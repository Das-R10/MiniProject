# backend/pipeline.py
"""
Run this from backend.main. This module:
- Creates embeddings with sentence-transformers
- Builds a simple clause graph (sequential + same-section + semantic)
- Generates weak labels with confidence
- Trains a small GNN (if torch_geometric available) or an MLP fallback
- Predicts labels and confidences
- Tries controlled GenAI rewriting (deterministic, with safety checks)
"""
print(" THIS PIPELINE FILE IS RUNNING")

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# transformers for GenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Utility
from collections import Counter
import itertools
import math
import warnings
warnings.filterwarnings("ignore")

# === Config ===
DEVICE = "cpu"  # CPU-first design
SIM_THRESHOLD = 0.6  # semantic similarity to add edges

# === Load models lazily to save startup time ===
_SENTENCE_MODEL = None
_GEN_TOKENIZER = None
_GEN_MODEL = None

def get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    return _SENTENCE_MODEL

def get_gen_model_and_tokenizer():
    global _GEN_MODEL, _GEN_TOKENIZER
    if _GEN_TOKENIZER is None:
        _GEN_TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-large")
    if _GEN_MODEL is None:
        _GEN_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        _GEN_MODEL.to(DEVICE)
    return _GEN_TOKENIZER, _GEN_MODEL

# === Weak label function (returns label,int and confidence float) ===
LABEL_MAP = {"Neutral": 0, "Pro": 1, "Con": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def weak_label_clause_v2(clause):
    text = clause["text"].lower()
    section = clause["section"].lower()

    score = 0.0

    con_patterns = {
        "sole discretion": 2.0,
        "without notice": 2.0,
        "at any time": 1.5,
        "reserves the right": 2.0,
        "immediately": 1.0,
        "modify": 1.0,
        "terminate": 1.5
    }

    pro_patterns = {
        "thirty (30) days": 1.5,
        "prior written notice": 2.0,
        "by either party": 1.0,
        "severance": 2.0,
        "compensation": 1.0
    }

    neutral_sections = {
        "definitions",
        "interpretation",
        "governing law",
        "commencement"
    }

    # HARD neutral sections
    if section in neutral_sections:
        return LABEL_MAP["Neutral"], 1.0

    # score accumulation
    for pat, w in con_patterns.items():
        if pat in text:
            score -= w

    for pat, w in pro_patterns.items():
        if pat in text:
            score += w

    # final decision
    if score <= -1.5:
        return LABEL_MAP["Con"], min(1.0, abs(score) / 3)
    elif score >= 1.5:
        return LABEL_MAP["Pro"], min(1.0, abs(score) / 3)
    else:
        return LABEL_MAP["Neutral"], 0.3

# === Graph builder ===
def build_graph(clauses, embeddings):
    # nodes: number of clauses
    n = len(clauses)
    edge_src, edge_dst, edge_labels = [], [], []

    # sequential edges
    for i in range(n-1):
        edge_src.extend([i, i+1])
        edge_dst.extend([i+1, i])
        edge_labels.extend(["sequential", "sequential"])

    # same-section edges
    for i, j in itertools.combinations(range(n), 2):
        if clauses[i]["section"] == clauses[j]["section"]:
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_labels.extend(["same_section", "same_section"])

    # semantic edges using cosine similarity
    emb_np = embeddings.cpu().numpy()
    sim_m = cosine_similarity(emb_np)
    for i in range(n):
        for j in range(i+1, n):
            if sim_m[i, j] >= SIM_THRESHOLD:
                edge_src.extend([i, j, j, i])
                edge_dst.extend([j, i, i, j])
                edge_labels.extend(["semantic", "semantic", "semantic", "semantic"])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2,0), dtype=torch.long)
    return edge_index, edge_labels

# === Model builders (GAT if available, else MLP fallback) ===
def try_import_torch_geometric():
    try:
        import torch_geometric
        from torch_geometric.nn import GATConv
        return True
    except Exception:
        return False

HAS_TG = try_import_torch_geometric()

if HAS_TG:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv

    class GATNodeClassifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
            self.gat2 = GATConv(hidden_channels * 4, num_classes, heads=1, concat=False)

        def forward(self, x, edge_index):
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.gat2(x, edge_index)
            return x
else:
    # Fallback: small MLP that uses the same node features (no graph ops)
    class MLPNodeClassifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, num_classes):
            super().__init__()
            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index=None):
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin2(x)
            return x

# === GenAI utilities ===
# ==========================================================
# RAG KNOWLEDGE BASE
# ==========================================================

RAG_EXAMPLES = [
    {
        "keywords": ["terminate", "termination"],
        "unsafe": "Employer may terminate employment immediately.",
        "safe": "Employer may terminate employment only after prior written notice of 15 days."
    },
    {
        "keywords": ["dismiss", "misconduct"],
        "unsafe": "Employee may be dismissed immediately for misconduct.",
        "safe": "Employee may be dismissed for misconduct only after fair investigation and written warning."
    },
    {
        "keywords": ["modify", "change terms"],
        "unsafe": "Employer may modify terms at its sole discretion.",
        "safe": "Employer may modify terms only after consultation and written agreement with the employee."
    },
    {
        "keywords": ["suspend"],
        "unsafe": "Employee may be suspended immediately.",
        "safe": "Employee suspension requires written notice and reasonable opportunity to respond."
    }
]

def retrieve_best_template(clause_text):

    text = clause_text.lower()

    if any(k in text for k in ["terminate", "termination"]):
        return "only after providing prior written notice of 15 days"

    if any(k in text for k in ["dismiss", "misconduct"]):
        return "only after fair investigation and written warning"

    if any(k in text for k in ["modify", "sole discretion"]):
        return "only after prior written notice and employee consultation"

    if "suspend" in text:
        return "only after written notice and opportunity to respond"

    return "with prior written notice and fair review"


def build_amendment(original):

    template = retrieve_best_template(original)

    original = original.rstrip(".")
    return f"{original}, {template}."


# ==========================================================
# FEW-SHOT ROLE BASED LEGAL PROMPT
# ==========================================================
def build_rag_prompt(clause_text, role="Employee"):

    return f"""
Rewrite unfair employment clauses to be fair.

Examples:

Original:
The company may terminate employment without notice.
Rewritten:
The company may terminate employment only after prior written notice of 15 days.

Original:
Employee may be dismissed immediately for misconduct.
Rewritten:
Employee may be dismissed for misconduct only after fair investigation and written warning.

Original:
Employer may modify terms at its sole discretion.
Rewritten:
Employer may modify terms only after prior written notice and employee consultation.

Now rewrite:

Original:
{clause_text}

Rewritten:
"""




# ==========================================================
# GENAI GENERATION
# ==========================================================
def generate_cpu_amendment(prompt):

    tokenizer, gen_model = get_gen_model_and_tokenizer()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    outputs = gen_model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.1
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

# ==========================================================
# CLEAN GENERATED OUTPUT
# ==========================================================
def clean_genai_output(original, generated):

    if generated is None:
        return None

    g = generated.strip()

    # remove prefixes
    if "Rewritten:" in g:
        g = g.split("Rewritten:")[-1].strip()

    # ONLY reject if absurdly short
    if len(g.split()) < 4:
        return None

    return g


def fallback_amendment(original):

    template = retrieve_best_template(original)

    return original.rstrip(".") + ", " + template + "."



# ==========================================================
# MERGE ORIGINAL + ADDITION
# ==========================================================
def merge_amendment(original, addition):

    if addition is None:
        addition = (
            "provided prior written notice and fair opportunity to respond are given"
        )

    addition = addition.strip().rstrip(".")
    original = original.rstrip(".")

    return f"{original}, {addition}."


# ==========================================================
# END-TO-END PIPELINE
# ==========================================================
def run_pipeline(clauses, role="Employee"):
    """
    Input: list of clause dicts
    Output: results with labels + amendments
    """

    results = []

    if len(clauses) == 0:
        return results

    # ===== Embeddings =====
    model = get_sentence_model()
    texts = [c["text"] for c in clauses]

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(DEVICE)

    # ===== Positional feature =====
    positions = np.array(
        [c.get("position", i+1) for i, c in enumerate(clauses)],
        dtype=np.float32
    )
    positions = positions / positions.max()
    pos_tensor = torch.tensor(positions).unsqueeze(1).float()

    # ===== Section IDs =====
    sections = list({c["section"] for c in clauses})
    section_to_id = {s: i for i, s in enumerate(sections)}

    section_ids = torch.tensor(
        [section_to_id[c["section"]] for c in clauses],
        dtype=torch.float
    ).unsqueeze(1)

    # ===== Node feature matrix =====
    X = torch.cat(
        [embeddings, pos_tensor.to(DEVICE), section_ids.to(DEVICE)],
        dim=1
    ).float()

    # ===== Graph =====
    edge_index, edge_labels = build_graph(clauses, embeddings)

    # ===== Weak labels =====
    labels = []
    label_conf = []

    for c in clauses:
        lbl, conf = weak_label_clause_v2(c)
        labels.append(lbl)
        label_conf.append(conf)

    labels = torch.tensor(labels, dtype=torch.long)
    label_conf = torch.tensor(label_conf, dtype=torch.float).to(DEVICE)

    # ===== Model =====
    in_ch = X.shape[1]
    num_classes = 3

    if HAS_TG:
        model_net = GATNodeClassifier(
            in_channels=in_ch,
            hidden_channels=64,
            num_classes=num_classes
        ).to(DEVICE)
    else:
        model_net = MLPNodeClassifier(
            in_channels=in_ch,
            hidden_channels=128,
            num_classes=num_classes
        ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model_net.parameters(),
        lr=0.004,
        weight_decay=1e-4
    )

    # ===== Training =====
    model_net.train()
    epochs = 30

    for epoch in range(1, epochs + 1):

        optimizer.zero_grad()

        if HAS_TG:
            logits = model_net(X, edge_index)
        else:
            logits = model_net(X, None)

        per_node_loss = F.cross_entropy(
            logits,
            labels,
            reduction="none"
        )

        loss = (per_node_loss * label_conf).mean()

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Train epoch {epoch:03d} loss={loss.item():.4f}")

    # ===== Inference =====
    model_net.eval()

    with torch.no_grad():
        if HAS_TG:
            logits = model_net(X, edge_index)
        else:
            logits = model_net(X, None)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # ===== Results =====
    clauses_for_genai = []

    for i, c in enumerate(clauses):

        pred = int(preds[i])
        conf = float(probs[i, pred])
        label_name = INV_LABEL_MAP[pred]

        results.append({
            "clause_id": c["clause_id"],
            "section": c["section"],
            "label": label_name,
            "confidence": conf,
            "original": c["text"],
            "amended": None
        })

        if label_name == "Con":
            clauses_for_genai.append((i, c))

    # ===== Load GenAI only if needed =====
    if clauses_for_genai:
        _ = get_gen_model_and_tokenizer()

   # ===== Amendment Generation =====
    for (i, c) in clauses_for_genai:

      amended = build_amendment(c["text"])

      results[i]["amended"] = amended





    return results

