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
def build_cpu_legal_prompt(clause_text):
    return (
        "Rewrite the following employment contract clause to be fairer to the employee.\n\n"
        "IMPORTANT:\n"
        "- You MUST change the wording.\n"
        "- You MUST add employee protection.\n"
        "- Do NOT repeat the original sentence.\n"
        "- Output ONLY one rewritten legal clause (no lists, no explanations).\n\n"
        f"Original clause:\n{clause_text}\n\n"
        "Rewritten clause:"
    )

def generate_cpu_amendment(prompt):
    tokenizer, gen_model = get_gen_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        num_beams=5,
        repetition_penalty=1.6,
        no_repeat_ngram_size=3,
        length_penalty=1.1
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Rewritten clause:" in text:
        text = text.split("Rewritten clause:")[-1]
    return text.strip()

def clean_genai_output(original, generated):
    if generated is None:
        return None
    original_norm = original.lower().strip()
    generated_norm = generated.lower().strip()

    # Reject verbatim copy
    if generated_norm == original_norm:
        return None

    # Reject too-short rewrites
    if len(generated.split()) < 8:
        return None

    return generated

def violates_semantic_scope(original, generated):
    if generated is None:
        return False
    forbidden_keywords = {"terminate", "termination", "dismiss", "fire", "suspend", "sever"}
    original_text = original.lower()
    generated_text = generated.lower()
    for word in forbidden_keywords:
        if (word in generated_text) and (word not in original_text):
            return True
    return False

# === End-to-end pipeline function ===
def run_pipeline(clauses):
    """
    Input: list of clause dicts (same shape used throughout your project)
    Output: list of result dicts: {clause_id, label_name, confidence, amended_text}
    """
    results = []

    if len(clauses) == 0:
        return results

    # 1) embeddings and features
    model = get_sentence_model()
    texts = [c["text"] for c in clauses]
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(DEVICE)

    # positional features
    positions = np.array([c.get("position", i+1) for i,c in enumerate(clauses)], dtype=np.float32)
    positions = positions / positions.max()
    pos_tensor = torch.tensor(positions).unsqueeze(1).float()

    # section ids
    sections = list({c["section"] for c in clauses})
    section_to_id = {s: i for i, s in enumerate(sections)}
    section_ids = torch.tensor([section_to_id[c["section"]] for c in clauses], dtype=torch.float).unsqueeze(1)

    # node feature matrix
    X = torch.cat([embeddings, pos_tensor.to(DEVICE), section_ids.to(DEVICE)], dim=1).float()

    # 2) build graph
    edge_index, edge_labels = build_graph(clauses, embeddings)

    # 3) weak labels + confidences
    labels = []
    label_conf = []
    for c in clauses:
        lbl, conf = weak_label_clause_v2(c)
        labels.append(lbl)
        label_conf.append(conf)
    labels = torch.tensor(labels, dtype=torch.long)
    label_conf = torch.tensor(label_conf, dtype=torch.float).to(DEVICE)

    # 4) class weighting safely
    counts = Counter(labels.tolist())
    total = sum(counts.values())
    class_weights = torch.tensor([ (total / counts[i]) if counts.get(i,0) > 0 else 0.0 for i in range(3)], dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # 5) build model (GAT if available)
    in_ch = X.shape[1]
    num_classes = 3
    if HAS_TG:
        # use GPU if available else CPU
        model_net = GATNodeClassifier(in_channels=in_ch, hidden_channels=64, num_classes=num_classes).to(DEVICE)
    else:
        model_net = MLPNodeClassifier(in_channels=in_ch, hidden_channels=128, num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(model_net.parameters(), lr=0.004, weight_decay=1e-4)

    # 6) weakly supervised training (small number of epochs)
    model_net.train()
    epochs = 60
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        if HAS_TG:
            logits = model_net(X, edge_index)
        else:
            logits = model_net(X, None)
        per_node_loss = F.cross_entropy(logits, labels, reduction="none")
        weighted = per_node_loss * label_conf.to(per_node_loss.device)
        loss = weighted.mean()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"Train epoch {epoch:03d} loss={loss.item():.4f}")

    # 7) inference
    model_net.eval()
    with torch.no_grad():
        if HAS_TG:
            logits = model_net(X, edge_index)
        else:
            logits = model_net(X, None)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    # 8) generate amendments for risky clauses with safety checks
    tokenizer, gen_model = None, None
    results = []
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
        # decide amendment candidates: Con and high confidence
        if label_name == "Con" and conf >= 0.7 and c["section"].lower() not in {"definitions","governing law","commencement","interpretation"}:
            clauses_for_genai.append((i, c, label_name, conf))

    # only load gen model if we need to generate
    if clauses_for_genai:
        _ = get_gen_model_and_tokenizer()  # will load models (takes time)

    for (i, c, label_name, conf) in clauses_for_genai:
        prompt = build_cpu_legal_prompt(c["text"])
        amended = None
        try:
            amended_raw = generate_cpu_amendment(prompt)
            amended_clean = clean_genai_output(c["text"], amended_raw)
            if amended_clean is not None and violates_semantic_scope(c["text"], amended_clean):
                amended_clean = None
            amended = amended_clean
        except Exception as e:
            amended = None

        if amended is None:
            amended = "[No safe automatic amendment generated]"

        results[i]["amended"] = amended

    # return results list
    return results
