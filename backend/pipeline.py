# backend/pipeline.py
print(" THIS PIPELINE FILE IS RUNNING")

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import itertools
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cpu"
SIM_THRESHOLD = 0.6

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
        "terminate": 1.5,
        "without any inquiry": 2.0,
        "without warning": 1.5,
        "without reason": 1.5,
    }

    pro_patterns = {
        "thirty (30) days": 1.5,
        "prior written notice": 2.0,
        "by either party": 1.0,
        "severance": 2.0,
        "compensation": 1.0,
        "entitled": 1.5,
        "shall be provided": 1.0,
        "paid leave": 1.5,
        "medical": 1.0,
        "insurance": 1.0,
    }

    neutral_sections = {
        "definitions", "interpretation", "governing law",
        "commencement", "introduction", "preamble"
    }

    # HARD neutral sections
    if any(ns in section for ns in neutral_sections):
        return LABEL_MAP["Neutral"], 1.0

    for pat, w in con_patterns.items():
        if pat in text:
            score -= w

    for pat, w in pro_patterns.items():
        if pat in text:
            score += w

    if score <= -1.5:
        return LABEL_MAP["Con"], min(1.0, abs(score) / 3)
    elif score >= 1.5:
        return LABEL_MAP["Pro"], min(1.0, abs(score) / 3)
    else:
        return LABEL_MAP["Neutral"], 0.3


def build_graph(clauses, embeddings):
    n = len(clauses)
    edge_src, edge_dst, edge_labels = [], [], []

    for i in range(n - 1):
        edge_src.extend([i, i + 1])
        edge_dst.extend([i + 1, i])
        edge_labels.extend(["sequential", "sequential"])

    for i, j in itertools.combinations(range(n), 2):
        if clauses[i]["section"] == clauses[j]["section"]:
            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_labels.extend(["same_section", "same_section"])

    emb_np = embeddings.cpu().numpy()
    sim_m = cosine_similarity(emb_np)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_m[i, j] >= SIM_THRESHOLD:
                edge_src.extend([i, j, j, i])
                edge_dst.extend([j, i, i, j])
                edge_labels.extend(["semantic"] * 4)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros((2, 0), dtype=torch.long)
    return edge_index, edge_labels


def try_import_torch_geometric():
    try:
        from torch_geometric.nn import GATConv
        return True
    except Exception:
        return False

HAS_TG = try_import_torch_geometric()

if HAS_TG:
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


def run_pipeline(clauses, role="Employee"):
    results = []
    if len(clauses) == 0:
        return results

    model = get_sentence_model()
    texts = [c["text"] for c in clauses]

    with torch.no_grad():
        embeddings = model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        ).to(DEVICE)

    positions = np.array(
        [c.get("position", i + 1) for i, c in enumerate(clauses)], dtype=np.float32
    )
    positions = positions / positions.max()
    pos_tensor = torch.tensor(positions).unsqueeze(1).float()

    sections = list({c["section"] for c in clauses})
    section_to_id = {s: i for i, s in enumerate(sections)}
    section_ids = torch.tensor(
        [section_to_id[c["section"]] for c in clauses], dtype=torch.float
    ).unsqueeze(1)

    X = torch.cat([embeddings, pos_tensor.to(DEVICE), section_ids.to(DEVICE)], dim=1).float()

    edge_index, edge_labels = build_graph(clauses, embeddings)

    labels = []
    label_conf = []
    for c in clauses:
        lbl, conf = weak_label_clause_v2(c)
        labels.append(lbl)
        label_conf.append(conf)

    labels = torch.tensor(labels, dtype=torch.long)
    label_conf = torch.tensor(label_conf, dtype=torch.float).to(DEVICE)

    in_ch = X.shape[1]
    num_classes = 3

    if HAS_TG:
        model_net = GATNodeClassifier(in_channels=in_ch, hidden_channels=64, num_classes=num_classes).to(DEVICE)
    else:
        model_net = MLPNodeClassifier(in_channels=in_ch, hidden_channels=128, num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(model_net.parameters(), lr=0.004, weight_decay=1e-4)

    model_net.train()
    for epoch in range(1, 31):
        optimizer.zero_grad()
        logits = model_net(X, edge_index) if HAS_TG else model_net(X, None)
        per_node_loss = F.cross_entropy(logits, labels, reduction="none")
        loss = (per_node_loss * label_conf).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Train epoch {epoch:03d} loss={loss.item():.4f}")

    model_net.eval()
    with torch.no_grad():
        logits = model_net(X, edge_index) if HAS_TG else model_net(X, None)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    clauses_for_amendment = []
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
            clauses_for_amendment.append((i, c))

    for (i, c) in clauses_for_amendment:
        results[i]["amended"] = build_amendment(c["text"])

    return results