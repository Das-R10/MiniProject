# backend/rag.py
import os
import faiss
import pickle
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from .pipeline import get_gen_model_and_tokenizer, violates_semantic_scope

# =========================
# Vector store (document RAG)
# =========================

EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2", device="cpu")
VECTOR_DIM = 768
STORE_PATH = "doc_kb"

class DocumentVectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.metadata = []

        if os.path.exists(STORE_PATH + ".index"):
            self.index = faiss.read_index(STORE_PATH + ".index")
            with open(STORE_PATH + ".pkl", "rb") as f:
                self.metadata = pickle.load(f)

    def clear(self):
        if os.path.exists(STORE_PATH + ".index"):
            os.remove(STORE_PATH + ".index")
        if os.path.exists(STORE_PATH + ".pkl"):
            os.remove(STORE_PATH + ".pkl")
        self.index = faiss.IndexFlatIP(VECTOR_DIM)
        self.metadata = []

    def add_clauses(self, clauses):
        texts = [c["text"] for c in clauses]
        embeddings = EMBED_MODEL.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        self.index.add(embeddings)

        for c in clauses:
            self.metadata.append({
                "clause_id": c["clause_id"],
                "text": c["text"],
                "section": c.get("section", "Unknown"),
                "page_no": c.get("page_no", 0)
            })

        faiss.write_index(self.index, STORE_PATH + ".index")
        with open(STORE_PATH + ".pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query, k=4):
        q_emb = EMBED_MODEL.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results


# =========================
# FastAPI QA Router
# =========================

router = APIRouter()
vecstore = DocumentVectorStore()

class QARequest(BaseModel):
    question: str

@router.post("/qa")
def document_qa(req: QARequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question required")

    retrieved = vecstore.search(question, k=4)

    if not retrieved:
        return {
            "answer": None,
            "explanation": "Not enough information in the uploaded document.",
            "evidence": []
        }

    refs = ""
    for i, r in enumerate(retrieved):
        refs += (
            f"[Ref {i+1}] Clause {r['clause_id']}\n"
            f"{r['text']}\n\n"
        )

    prompt = (
        "Answer the question using ONLY the passages below. "
        "If the answer is not present, say you cannot find it.\n\n"
        f"Passages:\n{refs}\n"
        f"Question: {question}\n\n"
        "Answer (cite like [Ref 1]):"
    )

    tokenizer, model = get_gen_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        num_beams=5
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if violates_semantic_scope(" ".join(r["text"] for r in retrieved), answer):
        return {
            "answer": None,
            "explanation": "Answer may go beyond document scope.",
            "evidence": retrieved
        }

    return {
        "answer": answer,
        "evidence": retrieved
    }
