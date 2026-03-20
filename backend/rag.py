# backend/rag.py
import logging
import uuid

import faiss
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .pipeline import get_sentence_model
from .constitution_loader import get_constitution_store

logger = logging.getLogger(__name__)

VECTOR_DIM = 768   # InLegalBERT output dim

# ─────────────────────────────────────────
# Lazy QA pipeline (roberta-base-squad2)
# ─────────────────────────────────────────
_QA_PIPELINE = None

def get_qa_pipeline():
    global _QA_PIPELINE
    if _QA_PIPELINE is None:
        from transformers import pipeline
        logger.info("Loading deepset/roberta-base-squad2 QA model...")
        _QA_PIPELINE = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1,   # CPU
        )
        logger.info("QA model loaded.")
    return _QA_PIPELINE


# ─────────────────────────────────────────
# Shared embed helper (uses InLegalBERT singleton)
# ─────────────────────────────────────────
def _embed(texts: list[str]) -> np.ndarray:
    model = get_sentence_model()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


# ─────────────────────────────────────────
# Per-session in-memory vector store
# ─────────────────────────────────────────
class DocumentVectorStore:
    def __init__(self):
        self.index    = faiss.IndexFlatIP(VECTOR_DIM)
        self.metadata = []

    def add_clauses(self, clauses: list[dict]):
        if not clauses:
            return
        texts      = [c["text"] for c in clauses]
        embeddings = _embed(texts)
        self.index.add(embeddings)

        for c in clauses:
            self.metadata.append({
                "clause_id" : c["clause_id"],
                "text"      : c["text"],
                "section"   : c.get("section", "Unknown"),
                "page_no"   : c.get("page_no", 0),
                "source"    : "contract",
            })
        logger.debug(f"VectorStore: {len(self.metadata)} clauses indexed.")

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q_emb = _embed([query])
        k     = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item          = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results


# ─────────────────────────────────────────
# Session registry — keyed by session_id
# ─────────────────────────────────────────
_stores: dict[str, DocumentVectorStore] = {}


def create_session() -> tuple[str, DocumentVectorStore]:
    session_id          = str(uuid.uuid4())
    store               = DocumentVectorStore()
    _stores[session_id] = store
    logger.debug(f"Created session {session_id}")
    return session_id, store


def get_store(session_id: str) -> DocumentVectorStore | None:
    return _stores.get(session_id)


def cleanup_session(session_id: str):
    _stores.pop(session_id, None)
    logger.debug(f"Cleaned up session {session_id}")


# ─────────────────────────────────────────
# FastAPI QA router
# ─────────────────────────────────────────
router = APIRouter()


class QARequest(BaseModel):
    question   : str
    session_id : str = ""


@router.post("/qa")
def document_qa(req: QARequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question required")

    # ── Retrieve from contract store ───────────────────────────────────
    store         = get_store(req.session_id) if req.session_id else None
    contract_hits = store.search(question, k=3) if store else []

    # ── Retrieve from Constitution store ──────────────────────────────
    const_store = get_constitution_store()
    const_hits  = const_store.search(question, k=2) if const_store else []

    for h in contract_hits:
        h["source"] = "contract"
    for h in const_hits:
        h["source"] = "constitution"

    retrieved = contract_hits + const_hits

    if not retrieved:
        return {
            "answer"     : None,
            "explanation": "No relevant information found in the document or constitution.",
            "evidence"   : [],
        }

    # ── Build context ──────────────────────────────────────────────────
    # Truncate each clause snippet to 200 chars so 5 hits stay well
    # under RoBERTa's 512-token window even before the question is added.
    context_parts = []
    for r in retrieved:
        source_tag = "[Constitution]" if r["source"] == "constitution" else "[Contract]"
        snippet    = r["text"][:200].replace("\n", " ")
        context_parts.append(
            f"{source_tag} Clause {r['clause_id']} ({r.get('section', '')}):\n{snippet}"
        )
    context = "\n\n".join(context_parts)

    # Hard cap at 1500 chars — keeps doc_stride (128) safely below
    # max_len after question tokens and special tokens are subtracted.
    if len(context) > 1500:
        context = context[:1500]

    # ── Run extractive QA ─────────────────────────────────────────────
    try:
        qa     = get_qa_pipeline()
        result = qa(
            question=question,
            context=context,
            max_answer_len=100,
            max_seq_len=512,
            doc_stride=128,           # stride < (512 - question_tokens - special_tokens)
            handle_impossible_answer=False,
        )
        answer = result.get("answer", "").strip()
        score  = round(float(result.get("score", 0.0)), 4)
    except Exception as e:
        logger.error(f"QA pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"QA model error: {e}")

    if not answer:
        answer = "Could not find a direct answer in the available clauses."

    return {
        "answer"  : answer,
        "score"   : score,
        "evidence": retrieved,
    }