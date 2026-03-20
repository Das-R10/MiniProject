# backend/constitution_loader.py
"""
Loads the Indian Constitution corpus from HuggingFace at startup
and indexes it into a dedicated FAISS vector store.

Dataset: nisaar/Constitution_of_India
Each row is one Article — stored as a clause-like dict so it's
compatible with DocumentVectorStore.add_clauses().
"""
import logging
import os
import pickle
import faiss
import numpy as np

logger = logging.getLogger(__name__)

CONST_STORE_PATH = "constitution_kb"
DATASET_NAME     = "nisaar/Constitution_of_India"
VECTOR_DIM       = 768   # InLegalBERT output dim


class ConstitutionVectorStore:
    """
    Immutable at runtime — loaded once at startup, never cleared.
    """

    def __init__(self, embed_fn):
        """
        embed_fn: callable that takes list[str] -> np.ndarray (normalized)
        """
        self.embed_fn = embed_fn
        self.index    = faiss.IndexFlatIP(VECTOR_DIM)
        self.metadata = []
        self._loaded  = False

    # ─────────────────────────────────────────
    # Load from disk cache or build from dataset
    # ─────────────────────────────────────────
    def load(self):
        if self._loaded:
            return

        idx_path = CONST_STORE_PATH + ".index"
        pkl_path = CONST_STORE_PATH + ".pkl"

        if os.path.exists(idx_path) and os.path.exists(pkl_path):
            logger.info("Loading Constitution index from disk cache...")
            self.index    = faiss.read_index(idx_path)
            with open(pkl_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Constitution index loaded — {len(self.metadata)} articles.")
            self._loaded = True
            return

        # Build from HuggingFace dataset
        logger.info("Downloading Indian Constitution dataset from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset(DATASET_NAME, split="train")
        except Exception as e:
            logger.warning(f"Could not load Constitution dataset: {e}. Constitution search disabled.")
            self._loaded = True
            return

        articles = self._parse_dataset(ds)
        if not articles:
            logger.warning("No articles parsed from Constitution dataset.")
            self._loaded = True
            return

        logger.info(f"Indexing {len(articles)} constitutional articles...")
        texts      = [a["text"] for a in articles]
        embeddings = self.embed_fn(texts)

        self.index.add(embeddings)
        self.metadata = articles

        faiss.write_index(self.index, idx_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info("Constitution index built and saved to disk.")
        self._loaded = True

    # ─────────────────────────────────────────
    # Parse dataset rows into clause-like dicts
    # ─────────────────────────────────────────
    def _parse_dataset(self, ds):
        articles = []
        # Try common column names used by the dataset
        text_cols    = ["article_text", "text", "content", "description", "Article"]
        id_cols      = ["article_id",   "id",   "article_number", "Article_Number"]
        title_cols   = ["article_title","title","heading",         "Article_Title"]

        col_names = ds.column_names
        text_col  = next((c for c in text_cols  if c in col_names), None)
        id_col    = next((c for c in id_cols    if c in col_names), None)
        title_col = next((c for c in title_cols if c in col_names), None)

        if text_col is None:
            # Fallback: use first string column
            for c in col_names:
                if ds.features[c].dtype == "string":
                    text_col = c
                    break

        if text_col is None:
            logger.warning(f"Could not identify text column in dataset. Columns: {col_names}")
            return []

        for i, row in enumerate(ds):
            text = str(row.get(text_col, "")).strip()
            if len(text.split()) < 5:
                continue

            art_id    = str(row.get(id_col,    i + 1)) if id_col    else str(i + 1)
            art_title = str(row.get(title_col, ""))    if title_col else ""

            articles.append({
                "clause_id" : f"Art.{art_id}",
                "text"      : text,
                "section"   : art_title or "Indian Constitution",
                "page_no"   : 0,
                "source"    : "constitution",
            })

        return articles

    # ─────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────
    def search(self, query: str, k: int = 2):
        if not self._loaded or self.index.ntotal == 0:
            return []

        q_emb  = self.embed_fn([query])
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if 0 <= idx < len(self.metadata):
                item          = self.metadata[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results


# ─────────────────────────────────────────
# Module-level singleton — initialised in main.py startup
# ─────────────────────────────────────────
_const_store: ConstitutionVectorStore | None = None


def get_constitution_store() -> ConstitutionVectorStore:
    return _const_store


def init_constitution_store(embed_fn):
    global _const_store
    _const_store = ConstitutionVectorStore(embed_fn)
    _const_store.load()
    return _const_store