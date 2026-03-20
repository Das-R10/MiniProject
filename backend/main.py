# backend/main.py
import logging
import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .clause_parser import extract_text_from_upload, split_into_clauses
from .pipeline import run_pipeline
from .rag import router as rag_router, create_session
from .translate_api import router as translate_router
from .constitution_loader import init_constitution_store

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Upload limits
# ─────────────────────────────────────────
MAX_FILE_SIZE   = 10 * 1024 * 1024   # 10 MB
ALLOWED_TYPES   = {"application/pdf", "text/plain"}
ALLOWED_EXTS    = {".pdf", ".txt"}

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(title="Legal Clause Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
app.include_router(translate_router, prefix="", tags=["translate"])
app.include_router(rag_router, tags=["qa"])


# ─────────────────────────────────────────
# Startup — pre-load models + constitution
# ─────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=== LexAnalyze startup ===")

    # Warm up embedding model (shared by pipeline + rag)
    from .pipeline import get_sentence_model
    embed_model = get_sentence_model()

    # Embed function wrapper for constitution loader
    def embed_fn(texts):
        return embed_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    # Load Indian Constitution corpus
    init_constitution_store(embed_fn)

    # Warm up QA model
    from .rag import get_qa_pipeline
    get_qa_pipeline()

    logger.info("=== Startup complete — ready to serve ===")


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # ── Validate file type ─────────────────────────────────────────────
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Upload a PDF or TXT file.",
        )

    # ── Validate file size ─────────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)//1024}KB). Maximum is 10MB.",
        )

    # Reset file pointer for downstream reader
    import io
    file.file = io.BytesIO(content)

    # ── Parse + split ──────────────────────────────────────────────────
    logger.info(f"Processing upload: {file.filename} ({len(content)//1024}KB)")
    raw_text = extract_text_from_upload(file)
    clauses  = split_into_clauses(raw_text)

    if not clauses:
        raise HTTPException(
            status_code=422,
            detail="No clauses could be extracted from the document.",
        )

    # ── Create session-scoped vector store ────────────────────────────
    session_id, store = create_session()
    store.add_clauses(clauses)

    # ── Run pipeline ───────────────────────────────────────────────────
    results = run_pipeline(clauses, role="Employee")

    return {
        "session_id" : session_id,
        "num_clauses": len(clauses),
        "clauses"    : clauses,
        "results"    : results,
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)