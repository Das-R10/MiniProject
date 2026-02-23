# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from .clause_parser import extract_text_from_upload, split_into_clauses
from .pipeline import run_pipeline
from fastapi.responses import FileResponse
from .translate_api import router as translate_router
from .rag import router as rag_router, vecstore

app = FastAPI(title="Legal Clause Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
app.include_router(translate_router, prefix="", tags=["translate"])
app.include_router(rag_router, tags=["qa"])

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw_text = extract_text_from_upload(file)
    clauses = split_into_clauses(raw_text)

    vecstore.clear()
    vecstore.add_clauses(clauses)

    results = run_pipeline(clauses)

    return {
        "num_clauses": len(clauses),
        "clauses": clauses,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)