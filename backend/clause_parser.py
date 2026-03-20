# backend/clause_parser.py
import re
from fastapi import UploadFile
import fitz  # PyMuPDF


def extract_text_from_upload(file: UploadFile) -> str:
    filename = file.filename.lower()
    content  = file.file.read()
    if filename.endswith(".pdf"):
        doc  = fitz.open(stream=content, filetype="pdf")
        text = ""
        for p in doc:
            text += p.get_text()
        return text
    else:
        try:
            return content.decode("utf-8")
        except Exception:
            return content.decode("latin-1")


def extract_section_name(first_line: str) -> str:
    line = first_line.strip()
    line = re.sub(r"^\d+(?:\.\d+)*[\.\)]?\s*", "", line).strip()
    if line and len(line.split()) <= 8 and not line.endswith(".") and line[0].isupper():
        return line
    return "Unknown"


def split_into_clauses(text: str):
    """
    Line-based clause splitter.

    Handles:
      - Standard:    "2.1 The Employee shall..."
      - Quoted:      '1.1 "Services" means...'
      - ALL CAPS:    "6.3 THE SERVICE PROVIDER EXPRESSLY DISCLAIMS..."
      - Section hdr: "10. Governing Law and Jurisdiction"

    Strategy: scan line by line for clause openers, slice between them.
    No DOTALL lookahead so nothing is silently truncated.
    """
    text  = text.replace("\r\n", "\n").replace("\r", "")
    lines = text.split("\n")

    # Clause opener: number + space + content starting with A-Z, quote, or ALL CAPS
    clause_open = re.compile(r'''^(\d+(?:\.\d+)*\.?)\s+([A-Z"'].+)$''')
    sep_pat     = re.compile(r"^[─━═\-]{4,}$")

    # Pass 1: find all start positions
    starts = []
    for i, line in enumerate(lines):
        m = clause_open.match(line.strip())
        if m:
            cid = m.group(1).rstrip(".")
            starts.append((i, cid, m.group(2).strip()))

    if not starts:
        return _fallback_paragraph_split(text)

    # Pass 2: slice body per clause
    clauses  = []
    position = 1

    for idx, (line_idx, cid, first_text) in enumerate(starts):
        end_idx    = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body_lines = lines[line_idx:end_idx]
        cleaned    = [l for l in body_lines if not sep_pat.match(l.strip())]
        clause_text = "\n".join(cleaned).strip()

        # Strip leading clause number
        clause_text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", clause_text, count=1).strip()

        if len(clause_text.split()) < 4:
            continue

        # Section name detection
        first_line = clause_text.splitlines()[0].strip()
        is_title   = (
            len(first_line.split()) <= 8
            and not first_line.endswith(".")
            and not first_line.endswith(",")
            and first_line[0].isupper()
        )
        section = first_line if is_title else extract_section_name(first_line)

        clauses.append({
            "clause_id"  : cid,
            "section"    : section,
            "text"       : clause_text,
            "position"   : position,
            "page_no"    : 0,
            "layout_type": "paragraph",
            "font_size"  : 11,
            "language"   : "en",
        })
        position += 1

    return clauses


def _fallback_paragraph_split(text: str):
    paragraphs = re.split(r"\n\s*\n+", text)
    clauses    = []
    position   = 1
    for para in paragraphs:
        para = para.strip()
        if len(para.split()) < 20:
            continue
        clauses.append({
            "clause_id"  : f"P{position}",
            "section"    : "Unknown",
            "text"       : para,
            "position"   : position,
            "page_no"    : 0,
            "layout_type": "paragraph",
            "font_size"  : 11,
            "language"   : "en",
        })
        position += 1
    return clauses