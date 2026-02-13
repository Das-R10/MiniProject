# backend/clause_parser.py
import re
from fastapi import UploadFile
import fitz  # PyMuPDF

def extract_text_from_upload(file: UploadFile) -> str:
    """
    Read PDF or text UploadFile and return plain text.
    """
    filename = file.filename.lower()
    content = file.file.read()
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for p in doc:
            text += p.get_text()
        return text
    else:
        # assume text file
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1")

import re

def remove_trailing_section_headers(text: str):
    """
    Removes trailing section headers like:
    '3. Compensation', '4. Confidentiality', etc.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Detect section headers: number + capitalized title
        if re.match(r"^\s*\d+\.\s+[A-Z][A-Za-z ]+$", line.strip()):
            break
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def fallback_paragraph_split(text: str):
    """
    Fallback clause splitter for paragraph-style legal documents.
    """
    paragraphs = re.split(r"\n\s*\n+", text)
    clauses = []
    position = 1

    for para in paragraphs:
        para = para.strip()
        if len(para.split()) < 40:  # ignore short junk
            continue

        clauses.append({
            "clause_id": f"P{position}",
            "section": "Unknown",
            "text": para,
            "position": position,
            "page_no": 0,
            "layout_type": "paragraph",
            "font_size": 11,
            "language": "en"
        })

        position += 1

    return clauses


def split_into_clauses(text: str):
    """
    Improved clause splitter for Indian legal documents.
    Handles:
    - Numbered clauses (1., 1.That, 2.1, 3))
    - Lettered clauses (a), b), c))
    """

    text = text.replace("\r", "")

    clauses = []
    position = 1

    # --- Numeric clauses ---
    numeric_clause_pattern = re.compile(
        r"\n\s*(\d+(?:\.\d+)*[\.\)]?)\s*([A-Z].*?)(?=\n\s*\d+(?:\.\d+)*[\.\)]?\s*|$)",
        re.DOTALL
    )

    for match in numeric_clause_pattern.finditer(text):
        clause_id = match.group(1)
        raw_text = match.group(2).strip()
        clause_text = remove_trailing_section_headers(raw_text)

        if len(clause_text.split()) < 4:
            continue

        clauses.append({
            "clause_id": clause_id,
            "section": "Unknown",
            "text": clause_text,
            "position": position,
            "page_no": 0,
            "layout_type": "paragraph",
            "font_size": 11,
            "language": "en"
        })

        position += 1

    # --- Lettered clauses ---
    lettered_clause_pattern = re.compile(
        r"\n\s*([a-z]\))\s+(.+?)(?=\n\s*[a-z]\)|$)",
        re.DOTALL | re.IGNORECASE
    )

    for match in lettered_clause_pattern.finditer(text):
        clause_id = match.group(1)  # a), b), c)
        clause_text = match.group(2).strip()
        clause_text = remove_trailing_section_headers(clause_text)

        if len(clause_text.split()) < 4:
            continue

        clauses.append({
            "clause_id": clause_id,
            "section": "Unknown",
            "text": clause_text,
            "position": position,
            "page_no": 0,
            "layout_type": "paragraph",
            "font_size": 11,
            "language": "en"
        })

        position += 1

    if not clauses:
        clauses = fallback_paragraph_split(text)

    return clauses

