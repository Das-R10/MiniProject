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
        try:
            return content.decode("utf-8")
        except Exception:
            return content.decode("latin-1")


def extract_section_name(first_line: str) -> str:
    """
    Extract a clean section name from the first line of a clause.
    E.g. 'Termination Without Notice' -> 'Termination Without Notice'
    """
    line = first_line.strip()
    # Remove leading number like "4." or "10.1"
    line = re.sub(r"^\d+(?:\.\d+)*[\.\)]?\s*", "", line).strip()
    # Only treat it as a section name if it looks like a title
    # (short, title-cased, no period inside)
    if line and len(line.split()) <= 6 and not line.endswith(".") and line[0].isupper():
        return line
    return "Unknown"


def remove_trailing_section_headers(text: str) -> str:
    """
    Removes trailing section headers like:
    '3. Compensation', '4. Confidentiality', etc.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
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
        if len(para.split()) < 40:
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
    Extracts section names from clause headings.
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

        # Extract section from first line
        first_line = clause_text.splitlines()[0]
        section = extract_section_name(first_line)

        clauses.append({
            "clause_id": clause_id,
            "section": section,
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
        clause_id = match.group(1)
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