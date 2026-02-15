#!/usr/bin/env python3
"""
make_analytics_review.py

Purpose:
1. Summarize Business Analytics materials.
2. Extract concrete Homework/Assignment details.
3. Suggest advanced execution strategies tailored to an MLE Tech Lead (Deep Learning/Transformer expert).

Usage:
  python make_analytics_review.py --course-dir ./ba_course
  python make_analytics_review.py --course-dir ./ba_course --override
"""

import argparse
import os
import sys
import pathlib
from typing import List, Tuple
from dataclasses import dataclass

# --- Hashlib patch for some environments ---
import hashlib as _hashlib
_orig_md5 = _hashlib.md5
def _md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)
    return _orig_md5(*args, **kwargs)
_hashlib.md5 = _md5

# --- Imports ---
try:
    from pypdf import PdfReader
    from openai import OpenAI
    # ReportLab for PDF generation
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except Exception as e:
    sys.stderr.write(f"ERROR: Missing dependencies. {e}\nTry: pip install openai pypdf reportlab\n")
    sys.exit(1)

# --- CONFIGURATION ---
# Override with env BAI_REVIEW_PERSONA for your own role/context; this is the default example.
DEFAULT_PERSONA = (
    "The student is an MLE Area Tech Lead at Coupang. "
    "They are an expert in Transformer model pretraining, Deep ML, and large-scale recommendation systems. "
    "They are taking this Business Analytics class to bridge technical depth with business decision-making."
)
USER_PERSONA = os.environ.get("BAI_REVIEW_PERSONA", DEFAULT_PERSONA)

@dataclass
class PdfChunk:
    file: pathlib.Path
    chunk_index: int
    text: str

def list_sections(course_dir: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in sorted(course_dir.iterdir()) if p.is_dir()]

def pick_section(sections: List[pathlib.Path]) -> pathlib.Path:
    print("Which week/module do you want to analyze?\n")
    for i, sec in enumerate(sections, 1):
        print(f"[{i}] {sec.name}")
    while True:
        sel = input("\nEnter number (or 0 to exit): ").strip()
        if not sel.isdigit(): continue
        idx = int(sel)
        if idx == 0: sys.exit(0)
        if 1 <= idx <= len(sections): return sections[idx-1]

def find_pdfs(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in root.rglob("*.pdf") if p.is_file()]

def extract_text_from_pdf(pdf_path: pathlib.Path, max_pages: int = None) -> str:
    parts = []
    try:
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
        use_n = total if max_pages is None else min(total, max_pages)
        for i in range(use_n):
            parts.append(reader.pages[i].extract_text() or "")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return "\n".join(parts)

def chunk_text(s: str, max_chars: int = 8000) -> List[str]:
    s = " ".join(s.split())
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        start = end
    return chunks

# ---------------------------------------------------------------------------
# PROMPT 1: The Scanner
# Scans raw PDF text to find (1) Content and (2) Assignment details
# ---------------------------------------------------------------------------
def summarize_chunk(client: OpenAI, model: str, ch: PdfChunk) -> str:
    prompt = f"""You are analyzing course materials for a 'Business Analytics and AI' class.
    
Document: {ch.file.name} (chunk {ch.chunk_index})

Your goal is to extract two things:
1. **Core Content**: The main analytical concepts, formulas, or business frameworks discussed.
2. **HOMEWORK / ASSIGNMENT DETAILS**: Look specifically for instructions, prompt questions, deliverables, due dates, or dataset descriptions.

If this chunk contains assignment instructions, flag them clearly with "⚠️ ASSIGNMENT FOUND:".

Format:
- **Concept**: [Brief summary]
- **Assignment Details**: [If any, list specific requirements found in this text]
- **Data/Technique**: [What data or algos are mentioned? e.g. Random Forest, AB Testing, Churn prediction]

TEXT START:
{ch.text}
TEXT END
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful teaching assistant."}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def summarize_document(client: OpenAI, model: str, file_path: pathlib.Path, per_file_page_cap: int, chunk_chars: int) -> str:
    text = extract_text_from_pdf(file_path, max_pages=per_file_page_cap)
    if not text.strip(): return ""
    chunks_text = chunk_text(text, max_chars=chunk_chars)
    
    digests = []
    for j, t in enumerate(chunks_text, 1):
        digests.append(summarize_chunk(client, model, PdfChunk(file=file_path, chunk_index=j, text=t)))
    
    # Condense the chunk summaries into a file summary
    joined = "\n\n".join(digests)
    prompt = f"""Summarize the document "{file_path.name}" based on the notes below.
If specific ASSIGNMENT INSTRUCTIONS were found, separate them out clearly at the top.

Structure:
1. **Is this an Assignment?** (Yes/No - if Yes, summarize deliverables)
2. **Key Concepts** (Bullet points)
3. **Technical Details** (Formulas, models mentioned)

NOTES:
{joined}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a concise editor."}, {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# PROMPT 2: The Strategist
# Synthesizes the review + Connects to your specific background
# ---------------------------------------------------------------------------
def synthesize_custom_plan(client: OpenAI, model: str, section_name: str, per_doc_summaries: List[Tuple[str, str]]) -> str:
    # Combine all notes
    joined_notes = "\n\n---\n\n".join([f"### File: {name}\n{summ}" for (name, summ) in per_doc_summaries])
    
    prompt = f"""You are a Strategic Technical Advisor. 
    
    CONTEXT:
    {USER_PERSONA}
    
    TASK:
    Review the notes from the 'Business Analytics' module below. Create a briefing document that:
    
    1. **Summarizes the Module**: High-level overview of what is being taught (3-4 bullet points).
    2. **Identifies the Homework**: Explicitly state what the assignment is. What are the deliverables? What data is used?
    3. **Strategy for an Expert (Crucial Step)**: 
       Given the user is an expert in Deep ML and Transformers at Coupang, suggest 3 specific "extra mile" directions or unique angles for this assignment. 
       - How can they apply advanced intuition (e.g., embeddings, attention mechanisms, complex loss functions) to this potentially simpler business problem?
       - If the assignment uses simple regression/classification, how might the user critique it or propose a 'Pro' version using modern stacks?
       - How does this relate to high-scale e-commerce (Coupang context)?

    INPUT NOTES:
    {joined_notes}
    """

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a technical mentor bridging the gap between MBA business logic and Deep Learning engineering."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()

# --- PDF Generation (Simple wrapper) ---
def _wrap_text_to_lines(text: str, width_chars: int = 95) -> List[str]:
    lines = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        s = para
        while len(s) > width_chars:
            cut = s.rfind(" ", 0, width_chars)
            if cut <= 0: cut = width_chars
            lines.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        lines.append(s)
    return lines

def write_pdf_reportlab(text: str, out_path: pathlib.Path, title: str):
    c = canvas.Canvas(str(out_path), pagesize=LETTER)
    width, height = LETTER
    left, top = 0.75 * inch, height - 0.75 * inch
    y = top
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, title)
    y -= 25
    
    # Body
    c.setFont("Helvetica", 10)
    lines = _wrap_text_to_lines(text, width_chars=100)
    for line in lines:
        if y < 1 * inch:
            c.showPage(); y = top; c.setFont("Helvetica", 10)
        c.drawString(left, y, line)
        y -= 12
    c.save()

def save_outputs(text: str, md_path: pathlib.Path, pdf_path: pathlib.Path, title: str):
    md_path.write_text(text, encoding="utf-8")
    try:
        write_pdf_reportlab(text, pdf_path, title)
    except Exception as e:
        print(f"Warning: PDF write failed: {e}")

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course-dir", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--override", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Please set OPENAI_API_KEY")

    course_dir = pathlib.Path(args.course_dir).resolve()
    section_dir = pick_section(list_sections(course_dir))
    
    pdfs = find_pdfs(section_dir)
    if not pdfs:
        print("No PDFs found.")
        return

    client = OpenAI(api_key=api_key)
    per_doc_summaries = []

    # 1. Summarize Files
    for pdf in pdfs:
        md_out = section_dir / f"{pdf.stem}.summary.md"
        pdf_out = section_dir / f"{pdf.stem}.summary.pdf"
        
        if md_out.exists() and not args.override:
            per_doc_summaries.append((pdf.name, md_out.read_text(encoding='utf-8')))
            continue
            
        print(f"Reading: {pdf.name}...")
        summary = summarize_document(client, args.model, pdf, 30, 8000)
        save_outputs(summary, md_out, pdf_out, pdf.name)
        per_doc_summaries.append((pdf.name, summary))

    # 2. Synthesize Strategic Plan
    print("\nSynthesizing Strategy & Homework Plan...")
    final_out = section_dir / "Section_Strategy_Plan.md"
    final_pdf = section_dir / "Section_Strategy_Plan.pdf"
    
    plan = synthesize_custom_plan(client, args.model, section_dir.name, per_doc_summaries)
    
    save_outputs(plan, final_out, final_pdf, f"Strategy: {section_dir.name}")
    
    print(f"\nDone! Report saved to:\n{final_out}")

if __name__ == "__main__":
    main()
