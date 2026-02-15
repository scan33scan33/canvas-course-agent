#!/usr/bin/env python3
"""
make_review_v2.py — Per-document summaries → section one-pager (+ PDFs)

What’s new vs make_review.py
- Summarize each PDF independently (document digests)
- Save each per-document summary as Markdown + PDF
- Then synthesize a section-level one-pager from all per-document summaries
- Save the section one-pager as Markdown + PDF
- Still skips regeneration unless --override is provided

Prereqs
  pip install openai pypdf reportlab
  export OPENAI_API_KEY=sk-...

Usage
  python make_review_v2.py --course-dir ./cto_course
  python make_review_v2.py --course-dir ./cto_course --override
  python make_review_v2.py --course-dir ./cto_course --model gpt-4o-mini
"""

import argparse
import os
import sys
import pathlib
from typing import List, Tuple
from dataclasses import dataclass

# --- add this near the top, before "from reportlab ..." imports ---
import hashlib as _hashlib
_orig_md5 = _hashlib.md5
def _md5(*args, **kwargs):
    kwargs.pop("usedforsecurity", None)  # ignore unsupported kw
    return _orig_md5(*args, **kwargs)
_hashlib.md5 = _md5

# --- third-party ---
try:
    from pypdf import PdfReader
except Exception:
    sys.stderr.write("ERROR: Please install pypdf (pip install pypdf)\n")
    raise

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    sys.stderr.write("ERROR: Please install openai (pip install openai)\n")
    raise

# PDF generation (ReportLab)
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, FrameBreak
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import utils
    from reportlab.lib.utils import simpleSplit
except Exception as e:
    canvas = None  # we will detect and instruct the user to install reportlab

ONE_PAGE_TARGET_WORDS = 450  # final one-pager target

@dataclass
class PdfChunk:
    file: pathlib.Path
    chunk_index: int
    text: str

def list_sections(course_dir: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in sorted(course_dir.iterdir()) if p.is_dir()]

def pick_section(sections: List[pathlib.Path]) -> pathlib.Path:
    print("Which section (module) do you want to review?\n")
    for i, sec in enumerate(sections, 1):
        print(f"[{i}] {sec.name}")
    while True:
        sel = input("\nEnter number (or 0 to exit): ").strip()
        if not sel.isdigit():
            print("Please enter a number.")
            continue
        idx = int(sel)
        if idx == 0:
            sys.exit(0)
        if 1 <= idx <= len(sections):
            return sections[idx-1]
        print("Invalid selection. Try again.")

def find_pdfs(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in root.rglob("*.pdf") if p.is_file()]

def extract_text_from_pdf(pdf_path: pathlib.Path, max_pages: int = None) -> str:
    parts = []
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    use_n = total if max_pages is None else min(total, max_pages)
    for i in range(use_n):
        try:
            parts.append(reader.pages[i].extract_text() or "")
        except Exception:
            continue
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


def summarize_chunk(client: OpenAI, model: str, ch: PdfChunk) -> str:
    prompt = f"""You are summarizing a reading for a Berkeley executive/CTO program.
Your job is not only to compress content, but to surface what will drive a **serious class discussion**.

Document: {ch.file.name} (chunk {ch.chunk_index})

From the text below, produce 3–6 tight bullet points (total 90–140 words) that cover:

1) **Core idea / mechanism**
   - What is the main claim, model, or mechanism?
   - Include concrete numbers or examples if present (e.g., 1.5% savings, 64% suppliers, cloud run-rate).

2) **Strategic tension or trade-off**
   - Cash cow vs. growth? Margin vs. scale? Control vs. trust? Speed vs. risk? Build vs. partner?
   - Note any implicit assumptions that could be challenged.

3) **Implications for this course’s themes**
   - Portfolio thinking (EOL, pricing, bets, J-curve).
   - Startup–corporate partnerships (power imbalance, termination, IP/data risk).
   - Automation/AI in negotiations and supplier relationships (trust, deadweight loss, contract design).
   - Governance, incentives, performance vs. power metrics.
   - The above are just examples. Do not need to literally include all points above. please be flexible.

4) **Discussion hooks**
   - 1–2 sharp questions or critiques that a thoughtful participant or professor might raise
     (e.g., “When does this harm supplier trust?”, “What if dependency risk flips?”).

Constraints:
- Bulleted format only.
- No generic study tips, no citations/URLs.
- Do NOT hallucinate cases: only use what can be reasonably inferred from the text.

TEXT START
{ch.text}
TEXT END
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an exacting teaching assistant for a graduate executive class. "
                    "You extract decision-grade insights, tensions, and debate prompts while staying faithful to the text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()



def summarize_document(client: OpenAI, model: str, file_path: pathlib.Path, per_file_page_cap: int, chunk_chars: int) -> str:
    """Return the per-document summary (≈250–400 words) built from chunk digests."""
    text = extract_text_from_pdf(file_path, max_pages=per_file_page_cap)
    if not text.strip():
        return ""
    chunks_text = chunk_text(text, max_chars=chunk_chars)
    digests = []
    for j, t in enumerate(chunks_text, 1):
        digests.append(summarize_chunk(client, model, PdfChunk(file=file_path, chunk_index=j, text=t)))
    # Compress digests to a single per-doc summary
    joined = "\n\n---\n\n".join(digests)
    prompt = f"""Combine the notes below into a single concise summary (≈300 words) of the document "{file_path.name}".
Highlight: key arguments, frameworks, examples, and 3–5 bullet takeaways at the end.
NOTES:
{joined}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert note condenser and editor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def synthesize_section_one_pager(client: OpenAI, model: str, section_name: str, per_doc_summaries: List[Tuple[str, str]]) -> str:
    """per_doc_summaries: list of (filename, summary_markdown)"""
    bullets = "\n".join([f"- {name}" for (name, _) in per_doc_summaries])
    joined = "\n\n---\n\n".join([f"### {name}\n\n{summ}" for (name, summ) in per_doc_summaries])
    prompt = f"""Create a single-page executive review (≈{ONE_PAGE_TARGET_WORDS} words) for the section "{section_name}".

Source documents:
{bullets}

Use the document summaries below as your notes. Your output must be self-contained and suitable for a busy CTO. Structure:
- Title
- 4–6 bullet key takeaways (impact-focused, specific)
- Core concepts (short paragraph)
- Frameworks/mental models (bulleted, with brief 'when to use')
- Action checklist for the coming week (5–7 items, imperative verbs)
- Reflection question (1 line)

Write in crisp, plain business English. Do not include citations or URLs.

DOCUMENT SUMMARIES:
{joined}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert executive coach and editor who writes concise, actionable one‑page briefs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# -------- PDF helpers (ReportLab) --------
def _wrap_text_to_lines(text: str, width_chars: int = 95) -> List[str]:
    # Fallback very simple wrapping for Type1 fonts
    lines = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        s = para
        while len(s) > width_chars:
            cut = s.rfind(" ", 0, width_chars)
            if cut <= 0:
                cut = width_chars
            lines.append(s[:cut].rstrip())
            s = s[cut:].lstrip()
        lines.append(s)
    return lines

def write_pdf_reportlab(text: str, out_path: pathlib.Path, title: str = None):
    if canvas is None:
        raise RuntimeError("ReportLab is not installed. pip install reportlab")
    # Simple single-column text PDF
    c = canvas.Canvas(str(out_path), pagesize=LETTER)
    width, height = LETTER
    left = 0.75 * inch
    right = width - 0.75 * inch
    top = height - 0.75 * inch
    bottom = 0.75 * inch
    y = top

    c.setFont("Times-Bold", 14)
    if title:
        title_lines = _wrap_text_to_lines(title, width_chars=80)
        for line in title_lines:
            if y < bottom + 20:
                c.showPage(); y = top; c.setFont("Times-Bold", 14)
            c.drawString(left, y, line)
            y -= 18
        y -= 8

    c.setFont("Times-Roman", 11)
    body_lines = _wrap_text_to_lines(text, width_chars=95)
    for line in body_lines:
        if y < bottom + 16:
            c.showPage(); y = top; c.setFont("Times-Roman", 11)
        if line.strip() == "":
            y -= 8
        c.drawString(left, y, line)
        y -= 14

    c.showPage()
    c.save()

def save_markdown_and_pdf(markdown_text: str, md_path: pathlib.Path, pdf_path: pathlib.Path, title: str):
    md_path.write_text(markdown_text, encoding="utf-8")
    try:
        write_pdf_reportlab(markdown_text, pdf_path, title=title)
    except Exception as e:
        sys.stderr.write(f"WARNING: Could not write PDF ({pdf_path.name}): {e}\n")

# -------------- Main workflow --------------
def list_sections_and_pick(course_dir: pathlib.Path) -> pathlib.Path:
    sections = list_sections(course_dir)
    if not sections:
        sys.stderr.write("No sections found. Did you run canvas_dump.py?\n")
        sys.exit(2)
    return pick_section(sections)

def main():
    ap = argparse.ArgumentParser(description="Summarize each PDF independently, then create a section one‑pager. Save as Markdown + PDF.")
    ap.add_argument("--course-dir", required=True, help="Path to course root (from canvas_dump.py)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    ap.add_argument("--per-file-page-cap", type=int, default=30, help="Max pages extracted per PDF (default 30)")
    ap.add_argument("--chunk-chars", type=int, default=8000, help="Max characters per chunk sent to the model (default 8000)")
    ap.add_argument("--override", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--section-out", default="section_one_pager.md", help="Section summary filename (Markdown)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("ERROR: Please set OPENAI_API_KEY.\n")
        sys.exit(2)

    course_dir = pathlib.Path(args.course_dir).resolve()
    if not course_dir.exists():
        sys.stderr.write(f"ERROR: {course_dir} does not exist.\n")
        sys.exit(2)

    section_dir = list_sections_and_pick(course_dir)
    print(f"\nSelected section: {section_dir.name}\n")

    # Gather PDFs
    pdfs = find_pdfs(section_dir)
    if not pdfs:
        print("No PDFs found in this section.")
        return

    client = OpenAI(api_key=api_key)

    per_doc_summaries: List[Tuple[str, str]] = []  # (filename, summary)
    for pdf_path in pdfs:
        # Paths for per-doc outputs
        stem = pdf_path.stem
        doc_md = section_dir / f"{stem}.summary.md"
        doc_pdf = section_dir / f"{stem}.summary.pdf"

        if doc_md.exists() and doc_pdf.exists() and not args.override:
            print(f"[skip] {pdf_path.name} (summaries already exist)")
            per_doc_summaries.append((pdf_path.name, doc_md.read_text(encoding='utf-8')))
            continue

        print(f"[doc] Summarizing: {pdf_path.name}")
        summary_markdown = summarize_document(client, args.model, pdf_path, args.per_file_page_cap, args.chunk_chars)
        if not summary_markdown.strip():
            print(f"[warn] No extractable text for {pdf_path.name}; skipped.")
            continue

        save_markdown_and_pdf(summary_markdown, doc_md, doc_pdf, title=f"{pdf_path.name} — Summary")
        per_doc_summaries.append((pdf_path.name, summary_markdown))

    if not per_doc_summaries:
        print("No per-document summaries were produced.")
        return

    # Section-level one‑pager
    section_md_path = section_dir / args.section_out
    section_pdf_path = section_md_path.with_suffix(".pdf")

    if section_md_path.exists() and section_pdf_path.exists() and not args.override:
        print(f"[skip] Section one‑pager already exists: {section_md_path}")
        print(str(section_md_path))
        return

    print("\nSynthesizing section one‑pager...")
    one_pager = synthesize_section_one_pager(
        client, args.model, section_dir.name, per_doc_summaries
    )
    save_markdown_and_pdf(one_pager, section_md_path, section_pdf_path, title=f"{section_dir.name} — One‑Page Review")

    print(f"\nSaved per-document summaries and section one‑pager under:\n{section_dir}\n")
    print(f"Section one‑pager: {section_md_path}\nPDF: {section_pdf_path}\n")

if __name__ == "__main__":
    main()
