#!/usr/bin/env python3
"""
make_review.py — Interactive "pick a section and generate a one‑page review" script

What it does
- Asks you to select a section (module) directory from a Canvas course dump (from canvas_dump.py)
- Recursively finds all PDFs under that section
- Extracts text locally and sends concise context to OpenAI to generate a one‑page review
- Skips rework if an output file already exists unless --override is provided

Prereqs
- pip install openai pypdf
- export OPENAI_API_KEY=sk-xxxx

Usage
  python make_review.py --course-dir ./cto_course --model gpt-4o-mini
  python make_review.py --course-dir ./cto_course --override
"""

import argparse
import os
import sys
import pathlib
from typing import List, Tuple
from dataclasses import dataclass

# --- third-party ---
try:
    from pypdf import PdfReader
except Exception as e:
    sys.stderr.write("ERROR: Please install pypdf (pip install pypdf)\n")
    raise

try:
    from openai import OpenAI
except Exception as e:
    sys.stderr.write("ERROR: Please install openai (pip install openai)\n")
    raise

# --------------------

ONE_PAGE_TARGET_WORDS = 450  # rough target

@dataclass
class PdfChunk:
    file: pathlib.Path
    chunk_index: int
    text: str

def list_sections(course_dir: pathlib.Path) -> List[pathlib.Path]:
    # A section is any immediate subdirectory with files inside
    sections = [p for p in sorted(course_dir.iterdir()) if p.is_dir()]
    return sections

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
    text_parts = []
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    total = len(pages)
    use_n = total if max_pages is None else min(total, max_pages)
    for i in range(use_n):
        try:
            text_parts.append(pages[i].extract_text() or "")
        except Exception:
            # If extract fails for a page, just skip it
            continue
    return "\n".join(text_parts)

def chunk_text(s: str, max_chars: int = 8000) -> List[str]:
    s = " ".join(s.split())  # collapse whitespace a bit
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        start = end
    return chunks

def summarize_chunks(client: OpenAI, model: str, chunks: List[PdfChunk]) -> List[str]:
    """Summarize each chunk down to ~120 words to build a compact digest."""
    digests = []
    for i, ch in enumerate(chunks, 1):
        prompt = f"""You are distilling a course PDF into a compact digest for later synthesis.
Document: {ch.file.name} (chunk {ch.chunk_index})
Task: Summarize the *key ideas* and *actionable takeaways* from the text below in 90–120 words. Avoid fluff. Preserve terminology, frameworks, and numbered lists if present.
TEXT START
{ch.text}
TEXT END"""
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise summarizer for graduate-level executive education content."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        digests.append(resp.choices[0].message.content.strip())
        print(ch.file.name, digests[-1])
    return digests

def synthesize_one_pager(client: OpenAI, model: str, section_name: str, pdf_filenames: List[str], digests: List[str]) -> str:
    joined = "\n\n---\n\n".join(digests)
    prompt = f"""Create a single-page executive review (≈{ONE_PAGE_TARGET_WORDS} words) for the section "{section_name}".
Source documents:
- """ + "\n- ".join(pdf_filenames) + f"""

Use the digests below as source notes. Your output must be *self-contained* and suitable for a busy CTO. Structure:
- Title
- 4–6 bullet key takeaways (impact-focused, specific)
- Core concepts (short paragraph)
- Frameworks/mental models (bulleted, with brief 'when to use')
- Action checklist for the coming week (5–7 items, imperative verbs)
- Reflection question (1 line)

Write in crisp, plain business English. Avoid quoting text. Do not include citations or urls.

DIGEST NOTES:
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

def collect_pdf_chunks(section_dir: pathlib.Path, per_file_page_cap: int, max_chars_per_chunk: int):
    pdfs = find_pdfs(section_dir)
    filenames = [p.name for p in pdfs]
    chunks = []
    for f in pdfs:
        text = extract_text_from_pdf(f, max_pages=per_file_page_cap)
        if not text.strip():
            continue
        for j, ch in enumerate(chunk_text(text, max_chars=max_chars_per_chunk), 1):
            chunks.append(PdfChunk(file=f, chunk_index=j, text=ch))
    return chunks, filenames

def main():
    import textwrap
    ap = argparse.ArgumentParser(description="Pick a section and generate a one‑page review using OpenAI.")
    ap.add_argument("--course-dir", required=True, help="Path to course root (from canvas_dump.py)")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    ap.add_argument("--per-file-page-cap", type=int, default=30, help="Max pages extracted per PDF (default 30). Increase if needed.")
    ap.add_argument("--chunk-chars", type=int, default=8000, help="Max characters per chunk sent to model (default 8000)")
    ap.add_argument("--override", action="store_true", help="Overwrite existing one-pager if present")
    ap.add_argument("--out-name", default="section_review.md", help="Output filename within the section directory")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("ERROR: Please set OPENAI_API_KEY.\n")
        sys.exit(2)

    course_dir = pathlib.Path(args.course_dir).resolve()
    if not course_dir.exists():
        sys.stderr.write(f"ERROR: {course_dir} does not exist.\n")
        sys.exit(2)

    sections = list_sections(course_dir)
    if not sections:
        sys.stderr.write("No sections (subdirectories) found. Did you run canvas_dump.py?\n")
        sys.exit(2)

    section = pick_section(sections)
    out_path = section / args.out_name
    if out_path.exists() and not args.override:
        print(f"Output already exists: {out_path} (use --override to regenerate)")
        print(str(out_path))
        return

    print(f"\nCollecting PDFs under: {section}")
    chunks, pdf_filenames = collect_pdf_chunks(section, args.per_file_page_cap, args.chunk_chars)
    if not chunks:
        print("No PDF text found in this section. (Are the materials files or links?)")
        return

    print(f"Found {len(pdf_filenames)} PDFs, {len(chunks)} text chunks. Summarizing...")

    client = OpenAI(api_key=api_key)

    # Stage 1: digest per chunk
    digests = summarize_chunks(client, args.model, chunks)

    # Stage 2: synthesize one‑page
    print("Synthesizing final one‑pager...")
    one_pager = synthesize_one_pager(client, args.model, section.name, pdf_filenames, digests)

    out_path.write_text(one_pager, encoding="utf-8")
    print(f"\nSaved: {out_path}\n")

    # Also echo the first ~50 lines so user gets a quick view
    preview = "\n".join(out_path.read_text(encoding="utf-8").splitlines()[:50])
    print("Preview:\n" + "-"*60 + "\n" + preview + "\n" + "-"*60)

if __name__ == "__main__":
    main()
