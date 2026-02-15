#!/usr/bin/env python3
"""
make_review.py

Generates AI-assisted review materials for Canvas course modules:
- Reads PDF, HTML, and JSON files
- Extracts assignment text often hidden in Canvas HTML pages
- Produces two summaries (module overview + assignment summary), strategy brief, and example answers
- Persona and prompts are loaded from text files (persona.txt, prompt_*.txt)

Usage:
  python make_review.py --course-dir ./ba_course
"""

import argparse
import os
import sys
import pathlib
import json
from typing import List, Tuple
from dataclasses import dataclass

# --- Hashlib patch ---
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
    from bs4 import BeautifulSoup  # New for HTML
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except Exception as e:
    sys.stderr.write(f"ERROR: Missing dependencies. {e}\nTry: pip install openai pypdf reportlab beautifulsoup4\n")
    sys.exit(1)

# --- CONFIGURATION ---
# Load persona and prompts from text files
SCRIPT_DIR = pathlib.Path(__file__).parent

def load_persona() -> str:
    """Load persona from persona.txt or env var BAI_REVIEW_PERSONA."""
    env_persona = os.environ.get("BAI_REVIEW_PERSONA")
    if env_persona:
        return env_persona.strip()
    persona_file = SCRIPT_DIR / "persona.txt"
    if persona_file.exists():
        return persona_file.read_text(encoding="utf-8").strip()
    # Fallback default
    return (
        "The student is an MLE Area Tech Lead at Coupang. "
        "They are an expert in Transformer model pretraining, Deep ML, and large-scale recommendation systems. "
        "They are taking this Business Analytics class to bridge technical depth with business decision-making."
    )

def load_prompt(filename: str) -> str:
    """Load a prompt template from a text file."""
    prompt_file = SCRIPT_DIR / filename
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt file not found: {filename}")

def load_system_prompts() -> dict:
    """Load system prompts from system_prompts.txt."""
    system_file = SCRIPT_DIR / "system_prompts.txt"
    prompts = {}
    if system_file.exists():
        for line in system_file.read_text(encoding="utf-8").strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                prompts[key.strip()] = value.strip()
    # Fallbacks
    prompts.setdefault("summarize_chunk", "You are a helpful teaching assistant.")
    prompts.setdefault("summarize_document", "You are a concise editor.")
    prompts.setdefault("synthesize_plan", "You are a technical mentor bridging MBA business logic and Deep Learning engineering.")
    prompts.setdefault("example_answers", "You are a helpful teaching assistant who writes clear, correct example solutions.")
    return prompts

USER_PERSONA = load_persona()
SYSTEM_PROMPTS = load_system_prompts()

@dataclass
class DocChunk:
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

# --- FILE HANDLING (Updated) ---

def find_files(root: pathlib.Path) -> List[pathlib.Path]:
    # We now look for pdf, html, and json
    extensions = {".pdf", ".html", ".json"}
    found = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            found.append(p)
    return sorted(found)

def extract_text(file_path: pathlib.Path, max_pages: int = 30) -> str:
    ext = file_path.suffix.lower()
    
    # 1. HTML Extraction
    if ext == ".html":
        try:
            soup = BeautifulSoup(file_path.read_text(errors='ignore'), 'html.parser')
            # Kill scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            return text
        except Exception as e:
            return f"[Error reading HTML: {e}]"

    # 2. JSON Extraction
    elif ext == ".json":
        try:
            data = json.loads(file_path.read_text(errors='ignore'))
            # If it's a list or dict, dump it to string; try to grab 'body' if generic canvas json
            if isinstance(data, dict):
                # Prioritize content fields usually found in Canvas JSONs
                content = []
                for key in ['title', 'description', 'body', 'message']:
                    if key in data and data[key]:
                        val = data[key]
                        # sometimes body is HTML string
                        if "<" in str(val) and ">" in str(val):
                            soup = BeautifulSoup(str(val), 'html.parser')
                            content.append(soup.get_text())
                        else:
                            content.append(str(val))
                if content:
                    return "\n".join(content)
                return json.dumps(data, indent=2)
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"[Error reading JSON: {e}]"

    # 3. PDF Extraction
    elif ext == ".pdf":
        parts = []
        try:
            reader = PdfReader(str(file_path))
            total = len(reader.pages)
            use_n = min(total, max_pages)
            for i in range(use_n):
                parts.append(reader.pages[i].extract_text() or "")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return "\n".join(parts)
    
    return ""

def chunk_text(s: str, max_chars: int = 8000) -> List[str]:
    # Simple clean up
    s = " ".join(s.split())
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        chunks.append(s[start:end])
        start = end
    return chunks

# --- AI WORKFLOW ---

def summarize_chunk(client: OpenAI, model: str, ch: DocChunk) -> str:
    prompt_template = load_prompt("prompt_summarize_chunk.txt")
    prompt = prompt_template.format(
        file_suffix=ch.file.suffix,
        file_name=ch.file.name,
        chunk_index=ch.chunk_index,
        chunk_text=ch.text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["summarize_chunk"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def summarize_document(client: OpenAI, model: str, file_path: pathlib.Path, per_file_page_cap: int, chunk_chars: int) -> str:
    text = extract_text(file_path, max_pages=per_file_page_cap)
    if not text or len(text) < 50: return "" # Skip empty files
    
    chunks_text = chunk_text(text, max_chars=chunk_chars)
    digests = []
    
    for j, t in enumerate(chunks_text, 1):
        digests.append(summarize_chunk(client, model, DocChunk(file=file_path, chunk_index=j, text=t)))
    
    joined = "\n\n".join(digests)
    prompt_template = load_prompt("prompt_summarize_document.txt")
    prompt = prompt_template.format(
        file_name=file_path.name,
        joined_digests=joined
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["summarize_document"]},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def synthesize_custom_plan(client: OpenAI, model: str, section_name: str, per_doc_summaries: List[Tuple[str, str]]) -> str:
    joined_notes = "\n\n---\n\n".join([f"### File: {name}\n{summ}" for (name, summ) in per_doc_summaries])
    
    prompt_template = load_prompt("prompt_synthesize_plan.txt")
    prompt = prompt_template.format(
        user_persona=USER_PERSONA,
        joined_notes=joined_notes
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["synthesize_plan"]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def generate_example_answers(
    client: OpenAI,
    model: str,
    section_name: str,
    per_doc_summaries: List[Tuple[str, str]],
    strategy_plan: str,
) -> str:
    """Generate example/sample answers to the assignment based on course materials and strategy."""
    joined_notes = "\n\n---\n\n".join([f"### File: {name}\n{summ}" for (name, summ) in per_doc_summaries])
    
    prompt_template = load_prompt("prompt_example_answers.txt")
    prompt = prompt_template.format(
        user_persona=USER_PERSONA,
        strategy_plan=strategy_plan,
        joined_notes=joined_notes
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS["example_answers"]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# --- PDF Gen ---
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
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, title)
    y -= 25
    
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
    
    files = find_files(section_dir)
    if not files:
        print("No compatible files (pdf/html/json) found.")
        return

    client = OpenAI(api_key=api_key)
    per_doc_summaries = []

    print(f"\nScanning {len(files)} files (PDF, HTML, JSON)...\n")

    for f in files:
        md_out = section_dir / f"{f.stem}.summary.md"
        pdf_out = section_dir / f"{f.stem}.summary.pdf"
        
        if md_out.exists() and not args.override:
            # print(f"[Skip] {f.name}")
            per_doc_summaries.append((f.name, md_out.read_text(encoding='utf-8')))
            continue
            
        print(f"Reading: {f.name}")
        summary = summarize_document(client, args.model, f, 30, 8000)
        if summary:
            save_outputs(summary, md_out, pdf_out, f.name)
            per_doc_summaries.append((f.name, summary))
        else:
            print(f"  -> Empty or unreadable, skipping.")

    print("\nSynthesizing Strategy & Homework Plan (two summaries + strategy)...")
    final_out = section_dir / "Section_Strategy_Plan.md"
    final_pdf = section_dir / "Section_Strategy_Plan.pdf"
    
    plan = synthesize_custom_plan(client, args.model, section_dir.name, per_doc_summaries)
    
    save_outputs(plan, final_out, final_pdf, f"Strategy: {section_dir.name}")

    print("Generating example answers to the assignment...")
    example_out = section_dir / "Section_Example_Answers.md"
    example_pdf = section_dir / "Section_Example_Answers.pdf"
    example_answers = generate_example_answers(
        client, args.model, section_dir.name, per_doc_summaries, plan
    )
    save_outputs(example_answers, example_out, example_pdf, f"Example Answers: {section_dir.name}")
    
    print(f"\nDone! Reports saved to:\n  {final_out}\n  {example_out}")

if __name__ == "__main__":
    main()
