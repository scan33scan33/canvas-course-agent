#!/usr/bin/env python3
"""
Canvas Course Downloader (Modules -> Week directories)

Features:
- Lists modules for a course and creates one directory per module (e.g., "Module 1: Week 2 (July 16, 2025)")
- Recursively fetches each module's items (files, pages, external links, assignments/discussions metadata)
- Downloads content only if not already present, unless --override is provided
- Handles pagination for Canvas API
- Interactive "review" prompt so you can pick a week's module to quickly preview items after download

Usage:
  export CANVAS_TOKEN="YOUR_API_TOKEN"
  python canvas_dump.py --base https://your-canvas.example.edu --course COURSE_ID --out ./my_course --per-page 100
  # Force re-downloads:
  python canvas_dump.py --base https://your-canvas.example.edu --course COURSE_ID --out ./my_course --override

Notes:
- Token must have at least read permissions for the course.
- File items are downloaded; Page/Assignment/Discussion/Quiz are saved as HTML/JSON/link stubs so you can review offline.

Improvements (follow links):
- Pages: after saving page HTML, parses it for links to Canvas files (e.g. Quick Reference PDFs, Video Transcripts)
  and downloads those files into the same module directory (--follow-page-links, default on).
- Pages: extracts iframe src and file links into a .urls.txt sidecar so you know what embedded content exists.
- Quizzes: fetches quiz questions via API and saves to .quiz.questions.json for full question text.
"""

import argparse
import os
import sys
import time
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple
import urllib.parse

import requests

# --------------- Helpers ---------------

def getenv_token() -> str:
    tok = os.environ.get("CANVAS_TOKEN")
    if not tok:
        sys.stderr.write("ERROR: Please set CANVAS_TOKEN environment variable.\n")
        sys.exit(2)
    return tok

def join_url(base: str, path: str) -> str:
    base = base.rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"

def paged_get(session: requests.Session, url: str, params: Optional[dict] = None) -> List[dict]:
    """Follow Canvas-style Link pagination."""
    items = []
    while url:
        resp = session.get(url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"GET {url} -> {resp.status_code} {resp.text[:300]}")
        try:
            batch = resp.json()
        except Exception:
            raise RuntimeError(f"Non-JSON response from {url}: {resp.text[:300]}")
        if isinstance(batch, list):
            items.extend(batch)
        else:
            items.append(batch)

        # Parse Link header
        link = resp.headers.get("Link", "")
        next_url = None
        if link:
            # Format: <url1>; rel="current", <url2>; rel="next", ...
            for part in link.split(","):
                m = re.search(r'<([^>]+)>;\s*rel="next"', part)
                if m:
                    next_url = m.group(1)
                    break
        url = next_url
        params = None  # only on first page
    return items

def safe_filename(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = name[:200]  # avoid extreme paths
    return name

def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: pathlib.Path, text: str, override: bool) -> None:
    if path.exists() and not override:
        return
    path.write_text(text, encoding="utf-8")

def write_json(path: pathlib.Path, obj, override: bool) -> None:
    if path.exists() and not override:
        return
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def stream_download(session: requests.Session, url: str, dest: pathlib.Path, override: bool) -> None:
    if dest.exists() and not override:
        return
    with session.get(url, stream=True) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"Download {url} -> {r.status_code} {r.text[:300]}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

# --------------- Canvas Fetchers ---------------

def get_modules(session: requests.Session, base: str, course_id: str, per_page: int) -> List[dict]:
    url = join_url(base, f"/api/v1/courses/{course_id}/modules")
    return paged_get(session, url, params={"per_page": per_page})

def get_module_items(session: requests.Session, base: str, course_id: str, module_id: int, per_page: int) -> List[dict]:
    url = join_url(base, f"/api/v1/courses/{course_id}/modules/{module_id}/items")
    return paged_get(session, url, params={"per_page": per_page, "include[]": ["content_details"]})

def get_file_meta(session: requests.Session, base: str, file_id: int) -> dict:
    url = join_url(base, f"/api/v1/files/{file_id}")
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET file {file_id} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()

def get_page(session: requests.Session, base: str, course_id: str, page_url: str) -> dict:
    url = join_url(base, f"/api/v1/courses/{course_id}/pages/{page_url}")
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET page {page_url} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()

def get_assignment(session: requests.Session, base: str, course_id: str, assignment_id: int) -> dict:
    url = join_url(base, f"/api/v1/courses/{course_id}/assignments/{assignment_id}")
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET assignment {assignment_id} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()

def get_discussion(session: requests.Session, base: str, course_id: str, topic_id: int) -> dict:
    url = join_url(base, f"/api/v1/courses/{course_id}/discussion_topics/{topic_id}")
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET discussion {topic_id} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()

def get_quiz(session: requests.Session, base: str, course_id: str, quiz_id: int) -> dict:
    url = join_url(base, f"/api/v1/courses/{course_id}/quizzes/{quiz_id}")
    resp = session.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"GET quiz {quiz_id} -> {resp.status_code} {resp.text[:300]}")
    return resp.json()

def get_quiz_questions(session: requests.Session, base: str, course_id: str, quiz_id: int, per_page: int = 50) -> List[dict]:
    """Fetch all questions for a quiz (paginated). Fails gracefully if endpoint not allowed."""
    url = join_url(base, f"/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions")
    try:
        return paged_get(session, url, params={"per_page": per_page})
    except Exception as e:
        # Some Canvas instances or New Quizzes may not expose this
        return []

def extract_canvas_file_ids_from_html(html: str, base: str, course_id: str) -> List[int]:
    """Find Canvas file IDs linked from same course in HTML (href or data-api-endpoint)."""
    cid = re.escape(str(course_id))
    # Match /courses/:course_id/files/:id or /api/v1/courses/:course_id/files/:id
    pattern = rf"/courses/{cid}/files/(\d+)|/api/v1/courses/{cid}/files/(\d+)"
    seen = set()
    for m in re.finditer(pattern, html, re.IGNORECASE):
        fid = int(m.group(1) or m.group(2))
        if fid not in seen:
            seen.add(fid)
    return sorted(seen)

def extract_embedded_urls_from_html(html: str) -> List[str]:
    """Extract iframe src and common embed URLs from HTML for reference (no fetch)."""
    urls = []
    # iframe src="..."
    for m in re.finditer(r'<iframe[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE):
        urls.append(m.group(1).strip())
    # data-api-endpoint (Canvas API URLs) - often File/Page
    for m in re.finditer(r'data-api-endpoint=["\']([^"\']+)["\']', html, re.IGNORECASE):
        urls.append(m.group(1).strip())
    return urls

def download_linked_canvas_files(
    session: requests.Session,
    base: str,
    course_id: str,
    mod_dir: pathlib.Path,
    html: str,
    override: bool,
) -> List[str]:
    """Parse HTML for Canvas file links, download those files into mod_dir. Returns list of downloaded filenames."""
    file_ids = extract_canvas_file_ids_from_html(html, base, course_id)
    downloaded = []
    for fid in file_ids:
        try:
            meta = get_file_meta(session, base, fid)
            fname = meta.get("filename") or meta.get("display_name") or f"file_{fid}"
            fname = safe_filename(fname)
            dest = mod_dir / fname
            if dest.exists() and not override:
                downloaded.append(fname)
                continue
            # Avoid overwriting a different file: use linked_FID_name if plain name exists
            if dest.exists() and override:
                dest = mod_dir / f"linked_{fid}_{fname}"
            download_url = meta.get("url") or meta.get("public_url") or meta.get("html_url")
            if not download_url:
                continue
            stream_download(session, download_url, dest, override=True)
            downloaded.append(dest.name)
        except Exception as e:
            # Log but don't fail the whole page
            err_path = mod_dir / f"linked_file_{fid}.ERROR.txt"
            write_text(err_path, f"{e}\n", True)
    return downloaded

# --------------- Core Logic ---------------

SUPPORTED_TYPES = {"File", "Page", "ExternalUrl", "Assignment", "Discussion", "Quiz"}

def process_item(
    session: requests.Session,
    base: str,
    course_id: str,
    item: dict,
    mod_dir: pathlib.Path,
    override: bool,
    follow_page_links: bool = True,
    fetch_quiz_questions: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Returns (status, preview_path) where preview_path is relative path string to a saved artifact.
    """
    itype = item.get("type")
    title = item.get("title") or itype
    title_safe = safe_filename(title)
    preview_rel = None

    try:
        if itype == "File":
            fid = item.get("content_id")
            meta = get_file_meta(session, base, fid)
            # Prefer original filename if available
            fname = meta.get("filename") or meta.get("display_name") or f"file_{fid}"
            fname = safe_filename(fname)
            dest = mod_dir / fname
            download_url = meta.get("url") or meta.get("public_url")
            if not download_url:
                # Fallback to html_url (will redirect to auth; still works with session cookies)
                download_url = meta.get("html_url")
            stream_download(session, download_url, dest, override)
            preview_rel = dest.name
            return (f"Downloaded file: {dest.name}", preview_rel)

        elif itype == "Page":
            page_url = item.get("page_url")  # slug identifier
            page = get_page(session, base, course_id, page_url)
            html = page.get("body") or ""
            dest = mod_dir / f"{title_safe}.html"
            write_text(dest, html, override)
            preview_rel = dest.name
            # Extract embedded URLs (iframes, data-api-endpoint) for reference
            embedded = extract_embedded_urls_from_html(html)
            if embedded:
                urls_dest = mod_dir / f"{title_safe}.urls.txt"
                write_text(urls_dest, "\n".join(embedded) + "\n", override)
            # Optionally download Canvas files linked from this page (e.g. PDFs, transcripts)
            if follow_page_links and html:
                linked = download_linked_canvas_files(session, base, course_id, mod_dir, html, override)
                if linked:
                    return (f"Saved page HTML: {dest.name} (+ {len(linked)} linked file(s): {', '.join(linked[:3])}{'...' if len(linked) > 3 else ''})", preview_rel)
            return (f"Saved page HTML: {dest.name}", preview_rel)

        elif itype == "ExternalUrl":
            ext = item.get("external_url")
            dest = mod_dir / f"{title_safe}.link.txt"
            write_text(dest, f"{ext}\n", override)
            preview_rel = dest.name
            return (f"Wrote external link: {dest.name}", preview_rel)

        elif itype == "Assignment":
            aid = item.get("content_id")
            a = get_assignment(session, base, course_id, aid)
            # Save JSON + description HTML if available
            write_json(mod_dir / f"{title_safe}.assignment.json", a, override)
            if a.get("description"):
                write_text(mod_dir / f"{title_safe}.assignment.html", a["description"], override)
            preview_rel = f"{title_safe}.assignment.json"
            return (f"Saved assignment metadata: {title_safe}.assignment.json", preview_rel)

        elif itype == "Discussion":
            did = item.get("content_id")
            d = get_discussion(session, base, course_id, did)
            write_json(mod_dir / f"{title_safe}.discussion.json", d, override)
            if d.get("message"):
                write_text(mod_dir / f"{title_safe}.discussion.html", d["message"], override)
            preview_rel = f"{title_safe}.discussion.json"
            return (f"Saved discussion metadata: {title_safe}.discussion.json", preview_rel)

        elif itype == "Quiz":
            qid = item.get("content_id")
            q = get_quiz(session, base, course_id, qid)
            write_json(mod_dir / f"{title_safe}.quiz.json", q, override)
            preview_rel = f"{title_safe}.quiz.json"
            if fetch_quiz_questions:
                questions = get_quiz_questions(session, base, course_id, qid, per_page=50)
                if questions:
                    write_json(mod_dir / f"{title_safe}.quiz.questions.json", questions, override)
                    return (f"Saved quiz metadata + {len(questions)} questions: {title_safe}.quiz.json", preview_rel)
            return (f"Saved quiz metadata: {title_safe}.quiz.json", preview_rel)

        else:
            # default: save the item JSON so nothing is lost
            write_json(mod_dir / f"{title_safe}.item.json", item, override)
            preview_rel = f"{title_safe}.item.json"
            return (f"Saved item JSON (unsupported type {itype})", preview_rel)
    except Exception as e:
        err_path = mod_dir / f"{title_safe}.ERROR.txt"
        write_text(err_path, f"{e}\n", True)
        return (f"ERROR on {title}: {e}", None)

def download_course(args) -> List[dict]:
    token = getenv_token()
    headers = {"Authorization": f"Bearer {token}"}

    session = requests.Session()
    session.headers.update(headers)

    outdir = pathlib.Path(args.out).resolve()
    ensure_dir(outdir)

    modules = get_modules(session, args.base, args.course, args.per_page)
    # Save manifest of modules
    manifest = {"course": args.course, "base": args.base, "downloaded_at": time.time(), "module_count": len(modules), "modules": modules}
    write_json(outdir / "manifest.modules.json", manifest, override=True)

    results = []
    for m in modules:
        name = m.get("name") or f"module_{m.get('id')}"
        module_dir = outdir / safe_filename(name)
        ensure_dir(module_dir)

        items = get_module_items(session, args.base, args.course, m["id"], args.per_page)
        write_json(module_dir / "manifest.items.json", items, override=True)

        # Per-item saving
        previews = []
        for it in items:
            status, preview_rel = process_item(
                session, args.base, args.course, it, module_dir, args.override,
                follow_page_links=args.follow_page_links,
                fetch_quiz_questions=args.quiz_questions,
            )
            previews.append({"title": it.get("title"), "type": it.get("type"), "status": status, "preview": preview_rel})
            print(status)
        results.append({"module": name, "dir": str(module_dir), "items": previews})

    write_json(outdir / "download.summary.json", results, override=True)
    return results

def pick_module_to_review(summaries: List[dict]) -> None:
    print("\nWhich week's module do you want to review?")
    for idx, m in enumerate(summaries, start=1):
        print(f"[{idx}] {m['module']}  ->  {m['dir']}")
    try:
        choice = int(input("Enter a number (or 0 to exit): ").strip())
    except Exception:
        return
    if choice <= 0 or choice > len(summaries):
        return
    m = summaries[choice - 1]
    # Show quick preview of items we saved
    print(f"\nPreview for: {m['module']}")
    for it in m["items"][:20]:
        print(f" - ({it['type']}) {it['title']}  [{it['status']}]")
    if len(m["items"]) > 20:
        print(f" ... and {len(m['items']) - 20} more")

def main():
    parser = argparse.ArgumentParser(description="Canvas course content downloader (modules -> week directories).")
    parser.add_argument("--base", required=True, help="Canvas base URL (e.g., https://your-canvas.example.edu)")
    parser.add_argument("--course", required=True, help="Course ID (numeric)")
    parser.add_argument("--out", default="./canvas_course", help="Output directory")
    parser.add_argument("--per-page", type=int, default=100, help="API page size (default 100)")
    parser.add_argument("--override", action="store_true", help="Override existing files")
    parser.add_argument("--no-follow-page-links", dest="follow_page_links", action="store_false", help="Do not download Canvas files linked from pages (e.g. PDFs, transcripts)")
    parser.add_argument("--no-quiz-questions", dest="quiz_questions", action="store_false", help="Do not fetch quiz questions (only save quiz metadata)")
    args = parser.parse_args()

    try:
        summaries = download_course(args)
        pick_module_to_review(summaries)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        sys.stderr.write(f"FATAL: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
