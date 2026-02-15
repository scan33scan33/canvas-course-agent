# Canvas Agent

Tools to download Canvas LMS course content and generate AI-assisted review materials: per-document summaries, module strategy briefs, and example assignment answers.

**Do not commit course content.** Use `.gitignore` (course directories and `.env` are ignored by default).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Environment variables (no secrets in repo)

- **Canvas download:** `CANVAS_TOKEN` — Canvas API token (read access to the course).
- **Review scripts:** `OPENAI_API_KEY` — OpenAI API key for summarization and example answers.
- **Optional:** `BAI_REVIEW_PERSONA` — Override the default “student persona” used in Business Analytics review scripts (e.g. role, background). If unset, a default example persona is used.

---

## Scripts

### 1. `canvas_dump.py` — Download a Canvas course by modules

- Lists modules and creates one directory per module.
- Downloads files, page HTML, assignment/discussion/quiz metadata.
- **Follows links:** Parses page HTML for Canvas file links (e.g. PDFs, transcripts) and downloads them; saves iframe/embed URLs to `.urls.txt`; fetches quiz questions when the API allows.

```bash
export CANVAS_TOKEN="your_canvas_api_token"
python canvas_dump.py --base https://your-canvas-instance.edu --course COURSE_ID --out ./my_course
# Re-download / refresh:
python canvas_dump.py --base https://your-canvas-instance.edu --course COURSE_ID --out ./my_course --override
```

Options: `--no-follow-page-links`, `--no-quiz-questions` to disable link-following or quiz-question fetch.

### 2. `make_bai_review_v2.py` — Business Analytics review (PDF/HTML/JSON)

- Reads PDF, HTML, and JSON in a chosen module directory.
- Produces two summaries (module overview + assignment summary), a strategy brief, and **example answers** for the assignment.
- Outputs: per-file `.summary.md`/`.pdf`, `Section_Strategy_Plan.md`/`.pdf`, `Section_Example_Answers.md`/`.pdf`.

```bash
export OPENAI_API_KEY="your_openai_key"
python make_bai_review_v2.py --course-dir ./my_course
```

Use `--override` to regenerate; `--model gpt-4o` (or `gpt-4o-mini`) to choose model.

### 3. Other review scripts

- `make_review_v3.py`, `make_review_v2.py`, `make_review.py` — Variants for PDF-based course review and section one-pagers.
- `make_bai_review.py` — Earlier Business Analytics review (PDF-focused).

Same pattern: `--course-dir` pointing at a course directory (e.g. after running `canvas_dump.py`), and `OPENAI_API_KEY` set.

---

## License

Use and modify as you like. If you redistribute, consider adding a `LICENSE` file (e.g. MIT).
