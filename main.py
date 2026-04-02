"""
Joblin/main.py
FastAPI application for Joblin.

Endpoints
---------
POST /parse/text   — accepts raw CV text (JSON body)
POST /parse/file   — accepts a .txt or .pdf file upload
POST /add-job      — embed and index a job posting into Elasticsearch
POST /enhance      — enhance a resume against a job description using Gemini
GET  /health       — health check
"""

import io
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
import yaml

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Config and prompt config are fixed server-side files — never sent by the client
_BASE_DIR = Path(__file__).parent
_APP_CONFIG = yaml.safe_load((_BASE_DIR / "config.yaml").read_text(encoding="utf-8")) or {}
_PROMPT_CONFIG = yaml.safe_load((_BASE_DIR / "prompt_config.yaml").read_text(encoding="utf-8")) or {}

from parser import parse_cv
from models import JobRequest, JobDocument, ParseTextRequest, ParsedCV, SkillGapRequest, SkillGapResponse, EnhanceRequest, EnhanceResponse 
from embedding_service import generate_job_embedding
from elastic_service import index_job
from Course_recommender import CourseRecommender

# ── optional PDF support (install with: pip install pdfplumber) ──────────────
try:
    import pdfplumber
    _PDF_SUPPORT = True
except ImportError:
    _PDF_SUPPORT = False


app = FastAPI(
    title="Joblin API",
    description="Green Joblin APIZZZ.",
    version="1.0.0",
)

_course_recommender = CourseRecommender()  # reads DATABASE_URL from env



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_pdf(data: bytes) -> str:
    if not _PDF_SUPPORT:
        raise HTTPException(
            status_code=422,
            detail="PDF support is not installed. Run: pip install pdfplumber",
        )
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(page.extract_text(layout=True) or "" for page in pdf.pages)


def _require_non_empty(text: str) -> None:
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="CV text is empty.")


def _build_prompt(prompt_config: dict, app_config: dict) -> str:
    """Assemble a structured prompt string from the server-side YAML config."""
    enhancer_cfg = prompt_config.get("CV_Enhancer", {})
    if not enhancer_cfg:
        raise RuntimeError("Missing 'CV_Enhancer' section in prompt_config.yaml.")

    parts = []

    role = enhancer_cfg.get("role")
    if role:
        parts.append(f"Role:\n{role.strip()}")

    instruction = enhancer_cfg.get("instruction")
    if not instruction:
        raise RuntimeError("Missing 'instruction' in CV_Enhancer config.")
    parts.append(f"Instruction:\n{instruction.strip()}")

    constraints = enhancer_cfg.get("output_constraints")
    if constraints:
        parts.append("Output Constraints:")
        parts.extend(
            [f"- {c}" for c in constraints]
            if isinstance(constraints, list)
            else [str(constraints)]
        )

    tone = enhancer_cfg.get("style_or_tone")
    if tone:
        parts.append("Style & Tone:")
        parts.extend(
            [f"- {t}" for t in tone]
            if isinstance(tone, list)
            else [str(tone)]
        )

    goal = enhancer_cfg.get("goal")
    if goal:
        parts.append(f"Goal:\n{goal.strip()}")

    reasoning_key = enhancer_cfg.get("reasoning_strategy")
    if reasoning_key:
        reasoning_text = app_config.get("reasoning_strategies", {}).get(reasoning_key)
        if reasoning_text:
            parts.append(f"Reasoning Strategy ({reasoning_key}):\n{reasoning_text.strip()}")

    parts.append("Now perform the task as instructed above.")
    return "\n\n".join(parts)


def _extract_llm_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return result.content
    if hasattr(result, "generations"):
        try:
            gens = result.generations
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, list) and first and hasattr(first[0], "text"):
                    return first[0].text
                if hasattr(first, "text"):
                    return first.text
        except Exception:
            pass
    return str(result)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect browser visits to the interactive docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/health", tags=["Utility"])
def health_check():
    """Returns service status and optional PDF support flag."""
    return {"status": "ok", "pdf_support": _PDF_SUPPORT}


@app.post(
    "/parse/text",
    response_model=ParsedCV,
    summary="Parse CV from raw text",
    tags=["Parser"],
)
def parse_text(body: ParseTextRequest):
    """
    Send raw CV text as a JSON body and receive structured data back.

    ```json
    { "text": "Jane Doe\\njane@example.com\\n..." }
    ```
    """
    _require_non_empty(body.text)
    return parse_cv(body.text)


@app.post(
    "/parse/file",
    response_model=ParsedCV,
    summary="Parse CV from an uploaded file",
    tags=["Parser"],
)
async def parse_file(file: UploadFile = File(...)):
    """
    Upload a **.txt** or **.pdf** file and receive structured CV data.

    - `text/plain` files are decoded as UTF-8.
    - `application/pdf` files require the `pdfplumber` package to be installed.
    """
    content_type = file.content_type or ""
    filename = file.filename or ""

    data = await file.read()

    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        text = _extract_text_from_pdf(data)
    elif content_type.startswith("text/") or filename.lower().endswith(".txt"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Send a .txt or .pdf file.",
        )

    _require_non_empty(text)
    return parse_cv(text)


@app.post(
    "/add-job",
    summary="Embed and index a job posting",
    tags=["Jobs"],
)
def add_job(job: JobRequest):
    """
    Generate a vector embedding for a job posting and index it into Elasticsearch.

    ```json
    {
      "job_id" : "123hhdu"
      "job_title": "Data Engineer",
      "job_description": "...",
      "skills": ["Python", "Spark", "AWS"]
    }
    ```
    """
    embedding = generate_job_embedding(job)
    job_doc = JobDocument(
        job_id=job.job_id,
        job_title=job.job_title,
        job_description=job.job_description,
        skills=job.skills,
        embedding=embedding,
    )
    index_job(job_doc)
    return {"message": "Job indexed successfully"}


@app.post(
    "/skill-gap",
    response_model=SkillGapResponse,
    summary="Return skills required by the job but absent from the candidate",
    tags=["Skills"],
)
def skill_gap(body: SkillGapRequest) -> SkillGapResponse:
    """
    Exact comparison (case-insensitive) between job skills and candidate skills.
    Returns only the skills the candidate is missing — no fuzzy matching.

    ```json
    {
      "job_skills": ["Python", "Docker", "Kubernetes"],
      "candidate_skills": ["Python", "Docker"]
    }
    ```
    """
    candidate_normalised = {s.strip().lower() for s in body.candidate_skills}
    missing = [
        skill for skill in body.job_skills
        if skill.strip().lower() not in candidate_normalised
    ]
    return SkillGapResponse(missing_skills=missing)


@app.post(
    "/recommend-courses",
    summary="Recommend courses based on a candidate's skills",
    tags=["Courses"],
)
def recommend_courses(body: SkillGapRequest, top_n: int = 6):
    """
    Given a candidate's skills, return the top matching courses ranked by
    skill overlap and rating.

    Reuses `SkillGapRequest.candidate_skills` — no new model needed.

    ```json
    {
      "job_skills": [],
      "candidate_skills": ["python", "fastapi", "postgresql"]
    }
    ```
    """
    if not body.candidate_skills:
        raise HTTPException(status_code=422, detail="Provide at least one skill.")
    try:
        courses = _course_recommender.recommend(
            skills=body.candidate_skills, top_n=top_n
        )
    except Exception as exc:
        logging.exception("Course recommender failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Course recommender error: {exc}")

    return {
        "skills": body.candidate_skills,
        "total": len(courses),
        "courses": [
            {
                "id": c.id,
                "title": c.title,
                "description": c.description,
                "skills": c.skills,
                "rating": c.rating,
                "level": c.level,
                "duration_hours": c.duration_hours,
                "instructor": c.instructor,
                "match_score": c.match_score,
            }
            for c in courses
        ],
    }


@app.post(
    response_model=EnhanceResponse,
    summary="Enhance a resume to match a job description",
    tags=["CV Enhancer"],
)
async def enhance_resume(
    resume: UploadFile = File(..., description="Resume file (.txt or .pdf)"),
    job_description: str = Form(..., description="Job description as plain text"),
):
    """
    Upload a **resume** file (`.txt` or `.pdf`) and provide the **job description**
    as a text field. Returns an enhanced resume aligned to that role.

    Config and prompt settings are fixed server-side — no YAML upload needed.

    ```
    curl -X POST /enhance \\
      -F "resume=@my_cv.pdf" \\
      -F "job_description=We are looking for a Senior Python Engineer..."
    ```
    """
    # Read resume file
    data = await resume.read()
    content_type = resume.content_type or ""
    filename = resume.filename or ""

    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        resume_text = _extract_text_from_pdf(data)
    else:
        try:
            resume_text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            resume_text = data.decode("latin-1").strip()

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Resume file is empty.")
    if not job_description.strip():
        raise HTTPException(status_code=422, detail="Job description is empty.")

    model_name = _APP_CONFIG.get("model", "mistralai/mistral-small-2603")
    reasoning = _APP_CONFIG.get("reasoning_strategies", "Self-Ask")

    try:
        template_str = _build_prompt(_PROMPT_CONFIG, _APP_CONFIG)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prompt = PromptTemplate.from_template(template_str)
    llm = ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            "HTTP-Referer": "https:localhost:8000",
            "X-Title": "Joblin",
        },
    )
    chain = RunnableSequence(prompt | llm)

    logging.info("Calling LLM for /enhance (model=%s)...", model_name)

    try:
        result = chain.invoke({"resume": resume_text, "job_description": job_description})
    except Exception as exc:
        logging.exception("LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    return EnhanceResponse(
        enhanced_resume=_extract_llm_text(result) or "",
        model_used=model_name,
        reasoning_strategy=str(reasoning),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (optional — use `uvicorn main:app` in production)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)