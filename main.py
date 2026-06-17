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

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# huggingface_hub looks for HF_TOKEN (newer) or HUGGINGFACE_HUB_TOKEN (older).
# Make sure both are set from whichever one is present in .env, so
# sentence-transformers/transformers pick up the token regardless of version.
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = _hf_token
    logging.info("HF_TOKEN loaded from .env (length=%d).", len(_hf_token))
else:
    logging.warning("HF_TOKEN not found in .env — Hugging Face requests will be unauthenticated.")

# Config and prompt config are fixed server-side files — never sent by the client
_BASE_DIR = Path(__file__).parent
_APP_CONFIG = yaml.safe_load((_BASE_DIR / "config.yaml").read_text(encoding="utf-8")) or {}
_PROMPT_CONFIG = yaml.safe_load((_BASE_DIR / "prompt_config.yaml").read_text(encoding="utf-8")) or {}

from parser import parse_cv
from models import JobRequest, JobDocument, ParseTextRequest, ParsedCV, SkillGapRequest, SkillGapResponse, EnhanceRequest, EnhanceResponse, JobSeekerRequest, SeekerEmbeddingResponse, SeekerDocument, NormalizeRequest, NormalizeResponse, JobUpdateRequest, SearchQueryRequest, KNNSearchRequest, KNNSearchResponse, SeekerUpdateRequest, BM25SearchRequest, BM25SearchResponse, HybridSearchRequest, HybridSearchResponse, RerankRequest, RerankResponse
from embedding_service import generate_job_embedding, generate_seeker_embedding, generate_query_embedding
from elastic_service import index_job, index_seeker, delete_job, update_job, knn_search_jobs, bm25_search_jobs, hybrid_search_jobs, delete_seeker, update_seeker
from reranker_service import rerank_jobs
from esco_service import normalize_job
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
    response_model=JobDocument,
    summary="Generate an embedding for a job posting",
    tags=["Jobs"],
)
def add_job(job: JobRequest):
    """
    Generate a vector embedding for a job posting and return the enriched job document.
    Use **/index-job** to persist it into Elasticsearch afterwards.

    ```json
    {
      "job_id": "123hhdu",
      "job_title": "Data Engineer",
      "job_description": "...",
      "skills": ["Python", "Spark", "AWS"]
    }
    ```
    """
    embedding = generate_job_embedding(job)
    return JobDocument(
        job_id=job.job_id,
        job_title=job.job_title,
        job_description=job.job_description,
        skills=job.skills,
        embedding=embedding,
    )


@app.post(
    "/index-job",
    summary="Index a job document into Elasticsearch",
    tags=["Jobs"],
)
def index_job_endpoint(job_doc: JobDocument):
    """
    Persist a **JobDocument** (including its embedding) into Elasticsearch.
    Call **/add-job** first to obtain the document with the embedding, then
    pass the result here to index it.

    ```json
    {
      "job_id": "123hhdu",
      "job_title": "Data Engineer",
      "job_description": "...",
      "skills": ["Python", "Spark", "AWS"],
      "embedding": [0.12, 0.34, ...]
    }
    ```
    """
    index_job(job_doc)
    return {"message": "Job indexed successfully", "job_id": job_doc.job_id}

@app.delete(
    "/delete-job/{job_id}",
    summary="Delete a job from Elasticsearch by job ID",
    tags=["Jobs"],
)
def delete_job_endpoint(job_id: str):
    """
    Permanently remove a job document from Elasticsearch using its **job_id**.

    ```
    DELETE /delete-job/123hhdu
    ```
    """
    try:
        return delete_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.patch(
    "/update-job/{job_id}",
    summary="Partially update a job in Elasticsearch by job ID",
    tags=["Jobs"],
)
def update_job_endpoint(job_id: str, updates: JobUpdateRequest):
    """
    Partially update a job document in Elasticsearch.
    Only the fields you provide are changed — omitted fields are left intact.

    ```json
    {
      "job_title": "Senior Data Engineer",
      "skills": ["Python", "Spark", "Kafka"]
    }
    ```
    """
    payload = updates.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail="No update fields provided.")
    try:
        return update_job(job_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post(
    "/seeker-embedding",
    response_model=SeekerEmbeddingResponse,
    summary="Generate an embedding for a job seeker",
    tags=["Jobs"],
)
def seeker_embedding(seeker: JobSeekerRequest):
    """
    Generate a vector embedding for a job seeker from their **title** and **skills**.

    The embedding has the same dimensionality as a job embedding so it can be
    used directly for cosine-similarity search against the jobs index.

    ```json
    {
      "title": "Data Engineer",
      "skills": ["Python", "Spark", "AWS"]
    }
    ```
    """
    embedding = generate_seeker_embedding(seeker)
    return SeekerEmbeddingResponse(title=seeker.title, skills=seeker.skills, embedding=embedding)


@app.post(
    "/index-seeker",
    summary="Index a job seeker document into Elasticsearch",
    tags=["Jobs"],
)
def index_seeker_endpoint(seeker_doc: SeekerDocument):
    """
    Persist a **SeekerDocument** (including its embedding) into Elasticsearch.
    Call **/seeker-embedding** first to obtain the document with the embedding,
    then pass the result here — adding a **seeker_id** — to index it.

    ```json
    {
      "seeker_id": "abc123",
      "title": "Data Engineer",
      "skills": ["Python", "Spark", "AWS"],
      "embedding": [0.12, 0.34, ...]
    }
    ```
    """
    index_seeker(seeker_doc)
    return {"message": "Job seeker indexed successfully", "seeker_id": seeker_doc.seeker_id}





@app.delete(
    "/delete-seeker/{seeker_id}",
    summary="Delete a job seeker from Elasticsearch by seeker ID",
    tags=["Jobs"],
)
def delete_seeker_endpoint(seeker_id: str):
    """
    Permanently remove a job seeker document from Elasticsearch using their **seeker_id**.

    ```
    DELETE /delete-seeker/abc123
    ```
    """
    try:
        return delete_seeker(seeker_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.patch(
    "/update-seeker/{seeker_id}",
    summary="Partially update a job seeker in Elasticsearch by seeker ID",
    tags=["Jobs"],
)
def update_seeker_endpoint(seeker_id: str, updates: SeekerUpdateRequest):
    """
    Partially update a job seeker document in Elasticsearch.
    Only the fields you provide are changed — omitted fields are left intact.

    ```json
    {
      "title": "Senior Data Engineer",
      "skills": ["Python", "Spark", "Kafka"]
    }
    ```
    """
    payload = updates.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail="No update fields provided.")
    try:
        return update_seeker(seeker_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post(
    "/search-query",
    summary="Embed a free-text search query, optionally blended with a seeker embedding",
    tags=["Search"],
)
def search_query(body: SearchQueryRequest):
    """
    Accepts a plain-text search query and returns its vector embedding.

    Optionally provide a **seeker_embedding** (from **/seeker-embedding**) to
    produce a joint embedding that blends the query intent with the seeker's
    profile via element-wise averaging — useful for personalised search.

    ```json
    {
      "query": "machine learning engineer with Python experience",
      "seeker_embedding": [0.12, 0.34, ...]
    }
    ```
    """
    query: str = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="'query' must not be empty.")

    seeker_embedding = body.seeker_embedding

    try:
        embedding = generate_query_embedding(query, seeker_embedding=seeker_embedding)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"query": query, "embedding": embedding}




@app.post(
    "/similarity",
    response_model=KNNSearchResponse,
    summary="ANN vector search — find the most similar jobs to a query embedding",
    tags=["Search"],
)
def similarity(body: KNNSearchRequest):
    """
    Run an **Approximate Nearest Neighbour (kNN)** search against the jobs index
    using the provided **query_embedding** (from **/search-query**).

    Returns the top-**k** most similar jobs ranked by cosine similarity score.

    - **k** — number of results to return (default 10)
    - **num_candidates** — ANN candidate pool size; higher = more accurate but slower (default 100)

    ```json
    {
      "query_embedding": [0.12, 0.34, ...],
      "k": 10,
      "num_candidates": 100
    }
    ```
    """
    try:
        results = knn_search_jobs(
            query_embedding=body.query_embedding,
            k=body.k,
            num_candidates=body.num_candidates,
        )
    except Exception as exc:
        logging.exception("kNN search failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"kNN search failed: {exc}")

    return KNNSearchResponse(total=len(results), results=results)


@app.post(
    "/bm25-search",
    response_model=BM25SearchResponse,
    summary="BM25 full-text search over jobs",
    tags=["Search"],
)
def bm25_search(body: BM25SearchRequest):
    """
    Run a **BM25** keyword search against the jobs index, matching the
    query text against `job_title`, `job_description` and `skills`.

    ```json
    {
      "query": "python data engineer remote",
      "k": 10
    }
    ```
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="'query' must not be empty.")

    try:
        results = bm25_search_jobs(query=query, k=body.k)
    except Exception as exc:
        logging.exception("BM25 search failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"BM25 search failed: {exc}")

    return BM25SearchResponse(total=len(results), results=results)


@app.post(
    "/hybrid-search",
    response_model=HybridSearchResponse,
    summary="Hybrid search — BM25 + vector kNN merged via Reciprocal Rank Fusion",
    tags=["Search"],
)
def hybrid_search(body: HybridSearchRequest):
    """
    Runs a **BM25** keyword search and a **vector kNN** search in parallel and
    merges the two ranked lists using **Reciprocal Rank Fusion (RRF)**.

    The query embedding is generated server-side from the `query` text
    (and optionally blended with `seeker_embedding` for personalisation).

    - **k** — number of merged results to return (default 10)
    - **num_candidates** — ANN candidate pool size for the vector search (default 100)
    - **rrf_k** — RRF smoothing constant (default 60)

    ```json
    {
      "query": "python data engineer remote",
      "k": 10
    }
    ```

    Response:
    ```json
    {
      "query": "python data engineer remote",
      "results": [
        {
          "job_id": "123",
          "job_title": "Senior Data Engineer",
          "job_description": "...",
          "skills": ["Python", "Spark"],
          "hybrid_score": 0.92,
          "bm25_score": 15.4,
          "embedding_score": 0.88
        }
      ]
    }
    ```
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="'query' must not be empty.")

    try:
        query_embedding = generate_query_embedding(query, seeker_embedding=body.seeker_embedding)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        results = hybrid_search_jobs(
            query=query,
            query_embedding=query_embedding,
            k=body.k,
            num_candidates=body.num_candidates,
            rrf_k=body.rrf_k,
        )
    except Exception as exc:
        logging.exception("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Hybrid search failed: {exc}")

    return HybridSearchResponse(query=query, results=results)


@app.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank a set of jobs against a query using a cross-encoder",
    tags=["Search"],
)
def rerank(body: RerankRequest):
    """
    Rerank a small candidate set of jobs against a free-text query using the
    **cross-encoder/ms-marco-MiniLM-L-6-v2** model.

    Unlike BM25 or vector kNN, a cross-encoder scores the **query and each
    job jointly** (no precomputed embeddings), which tends to give more
    accurate relevance ordering for a final-pass rerank of, e.g., the top-N
    results from **/hybrid-search**.

    Intended for small candidate lists (tens of jobs, not thousands) since
    scoring is O(n) model calls.

    ```json
    {
      "query": "entry level ai engineer with python",
      "jobs": [
        {"job_id": "123", "title": "AI Engineer", "description": "Python, TensorFlow, ML..."},
        {"job_id": "456", "title": "Backend Developer", "description": "Java, Spring..."}
      ]
    }
    ```
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="'query' must not be empty.")
    if not body.jobs:
        raise HTTPException(status_code=422, detail="'jobs' must not be empty.")

    try:
        results = rerank_jobs(query=query, jobs=body.jobs)
    except Exception as exc:
        logging.exception("Rerank failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Rerank failed: {exc}")

    return RerankResponse(query=query, results=results)



@app.post(
    "/normalize",
    response_model=NormalizeResponse,
    summary="Normalize a job title and skills against the ESCO dataset",
    tags=["ESCO"],
)
def normalize(body: NormalizeRequest):
    """
    Fuzzy-match a raw job title and skills list against the ESCO dataset and
    return the closest standardised label with a confidence score.

    - **title_score** — fuzzy match score for the title alone (0–100)
    - **confidence**  — weighted score: 60 % title + 40 % average skill score
    - **matched_skills** — per-skill ESCO match details

    ```json
    {
      "title": "Data Engineer",
      "skills": ["Python", "Spark", "AWS"]
    }
    ```
    """
    result = normalize_job(body.title, body.skills)
    return NormalizeResponse(**result)

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
    "/enhance",
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
        api_key=os.environ["CV_API_KEY"],
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