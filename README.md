# Joblin

**Joblin** is a FastAPI-based intelligent job matching and career development platform. It connects job seekers with relevant opportunities using hybrid search (BM25 + vector kNN), provides CV parsing and enhancement, performs skill gap analysis, and recommends courses to help candidates upskill.

---

## Features

- **CV Parsing** — Extract structured data from raw text or uploaded `.txt`/`.pdf` resumes
- **Job Indexing** — Generate vector embeddings for job postings and index them into Elasticsearch
- **Hybrid Job Search** — Combine BM25 full-text search and kNN vector search via Reciprocal Rank Fusion (RRF)
- **Reranking** — Cross-encoder reranking for final-pass result precision
- **Skill Gap Analysis** — Identify skills a candidate is missing for a target role
- **Course Recommendations** — Suggest relevant courses based on a candidate's existing skills
- **CV Enhancement** — Use an LLM (via OpenRouter) to rewrite a resume tailored to a specific job description
- **ESCO Normalization** — Fuzzy-match job titles and skills against the ESCO occupational taxonomy

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Search | Elasticsearch (kNN) |
| Keyword Search | Elasticsearch (BM25) |
| Embeddings | Sentence Transformers (HuggingFace) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (CV Enhancer) | OpenRouter (Mistral via LangChain) |
| ESCO Matching | Fuzzy string matching |
| PDF Parsing | pdfplumber |

---

## Project Structure

```
grad-project/
├── main.py                # FastAPI app and all API endpoints
├── models.py              # Pydantic request/response models
├── parser.py              # CV text parser
├── embedding_service.py   # Job, seeker, and query embedding generation
├── elastic_service.py     # Elasticsearch indexing, search, and CRUD
├── reranker_service.py    # Cross-encoder reranking
├── esco_service.py        # ESCO taxonomy normalization
├── skill_gap.py           # Skill gap helper logic
├── Course_recommender.py  # Course recommendation engine
├── build_qrels_pool.py    # Evaluation pool builder
├── evaluate_search.py     # Search evaluation metrics
├── config.yaml            # App configuration (model names, strategies)
├── prompt_config.yaml     # LLM prompt configuration for CV enhancement
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── embed_jobs.ipynb       # Notebook: batch-embed job listings
├── normalize_jobs.ipynb   # Notebook: normalize job data against ESCO
├── files/                 # Supporting data files
└── rankings/              # Search evaluation output files
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A running Elasticsearch instance
- API keys for HuggingFace and OpenRouter

### Installation

```bash
git clone https://github.com/hhm150456/grad-project.git
cd grad-project
pip install -r requirements.txt

# Optional: PDF resume support
pip install pdfplumber
```

### Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token
CV_API_KEY=your_openrouter_api_key
DATABASE_URL=your_database_url          # for course recommendations
ELASTICSEARCH_URL=http://localhost:9200 # or your ES endpoint
```

### Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Endpoints

### Parser
| Method | Endpoint | Description |
|---|---|---|
| POST | `/parse/text` | Parse a CV from raw text (JSON body) |
| POST | `/parse/file` | Parse a CV from an uploaded `.txt` or `.pdf` file |

### Jobs
| Method | Endpoint | Description |
|---|---|---|
| POST | `/add-job` | Generate an embedding for a job posting |
| POST | `/index-job` | Index a job document into Elasticsearch |
| DELETE | `/delete-job/{job_id}` | Remove a job from the index |
| PATCH | `/update-job/{job_id}` | Partially update a job document |
| POST | `/seeker-embedding` | Generate an embedding for a job seeker |
| POST | `/index-seeker` | Index a job seeker into Elasticsearch |
| DELETE | `/delete-seeker/{seeker_id}` | Remove a seeker from the index |
| PATCH | `/update-seeker/{seeker_id}` | Partially update a seeker document |

### Search
| Method | Endpoint | Description |
|---|---|---|
| POST | `/search-query` | Embed a free-text query (optionally blended with seeker profile) |
| POST | `/similarity` | kNN approximate nearest-neighbour vector search |
| POST | `/bm25-search` | BM25 full-text keyword search |
| POST | `/hybrid-search` | Hybrid BM25 + kNN search via Reciprocal Rank Fusion |
| POST | `/rerank` | Cross-encoder reranking of a candidate job set |

### Skills & Courses
| Method | Endpoint | Description |
|---|---|---|
| POST | `/skill-gap` | Identify skills missing from a candidate profile |
| POST | `/recommend-courses` | Recommend courses based on a candidate's skills |

### ESCO
| Method | Endpoint | Description |
|---|---|---|
| POST | `/normalize` | Normalize a job title and skills against the ESCO dataset |

### CV Enhancer
| Method | Endpoint | Description |
|---|---|---|
| POST | `/enhance` | Rewrite a resume to align with a target job description |

### Utility
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |

---

## Example Usage

**Parse a CV from text:**
```bash
curl -X POST http://localhost:8000/parse/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Jane Doe\njane@example.com\nPython, FastAPI, PostgreSQL..."}'
```

**Hybrid job search:**
```bash
curl -X POST http://localhost:8000/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{"query": "python data engineer remote", "k": 10}'
```

**Enhance a resume:**
```bash
curl -X POST http://localhost:8000/enhance \
  -F "resume=@my_cv.pdf" \
  -F "job_description=We are looking for a Senior Python Engineer..."
```

---

## Evaluation

The `build_qrels_pool.py` and `evaluate_search.py` scripts support offline evaluation of search quality using relevance judgments (qrels). Results are saved to the `rankings/` directory.

---

## License

This project was developed as a graduation project. Please contact the author for licensing details.
