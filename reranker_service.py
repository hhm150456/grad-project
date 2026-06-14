"""
reranker_service.py
Lightweight cross-encoder reranking for job search results.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers CrossEncoder)
to directly score (query, document) pairs — typically more accurate than
embedding-similarity for ranking a small candidate set, at the cost of being
O(n) model calls instead of O(1).
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = CrossEncoder(_MODEL_NAME)


def rerank_jobs(query: str, jobs: list) -> list:
    """
    Score and rank a list of jobs against a query using the cross-encoder.

    `jobs` is a list of objects/dicts with `job_id`, `title`, and
    `description` attributes/keys.

    Returns a list of dicts with job_id, title and rerank_score, sorted by
    rerank_score descending (highest relevance first).
    """
    if not jobs:
        return []

    def _field(job, name):
        return getattr(job, name) if hasattr(job, name) else job[name]

    pairs = [
        (query, f"{_field(job, 'title')}. {_field(job, 'description')}")
        for job in jobs
    ]

    scores = _model.predict(pairs)

    results = []
    for job, score in zip(jobs, scores):
        results.append({
            "job_id": _field(job, "job_id"),
            "title": _field(job, "title"),
            "rerank_score": float(score),
        })

    results.sort(key=lambda r: r["rerank_score"], reverse=True)
    return results
