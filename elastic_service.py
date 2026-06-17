import os

from elasticsearch import Elasticsearch
from models import JobDocument, SeekerDocument

INDEX_NAME = "jobs_strat2_normal_actual2"
SEEKER_INDEX_NAME = "job_seekers"
EMBEDDING_DIM = 3072

ES_URL = os.environ.get("ES_URL", "https://localhost:9200")
ES_USER = os.environ.get("ES_USER", "elastic")
ES_PASSWORD = os.environ.get("ES_PASSWORD")

if not ES_PASSWORD:
    raise RuntimeError(
        "ES_PASSWORD environment variable is not set. "
        "Add ES_PASSWORD=<your elastic user password> to your .env file."
    )

es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

def index_job(job: JobDocument):
    """
    Add a job document to Elasticsearch
    """

    document = {
        "job_id": job.job_id,
        "normal_job_title": job.job_title,
        "job_description": job.job_description,
        "skills": job.skills,
        "embedding": job.embedding
    }

    response = es.index(
        index=INDEX_NAME,
        id=job.job_id,
        document=document
    )

    return response


def create_index(dims: int = EMBEDDING_DIM):
    """
    Create Elasticsearch index with vector mapping
    """

    mapping = {
        "mappings": {
            "properties": {
                "job_id":          {"type": "keyword"},
                "normal_job_title":       {"type": "text"},
                "job_description": {"type": "text"},
                "skills":          {"type": "keyword"},
                "embedding": {
                    "type":       "dense_vector",
                    "dims":       dims,
                    "index":      True,
                    "similarity": "cosine",
                },
            }
        }
    }

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=mapping)

def index_seeker(seeker: SeekerDocument):
    """
    Add a job seeker document to Elasticsearch.
    """
    document = {
        "seeker_id": seeker.seeker_id,
        "title": seeker.title,
        "skills": seeker.skills,
        "embedding": seeker.embedding,
    }
    response = es.index(index=SEEKER_INDEX_NAME, id=seeker.seeker_id, document=document)
    return response


def create_seeker_index(dims: int):
    """
    Create the job seekers Elasticsearch index with vector mapping.
    """
    mapping = {
        "mappings": {
            "properties": {
                "seeker_id": {"type": "keyword"},
                "title":     {"type": "text"},
                "skills":    {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }
    if not es.indices.exists(index=SEEKER_INDEX_NAME):
        es.indices.create(index=SEEKER_INDEX_NAME, body=mapping)

def delete_job(job_id: str) -> dict:
    """
    Delete a job document from Elasticsearch by job_id.
    Raises a 404-style error if the document does not exist.
    """
    if not es.exists(index=INDEX_NAME, id=job_id):
        raise ValueError(f"Job '{job_id}' not found in index '{INDEX_NAME}'.")

    response = es.delete(index=INDEX_NAME, id=job_id)
    return {"deleted": True, "job_id": job_id, "result": response["result"]}


def update_job(job_id: str, updates: dict) -> dict:
    """
    Partially update a job document in Elasticsearch by job_id.
    Only the fields present in `updates` are changed; others are left intact.
    Raises a 404-style error if the document does not exist.
    """
    if not es.exists(index=INDEX_NAME, id=job_id):
        raise ValueError(f"Job '{job_id}' not found in index '{INDEX_NAME}'.")

    # The ES mapping stores the title under "normal_job_title"; translate
    # the public "job_title" field name so callers don't need to know that.
    if "job_title" in updates:
        updates = {**updates, "normal_job_title": updates.pop("job_title")}

    response = es.update(
        index=INDEX_NAME,
        id=job_id,
        body={"doc": updates},
    )
    return {"updated": True, "job_id": job_id, "result": response["result"]}

def knn_search_jobs(query_embedding: list, k: int = 10, num_candidates: int = 100) -> list:
    """
    Run an ANN (kNN) vector search against the jobs index using the provided
    query embedding. Returns the top-k closest jobs ranked by cosine similarity.

    k              — number of results to return
    num_candidates — size of the ANN candidate pool (higher = more accurate, slower)
    """
    response = es.search(
        index=INDEX_NAME,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": num_candidates,
            },
            "_source": ["job_id", "normal_job_title", "job_description", "skills"],
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "job_id":          hit["_source"]["job_id"],
            "job_title":       hit["_source"]["normal_job_title"],
            "job_description": hit["_source"]["job_description"],
            "skills":          hit["_source"]["skills"],
            "score":           hit["_score"],
        })
    return results

def bm25_search_jobs(query: str, k: int = 10) -> list:
    """
    Run a BM25 full-text search against the jobs index using the provided
    query string. Searches normal_job_title, job_description and skills fields.
    """
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["normal_job_title^2", "job_description", "skills"],
                }
            },
            "size": k,
            "_source": ["job_id", "normal_job_title", "job_description", "skills"],
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "job_id":          hit["_source"]["job_id"],
            "job_title":       hit["_source"]["normal_job_title"],
            "job_description": hit["_source"]["job_description"],
            "skills":          hit["_source"]["skills"],
            "score":           hit["_score"],
        })
    return results


def hybrid_search_jobs(
    query: str,
    query_embedding: list,
    k: int = 10,
    num_candidates: int = 100,
    rrf_k: int = 60,
) -> list:
    """
    Combine BM25 full-text search and kNN vector search results using
    Reciprocal Rank Fusion (RRF).

    For each result, RRF score contribution from a ranked list is
    1 / (rrf_k + rank), where rank is 1-based. Contributions from both
    lists are summed to produce the final hybrid_score.

    Returns a list of dicts containing job_id, job_title, job_description,
    skills, hybrid_score, bm25_score and embedding_score (the latter two
    are None if the job did not appear in that particular result list).
    """
    pool_size = max(k, num_candidates)
    bm25_results = bm25_search_jobs(query, k=pool_size)
    knn_results = knn_search_jobs(query_embedding, k=pool_size, num_candidates=num_candidates)

    combined: dict[str, dict] = {}

    def _get_entry(hit):
        job_id = hit["job_id"]
        return combined.setdefault(job_id, {
            "job_id": job_id,
            "job_title": hit["job_title"],
            "job_description": hit["job_description"],
            "skills": hit["skills"],
            "bm25_score": None,
            "embedding_score": None,
            "rrf_score": 0.0,
        })

    for rank, hit in enumerate(bm25_results, start=1):
        entry = _get_entry(hit)
        entry["bm25_score"] = hit["score"]
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, hit in enumerate(knn_results, start=1):
        entry = _get_entry(hit)
        entry["embedding_score"] = hit["score"]
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(combined.values(), key=lambda e: e["rrf_score"], reverse=True)

    results = []
    for entry in ranked[:k]:
        results.append({
            "job_id":          entry["job_id"],
            "job_title":       entry["job_title"],
            "job_description": entry["job_description"],
            "skills":          entry["skills"],
            "hybrid_score":    entry["rrf_score"],
            "bm25_score":      entry["bm25_score"],
            "embedding_score": entry["embedding_score"],
        })
    return results


def delete_seeker(seeker_id: str) -> dict:
    """
    Delete a job seeker document from Elasticsearch by seeker_id.
    Raises a ValueError if the document does not exist.
    """
    if not es.exists(index=SEEKER_INDEX_NAME, id=seeker_id):
        raise ValueError(f"Seeker '{seeker_id}' not found in index '{SEEKER_INDEX_NAME}'.")

    response = es.delete(index=SEEKER_INDEX_NAME, id=seeker_id)
    return {"deleted": True, "seeker_id": seeker_id, "result": response["result"]}


def update_seeker(seeker_id: str, updates: dict) -> dict:
    """
    Partially update a job seeker document in Elasticsearch by seeker_id.
    Only the fields present in `updates` are changed; others are left intact.
    Raises a ValueError if the document does not exist.
    """
    if not es.exists(index=SEEKER_INDEX_NAME, id=seeker_id):
        raise ValueError(f"Seeker '{seeker_id}' not found in index '{SEEKER_INDEX_NAME}'.")

    response = es.update(
        index=SEEKER_INDEX_NAME,
        id=seeker_id,
        body={"doc": updates},
    )
    return {"updated": True, "seeker_id": seeker_id, "result": response["result"]}