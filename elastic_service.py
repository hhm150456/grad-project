from elasticsearch import Elasticsearch
from models import JobDocument, SeekerDocument

INDEX_NAME = "jobs"
SEEKER_INDEX_NAME = "job_seekers"

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "iUC5UcRMUKI4JgfEn*rE"),
    verify_certs=False,
    ssl_show_warn=False,
)

def index_job(job: JobDocument):
    """
    Add a job document to Elasticsearch
    """

    document = {
        "job_id": job.job_id,
        "job_title": job.job_title,
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


def create_index(dims: int):
    """
    Create Elasticsearch index with vector mapping
    """

    mapping = {
        "mappings": {
            "properties": {
                "job_id": {"type": "keyword"},
                "job_title": {"type": "text"},
                "job_description": {"type": "text"},
                "skills": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine"
                }
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
            "_source": ["job_id", "job_title", "job_description", "skills"],
        },
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "job_id":          hit["_source"]["job_id"],
            "job_title":       hit["_source"]["job_title"],
            "job_description": hit["_source"]["job_description"],
            "skills":          hit["_source"]["skills"],
            "score":           hit["_score"],
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