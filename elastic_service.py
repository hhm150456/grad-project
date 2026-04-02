from elasticsearch import Elasticsearch
from models import JobDocument

INDEX_NAME = "jobs"

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
                "job_id": {"type": "text"},
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