from pydantic import BaseModel, field_validator
from typing import List


class JobRequest(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    skills: List[str]


class JobDocument(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    skills: List[str]
    embedding: List[float]

class ParseTextRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Jane Doe\njane@example.com\n+1 555 123 4567\n\nExperience\n..."
            }
        }
    }


class ParsedCV(BaseModel):
    personal_info: dict
    work_experience: list[dict]
    education: list[dict]
    skills: dict
    certifications: list[dict]
    languages: list[dict]


class SkillGapRequest(BaseModel):
    job_skills: list[str]
    candidate_skills: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_skills": ["Python", "FastAPI", "Docker", "Kubernetes"],
                "candidate_skills": ["Python", "FastAPI", "Docker"],
            }
        }
    }


class SkillGapResponse(BaseModel):
    missing_skills: list[str]


class EnhanceRequest(BaseModel):
    job_description: str

    model_config = {
        "json_schema_extra": {
            "example": {"job_description": "We are looking for a Senior Python Engineer..."}
        }
    }


class EnhanceResponse(BaseModel):
    enhanced_resume: str
    model_used: str
    reasoning_strategy: str

class JobSeekerRequest(BaseModel):
    title: str
    skills: list[str]

class SeekerEmbeddingResponse(BaseModel):
    title: str
    skills: list[str]
    embedding: list[float]

class SeekerDocument(BaseModel):
    seeker_id: str
    title: str
    skills: list[str]
    embedding: list[float]

class SkillMatch(BaseModel):
    input_skill: str
    matched_skill: str   # matched ESCO skill name from mapped_skills
    score: float

class NormalizeResponse(BaseModel):
    normalized_title: str
    occupation_uri: str | None  # ESCO occupation URI
    confidence: float
    title_score: float
    matched_skills: list[SkillMatch]

class NormalizeRequest(BaseModel):
    title: str
    skills: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Data Engineer",
                "skills": ["Python", "Spark", "AWS"],
            }
        }
    }


class JobUpdateRequest(BaseModel):
    job_title: str | None = None
    job_description: str | None = None
    skills: list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_title": "Senior Data Engineer",
                "skills": ["Python", "Spark", "Kafka"],
            }
        }
    }


class SearchQueryRequest(BaseModel):
    query: str
    seeker_embedding: list[float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "machine learning engineer with Python experience",
            }
        }
    }

    @field_validator("seeker_embedding")
    @classmethod
    def check_embedding_dim(cls, v: list[float]) -> list[float]:
        if len(v) != 3072:
            raise ValueError(
                f"seeker_embedding must have 3072 dimensions (got {len(v)}). "
                "Use the embedding returned by POST /seeker-embedding."
            )
        return v


class KNNSearchRequest(BaseModel):
    query_embedding: list[float]
    k: int = 10
    num_candidates: int = 100

    model_config = {
        "json_schema_extra": {
            "example": {
                "k": 10,
                "num_candidates": 100,
            }
        }
    }

    @field_validator("query_embedding")
    @classmethod
    def check_embedding_dim(cls, v: list[float]) -> list[float]:
        if len(v) != 3072:
            raise ValueError(
                f"query_embedding must have 3072 dimensions (got {len(v)}). "
                "Use the embedding returned by POST /search-query."
            )
        return v


class KNNSearchResult(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    skills: list[str]
    score: float


class KNNSearchResponse(BaseModel):
    total: int
    results: list[KNNSearchResult]


class BM25SearchRequest(BaseModel):
    query: str
    k: int = 10

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "python data engineer remote",
                "k": 10,
            }
        }
    }


class BM25SearchResult(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    skills: list[str]
    score: float


class BM25SearchResponse(BaseModel):
    total: int
    results: list[BM25SearchResult]


class HybridSearchRequest(BaseModel):
    query: str
    seeker_embedding: list[float] | None = None
    k: int = 10
    num_candidates: int = 100
    rrf_k: int = 60

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "python data engineer remote",
                "k": 10,
                "num_candidates": 100,
            }
        }
    }


class HybridSearchResult(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    skills: list[str]
    hybrid_score: float
    bm25_score: float | None = None
    embedding_score: float | None = None


class HybridSearchResponse(BaseModel):
    query: str
    results: list[HybridSearchResult]


class SeekerUpdateRequest(BaseModel):
    title: str | None = None
    skills: list[str] | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Senior Data Engineer",
                "skills": ["Python", "Spark", "Kafka"]
            }
        }
    }