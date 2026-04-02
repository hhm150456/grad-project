from pydantic import BaseModel
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