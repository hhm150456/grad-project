from sentence_transformers import SentenceTransformer
import numpy as np

jobbert = SentenceTransformer("TechWolf/JobBERT-v2")

def generate_job_embedding(job):

    title_emb = jobbert.encode(job.job_title)
    desc_emb = jobbert.encode(job.job_description)
    skills_emb = jobbert.encode(" ".join(job.skills))

    final_embedding = np.concatenate([
        title_emb,
        desc_emb,
        skills_emb
    ])

    return final_embedding.tolist()