from sentence_transformers import SentenceTransformer
import numpy as np



jobbert = SentenceTransformer("TechWolf/JobBERT-v2")

def generate_seeker_embedding(seeker):
    """
    Generate an embedding for a job seeker using their title and skills,
    mirroring the structure of the job embedding (title + skills only,
    description slot is zero-padded to keep vector dimensions identical).
    """
    title_emb  = jobbert.encode(seeker.title)
    skills_emb = jobbert.encode(" ".join(seeker.skills))

    # Zero-pad the description slot so the final vector has the same
    # dimensionality as a job embedding (title + desc + skills).
    desc_dim   = jobbert.get_embedding_dimension()
    desc_pad   = np.zeros(desc_dim)

    final_embedding = np.concatenate([title_emb, desc_pad, skills_emb])
    return final_embedding.tolist()


def generate_query_embedding(
    query: str,
    seeker_embedding: list | None = None,
) -> list:
    """
    Generate an embedding for a free-text search query, with an optional
    joint blend with a job seeker's embedding.

    Standalone (no seeker_embedding):
        The query text is encoded into the skills slot.
        Title and description slots are zero-padded so the final vector
        has the same dimensionality as a job embedding.

    Joint (seeker_embedding provided):
        The query-only vector and the seeker embedding are averaged
        element-wise, combining the user's search intent with their
        profile signal. The result has the same dimensionality.
    """
    dim = jobbert.get_embedding_dimension()

    title_pad = np.zeros(dim)
    desc_pad  = np.zeros(dim)
    query_emb = jobbert.encode(query)

    query_vector = np.concatenate([title_pad, desc_pad, query_emb])

    if seeker_embedding is None:
        return query_vector.tolist()

    seeker_vector = np.array(seeker_embedding, dtype=np.float32)
    if seeker_vector.shape != query_vector.shape:
        raise ValueError(
            f"seeker_embedding has {seeker_vector.shape[0]} dimensions "
            f"but expected {query_vector.shape[0]}."
        )
    query_vector = (query_vector + seeker_vector) / 2.0

    return query_vector.tolist()


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