"""
esco_service.py
Normalizes job titles and skills against the ESCO dataset stored in Postgres.

Table schema (esco_skills_with_mapped):
  occupationuri   – ESCO occupation URI
  occupationlabel – preferred occupation title  (used as the match target)
  skillLabel      – raw ESCO skill labels (comma-separated string)
  relationType    – essential / optional
  skillType       – knowledge / skill/competence
  skillUri        – skill URIs (comma-separated)
  iscoGroup       – ISCO-08 group code
  mapped_skills   – clean, normalised skill names (comma-separated string)
"""

import os
import re
import logging

import pandas as pd
from rapidfuzz import process, fuzz
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────────────────────────────────────

def _get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return create_engine(db_url)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded cache — populated on first call, never at import time
# ─────────────────────────────────────────────────────────────────────────────

_esco_df: pd.DataFrame | None = None
_skill_pool: list[str] | None = None


def _load_esco_data() -> pd.DataFrame:
    engine = _get_engine()
    query = text("""
        SELECT DISTINCT ON ("occupationuri")
            "occupationuri",
            "occupationlabel",
            "mapped_skills"
        FROM esco_skills_with_mapped
        WHERE "occupationlabel" IS NOT NULL
          AND "mapped_skills"   IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df["occupationlabel_clean"] = df["occupationlabel"].str.strip().str.lower()
    logger.info("Loaded %d ESCO occupations from Postgres.", len(df))
    return df


def _build_skill_pool(df: pd.DataFrame) -> list[str]:
    skills: set[str] = set()
    for raw in df["mapped_skills"].dropna():
        for s in raw.split(","):
            s = s.strip().lower()
            if s:
                skills.add(s)
    logger.info("Built ESCO skill pool with %d unique skills.", len(skills))
    return sorted(skills)


def _ensure_loaded() -> tuple[pd.DataFrame, list[str]]:
    """Load data from Postgres on first use and cache it for subsequent calls."""
    global _esco_df, _skill_pool
    if _esco_df is None or _skill_pool is None:
        _esco_df    = _load_esco_data()
        _skill_pool = _build_skill_pool(_esco_df)
    return _esco_df, _skill_pool


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy matchers
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_match_title(title: str, threshold: int = 80) -> dict | None:
    esco_df, _ = _ensure_loaded()

    title_clean = clean(title)
    choices     = esco_df["occupationlabel_clean"].tolist()

    match, score, idx = process.extractOne(
        title_clean,
        choices,
        scorer=fuzz.token_sort_ratio,
    )

    if score >= threshold:
        row = esco_df.iloc[idx]
        return {
            "matched_label":   match,
            "preferred_label": row["occupationlabel"],
            "occupation_uri":  row["occupationuri"],
            "score":           score,
        }
    return None


def fuzzy_match_skills(job_skills: list[str], threshold: int = 80) -> list[dict]:
    _, skill_pool = _ensure_loaded()

    matches = []
    for skill in job_skills:
        match, score, _ = process.extractOne(
            clean(skill),
            skill_pool,
            scorer=fuzz.token_sort_ratio,
        )
        if score >= threshold:
            matches.append({
                "input_skill":   skill,
                "matched_skill": match,
                "score":         score,
            })

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(title_score: float, skill_matches: list[dict]) -> float:
    if not skill_matches:
        return round(float(title_score), 2)
    avg_skill_score = sum(m["score"] for m in skill_matches) / len(skill_matches)
    return round(0.6 * title_score + 0.4 * avg_skill_score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline
# ─────────────────────────────────────────────────────────────────────────────

def normalize_job(job_title: str, job_skills: list[str]) -> dict:
    title_match   = fuzzy_match_title(job_title)
    skill_matches = fuzzy_match_skills(job_skills)

    if not title_match:
        return {
            "normalized_title": "unknown",
            "occupation_uri":   None,
            "confidence":       0.0,
            "title_score":      0,
            "matched_skills":   skill_matches,
        }

    return {
        "normalized_title": title_match["preferred_label"],
        "occupation_uri":   title_match["occupation_uri"],
        "confidence":       compute_final_score(title_match["score"], skill_matches),
        "title_score":      title_match["score"],
        "matched_skills":   skill_matches,
    }