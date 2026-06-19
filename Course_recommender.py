"""
course_recommender.recommender
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recommends courses from a PostgreSQL database based on a list of skills.

Scoring formula (per course):
    score = matching_skills / total_requested_skills

Usage
-----
    from course_recommender import CourseRecommender

    rec = CourseRecommender(dsn="postgresql://user:pass@localhost/courses_db")
    results = rec.recommend(skills=["python", "fastapi"], top_n=5)
    for course in results:
        print(course)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import psycopg2
import psycopg2.extras



@dataclass
class Course:
    title: str
    url: str
    description: str
    skills: List[str]
    level: str
    match_score: float

    def __str__(self) -> str:
        bar_width = 20
        filled = round(self.match_score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = round(self.match_score * 100)
        return (
            f"[{pct:3d}%] {bar}  "
            f"{self.level:<14}  "
            f"{self.title}  ({self.url})"
        )




class CourseRecommender:
    """
    Recommends courses stored in a PostgreSQL ``OfferingPosts`` table.

    Parameters
    ----------
    dsn:
        libpq connection string, e.g.
        ``"postgresql://user:pass@localhost:5432/courses_db"``.
        Falls back to the ``DATABASE_URL`` environment variable if omitted.
    """

    _QUERY = """
        WITH PostSkills AS (
            -- Group all skills into a PostgreSQL array for each OfferingPost
            SELECT
                ops."OfferingPostsId",
                ARRAY_AGG(s."Name") AS skills_array
            FROM "OfferingPostSkills" ops
            JOIN "Skills" s ON ops."ProvidedSkillsId" = s."Id"
            GROUP BY ops."OfferingPostsId"
        )
        SELECT
            op."Title" AS title,
            op."EnrollmentUrl" AS url,
            op."Description" AS description,
            ps.skills_array AS skills,
            op."DifficultyLevel" AS level,
            ROUND(
                (
                    CARDINALITY(
                        ARRAY(
                            SELECT UNNEST(ps.skills_array)
                            INTERSECT
                            SELECT UNNEST(%(skills)s::text[])
                        )
                    )::float / %(n_skills)s
                )::numeric,
                4
            ) AS match_score
        FROM "OfferingPosts" op
        JOIN PostSkills ps ON op."Id" = ps."OfferingPostsId"
        -- Filter only posts that have at least one overlapping skill
        WHERE ps.skills_array && %(skills)s::text[]
        ORDER BY match_score DESC
        LIMIT %(top_n)s;
    """

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError(
                "Provide a DSN or set the DATABASE_URL environment variable."
            )
        self._conn: Optional[psycopg2.extensions.connection] = None

    

    def connect(self) -> "CourseRecommender":
        """Open the database connection. Called automatically on first use."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
        elif self._conn.status == psycopg2.extensions.STATUS_IN_TRANSACTION:
            self._conn.rollback()
        return self

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "CourseRecommender":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    
    def recommend(
        self,
        skills: List[str],
        top_n: int = 6,
    ) -> List[Course]:
        """
        Return the top *top_n* courses that best match *skills*.

        Parameters
        ----------
        skills:
            List of skill strings, e.g. ``["python", "fastapi"]``.
            Matching is case-insensitive.
        top_n:
            Maximum number of courses to return (default 6).

        Returns
        -------
        List[Course]
            Courses sorted by ``match_score`` descending, then by
            ``rating`` descending.  Empty list if no courses match.

        Raises
        ------
        ValueError
            If *skills* is empty.
        psycopg2.Error
            On any database error.
        """
        if not skills:
            raise ValueError("Provide at least one skill.")

        normalised = [s.strip().lower() for s in skills]
        self.connect()

        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    self._QUERY,
                    {
                        "skills": normalised,
                        "n_skills": len(normalised),
                        "top_n": top_n,
                    },
                )
                rows = cur.fetchall()
        except Exception:
            self._conn.rollback()
            raise

        return [
            Course(
                title=row["title"],
                url=row["url"],
                description=row["description"],
                skills=list(row["skills"]),
                level=row["level"],
                match_score=float(row["match_score"]),
            )
            for row in rows
        ]