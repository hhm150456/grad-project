"""
skill_gap.py
~~~~~~~~~~~~
Standalone skill gap analysis module.

Usage
-----
    from skill_gap import analyse_skill_gap, SkillGapResult

    result = analyse_skill_gap(
        job_skills=["Python", "React", "Docker", "Kubernetes"],
        candidate_skills=["python", "docker"],
    )

    print(result.match_rate)          # 50.0
    for gap in result.missing_skills:
        print(gap.priority, gap.skill, gap.priority_label)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MATCH_THRESHOLD: float = 0.80
"""Similarity score at or above which a candidate skill counts as a match."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MissingSkill:
    """A job-required skill absent (or weakly present) in the candidate profile."""

    skill: str
    priority: int          # 1 = highest priority (biggest gap)
    priority_label: str    # "Critical" | "Important" | "Nice to have"
    match_score: float     # 0.0 = no similarity; 1.0 = exact match


@dataclass
class SkillGapResult:
    """Full output of a skill gap analysis."""

    missing_skills: List[MissingSkill]
    matched_skills: List[str]
    total_job_skills: int
    total_candidate_skills: int
    match_rate: float      # percentage of job skills covered by the candidate

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def critical(self) -> List[MissingSkill]:
        """Missing skills labelled 'Critical'."""
        return [m for m in self.missing_skills if m.priority_label == "Critical"]

    @property
    def important(self) -> List[MissingSkill]:
        """Missing skills labelled 'Important'."""
        return [m for m in self.missing_skills if m.priority_label == "Important"]

    @property
    def nice_to_have(self) -> List[MissingSkill]:
        """Missing skills labelled 'Nice to have'."""
        return [m for m in self.missing_skills if m.priority_label == "Nice to have"]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Match rate  : {self.match_rate}%  "
            f"({len(self.matched_skills)}/{self.total_job_skills} skills matched)",
            f"Missing     : {len(self.missing_skills)}  "
            f"(Critical={len(self.critical)}, Important={len(self.important)}, "
            f"Nice to have={len(self.nice_to_have)})",
        ]
        if self.missing_skills:
            lines.append("\nPrioritised gaps:")
            for m in self.missing_skills:
                bar = "█" * int(m.match_score * 10) + "░" * (10 - int(m.match_score * 10))
                lines.append(
                    f"  [{m.priority:>2}] {m.skill:<28} "
                    f"{m.priority_label:<14} {bar} {m.match_score:.2f}"
                )
        if self.matched_skills:
            lines.append("\nMatched skills:")
            lines.append("  " + ", ".join(self.matched_skills))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _normalise(skill: str) -> str:
    return skill.strip().lower()


def _best_match_score(skill: str, candidate_skills: List[str]) -> float:
    """
    Return the highest similarity score between *skill* and any skill in
    *candidate_skills*.  Scores range from 0.0 (no similarity) to 1.0 (exact).
    """
    norm_skill = _normalise(skill)
    best = 0.0

    for cs in candidate_skills:
        norm_cs = _normalise(cs)

        if norm_skill == norm_cs:
            return 1.0

        # Substring containment handles e.g. "Python" vs "Python 3"
        if norm_skill in norm_cs or norm_cs in norm_skill:
            best = max(best, 0.9)
            continue

        ratio = SequenceMatcher(None, norm_skill, norm_cs).ratio()
        best = max(best, ratio)

    return best


def _priority_label(rank: int, total: int) -> str:
    if total == 0:
        return "Nice to have"
    third = total / 3
    if rank < third:
        return "Critical"
    if rank < 2 * third:
        return "Important"
    return "Nice to have"


def analyse_skill_gap(
    job_skills: List[str],
    candidate_skills: List[str],
    match_threshold: float = MATCH_THRESHOLD,
) -> SkillGapResult:
    """
    Compare *job_skills* against *candidate_skills* and return a
    :class:`SkillGapResult` with missing skills sorted by priority.

    Parameters
    ----------
    job_skills:
        Skills required by the role (normalised strings).
    candidate_skills:
        Skills the candidate possesses (normalised strings).
    match_threshold:
        Similarity score (0–1) above which a skill is considered matched.
        Defaults to :data:`MATCH_THRESHOLD` (0.80).

    Returns
    -------
    SkillGapResult
    """
    matched: List[str] = []
    missing_raw: List[tuple[str, float]] = []

    for skill in job_skills:
        score = _best_match_score(skill, candidate_skills)
        if score >= match_threshold:
            matched.append(skill)
        else:
            missing_raw.append((skill, score))

    # Lowest match score → highest priority (biggest gap first)
    missing_raw.sort(key=lambda x: x[1])
    total_missing = len(missing_raw)

    missing_skills = [
        MissingSkill(
            skill=skill,
            priority=idx + 1,
            priority_label=_priority_label(idx, total_missing),
            match_score=round(score, 3),
        )
        for idx, (skill, score) in enumerate(missing_raw)
    ]

    match_rate = round(len(matched) / len(job_skills) * 100, 1) if job_skills else 0.0

    return SkillGapResult(
        missing_skills=missing_skills,
        matched_skills=matched,
        total_job_skills=len(job_skills),
        total_candidate_skills=len(candidate_skills),
        match_rate=match_rate,
    )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _job = [
        "Python", "FastAPI", "PostgreSQL", "Docker",
        "Kubernetes", "React", "TypeScript",
    ]
    _candidate = ["python", "fastapi", "postgresql", "docker", "javascript"]

    result = analyse_skill_gap(_job, _candidate)
    print(result.summary())
