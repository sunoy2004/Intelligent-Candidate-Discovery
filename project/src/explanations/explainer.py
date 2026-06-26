"""Hallucination-safe explanation generation."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _matched_skills(skill_names: list[str], job_required: list[str]) -> list[str]:
    req_lower = {r.lower() for r in job_required}
    matched = []
    for s in skill_names:
        sl = s.lower()
        if any(r in sl or sl in r for r in req_lower):
            matched.append(s)
    return matched[:4]


def explain_candidate(
    row: pd.Series,
    rank: int,
    job_features: dict[str, Any],
    component_scores: dict[str, float] | None = None,
) -> str:
    skill_names = json.loads(row["skill_names"])
    matched = _matched_skills(skill_names, job_features.get("required_skills", []))
    title = row["current_title"]
    yoe = row["years_experience"]
    location = row["location"]
    response = row["recruiter_response_rate"]
    notice = row["notice_period_days"]

    concerns: list[str] = []
    if notice > 60:
        concerns.append(f"notice period {int(notice)} days")
    if row.get("consulting_only_flag", 0):
        concerns.append("consulting-only career")
    if row.get("is_stale_6mo", 0):
        concerns.append("limited recent activity")
    if row.get("ai_keyword_count", 0) >= 7 and component_scores and component_scores.get("title_fit", 1) < 0.4:
        concerns.append("title-skill mismatch")

    skills_part = ", ".join(matched) if matched else "adjacent technical skills"
    concern_part = f" Concern: {concerns[0]}." if concerns and rank > 15 else ""

    if rank <= 10:
        return (
            f"{title} with {yoe:.1f} yrs; strong fit on {skills_part}; "
            f"response rate {response:.2f}; {location}.{concern_part}"
        )
    if rank <= 50:
        return (
            f"{title}, {yoe:.1f} yrs experience; matches {skills_part}; "
            f"engagement response {response:.2f}.{concern_part}"
        )
    return (
        f"{title} with {yoe:.1f} yrs; partial match on {skills_part}; "
        f"response rate {response:.2f}.{concern_part}"
    )


def build_explanations(
    df: pd.DataFrame,
    ranked_ids: list[str],
    job_features: dict[str, Any],
) -> dict[str, str]:
    indexed = df.set_index("candidate_id")
    explanations: dict[str, str] = {}
    for rank, cid in enumerate(ranked_ids, start=1):
        row = indexed.loc[cid]
        explanations[cid] = explain_candidate(row, rank, job_features)
    return explanations
