"""Stream parser for candidates.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from src.parser.models import (
    Candidate,
    CareerEntry,
    EducationEntry,
    RedrobSignals,
    Skill,
)

logger = logging.getLogger(__name__)


def _parse_signals(data: dict) -> RedrobSignals:
    sig = data["redrob_signals"]
    salary = sig["expected_salary_range_inr_lpa"]
    return RedrobSignals(
        profile_completeness_score=float(sig["profile_completeness_score"]),
        signup_date=sig["signup_date"],
        last_active_date=sig["last_active_date"],
        open_to_work_flag=bool(sig["open_to_work_flag"]),
        profile_views_received_30d=int(sig["profile_views_received_30d"]),
        applications_submitted_30d=int(sig["applications_submitted_30d"]),
        recruiter_response_rate=float(sig["recruiter_response_rate"]),
        avg_response_time_hours=float(sig["avg_response_time_hours"]),
        skill_assessment_scores=dict(sig.get("skill_assessment_scores") or {}),
        connection_count=int(sig["connection_count"]),
        endorsements_received=int(sig["endorsements_received"]),
        notice_period_days=int(sig["notice_period_days"]),
        expected_salary_min=float(salary["min"]),
        expected_salary_max=float(salary["max"]),
        preferred_work_mode=sig["preferred_work_mode"],
        willing_to_relocate=bool(sig["willing_to_relocate"]),
        github_activity_score=float(sig["github_activity_score"]),
        search_appearance_30d=int(sig["search_appearance_30d"]),
        saved_by_recruiters_30d=int(sig["saved_by_recruiters_30d"]),
        interview_completion_rate=float(sig["interview_completion_rate"]),
        offer_acceptance_rate=float(sig["offer_acceptance_rate"]),
        verified_email=bool(sig["verified_email"]),
        verified_phone=bool(sig["verified_phone"]),
        linkedin_connected=bool(sig["linkedin_connected"]),
    )


def parse_candidate_dict(data: dict) -> Candidate:
    skills = [
        Skill(
            name=s["name"],
            proficiency=s["proficiency"],
            endorsements=int(s["endorsements"]),
            duration_months=int(s.get("duration_months", 0)),
        )
        for s in data.get("skills", [])
    ]
    career = [
        CareerEntry(
            company=c["company"],
            title=c["title"],
            start_date=c["start_date"],
            end_date=c.get("end_date"),
            duration_months=int(c["duration_months"]),
            is_current=bool(c["is_current"]),
            industry=c["industry"],
            company_size=c["company_size"],
            description=c.get("description", ""),
        )
        for c in data.get("career_history", [])
    ]
    education = [
        EducationEntry(
            institution=e["institution"],
            degree=e["degree"],
            field_of_study=e["field_of_study"],
            start_year=int(e["start_year"]),
            end_year=int(e["end_year"]),
            grade=e.get("grade"),
            tier=e.get("tier", "unknown"),
        )
        for e in data.get("education", [])
    ]
    return Candidate(
        candidate_id=data["candidate_id"],
        profile=data["profile"],
        career_history=career,
        education=education,
        skills=skills,
        certifications=list(data.get("certifications") or []),
        languages=list(data.get("languages") or []),
        redrob_signals=_parse_signals(data),
        raw=data,
    )


def stream_candidates(path: Path, limit: int | None = None) -> Iterator[Candidate]:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield parse_candidate_dict(json.loads(line))
            except (KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed record: %s", exc)
                continue
            count += 1
            if limit is not None and count >= limit:
                break


def load_candidates_json(path: Path) -> list[Candidate]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [parse_candidate_dict(item) for item in data]
