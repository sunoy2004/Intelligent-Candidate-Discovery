"""Candidate data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int = 0


@dataclass
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None = None
    tier: str = "unknown"


@dataclass
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class Candidate:
    candidate_id: str
    profile: dict[str, Any]
    career_history: list[CareerEntry]
    education: list[EducationEntry]
    skills: list[Skill]
    certifications: list[dict[str, Any]]
    languages: list[dict[str, Any]]
    redrob_signals: RedrobSignals
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
