"""Build tabular candidate features."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_pipeline_config, project_path, resolve_path, save_json
from src.parser.candidate_parser import stream_candidates
from src.parser.models import Candidate

logger = logging.getLogger(__name__)

CONSULTING = {"TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini", "HCL", "Tech Mahindra"}
PRODUCT_INDUSTRIES = {"Software", "Fintech", "E-commerce", "EdTech"}
SMALL_SIZES = {"1-10", "11-50", "51-200"}
AI_KEYWORDS = {
    "rag", "llm", "llms", "nlp", "pytorch", "tensorflow", "machine learning", "deep learning",
    "mlops", "langchain", "milvus", "pinecone", "faiss", "embedding", "vector", "fine-tuning",
    "transformers", "xgboost", "feature engineering", "model deployment", "computer vision",
}
TEMPLATE_SNIPPET = "Lately I've been curious about how AI tools could augment my work"
RETRIEVAL_RE = re.compile(
    r"\b(retrieval|ranking|recommendation|embedding|vector search|hybrid search|search system|ndcg|mrr|map)\b",
    re.I,
)
LEADERSHIP_RE = re.compile(r"\b(lead|led|managed|mentor|team of|ownership|owned)\b", re.I)
SHIP_RE = re.compile(r"\b(shipped|deployed|production|end-to-end|scaled)\b", re.I)
TIER_MAP = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1, "unknown": 0}
SENIORITY = [
    ("intern", 1),
    ("junior", 2),
    ("engineer", 3),
    ("senior", 4),
    ("lead", 5),
    ("staff", 6),
    ("principal", 7),
]


def _seniority_level(title: str) -> int:
    t = title.lower()
    level = 2
    for key, val in SENIORITY:
        if key in t:
            level = max(level, val)
    return level


def _candidate_row(candidate: Candidate, reference_date: str) -> dict[str, Any]:
    p = candidate.profile
    sig = candidate.redrob_signals
    ref = datetime.strptime(reference_date, "%Y-%m-%d")
    last_active = datetime.strptime(sig.last_active_date, "%Y-%m-%d")
    days_since_active = (ref - last_active).days

    career_text = " ".join(c.description for c in candidate.career_history)
    title_text = f"{p.get('current_title', '')} {p.get('headline', '')}".strip()
    summary_text = p.get("summary", "")

    total_months = sum(c.duration_months for c in candidate.career_history)
    product_months = sum(
        c.duration_months for c in candidate.career_history if c.industry in PRODUCT_INDUSTRIES
    )
    companies = {c.company for c in candidate.career_history}
    consulting_only = bool(companies) and all(
        any(cn in co for cn in CONSULTING) for co in companies
    )

    skill_names = [s.name for s in candidate.skills]
    skill_names_lower = [s.lower() for s in skill_names]
    ai_keyword_count = sum(
        1 for s in candidate.skills
        if s.name.lower() in AI_KEYWORDS
        or any(k in s.name.lower() for k in AI_KEYWORDS)
    )

    assessments = sig.skill_assessment_scores
    assessment_mismatch = False
    prof_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    for skill in candidate.skills:
        if skill.name in assessments and assessments[skill.name] < 30:
            if prof_map.get(skill.proficiency, 0) >= 0.75:
                assessment_mismatch = True
                break

    expert_zero = any(s.proficiency == "expert" and s.duration_months == 0 for s in candidate.skills)

    edu_tiers = [TIER_MAP.get(e.tier, 0) for e in candidate.education]
    max_tier = max(edu_tiers) if edu_tiers else 0
    field_cs = any(
        any(k in e.field_of_study.lower() for k in ("computer", "software", "information", "data"))
        for e in candidate.education
    )

    seniority_levels = [_seniority_level(c.title) for c in candidate.career_history]
    promotion_score = 0.0
    if len(seniority_levels) >= 2:
        promotion_score = max(0.0, (seniority_levels[-1] - seniority_levels[0]) / 6.0)

    short_stints = sum(1 for c in candidate.career_history if c.duration_months < 18)
    avg_tenure = (
        total_months / len(candidate.career_history) if candidate.career_history else 0.0
    )

    loc = p.get("location", "").lower()
    in_pune_noida = "pune" in loc or "noida" in loc

    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "years_experience": float(p.get("years_of_experience", 0)),
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
        "industry": p.get("current_industry", ""),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "company_size": p.get("current_company_size", ""),
        "career_text": career_text,
        "title_text": title_text,
        "summary_text": summary_text,
        "skill_names": json.dumps(skill_names),
        "skill_count": len(candidate.skills),
        "ai_keyword_count": ai_keyword_count,
        "num_roles": len(candidate.career_history),
        "avg_tenure_months": avg_tenure,
        "promotion_score": promotion_score,
        "leadership_mentions": len(LEADERSHIP_RE.findall(career_text)),
        "ship_mentions": len(SHIP_RE.findall(career_text)),
        "product_months_ratio": product_months / max(total_months, 1),
        "consulting_only_flag": int(consulting_only),
        "retrieval_ranking_mentions": len(RETRIEVAL_RE.findall(career_text)),
        "max_edu_tier": max_tier,
        "field_cs_related": int(field_cs),
        "total_tenure_months": total_months,
        "short_stints": short_stints,
        "template_summary": int(TEMPLATE_SNIPPET in summary_text),
        "assessment_mismatch": int(assessment_mismatch),
        "expert_zero_duration": int(expert_zero),
        "in_pune_noida": int(in_pune_noida),
        "days_since_active": days_since_active,
        "is_stale_6mo": int(days_since_active > 183),
        "profile_completeness_score": sig.profile_completeness_score,
        "open_to_work_flag": int(sig.open_to_work_flag),
        "profile_views_received_30d": sig.profile_views_received_30d,
        "applications_submitted_30d": sig.applications_submitted_30d,
        "recruiter_response_rate": sig.recruiter_response_rate,
        "avg_response_time_hours": sig.avg_response_time_hours,
        "connection_count": sig.connection_count,
        "endorsements_received": sig.endorsements_received,
        "notice_period_days": sig.notice_period_days,
        "preferred_work_mode": sig.preferred_work_mode,
        "willing_to_relocate": int(sig.willing_to_relocate),
        "github_activity_score": sig.github_activity_score,
        "search_appearance_30d": sig.search_appearance_30d,
        "saved_by_recruiters_30d": sig.saved_by_recruiters_30d,
        "interview_completion_rate": sig.interview_completion_rate,
        "offer_acceptance_rate": sig.offer_acceptance_rate,
        "verified_email": int(sig.verified_email),
        "verified_phone": int(sig.verified_phone),
        "linkedin_connected": int(sig.linkedin_connected),
        "skill_assessment_json": json.dumps(assessments),
    }
    return row


def build_features(candidates_path: Path, output_parquet: Path, output_ids: Path, limit: int | None = None) -> pd.DataFrame:
    pipeline = load_pipeline_config()
    reference_date = pipeline["ranking"]["reference_date"]
    rows: list[dict[str, Any]] = []
    for candidate in stream_candidates(candidates_path, limit=limit):
        rows.append(_candidate_row(candidate, reference_date))
    df = pd.DataFrame(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    save_json(output_ids, df["candidate_id"].tolist())
    logger.info("Wrote %d feature rows to %s", len(df), output_parquet)
    return df


def run_feature_build(limit: int | None = None) -> pd.DataFrame:
    pipeline = load_pipeline_config()
    candidates_path = resolve_path(pipeline["paths"]["candidates_jsonl"])
    output_parquet = resolve_path(pipeline["paths"]["candidate_features"])
    output_ids = resolve_path(pipeline["paths"]["candidate_ids"])
    return build_features(candidates_path, output_parquet, output_ids, limit=limit)
