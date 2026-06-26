"""Trap and honeypot detection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

NON_TECH_TITLES = (
    "business analyst", "hr manager", "accountant", "marketing manager",
    "content writer", "graphic designer", "sales executive", "customer support",
    "operations manager", "mechanical engineer", "civil engineer", "project manager",
)


def _title_is_non_tech(titles: pd.Series) -> np.ndarray:
    lowered = titles.str.lower()
    mask = np.zeros(len(titles), dtype=bool)
    for t in NON_TECH_TITLES:
        mask |= lowered.str.contains(t, regex=False)
    return mask


def detect_traps(df: pd.DataFrame) -> np.ndarray:
    yoe_months = df["years_experience"].astype(float).to_numpy() * 12.0
    total_months = df["total_tenure_months"].astype(float).to_numpy()
    penalties = np.zeros(len(df), dtype=np.float32)

    penalties += np.where(total_months > yoe_months + 24, 0.35, 0.0)
    penalties += np.where(df["expert_zero_duration"].astype(int).to_numpy() > 0, 0.20, 0.0)

    non_tech = _title_is_non_tech(df["current_title"])
    penalties += np.where(non_tech & (df["ai_keyword_count"].to_numpy() >= 7), 0.30, 0.0)
    penalties += np.where(df["consulting_only_flag"].astype(int).to_numpy() > 0, 0.10, 0.0)
    penalties += np.where(df["assessment_mismatch"].astype(int).to_numpy() > 0, 0.10, 0.0)
    penalties += np.where(
        (df["template_summary"].astype(int).to_numpy() > 0) & (df["ai_keyword_count"].to_numpy() >= 6),
        0.10,
        0.0,
    )
    penalties += np.where(
        (df["is_stale_6mo"].astype(int).to_numpy() > 0)
        & (df["recruiter_response_rate"].astype(float).to_numpy() < 0.1),
        0.15,
        0.0,
    )

    # langchain without core — row-wise only for skill json
    for idx in range(len(df)):
        skill_names = json.loads(df.iloc[idx]["skill_names"])
        names_lower = [s.lower() for s in skill_names]
        has_lc = any("langchain" in s for s in names_lower)
        has_core = any(
            any(k in s for k in ("pytorch", "tensorflow", "machine learning", "deep learning", "mlops", "rag"))
            for s in names_lower
        )
        if has_lc and not has_core:
            penalties[idx] += 0.15

    return np.clip(penalties, 0.0, 1.0)
