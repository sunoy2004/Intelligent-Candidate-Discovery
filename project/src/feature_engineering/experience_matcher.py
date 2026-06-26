"""Experience and title matching."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml


def _title_fit(title: str, taxonomy: dict) -> float:
    title = title.strip()
    if title in taxonomy.get("tier_a", []):
        return float(taxonomy.get("tier_a_score", 1.0))
    if title in taxonomy.get("tier_b", []):
        return float(taxonomy.get("tier_b_score", 0.7))
    if title in taxonomy.get("tier_c", []):
        return float(taxonomy.get("tier_c_score", 0.2))
    t = title.lower()
    if any(k in t for k in ("machine learning", "ml engineer", "ai engineer", "data scientist", "search engineer")):
        return 0.95
    if any(k in t for k in ("software", "backend", "cloud", "data engineer", "platform")):
        return 0.65
    return float(taxonomy.get("default_score", 0.4))


def _yoe_score(yoe: float, y_min: float, y_max: float) -> float:
    center = (y_min + y_max) / 2.0
    width = max(y_max - y_min, 1.0)
    return math.exp(-((yoe - center) ** 2) / (2 * (width / 2) ** 2))


def score_experience(df: pd.DataFrame, job_features: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    taxonomy = load_yaml("title_taxonomy.yaml")
    y_min = float(job_features.get("years_experience_min", 5))
    y_max = float(job_features.get("years_experience_max", 9))
    country_pref = job_features.get("country_preferred", "India")

    title_scores = df["current_title"].apply(lambda t: _title_fit(str(t), taxonomy)).astype(np.float32).to_numpy()
    yoe_s = df["years_experience"].astype(float).apply(lambda y: _yoe_score(y, y_min, y_max)).astype(np.float32).to_numpy()
    product_s = df["product_months_ratio"].astype(float).to_numpy()
    retrieval_s = np.clip(df["retrieval_ranking_mentions"].astype(float).to_numpy() / 3.0, 0.0, 1.0)

    loc_s = np.full(len(df), 0.3, dtype=np.float32)
    india_mask = df["country"] == country_pref
    loc_s = np.where(india_mask.to_numpy(), 0.7, loc_s)
    loc_s = np.where(india_mask.to_numpy() & (df["in_pune_noida"].astype(int).to_numpy() > 0), 1.0, loc_s)
    loc_s = np.where(
        india_mask.to_numpy()
        & (df["in_pune_noida"].astype(int).to_numpy() == 0)
        & (df["willing_to_relocate"].astype(int).to_numpy() > 0),
        0.85,
        loc_s,
    )

    scores = (
        0.35 * title_scores
        + 0.20 * yoe_s
        + 0.20 * product_s
        + 0.15 * retrieval_s
        + 0.10 * loc_s
    )
    return np.clip(scores, 0.0, 1.0).astype(np.float32), title_scores
