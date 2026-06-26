"""Behavioral signal scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_weights_config


def _norm_series(series: pd.Series, lo: float, hi: float) -> np.ndarray:
    return np.clip((series.astype(float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def score_behavior(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return behavior_score and availability_modifier."""
    weights_cfg = load_weights_config()

    response = df["recruiter_response_rate"].astype(float).to_numpy()
    recency = 1.0 - _norm_series(df["days_since_active"], 0, 365)
    open_flag = df["open_to_work_flag"].astype(float).to_numpy()
    saved = _norm_series(df["saved_by_recruiters_30d"], 0, 15)
    interview = df["interview_completion_rate"].astype(float).to_numpy()
    notice_penalty = _norm_series(df["notice_period_days"], 30, 120)

    github = df["github_activity_score"].astype(float).to_numpy()
    github_s = np.where(github >= 0, _norm_series(df["github_activity_score"], 0, 100), 0.3)

    behavior = np.clip(
        0.30 * response
        + 0.25 * recency
        + 0.15 * open_flag
        + 0.15 * saved
        + 0.10 * interview
        + 0.05 * github_s,
        0.0,
        1.0,
    ).astype(np.float32)

    modifier = np.ones(len(df), dtype=np.float32)
    if weights_cfg.get("availability_multiplier", True):
        mod = 0.85 + 0.10 * response + 0.10 * recency + 0.05 * open_flag - 0.15 * notice_penalty
        mod = np.where(df["is_stale_6mo"].astype(int).to_numpy() > 0, mod - 0.15, mod)
        modifier = np.clip(
            mod,
            weights_cfg.get("availability_min", 0.70),
            weights_cfg.get("availability_max", 1.10),
        ).astype(np.float32)

    return behavior, modifier
