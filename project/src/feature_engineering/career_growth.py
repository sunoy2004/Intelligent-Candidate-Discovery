"""Career growth scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_career_growth(df: pd.DataFrame) -> np.ndarray:
    promo = df["promotion_score"].astype(float).to_numpy()
    leadership = np.clip(df["leadership_mentions"].astype(float).to_numpy() / 3.0, 0.0, 1.0)
    short_penalty = np.clip(df["short_stints"].astype(float).to_numpy() / 4.0, 0.0, 0.5)
    tenure = np.clip(df["avg_tenure_months"].astype(float).to_numpy() / 36.0, 0.0, 1.0)
    scores = 0.55 * promo + 0.35 * leadership + 0.10 * tenure - short_penalty
    return np.clip(scores, 0.0, 1.0).astype(np.float32)
