"""Startup and product fit scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_startup_fit(df: pd.DataFrame) -> np.ndarray:
    product = df["product_months_ratio"].astype(float).to_numpy()
    ship = np.clip(df["ship_mentions"].astype(float).to_numpy() / 2.0, 0.0, 1.0)
    consulting_penalty = np.where(df["consulting_only_flag"].astype(int).to_numpy() > 0, 0.25, 0.0)
    leadership = np.clip(df["leadership_mentions"].astype(float).to_numpy() / 2.0, 0.0, 1.0)
    scores = 0.45 * product + 0.35 * ship + 0.20 * leadership - consulting_penalty
    return np.clip(scores, 0.0, 1.0).astype(np.float32)
