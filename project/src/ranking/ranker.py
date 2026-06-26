"""Weighted ranking engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_json, load_pipeline_config, load_weights_config, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class RankingResult:
    candidate_ids: list[str]
    final_scores: np.ndarray
    component_scores: dict[str, np.ndarray]


def _load_components() -> dict[str, np.ndarray]:
    path = resolve_path("outputs/component_scores.npz")
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}
    return {}


def compute_all_scores(df: pd.DataFrame, job_features: dict) -> RankingResult:
    weights = load_weights_config()
    components = _load_components()

    if components:
        semantic = components["semantic"]
        skill = components["skill"]
        experience = components["experience"]
        behavior = components["behavior"]
        startup = components["startup"]
        growth = components["growth"]
        trap = components["trap"]
        availability_mod = components["availability_mod"]
        title_scores = components["title_fit"]
    else:
        from src.behavioral.behavior_scorer import score_behavior
        from src.feature_engineering.career_growth import score_career_growth
        from src.feature_engineering.experience_matcher import score_experience
        from src.feature_engineering.skill_matcher import score_skills
        from src.startup_fit.startup_scorer import score_startup_fit
        from src.trap_detection.trap_detector import detect_traps

        pipeline = load_pipeline_config()
        sem_path = resolve_path(pipeline["paths"]["semantic_scores"])
        semantic = np.load(sem_path).astype(np.float32)
        semantic = (semantic - semantic.min()) / max(semantic.max() - semantic.min(), 1e-9)
        experience, title_scores = score_experience(df, job_features)
        skill = score_skills(df, job_features, title_scores)
        behavior, availability_mod = score_behavior(df)
        startup = score_startup_fit(df)
        growth = score_career_growth(df)
        trap = detect_traps(df)

    trap_penalty = np.clip(trap * weights.get("trap_penalty_max", 0.35), 0.0, weights.get("trap_penalty_max", 0.35))

    base = (
        weights["semantic_score"] * semantic
        + weights["skill_score"] * skill
        + weights["experience_score"] * experience
        + weights["behavior_score"] * behavior
        + weights["startup_score"] * startup
        + weights["career_growth_score"] * growth
    )
    final = (base - trap_penalty) * availability_mod
    final = np.clip(final, 0.0, None)
    final = np.where(trap > 0.5, final * 0.25, final)

    return RankingResult(
        candidate_ids=df["candidate_id"].tolist(),
        final_scores=final.astype(np.float32),
        component_scores={
            "semantic": semantic,
            "skill": skill,
            "experience": experience,
            "behavior": behavior,
            "startup": startup,
            "growth": growth,
            "trap_penalty": trap_penalty,
            "availability_mod": availability_mod,
            "title_fit": title_scores,
        },
    )


def get_top_k(result: RankingResult, k: int) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    order = np.argsort(-result.final_scores, kind="mergesort")
    if len(order) > k:
        order = order[:k]
    ids = [result.candidate_ids[i] for i in order]
    scores = result.final_scores[order]
    components = {name: arr[order] for name, arr in result.component_scores.items()}
    return ids, scores, components


def load_dataframe() -> pd.DataFrame:
    pipeline = load_pipeline_config()
    return pd.read_parquet(resolve_path(pipeline["paths"]["candidate_features"]))


def load_job_features() -> dict:
    pipeline = load_pipeline_config()
    return load_json(resolve_path(pipeline["paths"]["job_features"]))
