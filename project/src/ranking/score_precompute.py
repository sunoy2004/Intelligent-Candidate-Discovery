"""Precompute component scores for fast ranking."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.behavioral.behavior_scorer import score_behavior
from src.config import load_json, load_pipeline_config, resolve_path
from src.feature_engineering.career_growth import score_career_growth
from src.feature_engineering.experience_matcher import score_experience
from src.feature_engineering.skill_matcher import score_skills
from src.startup_fit.startup_scorer import score_startup_fit
from src.trap_detection.trap_detector import detect_traps

logger = logging.getLogger(__name__)


def run_score_precompute(df: pd.DataFrame | None = None) -> None:
    pipeline = load_pipeline_config()
    if df is None:
        df = pd.read_parquet(resolve_path(pipeline["paths"]["candidate_features"]))
    job_features = load_json(resolve_path(pipeline["paths"]["job_features"]))

    sem_path = resolve_path(pipeline["paths"]["semantic_scores"])
    semantic = np.load(sem_path).astype(np.float32)
    semantic = (semantic - semantic.min()) / max(semantic.max() - semantic.min(), 1e-9)

    experience, title_fit = score_experience(df, job_features)
    skill = score_skills(df, job_features, title_fit)
    behavior, availability_mod = score_behavior(df)
    startup = score_startup_fit(df)
    growth = score_career_growth(df)
    trap = detect_traps(df)

    out = resolve_path("outputs/component_scores.npz")
    np.savez(
        out,
        semantic=semantic,
        skill=skill,
        experience=experience,
        behavior=behavior,
        startup=startup,
        growth=growth,
        trap=trap,
        availability_mod=availability_mod,
        title_fit=title_fit,
    )
    logger.info("Saved component scores to %s", out)
