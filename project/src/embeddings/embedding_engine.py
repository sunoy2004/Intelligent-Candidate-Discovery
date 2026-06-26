"""Semantic embedding and similarity scoring."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import load_pipeline_config, resolve_path

logger = logging.getLogger(__name__)


def _fuse_text(row: pd.Series, weights: dict[str, float]) -> str:
    """Compose a single document emphasizing career over summary."""
    career = str(row.get("career_text", ""))[:3000]
    title = str(row.get("title_text", ""))[:400]
    summary = str(row.get("summary_text", ""))[:600]
    repeat_career = max(1, int(weights["career"] / max(weights["title"], 0.01)))
    career_block = " ".join([career] * repeat_career)
    return f"Title: {title}\nCareer: {career_block}\nSummary: {summary}"


def _encode_batch(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def build_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pipeline = load_pipeline_config()
    weights = pipeline["embedding_weights"]
    model_name = pipeline["models"]["embedding_model"]
    batch_size = pipeline["models"]["embedding_batch_size"]

    model = SentenceTransformer(model_name, cache_folder=str(resolve_path("models")))
    fused_texts = [_fuse_text(row, weights) for _, row in df.iterrows()]
    fused = _encode_batch(model, fused_texts, batch_size)

    jd_path = resolve_path(pipeline["paths"]["jd_embedding"])
    if not jd_path.exists():
        from src.jd_analysis.job_analyzer import embed_jd

        jd_text = resolve_path(pipeline["paths"]["job_description"]).read_text(encoding="utf-8")
        jd_emb = embed_jd(jd_text, model_name, jd_path)
    else:
        jd_emb = np.load(jd_path)

    semantic_scores = fused @ jd_emb
    return fused.astype(np.float32), semantic_scores.astype(np.float32)


def run_embeddings(df: pd.DataFrame | None = None) -> None:
    pipeline = load_pipeline_config()
    if df is None:
        df = pd.read_parquet(resolve_path(pipeline["paths"]["candidate_features"]))

    fused, semantic_scores = build_embeddings(df)
    emb_path = resolve_path(pipeline["paths"]["candidate_embeddings"])
    sem_path = resolve_path(pipeline["paths"]["semantic_scores"])
    np.save(emb_path, fused)
    np.save(sem_path, semantic_scores)
    logger.info("Saved embeddings %s and semantic scores %s", emb_path, sem_path)
