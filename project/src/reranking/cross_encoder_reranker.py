"""Cross-encoder reranking for top candidates."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.config import load_pipeline_config, load_weights_config, resolve_path

logger = logging.getLogger(__name__)


def rerank_top(
    df: pd.DataFrame,
    candidate_ids: list[str],
    base_scores: np.ndarray,
    jd_text: str,
) -> tuple[list[str], np.ndarray]:
    weights = load_weights_config()
    pipeline = load_pipeline_config()
    blend = float(weights.get("rerank_blend", 0.30))
    batch_size = int(pipeline["models"]["reranker_batch_size"])
    model_name = pipeline["models"]["reranker_model"]

    subset = df.set_index("candidate_id").loc[candidate_ids].reset_index()
    pairs = [
        (jd_text[:1500], f"{row['title_text']} {row['career_text']}"[:2000])
        for _, row in subset.iterrows()
    ]

    model = CrossEncoder(model_name, max_length=512, cache_folder=str(resolve_path("models")))
    rerank_raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    rerank_raw = np.array(rerank_raw, dtype=np.float32)
    rerank_norm = (rerank_raw - rerank_raw.min()) / max(rerank_raw.max() - rerank_raw.min(), 1e-9)

    base_norm = (base_scores - base_scores.min()) / max(base_scores.max() - base_scores.min(), 1e-9)
    combined = (1.0 - blend) * base_norm + blend * rerank_norm

    order = np.argsort(-combined, kind="mergesort")
    sorted_ids = [candidate_ids[i] for i in order]
    sorted_scores = combined[order]
    return sorted_ids, sorted_scores
