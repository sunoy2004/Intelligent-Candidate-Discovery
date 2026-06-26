"""Fast semantic similarity via chunked hashing TF-IDF."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

from src.config import load_pipeline_config, resolve_path
from src.embeddings.embedding_engine import _fuse_text

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000


def _batched_docs(df: pd.DataFrame, weights: dict) -> list[list[str]]:
    docs = [_fuse_text(row, weights) for _, row in df.iterrows()]
    return [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]


def build_tfidf_semantic(df: pd.DataFrame) -> np.ndarray:
    pipeline = load_pipeline_config()
    weights = pipeline["embedding_weights"]
    jd_text = resolve_path(pipeline["paths"]["job_description"]).read_text(encoding="utf-8")

    hasher = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        ngram_range=(1, 2),
        lowercase=True,
        dtype=np.float32,
    )
    tfidf = TfidfTransformer(sublinear_tf=True)

    batches = _batched_docs(df, weights)
    first = hasher.transform(batches[0])
    tfidf.fit(first)

    jd_vec = tfidf.transform(hasher.transform([jd_text]))
    scores_list: list[np.ndarray] = []

    for i, batch in enumerate(batches):
        matrix = tfidf.transform(hasher.transform(batch))
        batch_scores = linear_kernel(jd_vec, matrix).ravel().astype(np.float32)
        scores_list.append(batch_scores)
        logger.info("Semantic batch %d/%d", i + 1, len(batches))

    return np.concatenate(scores_list)


def run_tfidf_semantic(df: pd.DataFrame | None = None) -> None:
    pipeline = load_pipeline_config()
    if df is None:
        df = pd.read_parquet(resolve_path(pipeline["paths"]["candidate_features"]))

    scores = build_tfidf_semantic(df)
    sem_path = resolve_path(pipeline["paths"]["semantic_scores"])
    np.save(sem_path, scores)
    logger.info("Saved TF-IDF semantic scores (%d) to %s", len(scores), sem_path)
