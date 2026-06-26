"""Pipeline orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import load_json, load_pipeline_config, resolve_path, save_json
from src.embeddings.tfidf_semantic import run_tfidf_semantic
from src.explanations.explainer import build_explanations
from src.feature_engineering.feature_builder import run_feature_build
from src.jd_analysis.job_analyzer import run_job_analysis
from src.ranking.ranker import compute_all_scores, get_top_k, load_dataframe, load_job_features
from src.ranking.score_precompute import run_score_precompute
from src.reranking.cross_encoder_reranker import rerank_top
from src.submission.submission_writer import validate_submission, write_submission

logger = logging.getLogger(__name__)


def precompute(limit: int | None = None) -> None:
    pipeline = load_pipeline_config()
    features_path = resolve_path(pipeline["paths"]["candidate_features"])

    if not features_path.exists() or limit is not None:
        logger.info("Stage 1: Job analysis")
        run_job_analysis()
        logger.info("Stage 2: Feature extraction")
        df = run_feature_build(limit=limit)
    else:
        logger.info("Stage 1-2: Skipping (using cached features)")
        import pandas as pd

        df = pd.read_parquet(features_path)
        if not resolve_path(pipeline["paths"]["job_features"]).exists():
            run_job_analysis()

    method = pipeline.get("semantic_method", "tfidf")
    logger.info("Stage 3: Semantic scoring (%s)", method)
    if method == "bge":
        from src.embeddings.embedding_engine import run_embeddings

        run_embeddings(df)
    else:
        run_tfidf_semantic(df)

    logger.info("Stage 4: Component scores")
    run_score_precompute(df)
    logger.info("Precompute complete (%d candidates)", len(df))


def rank_candidates(output_path: Path | None = None) -> None:
    pipeline = load_pipeline_config()
    weights = __import__("src.config", fromlist=["load_weights_config"]).load_weights_config()
    top_k_prerank = int(weights.get("top_k_prerank", pipeline["ranking"]["top_k_prerank"]))
    top_k_final = int(pipeline["ranking"]["top_k_final"])

    t0 = time.time()
    df = load_dataframe()
    job_features = load_job_features()
    jd_text = resolve_path(pipeline["paths"]["job_description"]).read_text(encoding="utf-8")

    result = compute_all_scores(df, job_features)
    top_ids, top_scores, _ = get_top_k(result, top_k_prerank)

    logger.info("Reranking top %d candidates", len(top_ids))
    final_ids, final_scores = rerank_top(df, top_ids, top_scores, jd_text)
    final_ids = final_ids[:top_k_final]
    final_scores = final_scores[:top_k_final]

    explanations = build_explanations(df, final_ids, job_features)
    out_csv = output_path or resolve_path(pipeline["paths"]["submission"])
    write_submission(out_csv, final_ids, final_scores, explanations)

    exp_path = resolve_path(pipeline["paths"]["explanations"])
    save_json(exp_path, explanations)

    validator = resolve_path(pipeline["paths"]["validator"])
    if not validate_submission(out_csv, validator):
        raise RuntimeError("Submission validation failed")

    elapsed = time.time() - t0
    logger.info("Ranking complete in %.1f seconds", elapsed)
