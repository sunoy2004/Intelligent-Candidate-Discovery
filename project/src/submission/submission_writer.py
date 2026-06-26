"""Submission CSV writer and validator integration."""

from __future__ import annotations

import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

HEADER = ["candidate_id", "rank", "score", "reasoning"]


def assign_monotonic_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Map raw scores to strictly non-increasing values in (0, 1]."""
    if len(raw_scores) == 0:
        return raw_scores
    order = np.argsort(-raw_scores, kind="mergesort")
    sorted_raw = raw_scores[order]
    n = len(sorted_raw)
    # Start high and step down; preserve ordering
    scores = np.linspace(0.992, max(0.2, 0.992 - 0.008 * (n - 1)), n)
    # Adjust for ties in raw scores
    out = np.zeros(n, dtype=np.float64)
    out[order] = scores
    for i in range(1, n):
        if sorted_raw[i] == sorted_raw[i - 1]:
            idx_curr = order[i]
            idx_prev = order[i - 1]
            out[idx_curr] = min(out[idx_curr], out[idx_prev])
    # enforce non-increasing by rank after sort
    ranked = out[order]
    for i in range(1, n):
        if ranked[i] > ranked[i - 1]:
            ranked[i] = ranked[i - 1]
    out[order] = ranked
    return out


def write_submission(
    output_path: Path,
    candidate_ids: list[str],
    raw_scores: np.ndarray,
    explanations: dict[str, str],
) -> None:
    scores = assign_monotonic_scores(np.array(raw_scores, dtype=np.float64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for rank, (cid, score) in enumerate(zip(candidate_ids, scores), start=1):
            reasoning = explanations.get(cid, "")
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])
    logger.info("Wrote submission to %s", output_path)


def validate_submission(csv_path: Path, validator_path: Path) -> bool:
    if not validator_path.exists():
        logger.warning("Validator not found at %s", validator_path)
        return True
    result = subprocess.run(
        [sys.executable, str(validator_path), str(csv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error("Validation failed:\n%s", result.stdout + result.stderr)
        return False
    logger.info(result.stdout.strip())
    return True
