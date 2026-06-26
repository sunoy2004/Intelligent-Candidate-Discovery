#!/usr/bin/env python3
"""Hackathon reproduction entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import rank_candidates


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Produce submission CSV from cached features")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (used for metadata)")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    args = parser.parse_args()

    candidates_path = Path(args.candidates).resolve()
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    out_path = Path(args.out).resolve()
    rank_candidates(output_path=out_path)


if __name__ == "__main__":
    main()
