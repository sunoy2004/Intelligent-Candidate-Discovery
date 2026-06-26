#!/usr/bin/env python3
"""Main entry point for the Redrob ranking engine."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import precompute, rank_candidates


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Redrob candidate ranking pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("precompute", help="Build features and embeddings offline")
    p_pre.add_argument("--limit", type=int, default=None, help="Limit candidates for testing")

    sub.add_parser("rank", help="Run ranking and write submission.csv")

    args = parser.parse_args()
    if args.command == "precompute":
        precompute(limit=args.limit)
    elif args.command == "rank":
        rank_candidates()


if __name__ == "__main__":
    main()
