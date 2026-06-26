#!/usr/bin/env python3
"""Thin wrapper for precompute stage."""

from main import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "precompute", *sys.argv[1:]]
    main()
