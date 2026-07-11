#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO: public development evaluator."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("final_artifact/artifact.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    raise NotImplementedError(
        "Replace this template with a fast, machine-readable evaluator."
    )


if __name__ == "__main__":
    main()

