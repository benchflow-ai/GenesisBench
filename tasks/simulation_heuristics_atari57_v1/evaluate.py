#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
RUNTIME_CANDIDATES = (
    TASK_DIR / "_runtime",
    TASK_DIR.parent / "src",
    TASK_DIR.parent.parent / "src",
)
for candidate in RUNTIME_CANDIDATES:
    if (candidate / "genesisbench").is_dir():
        sys.path.insert(0, str(candidate))
        break

from genesisbench.atari57 import (  # noqa: E402
    OBSERVATION_MODES,
    evaluate_atari57_artifact,
    load_atari57_artifact,
    load_hns_references,
)


DEFAULT_GAMES = (
    "Breakout-v5",
    "Freeway-v5",
    "MontezumaRevenge-v5",
    "Pong-v5",
    "Seaquest-v5",
    "Skiing-v5",
)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_strings(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an aggregate Atari57 policy artifact."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("final_artifact"),
    )
    parser.add_argument(
        "--games",
        default=",".join(DEFAULT_GAMES),
        help="Comma-separated Atari57 environment ids.",
    )
    parser.add_argument(
        "--obs-modes",
        default=",".join(OBSERVATION_MODES),
        help="Comma-separated observation modes.",
    )
    parser.add_argument(
        "--seeds",
        default="101,202,303",
        help=(
            "Three comma-separated seeds mapped in order to repeat-specific "
            "policy slots 0, 1, and 2."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--hns-table",
        type=Path,
        default=TASK_DIR / "task_context" / "atari57_games.csv",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 342-slot manifest and interaction ledger only.",
    )
    parser.add_argument("--json-output-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_atari57_artifact(args.artifact)
    budget = asdict(artifact.interaction_budget)
    if args.validate_only:
        payload = {
            "valid": True,
            "artifact": str(artifact.root),
            "policy_slots": len(artifact.policies),
            "interaction_budget": budget,
            "full_article_reproduction": (
                budget["completed_trajectories"] == budget["planned_trajectories"]
            ),
        }
    else:
        references = load_hns_references(args.hns_table)
        result = evaluate_atari57_artifact(
            artifact,
            games=_csv_strings(args.games),
            obs_modes=_csv_strings(args.obs_modes),
            seeds=_csv_ints(args.seeds),
            max_steps=args.max_steps,
            hns_references=references,
        )
        payload = {
            **result.to_dict(),
            "suite": "public_representative_subset",
            "interaction_budget": budget,
            "full_article_reproduction": False,
        }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output_file is not None:
        args.json_output_file.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_file.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
