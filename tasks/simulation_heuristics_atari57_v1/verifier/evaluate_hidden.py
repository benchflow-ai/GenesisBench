#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


VERIFIER_DIR = Path(__file__).resolve().parent
RUNTIME_CANDIDATES = [Path("/opt/genesisbench")]
RUNTIME_CANDIDATES.extend(ancestor / "src" for ancestor in VERIFIER_DIR.parents)
for candidate in RUNTIME_CANDIDATES:
    if (candidate / "genesisbench").is_dir():
        sys.path.insert(0, str(candidate))
        break

from genesisbench.atari57 import (  # noqa: E402
    ATARI57_GAMES,
    OBSERVATION_MODES,
    Atari57Artifact,
    Atari57ArtifactError,
    deterministic_test_env_factory,
    evaluate_atari57_artifact,
    load_atari57_artifact,
    load_hns_references,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hidden aggregate Atari57 evaluation."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=VERIFIER_DIR / "config.toml",
        help="Evaluation suite config; production can inject a private file.",
    )
    parser.add_argument(
        "--anchors",
        type=Path,
        default=VERIFIER_DIR / "anchors.json",
        help="Normalization anchors matching the selected suite.",
    )
    return parser.parse_args()


def _sequence(
    value: object,
    *,
    all_values: tuple[str, ...],
    shorthand: str,
    field: str,
) -> tuple[str, ...]:
    if value == shorthand:
        return all_values
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be {shorthand!r} or an array")
    return tuple(value)


def _evaluate_raw(
    artifact: Atari57Artifact,
    *,
    config_path: Path,
    evaluation: dict[str, Any],
) -> Any:
    backend = evaluation.get("backend", "envpool")
    if backend == "envpool":
        env_factory = None
    elif backend == "deterministic":
        env_factory = deterministic_test_env_factory
    else:
        raise ValueError(f"Unsupported evaluation backend: {backend!r}")

    games = _sequence(
        evaluation.get("games", "atari57"),
        all_values=ATARI57_GAMES,
        shorthand="atari57",
        field="evaluation.games",
    )
    obs_modes = _sequence(
        evaluation.get("obs_modes", "both"),
        all_values=OBSERVATION_MODES,
        shorthand="both",
        field="evaluation.obs_modes",
    )
    seeds = evaluation.get("seeds")
    if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("evaluation.seeds must be an integer array")
    if len(seeds) != 3:
        raise ValueError(
            "Hidden Atari57 evaluation requires exactly three repeat seeds"
        )
    hns_path = config_path.parent / evaluation["hns_table"]
    references = load_hns_references(hns_path)
    result = evaluate_atari57_artifact(
        artifact,
        games=games,
        obs_modes=obs_modes,
        seeds=seeds,
        max_steps=int(evaluation["max_steps"]),
        hns_references=references,
        env_factory=env_factory,
    )
    return result


def _anchor_score(anchors: dict[str, Any], name: str) -> float:
    anchor = anchors[name]
    score = anchor.get("score")
    if not isinstance(score, int | float):
        raise ValueError(f"{name} must declare a numeric score")
    return float(score)


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


def _zero_payload(
    *,
    starter_score: float,
    reference_score: float,
    reason: str,
    artifact: Atari57Artifact | None,
) -> dict[str, Any]:
    return {
        "score": 0.0,
        "normalized_score": 0.0,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "evaluation": None,
        "interaction_budget": (
            asdict(artifact.interaction_budget) if artifact is not None else None
        ),
        "protocol_complete": False,
        "disqualification_reason": reason,
    }


def _is_protocol_complete(artifact: Atari57Artifact) -> bool:
    budget = artifact.interaction_budget
    return budget.completed_trajectories == budget.planned_trajectories == 342


def main() -> None:
    args = parse_args()
    config = tomllib.loads(args.config.read_text())
    anchors = json.loads(args.anchors.read_text())
    evaluation = config["evaluation"]
    starter_score = _anchor_score(anchors, "starter_policy")
    reference_score = _anchor_score(anchors, "reference_policy")
    if reference_score == starter_score:
        raise ValueError("starter and reference anchors must have different scores")

    try:
        artifact = load_atari57_artifact(args.artifact)
    except (Atari57ArtifactError, FileNotFoundError) as error:
        _emit(
            _zero_payload(
                starter_score=starter_score,
                reference_score=reference_score,
                reason=f"Invalid aggregate artifact: {error}",
                artifact=None,
            ),
            args.output,
        )
        return

    protocol_complete = _is_protocol_complete(artifact)
    if (
        evaluation.get("require_complete_search_ledger", False)
        and not protocol_complete
    ):
        completed = artifact.interaction_budget.completed_trajectories
        _emit(
            _zero_payload(
                starter_score=starter_score,
                reference_score=reference_score,
                reason=(
                    "Official suite requires 342 completed search-ledger "
                    f"records; found {completed}."
                ),
                artifact=artifact,
            ),
            args.output,
        )
        return

    result = _evaluate_raw(
        artifact,
        config_path=args.config,
        evaluation=evaluation,
    )
    normalized_score = round(
        100.0 * (result.score - starter_score) / (reference_score - starter_score),
        9,
    )
    payload = {
        "score": result.score,
        "normalized_score": normalized_score,
        "starter_score": starter_score,
        "reference_score": reference_score,
        "evaluation": result.to_dict(),
        "interaction_budget": asdict(artifact.interaction_budget),
        "protocol_complete": protocol_complete,
        "disqualification_reason": None,
    }
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
