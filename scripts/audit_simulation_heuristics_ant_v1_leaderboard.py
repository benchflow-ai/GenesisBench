#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "gpt-5.6-sol": "xhigh",
    "gpt-5.5": "xhigh",
    "claude-opus-4.8": "max",
    "gpt-5.4-mini": "xhigh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the generated Simulation Heuristics Ant v1 leaderboard."
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "simulation_heuristics_ant_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.leaderboard.read_text())
    assert payload["benchmark"] == "simulation_heuristics_ant_v1"
    rows = payload["rows"]
    by_model = {row["model_id"]: row for row in rows}
    assert set(by_model) == set(EXPECTED_MODELS), (
        set(EXPECTED_MODELS) - set(by_model),
        set(by_model) - set(EXPECTED_MODELS),
    )

    for model_id, expected_reasoning in EXPECTED_MODELS.items():
        row = by_model[model_id]
        assert row["runtime"] == "docker"
        assert row["budget_minutes"] == 30
        assert row["reasoning"] == expected_reasoning
        assert isinstance(row["score"], (int, float))
        assert isinstance(row["normalized_score"], (int, float))
        assert isinstance(row["hidden_nominal"], (int, float))
        assert isinstance(row["hidden_robustness"], (int, float))

        submission_path = REPO_ROOT / row["submission_path"]
        submission_dir = submission_path.parent
        metadata = json.loads((submission_dir / "metadata.json").read_text())
        score = json.loads((submission_dir / "score.json").read_text())
        assert metadata["runtime"] == "docker"
        assert metadata["budget_minutes"] == 30
        assert metadata["reasoning"] == expected_reasoning
        assert score["score"] == row["score"]
        assert submission_path.is_file()
        assert score["hidden_nominal"]["policy_path"] == row["submission_path"]
        assert score["hidden_robustness"]["policy_path"] == row["submission_path"]

        source_run = (
            REPO_ROOT
            / "leaderboard"
            / "runs"
            / metadata["source_run_id"]
        )
        if source_run.is_dir():
            agent_summary_path = source_run / "agent_summary.json"
            if agent_summary_path.is_file():
                agent_summary = json.loads(agent_summary_path.read_text())
                assert agent_summary["status"] != "error"
            assert (source_run / "events.jsonl").stat().st_size > 0
            assert not (source_run / ".provider_env.json").exists()

    ranks = [row["rank"] for row in rows]
    assert ranks == list(range(1, len(rows) + 1))
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores, reverse=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "models": sorted(by_model),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
