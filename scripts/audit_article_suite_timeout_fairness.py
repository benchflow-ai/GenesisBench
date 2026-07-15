#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments" / "article_suite" / "protocol.toml"
SUBMISSIONS_ROOT = REPO_ROOT / "leaderboard" / "article_suite_submissions"
DEFAULT_OUTPUT = (
    REPO_ROOT / "leaderboard" / "article_suite_timeout_fairness_audit.json"
)

HISTORICALLY_AFFECTED_CELLS = (
    ("gpt-5.5", "simulation_heuristics_breakout_rgb_v1", 1, "pty_900s"),
    ("gpt-5.5", "simulation_heuristics_ant_v1", 4, "pty_900s"),
    ("gpt-5.5", "simulation_heuristics_montezuma_v1", 4, "idle_600s"),
    ("gpt-5.5", "simulation_heuristics_pong_ram_v1", 4, "pty_900s"),
    ("gpt-5.6-sol", "simulation_heuristics_halfcheetah_v1", 1, "pty_900s"),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d3_v1", 1, "pty_900s"),
    ("gpt-5.6-sol", "simulation_heuristics_ant_v1", 2, "pty_900s"),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d1_v1", 3, "idle_600s"),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d3_v1", 3, "pty_900s"),
    ("gpt-5.6-sol", "simulation_heuristics_breakout_ram_v1", 4, "pty_900s"),
)

RERUN_SCORE_CHANGES = (
    ("gpt-5.5", "simulation_heuristics_breakout_rgb_v1", 1, 21.299638989, 99.277978339),
    ("gpt-5.5", "simulation_heuristics_ant_v1", 4, -1.398078834, 0.0),
    ("gpt-5.5", "simulation_heuristics_montezuma_v1", 4, 0.0, 0.0),
    ("gpt-5.5", "simulation_heuristics_pong_ram_v1", 4, 25.0, 50.0),
    ("gpt-5.6-sol", "simulation_heuristics_halfcheetah_v1", 1, 26.12140084, 7.417455939),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d3_v1", 1, 40.106951872, 4.879679144),
    ("gpt-5.6-sol", "simulation_heuristics_ant_v1", 2, -7.591479, -13.495063777),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d1_v1", 3, 31.015418049, 113.009039251),
    ("gpt-5.6-sol", "simulation_heuristics_vizdoom_d3_v1", 3, 0.0, 22.961229947),
    ("gpt-5.6-sol", "simulation_heuristics_breakout_ram_v1", 4, 18.448637317, 100.0),
)

MODEL_IQM_CHANGES = (
    ("gpt-5.5", 10.662970787652174, 13.868245803826087),
    ("gpt-5.6-sol", 36.651764607913044, 40.67930709160869),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit timeout fairness of published article-suite cells."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _task_document(task: str) -> dict[str, Any]:
    path = REPO_ROOT / "tasks" / task / "task.md"
    return yaml.safe_load(path.read_text().split("---", 2)[1])


def main() -> None:
    args = parse_args()
    protocol = tomllib.loads(PROTOCOL_PATH.read_text())
    fairness = protocol["fairness"]
    expected_idle = int(fairness["agent_idle_timeout_sec"])
    expected_pty = int(fairness["daytona_pty_readline_timeout_sec"])
    metadata_paths = sorted(
        SUBMISSIONS_ROOT.glob("*/*/trial-*/metadata.json")
    )

    issues: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    wall_clock_timeout_cells: list[dict[str, Any]] = []
    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        model_id = metadata["model"]["id"]
        task = metadata["task"]
        trial = int(metadata["trial"])
        task_document = _task_document(task)
        expected_agent = int(task_document["agent"]["timeout_sec"])
        expected_verifier = int(task_document["verifier"]["timeout_sec"])
        expected_build = int(
            task_document["environment"]["build_timeout_sec"]
        )

        cell = {
            "model_id": model_id,
            "task": task,
            "trial": trial,
            "published_protocol_version": metadata.get(
                "published_protocol_version"
            ),
            "source_protocol_version": metadata.get("source_protocol_version"),
            "agent_timeout_sec": metadata.get("agent_timeout_sec"),
            "source_agent_idle_timeout_sec": metadata.get(
                "source_agent_idle_timeout_sec"
            ),
            "source_daytona_pty_readline_timeout_sec": metadata.get(
                "source_daytona_pty_readline_timeout_sec"
            ),
            "protocol_agent_idle_timeout_sec": metadata.get(
                "protocol_agent_idle_timeout_sec"
            ),
            "protocol_daytona_pty_readline_timeout_sec": metadata.get(
                "protocol_daytona_pty_readline_timeout_sec"
            ),
            "verifier_timeout_sec": expected_verifier,
            "build_timeout_sec": expected_build,
            "terminal_error_category": metadata.get(
                "terminal_error_category"
            ),
            "agent_timeout_info": metadata.get("agent_timeout_info"),
            "idle_timeout_info": metadata.get("idle_timeout_info"),
            "transport_error_info": metadata.get("transport_error_info"),
            "verifier_timeout_info": metadata.get("verifier_timeout_info"),
            "source_run_id": metadata.get("source_run_id"),
        }
        cells.append(cell)

        checks = (
            (
                "protocol_version",
                metadata.get("published_protocol_version"),
                protocol["version"],
            ),
            (
                "agent_timeout_sec",
                metadata.get("agent_timeout_sec"),
                expected_agent,
            ),
            (
                "protocol_agent_idle_timeout_sec",
                metadata.get("protocol_agent_idle_timeout_sec"),
                expected_idle,
            ),
            (
                "protocol_daytona_pty_readline_timeout_sec",
                metadata.get("protocol_daytona_pty_readline_timeout_sec"),
                expected_pty,
            ),
        )
        for field, actual, expected in checks:
            if actual != expected:
                issues.append(
                    {
                        "model_id": model_id,
                        "task": task,
                        "trial": trial,
                        "field": field,
                        "actual": actual,
                        "expected": expected,
                    }
                )

        historical_key = (model_id, task, trial)
        affected_keys = {
            (affected_model, affected_task, affected_trial)
            for (
                affected_model,
                affected_task,
                affected_trial,
                _,
            ) in HISTORICALLY_AFFECTED_CELLS
        }
        if historical_key in affected_keys:
            for field, actual, expected in (
                (
                    "source_protocol_version",
                    metadata.get("source_protocol_version"),
                    protocol["version"],
                ),
                (
                    "source_agent_idle_timeout_sec",
                    metadata.get("source_agent_idle_timeout_sec"),
                    expected_idle,
                ),
                (
                    "source_daytona_pty_readline_timeout_sec",
                    metadata.get("source_daytona_pty_readline_timeout_sec"),
                    expected_pty,
                ),
            ):
                if actual != expected:
                    issues.append(
                        {
                            "model_id": model_id,
                            "task": task,
                            "trial": trial,
                            "field": field,
                            "actual": actual,
                            "expected": expected,
                        }
                    )

        if metadata.get("idle_timeout_info") is not None:
            issues.append(
                {
                    "model_id": model_id,
                    "task": task,
                    "trial": trial,
                    "field": "idle_timeout_info",
                    "actual": metadata["idle_timeout_info"],
                    "expected": None,
                }
            )
        terminal_error = metadata.get("terminal_error") or ""
        if "PTY readline timeout" in terminal_error:
            issues.append(
                {
                    "model_id": model_id,
                    "task": task,
                    "trial": trial,
                    "field": "terminal_error",
                    "actual": terminal_error,
                    "expected": "no PTY timeout",
                }
            )

        agent_timeout = metadata.get("agent_timeout_info")
        if agent_timeout is not None:
            timeout_sec = agent_timeout.get("timeout_sec")
            if (
                agent_timeout.get("reason") != "wall_clock_timeout"
                or not isinstance(timeout_sec, int | float)
                or not math.isclose(float(timeout_sec), expected_agent)
            ):
                issues.append(
                    {
                        "model_id": model_id,
                        "task": task,
                        "trial": trial,
                        "field": "agent_timeout_info",
                        "actual": agent_timeout,
                        "expected": {
                            "reason": "wall_clock_timeout",
                            "timeout_sec": expected_agent,
                        },
                    }
                )
            else:
                wall_clock_timeout_cells.append(
                    {
                        "model_id": model_id,
                        "task": task,
                        "trial": trial,
                        "timeout_sec": timeout_sec,
                    }
                )

    expected_cells = 4 * 9 * int(protocol["trials"])
    if len(cells) != expected_cells:
        issues.append(
            {
                "field": "selected_cell_count",
                "actual": len(cells),
                "expected": expected_cells,
            }
        )

    report = {
        "status": "ok" if not issues else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol_version": protocol["version"],
        "expected_cells": expected_cells,
        "selected_cells": len(cells),
        "fairness": {
            "agent_idle_timeout_sec": expected_idle,
            "daytona_pty_readline_timeout_sec": expected_pty,
            "interpretation": (
                "historically affected cells rerun under standardized "
                "safeguards; unaffected source cells retained only when no "
                "idle or PTY timeout influenced the selected result"
            ),
        },
        "historically_affected_cells": [
            {
                "model_id": model_id,
                "task": task,
                "trial": trial,
                "reason": reason,
            }
            for model_id, task, trial, reason in HISTORICALLY_AFFECTED_CELLS
        ],
        "rerun_cell_count": len(HISTORICALLY_AFFECTED_CELLS),
        "rerun_score_changes": [
            {
                "model_id": model_id,
                "task": task,
                "trial": trial,
                "before_normalized_score": before,
                "after_normalized_score": after,
                "delta": after - before,
            }
            for model_id, task, trial, before, after in RERUN_SCORE_CHANGES
        ],
        "model_iqm_changes": [
            {
                "model_id": model_id,
                "before_iqm": before,
                "after_iqm": after,
                "delta": after - before,
            }
            for model_id, before, after in MODEL_IQM_CHANGES
        ],
        "selected_timeout_influenced_cells": issues,
        "shared_wall_clock_timeout_cells": wall_clock_timeout_cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
