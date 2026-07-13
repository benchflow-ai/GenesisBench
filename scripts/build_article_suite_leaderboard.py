#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow._utils.task_authoring import task_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "experiments" / "article_suite" / "models.toml"
TASKS = (
    "simulation_heuristics_ant_v1",
    "simulation_heuristics_pong_ram_v1",
    "simulation_heuristics_breakout_ram_v1",
    "simulation_heuristics_breakout_rgb_v1",
    "simulation_heuristics_halfcheetah_v1",
    "simulation_heuristics_vizdoom_d1_v1",
    "simulation_heuristics_vizdoom_d3_v1",
    "simulation_heuristics_atari57_v1",
    "simulation_heuristics_montezuma_v1",
)
TASK_DIGEST_COMPATIBILITY = {
    "simulation_heuristics_ant_v1": {
        "sha256:bbb533da0cb86459f4d49dee667e6c73ac54c0188bc40e54e911d50ef3c3bc38": (
            "Score-equivalent to the current task. The only later change adds "
            "a fail-closed internal timeout for candidates whose verifier "
            "would otherwise exceed BenchFlow's deadline."
        )
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the aggregate nine-task article-suite leaderboard."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "runs" / "article_suite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "article_suite.json",
    )
    return parser.parse_args()


def _result_files(model_root: Path) -> list[Path]:
    jobs = model_root / "jobs"
    candidates = sorted(
        path
        for path in jobs.glob("*/results.jsonl")
        if path.parent.name != jobs.name
    )
    if not candidates:
        candidates = sorted(jobs.glob("results.jsonl"))
    return candidates


def _load_results(
    model_root: Path,
    *,
    expected_tasks: tuple[str, ...] = TASKS,
    allow_partial_errors: bool = False,
) -> dict[str, dict[str, Any]]:
    files = _result_files(model_root)
    if len(files) != 1:
        raise RuntimeError(
            f"{model_root} must contain exactly one top-level results.jsonl; "
            f"found {len(files)}"
        )
    rows: dict[str, dict[str, Any]] = {}
    for line in files[0].read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_name = row.get("info", {}).get("task_name")
        if task_name not in expected_tasks:
            continue
        error = row.get("error")
        rollout_dir = row.get("info", {}).get("rollout_dir")
        score_details = (
            Path(rollout_dir) / "verifier" / "genesis-score.json"
            if isinstance(rollout_dir, str)
            else None
        )
        has_score_details = bool(
            score_details is not None and score_details.is_file()
        )
        if error is not None and not (
            isinstance(error, dict)
            and error.get("error") == "missing_llm_trajectory"
        ) and not has_score_details:
            if allow_partial_errors:
                continue
            raise RuntimeError(f"{model_root.name}/{task_name} contains an error")
        reward = row.get("reward")
        if (
            not isinstance(reward, int | float)
            or isinstance(reward, bool)
            or not math.isfinite(float(reward))
        ):
            raise RuntimeError(
                f"{model_root.name}/{task_name} has invalid reward {reward!r}"
            )
        rows[task_name] = row
    missing = sorted(set(expected_tasks) - set(rows))
    if missing and not allow_partial_errors:
        raise RuntimeError(
            f"{model_root.name} is missing article-suite tasks: "
            + ", ".join(missing)
        )
    return rows


def _normalized_task_score(row: dict[str, Any]) -> float:
    rollout_dir = row.get("info", {}).get("rollout_dir")
    if isinstance(rollout_dir, str):
        details_path = Path(rollout_dir) / "verifier" / "genesis-score.json"
        if details_path.is_file():
            details = json.loads(details_path.read_text())
            normalized = details.get("normalized_score")
            if (
                isinstance(normalized, int | float)
                and not isinstance(normalized, bool)
                and math.isfinite(float(normalized))
            ):
                return float(normalized)
    return 100.0 * float(row["reward"])


def _sanitize_score_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "submitted_artifact"
                if key in {"policy_path", "artifact_path"}
                else _sanitize_score_paths(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_score_paths(item) for item in value]
    return value


def _digest_compatibility_note(
    task: str,
    recorded_digest: str,
    current_digest: str,
) -> str | None:
    if recorded_digest == current_digest:
        return None
    return TASK_DIGEST_COMPATIBILITY.get(task, {}).get(recorded_digest)


def _expected_models() -> dict[str, dict[str, Any]]:
    payload = tomllib.loads(MODELS_PATH.read_text())
    return {model["id"]: model for model in payload["models"]}


def _latest_model_runs(runs_root: Path) -> dict[str, Path]:
    selected: dict[str, tuple[float, Path]] = {}
    for metadata_path in runs_root.glob("*/*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dry_run"):
            continue
        model_id = metadata.get("model", {}).get("id")
        finished_at = metadata.get("finished_at")
        if not isinstance(model_id, str) or not isinstance(
            finished_at,
            int | float,
        ):
            continue
        current = selected.get(model_id)
        if current is None or float(finished_at) > current[0]:
            selected[model_id] = (float(finished_at), metadata_path.parent)
    return {model_id: item[1] for model_id, item in selected.items()}


def _latest_task_results(
    runs_root: Path,
) -> dict[
    str,
    dict[str, tuple[float, dict[str, Any], Path, str]],
]:
    selected: dict[
        str,
        dict[str, tuple[float, dict[str, Any], Path, str]],
    ] = {}
    for metadata_path in runs_root.glob("*/*/run_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("dry_run"):
            continue
        model_id = metadata.get("model", {}).get("id")
        finished_at = metadata.get("finished_at")
        configured_tasks = metadata.get("tasks")
        if (
            not isinstance(model_id, str)
            or not isinstance(finished_at, int | float)
            or not isinstance(configured_tasks, list)
            or not configured_tasks
            or not all(task in TASKS for task in configured_tasks)
        ):
            continue
        model_root = metadata_path.parent
        manifest_path = model_root / "task_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest_digests = {
            item.get("task_id"): item.get("digest")
            for item in manifest.get("tasks", [])
            if isinstance(item, dict)
        }
        if any(
            not isinstance(manifest_digests.get(task), str)
            for task in configured_tasks
        ):
            continue
        try:
            rows = _load_results(
                model_root,
                expected_tasks=tuple(configured_tasks),
                allow_partial_errors=True,
            )
        except RuntimeError:
            continue
        model_rows = selected.setdefault(model_id, {})
        for task, row in rows.items():
            current = model_rows.get(task)
            if current is None or float(finished_at) > current[0]:
                model_rows[task] = (
                    float(finished_at),
                    row,
                    model_root,
                    manifest_digests[task],
                )
    return selected


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    expected_models = _expected_models()
    task_results = _latest_task_results(runs_root)
    missing_models = sorted(set(expected_models) - set(task_results))
    if missing_models:
        raise RuntimeError(
            "Missing completed article-suite model runs: "
            + ", ".join(missing_models)
        )
    ranked: list[dict[str, Any]] = []
    submissions_root = args.output.parent / "article_suite_submissions"
    for model_id in expected_models:
        model_results = task_results[model_id]
        missing_tasks = sorted(set(TASKS) - set(model_results))
        if missing_tasks:
            raise RuntimeError(
                f"{model_id} is missing completed task results: "
                + ", ".join(missing_tasks)
            )
        stale_tasks = [
            task
            for task in TASKS
            if _digest_compatibility_note(
                task,
                model_results[task][3],
                task_digest(REPO_ROOT / "tasks" / task),
            )
            is None
            and model_results[task][3]
            != task_digest(REPO_ROOT / "tasks" / task)
        ]
        if stale_tasks:
            raise RuntimeError(
                f"{model_id} has stale task digests: "
                + ", ".join(stale_tasks)
            )
        latest_task = max(
            model_results.values(),
            key=lambda item: item[0],
        )
        metadata = json.loads(
            (latest_task[2] / "run_metadata.json").read_text()
        )
        task_scores = {
            task: _normalized_task_score(model_results[task][1])
            for task in TASKS
        }
        submission_details: dict[str, str] = {}
        for task in TASKS:
            _, result_row, model_root, digest = model_results[task]
            current_digest = task_digest(REPO_ROOT / "tasks" / task)
            compatibility_note = _digest_compatibility_note(
                task,
                digest,
                current_digest,
            )
            task_metadata = json.loads(
                (model_root / "run_metadata.json").read_text()
            )
            rollout_dir = Path(result_row["info"]["rollout_dir"])
            source_score = rollout_dir / "verifier" / "genesis-score.json"
            if not source_score.is_file():
                raise RuntimeError(
                    f"{model_id}/{task} has no verifier genesis-score.json"
                )
            destination = submissions_root / model_id / task
            destination.mkdir(parents=True, exist_ok=True)
            sanitized_score = _sanitize_score_paths(
                json.loads(source_score.read_text())
            )
            (destination / "score.json").write_text(
                json.dumps(sanitized_score, indent=2, sort_keys=True) + "\n"
            )
            source_run = str(model_root.relative_to(REPO_ROOT))
            metadata_payload = {
                "benchmark": "learning_beyond_gradients_article_suite",
                "task": task,
                "task_digest": digest,
                "current_task_digest": current_digest,
                "digest_compatibility_note": compatibility_note,
                "model": task_metadata["model"],
                "harness": task_metadata["harness"],
                "provider_reasoning_effort": task_metadata[
                    "provider_reasoning_effort"
                ],
                "normalized_score": task_scores[task],
                "source_run_id": source_run,
                "finished_at": task_metadata["finished_at"],
                "tool_calls": result_row.get("metrics", {}).get(
                    "n_tool_calls",
                    result_row.get("total_tool_calls"),
                ),
                "token_usage": result_row.get("token_usage"),
            }
            (destination / "metadata.json").write_text(
                json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n"
            )
            submission_details[task] = str(
                (destination / "score.json").relative_to(REPO_ROOT)
            )
        average = sum(task_scores.values()) / len(TASKS)
        ranked.append(
            {
                "model_id": metadata["model"]["id"],
                "model": metadata["model"]["display_name"],
                "harness": metadata["harness"],
                "provider_reasoning_effort": metadata[
                    "provider_reasoning_effort"
                ],
                "average_normalized_score": average,
                "task_scores": task_scores,
                "submission_details": submission_details,
                "source_runs": {
                    task: str(
                        model_results[task][2].relative_to(REPO_ROOT)
                    )
                    for task in TASKS
                },
            }
        )
    ranked.sort(
        key=lambda row: row["average_normalized_score"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    payload = {
        "benchmark": "learning_beyond_gradients_article_suite",
        "task_count": len(TASKS),
        "tasks": list(TASKS),
        "aggregation": "arithmetic_mean_of_normalized_task_scores",
        "task_digest_compatibility": TASK_DIGEST_COMPATIBILITY,
        "source_runs": {
            model_id: {
                task: str(task_results[model_id][task][2].relative_to(REPO_ROOT))
                for task in TASKS
            }
            for model_id in expected_models
        },
        "generated_at": datetime.fromtimestamp(
            max(
                item[0]
                for model_results in task_results.values()
                for item in model_results.values()
            ),
            UTC,
        ).isoformat(),
        "rows": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    labels = {
        "simulation_heuristics_ant_v1": "Ant",
        "simulation_heuristics_pong_ram_v1": "Pong",
        "simulation_heuristics_breakout_ram_v1": "Breakout RAM",
        "simulation_heuristics_breakout_rgb_v1": "Breakout RGB",
        "simulation_heuristics_halfcheetah_v1": "HalfCheetah",
        "simulation_heuristics_vizdoom_d1_v1": "Doom D1",
        "simulation_heuristics_vizdoom_d3_v1": "Doom D3",
        "simulation_heuristics_atari57_v1": "Atari57",
        "simulation_heuristics_montezuma_v1": "Montezuma",
    }
    header = [
        "Rank",
        "Model",
        "Harness",
        "Effort",
        "Average",
        *(labels[task] for task in TASKS),
    ]
    alignment = ["---:", "---", "---", "---", "---:", *(["---:"] * len(TASKS))]
    markdown = [
        "# GenesisBench Learning Beyond Gradients Article Suite",
        "",
        "The aggregate score is the arithmetic mean of nine normalized task "
        "scores. Starter policies map to 0 and trusted article-level references "
        "map to 100.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for row in ranked:
        values = [
            str(row["rank"]),
            row["model"],
            row["harness"],
            row["provider_reasoning_effort"],
            f"{row['average_normalized_score']:.2f}",
            *(f"{row['task_scores'][task]:.2f}" for task in TASKS),
        ]
        markdown.append("| " + " | ".join(values) + " |")
    (args.output.parent / "ARTICLE_SUITE.md").write_text(
        "\n".join(markdown) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
