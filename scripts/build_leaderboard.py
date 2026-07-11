#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "gpt-5.6-sol",
    "gpt-5.5",
    "claude-opus-4.8",
    "gpt-5.4-mini",
}
APACHE_NOTICE = """# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""


def _sanitize_score_paths(value: Any, *, policy_path: str) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                policy_path
                if key == "policy_path"
                else _sanitize_score_paths(item, policy_path=policy_path)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_score_paths(item, policy_path=policy_path)
            for item in value
        ]
    return value


def _package_policy(source: Path, destination: Path) -> None:
    content = source.read_text()
    if "Licensed under the Apache License, Version 2.0" not in content:
        content = APACHE_NOTICE + content
    destination.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Ant leaderboard.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "ant_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for score_path in args.runs_root.glob("*/*/score.json"):
        run_dir = score_path.parent
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            continue
        score = json.loads(score_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        agent_summary_path = run_dir / "agent_summary.json"
        if agent_summary_path.is_file():
            agent_summary = json.loads(agent_summary_path.read_text())
            if agent_summary.get("status") == "error":
                continue
        if metadata.get("runtime") != "docker":
            continue
        if metadata.get("score_return_code") != 0:
            continue
        if metadata["model"]["id"] not in EXPECTED_MODELS:
            continue
        rows.append(
            {
                "model_id": metadata["model"]["id"],
                "model": metadata["model"]["display_name"],
                "harness": metadata["harness"],
                "score": score["score"],
                "normalized_score": score["normalized_score"],
                "hidden_nominal": score["hidden_nominal"]["mean_return"],
                "hidden_robustness": score["hidden_robustness"]["mean_return"],
                "fall_rate": (
                    0.7 * score["hidden_nominal"]["fall_rate"]
                    + 0.3 * score["hidden_robustness"]["fall_rate"]
                ),
                "budget_minutes": metadata["budget_minutes"],
                "runtime": metadata["runtime"],
                "reasoning": metadata["model"].get(
                    "reasoning_effort",
                    metadata["model"].get("thinking_effort"),
                ),
                "_source_run_dir": run_dir,
                "run_id": str(run_dir.relative_to(args.runs_root)),
                "finished_at": metadata["finished_at"],
            }
        )
    latest_by_model: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["finished_at"]):
        latest_by_model[row["model_id"]] = row
    ranked = sorted(
        latest_by_model.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        submission_dir = args.output.parent / "submissions" / row["model_id"]
        submission_dir.mkdir(parents=True, exist_ok=True)
        source_run_dir = row.pop("_source_run_dir")
        _package_policy(
            source_run_dir / "workspace" / "final_policy" / "policy.py",
            submission_dir / "policy.py",
        )
        packaged_policy = str(
            (submission_dir / "policy.py").relative_to(REPO_ROOT)
        )
        sanitized_score = _sanitize_score_paths(
            json.loads((source_run_dir / "score.json").read_text()),
            policy_path=packaged_policy,
        )
        (submission_dir / "score.json").write_text(
            json.dumps(sanitized_score, indent=2, sort_keys=True) + "\n"
        )
        (submission_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "benchmark": "ant_v1",
                    "model_id": row["model_id"],
                    "model": row["model"],
                    "harness": row["harness"],
                    "reasoning": row["reasoning"],
                    "budget_minutes": row["budget_minutes"],
                    "runtime": row["runtime"],
                    "source_run_id": row["run_id"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        row["submission_path"] = packaged_policy

    payload = {
        "benchmark": "ant_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# GenesisBench Ant v1 Leaderboard",
        "",
        "![GenesisBench Ant v1 leaderboard](ant_v1_leaderboard.png)",
        "",
        "| Rank | Agent model | Harness | Reasoning | Score | Normalized | Nominal | Robust | Fall rate |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        markdown.append(
            "| {rank} | {model} | {harness} | {reasoning} | {score:.2f} | "
            "{normalized_score:.2f} | {hidden_nominal:.2f} | "
            "{hidden_robustness:.2f} | {fall_rate:.1%} |".format(**row)
        )
    (args.output.parent / "README.md").write_text("\n".join(markdown) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
