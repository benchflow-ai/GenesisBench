#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_MODULES = {
    "simulation_heuristics_ant_v1": "ant.py",
    "simulation_heuristics_atari57_v1": "atari57.py",
    "simulation_heuristics_breakout_ram_v1": "breakout.py",
    "simulation_heuristics_breakout_rgb_v1": "breakout.py",
    "simulation_heuristics_halfcheetah_v1": "halfcheetah.py",
    "simulation_heuristics_montezuma_v1": "montezuma.py",
    "simulation_heuristics_pong_ram_v1": "pong.py",
    "simulation_heuristics_vizdoom_d1_v1": "vizdoom.py",
    "simulation_heuristics_vizdoom_d3_v1": "vizdoom.py",
}


def _write_repository_validator(frontier_root: Path) -> None:
    path = frontier_root / ".github" / "scripts" / "validate_repository.py"
    path.write_text(
        '''#!/usr/bin/env python3
"""Validate the FrontierPhysics public task layout."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    problems: list[str] = []
    tasks_root = ROOT / "tasks"
    task_names = sorted(
        path.name
        for path in tasks_root.iterdir()
        if path.is_dir() and (path / "task.md").is_file()
    )

    if not task_names:
        problems.append("tasks/ must contain at least one task package")

    for forbidden in ("example_tasks", "tasks-extra"):
        if (ROOT / forbidden).exists():
            problems.append(f"forbidden public task directory exists: {forbidden}/")

    task_files = sorted(
        path
        for path in ROOT.glob("**/task.md")
        if ".venv" not in path.parts
        and ".git" not in path.parts
        and "example-tasks" not in path.relative_to(ROOT).parts
    )
    expected_files = sorted(
        tasks_root / task_name / "task.md" for task_name in task_names
    )
    if task_files != expected_files:
        problems.append(
            "task.md layout mismatch; expected "
            f"{[path.relative_to(ROOT).as_posix() for path in expected_files]}; found "
            f"{[path.relative_to(ROOT).as_posix() for path in task_files]}"
        )

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1

    print("OK: public task layout matches repository policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    )


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _private_task_id(source_task: str) -> str:
    return source_task.replace("_", "-")


def _private_dockerfile(source_task: str, private_task: str) -> str:
    text = (
        REPO_ROOT
        / "tasks"
        / source_task
        / "environment"
        / "Dockerfile"
    ).read_text()
    replacements = {
        "COPY security/restrict_exec.c /tmp/restrict_exec.c": (
            "COPY restrict_exec.c /tmp/restrict_exec.c"
        ),
        "COPY src/genesisbench /opt/genesisbench/genesisbench": (
            "COPY genesisbench /opt/genesisbench/genesisbench"
        ),
        f"COPY tasks/{source_task}/evaluate.py /app/evaluate.py": (
            "COPY evaluate.py /app/evaluate.py"
        ),
        f"COPY tasks/{source_task}/starter_policy /app/starter_policy": (
            "COPY starter_policy /app/starter_policy"
        ),
        f"COPY tasks/{source_task}/starter_artifact /app/starter_artifact": (
            "COPY starter_artifact /app/starter_artifact"
        ),
        f"COPY tasks/{source_task}/task_context /app/task_context": (
            "COPY task_context /app/task_context"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if f"tasks/{source_task}" in text:
        raise RuntimeError(
            f"unconverted source path remains in {private_task} Dockerfile"
        )
    return text


def _private_test_script(
    *,
    artifact: str,
    timeout_seconds: int,
) -> str:
    submission = (
        "/app/final_artifact"
        if artifact == "/app/final_artifact"
        else "/app/final_policy"
    )
    log_name = "final_artifact" if artifact.endswith("artifact") else "final_policy"
    return f"""#!/bin/bash
set -u

mkdir -p /logs/verifier
artifact={artifact}
submission={submission}
evaluator_log=/logs/verifier/evaluator-output.txt
integrity_report=/logs/verifier/integrity.json

if [ -e "$artifact" ]; then
  set +e
  python3 -m genesisbench.integrity \\
    --submission "$submission" \\
    --trajectory /logs/agent/acp_trajectory.jsonl \\
    --config /verifier/integrity.json \\
    --output "$integrity_report" \\
    > /logs/verifier/integrity-output.txt 2>&1
  integrity_status=$?
  set -e
else
  integrity_status=2
  printf 'missing submission artifact: %s\\n' "$artifact" \\
    > /logs/verifier/integrity-output.txt
fi

cat /logs/verifier/integrity-output.txt

if [ "$integrity_status" -ne 0 ]; then
  if [ -e "$submission" ]; then
    cp -R "$submission" /logs/verifier/{log_name}
  fi
  printf '0\\n' > /logs/verifier/reward.txt
  python3 - <<'PY_INTEGRITY'
import json
from pathlib import Path

Path("/logs/verifier/reward.json").write_text(
    json.dumps({{"reward": 0}}, indent=2) + "\\n"
)
PY_INTEGRITY
  exit 0
fi

export GENESISBENCH_POLICY_ISOLATION=required
set +e
timeout --signal=TERM --kill-after=30s {timeout_seconds}s \\
  python3 /verifier/evaluate_hidden.py \\
    "$artifact" \\
    --config /verifier/config.toml \\
    --anchors /verifier/anchors.json \\
    --output /logs/verifier/genesis-score.json \\
    > "$evaluator_log" 2>&1
evaluation_status=$?
set -e

cat "$evaluator_log"
printf '%s\\n' "$evaluation_status" > /logs/verifier/evaluation-status.txt

if [ "$evaluation_status" -ne 0 ] && [ ! -f /logs/verifier/genesis-score.json ]; then
  EVALUATION_STATUS="$evaluation_status" python3 - <<'PY_EVALUATION'
import json
import os
from pathlib import Path

Path("/logs/verifier/genesis-score.json").write_text(
    json.dumps(
        {{
            "score": None,
            "normalized_score": None,
            "evaluation_status": int(os.environ["EVALUATION_STATUS"]),
            "verifier_timeout": int(os.environ["EVALUATION_STATUS"]) in {{124, 137}},
        }},
        indent=2,
        sort_keys=True,
    )
    + "\\n"
)
PY_EVALUATION
fi

if [ -e "$submission" ]; then
  cp -R "$submission" /logs/verifier/{log_name}
fi

set +e
python3 -m pytest \\
  -p no:cacheprovider \\
  --ctrf /logs/verifier/ctrf.json \\
  /verifier/test_security.py \\
  /verifier/test_outputs.py \\
  -rA -v > /logs/verifier/output.txt 2>&1
RC=$?
set -e

cat /logs/verifier/output.txt

if [ "$RC" -eq 0 ]; then
  reward=1
else
  reward=0
fi
printf '%s\\n' "$reward" > /logs/verifier/reward.txt
REWARD="$reward" python3 - <<'PY_REWARD'
import json
import os
from pathlib import Path

reward = int(os.environ["REWARD"])
Path("/logs/verifier/reward.json").write_text(
    json.dumps({{"reward": reward}}, indent=2) + "\\n"
)
PY_REWARD

exit 0
"""


def _write_security_test(verifier: Path) -> None:
    (verifier / "test_security.py").write_text(
        '''"""Adversarial checks for the shared isolated policy worker."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/genesisbench")

from genesisbench.policy_isolation import IsolatedPolicy


def test_policy_worker_cannot_read_or_write_verifier_paths(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.py"
    policy_path.write_text(
        """
from pathlib import Path


class Policy:
    def reset(self, seed=0):
        del seed

    def act(self, observation):
        del observation
        breaches = 0
        try:
            Path("/verifier/config.toml").read_text()
            breaches += 1
        except OSError:
            pass
        try:
            Path("/logs/verifier/reward.txt").write_text("1\\\\n")
            breaches += 1
        except OSError:
            pass
        return breaches
"""
    )
    policy = IsolatedPolicy(policy_path=policy_path, init_kwargs={"seed": 0})
    try:
        policy.reset(seed=0)
        assert policy.act(np.zeros(4, dtype=np.float64)) == 0
    finally:
        policy.close()
'''
    )


def _add_integrity_test(path: Path) -> None:
    text = path.read_text()
    if "test_reward_integrity_passed" not in text:
        marker = 'EVALUATION_STATUS_PATH = Path("/logs/verifier/evaluation-status.txt")\n'
        addition = marker + 'INTEGRITY_PATH = Path("/logs/verifier/integrity.json")\n'
        if marker not in text:
            raise RuntimeError(f"unexpected test_outputs.py layout: {path}")
        text = text.replace(marker, addition, 1)
        insertion = """

def test_reward_integrity_passed() -> None:
    integrity = json.loads(INTEGRITY_PATH.read_text())
    assert integrity["passed"] is True, integrity["violations"]
"""
        fixture_marker = "\n\n@pytest.fixture(scope=\"module\")\ndef threshold()"
        if fixture_marker not in text:
            raise RuntimeError(f"fixture marker is missing: {path}")
        text = text.replace(fixture_marker, insertion + fixture_marker, 1)
    if "test_threshold_requires_real_improvement" not in text:
        text += """


def test_threshold_requires_real_improvement(
    threshold: dict[str, Any],
) -> None:
    required = float(threshold["value"])
    starter = float(threshold["starter_score"])
    reference = float(threshold["reference_score"])
    tolerance = float(threshold.get("absolute_tolerance", 0.0))
    assert required > starter + tolerance, (
        f"acceptance threshold {required:.12g} must exceed starter "
        f"{starter:.12g}"
    )
    assert required <= reference + tolerance, (
        f"acceptance threshold {required:.12g} exceeds reference "
        f"{reference:.12g}"
    )
"""
    path.write_text(text)


def sync_task(
    *,
    frontier_root: Path,
    source_task: str,
    timeout_seconds: int,
) -> Path:
    if source_task not in TASK_MODULES:
        raise ValueError(f"unsupported task: {source_task}")
    _write_repository_validator(frontier_root)
    private_task = _private_task_id(source_task)
    source = REPO_ROOT / "tasks" / source_task
    destination = frontier_root / "tasks" / private_task
    if not destination.is_dir():
        raise FileNotFoundError(destination)

    environment = destination / "environment"
    _copy_tree(source / "task_context", environment / "task_context")

    runtime = environment / "genesisbench"
    runtime.mkdir(parents=True, exist_ok=True)
    for filename in (
        TASK_MODULES[source_task],
        "integrity.py",
        "policy_isolation.py",
    ):
        shutil.copy2(REPO_ROOT / "src" / "genesisbench" / filename, runtime / filename)
    shutil.copy2(REPO_ROOT / "security" / "restrict_exec.c", environment / "restrict_exec.c")
    (environment / "Dockerfile").write_text(
        _private_dockerfile(source_task, private_task)
    )

    verifier = destination / "verifier"
    shutil.copy2(source / "verifier" / "integrity.json", verifier / "integrity.json")
    shutil.copy2(source / "verifier" / "verifier.md", verifier / "verifier.md")
    if (source / "verifier" / "rubrics").is_dir():
        _copy_tree(source / "verifier" / "rubrics", verifier / "rubrics")
    _write_security_test(verifier)
    artifact = (
        "/app/final_artifact"
        if "atari57" in source_task
        else "/app/final_policy/policy.py"
    )
    (verifier / "test.sh").write_text(
        _private_test_script(
            artifact=artifact,
            timeout_seconds=timeout_seconds,
        )
    )
    (verifier / "test.sh").chmod(0o755)
    _add_integrity_test(verifier / "test_outputs.py")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-root", type=Path, required=True)
    parser.add_argument("--source-task", choices=sorted(TASK_MODULES), required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = sync_task(
        frontier_root=args.frontier_root.resolve(),
        source_task=args.source_task,
        timeout_seconds=args.timeout_seconds,
    )
    print(destination)


if __name__ == "__main__":
    main()
