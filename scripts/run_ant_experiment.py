#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = REPO_ROOT / "experiments" / "ant_v1" / "models.toml"
DOCKER_IMAGE = os.environ.get(
    "GENESISBENCH_DOCKER_IMAGE",
    "genesisbench-ant-runner:latest",
)


def _default_docker_platform() -> str:
    configured = os.environ.get("GENESISBENCH_DOCKER_PLATFORM")
    if configured:
        return configured
    machine = platform.machine().lower()
    return "linux/arm64" if machine in {"arm64", "aarch64"} else "linux/amd64"


def _default_env_path() -> Path:
    return Path(
        os.environ.get("GENESISBENCH_ENV_FILE", REPO_ROOT / ".env")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and score one OpenHands Ant experiment."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path, default=_default_env_path())
    parser.add_argument(
        "--docker-platform",
        default=_default_docker_platform(),
    )
    parser.add_argument(
        "--openhands-python",
        type=Path,
        default=None,
        help="Python from an OpenHands SDK installation for --runtime local.",
    )
    parser.add_argument(
        "--runtime",
        choices=("docker", "local"),
        default="docker",
    )
    return parser.parse_args()


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        result[key] = value
    return result


def _provider_environment(path: Path) -> dict[str, str]:
    result = {}
    if path.is_file():
        result.update(_read_env(path))
    result.update(os.environ)
    return result


def _require_provider_keys(
    model: dict[str, Any],
    provider_env: dict[str, str],
) -> None:
    required = {
        "azure": ("AZURE_API_ENDPOINT", "AZURE_API_KEY"),
        "claude_acp": ("CLAUDE_CODE_OAUTH_TOKEN",),
    }[model["provider"]]
    missing = [key for key in required if not provider_env.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing credentials for {model['id']}: {joined}. "
            "Set environment variables or pass --env-file."
        )


def _resolve_openhands_python(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    configured = os.environ.get("GENESISBENCH_OPENHANDS_PYTHON")
    if configured:
        return _resolve_openhands_python(Path(configured))
    executable = shutil.which("openhands")
    if executable:
        first_line = Path(executable).read_text().splitlines()[0]
        if first_line.startswith("#!"):
            candidate = Path(first_line[2:])
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "Unable to locate an OpenHands Python runtime. Set "
        "GENESISBENCH_OPENHANDS_PYTHON or pass --openhands-python."
    )


def _load_model(model_id: str) -> dict[str, Any]:
    config = tomllib.loads(MODELS_PATH.read_text())
    for model in config["models"]:
        if model["id"] == model_id:
            return model
    choices = ", ".join(model["id"] for model in config["models"])
    raise ValueError(f"Unknown model {model_id!r}; choose one of: {choices}")


def _write_timer(workspace: Path, deadline: float) -> None:
    timer = workspace / "timer.sh"
    timer.write_text(
        "#!/bin/sh\n"
        "python - <<'PY'\n"
        "import time\n"
        f"remaining = max(0, int({deadline!r} - time.time()))\n"
        "print(f'remaining_seconds={remaining}')\n"
        "PY\n"
    )
    timer.chmod(0o755)


def _terminate(
    process: subprocess.Popen[str],
    *,
    container_name: str | None = None,
) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if container_name is not None:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> None:
    args = parse_args()
    model = _load_model(args.model)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else REPO_ROOT / "leaderboard" / "runs" / timestamp
    )
    run_dir = output_root / model["id"]
    workspace = run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "prepare_task.py"),
            "ant_v1",
            str(workspace),
            "--force",
        ],
        check=True,
    )
    deadline = time.time() + args.minutes * 60
    _write_timer(workspace, deadline)

    model_config_path = run_dir / "model_config.json"
    model_config_path.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n"
    )
    provider_env = _provider_environment(args.env_file.expanduser())
    _require_provider_keys(model, provider_env)
    safe_env = {
        key: provider_env[key]
        for key in (
            "AZURE_API_ENDPOINT",
            "AZURE_API_KEY",
            "AZURE_API_VERSION",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_REGION",
            "CLAUDE_CODE_OAUTH_TOKEN",
        )
        if key in provider_env
    }
    env_json_path = run_dir / ".provider_env.json"
    env_json_path.write_text(json.dumps(safe_env))
    env_json_path.chmod(0o600)

    child_env = os.environ.copy()
    child_env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    container_name: str | None = None
    if args.runtime == "docker":
        image_check = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if image_check.returncode != 0:
            raise RuntimeError(
                "Missing Docker image. Run scripts/build_ant_runner_image.sh"
            )
        container_name = (
            "genesisbench-ant-"
            + model["id"].replace(".", "-")
            + "-"
            + timestamp.lower()
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            args.docker_platform,
            "--name",
            container_name,
            "--volume",
            f"{workspace}:/workspace",
            "--volume",
            f"{run_dir}:/artifacts",
            "--workdir",
            "/workspace",
            "--env",
            "OPENHANDS_SUPPRESS_BANNER=1",
            DOCKER_IMAGE,
            "--workspace",
            "/workspace",
            "--run-dir",
            "/artifacts",
            "--model-config",
            "/artifacts/model_config.json",
            "--env-json",
            "/artifacts/.provider_env.json",
            "--max-iterations",
            str(args.max_iterations),
        ]
    else:
        openhands_python = _resolve_openhands_python(args.openhands_python)
        child_env["PATH"] = (
            str(Path(sys.executable).parent)
            + os.pathsep
            + child_env.get("PATH", "")
        )
        command = [
            str(openhands_python),
            str(REPO_ROOT / "scripts" / "run_openhands_agent.py"),
            "--workspace",
            str(workspace),
            "--run-dir",
            str(run_dir),
            "--model-config",
            str(model_config_path),
            "--env-json",
            str(env_json_path),
            "--max-iterations",
            str(args.max_iterations),
        ]
    started_at = time.time()
    timed_out = False
    with (run_dir / "agent_stdout.log").open("w") as stdout, (
        run_dir / "agent_stderr.log"
    ).open("w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=child_env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=args.minutes * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process, container_name=container_name)
            return_code = process.returncode

    env_json_path.unlink(missing_ok=True)
    policy_path = workspace / "final_policy" / "policy.py"
    score_path = run_dir / "score.json"
    score_process = subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "tasks"
                / "ant_v1"
                / "verifier"
                / "evaluate_hidden.py"
            ),
            str(policy_path),
            "--output",
            str(score_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    (run_dir / "score_stdout.log").write_text(score_process.stdout)
    (run_dir / "score_stderr.log").write_text(score_process.stderr)

    metadata = {
        "benchmark": "ant_v1",
        "harness": (
            "OpenHands SDK ACPAgent"
            if model["provider"] == "claude_acp"
            else "OpenHands SDK"
        ),
        "openhands_python": (
            str(openhands_python) if args.runtime == "local" else None
        ),
        "runtime": args.runtime,
        "docker_image": DOCKER_IMAGE if args.runtime == "docker" else None,
        "docker_platform": (
            args.docker_platform if args.runtime == "docker" else None
        ),
        "model": model,
        "budget_minutes": args.minutes,
        "max_iterations": args.max_iterations,
        "started_at": started_at,
        "finished_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "timed_out": timed_out,
        "agent_return_code": return_code,
        "score_return_code": score_process.returncode,
        "score_file": str(score_path),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(run_dir)
    if score_process.returncode != 0:
        print(score_process.stderr, file=sys.stderr)
        raise SystemExit(score_process.returncode)


if __name__ == "__main__":
    main()
