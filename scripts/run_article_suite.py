#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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
OPENCODE_AGENT_ENV = {
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "1",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
}


def _default_env_path() -> Path:
    configured = os.environ.get("GENESISBENCH_ENV_FILE")
    if configured:
        return Path(configured).expanduser()
    return REPO_ROOT / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the nine-task article suite through OpenCode."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model")
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="Run only this task; repeat for multiple tasks. Defaults to all nine.",
    )
    parser.add_argument("--env-file", type=Path, default=_default_env_path())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "leaderboard" / "runs" / "article_suite",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sandbox", choices=("docker", "daytona"), default="docker")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
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


def _models() -> list[dict[str, Any]]:
    payload = tomllib.loads(MODELS_PATH.read_text())
    return list(payload["models"])


def _select_models(model_id: str | None, all_models: bool) -> list[dict[str, Any]]:
    models = _models()
    if all_models:
        return models
    selected = [model for model in models if model["id"] == model_id]
    if not selected:
        choices = ", ".join(model["id"] for model in models)
        raise ValueError(f"Unknown model {model_id!r}; choose one of: {choices}")
    return selected


def _validate_tasks(tasks: tuple[str, ...]) -> None:
    missing = [
        task
        for task in tasks
        if not (REPO_ROOT / "tasks" / task / "task.md").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Article suite task packages are missing: " + ", ".join(missing)
        )


def _require_credentials(model: dict[str, Any], env: dict[str, str]) -> None:
    required = {
        "azure": ("AZURE_API_ENDPOINT", "AZURE_API_KEY"),
        "claude_oauth": ("CLAUDE_CODE_OAUTH_TOKEN",),
    }[model["provider"]]
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing credentials for {model['id']}: {', '.join(missing)}"
        )


def _scoped_provider_env(
    model: dict[str, Any],
    provider_env: dict[str, str],
) -> dict[str, str]:
    scoped = provider_env.copy()
    if model["provider"] == "azure":
        for key in (
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_OAUTH_TOKEN",
        ):
            scoped.pop(key, None)
    elif model["provider"] == "claude_oauth":
        for key in (
            "AZURE_API_ENDPOINT",
            "AZURE_API_KEY",
            "AZURE_API_VERSION",
            "AZURE_RESOURCE",
            "AZURE_RESOURCE_NAME",
            "CLAUDE_OAUTH_TOKEN",
        ):
            scoped.pop(key, None)
        # BenchFlow's provider resolver expects the standard Anthropic slot.
        # The OpenCode plugin intercepts the request and uses the separately
        # materialized OAuth credential, so a non-secret placeholder is enough.
        scoped["ANTHROPIC_API_KEY"] = "oauth-plugin"
    return scoped


def build_command(
    *,
    model: dict[str, Any],
    jobs_dir: Path,
    sandbox: str,
    concurrency: int,
    artifact_dir: Path,
    tasks: tuple[str, ...] = TASKS,
    azure_resource_name: str | None = None,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/bench_opencode.py",
        "eval",
        "run",
        "--tasks-dir",
        "tasks",
        "--agent",
        "opencode",
        "--model",
        model["model"],
        "--sandbox",
        sandbox,
        "--skill-mode",
        "no-skill",
        "--loop-strategy",
        "single-shot",
        "--usage-tracking",
        "off",
        "--concurrency",
        str(concurrency),
        "--build-concurrency",
        "1",
        "--agent-idle-timeout",
        "600",
        "--expected-tasks",
        str(len(tasks)),
        "--jobs-dir",
        str(jobs_dir),
        "--health-summary-out",
        str(artifact_dir / "health.json"),
        "--task-manifest-out",
        str(artifact_dir / "task_manifest.json"),
        "--run-config-out",
        str(artifact_dir / "run_config.json"),
        "--context-root",
        ".",
    ]
    command.extend(
        [
            "--agent-env",
            "OPENCODE_CONFIG_CONTENT="
            + json.dumps(
                _opencode_config(model),
                separators=(",", ":"),
                sort_keys=True,
            ),
        ]
    )
    for key, value in OPENCODE_AGENT_ENV.items():
        command.extend(["--agent-env", f"{key}={value}"])
    if azure_resource_name:
        command.extend(
            ["--agent-env", f"AZURE_RESOURCE_NAME={azure_resource_name}"]
        )
    for task in tasks:
        command.extend(["--include", task])
    return command


def _opencode_config(model: dict[str, Any]) -> dict[str, Any]:
    provider = model["model"].split("/", 1)[0]
    bare_model = model["model"].split("/", 1)[1]
    effort = model["provider_reasoning_effort"]
    definition: dict[str, Any] = {
        "name": model["display_name"],
        "reasoning": True,
    }
    if provider == "azure":
        definition["options"] = {
            "reasoningEffort": effort,
            "reasoningSummary": "auto",
        }
        definition["variants"] = {
            effort: {
                "reasoningEffort": effort,
                "reasoningSummary": "auto",
                "include": ["reasoning.encrypted_content"],
            }
        }
    elif provider == "anthropic":
        budget = 31_999 if effort == "max" else 16_000
        definition["options"] = {
            "thinking": {
                "type": "enabled",
                "budgetTokens": budget,
            }
        }
        definition["variants"] = {
            effort: {
                "thinking": {
                    "type": "enabled",
                    "budgetTokens": budget,
                }
            }
        }
    config: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "models": {
                    bare_model: definition,
                }
            }
        },
    }
    if model["provider"] == "claude_oauth":
        config["plugin"] = ["opencode-claude-auth@2.0.0"]
    return config


def _azure_resource_name(env: dict[str, str]) -> str:
    explicit = env.get("AZURE_RESOURCE_NAME", "").strip()
    if explicit:
        return explicit
    endpoint = env.get("AZURE_API_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("AZURE_API_ENDPOINT is required")
    parsed = urlparse(
        endpoint if "://" in endpoint else f"https://{endpoint}"
    )
    host = (parsed.hostname or "").strip()
    suffix = ".openai.azure.com"
    if not host.endswith(suffix):
        raise RuntimeError(
            "AZURE_API_ENDPOINT must use an *.openai.azure.com host"
        )
    resource = host[: -len(suffix)]
    if not resource:
        raise RuntimeError("Could not derive Azure resource name")
    return resource


def _docker_ready() -> bool:
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    tasks = tuple(args.task or TASKS)
    _validate_tasks(tasks)
    models = _select_models(args.model, args.all_models)
    provider_env = os.environ.copy()
    provider_env.update(_read_env(args.env_file.expanduser()))
    if args.sandbox == "daytona":
        if not provider_env.get("DAYTONA_API_KEY"):
            raise RuntimeError("DAYTONA_API_KEY is required for --sandbox daytona")
        provider_env.setdefault("DAYTONA_TARGET", "us")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_root = args.output_root.resolve() / timestamp
    batch_root.mkdir(parents=True, exist_ok=True)

    if args.sandbox == "docker" and not args.dry_run and not _docker_ready():
        raise RuntimeError(
            "Docker is not ready. Start Docker Desktop before running the "
            "authoritative article-suite evaluation."
        )

    batch_manifest: dict[str, Any] = {
        "suite": "learning_beyond_gradients_article",
        "harness": "opencode",
        "tasks": list(tasks),
        "created_at": datetime.now(UTC).isoformat(),
        "models": [],
    }
    for model in models:
        _require_credentials(model, provider_env)
        model_root = batch_root / model["id"]
        jobs_dir = model_root / "jobs"
        model_root.mkdir(parents=True, exist_ok=True)
        run_env = _scoped_provider_env(model, provider_env)
        azure_resource_name = None
        if model["provider"] == "azure":
            azure_resource_name = _azure_resource_name(run_env)
            run_env["AZURE_RESOURCE_NAME"] = azure_resource_name
        command = build_command(
            model=model,
            jobs_dir=jobs_dir,
            sandbox=args.sandbox,
            concurrency=args.concurrency,
            artifact_dir=model_root,
            tasks=tasks,
            azure_resource_name=azure_resource_name,
        )
        run_env["BENCHFLOW_REASONING_EFFORT"] = model[
            "provider_reasoning_effort"
        ]
        started_at = time.time()
        metadata = {
            "model": model,
            "harness": "opencode",
            "sandbox": args.sandbox,
            "tasks": list(tasks),
            "command": command,
            "provider_env_keys": sorted(
                key
                for key in (
                    "AZURE_API_ENDPOINT",
                    "AZURE_API_KEY",
                    "CLAUDE_CODE_OAUTH_TOKEN",
                )
                if run_env.get(key)
            ),
            "provider_reasoning_effort": model["provider_reasoning_effort"],
            "started_at": started_at,
            "finished_at": None,
            "elapsed_seconds": None,
            "return_code": None,
            "status": "running",
            "dry_run": args.dry_run,
        }
        metadata_path = model_root / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        batch_entry = {
            "id": model["id"],
            "run_metadata": str(metadata_path.relative_to(batch_root)),
            "return_code": None,
            "status": "running",
        }
        batch_manifest["models"].append(batch_entry)
        (batch_root / "batch_manifest.json").write_text(
            json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n"
        )
        return_code = 0
        status = "completed"
        try:
            if args.dry_run:
                print(" ".join(command))
            else:
                return_code = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=run_env,
                    check=False,
                ).returncode
            if return_code != 0:
                status = "failed"
        except KeyboardInterrupt:
            return_code = 130
            status = "interrupted"
            raise
        finally:
            metadata.update(
                {
                    "finished_at": time.time(),
                    "elapsed_seconds": time.time() - started_at,
                    "return_code": return_code,
                    "status": status,
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            batch_entry["return_code"] = return_code
            batch_entry["status"] = status
            (batch_root / "batch_manifest.json").write_text(
                json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n"
            )
        if return_code != 0:
            raise SystemExit(return_code)

    (batch_root / "batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(batch_root)


if __name__ == "__main__":
    main()
