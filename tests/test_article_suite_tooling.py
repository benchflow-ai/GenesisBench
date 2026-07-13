from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_article_suite_leaderboard as leaderboard
from scripts import bench_opencode
from scripts import run_article_suite as runner


def test_article_suite_declares_exactly_nine_unique_tasks() -> None:
    assert len(runner.TASKS) == 9
    assert len(set(runner.TASKS)) == 9
    assert runner.TASKS == leaderboard.TASKS
    assert "simulation_heuristics_ant_v1" in runner.TASKS


def test_command_uses_opencode_and_fail_closed_artifacts(tmp_path: Path) -> None:
    model = {
        "id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "model": "azure/gpt-5.6-sol",
        "provider": "azure",
        "provider_reasoning_effort": "max",
    }
    command = runner.build_command(
        model=model,
        jobs_dir=tmp_path / "jobs",
        sandbox="docker",
        concurrency=1,
        artifact_dir=tmp_path,
        tasks=runner.TASKS,
        azure_resource_name="example-resource",
    )

    assert command[command.index("--agent") + 1] == "opencode"
    assert command[command.index("--model") + 1] == "azure/gpt-5.6-sol"
    assert command[command.index("--usage-tracking") + 1] == "off"
    assert command[command.index("--expected-tasks") + 1] == "9"
    assert command.count("--include") == 9
    assert "--reasoning-effort" not in command
    assert "scripts/bench_opencode.py" in command
    assert "AZURE_RESOURCE_NAME=example-resource" in command


def test_benchflow_opencode_shim_uses_native_provider_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bench_opencode.litellm_runtime._NATIVE_PROTOCOL_AGENTS
    monkeypatch.setattr(
        bench_opencode.litellm_runtime,
        "_NATIVE_PROTOCOL_AGENTS",
        original,
    )

    bench_opencode.apply_opencode_direct_provider_mode()

    assert "opencode" in (
        bench_opencode.litellm_runtime._NATIVE_PROTOCOL_AGENTS
    )


def test_benchflow_opencode_shim_writes_claude_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bench_opencode.AGENTS["opencode"]
    monkeypatch.setitem(bench_opencode.AGENTS, "opencode", original)

    bench_opencode.apply_opencode_claude_oauth_support()

    credentials = bench_opencode.AGENTS["opencode"].credential_files
    oauth = [
        item
        for item in credentials
        if item.path == "{home}/.claude/.credentials.json"
    ]
    assert len(oauth) == 1
    assert oauth[0].env_source == "CLAUDE_CODE_OAUTH_TOKEN"
    assert "claudeAiOauth" in oauth[0].template


def test_opencode_config_pins_gpt_5_6_sol_to_max() -> None:
    model = {
        "id": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "model": "azure/gpt-5.6-sol",
        "provider": "azure",
        "provider_reasoning_effort": "max",
    }

    config = runner._opencode_config(model)
    definition = config["provider"]["azure"]["models"]["gpt-5.6-sol"]

    assert definition["options"]["reasoningEffort"] == "max"
    assert definition["variants"]["max"]["reasoningEffort"] == "max"


def test_opencode_config_pins_claude_oauth_plugin() -> None:
    model = {
        "id": "claude-opus-4.8",
        "display_name": "Claude Opus 4.8",
        "model": "anthropic/claude-opus-4-8",
        "provider": "claude_oauth",
        "provider_reasoning_effort": "max",
    }

    config = runner._opencode_config(model)
    definition = config["provider"]["anthropic"]["models"][
        "claude-opus-4-8"
    ]

    assert config["plugin"] == ["opencode-claude-auth@2.0.0"]
    assert definition["variants"]["max"]["thinking"]["budgetTokens"] == 31_999


def test_azure_resource_name_is_derived_without_exposing_endpoint() -> None:
    assert (
        runner._azure_resource_name(
            {
                "AZURE_API_ENDPOINT": (
                    "https://example-resource.openai.azure.com/"
                )
            }
        )
        == "example-resource"
    )


def test_provider_environment_is_least_privilege() -> None:
    shared = {
        "AZURE_API_ENDPOINT": "https://example.openai.azure.com/",
        "AZURE_API_KEY": "azure-secret",
        "ANTHROPIC_API_KEY": "anthropic-or-azure-secret",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
    }
    azure = runner._scoped_provider_env(
        {"provider": "azure"},
        shared,
    )
    claude = runner._scoped_provider_env(
        {"provider": "claude_oauth"},
        shared,
    )

    assert azure["AZURE_API_KEY"] == "azure-secret"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in azure
    assert "ANTHROPIC_API_KEY" not in azure
    assert claude["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-secret"
    assert claude["ANTHROPIC_API_KEY"] == "oauth-plugin"
    assert "AZURE_API_KEY" not in claude
    assert "AZURE_API_ENDPOINT" not in claude


def test_aggregate_loader_requires_every_task(tmp_path: Path) -> None:
    model_root = tmp_path / "gpt-5.6-sol"
    job = model_root / "jobs" / "run"
    job.mkdir(parents=True)
    rows = []
    for task in leaderboard.TASKS[:-1]:
        rows.append(
            {
                "info": {"task_name": task},
                "reward": 1.0,
                "error": None,
            }
        )
    (job / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    with pytest.raises(RuntimeError, match="missing article-suite tasks"):
        leaderboard._load_results(model_root)


def test_aggregate_loader_rejects_task_errors(tmp_path: Path) -> None:
    model_root = tmp_path / "gpt-5.6-sol"
    job = model_root / "jobs" / "run"
    job.mkdir(parents=True)
    rows = [
        {
            "info": {"task_name": task},
            "reward": 1.0,
            "error": {"error": "agent_error"} if index == 0 else None,
        }
        for index, task in enumerate(leaderboard.TASKS)
    ]
    (job / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    with pytest.raises(RuntimeError, match="contains an error"):
        leaderboard._load_results(model_root)

    partial = leaderboard._load_results(
        model_root,
        allow_partial_errors=True,
    )
    assert len(partial) == 8


def test_aggregate_loader_accepts_direct_provider_training_export_warning(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "gpt-5.6-sol"
    job = model_root / "jobs" / "run"
    job.mkdir(parents=True)
    rows = [
        {
            "info": {"task_name": task},
            "reward": 0.5,
            "error": {
                "error": "missing_llm_trajectory",
                "error_chain_str": "not training-ready",
            },
        }
        for task in leaderboard.TASKS
    ]
    (job / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    loaded = leaderboard._load_results(model_root)

    assert len(loaded) == 9


def test_normalized_score_prefers_unbounded_verifier_details(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout"
    verifier = rollout / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "genesis-score.json").write_text(
        json.dumps({"normalized_score": 103.811802037})
    )
    row = {
        "info": {"rollout_dir": str(rollout)},
        "reward": 1.0,
    }

    assert leaderboard._normalized_task_score(row) == 103.811802037


def test_scored_agent_timeout_is_a_valid_leaderboard_result(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "claude-opus-4.8"
    job = model_root / "jobs" / "run"
    job.mkdir(parents=True)
    rows = []
    for index, task in enumerate(leaderboard.TASKS):
        rollout = job / f"task-{index}"
        verifier = rollout / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "genesis-score.json").write_text(
            json.dumps({"normalized_score": 0.0})
        )
        rows.append(
            {
                "info": {
                    "task_name": task,
                    "rollout_dir": str(rollout),
                },
                "reward": 0.0,
                "error": (
                    {"error": "agent_timeout"}
                    if index == 0
                    else {"error": "missing_llm_trajectory"}
                ),
            }
        )
    (job / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )

    loaded = leaderboard._load_results(model_root)

    assert len(loaded) == 9


def test_latest_model_runs_can_resume_across_batches(tmp_path: Path) -> None:
    for batch, model_id, finished_at in (
        ("batch-a", "gpt-5.6-sol", 1.0),
        ("batch-b", "gpt-5.6-sol", 2.0),
        ("batch-c", "gpt-5.5", 3.0),
    ):
        model_root = tmp_path / batch / model_id
        model_root.mkdir(parents=True)
        (model_root / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": {"id": model_id},
                    "return_code": 0,
                    "dry_run": False,
                    "finished_at": finished_at,
                }
            )
        )
    selected = leaderboard._latest_model_runs(tmp_path)

    assert selected["gpt-5.6-sol"] == (
        tmp_path / "batch-b" / "gpt-5.6-sol"
    )
    assert selected["gpt-5.5"] == tmp_path / "batch-c" / "gpt-5.5"


def test_latest_task_results_merges_partial_runs(tmp_path: Path) -> None:
    model_id = "gpt-5.6-sol"
    for index, task in enumerate(leaderboard.TASKS[:2], start=1):
        model_root = tmp_path / f"batch-{index}" / model_id
        job = model_root / "jobs" / "run"
        job.mkdir(parents=True)
        (model_root / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model": {"id": model_id},
                    "return_code": 0,
                    "dry_run": False,
                    "finished_at": float(index),
                    "tasks": [task],
                }
            )
        )
        (model_root / "task_manifest.json").write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task_id": task,
                            "digest": f"sha256:digest-{index}",
                        }
                    ]
                }
            )
        )
        (job / "results.jsonl").write_text(
            json.dumps(
                {
                    "info": {"task_name": task},
                    "reward": index / 10,
                    "error": None,
                }
            )
            + "\n"
        )

    selected = leaderboard._latest_task_results(tmp_path)

    assert set(selected[model_id]) == set(leaderboard.TASKS[:2])
    assert selected[model_id][leaderboard.TASKS[0]][1]["reward"] == 0.1
    assert selected[model_id][leaderboard.TASKS[1]][1]["reward"] == 0.2
    assert selected[model_id][leaderboard.TASKS[1]][3] == "sha256:digest-2"
