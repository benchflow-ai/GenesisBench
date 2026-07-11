#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from openhands.sdk import Agent, Conversation, Event, LLM
from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.tool import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one OpenHands agent on a prepared Ant task."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--env-json", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=500)
    return parser.parse_args()


def _event_json(event: Event) -> str:
    try:
        return event.model_dump_json()
    except Exception:
        return json.dumps(
            {
                "type": type(event).__name__,
                "rendered": str(event),
            },
            ensure_ascii=False,
        )


def _create_llm(config: dict[str, Any], env: dict[str, str]) -> LLM:
    common: dict[str, Any] = {
        "model": config["model"],
        "usage_id": config["id"],
        "timeout": 300,
        "num_retries": 5,
        "max_output_tokens": 32768,
        "reasoning_summary": "detailed",
        "caching_prompt": True,
    }
    provider = config["provider"]
    if provider == "azure":
        common.update(
            {
                "api_key": SecretStr(env["AZURE_API_KEY"]),
                "base_url": env["AZURE_API_ENDPOINT"],
                "api_version": env.get("AZURE_API_VERSION", "preview"),
                "reasoning_effort": config.get("reasoning_effort", "xhigh"),
                "extended_thinking_budget": None,
            }
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return LLM(**common)


def _create_agent(
    config: dict[str, Any],
    env: dict[str, str],
    run_dir: Path,
) -> tuple[Agent | ACPAgent, LLM | None]:
    if config["provider"] == "claude_acp":
        claude_config_dir = run_dir / "claude_config"
        claude_config_dir.mkdir(parents=True, exist_ok=True)
        (claude_config_dir / "settings.json").write_text(
            json.dumps(
                {
                    "model": config["model"],
                    "effortLevel": config.get("thinking_effort", "max"),
                    "permissions": {"defaultMode": "bypassPermissions"},
                },
                indent=2,
            )
            + "\n"
        )
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = env["CLAUDE_CODE_OAUTH_TOKEN"]
        os.environ["CLAUDE_CONFIG_DIR"] = str(claude_config_dir)
        os.environ["ANTHROPIC_MODEL"] = config["model"]
        acp_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
        }
        acp_binary = shutil.which("claude-agent-acp")
        acp_command = (
            [acp_binary]
            if acp_binary is not None
            else ["npx", "-y", "@agentclientprotocol/claude-agent-acp"]
        )
        return (
            ACPAgent(
                acp_command=acp_command,
                acp_env=acp_env,
                acp_model=config["model"],
                acp_session_mode="acceptEdits",
                acp_prompt_timeout=1800.0,
            ),
            None,
        )

    llm = _create_llm(config, env)
    tools = [
        Tool(
            name=TerminalTool.name,
            params={"no_change_timeout_seconds": 5},
        ),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
    return Agent(llm=llm, tools=tools), llm


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.model_config.read_text())
    env = json.loads(args.env_json.read_text())
    args.env_json.unlink(missing_ok=True)

    event_path = run_dir / "events.jsonl"
    event_file = event_path.open("a", encoding="utf-8")

    def callback(event: Event) -> None:
        event_file.write(_event_json(event) + "\n")
        event_file.flush()

    agent, llm = _create_agent(config, env, run_dir)
    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        persistence_dir=run_dir / "conversation",
        callbacks=[callback],
        max_iteration_per_run=args.max_iterations,
        visualizer=None,
        delete_on_close=False,
        tags={
            "benchmark": "simulation_heuristics_ant_v1",
            "model": config["id"],
        },
    )

    sys.path.insert(0, str(workspace / "_runtime"))
    from genesisbench.task_document import TaskDocument

    prompt = TaskDocument.from_path(workspace / "task.md").instruction
    started_at = time.time()
    status = "completed"
    error: str | None = None
    try:
        conversation.send_message(prompt)
        conversation.run()
    except Exception as exception:
        status = "error"
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        event_file.close()
        summary = {
            "status": status,
            "error": error,
            "model": config,
            "started_at": started_at,
            "finished_at": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "metrics": llm.metrics.model_dump(mode="json") if llm else None,
        }
        (run_dir / "agent_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
