#!/usr/bin/env python3
"""Run BenchFlow with OpenCode's provider-native protocol enabled.

BenchFlow 0.6.5 normally routes OpenCode through a Chat Completions LiteLLM
gateway. Current Azure GPT-5.6 Sol tool use requires OpenCode's native Azure
Responses-API path, so this entrypoint keeps BenchFlow's sandbox, ACP, and
verifier lifecycle while bypassing only that incompatible model gateway.
"""

from __future__ import annotations

from dataclasses import replace

from benchflow.agents.registry import AGENTS, CredentialFile
from benchflow.providers import litellm_runtime


def apply_opencode_direct_provider_mode() -> None:
    """Let OpenCode speak its provider-native protocol without LiteLLM.

    Azure GPT-5.6 Sol currently rejects the function-call transformation made
    by BenchFlow 0.6.5's chat-completions gateway. OpenCode's native Azure
    Responses-API path is verified separately and preserves tool use.
    """

    litellm_runtime._NATIVE_PROTOCOL_AGENTS = frozenset(  # noqa: SLF001
        {*litellm_runtime._NATIVE_PROTOCOL_AGENTS, "opencode"}  # noqa: SLF001
    )


def apply_opencode_claude_oauth_support() -> None:
    """Materialize the provided Claude OAuth token for the OpenCode plugin."""

    config = AGENTS["opencode"]
    credential_path = "{home}/.claude/.credentials.json"
    if any(item.path == credential_path for item in config.credential_files):
        return
    credential = CredentialFile(
        path=credential_path,
        env_source="CLAUDE_CODE_OAUTH_TOKEN",
        template=(
            '{{"claudeAiOauth":{{"accessToken":"{value}",'
            '"refreshToken":"","expiresAt":4102444800000}}}}'
        ),
    )
    AGENTS["opencode"] = replace(
        config,
        credential_files=[*config.credential_files, credential],
    )


def main() -> None:
    apply_opencode_direct_provider_mode()
    apply_opencode_claude_oauth_support()
    from benchflow.cli.main import app

    app()


if __name__ == "__main__":
    main()
