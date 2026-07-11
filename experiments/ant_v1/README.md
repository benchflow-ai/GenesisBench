# Ant v1 OpenHands sweep

This experiment runs the same GenesisBench Ant policy-improvement task with
four model configurations through OpenHands.

## Models

| ID | Provider route | Reasoning |
| --- | --- | --- |
| `gpt-5.6-sol` | Azure `gpt-5.6-sol` | `xhigh` |
| `gpt-5.5` | Azure `gpt-5.5` | `xhigh` |
| `claude-opus-4.8` | OpenHands `ACPAgent` + Claude Agent ACP | `max` |
| `gpt-5.4-mini` | Azure `gpt-5.4-mini` | `xhigh` |

The three GPT models use the standard OpenHands SDK agent loop. Opus 4.8 uses
OpenHands' supported `ACPAgent` integration because the available credential is
a Claude subscription credential rather than an Anthropic API key. Both paths
run inside an OpenHands `Conversation` and receive isolated copies of the same
task.

## Run one model

```bash
uv sync --extra dev
uv run python scripts/run_ant_experiment.py \
  --model gpt-5.6-sol \
  --minutes 30 \
  --max-iterations 500
```

The authoritative runtime is Docker. Build it once:

```bash
sh scripts/build_ant_runner_image.sh
```

`--runtime local` is available only for harness development. Local runs are
not leaderboard-valid because a local terminal could traverse outside the task
directory.

Copy the credential template and fill only the providers you plan to run:

```bash
cp .env.example .env
```

Alternatively, pass another file:

```bash
uv run python scripts/run_ant_experiment.py \
  --model gpt-5.6-sol \
  --env-file /secure/path/provider.env
```

Credentials are written only to a mode-`0600` temporary run file, removed
after the agent exits, and never copied into the task workspace.

Docker defaults to the host architecture. Override it when needed:

```bash
GENESISBENCH_DOCKER_PLATFORM=linux/amd64 \
  sh scripts/build_ant_runner_image.sh
```

## Run artifacts

Each run stores:

```text
leaderboard/runs/<timestamp>/<model>/
  workspace/             # isolated public task and submitted final policy
  events.jsonl           # OpenHands event stream
  conversation/          # OpenHands persisted conversation
  model_config.json      # non-secret model configuration
  agent_summary.json     # usage and timing when the agent exits normally
  run_metadata.json      # budget, timeout, process, and scoring metadata
  score.json             # authoritative hidden evaluation
```

The model never receives `tasks/ant_v1/verifier/`. Scoring happens after the
agent exits, in the GenesisBench repository's clean evaluator.

`leaderboard/runs/` is gitignored. Durable public artifacts are the sanitized
policies, score files, metadata, chart, and report under `leaderboard/`.

## Rebuild the leaderboard

```bash
uv run python scripts/build_leaderboard.py
```

This selects the latest completed score for each model and writes:

```text
leaderboard/ant_v1.json
leaderboard/README.md
```

## Fairness notes

- All agents receive the same task files and starter policy.
- Authoritative runs execute in a container that mounts the public task but not
  the hidden verifier.
- All agents receive the same wall-clock budget.
- GPT reasoning effort is `xhigh`, the highest effort exposed by the installed
  OpenHands SDK for those routes.
- Opus effort is `max`, configured in its isolated Claude settings.
- Public development seeds are visible.
- Final nominal seeds and dynamics configurations are not copied to the agent.
- The leaderboard score comes only from `evaluate_hidden.py`, never from the
  agent's self-reported development results.
