# Learning Beyond Gradients article suite

This experiment evaluates the four canonical GenesisBench model routes across
the nine task packages derived from the article:

1. `simulation_heuristics_ant_v1`
2. `simulation_heuristics_pong_ram_v1`
3. `simulation_heuristics_breakout_ram_v1`
4. `simulation_heuristics_breakout_rgb_v1`
5. `simulation_heuristics_halfcheetah_v1`
6. `simulation_heuristics_vizdoom_d1_v1`
7. `simulation_heuristics_vizdoom_d3_v1`
8. `simulation_heuristics_atari57_v1`
9. `simulation_heuristics_montezuma_v1`

Every run uses BenchFlow's registered `opencode` ACP harness. OpenHands is not
part of this suite.

## Inference settings

| Model | Exact route | Provider-specific reasoning setting |
| --- | --- | --- |
| GPT-5.6 Sol | `azure/gpt-5.6-sol` | `max` |
| GPT-5.5 | `azure/gpt-5.5` | `xhigh` |
| Claude Opus 4.8 | `anthropic/claude-opus-4-8` via Claude OAuth and the pinned OpenCode plugin | `max` |
| GPT-5.4 Mini | `azure/gpt-5.4-mini` | `xhigh` |

The setting names are categorical labels exposed by each provider integration,
not a common numeric measure of inference compute. All four routes use the
OpenCode harness; the route and reasoning setting are stored with every
published result.

## Credentials

Credential values are read from the process environment or an env file. They
are never copied into a task workspace or committed artifact.

The Azure routes require:

```text
AZURE_API_ENDPOINT
AZURE_API_KEY
```

The Claude Opus route uses the pinned OpenCode Claude-auth plugin and requires:

```text
CLAUDE_CODE_OAUTH_TOKEN
```

## Run

Docker must be running because the authoritative suite uses isolated BenchFlow
task environments:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --model gpt-5.6-sol
```

Atari57 requests more CPU, memory, and storage than the current Daytona account
allows. The all-nine commands therefore use the default local Docker sandbox.
Use Daytona only for selected non-Atari tasks.

Run all four models sequentially:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --all-models
```

Long runs are resumable at task granularity. For example:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --model gpt-5.6-sol \
  --task simulation_heuristics_halfcheetah_v1 \
  --sandbox daytona
```

The leaderboard builder selects the latest successful result for every
model/task pair across all run batches.

GPT-5.6 Sol is routed directly through OpenCode's Azure Responses-API provider
with reasoning effort `max`. BenchFlow still owns sandboxing, task staging,
ACP trajectory capture, and verifier execution. Trusted LiteLLM usage tracking
is disabled for this suite because BenchFlow 0.6.5's chat-completions gateway
cannot faithfully transform GPT-5.6 Sol tool calls; the run metadata records
this limitation explicitly.

## Build all 10 leaderboards

```bash
uv run python scripts/build_article_suite_leaderboard.py
```

The builder writes:

- `leaderboard/ARTICLE_SUITE.md`: one nine-panel task image followed by the
  final IQM leaderboard image;
- `leaderboard/article_suite.json`: the same 10 leaderboards plus the
  model-centric score records used for reproducibility.
- `leaderboard/article_suite_task_leaderboards.png`: nine task-specific
  leaderboard panels using native raw environment scores;
- `leaderboard/article_suite_final_leaderboard.png`: the final cross-task
  ranking displayed as `IQM + 100` so every current plotted value is positive.

Each task maps its starter policy to `0` and its trusted article-level reference
to `100`. The final primary aggregate is the interquartile mean: remove the two
lowest and two highest of the nine normalized scores, then average the middle
five. Arithmetic mean and median remain secondary diagnostics. The top-level
repository README shows only the final image. The additive display offset does
not affect ranking or score gaps; raw IQM remains available in the JSON.
