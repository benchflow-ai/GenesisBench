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

## Experiment protocol

The current leaderboard protocol is defined in `protocol.toml`:

- five independent full-suite trials per model;
- three times the original agent wall-clock timeout for every task;
- task scores reported as mean ± sample standard deviation across trials;
- final score computed as RLiable-style IQM over all `5 × 9 = 45`
  normalized trial-task scores;
- variability shown as the sample standard deviation of the five per-trial
  nine-task IQMs.

The resulting matrix is:

```text
4 models × 5 trials × 9 tasks = 180 model-task runs
```

| Tasks | Previous cap | Current cap |
| --- | ---: | ---: |
| Ant, Pong, both Breakouts, Montezuma, VizDoom D1/D3 | 30 min | 90 min |
| HalfCheetah | 60 min | 180 min |
| Atari57 aggregate search | 7 days | 21 days |

Atari57's cap is exceptional because one task contains 342 accounted search
slots. It can be scheduled separately when operating capacity is limited; the
cap is a maximum, not an expected runtime.

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
task environments. Local Docker runs default to calibrated `linux/amd64`
images, including on Apple Silicon:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --model gpt-5.6-sol
```

Use `--docker-platform` only when a task's verifier has a matching trusted
platform calibration. The current VizDoom reproducibility anchors support
`linux-x86_64`, so uncalibrated `linux-arm64` runs fail closed.

Atari57 requests more CPU, memory, and storage than the current Daytona account
allows. The all-nine commands therefore use the default local Docker sandbox.
Use Daytona only for selected non-Atari tasks.

Run all four models sequentially:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --all-models \
  --trials 5 \
  --batch-id article-suite-v2
```

Runs are resumable at model-trial granularity. Reusing `--batch-id` skips
completed trials. Models may also be scheduled into separate batch directories
for parallel execution. The leaderboard builder chooses one complete
five-trial batch per model and never mixes trials from different batches within
the same model.

Use `--trial` to schedule or repair selected trials:

```bash
uv run python scripts/run_article_suite.py \
  --env-file /path/to/credentials.env \
  --model gpt-5.6-sol \
  --task simulation_heuristics_halfcheetah_v1 \
  --trial 3 \
  --batch-id article-suite-v2 \
  --sandbox daytona
```

The leaderboard builder selects the latest complete per-model batch matching
`protocol.toml`; it never mixes older single-run results or partial batches into
the five-trial leaderboard.

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
to `100`. The final leaderboard flattens the five-by-nine normalized score
matrix, removes the lowest 11 and highest 11 values, and averages the middle
23. Per-task sample standard deviations and the sample standard deviation of
the five trial-level IQMs remain visible diagnostics. The additive display
offset does not affect ranking or score gaps; raw IQM remains available in the
JSON.
