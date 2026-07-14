# Learning Beyond Gradients task suite

GenesisBench represents each experiment-family row in the article as one
native task package. Together with the original Ant task, the suite contains
nine tasks.

| Task | Article contract | Trusted reproduction |
| --- | --- | --- |
| `simulation_heuristics_ant_v1` | CPG/PD through adaptive residual MPC | Gymnasium mean 5895.932216 with exact source/action parity |
| `simulation_heuristics_pong_ram_v1` | RAM-only Pong, score 21 | 21 on hidden seeds and reset variants |
| `simulation_heuristics_breakout_ram_v1` | `387 → 507 → 839 → 864` | 864 on nominal and shifted starts |
| `simulation_heuristics_breakout_rgb_v1` | `310 → 428 → 864` pixel transfer | 864 on nominal and shifted starts |
| `simulation_heuristics_halfcheetah_v1` | staged-tree MPC, seeds 100–104 | mean 11836.693449819431 |
| `simulation_heuristics_vizdoom_d1_v1` | screen CV + `HEALTH` | mean 0.9440999741666019 |
| `simulation_heuristics_vizdoom_d3_v1` | screen CV + five public variables | returns `[545,475,480,440,690,500,600,595,530,715]` |
| `simulation_heuristics_atari57_v1` | 57 games × 2 modes × 3 searches | complete 342-slot and 6.84B-frame protocol |
| `simulation_heuristics_montezuma_v1` | 400-point macro route plus recovery | 400 in 1,769 steps with recovery variants |

## Common benchmark boundary

Each task follows the original Ant package:

```text
fixed public starter
→ OpenCode research in an isolated sandbox
→ public development evaluator
→ standardized final artifact
→ hidden verifier
→ starter/reference normalized score
```

The task workspace never contains `verifier/`, `oracle/`, or `evidence/`.

## Atari57 exception

Atari57 is one aggregate task rather than 114 standalone tasks. Its artifact
resolves one policy for every `(game, observation mode, repeat index)` tuple:

```text
57 × 2 × 3 = 342 policy/search slots
```

An official result is eligible only when the interaction ledger records all
342 searches at 20,000,000 frames each. Incomplete submissions return a clean
zero and are not presented as article reproductions.

## Model and harness contract

All new leaderboard runs use the BenchFlow `opencode` ACP harness. The canonical
matrix is:

| Model | Exact route | Harness | Provider reasoning setting |
| --- | --- | --- | --- |
| GPT-5.6 Sol | Azure `azure/gpt-5.6-sol` | OpenCode | `max` |
| GPT-5.5 | Azure `azure/gpt-5.5` | OpenCode | `xhigh` |
| Claude Opus 4.8 | Claude OAuth `anthropic/claude-opus-4-8` through the pinned OpenCode plugin | OpenCode | `max` |
| GPT-5.4 Mini | Azure `azure/gpt-5.4-mini` | OpenCode | `xhigh` |

`max` and `xhigh` are provider-specific categorical labels. They indicate the
configured reasoning setting for that route; they are not interchangeable
units and should not be read as a shared numeric inference-compute scale.

OpenCode talks directly to the provider because BenchFlow 0.6.5's
chat-completions gateway cannot faithfully transform Azure GPT-5.6 Sol tool
calls. BenchFlow continues to own task staging, Daytona/Docker isolation, ACP
trajectory capture, timing, and verifier execution.

## Leaderboard outputs

The offline article-suite report contains 10 independent leaderboards in a
fixed order:

1. one leaderboard for each of the nine article-derived tasks;
2. the final cross-task leaderboard.

Each task first receives an unbounded anchor-normalized score:

```text
task score = 100 * (candidate - starter) / (reference - starter)
```

Each model runs five independent trials. Task leaderboards report mean ± sample
standard deviation across those trials. The primary cross-task score is the
RLiable-style 25% trimmed interquartile mean (IQM) over the complete `5 × 9`
score matrix: flatten all 45 normalized scores, remove the lowest 11 and
highest 11, and average the middle 23. The displayed `±` value is the sample
standard deviation of the five per-trial nine-task IQMs.

The repository README intentionally shows only the final leaderboard image.
The nine task panels use each environment's native raw score. The final chart
uses a positive plot index equal to `IQM + 100`, while raw IQM remains the
official ranking metric. Both images live in `leaderboard/ARTICLE_SUITE.md`;
the matching machine-readable structure lives in
`leaderboard/article_suite.json`. See `docs/article-suite-scoring.md` for the
research rationale and statistical limitations.

The runner and resumable leaderboard builder live in:

- `scripts/run_article_suite.py`
- `scripts/build_article_suite_leaderboard.py`
- `experiments/article_suite/`
