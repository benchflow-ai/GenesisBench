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

| Model | Provider | Effort |
| --- | --- | --- |
| GPT-5.6 Sol | Azure | `max` |
| GPT-5.5 | Azure | `xhigh` |
| Claude Opus 4.8 | Claude OAuth through pinned OpenCode plugin | `max` |
| GPT-5.4 Mini | Azure | `xhigh` |

OpenCode talks directly to the provider because BenchFlow 0.6.5's
chat-completions gateway cannot faithfully transform Azure GPT-5.6 Sol tool
calls. BenchFlow continues to own task staging, Daytona/Docker isolation, ACP
trajectory capture, timing, and verifier execution.

## Aggregate score

The article-suite leaderboard reports every normalized task score and their
unweighted arithmetic mean:

```text
average = sum(nine normalized task scores) / 9
```

The runner and resumable leaderboard builder live in:

- `scripts/run_article_suite.py`
- `scripts/build_article_suite_leaderboard.py`
- `experiments/article_suite/`
