---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_breakout_ram_v1
  description: Improve a RAM-only programmatic Atari Breakout policy toward the 864-point article result.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - breakout
    - atari
    - ram
metadata:
  category: game-control
  difficulty: medium
  tags:
    - atari
    - discrete-control
    - policy-search
    - ram-observation
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 1
      max_steps: 27000
      seeds: [0]
      observation_mode: ram
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 1800
  user: agent
  network_mode: public
verifier:
  timeout_sec: 300
  user: root
  network_mode: no-network
environment:
  build_timeout_sec: 1200
  cpus: 4
  memory_mb: 4096
  storage_mb: 10240
  network_mode: public
  allow_internet: true
  workdir: /app
benchflow:
  document_version: "0.6"
---
# Improve the RAM-only Breakout policy

You are given the article's 387-point programmatic controller for EnvPool
`Breakout-v5`. Improve it through autonomous research, coding, and
experimentation.

## Objective

Maximize the native Breakout game score. The article experiment progressed
through `387 -> 507 -> 839 -> 864`; `864` is the target reproduced by the
packaged reference policy.

## Observation and action contract

The policy receives only a NumPy-compatible `uint8` RAM vector with shape
`(128,)`. It does not receive pixels, reward, `info`, lives, the ALE object, or
hidden evaluator configuration.

Return one integer action:

```text
0 = NOOP
1 = FIRE
2 = RIGHT
3 = LEFT
```

The evaluator runs one ALE frame per policy action with no frame skipping.

## Starting point

`starter_policy/policy.py` reproduces the article's `387` node: it predicts
reflected paddle interceptions but has no stuck-loop perturbation, fast-low-ball
special case, or late-game offset release.

## Development evaluation

Run:

```bash
python evaluate.py --policy starter_policy/policy.py
```

Run the full article-scale horizon with:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 1 \
  --max-steps 108000
```

The public evaluator uses a nominal visible setup. The final evaluator uses
unseen seeds and shifted initial paddle configurations. A step-indexed replay
of one public trajectory is not a robust solution.

## Required submission

Store your best policy at:

```text
final_policy/policy.py
```

It must define either:

```python
class Policy:
    def __init__(self, seed: int = 0): ...
    def reset(self, seed: int = 0) -> None: ...
    def act(self, observation) -> int: ...
```

or `make_policy(seed=0)` returning an object with `reset` and `act`.

## Rules

1. Work autonomously. Do not ask for user feedback.
2. Do not modify `evaluate.py` or trusted runtime files.
3. Do not access `/oracle`, `/verifier`, or hidden configuration.
4. Do not copy a completed Breakout solution from the internet.
5. The final policy receives the 128 RAM bytes only.
6. The final policy must run offline with the installed packages.
7. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
