---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_breakout_rgb_v1
  description: Improve a pixel-only Atari Breakout policy toward the article's 864-point RAM-to-RGB transfer result.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - breakout
    - atari
    - computer-vision
metadata:
  category: game-control
  difficulty: hard
  tags:
    - atari
    - discrete-control
    - policy-search
    - rgb-observation
    - computer-vision
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 1
      max_steps: 30000
      seeds: [0]
      observation_mode: rgb
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 5400
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
# Improve the pixel-only Breakout policy

You are given the article's initial 310-point programmatic vision controller
for EnvPool `Breakout-v5`. Improve it through autonomous research, coding, and
experimentation.

## Objective

Maximize the native Breakout game score. The article first developed the
controller structure with RAM, then transferred the state-reading layer to
pixels. The RGB transfer progressed through `310 -> 428 -> 864`; this package
preserves the pixel-only `864` target.

## Observation and action contract

The policy receives only one EnvPool RGB `uint8` tensor with shape
`(3, 210, 160)`. It
does not receive RAM, reward, `info`, lives, emulator state, object labels, or
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

`starter_policy/policy.py` reproduces the article's `310` RGB node. The
intermediate `428` node increases ordinary chase lookahead; the `864` node also
applies stuck-offset release and lag compensation throughout the transfer
policy.

## Development evaluation

Run:

```bash
python evaluate.py --policy starter_policy/policy.py
```

Run the article-scale RGB horizon with:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 1 \
  --max-steps 30000
```

The final evaluator uses unseen seeds and a shifted initial paddle state. A
memorized action tape from the public trajectory is unlikely to transfer.

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
5. The final policy receives pixels only. Reading RAM or emulator internals is
   prohibited.
6. The final policy must run offline with the installed packages.
7. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
