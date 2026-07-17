---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_pong_ram_v1
  description: Improve a programmatic Atari Pong controller that acts from the current 128-byte RAM state.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - atari
    - pong
    - ram
metadata:
  category: game-control
  difficulty: medium
  tags:
    - atari
    - discrete-control
    - policy-search
    - programmatic-control
    - ram
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 3
      max_steps: 27000
      seeds: [0, 1, 2]
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 5400
  user: agent
  network_mode: no-network
verifier:
  timeout_sec: 600
  user: root
  network_mode: no-network
  hardening:
    cleanup_conftests: true
environment:
  build_timeout_sec: 1200
  cpus: 4
  memory_mb: 4096
  storage_mb: 10240
  network_mode: no-network
  allow_internet: false
  workdir: /app
benchflow:
  document_version: "0.6"
---
# Improve the Atari Pong RAM controller

You are given a working but weak programmatic controller for EnvPool
`Pong-v5`. Improve it through autonomous coding and experimentation.

## Objective

Maximize the native Pong episode score:

```text
points won - points lost
```

An episode ends when either side reaches 21 points. The target score is the
perfect result:

```text
+21
```

The evaluator uses the article experiment's Atari setup:

- current 128-byte RAM state as the policy observation;
- minimal Pong discrete action space;
- frame skip `1`;
- unclipped rewards;
- no sticky actions;
- automatic fire reset;
- no episodic-life wrapper.

## Starting point

`starter_policy/policy.py` contains a late reactive paddle tracker. It decodes
the ball and controlled paddle from RAM, but begins chasing too late and does
not predict wall bounces or shape outgoing returns.

You may replace it, tune it, add search scripts, or build a more capable
stateful controller. The submitted policy must remain a directly inspectable
programmatic controller; do not download or bundle a pretrained neural
network.

## Development evaluation

Run:

```bash
python evaluate.py --policy starter_policy/policy.py
```

Evaluate your current submission:

```bash
python evaluate.py --policy final_policy/policy.py
```

For a quick smoke test:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 1 \
  --max-steps 4000
```

The public evaluator uses visible development seeds. The final evaluator uses
unseen seeds and multiple random no-op reset windows while keeping the article
environment semantics fixed. A controller that depends on one memorized serve
timing is unlikely to score well.

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

or:

```python
def make_policy(seed: int = 0): ...
```

The returned object must implement `reset` and `act`.

`observation` is a NumPy-compatible vector with shape `(128,)` and dtype
`uint8`. `act` must return one finite integer in the environment's minimal
discrete action space. The standard controller actions are `0` for no-op, `2`
for paddle up, and `3` for paddle down.

## Rules

1. Work autonomously. Do not ask for user feedback.
2. Use the fixed wall-clock budget for iterative improvement.
3. Do not modify `evaluate.py` or trusted runtime files.
4. Do not access `/oracle`, `/verifier`, or reconstruct hidden seeds/config.
5. Do not fetch or copy a completed Pong solution or upstream policy.
6. The policy receives RAM only. It does not receive reward, `info`, emulator
   objects, screenshots, or hidden evaluator parameters.
7. The policy may keep recurrent state and load files stored below
   `final_policy/`.
8. The final policy must run offline with packages installed in the task
   environment.
9. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
