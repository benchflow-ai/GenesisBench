---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_montezuma_v1
  description: Improve a native-image Montezuma policy beyond a brittle open-loop replay.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - atari
    - montezuma
    - visual-control
    - long-horizon-planning
metadata:
  category: long-horizon-control
  difficulty: hard
  tags:
    - atari
    - native-image
    - macro-actions
    - recovery
    - policy-search
  reference_task: true
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 1
      max_steps: 2600
      seeds: [10001]
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 1800
  user: agent
  network_mode: public
verifier:
  timeout_sec: 600
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
# Improve a recoverable Montezuma's Revenge policy

You are given a valid but weak programmatic controller for EnvPool
`MontezumaRevenge-v5`. Improve it through autonomous research, coding, and
experimentation using only the native RGB observation.

## Why this is a boundary task

The published experiment behind this task documented two complementary facts:

- an earlier state-graph search reduced the reported key distance from `72` to
  `28`, while the game score remained `0`;
- a later unattended native-image run produced a real `400`-point replay using
  `86` macro-actions and `1769` environment steps, but the route was mostly
  open-loop.

The goal is not merely to paste that timeline. Build a policy that can reproduce
the first `400` points and re-enter its plan after timing shifts or from an
intermediate visual state.

## Development evaluation

Run:

```bash
python evaluate.py --policy final_policy/policy.py
```

Test a fresh-policy handoff after your own policy has reached an intermediate
state:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --bootstrap-steps 512 \
  --initial-noops 4
```

The public evaluator exposes development seeds and perturbations. Final scoring
uses unseen seeds and a private mix of startup no-ops and recovery checkpoints.

## Required submission

Store the best policy at:

```text
final_policy/policy.py
```

It must define `Policy` or `make_policy`:

```python
class Policy:
    def __init__(self, seed: int = 0): ...
    def reset(self, seed: int = 0) -> None: ...
    def act(self, observation) -> int: ...
```

`observation` is one native `uint8` RGB frame with shape `(3, 210, 160)`.
`act` must return one integer Atari action in `[0, 17]`. The policy may keep
recurrent state and load files stored below `final_policy/`.

## Evaluation objective

Each episode is capped at the boundary target of `400` points. Final scoring
combines:

1. direct reproduction from reset;
2. unseen deterministic seeds;
3. controlled no-op timing variants;
4. fresh-policy recovery from hidden intermediate checkpoints.

A copied action counter can receive credit for reproduction but cannot receive
full credit without using the image to identify and recover state.

## Rules

1. Work autonomously within the fixed wall-clock budget.
2. Do not modify `evaluate.py`, `_runtime/`, or other trusted task files.
3. Do not access `/oracle`, `/verifier`, hidden configurations, or hidden
   trajectory data.
4. The policy receives native RGB images only. Do not use RAM, ALE state,
   emulator objects, reward, `info`, lives, room identifiers, or hidden timing.
5. Do not fetch or copy a completed Montezuma solution. Research ideas are
   allowed, but the submitted artifact must be your own implementation.
6. The final policy must run offline with the packages installed in the task
   image.
7. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working artifact in place.
