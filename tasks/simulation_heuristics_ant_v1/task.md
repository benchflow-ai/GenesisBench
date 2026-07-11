---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_ant_v1
  description: Improve a programmatic Ant locomotion policy under a fixed autonomous research budget.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - ant
    - mujoco
    - continuous-control
metadata:
  category: locomotion
  difficulty: medium
  tags:
    - mujoco
    - continuous-control
    - policy-search
    - robotics
  reference_task: true
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 3
      max_steps: 300
      seeds: [0, 1, 2]
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
# Improve the Ant locomotion policy

You are given a working but weak programmatic controller for Gymnasium
`Ant-v5`. Improve the policy as much as possible through autonomous research,
coding, and experimentation.

## Objective

Maximize the final Ant episode return. The return is the native Ant reward:

```text
forward velocity + healthy reward - control cost
```

The environment has:

- a 27-dimensional observation;
- eight continuous actions in `[-1, 1]`;
- a maximum of 1,000 steps per episode;
- early termination when the Ant becomes unhealthy.

## Starting point

`starter_policy/policy.py` contains a working rhythmic CPG/PD controller. You
may modify it, replace it, add training or search code, bundle learned weights,
or implement model-predictive control.

## Development evaluation

Run:

```bash
python evaluate.py --policy starter_policy/policy.py
```

For faster iteration:

```bash
python evaluate.py \
  --policy path/to/policy.py \
  --episodes 1 \
  --max-steps 150
```

The public evaluator uses visible development seeds. The final evaluator uses
unseen seeds and conservative unseen dynamics variants. A policy that memorizes
the development episodes is unlikely to score well.

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
    def act(self, observation) -> numpy.ndarray: ...
```

or:

```python
def make_policy(seed: int = 0): ...
```

The returned object must implement `reset` and `act`.

Each action must be a finite NumPy-compatible array with shape `(8,)`. Values
outside `[-1, 1]` are clipped.

## Rules

1. Work autonomously. Do not ask for user feedback.
2. You have a fixed wall-clock budget. Use it for iterative improvement.
3. Do not modify `evaluate.py` or trusted runtime files.
4. Do not access `/oracle`, `/verifier`, or reconstruct hidden evaluation
   seeds or dynamics files.
5. Do not copy a completed Ant solution from the internet.
6. The final policy receives observations only. It does not receive reward,
   environment `info`, simulator objects, or hidden variant parameters.
7. The final policy must run offline with the packages already installed.
8. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
