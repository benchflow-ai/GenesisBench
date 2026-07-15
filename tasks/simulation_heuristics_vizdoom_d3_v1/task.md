---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_vizdoom_d3_v1
  description: Improve a screen-CV VizDoom D3 battle policy under a fixed autonomous research budget.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - vizdoom
    - computer-vision
    - first-person-control
metadata:
  category: visual-control
  difficulty: hard
  tags:
    - vizdoom
    - screen-cv
    - programmatic-policy
    - combat
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 10
      max_steps: 1050
      seeds: [0]
      frame_skip: 2
      render_size: [640, 480]
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 5400
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
# Improve the VizDoom D3 screen-CV battle policy

You are given a working but weak programmatic controller for EnvPool
`D3Battle-v1`. Improve it through autonomous coding and repeated development
evaluation.

## Objective

Maximize:

```text
episode reward = DAMAGECOUNT + 10 * KILLCOUNT
```

The controller must close three loops from screen pixels:

1. detect, aim at, and attack visible enemies;
2. seek health or ammunition when resources are low;
3. explore and recover from walls or repeated views when no target is visible.

The visible reference target is:

```text
10-seed mean = 557
10-seed min  = 440
```

The task runtime is pinned to the article's declared EnvPool `1.1.1`.

## Observable inputs

At each step your policy receives exactly:

1. an immutable `uint8` screen frame with shape `(480, 640, 3)`;
2. an immutable mapping containing public `HEALTH`, `AMMO2`, `HITCOUNT`,
   `DAMAGECOUNT`, and `KILLCOUNT`.

It does not receive reward, environment objects, map data, labels, object
coordinates, seeds, or a vector lane identifier.

## Development evaluation

Run:

```bash
python evaluate.py --policy final_policy/policy.py
```

For a shorter smoke test:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 2 \
  --max-steps 100
```

The public evaluator uses batch seed `0`. Final evaluation uses different,
unseen seed batches.

## Required submission

Store the best policy at:

```text
final_policy/policy.py
```

It must define `Policy` or `make_policy`:

```python
class Policy:
    def reset(self) -> None:
        ...

    def act(self, frame, variables):
        ...
```

Return a finite NumPy-compatible array with shape `(8,)`:

```text
[ATTACK, SPEED, FORWARD, BACKWARD, RIGHT, LEFT, TURN180, TURN_DELTA]
```

The first seven channels must be in `[0, 1]`; turn delta must be in
`[-12, 12]`.

## Rules

1. Work autonomously and use the fixed wall-clock budget well.
2. Do not modify `evaluate.py`, `_runtime/`, or trusted task files.
3. Do not access `/oracle`, `/verifier`, hidden configs, or hidden seeds.
4. Do not import EnvPool or VizDoom from the policy.
5. Do not read WAD/config files, map geometry, labels, object coordinates,
   automaps, or simulator state.
6. Do not hard-code routes or behavior for the public seed batch.
7. Use screen CV plus the five allowlisted public variables only.
8. The final policy must run offline with the installed packages.
9. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
