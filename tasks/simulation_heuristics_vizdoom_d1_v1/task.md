---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_vizdoom_d1_v1
  description: Improve a rendered-pixel VizDoom D1 medikit policy under a fixed autonomous research budget.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - vizdoom
    - computer-vision
    - visual-control
metadata:
  category: visual-control
  difficulty: medium
  tags:
    - vizdoom
    - screen-cv
    - programmatic-policy
    - sparse-reward
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 10
      max_steps: 2100
      seeds: [0]
      frame_skip: 1
      render_size: [240, 180]
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
# Improve the VizDoom D1 screen policy

You are given a working but weak programmatic controller for EnvPool
`D1Basic-v1`. Improve it through autonomous coding and repeated development
evaluation.

## Objective

Maximize mean episode reward over ten D1 lanes. The useful behavior is to find
the medikit from pixels, stage near it while health is high, and collect it
after health decay makes the pickup valuable.

The published reference target for the visible seed batch is approximately:

```text
10-seed mean = 0.9441
10-seed min  = 0.2900
```

The task runtime is pinned to the article's declared EnvPool `1.1.1`.

## Observable inputs

At each step your policy receives exactly:

1. an immutable `uint8` rendered frame with shape `(180, 240, 3)`;
2. an immutable mapping containing only public `HEALTH`.

It does not receive reward, environment objects, map data, labels, object
coordinates, the EnvPool base seed, or a vector lane identifier.

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
  --max-steps 200
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

    def act(self, frame, variables) -> int:
        ...
```

The returned action must be an integer in `[0, 5]`:

```text
0 NONE
1 TURN_RIGHT
2 TURN_LEFT
3 FORWARD
4 FORWARD_RIGHT
5 FORWARD_LEFT
```

## Rules

1. Work autonomously and use the fixed wall-clock budget well.
2. Do not modify `evaluate.py`, `_runtime/`, or trusted task files.
3. Do not access `/oracle`, `/verifier`, hidden configs, or hidden seeds.
4. Do not import EnvPool or VizDoom from the policy.
5. Do not read WAD/config files, map geometry, labels, object coordinates,
   automaps, or simulator state.
6. Do not hard-code routes or behavior for the public seed batch.
7. The final controller must use rendered pixels plus public `HEALTH` only.
8. The final policy must run offline with the installed packages.
9. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
