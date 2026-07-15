---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_halfcheetah_v1
  description: Improve an interpretable HalfCheetah controller under a fixed autonomous research budget.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - halfcheetah
    - mujoco
    - model-predictive-control
metadata:
  category: locomotion
  difficulty: hard
  tags:
    - mujoco
    - continuous-control
    - policy-search
    - model-predictive-control
    - robotics
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
    development:
      episodes: 3
      max_steps: 300
      seeds: [100, 101, 102]
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 10800
  user: agent
  network_mode: public
verifier:
  timeout_sec: 4200
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
# Improve the HalfCheetah locomotion policy

You are given a strong, interpretable programmatic controller for Gymnasium
`HalfCheetah-v5`. Improve it as much as possible through autonomous research,
coding, and experimentation. The policy must remain a code-based controller:
do not train or embed a neural network.

## Objective

Maximize the final HalfCheetah episode return:

```text
x velocity - 0.1 * sum(action ** 2)
```

The environment has:

- a 17-dimensional observation;
- six continuous actions in `[-1, 1]`;
- a maximum of 1,000 steps per episode;
- no health termination in the standard task.

## Starting point

`starter_policy/policy.py` contains the article's non-MPC baseline: a
two-rate asymmetric central pattern generator whose Fourier joint targets are
tracked by a PD controller. The published artifact reported about `4,799.7`
mean return over seeds `100..109` for this controller family.

The reference result adds online staged-tree MPC around that interpretable
gait. The article's five-episode rerun on seeds `100..104` reported:

```text
mean 11836.693, min 11735.0, max 12041.2
```

The trusted oracle reimplements that controller: top-K two-level action-tree
search, a 14-step closed-loop CPG/PD tail, and a swing-amplitude schedule that
changes at steps 300 and 900.

## Development evaluation

Run:

```bash
python evaluate.py --policy final_policy/policy.py
```

For a quick iteration:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 1 \
  --max-steps 100
```

For the full published five-seed reproduction:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --episodes 5 \
  --max-steps 1000 \
  --seed 100
```

The full staged-tree reference is intentionally compute-intensive. Internal
MuJoCo planning rollouts do not count as external episode steps, but they do
consume your wall-clock budget and appear in action-latency metrics.

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

Each action must be a finite NumPy-compatible array with shape `(6,)`. Values
outside `[-1, 1]` are clipped.

For model-based planning, a policy may optionally implement:

```python
def configure_simulator(
    self,
    *,
    model_xml_path: str,
    frame_skip: int,
) -> None: ...
```

The evaluator calls this once per episode with a read-only path to the
episode's MuJoCo model. It does not expose the live environment, reward, or
hidden evaluation configuration.

## Rules

1. Work autonomously. Do not ask for user feedback.
2. You have a fixed wall-clock budget. Use it for iterative improvement.
3. Keep the final controller interpretable and non-neural.
4. Do not modify `evaluate.py` or trusted runtime files.
5. Do not access `/oracle`, `/verifier`, or reconstruct hidden seeds or
   dynamics configuration.
6. Do not copy a completed HalfCheetah solution from the internet.
7. The final policy receives observations, its reset seed, and the optional
   simulator-model configuration hook only.
8. The final policy must run offline with the packages already installed.
9. Before finishing, evaluate `final_policy/policy.py` and leave the best
   working version in place.
