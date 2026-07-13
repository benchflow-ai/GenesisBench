# Simulation Heuristics Ant v1 Design Rationale

This document explains why `tasks/simulation_heuristics_ant_v1/` has its current benchmark shape.
The runnable contract and contribution instructions live in
`tasks/simulation_heuristics_ant_v1/README.md` and `tasks/README.md`.

## Research question

Can a coding agent autonomously improve a robot-control policy by repeatedly:

1. reading environment and policy code;
2. running simulator experiments;
3. editing controller or search code;
4. comparing measured returns;
5. submitting the best final policy?

This is the robotics analogue of PostTrainBench's fixed-starting-model,
bounded-research, independently-evaluated workflow.

## Source experiment

The task is motivated by the Ant experiment in
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/).
That experiment evolved an interpretable controller through:

- a rhythmic CPG/PD gait;
- torso orientation feedback;
- higher gait harmonics;
- short-horizon residual MPC;
- warm-started planning and adaptive gait timing.

The public artifacts report a five-episode mean near `6005` in EnvPool's
`Ant-v5`: mean `6005.521`, minimum `5776.805`, and maximum `6146.208`.

GenesisBench keeps the public weak CPG/PD starter so agents still face an
authentic policy-improvement problem. The trusted oracle and normalization
anchor now reproduce the article's final controller instead of using an
unrelated stronger rhythmic policy.

That reference combines:

- speed-adaptive phase increment and stance duty;
- higher-harmonic CPG joint targets;
- posture and heading feedback;
- a 10-step copied-MuJoCo planning horizon;
- 96 residual-action candidates per external step;
- temporal residual smoothing and warm-start plan decay;
- forward, control, posture, yaw, height, health, and terminal-velocity
  objective terms.

The source used EnvPool `1.1.1`; GenesisBench uses Gymnasium. The supplied
`ant_envpool.xml` and Gymnasium's Ant XML parse to the same MuJoCo model, but
the two runtimes use different reset random-number streams. Machine-readable
source hashes and both result families live in
`tasks/simulation_heuristics_ant_v1/evidence/source_provenance.json`.

On the development host, both the imported source policy and the GenesisBench
adaptation produced the same Gymnasium seed-`0..4` returns: mean
`5895.932216`, minimum `5791.444245`, and maximum `6131.400491`. A separate
50-step action-parity probe had maximum absolute difference `0.0`.

## Benchmark translation

| PostTrainBench concept | GenesisBench Simulation Heuristics Ant v1 |
| --- | --- |
| Base model | Starter CPG/PD controller |
| Training research | Controller/search/training research |
| Public `evaluate.py` | Development simulator feedback |
| `final_model/` | `final_policy/` |
| Held-out benchmark score | Hidden Ant suite return |
| Evaluation-tampering defense | Verifier excluded from agent container |

## Starting artifact

The starter policy is intentionally:

- runnable before the agent starts;
- substantially better than random;
- simple enough to understand and modify;
- weaker than the frozen reference controller;
- expressed through the same final policy interface.

Starting from an empty file would overemphasize one-shot controller invention.
Starting from the published final MPC controller would make the task too easy
and contaminated.

## Public development loop

Agents can query short nominal episodes through `evaluate.py`. Short episodes
make parameter search and controller debugging fast enough for repeated
iteration within the wall-clock budget.

The public evaluator reports raw return, forward progress, fall rate, invalid
policy rate, and action latency. It never determines the leaderboard score.

## Final artifact

The only scored artifact is:

```text
final_policy/
  policy.py
```

It must expose `Policy` or `make_policy` and produce finite eight-dimensional
actions from observations. The final evaluator imports this artifact in a clean
process after the agent exits.

Policies may optionally expose:

```python
configure_simulator(model_xml_path=..., frame_skip=...)
```

The evaluator supplies a copied model matching the current episode. It never
passes the live `Env`, mutable scored `MjData`, reward, `info`, or hidden suite
configuration. This preserves the article's model-based planning capability
without letting a policy step or mutate the scored simulator.

## Hidden evaluation

The reproducibility suite evaluates:

- unseen nominal initialization seeds;
- lower density and friction;
- higher density and friction;
- changed damping;
- weaker actuators.

All episodes use the full 1,000-step Ant horizon.

The checked-in config exists so contributors can reproduce and test the task.
A hosted public leaderboard should inject a private config and matching private
anchors through the verifier's `--config` and `--anchors` arguments.

## Score

```text
raw score =
    0.70 * hidden nominal mean return
  + 0.30 * hidden dynamics-robustness mean return
```

Normalization is:

```text
100 * (raw - starter_raw) / (reference_raw - starter_raw)
```

The checked-in verifier evaluates trusted copies of the starter and reference
policies on the same platform as the submission. This keeps `0` and `100`
stable across small MuJoCo platform differences. A hosted private suite may
instead inject fixed scores through its private anchors file.

The reference and oracle policy files are byte-identical. The verifier caches
evaluations by `policy.py` SHA-256 fingerprint so an oracle run evaluates the
expensive MPC controller once rather than once as the submission and again as
the reference anchor.

Interpretation:

- `0`: matches the starter;
- `100`: matches the frozen stronger reference;
- above `100`: exceeds the reference;
- below `0`: regresses from the starter.

Raw nominal and robustness returns remain published so the normalized scalar
does not hide failure modes.

## Resource model

The first published sweep uses:

- 30 wall-clock minutes per model;
- equal task files and starter policy;
- isolated Docker workspaces;
- each model's highest supported reasoning level.

Simulation Heuristics Ant v1 currently enforces wall-clock time but does not
centrally meter every simulator step. Future hosted versions should add an
authoritative interaction meter before making sample-efficiency claims across
agents.

Internal MPC transitions must also remain separate from external environment
steps in any future accounting. The final reference performs roughly
`96 x 10` copied-model environment steps for every scored external action, so
full hidden evaluation is intentionally a long-running publication check.

Ordinary unit tests use an injected one-step hidden config. The full
five-seed, 1,000-step reproduction is opt-in through:

```bash
GENESISBENCH_RUN_SLOW_ANT_MPC=1 \
  uv run pytest -q tests/test_simulation_heuristics_ant_v1.py
```

## Benchmark integrity

- The agent container receives the public task but not `verifier/`.
- The agent receives only a copied model path for optional MPC, never the live
  scored environment.
- Credentials are supplied through a temporary mode-`0600` file and removed
  after agent startup.
- Final scores come from the independently imported final policy.
- Invalid actions receive a fixed failure result instead of crashing scoring.
- Public packaged score files contain only relative paths.
- Raw trajectories are excluded from the source distribution.

## What Simulation Heuristics Ant v1 establishes

Simulation Heuristics Ant v1 demonstrates the full GenesisBench task lifecycle:

```text
task scaffold
→ public workspace preparation
→ autonomous OpenHands run
→ clean final evaluation
→ packaged submission
→ audited leaderboard
```

It is a simulation benchmark and does not establish real-robot transfer.
Future tasks should add manipulation, navigation, richer sensing, recovery
cases, and eventually hardware evaluation without weakening the clean
artifact/evaluator boundary.
