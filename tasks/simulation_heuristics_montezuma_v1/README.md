# Simulation Heuristics Montezuma v1 — Boundary Reference Task

`simulation_heuristics_montezuma_v1` turns the Montezuma boundary experiment
from *Learning Beyond Gradients* into a native GenesisBench task:

```text
native RGB starter
→ public development evaluation
→ final_policy/policy.py
→ hidden seed/no-op/recovery suites
→ platform-local 0/100 anchors
```

## Historical result preserved by the task

The article reports that an earlier graph search improved key distance
`72 → 28` without earning reward. A later native-image run replayed `400`
points in `1769` environment steps with `86` macro-actions, but was mostly
open-loop. The trusted reference reproduces that boundary result while using
image-based trajectory re-entry to survive the checked-in recovery suite.

## Score

Each episode contributes at most `400` raw points:

```text
raw score =
    0.25 * reproduction
  + 0.10 * hidden deterministic seeds
  + 0.10 * startup no-op variants
  + 0.55 * intermediate-state recovery
```

The verifier evaluates the checked-in starter and trusted reference on the
same platform as the submission. They normalize to exactly `0` and `100`.
A plain copied open-loop replay receives reproduction credit but fails the
recovery handoffs and therefore cannot reach the reference anchor.

## Local commands

Install EnvPool only for the command being run:

```bash
uv run --with envpool==1.1.1 \
  python tasks/simulation_heuristics_montezuma_v1/evaluate.py \
  --policy tasks/simulation_heuristics_montezuma_v1/starter_policy/policy.py
```

Run the hidden evaluator:

```bash
uv run --with envpool==1.1.1 \
  python tasks/simulation_heuristics_montezuma_v1/verifier/evaluate_hidden.py \
  tasks/simulation_heuristics_montezuma_v1/oracle/policy.py
```

Validate the native task package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_montezuma_v1 \
  --level publication-grade
```

Run the oracle through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_montezuma_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

The prepared agent workspace excludes `oracle/`, `verifier/`, and `evidence/`.
Official leaderboard runs should inject a private suite and matching anchors.

## Provenance

The evaluator, starter, and recovery implementation are independent
GenesisBench code. The calibration trajectory is attributed in
`task_context/provenance.md`; retain that notice when redistributing the task.
