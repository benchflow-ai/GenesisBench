# Simulation Heuristics Breakout RGB v1

This native GenesisBench task packages the article's RAM-to-RGB Breakout
transfer as a strict pixel-only policy-improvement benchmark.

## Contract

The agent receives `task.md`, `starter_policy/`, `evaluate.py`, and
`task_context/`. It must leave:

```text
final_policy/
  policy.py
```

The policy receives raw RGB pixels only. The evaluator does not pass RAM,
reward, `info`, or object labels, and the hidden verifier is not copied into
the agent workspace.

## Score

```text
raw score =
    0.50 * hidden nominal return
  + 0.50 * hidden shifted-start return
```

Normalization maps the checked-in starter to `0` and the frozen pixel-only
864-point reference to `100`. Both trusted anchors are evaluated on the same
runtime as the submission. Scores above `100` remain representable.

## Local verification

Use a Linux environment with `envpool==1.1.1`, then run:

```bash
python tasks/simulation_heuristics_breakout_rgb_v1/evaluate.py \
  --policy tasks/simulation_heuristics_breakout_rgb_v1/starter_policy/policy.py

python tasks/simulation_heuristics_breakout_rgb_v1/verifier/evaluate_hidden.py \
  tasks/simulation_heuristics_breakout_rgb_v1/oracle/policy.py
```

Validate the native package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_breakout_rgb_v1 \
  --level publication-grade
```

Run the trusted oracle through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_breakout_rgb_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

## Public and private suites

The checked-in verifier config is a reproducibility suite. A hosted leaderboard
should inject a private config and matching private anchors:

```bash
python verifier/evaluate_hidden.py final_policy/policy.py \
  --config /private/final-suite.toml \
  --anchors /private/final-anchors.json
```

See `task_context/article_progression.md` for the `310 -> 428 -> 864` transfer
mechanisms and `task_context/provenance.md` for revision/licensing notes.
