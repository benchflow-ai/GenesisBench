# Contributing GenesisBench Tasks

GenesisBench tasks evaluate whether a coding agent can improve a physical
policy or robotics software artifact under a fixed resource budget.

Use `tasks/ant_v1/` as the canonical working example and `tasks/_template/` as
the scaffold source.

## Create a task

```bash
uv sync --extra dev

uv run python scripts/create_task.py my_robot_task \
  --title "My Robot Policy Improvement Task"
```

This creates `tasks/my_robot_task/` without changing the template.

## Required benchmark shape

Every task must define:

1. **Fixed starting artifact** — a controller, planner, policy, model, or
   training system that runs before the agent starts.
2. **Agent-facing objective** — one clear optimization goal in `prompt.md`.
3. **Queryable development feedback** — a reasonably fast `evaluate.py`.
4. **Bounded resources** — wall-clock plus simulator interactions, compute, or
   physical trials where relevant.
5. **Standard final artifact** — declared in `[submission]` in `task.toml`.
6. **Clean final verifier** — executed after the agent exits and not exposed in
   the task workspace.
7. **Robust final suite** — unseen seeds, scenarios, dynamics, or hardware
   conditions that measure generalization rather than one public trajectory.
8. **Reproducible score** — deterministic where possible; otherwise specify
   seeds, confidence intervals, and rerun policy.

## Required files

```text
tasks/<task_name>/
  README.md
  benchmark.txt
  task.toml
  prompt.md
  evaluate.py
  <starter path declared in task.toml>
  task_context/
  verifier/
    evaluate_hidden.py
```

Task-specific files and assets may be added as needed.

## Public workspace boundary

`scripts/prepare_task.py` copies the agent-visible task and deliberately
excludes `verifier/`. Never rely only on prompt instructions to protect hidden
evaluation data.

The checked-in verifier can be a reproducibility suite. An official public
leaderboard should inject a private final suite when public source access would
otherwise reveal seeds, scenarios, answers, or dynamics parameters.

## `task.toml` core fields

```toml
version = "1.0"
name = "my_robot_task"
title = "My Robot Policy Improvement Task"

[metadata]
description = "One-sentence task description."
author = "Your Name"
category = "locomotion"
difficulty = "medium"
tags = ["robotics"]
reference_task = false

[starter]
path = "starter_policy"

[submission]
directory = "final_policy"
entrypoint = "policy.py"

[budget]
wall_clock_minutes = 30

[verifier]
entrypoint = "verifier/evaluate_hidden.py"
supports_private_config = true
```

Additional environment and evaluation fields are task-specific.

## Development and final evaluation

Development evaluation should be:

- fast enough for repeated iteration;
- representative enough to guide research;
- explicit about charged simulator or hardware interactions;
- machine-readable;
- unable to mutate the official final score.

Final evaluation should:

- import only the submitted artifact;
- run in a clean process or container;
- reject malformed outputs safely;
- record raw metrics alongside the scalar score;
- include robustness conditions;
- use full task horizons unless the benchmark explicitly studies short-horizon
  behavior.

## Acceptance checklist

Before opening a contribution:

```bash
uv run python scripts/validate_tasks.py
uv run pytest -q
uv run ruff check .
```

Also demonstrate:

- the starter runs;
- the submitted artifact contract is enforced;
- random, starter, and reference anchors are meaningfully separated;
- hidden evaluation cannot be read from the prepared workspace;
- the score reproduces in a clean process;
- one real coding-agent canary completes end to end;
- licenses and provenance are recorded for vendored assets.

## Avoid these benchmark failures

- Evaluating only one public seed.
- Shipping the strongest public solution as the starter.
- Letting the policy read reward, hidden dynamics, or simulator internals unless
  those are explicitly part of the task.
- Counting internal planning rollouts as external environment steps without
  defining that accounting.
- Trusting an agent-reported score instead of independently evaluating its
  final artifact.
- Publishing simulation results as evidence of real-robot transfer.

