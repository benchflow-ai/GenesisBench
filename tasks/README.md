# Contributing GenesisBench Tasks

GenesisBench tasks evaluate whether a coding agent can improve a physical
policy or robotics software artifact under a fixed resource budget.

Use `tasks/simulation_heuristics_ant_v1/` as the canonical working example and `tasks/_template/` as
the scaffold source.

GenesisBench follows BenchFlow `0.6.5`'s
[native task package standard](https://github.com/benchflow-ai/benchflow/blob/main/docs/task-standard.md).

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
2. **Agent-facing objective** — one clear optimization goal in the `task.md`
   body.
3. **Queryable development feedback** — a reasonably fast `evaluate.py`.
4. **Bounded resources** — wall-clock plus simulator interactions, compute, or
   physical trials where relevant.
5. **Standard final artifact** — declared under
   `metadata.genesisbench.submission` in `task.md`.
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
  task.md
  evaluate.py
  environment/
    Dockerfile
  <starter path declared in task.md>
  task_context/
  oracle/
    solve.sh
  verifier/
    verifier.md
    test.sh
    evaluate_hidden.py
```

Task-specific files and assets may be added as needed.

## Container layout

Each current task owns its agent environment at:

```text
tasks/<task_name>/environment/Dockerfile
```

BenchFlow builds that task-local Dockerfile with the repository root as its
context. There is intentionally no parallel top-level `containers/<task>/`
tree. The former `containers/simulation_heuristics_ant_v1` image was an
OpenHands-specific runner for the retired Ant-only sweep, not a task
environment.

## Public workspace boundary

`scripts/prepare_task.py` copies the agent-visible task and deliberately
excludes `verifier/`, `oracle/`, and `evidence/`. Never rely only on prompt
instructions to protect hidden evaluation data.

The checked-in verifier can be a reproducibility suite. An official public
leaderboard should inject a private final suite when public source access would
otherwise reveal seeds, scenarios, answers, or dynamics parameters.

## `task.md` core fields

```markdown
---
schema_version: "1.3"
task:
  name: genesisbench/my_robot_task
  description: One-sentence task description.
  authors:
    - name: Your Name
metadata:
  category: locomotion
  difficulty: medium
  tags: [robotics]
  reference_task: false
  genesisbench:
    starter:
      path: starter_policy
    submission:
      directory: final_policy
      entrypoint: policy.py
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 300
environment:
  cpus: 1
  memory_mb: 2048
  workdir: /app
benchflow:
  document_version: "0.6"
---

Write the complete agent instruction here.
```

This follows BenchFlow `0.6.5`'s native task document format. Do not add
`task.toml`, `instruction.md`, or `prompt.md` mirrors.

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
uv run bench tasks check tasks/<task_name> --level publication-grade
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
