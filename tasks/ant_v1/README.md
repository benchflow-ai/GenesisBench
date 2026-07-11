# Ant v1 — Reference Task

`ant_v1` is the canonical, end-to-end example for GenesisBench contributors.
It demonstrates how to turn an open-ended robotics research loop into a
repeatable coding-agent benchmark:

```text
starter policy
→ agent edits/searches/trains
→ public development evaluation
→ standardized final artifact
→ clean hidden evaluation
→ scalar leaderboard score
```

## Task contract

The agent receives:

- `prompt.md`: objective, allowed methods, submission contract, and rules;
- `starter_policy/`: the fixed starting controller;
- `evaluate.py`: a queryable development evaluator;
- `task_context/`: stable API and reward documentation;
- `_runtime/`: copied into the isolated workspace by the preparation script.

The agent must produce:

```text
final_policy/
  policy.py
```

After the agent exits, GenesisBench runs `verifier/evaluate_hidden.py` without
mounting `verifier/` into the agent container.

## Score

```text
raw score =
    0.70 * hidden nominal mean return
  + 0.30 * hidden dynamics-robustness mean return
```

The normalized score maps:

- the checked-in starter controller to `0`;
- the checked-in stronger reference controller to `100`.

Scores above `100` are valid.

## Run locally

```bash
uv sync --extra dev

uv run python tasks/ant_v1/evaluate.py \
  --policy tasks/ant_v1/starter_policy/policy.py

uv run python tasks/ant_v1/verifier/evaluate_hidden.py \
  tasks/ant_v1/starter_policy/policy.py
```

Prepare exactly the public workspace an agent receives:

```bash
uv run python scripts/prepare_task.py \
  ant_v1 \
  /tmp/genesisbench-ant-v1 \
  --force
```

The prepared directory must not contain `verifier/`.

## Files contributors should study

| File | Why it matters |
| --- | --- |
| `task.toml` | Machine-readable task metadata and artifact paths |
| `prompt.md` | Agent-facing research objective and constraints |
| `evaluate.py` | Fast public feedback loop |
| `starter_policy/policy.py` | Fixed initial artifact |
| `task_context/policy_api.md` | Stable policy interface |
| `verifier/evaluate_hidden.py` | Clean final score calculation |
| `verifier/config.toml` | Reproducibility evaluation suite |
| `verifier/anchors.json` | Frozen normalization anchors |

## Public versus private final suites

The checked-in verifier config makes the task reproducible and testable. For a
public leaderboard, the official evaluator should inject a private config and
matching private anchors:

```bash
python verifier/evaluate_hidden.py final_policy/policy.py \
  --config /private/final-suite.toml \
  --anchors /private/final-anchors.json
```

This prevents agents with internet access from learning final seeds or dynamics
parameters from the public repository.

## Use this task as the example

Start a new contribution with:

```bash
uv run python scripts/create_task.py my_robot_task \
  --title "My Robot Policy Improvement Task"
```

Then follow `tasks/README.md`. Do not copy Ant-specific reward assumptions,
observation indexing, or perturbation ranges into a different robot task.

