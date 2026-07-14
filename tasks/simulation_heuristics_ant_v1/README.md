# Simulation Heuristics Ant v1 — Reference Task

`simulation_heuristics_ant_v1` is the canonical, end-to-end example for
GenesisBench contributors. It demonstrates how to turn an open-ended robotics
research loop into a repeatable coding-agent benchmark:

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

- `task.md`: BenchFlow-native config plus the full agent instruction body;
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
- the checked-in article residual-MPC reference controller to `100`.

The verifier evaluates trusted copies of both anchors on the same platform as
the submitted policy, avoiding small MuJoCo macOS/Linux drift. Scores above
`100` are valid. It caches evaluations by `policy.py` fingerprint, so an oracle
submission byte-identical to the reference is not evaluated twice.

## Article reference

The trusted reference adapts the final Ant controller from *Learning Beyond
Gradients* at revision
`3555c2956c257d49a5015b782cbe485b14fd659e`.

It combines a speed-adaptive CPG/PD gait with warm-started residual MPC:

```text
96 candidate plans per external step
× 10-step horizon
× copied MuJoCo model
```

The article's EnvPool rerun reported seeds `0..4` with mean `6005.521`,
minimum `5776.805`, and maximum `6146.208`.

Gymnasium and the supplied EnvPool XML parse to the same MuJoCo model. Their
reset random-number streams differ, so platform-specific results are recorded
separately in `evidence/source_provenance.json`.

On the development host, the GenesisBench oracle reproduced the imported
source controller exactly under Gymnasium:

```text
seeds 0..4
mean 5895.932216
min  5791.444245
max  6131.400491
```

All five episodes reached 1,000 steps. The adapted and source controllers also
selected byte-identical floating-point actions over a 50-step parity probe.

## Run locally

```bash
uv sync --extra dev

uv run python tasks/simulation_heuristics_ant_v1/evaluate.py \
  --policy tasks/simulation_heuristics_ant_v1/starter_policy/policy.py

uv run python tasks/simulation_heuristics_ant_v1/verifier/evaluate_hidden.py \
  tasks/simulation_heuristics_ant_v1/starter_policy/policy.py
```

The hidden-evaluator command runs the full suite and calibrates the MPC
reference locally, so it can take tens of minutes. Use the focused pytest suite
for ordinary development.

Run a one-step MPC smoke:

```bash
uv run python tasks/simulation_heuristics_ant_v1/evaluate.py \
  --policy tasks/simulation_heuristics_ant_v1/oracle/policy.py \
  --episodes 1 \
  --max-steps 1 \
  --seed 0
```

Run the full five-seed article-reference reproduction:

```bash
uv run python tasks/simulation_heuristics_ant_v1/evaluate.py \
  --policy tasks/simulation_heuristics_ant_v1/oracle/policy.py \
  --episodes 5 \
  --max-steps 1000 \
  --seed 0
```

The exact planner is intentionally CPU-heavy. Use the gated regression test
when a full reproduction is required:

```bash
GENESISBENCH_RUN_SLOW_ANT_MPC=1 \
  uv run pytest -q tests/test_simulation_heuristics_ant_v1.py
```

Validate the native BenchFlow package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_ant_v1 \
  --level publication-grade
```

Run the trusted oracle end to end through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_ant_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

The checked-in `verifier/config.toml` is the full publication-grade suite and
can take tens of minutes with the residual-MPC reference. CI temporarily swaps
in `verifier/config_smoke.toml` for the command above. The smoke suite still
exercises the Docker environment, copied-model planning, a dynamics variant,
anchor normalization, and reward emission, while keeping the GitHub Actions
gate bounded.

`--context-root .` lets BenchFlow stage the repo-root sources referenced by the
task's `environment/Dockerfile` into its isolated build context.

Prepare exactly the public workspace an agent receives:

```bash
uv run python scripts/prepare_task.py \
  simulation_heuristics_ant_v1 \
  /tmp/genesisbench-simulation-heuristics-ant-v1 \
  --force
```

The prepared coding-agent workspace must not contain `verifier/`, `oracle/`,
or `evidence/`.

## Files contributors should study

| File | Why it matters |
| --- | --- |
| `task.md` | BenchFlow-native metadata and agent-facing instruction |
| `evaluate.py` | Fast public feedback loop |
| `starter_policy/policy.py` | Fixed initial artifact |
| `task_context/policy_api.md` | Stable policy interface |
| `verifier/evaluate_hidden.py` | Clean final score calculation |
| `verifier/config.toml` | Reproducibility evaluation suite |
| `verifier/config_smoke.toml` | One-step CI integration suite |
| `verifier/anchors.json` | Frozen normalization anchors |
| `oracle/solve.sh` | BenchFlow oracle entrypoint |
| `evidence/source_provenance.json` | Source hashes, license, and reproduction metrics |

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
