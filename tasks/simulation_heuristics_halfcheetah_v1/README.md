# Simulation Heuristics HalfCheetah v1

`simulation_heuristics_halfcheetah_v1` packages the MuJoCo HalfCheetah article
experiment from *Learning Beyond Gradients* as a native GenesisBench task.

The research progression remains visible in the benchmark:

```text
asymmetric Fourier CPG
→ joint-space PD tracking
→ copied-model one-step action scoring
→ top-K two-level online tree
→ staged swing-amplitude schedule
```

## Task contract

The agent receives:

- `task.md`: BenchFlow-native config and complete instruction;
- `starter_policy/policy.py`: the asymmetric CPG/PD baseline;
- `evaluate.py`: queryable development feedback;
- `task_context/`: reward, policy API, and article context;
- `_runtime/`: copied into the isolated workspace by the preparation script.

The preparation step creates `final_policy/policy.py` from the starter. The
clean verifier and oracle are excluded from the prepared agent workspace.

## Published reference

The article reports the staged-tree policy on seeds `100..104`:

```text
mean 11836.693
min  11735.0
max  12041.2
```

The trusted policy reimplements the published controller while adapting it to
an observation-only action interface. The optional configuration hook supplies
a copied MuJoCo XML path, not the live environment.

## Score

```text
raw score =
    0.70 * hidden nominal mean return
  + 0.30 * hidden dynamics-robustness mean return
```

The normalized score maps:

- the checked-in asymmetric CPG/PD starter to `0`;
- the checked-in staged-tree reference to `100`.

Both anchors are evaluated locally with the same runtime and dynamics suite as
the submitted policy. Scores above `100` are valid.

## Run locally

```bash
uv sync --extra dev

uv run python tasks/simulation_heuristics_halfcheetah_v1/evaluate.py \
  --policy tasks/simulation_heuristics_halfcheetah_v1/starter_policy/policy.py
```

Run a one-step reference smoke:

```bash
uv run python tasks/simulation_heuristics_halfcheetah_v1/evaluate.py \
  --policy tasks/simulation_heuristics_halfcheetah_v1/oracle/policy.py \
  --episodes 1 \
  --max-steps 1 \
  --seed 100
```

Run the full article reproduction:

```bash
uv run python tasks/simulation_heuristics_halfcheetah_v1/evaluate.py \
  --policy tasks/simulation_heuristics_halfcheetah_v1/oracle/policy.py \
  --episodes 5 \
  --max-steps 1000 \
  --seed 100
```

The exact tree planner is CPU-heavy. One 1,000-step seed took roughly six
minutes in the source script and eight minutes through the benchmark evaluator
on the development host, so the five-seed and full hidden suites are
long-running publication checks rather than unit tests.

Validate the native task:

```bash
uv run python scripts/validate_tasks.py \
  --task simulation_heuristics_halfcheetah_v1

uv run bench tasks check \
  tasks/simulation_heuristics_halfcheetah_v1 \
  --level publication-grade
```

Run the trusted oracle through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_halfcheetah_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

Prepare the public workspace:

```bash
uv run python scripts/prepare_task.py \
  simulation_heuristics_halfcheetah_v1 \
  /tmp/genesisbench-simulation-heuristics-halfcheetah-v1 \
  --force
```

The prepared workspace must not contain `verifier/`, `oracle/`, or `evidence/`.

## Provenance

The inspected article artifact repository did not contain an explicit license
at commit `3555c2956c257d49a5015b782cbe485b14fd659e`. No upstream source file is
vendored. The implementation is a GPL-3.0-or-later policy-API adaptation; the
published numeric controller parameters are retained as reproducibility data.
See `evidence/source_provenance.md` for the exact evidence trail.
