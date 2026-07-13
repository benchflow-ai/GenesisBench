# Simulation Heuristics Pong RAM v1

`simulation_heuristics_pong_ram_v1` packages the Pong 21 experiment from
*Learning Beyond Gradients* as a reproducible GenesisBench policy-improvement
task.

The benchmark loop is:

```text
weak RAM paddle tracker
→ agent experiments and edits code
→ public development episodes
→ final_policy/policy.py
→ clean hidden evaluation
→ locally normalized score
```

## Task contract

The public workspace contains:

- `task.md`: BenchFlow-native metadata and complete agent instructions;
- `starter_policy/policy.py`: a weak late reactive controller;
- `evaluate.py`: a queryable development evaluator;
- `task_context/`: stable policy, scoring, and provenance documentation;
- `_runtime/`: the trusted GenesisBench runtime copied by task preparation.

The required artifact is:

```text
final_policy/
  policy.py
```

The policy sees only the current 128-byte Atari RAM vector. Reward, emulator
objects, hidden seeds, and hidden reset configuration remain inside the trusted
evaluator.

## Environment semantics

The package preserves the article experiment's EnvPool `Pong-v5` setup:

- EnvPool `1.1.1`;
- frame skip `1`;
- minimal action space;
- automatic fire reset;
- unclipped reward;
- no sticky actions;
- full-game episodes rather than episodic lives.

The native episode score is the point differential in `[-21, 21]`. A score of
`21` means winning 21–0 and is the experiment target.

## Final score

The checked-in reproduction suite computes:

```text
raw score =
    0.70 * hidden nominal mean Pong score
  + 0.30 * hidden randomized-reset mean Pong score
```

Both suites use unseen seeds. The reset-robustness suite widens EnvPool's
random no-op reset window while preserving frame skip, reward, action, and
sticky-action semantics.

The normalized score maps:

- the checked-in weak starter controller to `0`;
- the checked-in Pong-21 reference controller to `100`.

Both anchors are rerun locally on the same runtime as the submission. Scores
above `100` are permitted by the formula, although `21` is the native Pong
maximum.

## Run locally

EnvPool is task-specific and intentionally is not added to the repository-wide
Python dependencies. Run an evaluator with an ephemeral dependency:

```bash
PYTHONPATH=src uv run --with envpool==1.1.1 \
  python tasks/simulation_heuristics_pong_ram_v1/evaluate.py \
  --policy tasks/simulation_heuristics_pong_ram_v1/starter_policy/policy.py
```

Reproduce the article target:

```bash
PYTHONPATH=src uv run --with envpool==1.1.1 \
  python tasks/simulation_heuristics_pong_ram_v1/evaluate.py \
  --policy tasks/simulation_heuristics_pong_ram_v1/oracle/policy.py \
  --episodes 1 \
  --seed 0
```

The result should report a mean score of `21.0`.

Run the checked-in hidden reproduction suite:

```bash
PYTHONPATH=src uv run --with envpool==1.1.1 \
  python tasks/simulation_heuristics_pong_ram_v1/verifier/evaluate_hidden.py \
  tasks/simulation_heuristics_pong_ram_v1/oracle/policy.py
```

Validate the native task package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_pong_ram_v1 \
  --level publication-grade
```

Run the trusted oracle through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_pong_ram_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

Prepare exactly the public workspace an agent receives:

```bash
uv run python scripts/prepare_task.py \
  simulation_heuristics_pong_ram_v1 \
  /tmp/genesisbench-simulation-heuristics-pong-ram-v1 \
  --force
```

The prepared workspace must not contain `verifier/`, `oracle/`, or
`evidence/`.

## Public versus private final suites

The checked-in verifier is a transparent reproduction suite. A public
leaderboard should inject a private config and matching private anchors:

```bash
python verifier/evaluate_hidden.py final_policy/policy.py \
  --config /private/final-suite.toml \
  --anchors /private/final-anchors.json
```

Private suites should retain the documented Atari semantics while changing
seeds and reset windows. They should not introduce sticky actions or a
different frame skip under the same benchmark name.

## Provenance and licensing

The experiment semantics, RAM coordinate decoding, and geometric reference
controller are derived from:

- Jiayi Weng, *Learning Beyond Gradients* (2026);
- `Trinkle23897/learning-beyond-gradients`;
- source artifact `atari/pong/heuristic_pong.py`;
- source commit `3555c2956c257d49a5015b782cbe485b14fd659e`.

The source artifact carries the Copyright 2021 Garena Online Private Limited
notice and Apache License 2.0 header. GenesisBench retains that notice on
policy files and reimplements the policy-only interface and evaluator cleanly.
No source videos, images, trial logs, or ROM files are vendored into this task.
See `task_context/provenance.md` for the detailed boundary.
