# Simulation Heuristics VizDoom D3 v1

This task packages the D3 Battle experiment from *Learning Beyond Gradients*
as a native GenesisBench policy-improvement benchmark.

The public controller boundary contains only:

- the current three-channel screen image;
- public `HEALTH`, `AMMO2`, `HITCOUNT`, `DAMAGECOUNT`, and `KILLCOUNT`;
- an eight-value action vector for attack, movement, and turning.

Map files, object labels, coordinates, automaps, environment objects, seeds,
and vector lane ids never cross the boundary. Policy source is audited for
direct simulator and privileged-map access.

## Article reproduction target

With EnvPool `1.1.1`, batch seed `0`, ten lanes, frame skip `2`, `640 x 480`
screen input, and reward `DAMAGECOUNT + 10 * KILLCOUNT`, the frozen reference
policy reproduces:

```text
mean = 557.0
min  = 440.0
rewards = [545, 475, 480, 440, 690, 500, 600, 595, 530, 715]
```

## Task flow

```text
starter_policy/policy.py
→ public evaluate.py
→ prepared final_policy/policy.py
→ hidden unseen-seed evaluation
→ local starter/reference normalization
```

The shared task preparer creates `final_policy/policy.py` from the starter in
the agent workspace. A checked-in `final_policy/` would conflict with that
native preparation step, so the source package keeps only the starter and
oracle copies.

## Hidden verifier isolation

EnvPool `1.1.1` cannot safely create multiple D3 VizDoom instances
sequentially in one Linux verifier process and working directory. The hidden
evaluator therefore runs each seed suite in a fresh subprocess with a unique
working directory.

The checked-in starter/reference scores were calibrated in clean, separate
EnvPool `1.1.1` runs. Their per-suite means and the exact hidden config are
stored in `verifier/anchors.json`; the verifier rejects numeric anchors when
that calibration metadata drifts.

## Run locally

```bash
python tasks/simulation_heuristics_vizdoom_d3_v1/evaluate.py \
  --policy tasks/simulation_heuristics_vizdoom_d3_v1/oracle/policy.py
```

Validate the native task package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_vizdoom_d3_v1 \
  --level publication-grade
```

## Provenance and licensing

The behavioral contract and target vector come from
`Trinkle23897/learning-beyond-gradients` commit
`3555c2956c257d49a5015b782cbe485b14fd659e`.
The source repository's runtime notes explicitly pin the experiments to
EnvPool `1.1.1`; the task Docker image and evaluator enforce the same version.

The pinned source repository did not declare a license. This package therefore
does not vendor its implementation. The GenesisBench runtime, API, starter,
and minimized reference controller are independently written from the public
experiment description and reproduced behavior. Third-party runtime packages
retain their own licenses.
