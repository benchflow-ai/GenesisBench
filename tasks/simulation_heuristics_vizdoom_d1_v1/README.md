# Simulation Heuristics VizDoom D1 v1

This task turns the D1 Basic experiment from *Learning Beyond Gradients* into
a clean GenesisBench policy-improvement package.

The public policy contract is deliberately narrow:

- input image: the `240 x 180` RGB frame returned by EnvPool `render()`;
- public variable: `HEALTH`;
- output: one of the six combined D1 actions.

The policy never receives the EnvPool environment, WAD/config paths, map
geometry, labels, object coordinates, the base seed, or the vector lane id.
The evaluator also audits policy source for direct simulator or privileged-map
access.

## Article reproduction target

With EnvPool `1.1.1`, batch seed `0`, ten lanes, frame skip `1`, and 2,100
steps, the frozen reference policy reproduces:

```text
mean = 0.9440999741666019
min  = 0.28999998047947884
rewards = [
  1.0799999684095383,
  1.0399999842047691,
  0.28999998047947884,
  0.9279999658465385,
  1.0799999684095383,
  1.0099999718368053,
  1.0349999852478504,
  1.004999976605177,
  0.9439999675378203,
  1.0289999730885029,
]
```

## Task flow

```text
starter_policy/policy.py
→ public evaluate.py
→ prepared final_policy/policy.py
→ hidden unseen-seed evaluation
→ local starter/reference normalization
```

`final_policy/` is created in the prepared agent workspace from
`starter_policy/`; it is intentionally not checked into the source package
because the shared task preparer owns that copy step.

## Hidden verifier isolation

EnvPool `1.1.1` cannot safely create multiple D1 VizDoom instances
sequentially in one Linux verifier process and working directory. The hidden
evaluator therefore runs each seed suite in a fresh subprocess with a unique
working directory.

Before each suite, the root verifier terminates only stale task-owned
`/app/evaluate.py` and VizDoom processes, then removes exact `ViZDoom*`
artifacts owned by the task from `/dev/shm` and
`/tmp/boost_interprocess`. This releases shared memory left by timed-out agent
development evaluations without touching unrelated resources.

The checked-in starter/reference scores were calibrated in clean, separate
EnvPool `1.1.1` runs. D1 screen-CV returns differ between the supported
`darwin-arm64` and `linux-x86_64` runtimes, so each platform has an explicit
profile. Per-suite means and the exact hidden config are stored in
`verifier/anchors.json`; missing or stale platform calibration fails closed.

## Run locally

```bash
python tasks/simulation_heuristics_vizdoom_d1_v1/evaluate.py \
  --policy tasks/simulation_heuristics_vizdoom_d1_v1/oracle/policy.py
```

Validate the native task package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_vizdoom_d1_v1 \
  --level publication-grade
```

## Provenance and licensing

The behavioral contract and published target come from
`Trinkle23897/learning-beyond-gradients` commit
`3555c2956c257d49a5015b782cbe485b14fd659e`.
The source repository's runtime notes explicitly pin the experiments to
EnvPool `1.1.1`; the task Docker image and evaluator enforce the same version.

That repository did not declare a source license at the pinned revision.
Accordingly, this package does not vendor its policy source. The evaluator,
task API, starter, and compact reference policy are independently implemented
for GenesisBench from the public experiment description and measured
behavior. EnvPool is installed from its published package and retains its own
license.
