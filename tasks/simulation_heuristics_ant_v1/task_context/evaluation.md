# Evaluation

Per-step reward:

```text
reward = x_velocity + 1.0 - 0.5 * sum(action ** 2)
```

The healthy reward is received while the torso state is finite and torso
height is within `[0.2, 1.0]`.

The public evaluator reports:

- mean, minimum, and maximum return;
- mean final x-position;
- fall rate;
- invalid-policy rate;
- mean action latency;
- per-episode metrics.

The checked-in hidden suite runs full 1,000-step episodes over unseen nominal
seeds and two conservative dynamics variants. Its raw score is:

```text
0.70 * hidden nominal mean return
+ 0.30 * hidden dynamics-robustness mean return
```

The score is normalized locally so the weak public CPG/PD starter maps to `0`
and the article residual-MPC reference maps to `100`. Invalid policies receive
a fixed failure return for the affected episode.

## Article reference

The *Learning Beyond Gradients* EnvPool rerun reported seeds `0..4` with mean
`6005.521`, minimum `5776.805`, and maximum `6146.208`.

GenesisBench runs Gymnasium rather than EnvPool. The supplied article XML and
Gymnasium's Ant XML parse to the same MuJoCo model, but reset RNG behavior
differs between runtimes. Exact source-policy checks on this platform are
recorded under `evidence/source_provenance.json`.

Internal MPC model transitions are not external environment steps. They do
count against verifier wall-clock limits and action-latency metrics.
