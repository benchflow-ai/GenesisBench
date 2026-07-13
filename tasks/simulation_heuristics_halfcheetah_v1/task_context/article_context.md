# Article experiment context

This task adapts the MuJoCo HalfCheetah experiment published with *Learning
Beyond Gradients* (Jiayi Weng, May 2026).

The source artifact describes an interpretable progression:

1. periodic Fourier torque rules;
2. proprioceptive reflexes;
3. a symmetric PD/CPG target-angle gait;
4. an asymmetric two-rate PD/CPG gait;
5. one-step model-predictive action selection;
6. a two-level top-K action tree;
7. staged swing-amplitude target schedules.

The final named policy,
`mpc-staged-tree-asym-pd-cpg`, uses a 14-step closed-loop CPG/PD tail, a
terminal velocity term, top-K width 8, high-gain undamped PD tracking inside
the planner, and target schedules:

```text
steps   0..299: harmonic amplitude 1.15, front lower-leg bias 0.15
steps 300..899: harmonic amplitude 1.18, front lower-leg bias 0.20
steps 900..999: harmonic amplitude 1.15, front lower-leg bias 0.15
```

The article reports five seeds `100..104` with mean return `11836.693`, minimum
`11735.0`, and maximum `12041.2`.

No upstream source file is vendored into the public workspace. The GenesisBench
implementation is a policy-API adaptation under this repository's GPL-3.0
license. Numeric controller parameters are retained as experiment data needed
to reproduce the published result. Detailed provenance is kept outside the
agent workspace under `evidence/`.
