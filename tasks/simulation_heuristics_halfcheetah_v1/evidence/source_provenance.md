# Source provenance and licensing review

## Evidence inspected

- Artifact repository:
  `https://github.com/Trinkle23897/learning-beyond-gradients`
- Inspected commit:
  `3555c2956c257d49a5015b782cbe485b14fd659e`
- Article source:
  `learning-beyond-gradient.en.md`
- Experiment source:
  `mujoco/halfcheetah/heuristic_halfcheetah_v5.py`
- Iteration record:
  `mujoco/halfcheetah/heuristic_halfcheetah_v5_log.md`

The article identifies the work as *Learning Beyond Gradients*, Jiayi Weng,
May 2026. Its appendix reports a five-episode mean of `11836.693` for seeds
`100..104`.

## Licensing treatment

No `LICENSE`, `COPYING`, or equivalent repository-level license was present in
the inspected source tree at the commit above. Therefore GenesisBench does not
vendor the upstream Python file or present it as licensed source code.

The task contains a newly written policy-API adaptation under GenesisBench's
GPL-3.0-or-later licensing. It preserves the published algorithmic semantics
and numeric controller parameters as experiment data required for
reproducibility:

- asymmetric two-rate CPG/PD gait;
- copied-model, short-horizon scoring;
- top-K two-level action tree;
- staged swing-amplitude schedule;
- published seeds and aggregate target.

The implementation reconstructs a local MuJoCo planning state from the public
17-value observation rather than receiving the live environment object used by
the original script.

## Local source check

On July 12, 2026, the unmodified source command for seed 100 produced
`12041.189857475818`. A 20-step parity probe showed the GenesisBench oracle and
source planner selecting identical actions at every checked step. The complete
GenesisBench oracle run for seed 100 also returned exactly
`12041.189857475818` over 1,000 steps.

The full packaged five-seed reproduction subsequently produced mean
`11836.693449819431`, minimum `11735.02927325886`, and maximum
`12041.189857475818`. See `evidence/article_reproduction.json` for the
machine-readable command and per-seed returns.
