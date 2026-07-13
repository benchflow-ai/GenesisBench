# History, provenance, and licensing

This task preserves the boundary account published in *Learning Beyond
Gradients*:

- the article reports an earlier state-graph search moving key distance from
  `72` to `28` while score stayed `0`;
- the repaired replay `repair_replay_r1_t19734` scored `400`, used `86`
  macro-actions and `1769` environment steps, and was described as mostly
  open-loop.

The calibration source inspected for this task was:

```text
article: https://trinkle23897.github.io/learning-beyond-gradients/
repository: https://github.com/Trinkle23897/learning-beyond-gradients
commit: 3555c2956c257d49a5015b782cbe485b14fd659e
artifact: atari/montezuma/heuristic_montezuma_400_macros.json
environment: EnvPool 1.1.1 MontezumaRevenge-v5
seed: 10001
```

GenesisBench independently implements the evaluator, starter, recovery
protocol, image matching, serialization, and verifier. No upstream Python
implementation is copied.

The checked-in `reference_trajectory.npz` contains only the expanded action
ids plus native-image hashes and downsampled image features regenerated with
EnvPool `1.1.1`; it contains no RAM or reward trace. Its SHA-256 is:

```text
72f7211be1c73d556c727b5ef1dc1fbd6aeddc7fe96d44ce99c764089f147aa1
```

The inspected upstream snapshot did not contain a top-level license file, and
the replay artifact did not carry a file-level license grant. The expanded
action trace and regenerated native-image fingerprints are therefore retained
as provenance-tagged benchmark calibration data; this task does not assert a
general relicensing of the upstream artifact. Keep this notice and confirm any
additional permission needed before redistributing the calibration data outside
this benchmark.
