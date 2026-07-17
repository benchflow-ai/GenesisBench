# History, provenance, and licensing

This task preserves the boundary account published in *Learning Beyond
Gradients*:

- the article reports an earlier state-graph search moving key distance from
  `72` to `28` while score stayed `0`;
- the repaired replay `repair_replay_r1_t19734` scored `400`, used `86`
  macro-actions and `1769` environment steps, and was described as mostly
  open-loop.

Exact upstream repository paths, revisions, answer artifacts, and replay seeds
are retained outside the agent image in maintainer-only provenance records.

GenesisBench independently implements the evaluator, starter, recovery
protocol, image matching, serialization, and verifier. No upstream Python
implementation is copied.

The verifier-side reference trajectory contains expanded action ids plus
native-image hashes and downsampled image features regenerated with the pinned
runtime; it contains no RAM or reward trace and is not included in the agent
image.

The inspected upstream snapshot did not contain a top-level license file, and
the replay artifact did not carry a file-level license grant. The expanded
action trace and regenerated native-image fingerprints are therefore retained
as provenance-tagged benchmark calibration data; this task does not assert a
general relicensing of the upstream artifact. Keep this notice and confirm any
additional permission needed before redistributing the calibration data outside
this benchmark.
