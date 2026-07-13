# Evaluation

## Public development suite

The default public command runs one ten-lane EnvPool batch:

```text
scenario: D1Basic-v1
EnvPool: 1.1.1
batch seed: 0
episodes: 10
frame skip: 1
maximum steps: 2,100
render size: 240 x 180
```

EnvPool's native D1 reward is accumulated without reshaping. The evaluator
reports mean, minimum, and maximum return, per-lane returns and lengths,
invalid-policy rate, final public variables, and action latency.

## Hidden suite

The checked-in reproducibility verifier uses seed batches that do not overlap
the public batch. A hosted leaderboard can inject a private config and matching
private anchors.

Raw score is the weighted mean return across hidden suites. The normalized
GenesisBench score maps:

- the checked-in starter to `0`;
- the frozen screen-CV reference to `100`.

Scores above `100` are valid. The checked-in reproducibility anchors are
numeric scores generated from clean, separate EnvPool `1.1.1` processes. The
verifier selects an explicit `darwin-arm64` or `linux-x86_64` profile and
validates its per-suite means against the exact selected config before using
it. Missing platform calibration fails closed. Candidate seed suites run in
fresh subprocesses and unique working directories because the EnvPool `1.1.1`
D1 runtime is not safely re-entrant in one Linux process. Private path-based
anchors remain supported and use the same subprocess isolation.

The verifier also clears stale D1 state before every suite. Cleanup is limited
to task-owned `/app/evaluate.py`/VizDoom processes and exact `ViZDoom*`
shared-memory artifacts owned by `root` or `agent`; unrelated shared-memory
files are preserved.

## Failure handling

A policy receives the configured failure return for an episode if it:

- fails to import or instantiate;
- returns an invalid action;
- raises during `act`;
- violates the privileged-state source audit.
