# Evaluation

## Public development suite

The default public command runs one ten-lane EnvPool batch:

```text
scenario: D3Battle-v1
EnvPool: 1.1.1
batch seed: 0
episodes: 10
frame skip: 2
maximum steps: 1,050
screen size: 640 x 480
reward: DAMAGECOUNT + 10 * KILLCOUNT
```

Weapon and crosshair rendering are disabled. The configured action channels
are attack, speed, forward/backward movement, strafing, 180-degree turn, and
horizontal turn delta.

The evaluator reports mean, minimum, and maximum return, per-lane returns and
lengths, invalid-policy rate, final allowlisted variables, and action latency.

## Hidden suite

The checked-in reproducibility verifier uses seed batches that do not overlap
the public batch. A hosted leaderboard can inject a private config and matching
private anchors.

Raw score is the weighted mean return across hidden suites. The normalized
GenesisBench score maps:

- the checked-in sweep-and-fire starter to `0`;
- the frozen screen-CV battle reference to `100`.

Scores above `100` are valid. The checked-in reproducibility anchors are
numeric scores generated
from clean, separate EnvPool `1.1.1` processes. The verifier validates their
per-suite means against the exact selected config before using them. Candidate
seed suites run in fresh subprocesses and unique working directories because
the EnvPool `1.1.1` D3 runtime is not safely re-entrant in one Linux process.
Private path-based anchors remain supported and use the same subprocess
isolation.

## Failure handling

A policy receives the configured failure return for an episode if it:

- fails to import or instantiate;
- returns an invalid action;
- raises during `act`;
- violates the privileged-state source audit.
