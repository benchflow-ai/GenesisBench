# Simulation Heuristics Atari57 v1 verifier rubric

## `task_success`

The scored quantity is the normalized hidden-suite primary aggregate:

1. mean HNS over three evaluation repeats for each game/mode;
2. better mode mean for each game;
3. median over all 57 games;
4. numeric normalization where HNS `0.0` maps to `0` and article-reference
   HNS `0.8283015254994576` maps to `1`.

The canonical reward is clamped to `[0, 1]`; the details JSON may contain
negative normalized scores or values above `100`.

## Publication eligibility

Artifact validity is mandatory. Interaction accounting is reported separately.
The checked-in full suite sets `require_complete_search_ledger=true`, making
342 completed records with at least 20M counted frames and the required
evidence files an eligibility gate. Incomplete artifacts return reward and
normalized score zero without starting EnvPool. The deterministic smoke config
disables this gate only for contract testing.

The looser best-single-run HNS is diagnostic only and never replaces the
primary best-input-mean metric.
