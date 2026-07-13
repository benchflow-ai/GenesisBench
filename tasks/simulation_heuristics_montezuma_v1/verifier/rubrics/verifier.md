# Montezuma's Revenge Boundary Policy Improvement Verifier Rubric

- `task_success`: normalized weighted native-game score, with each episode
  capped at `400`, the starter mapped to `0`, and the trusted image-synchronized
  reference mapped to `1`.

The checked-in weighting is `25%` reproduction, `10%` deterministic hidden
seeds, `10%` startup no-op variants, and `55%` intermediate-state recovery.
