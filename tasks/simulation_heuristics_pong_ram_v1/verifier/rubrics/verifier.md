# Simulation Heuristics Pong RAM v1 Verifier Rubric

- `task_success`: weighted hidden-suite native Pong score, normalized so the
  weak starter maps to `0` and the frozen Pong-21 reference maps to `1`.

The hidden score combines:

- `70%` unseen nominal seeds;
- `30%` unseen seeds with a wider random no-op reset window.

Malformed or failing policies receive `-21` for each affected episode. The
BenchFlow reward is clamped to `[0, 1]`, while the detailed output retains raw
scores and unclamped normalization.
