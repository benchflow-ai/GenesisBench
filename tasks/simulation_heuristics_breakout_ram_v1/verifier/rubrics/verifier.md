# Simulation Heuristics Breakout RAM v1 Verifier Rubric

- `task_success`: weighted hidden-suite Breakout return normalized so the
  public starter maps to `0` and the frozen 864-point RAM reference maps to
  `1`.
- Invalid or out-of-range actions fail the affected episode.
- The canonical reward is clamped to `[0, 1]`; detailed JSON retains raw and
  unclamped normalized scores.
