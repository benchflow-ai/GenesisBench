# Simulation Heuristics Breakout RGB v1 Verifier Rubric

- `task_success`: weighted hidden-suite Breakout return normalized so the
  public starter maps to `0` and the frozen 864-point pixel-only reference maps
  to `1`.
- Any attempt to return malformed/out-of-range actions fails the affected
  episode; the policy process receives RGB frames only.
- The canonical reward is clamped to `[0, 1]`; detailed JSON retains raw and
  unclamped normalized scores.
