# Article-suite timeout fairness audit

The published five-trial leaderboard was audited cell by cell against the
selected source trajectories for all `4 × 9 × 5 = 180` model-task-trial cells.

## Fairness policy

- Task wall-clock, verifier, build, hidden-suite, and task-digest settings are
  identical between models within each task.
- Protocol v2.2 sets both the agent-idle and Daytona PTY-read safeguards to
  `3,600` seconds for every model.
- Infrastructure timeout attempts are not scored as model performance. Only
  affected model-task-trial cells are rerun.

## Audit and repairs

The historical audit found 10 cells whose result or retry path was affected by
the earlier model-specific safeguards:

- 8 cells had a `900`-second GPT Daytona PTY timeout in their retry history.
- 2 selected GPT cells ended at the earlier `600`-second idle watchdog.

All 10 cells were rerun with protocol v2.2. The final audit reports:

```text
selected cells: 180
selected idle/PTY-influenced cells: 0
historically affected cells rerun: 10
```

Two Claude Ant cells reached the shared task wall-clock limit of `5,400`
seconds. They remain valid because the same primary limit applied to every
model; they were not caused by the earlier idle/PTY difference.

## Leaderboard impact

| Model | Previous IQM | Fair IQM | Change |
| --- | ---: | ---: | ---: |
| GPT-5.5 | `10.66` | `13.87` | `+3.21` |
| GPT-5.6 Sol | `36.65` | `40.68` | `+4.03` |
| Claude Opus 4.8 | `62.03` | `62.03` | `0.00` |
| GPT-5.4 Mini | `-17.41` | `-17.41` | `0.00` |

The ranking order did not change.

The machine-readable audit is
[`leaderboard/article_suite_timeout_fairness_audit.json`](../leaderboard/article_suite_timeout_fairness_audit.json).
Reproduce it with:

```bash
uv run python scripts/audit_article_suite_timeout_fairness.py
```
