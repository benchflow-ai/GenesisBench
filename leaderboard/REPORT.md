# GenesisBench Simulation Heuristics Ant v1 Results

## Setup

- Task: improve a provided Ant CPG/PD controller through iterative coding and
  simulator evaluation.
- Agent harness: OpenHands.
- Runtime: isolated Docker task workspace; the hidden verifier is not mounted.
- Budget: 30 wall-clock minutes per model.
- GPT reasoning: `xhigh`.
- Claude Opus 4.8 reasoning: `max`.
- Final evaluation: full 1,000-step episodes.
- Score:

```text
0.70 * hidden nominal mean return
+ 0.30 * hidden dynamics-robustness mean return
```

The normalized score maps the frozen starter controller to `0` and the frozen
higher-harmonic reference controller to `100`. Scores above `100` are allowed.
The current verifier recalibrates those trusted policies on the evaluation
platform to avoid small MuJoCo platform drift; the table records the original
published sweep.

## Leaderboard

| Rank | Model | Final score | Normalized | Nominal | Robust |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | GPT-5.6 Sol | 3417.86 | 150.24 | 3399.54 | 3460.62 |
| 2 | GPT-5.5 | 2382.23 | 0.00 | 2271.38 | 2640.87 |
| 3 | GPT-5.4 Mini | 2369.61 | -1.83 | 2296.00 | 2541.39 |
| 4 | Claude Opus 4.8 | 2235.71 | -21.26 | 2276.09 | 2141.48 |

All submitted policies had a `0%` fall rate on the hidden suite. Differences
therefore came primarily from forward velocity and control cost, not merely
surviving for more steps.

## Interpretation

- GPT-5.6 Sol produced the only submission that clearly exceeded both frozen
  anchors. Its hidden robustness return was also higher than its nominal return.
- GPT-5.5's first run discovered a much stronger development candidate but
  ended on a provider disconnect and was excluded as infrastructure failure.
  The clean retry consumed the full budget without writing a new
  `final_policy/policy.py`, so its authoritative score is exactly the starter
  score. The benchmark intentionally scores the submitted artifact rather than
  intermediate logs or self-reported experiments.
- GPT-5.4 Mini submitted a modified controller, but its hidden score was
  slightly below the starter.
- Claude Opus 4.8 submitted a modified controller that regressed more on the
  hidden dynamics suite than on the nominal suite.

## Evidence

- Machine-readable leaderboard: `leaderboard/simulation_heuristics_ant_v1.json`
- Packaged policies and scores: `leaderboard/submissions/`
- Deterministic verifier: `tasks/simulation_heuristics_ant_v1/verifier/evaluate_hidden.py`
- Run and audit commands: `experiments/simulation_heuristics_ant_v1/README.md`

Every packaged policy was re-evaluated after packaging, and all four scores
matched their recorded score exactly.

Raw OpenHands trajectories are intentionally excluded from the source
distribution because they are large and can contain provider-specific runtime
metadata. Packaged metadata retains a non-secret source run identifier.

## Limitations

- This is one agent run per model, not a multi-trial estimate of model quality.
- Opus uses OpenHands `ACPAgent` with Claude Agent ACP because its available
  credential is a Claude subscription credential. The three GPT models use the
  standard OpenHands SDK agent loop.
- The task is simulation-only and does not establish real-robot transfer.
- The hidden dynamics suite is deliberately conservative and should be expanded
  only after additional calibration and repeated agent trials.
