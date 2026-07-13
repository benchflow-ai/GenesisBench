# Evaluation

Per-step reward:

```text
reward = x_velocity - 0.1 * sum(action ** 2)
```

Standard `HalfCheetah-v5` runs for 1,000 steps without health termination.
The public evaluator reports:

- mean, minimum, and maximum return;
- mean final x-position;
- mean accumulated forward reward and control cost;
- invalid-policy rate;
- mean action latency;
- per-episode metrics.

Actions are clipped to `[-1, 1]` and quantized to `float32`, matching the
published experiment's execution path.

## Public and hidden suites

The public development defaults are three 300-step episodes on seeds
`100..102`. Use `--max-steps 1000` for full episodes.

The checked-in hidden reproducibility suite runs full 1,000-step episodes on
unseen nominal seeds and on conservative mass, friction, damping, and actuator
variants. Its raw score is:

```text
0.70 * hidden nominal mean return
+ 0.30 * hidden dynamics-robustness mean return
```

The score is normalized on the evaluation machine so the public asymmetric
CPG/PD starter maps to `0` and the trusted staged-tree reference maps to `100`.
Scores above `100` are valid.

The checked-in suite is reproducible, not secret. An official leaderboard
should inject a private config and matching private anchors.

## Planning cost

Calls made by a policy inside its own copied MuJoCo model are internal planning
rollouts. They do not increment the external episode-step count. They do count
against wall-clock limits and are reflected in mean action latency.

The trusted article controller evaluates hundreds of first actions plus a
second tree level at every external step, so a full five-seed reproduction can
take tens of minutes on a CPU.
