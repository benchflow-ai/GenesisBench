# Evaluation

Per-step reward:

```text
reward = x_velocity + 1.0 - 0.5 * sum(action ** 2)
```

The healthy reward is received while the torso state is finite and torso
height is within `[0.2, 1.0]`.

The public evaluator reports:

- mean, minimum, and maximum return;
- mean final x-position;
- fall rate;
- invalid-policy rate;
- mean action latency;
- per-episode metrics.

The final leaderboard score is the mean return over hidden nominal episodes
and hidden conservative dynamics variants. Invalid policies receive a fixed
failure return for the affected episode.

