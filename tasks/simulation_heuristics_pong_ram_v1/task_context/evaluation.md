# Evaluation

## Native environment

The evaluator uses EnvPool `Pong-v5` with:

```text
frame_skip = 1
reward_clip = false
repeat_action_probability = 0.0
use_fire_reset = true
episodic_life = false
full_action_space = false
```

Each environment step advances one Atari frame. The full episode is capped at
27,000 policy decisions as a safety limit.

## Native score

Pong emits:

```text
+1 when the controlled right paddle wins a point
-1 when the opponent wins a point
```

The episode ends when either side reaches 21 points. Therefore:

```text
episode score = points_for - points_against
range = [-21, 21]
target = 21
```

## Public metrics

The development evaluator reports:

- mean, minimum, and maximum native score;
- win rate;
- perfect-score rate;
- invalid-policy rate;
- mean policy action latency;
- per-episode seeds, variants, point totals, lengths, and errors.

## Final suite

The reproduction verifier combines:

```text
0.70 * unseen-seed nominal mean score
+ 0.30 * unseen-seed randomized-reset mean score
```

Both suites keep the article's frame skip and zero-sticky-action semantics.
The randomized-reset suite only widens the random no-op window before a game
begins. Official leaderboard runs may inject private seeds and reset windows.

Invalid actions, import errors, constructor/reset errors, and policy exceptions
receive a native failure score of `-21` for the affected episode.

The raw score is normalized against locally rerun starter and reference
policies. Starter maps to `0`; the Pong-21 reference maps to `100`.
