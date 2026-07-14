# Article-suite scoring methodology

GenesisBench separates per-task normalization from cross-task aggregation.

## Per-task normalized score

Every task evaluates a candidate on its hidden suite and compares the resulting
raw score with two frozen anchors:

```text
normalized_task_score =
    100 * (candidate_score - starter_score)
        / (reference_score - starter_score)
```

The public starter maps to `0`; the trusted article-level reference maps to
`100`. Scores are intentionally unbounded:

- negative scores perform below the starter;
- scores above `100` outperform the reference.

This follows the same family of baseline normalization used by benchmarks such
as [D4RL](https://arxiv.org/abs/2004.07219), which normalizes returns between
task-specific lower and upper reference scores.

## Five-trial estimator

Each model runs the complete nine-task suite five independent times. For task
`t`, the published task score is the arithmetic mean of its five normalized
trial scores. The task leaderboard also publishes their sample standard
deviation.

## Final normalized score

Within each trial, the primary cross-task metric is the interquartile mean
(IQM), implemented as the 25% trimmed mean used by
[RLiable](https://github.com/google-research/rliable):

```text
scores = sort(the nine normalized task scores for one trial)
trim_count = floor(0.25 * 9) = 2
trial_iqm = mean(scores[2:7])
```

The final model score is:

```text
final_normalized_score = mean(trial_iqm[1:5])
```

The leaderboard also publishes the sample standard deviation of the five
trial-level IQMs.

The JSON also publishes:

- `arithmetic_mean_normalized_score`;
- `median_normalized_score`;
- the original `average_normalized_score` as a backward-compatible alias for
  the arithmetic mean.

## Why IQM is primary

The original GenesisBench aggregate was the unweighted arithmetic mean of nine
normalized task scores. That is simple and remains useful, but a single very
large positive or negative task can move the final rank substantially.

The NeurIPS paper
[*Deep Reinforcement Learning at the Edge of the Statistical
Precipice*](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
recommends robust aggregate metrics such as IQM, together with performance
profiles and uncertainty estimates. Atari research also commonly reports both
median and mean human-normalized scores; the
[Agent57 paper](https://arxiv.org/abs/2003.13350) explicitly notes that a high
average can hide weak performance on many individual games.

GenesisBench therefore uses:

1. task-specific native raw-score leaderboards for visibility;
2. IQM as the primary cross-task rank;
3. arithmetic mean and median as secondary diagnostics.

## Positive display index

The final image uses a fixed additive transform:

```text
positive_display_score = final_normalized_score + 100
```

This is presentation-only. Raw IQM remains the official ranking field.

A fixed offset is preferred over cohort min-max scaling because it:

- preserves every model-to-model difference exactly;
- does not change when another model is added;
- gives a stable interpretation: an aggregate starter-level IQM of `0`
  displays as `100`.

Min-max scaling would force the current best and worst models to arbitrary
endpoints and would rewrite every displayed score whenever the comparison set
changes. Clipping negative IQM values to zero would erase meaningful
differences.

## Statistical limitation

Five trials expose run-to-run variability and are materially stronger than a
single run, but they remain a small sample. The leaderboard reports sample
standard deviation rather than claiming narrow confidence intervals. A larger
future release can add stratified-bootstrap confidence intervals, performance
profiles, and pairwise probability of improvement following RLiable.
