# GenesisBench Learning Beyond Gradients Article Suite

The first image contains the nine independently ranked task leaderboards. The second image contains the final cross-task ranking.

The nine task panels use each environment's native raw score. The final scientific metric remains unbounded IQM.

## Nine task leaderboards

![Nine task-specific GenesisBench leaderboards](article_suite_task_leaderboards.png)

## Final normalized score

The primary score is the interquartile mean (IQM): sort the nine task scores, remove the lowest two and highest two, then average the middle five. The image uses a plot-only positive display index equal to `IQM + 100`; raw IQM, arithmetic mean, and median remain in the JSON.

![Final GenesisBench article-suite leaderboard](article_suite_final_leaderboard.png)

Machine-readable rankings and score-detail paths are available in [`article_suite.json`](article_suite.json). The scoring rationale is documented in [`docs/article-suite-scoring.md`](../docs/article-suite-scoring.md).
