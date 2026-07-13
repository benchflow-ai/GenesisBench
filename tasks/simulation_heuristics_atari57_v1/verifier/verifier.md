---
document_version: "0.3"
verifier:
  name: simulation_heuristics_atari57_v1
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success:
        weight: 1.0
        source: deterministic
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/genesis-score.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## Verifier intent

Resolve the aggregate artifact into 342 repeat-specific policy slots. If all
342 search records are complete, run each policy on its corresponding hidden
seed, compute article-style HNS aggregation, and normalize the strict median
against numeric HNS anchors. Incomplete artifacts receive zero before EnvPool
starts.

The full details file retains raw HNS, best-single-run diagnostics, per-game
metrics, evaluation-step counts, and the 342-search interaction ledger.
