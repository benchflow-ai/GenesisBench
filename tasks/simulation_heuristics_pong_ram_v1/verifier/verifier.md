---
document_version: "0.3"
verifier:
  name: simulation_heuristics_pong_ram_v1
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

## verifier intent

Run the submitted Atari Pong RAM policy on unseen nominal seeds and unseen
randomized-reset seeds while preserving the article's frame-skip-1,
unclipped-reward, zero-sticky-action semantics.

BenchFlow's canonical reward is the locally normalized GenesisBench score
clamped to `[0, 1]`. Full native Pong scores, point totals, errors, suite
weights, and unclamped normalization remain in `genesis-score.json`.
