---
document_version: "0.3"
verifier:
  name: simulation_heuristics_ant_v1
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

Run the submitted Ant policy on hidden nominal seeds and conservative hidden
dynamics variants. BenchFlow's canonical reward is the normalized GenesisBench
score clamped to `[0, 1]`; the full raw and normalized metrics remain available
in `genesis-score.json`.

