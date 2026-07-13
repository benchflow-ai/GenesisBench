---
document_version: "0.3"
verifier:
  name: simulation_heuristics_halfcheetah_v1
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

Run the submitted HalfCheetah policy on full-horizon hidden nominal seeds and
conservative hidden dynamics variants. BenchFlow's canonical reward is the
normalized GenesisBench score clamped to `[0, 1]`; full raw and normalized
metrics remain available in `genesis-score.json`.

Normalization evaluates trusted local copies of the starter and staged-tree
reference on the same platform. Byte-identical policies are evaluated once per
verifier invocation so the oracle does not pay the reference MPC cost twice.
