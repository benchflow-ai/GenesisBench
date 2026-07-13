---
document_version: "0.3"
verifier:
  name: simulation_heuristics_montezuma_v1
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

Run the submitted native-image policy on direct reproduction, deterministic
hidden seeds, startup no-op variants, and fresh-policy checkpoint recovery.

Each episode is capped at the `400`-point boundary. BenchFlow's canonical
reward is the normalized GenesisBench score clamped to `[0, 1]`; full suite,
raw-score, recovery, latency, and anchor metrics remain available in
`genesis-score.json`.

The checked-in suite is reproducible. Official evaluation should inject a
private config and matching anchors so checkpoint/no-op details are not
authoring targets.
