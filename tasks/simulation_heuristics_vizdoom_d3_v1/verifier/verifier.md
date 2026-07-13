---
document_version: "0.3"
verifier:
  name: simulation_heuristics_vizdoom_d3_v1
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

Run the submitted D3 screen-CV policy on hidden EnvPool seed batches. Only the
screen and five allowlisted public game variables cross the trusted policy
boundary.

BenchFlow's canonical reward is the locally normalized GenesisBench score
clamped to `[0, 1]`. Raw combat returns, anchor scores, invalid-policy details,
and per-suite metrics remain available in `genesis-score.json`.

Each candidate suite runs in a fresh subprocess and unique working directory.
This prevents EnvPool `1.1.1` from reusing its process-global D3 runtime or
colliding on `./_vizdoom`. Checked-in numeric anchors include exact config and
per-suite calibration metadata; mismatched calibration fails closed.
