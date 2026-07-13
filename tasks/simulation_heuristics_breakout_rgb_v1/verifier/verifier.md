---
document_version: "0.3"
verifier:
  name: simulation_heuristics_breakout_rgb_v1
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

Run the submitted policy on raw RGB frames only, using a full 30,000-frame
nominal episode and a shifted-start episode. The shift moves the paddle before
the policy is instantiated, changing action timing without exposing RAM,
reward, `info`, or hidden state.

The raw score is a weighted native Breakout return. The normalized score maps
the trusted RGB starter to `0` and the trusted pixel-only 864-point reference
to `100`. BenchFlow's reward clamps normalized score divided by 100 to `[0, 1]`;
full raw, anchor, suite, completion, invalid-action, and latency metrics remain
in `genesis-score.json`.

The checked-in suite is reproducible. Hosted evaluation should inject private
seeds, prefixes, and matching anchors.
