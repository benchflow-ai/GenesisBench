---
document_version: "0.3"
verifier:
  name: {{task_name}}
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
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_success: 1.0
---

## verifier intent

TODO: Describe exactly what the verifier measures and which task outputs it
reads.

