# Evaluation

The runtime is pinned to EnvPool `1.1.1` with:

```text
MontezumaRevenge-v5
native 210x160 RGB
frame_skip = 1
noop_max = 1
use_fire_reset = true
episodic_life = false
reward_clip = false
repeat_action_probability = 0
full_action_space = false
```

The boundary target is the first `400` game points. Per-episode scoring clips
the native return into `[0, 400]`; reaching more than `400` is useful research
evidence but does not raise this v1 boundary score.

The checked-in final suite combines:

- `25%` direct reproduction from reset;
- `10%` unseen deterministic seeds;
- `10%` startup no-op timing variants;
- `55%` fresh-policy recovery from intermediate trajectory checkpoints.

For a recovery episode, a trusted policy first advances the environment to a
checkpoint. Optional no-ops are then applied. A new submitted policy instance
receives only the resulting RGB frame and must continue. Reward earned before
the checked-in checkpoints is zero.

The public evaluator can exercise the same handoff mechanism using the
candidate policy itself as the trusted prelude:

```bash
python evaluate.py \
  --policy final_policy/policy.py \
  --bootstrap-steps 512 \
  --initial-noops 4
```

Official leaderboard evaluation can inject private seeds, checkpoints, no-op
counts, and matching platform-local anchors.
