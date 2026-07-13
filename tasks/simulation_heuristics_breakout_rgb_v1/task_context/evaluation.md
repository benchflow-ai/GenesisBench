# Evaluation

The environment is EnvPool `1.1.1` `Breakout-v5` with the article's settings:

- raw channel-first `3 x 210 x 160` RGB observations;
- the reduced four-action set;
- `frameskip=1`;
- `noop_max=1`;
- `use_fire_reset=True`;
- `episodic_life=False`;
- `repeat_action_probability=0`;
- `reward_clip=False`;
- a final horizon of 30,000 policy actions.

The public evaluator reports mean/minimum/maximum return, 864-point completion
rate, invalid-policy rate, action latency, and per-episode details.

The checked-in final suite combines a nominal start with a shifted initial
paddle position. The zero-reward prefix changes action timing before policy
control begins and penalizes fixed single-trajectory replays while preserving
the pixel-only observation contract.

Official hosted evaluation may inject a private config with different seeds
and prefixes. Invalid actions receive the configured failure return.
