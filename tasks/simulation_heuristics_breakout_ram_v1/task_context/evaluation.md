# Evaluation

The environment is EnvPool `1.1.1` `Breakout-v5` with the article's settings:

- RAM observations;
- the reduced four-action set;
- `frameskip=1`;
- `noop_max=1`;
- `use_fire_reset=True`;
- `episodic_life=False`;
- `repeat_action_probability=0`;
- `reward_clip=False`;
- a full final horizon of 108,000 policy actions.

The public evaluator reports mean/minimum/maximum return, 864-point completion
rate, invalid-policy rate, action latency, and per-episode details.

The checked-in final suite combines a nominal start with shifted initial paddle
positions. Those zero-reward prefixes change the trajectory before policy
control begins and make a memorized action tape brittle while preserving the
same observation and reward contract.

Official hosted evaluation may inject a private config with different seeds
and prefixes. Invalid actions receive the configured failure return.
