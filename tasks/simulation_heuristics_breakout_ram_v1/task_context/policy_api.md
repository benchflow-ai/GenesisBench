# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.

Define `Policy` or `make_policy`. The policy is instantiated once per episode
and reset before the first scored action:

```python
class Policy:
    def __init__(self, seed: int = 0):
        ...

    def reset(self, seed: int = 0) -> None:
        ...

    def act(self, observation):
        ...
```

`observation` is exactly one EnvPool `Breakout-v5` 128-byte `uint8` RAM vector,
extracted from the environment's per-step `info["ram"]` batch.

The only valid outputs are scalar integer actions `0`, `1`, `2`, and `3`,
corresponding to `NOOP`, `FIRE`, `RIGHT`, and `LEFT`. The policy may keep
recurrent state and load files stored below `final_policy/`.

No reward, pixels, `info`, lives counter, emulator object, or hidden variant
metadata is passed to the policy.
