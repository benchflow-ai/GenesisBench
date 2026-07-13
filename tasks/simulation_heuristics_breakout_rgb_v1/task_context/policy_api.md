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

`observation` is exactly one channel-first `uint8` RGB frame from EnvPool
`1.1.1` `Breakout-v5`, with shape `(3, 210, 160)`.

The only valid outputs are scalar integer actions `0`, `1`, `2`, and `3`,
corresponding to `NOOP`, `FIRE`, `RIGHT`, and `LEFT`. The policy may keep
recurrent state and load files stored below `final_policy/`.

No RAM, reward, `info`, lives counter, object labels, emulator object, or
hidden variant metadata is passed to the policy.
