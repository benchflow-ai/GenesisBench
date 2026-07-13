# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.

Define `Policy` or `make_policy`. The policy is instantiated once per episode
and reset before the first action.

```python
class Policy:
    def __init__(self, seed: int = 0):
        ...

    def reset(self, seed: int = 0) -> None:
        ...

    def act(self, observation):
        ...
```

`observation` is the current Atari 2600 RAM image:

```text
shape: (128,)
dtype: uint8
range: 0..255
```

No frame stack is supplied. A policy may retain previous RAM observations,
velocity estimates, rally state, or other recurrent state internally.

`act` must return one finite integer accepted by EnvPool's minimal Pong action
space. The standard paddle controller uses:

```text
0 = no-op
2 = paddle up
3 = paddle down
```

Other in-range minimal actions are accepted, but the evaluator rejects arrays
with more than one value, fractional actions, NaN/infinity, and out-of-range
integers. Invalid policies receive the configured failure score for that
episode.

The policy does not receive reward, score, `info`, screenshots, emulator
objects, hidden seeds, or reset-window parameters. It may load files stored
below `final_policy/`.
