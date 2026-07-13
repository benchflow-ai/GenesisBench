# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.
Define `Policy` or `make_policy`:

```python
class Policy:
    def __init__(self, seed: int = 0):
        ...

    def reset(self, seed: int = 0) -> None:
        ...

    def act(self, observation) -> int:
        ...
```

The policy receives exactly one native EnvPool RGB observation:

```text
shape: (3, 210, 160)
dtype: uint8
layout: channel, height, width
```

The evaluator never passes reward, `info`, RAM, lives, ALE state, room ids,
emulator objects, bootstrap length, or no-op count.

`act` returns one scalar integer in the full 18-action Atari space:

| ID | Action | ID | Action |
| ---: | --- | ---: | --- |
| 0 | NOOP | 9 | DOWNLEFT |
| 1 | FIRE | 10 | UPFIRE |
| 2 | UP | 11 | RIGHTFIRE |
| 3 | RIGHT | 12 | LEFTFIRE |
| 4 | LEFT | 13 | DOWNFIRE |
| 5 | DOWN | 14 | UPRIGHTFIRE |
| 6 | UPRIGHT | 15 | UPLEFTFIRE |
| 7 | UPLEFT | 16 | DOWNRIGHTFIRE |
| 8 | DOWNRIGHT | 17 | DOWNLEFTFIRE |

The policy may keep recurrent state and load assets stored below
`final_policy/`. Every episode, including a recovery handoff, uses a fresh
policy instance followed by `reset(seed=...)`.
