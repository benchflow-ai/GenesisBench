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

`observation` is a one-dimensional NumPy array with 27 values:

```text
qpos[2:]  # 13 values
qvel[:]   # 14 values
```

The first two global x/y positions are omitted. The eight actions correspond
to the eight Ant hinge actuators and must have shape `(8,)`.

The policy may keep recurrent state and may load files stored below
`final_policy/`.

