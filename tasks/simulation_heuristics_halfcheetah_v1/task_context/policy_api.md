# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.

The final verifier runs the policy in an isolated process and passes
`seed=0` to constructors and `reset`. Environment reset seeds remain private;
policies must adapt from observations rather than branch on hidden seed IDs.

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

`observation` is a one-dimensional NumPy array with 17 values:

```text
qpos[1:]  # 8 values: root z, torso pitch, six joint angles
qvel[:]   # 9 values: root x/z velocity, pitch rate, six joint velocities
```

The global x-position is omitted. The six actions are ordered:

```text
bthigh, bshin, bfoot, fthigh, fshin, ffoot
```

Each action must have shape `(6,)`. The policy may keep recurrent state and
may load files stored below `final_policy/`.

## Optional simulator configuration

Model-based policies may define:

```python
def configure_simulator(
    self,
    *,
    model_xml_path: str,
    frame_skip: int,
) -> None:
    ...
```

Call order is:

```text
instantiate policy
configure_simulator(...)  # when implemented
reset(seed=...)
act(observation) repeated for the episode
```

`model_xml_path` identifies the dynamics used by that episode, including a
hidden robustness variant when applicable. The policy receives no live
`gymnasium.Env`, `MjData`, reward, or `info` object.

HalfCheetah's omitted x-position is dynamically translation-invariant, so a
planner can reconstruct a local `MjData` state by setting x to any convenient
origin and copying the 17 observed values into the remaining qpos/qvel fields.
