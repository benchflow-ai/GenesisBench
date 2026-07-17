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

`observation` is a one-dimensional NumPy array with 27 values:

```text
qpos[2:]  # 13 values
qvel[:]   # 14 values
```

The first two global x/y positions are omitted. The eight actions correspond
to the eight Ant hinge actuators and must have shape `(8,)`.

The policy may keep recurrent state and may load files stored below
`final_policy/`.

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
reset the environment
instantiate policy
configure_simulator(...)  # when implemented
reset(seed=...)
act(observation) repeated for the episode
```

`model_xml_path` is a private per-episode XML copy matching the current
episode, including a conservative hidden dynamics variant when applicable.
Policies may load that model into their own `MjModel`/`MjData` for planning.

The policy never receives the live `gymnasium.Env`, its mutable `MjData`,
reward, `info`, or hidden suite configuration. The copied model therefore
enables MPC without allowing the policy to step or mutate the scored
environment.
