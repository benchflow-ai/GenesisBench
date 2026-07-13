# Aggregate policy API

The artifact manifest resolves one policy module for each of the 342
`env_id / obs_mode / repeat_index` slots. Multiple slots may share a module,
but the evaluator instantiates the resolved module separately for every repeat.

A module defines `Policy` or `make_policy`. The evaluator supplies any accepted
arguments from:

```python
env_id: str
obs_mode: str
repeat_index: int
action_count: int
seed: int
config: dict
```

The resulting object implements:

```python
class Policy:
    def reset(self, seed: int = 0) -> None:
        ...

    def act(self, observation, info=None) -> int:
        ...
```

The action must be one integer in `[0, action_count)`.

For `native_obs`, `info` is always `None`. For `ram`, `info` is a dictionary
containing only `ram` when EnvPool exposes it. Batch dimension one is removed
from both observations and RAM before the policy call.

Policies may keep recurrent state and load files below `final_artifact/`.
They must run offline in the task image.
