# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.

Define `Policy` or `make_policy`:

```python
class Policy:
    def __init__(self):
        ...

    def reset(self) -> None:
        ...

    def act(self, frame, variables) -> int:
        ...
```

The evaluator creates one policy instance per active EnvPool lane and calls
`reset()` before the first action.

## Frame

`frame` is an immutable, C-contiguous NumPy `uint8` array:

```text
shape = (render_height, render_width, 3)
public default = (180, 240, 3)
```

It is the RGB image returned by EnvPool `render()`. Copy it before attempting
in-place edits.

## Variables

`variables` is an immutable mapping with exactly one key:

```text
HEALTH: float
```

No reward, seed, lane id, map state, labels, or object coordinates are
included.

## Actions

Return a scalar integer:

| Value | Action |
| ---: | --- |
| 0 | no-op |
| 1 | turn right |
| 2 | turn left |
| 3 | move forward |
| 4 | move forward and right |
| 5 | move forward and left |

Non-integral, out-of-range, or non-finite actions invalidate the episode.

