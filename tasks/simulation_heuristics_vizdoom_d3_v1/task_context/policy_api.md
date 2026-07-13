# Policy API

The evaluator imports `final_policy/policy.py` in a clean Python process.

Define `Policy` or `make_policy`:

```python
class Policy:
    def __init__(self):
        ...

    def reset(self) -> None:
        ...

    def act(self, frame, variables):
        ...
```

One policy instance is created per active EnvPool lane and reset before the
first action.

## Frame

`frame` is an immutable, C-contiguous NumPy `uint8` array:

```text
shape = (render_height, render_width, 3)
public default = (480, 640, 3)
```

The three channels are the raw `CRCGCB` screen format configured for the
article experiment. Treat them as ordered image channels rather than assuming
standard sRGB colorimetry.

## Variables

`variables` is an immutable mapping with exactly:

```text
HEALTH: float
AMMO2: float
HITCOUNT: float
DAMAGECOUNT: float
KILLCOUNT: float
```

No reward, seed, lane id, map state, labels, or object coordinates are
included.

## Actions

Return a NumPy-compatible vector with shape `(8,)`:

| Index | Channel | Valid range |
| ---: | --- | --- |
| 0 | attack | `[0, 1]` |
| 1 | speed | `[0, 1]` |
| 2 | move forward | `[0, 1]` |
| 3 | move backward | `[0, 1]` |
| 4 | move right | `[0, 1]` |
| 5 | move left | `[0, 1]` |
| 6 | turn 180 degrees | `[0, 1]` |
| 7 | horizontal turn delta | `[-12, 12]` |

Non-finite, out-of-range, or incorrectly shaped actions invalidate the
episode.

