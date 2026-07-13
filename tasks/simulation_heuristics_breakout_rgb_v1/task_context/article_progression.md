# Article progression

The RGB package represents transfer of the already-developed geometry
controller to pixels:

| Node | Score | Mechanism |
| --- | ---: | --- |
| Starter | 310 | Pixel segmentation with ordinary chase lookahead of six |
| Intermediate | 428 | Increase ordinary chase lookahead from six to eight |
| Reference | 864 | Apply stuck-offset release and paddle-lag compensation throughout |

The checked-in starter is the `310` node and the trusted reference is the
`864` node. The intermediate `428` behavior is regression-tested by changing
only `CHASE_LEAD_STEPS` from `6.0` to `8.0` in the starter policy.

All three nodes receive channel-first RGB pixels only. Brick changes, ball
velocity, and no-progress duration are inferred from successive frames; RAM,
reward, and `info` are never passed to the policy.
