# Article progression

The packaged policies preserve the RAM experiment's reported control path:

| Node | Score | Mechanism |
| --- | ---: | --- |
| Starter | 387 | Reflected interception with no trajectory perturbation |
| Intermediate | 507 | Resettable stuck-loop offset cycling |
| Intermediate | 839 | Fast-low-ball lead of three frames |
| Reference | 864 | Post-432 offset release plus two-pixel paddle-lag compensation |

The checked-in starter is the `387` node and the trusted reference is the
`864` node. Both are observation-only adaptations: reward progress and lives
needed by the controller are inferred from the 128 RAM bytes rather than
passed as side channels.
