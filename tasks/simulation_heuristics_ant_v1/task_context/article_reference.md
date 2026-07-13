# Learning Beyond Gradients Ant reference

The trusted reference adapts the final `mpc` configuration from
`mujoco/ant/heuristic_ant.py` at revision
`3555c2956c257d49a5015b782cbe485b14fd659e`.

Its controller is composed of:

- an asymmetric four-leg CPG with second- and third-order harmonics;
- PD tracking plus roll, pitch, yaw, and angular-rate feedback;
- phase increment and stance duty adapted to measured forward velocity;
- a 10-step copied-MuJoCo planning horizon;
- 96 sampled residual-action plans per external step;
- temporally smoothed residual noise;
- warm-start plan shifting with decay `0.504186948858276`;
- forward, control, posture, yaw, height, health, and terminal joint-velocity
  terms in the planning objective.

The article's EnvPool command used five episodes from seed `0` and reported:

```text
mean 6005.521
min  5776.805
max  6146.208
```

The policy's planning RNG is reset to seed `12` every episode. Environment
reset seeds remain controlled independently by GenesisBench.

The source controller and XML are Apache-2.0 licensed by Garena Online Private
Limited. Machine-readable hashes, licensing, target metrics, and platform
reproduction results live in `evidence/source_provenance.json`, which is
excluded from the public agent workspace.
