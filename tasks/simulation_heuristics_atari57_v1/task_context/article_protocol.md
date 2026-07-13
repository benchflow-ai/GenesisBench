# Source article protocol

The supplied source artifacts define one unattended heuristic-search run for
each:

```text
57 Atari games x {ram, native_obs} x 3 repeats
```

That is 342 coding-agent search trajectories.

Each trajectory:

- uses EnvPool `1.1.1`;
- targets 20,000,000 counted frames;
- counts every probe, debug rollout, and trial;
- keeps `frame_skip=1`, `reward_clip=False`, and sticky action probability
  `0.0`;
- forbids neural-network training and environment-source/hidden-state access;
- writes a best policy and experiment history;
- simplifies a new best policy and rechecks it before retaining it.

The article reported these checkpoints from its aggregate curve:

| Mode | Counted steps | Median HNS |
| --- | ---: | ---: |
| `native_obs` | 988,645 | 0.31874552826138824 |
| `ram` | 988,645 | 0.25770816471064345 |
| `native_obs` | 9,746,987 | 0.8079186493157826 |
| `ram` | 9,746,987 | 0.5914131823634771 |

Its final cross-game summaries were:

| Method | Mean HNS | Median HNS | Games with HNS >= 1 |
| --- | ---: | ---: | ---: |
| OpenAI Baselines PPO2 | 4.54578947368421 | 0.8 | 28 |
| CleanRL EnvPool PPO | 6.841228070175438 | 0.98 | 28 |
| Codex best input mean | 2.3427762956086453 | 0.8283015254994576 | 26 |
| Codex best single run | 3.8967649230310704 | 1.1813031161473089 | 36 |

“Best single run” is intentionally reported as a looser diagnostic. The
GenesisBench primary metric uses best-input mean.

These are historical source measurements. They are not embedded as evaluator
answers, and the deterministic smoke tests do not claim to reproduce them.
