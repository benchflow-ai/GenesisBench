---
schema_version: "1.3"
task:
  name: genesisbench/simulation_heuristics_atari57_v1
  description: Improve one aggregate 342-policy Atari57 submission produced by 342 independently accounted heuristic-search trajectories.
  authors:
    - name: GenesisBench Contributors
  keywords:
    - simulation
    - heuristics
    - atari57
    - envpool
    - batch-experiment
metadata:
  category: atari
  difficulty: hard
  tags:
    - atari57
    - programmatic-policy
    - aggregate-evaluation
    - sample-efficiency
    - batch-search
  reference_task: false
  genesisbench:
    starter:
      path: starter_artifact
    submission:
      directory: final_artifact
      entrypoint: manifest.json
    development:
      games:
        - Breakout-v5
        - Freeway-v5
        - MontezumaRevenge-v5
        - Pong-v5
        - Seaquest-v5
        - Skiing-v5
      observation_modes:
        - ram
        - native_obs
      repeats: 3
      seeds: [101, 202, 303]
      max_steps: 5000
    verifier:
      reproduction_config: verifier/config.toml
      anchors: verifier/anchors.json
      supports_private_config: true
agent:
  timeout_sec: 1814400
  user: agent
  network_mode: no-network
verifier:
  timeout_sec: 172800
  user: root
  network_mode: no-network
  hardening:
    cleanup_conftests: true
environment:
  build_timeout_sec: 1800
  cpus: 14
  memory_mb: 8192
  storage_mb: 102400
  network_mode: no-network
  allow_internet: false
  workdir: /app
benchflow:
  document_version: "0.6"
---
# Scale heuristic search across Atari57

Improve a single aggregate submission containing one programmatic policy slot
for every Atari57 game, observation mode, and independent repeat. This is the
batch/scaling experiment. Do not split it into standalone tasks.

## Article protocol

The experiment matrix is:

```text
57 games x 2 observation modes x 3 repeats = 342 search trajectories
```

Each search trajectory has a target of `20,000,000` Atari frames. Because
`frame_skip=1`, one environment step is one counted frame. The complete target
across the matrix is:

```text
342 x 20,000,000 = 6,840,000,000 counted environment steps
```

Every environment call used for probing, debugging, tuning, comparison, or
final validation must be included in the corresponding ledger record. Never
invent completed searches or interaction counts. A partial submission remains
evaluable with the public development tool, but the official verifier assigns
it zero rather than treating it as an article-faithful reproduction.

The three repeats must be independent search trajectories. Use `repeat_index`
as a seed offset or other decorrelation mechanism rather than replaying the
same search three times.

## Fixed Atari environment

All search and evaluation environments use EnvPool `1.1.1` and these exact
settings:

```python
envpool.make_gym(
    ENV_ID,
    num_envs=1,
    batch_size=1,
    seed=seed,
    img_height=210,
    img_width=160,
    stack_num=1,
    gray_scale=False,
    frame_skip=1,
    noop_max=1,
    use_fire_reset=True,
    episodic_life=False,
    reward_clip=False,
    repeat_action_probability=0.0,
    full_action_space=False,
)
```

In particular, `frame_skip=1`, `reward_clip=false`, and sticky action is
disabled with `repeat_action_probability=0.0`.

## Observation modes

- `native_obs`: the policy may use only the observation returned by
  `reset`/`step` plus its own internal state. The evaluator passes no `info`.
- `ram`: the policy may additionally use the public `info["ram"]` array. The
  evaluator removes every other `info` field before calling the policy.

Do not read EnvPool implementation files, tests, ROM internals, emulator state,
or hidden evaluator data. Do not use `render()` as an extra policy input. Do
not train a neural network or copy completed Atari policies from the internet.

## Starting artifact

`starter_artifact/` is a valid aggregate artifact. Its no-op policy is shared
by all 342 game/mode/repeat slots, and its sparse interaction ledger honestly
reports zero completed searches.

The manifest supports a shared `policy_defaults` entry plus per-slot overrides.
You may share code between games or use a different module for every search.
The resolved artifact must still cover every tuple in:

```text
Atari57 games x {ram, native_obs} x {0, 1, 2}
```

## Public development evaluation

First validate the aggregate artifact without requiring EnvPool:

```bash
python evaluate.py --artifact final_artifact --validate-only
```

Run the representative public subset:

```bash
python evaluate.py --artifact final_artifact
```

The public subset evaluates all three repeat-specific policy slots on public
seeds for dense reward, sparse reward, negative-return, visual tracking, and
resource-management games. It is iteration feedback only; it is not evidence
that the complete 57-game protocol was reproduced.

## Required final artifact

Write the aggregate submission to:

```text
final_artifact/
  manifest.json
  interaction_ledger.json
  policy.py                 # optional shared default
  policies/                 # optional per-game/mode/repeat modules
  searches/                 # evidence for completed search trajectories
```

`manifest.json` must retain the fixed protocol block. `policy_defaults` defines
the fallback module/configuration. Entries in `policies` override a specific
`env_id`, `obs_mode`, and `repeat_index`.

Each selected module must define `Policy` or `make_policy`. A policy receives
`env_id`, `obs_mode`, `repeat_index`, `action_count`, `seed`, and its manifest
`config` when those arguments are accepted. Its action API is:

```python
class Policy:
    def reset(self, seed: int = 0) -> None: ...
    def act(self, observation, info=None) -> int: ...
```

Return exactly one integer in `[0, action_count)`.

`interaction_ledger.json` contains sparse records keyed by
`env_id / obs_mode / repeat_index`. Omitted entries count as `not_run`.
Use `status="complete"` only after counting at least `20,000,000` environment
steps for that search trajectory and retaining an `evidence_path` directory
with `policy.py`, `trials.jsonl`, `summary.csv`, `sample_efficiency.png`, and
`README.md`. Each record's best policy remains a separate repeat-specific
manifest slot. Do not collapse the three independent searches into one policy.

## Final scoring

For each evaluated episode:

```text
HNS = (episode_return - random_score) / (human_score - random_score)
```

The aggregate follows the article's strict best-input-mean convention:

1. mean HNS over the three repeats for each game and observation mode;
2. for each game, select the higher of the two mode means;
3. take the median across all 57 games.

The verifier also reports mean HNS and the looser best-single-run median, but
the primary score is the strict median above. Natural HNS `0.0` maps to
normalized score `0`; the article's reported best-input-mean median HNS
`0.8283015254994576` maps to `100`. Values above `100` are valid.

For context, the source article reported a final best-input-mean median HNS of
`0.8283015254994576` and a best-single-run median HNS of
`1.1813031161473089`. The first is intentionally the numeric normalization
reference. Neither value is a claim made by the starter or oracle artifacts.

## Finish criteria

Before exiting:

1. run `--validate-only`;
2. run as much public evaluation as the available runtime permits;
3. leave the best working aggregate artifact in `final_artifact/`;
4. keep every interaction count honest;
5. document incomplete searches as incomplete rather than claiming a full
   6.84B-frame reproduction.
