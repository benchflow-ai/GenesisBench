# Aggregate artifact contract

## Layout

```text
final_artifact/
  manifest.json
  interaction_ledger.json
  policy.py
  policies/
```

Only the manifest and ledger are mandatory filenames. Policy modules may use
any safe relative path below the artifact root.

## Manifest

The fixed `protocol` block identifies EnvPool `1.1.1`, the two observation
modes, three search repeats, the 20M-frame per-search target, and the exact
Atari wrapper settings.

`policy_defaults` contains a default `module` and `config`. `policies` is a
sparse override array:

```json
{
  "env_id": "Breakout-v5",
  "obs_mode": "native_obs",
  "repeat_index": 1,
  "module": "policies/breakout_native_repeat_1.py",
  "config": {"threshold": 17}
}
```

The evaluator expands defaults plus overrides into exactly 342
`env_id / obs_mode / repeat_index` slots and rejects unknown games, modes,
repeat indices, duplicate overrides, unsafe paths, or missing modules. Shared
defaults are allowed, but repeat identity is never discarded.

## Interaction ledger

The ledger is sparse. Each explicit record has:

```json
{
  "env_id": "Breakout-v5",
  "obs_mode": "native_obs",
  "repeat_index": 0,
  "cumulative_env_steps": 20001234,
  "cumulative_episodes": 731,
  "status": "complete"
}
```

Allowed statuses are `not_run`, `running`, `complete`, and `failed`. Omitted
matrix entries are treated as `not_run`. `complete` requires at least
20,000,000 counted environment steps and an `evidence_path` containing the
article outputs:

```text
policy.py
trials.jsonl
summary.csv
sample_efficiency.png
README.md
```

The ledger is an auditable claim, not a mechanism that can prove no step was
omitted.

The checked-in starter and reference keep an empty ledger. That is deliberate:
they validate the software path without claiming article-scale search. The full
verifier disqualifies both before EnvPool execution.
