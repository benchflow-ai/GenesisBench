# Simulation Heuristics Atari57 v1

This package turns the article's Atari57 scaling experiment into one aggregate
GenesisBench task. The unit of submission is a manifest resolving 342 policy
slots, not a directory of duplicated standalone tasks.

## Contract at a glance

```text
57 games
x 2 observation modes
x 3 independent searches
= 342 search trajectories

342 x 20,000,000 frames
= 6,840,000,000-frame full-search target
```

The final evaluator runs each repeat-specific policy on its corresponding
evaluation seed, computes per-episode human-normalized score (HNS), averages
the three independent policies within each game/mode, selects the better mode
per game, and takes the median over games.

## Local commands

Validate the starter without EnvPool:

```bash
uv run python tasks/simulation_heuristics_atari57_v1/evaluate.py \
  --artifact tasks/simulation_heuristics_atari57_v1/starter_artifact \
  --validate-only
```

Run the deterministic verifier smoke suite:

```bash
uv run python \
  tasks/simulation_heuristics_atari57_v1/verifier/evaluate_hidden.py \
  tasks/simulation_heuristics_atari57_v1/starter_artifact \
  --config \
  tasks/simulation_heuristics_atari57_v1/verifier/config_smoke.toml
```

Run the real public subset inside an environment containing exactly
`envpool==1.1.1`:

```bash
python tasks/simulation_heuristics_atari57_v1/evaluate.py \
  --artifact tasks/simulation_heuristics_atari57_v1/starter_artifact
```

Validate the native package:

```bash
uv run bench tasks check \
  tasks/simulation_heuristics_atari57_v1 \
  --level publication-grade
```

Run the contract oracle through BenchFlow:

```bash
uv run bench eval run \
  --tasks-dir tasks/simulation_heuristics_atari57_v1 \
  --agent oracle \
  --sandbox docker \
  --context-root .
```

## Full runtime versus deterministic tests

The checked-in `verifier/config.toml` requires all 342 completed search records
before starting the complete 57-game × 2-mode × 3-repeat EnvPool evaluation.
Incomplete artifacts return zero immediately. `config_smoke.toml` disables
that eligibility gate only for deterministic contract tests.

The bundled seeded-random oracle/reference artifact is only a runnable software
contract artifact. Its ledger remains at zero rather than pretending that the
342 article-scale searches were run, so the full verifier disqualifies it with
reward zero. It is not the `100` anchor and must not be cited as a reproduction
of the article's measured HNS.

## Source-faithful reference metrics

The task context preserves the supplied article measurements:

- native-observation median HNS `0.31874552826138824` at `988,645` steps;
- RAM median HNS `0.25770816471064345` at `988,645` steps;
- native-observation median HNS `0.8079186493157826` at `9,746,987` steps;
- RAM median HNS `0.5914131823634771` at `9,746,987` steps;
- final best-input-mean median HNS `0.8283015254994576`, used as the numeric
  normalized-score `100` anchor;
- final best-single-run median HNS `1.1813031161473089`.

The score anchor is intentionally numeric; the remaining values are context,
not hard-coded evaluator outputs.

## Package map

| Path | Purpose |
| --- | --- |
| `task.md` | BenchFlow-native task and agent contract |
| `starter_artifact/` | Honest zero-search, 342-slot starter |
| `evaluate.py` | Representative public EnvPool evaluator |
| `task_context/` | Protocol, artifact, policy, scoring, and source metrics |
| `verifier/config.toml` | Full aggregate reproduction suite |
| `verifier/config_smoke.toml` | Deterministic lightweight contract suite |
| `verifier/evaluate_hidden.py` | Hidden aggregate HNS scorer |
| `verifier/anchors.json` | Numeric HNS `0` and article `100` anchors |
| `oracle/reference_artifact/` | Runnable non-target contract artifact |

Production deployments may inject private evaluation seeds, but the published
numeric HNS anchors and complete-ledger eligibility gate remain explicit.
