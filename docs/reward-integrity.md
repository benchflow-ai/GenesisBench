# Reward integrity

GenesisBench scores scientific or control performance only after a deterministic
integrity gate passes.

## Threat model

Submitted code is untrusted. It must not:

- access `/oracle`, `/verifier`, verifier logs, or hidden configuration;
- use network retrieval or copy a completed upstream answer;
- modify trusted evaluators or runtime modules;
- create symlinks, import hooks, compiled extensions, or special files;
- monkeypatch the verifier or execute dynamic code;
- tamper with reward artifacts;
- fabricate experiment-accounting records.

## Prevention

All article-suite tasks run the agent, environment, and verifier without
network access. OpenCode web retrieval, external-directory access, and
subagents are denied by the managed run configuration.

The task image keeps `/app/evaluate.py`, starter artifacts, task context, and
`/opt/genesisbench` root-owned and read-only. Only the submission directory and
`/app/work` are agent-writable.

The verifier never imports submitted Python into its root process. A persistent
policy worker runs as UID `agent` under Landlock with access only to:

- system libraries;
- a copied submission bundle;
- a shared observation buffer;
- an episode-specific copied MuJoCo XML when the task permits model-based
  planning.

The worker cannot read `/oracle`, `/verifier`, `/logs`, or hidden suite files.

## Detection

Before scientific scoring, `genesisbench.integrity` audits:

- the root-published ACP trajectory;
- the submitted artifact tree;
- prohibited upstream answer hashes;
- forbidden imports, paths, file types, symlinks, and dynamic-code hooks;
- untrusted Atari57 completion claims.

The report is preserved at `/logs/verifier/integrity.json`.

## Reward composition

Integrity is a binary multiplier:

```text
final_reward = scientific_reward  if integrity_pass
final_reward = 0                  otherwise
```

`reward.json` keeps BenchFlow's numeric reward-only contract. The scientific
score and violation evidence remain separately reviewable in
`genesis-score.json` and `integrity.json`, so confirmed misconduct is not
conflated with an ordinary low-performing policy.

## Required acceptance tests

Every task change must keep these gates green:

1. schema and task validation;
2. starter/reference/oracle regression tests;
3. clean submission passes integrity;
4. web retrieval, hidden-path access, source copying, symlinks, import hooks,
   reward tampering, and fabricated accounting receive reward zero;
5. Docker/Daytona policy-isolation smoke;
6. complete repository test suite and lint.
