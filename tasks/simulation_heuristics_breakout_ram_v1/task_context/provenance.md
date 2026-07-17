# Provenance and licensing

This task reimplements the RAM Breakout experiment described in **Learning
Beyond Gradients**.

- Reported progression: `387 -> 507 -> 839 -> 864`

Exact upstream repository paths, revisions, and answer-file hashes are retained
outside the agent image in the repository's maintainer provenance records.

The starter and reference policies adapt the published geometric controller
structure and retain its `Copyright 2021 Garena Online Private Limited` and
Apache License 2.0 notice. GenesisBench includes the Apache 2.0 text at
`LICENSES/Apache-2.0.txt`.

No Atari ROM file is copied into this repository. The task image pins
`envpool==1.1.1`, the runtime used by the article, and relies on the Atari
assets distributed with that EnvPool wheel.
