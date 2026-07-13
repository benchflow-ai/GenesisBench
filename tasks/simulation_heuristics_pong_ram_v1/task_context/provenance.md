# Provenance and licensing

## Source experiment

This task packages the Pong 21 reproduction described in:

```text
Jiayi Weng, Learning Beyond Gradients, 2026
https://github.com/Trinkle23897/learning-beyond-gradients
```

The implementation was derived from the source artifact:

```text
atari/pong/heuristic_pong.py
commit 3555c2956c257d49a5015b782cbe485b14fd659e
```

That artifact documents the expected seed-0 result:

```text
episode score = 21.0
mean score = 21.000
```

## Reimplementation boundary

GenesisBench does not copy the source script wholesale. It reimplements:

- the policy-only `Policy.reset` / `Policy.act` contract;
- the trusted RAM-only evaluator;
- action and error validation;
- multi-seed reset-robustness evaluation;
- local starter/reference normalization;
- BenchFlow task, verifier, Docker, and oracle packaging.

The reference preserves the source experiment's observable controller
semantics: RAM coordinate decoding, recurrent velocity estimation, reflected
intercept prediction, paddle deadband, and a small return-angle bias.

No source videos, images, reports, trial records, or ROM files are copied into
the repository.

## License notice

The source policy file carries:

```text
Copyright 2021 Garena Online Private Limited
Licensed under the Apache License, Version 2.0
```

The starter, oracle, and trusted anchor policy files retain that notice and
identify the exact upstream artifact and commit. The surrounding GenesisBench
task packaging is authored for this repository.
