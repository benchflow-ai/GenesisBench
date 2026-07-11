# Security Policy

## Reporting a vulnerability

Do not open a public issue for vulnerabilities involving credentials, sandbox
escape, evaluator leakage, arbitrary code execution, or access to hidden
benchmark assets.

Use GitHub's private vulnerability reporting for this repository. Include:

- affected revision;
- reproduction steps;
- expected impact;
- any suggested mitigation.

## Secrets

Never commit provider keys, OAuth tokens, private evaluator configurations, or
raw agent workspaces containing credentials. Use `.env.example` as the public
configuration reference.

## Benchmark integrity

Evaluator tampering, hidden-suite access, resource-meter bypass, and
answer-source contamination are treated as security and benchmark-validity
issues.
