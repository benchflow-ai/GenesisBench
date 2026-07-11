# Contributing to GenesisBench

Thank you for contributing.

## Add a benchmark task

GenesisBench welcomes robotics tasks where a coding agent can iteratively
improve a policy, controller, planner, data-generation system, or training
pipeline and receive an independently verified final score.

Start here:

1. Read `tasks/README.md`.
2. Study the complete reference task in `tasks/simulation_heuristics_ant_v1/`.
3. Scaffold a task with `scripts/create_task.py`.
4. Validate it with `scripts/validate_tasks.py`.
5. Include a real end-to-end agent canary and reproducible score evidence.

## General contribution checks

```bash
uv sync --extra dev
uv run python scripts/validate_tasks.py
uv run bench tasks check \
  tasks/simulation_heuristics_ant_v1 \
  --level publication-grade
uv run pytest -q
uv run ruff check .
```

Do not include API keys, provider credentials, private evaluator assets, or
large generated run directories in a contribution.

By participating, you agree to follow `CODE_OF_CONDUCT.md`. Report security or
benchmark-integrity vulnerabilities according to `SECURITY.md`.
