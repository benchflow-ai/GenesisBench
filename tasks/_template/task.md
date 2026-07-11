---
schema_version: "1.3"
task:
  name: genesisbench/{{task_name}}
  description: {{task_title_yaml}}
  authors:
    - name: {{author_yaml}}
  keywords:
    - robotics
metadata:
  category: TODO
  difficulty: medium
  tags:
    - robotics
  reference_task: false
  genesisbench:
    starter:
      path: starter_artifact
    submission:
      directory: final_artifact
      entrypoint: artifact.py
    development:
      episodes: 1
      max_steps: 1
      seeds: [0]
    verifier:
      reproduction_config: verifier/config.toml
      supports_private_config: true
agent:
  timeout_sec: 1800
verifier:
  timeout_sec: 300
environment:
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
  workdir: /app
benchflow:
  document_version: "0.6"
---
# {{task_title}}

TODO: State the artifact the coding agent receives, the behavior it must
improve, the resource budget, allowed methods, prohibited information, and the
required final artifact.

The final artifact must be written to:

```text
final_artifact/artifact.py
```

The final score is computed independently after the agent exits.

