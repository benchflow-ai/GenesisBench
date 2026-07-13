#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
policy=/app/final_policy/policy.py

if [ ! -f "$policy" ]; then
  printf '0.0\n' > /logs/verifier/reward.txt
  printf '{"reward": 0.0}\n' > /logs/verifier/reward.json
  exit 0
fi

python /verifier/evaluate_hidden.py \
  "$policy" \
  --config /verifier/config.toml \
  --anchors /verifier/anchors.json \
  --output /logs/verifier/genesis-score.json

python - <<'PY'
import json
from pathlib import Path

score = json.loads(Path("/logs/verifier/genesis-score.json").read_text())
reward = max(0.0, min(1.0, float(score["normalized_score"]) / 100.0))
Path("/logs/verifier/reward.txt").write_text(f"{reward}\n")
Path("/logs/verifier/reward.json").write_text(
    json.dumps({"reward": reward}, indent=2) + "\n"
)
PY
