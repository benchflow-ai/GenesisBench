#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
artifact=/app/final_artifact
submission=/app/final_artifact
integrity_report=/logs/verifier/integrity.json

if [ ! -f "$artifact/manifest.json" ]; then
  printf '0.0\n' > /logs/verifier/reward.txt
  printf '{"reward": 0.0}\n' > /logs/verifier/reward.json
  exit 0
fi

set +e
python -m genesisbench.integrity \
  --submission "$submission" \
  --trajectory /logs/agent/acp_trajectory.jsonl \
  --config /verifier/integrity.json \
  --output "$integrity_report"
integrity_status=$?
set -e
if [ "$integrity_status" -ne 0 ]; then
  cp -R "$submission" /logs/verifier/final_artifact
  printf '0.0\n' > /logs/verifier/reward.txt
  python - <<'PY'
import json
from pathlib import Path

Path("/logs/verifier/reward.json").write_text(
    json.dumps({"reward": 0.0}, indent=2) + "\n"
)
PY
  exit 0
fi

export GENESISBENCH_POLICY_ISOLATION=required
python /verifier/evaluate_hidden.py \
  "$artifact" \
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

cp -R "$submission" /logs/verifier/final_artifact
