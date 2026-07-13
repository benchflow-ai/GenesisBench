#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
policy=/app/final_policy/policy.py

if [ ! -f "$policy" ]; then
  printf '0.0\n' > /logs/verifier/reward.txt
  printf '{"reward": 0.0}\n' > /logs/verifier/reward.json
  exit 0
fi

set +e
timeout --signal=TERM --kill-after=30s 3900s \
  python /verifier/evaluate_hidden.py \
  "$policy" \
  --config /verifier/config.toml \
  --anchors /verifier/anchors.json \
  --output /logs/verifier/genesis-score.json
evaluation_status=$?
set -e

if [ "$evaluation_status" -eq 124 ] || [ "$evaluation_status" -eq 137 ]; then
  python - <<'PY'
import json
from pathlib import Path

Path("/logs/verifier/genesis-score.json").write_text(
    json.dumps(
        {
            "score": None,
            "normalized_score": 0.0,
            "verifier_timeout": True,
            "timeout_seconds": 3900,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY
elif [ "$evaluation_status" -ne 0 ]; then
  exit "$evaluation_status"
fi

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
