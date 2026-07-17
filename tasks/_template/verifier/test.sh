#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
submission=/app/final_artifact

set +e
python -m genesisbench.integrity \
  --submission "$submission" \
  --trajectory /logs/agent/acp_trajectory.jsonl \
  --config /verifier/integrity.json \
  --output /logs/verifier/integrity.json
integrity_status=$?
set -e

if [ "$integrity_status" -ne 0 ]; then
  echo "0.0" > /logs/verifier/reward.txt
  echo '{"reward": 0.0, "integrity_pass": false}' \
    > /logs/verifier/reward.json
  exit 0
fi

export GENESISBENCH_POLICY_ISOLATION=required
echo "0.0" > /logs/verifier/reward.txt
echo '{"reward": 0.0, "integrity_pass": true}' \
  > /logs/verifier/reward.json
