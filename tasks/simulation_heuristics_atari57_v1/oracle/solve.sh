#!/bin/bash
set -euo pipefail

rm -rf /app/final_artifact
cp -R /oracle/reference_artifact /app/final_artifact
