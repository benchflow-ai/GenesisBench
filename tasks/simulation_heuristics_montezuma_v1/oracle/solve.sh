#!/bin/bash
set -euo pipefail

mkdir -p /app/final_policy
cp /oracle/policy.py /app/final_policy/policy.py
cp /oracle/reference_trajectory.npz /app/final_policy/reference_trajectory.npz
