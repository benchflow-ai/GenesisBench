#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

case "$(uname -m)" in
  arm64|aarch64) default_platform="linux/arm64" ;;
  *) default_platform="linux/amd64" ;;
esac

platform="${GENESISBENCH_DOCKER_PLATFORM:-$default_platform}"
image="${GENESISBENCH_DOCKER_IMAGE:-genesisbench-simulation-heuristics-ant-v1-runner:latest}"

docker build \
  --platform "$platform" \
  --tag "$image" \
  --file containers/simulation_heuristics_ant_v1/Dockerfile \
  .
