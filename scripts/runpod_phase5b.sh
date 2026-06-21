#!/usr/bin/env bash
# Phase 5b hierarchical progressive on RunPod.
# Usage: RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_phase5b.sh phase5b_v1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-phase5b_v1}"
EXTRA=()
[[ "${RETRAIN_PROG:-}" == "1" ]] && EXTRA+=(--retrain-prog)
exec "$ROOT/scripts/runpod_train.sh" experiments/phase5b_hierarchical_progressive.py --cloud-resume --tag "$TAG" "${EXTRA[@]}"
