#!/usr/bin/env bash
# Phase 5c scale hierarchy + anchors on RunPod.
# Usage: RETRAIN_PROG=1 RETRAIN_ANCHORS=1 RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_phase5c.sh phase5c_v2
# Usage: PROFILE=fine RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_phase5c.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-phase5c_v1}"
EXTRA=()
[[ "${RETRAIN_PROG:-}" == "1" ]] && EXTRA+=(--retrain-prog)
[[ "${RETRAIN_ANCHORS:-}" == "1" ]] && EXTRA+=(--retrain-anchors)
[[ -n "${ABLATION:-}" ]] && EXTRA+=(--ablation "$ABLATION")
[[ -n "${PROFILE:-}" ]] && EXTRA+=(--profile "$PROFILE")
exec "$ROOT/scripts/runpod_train.sh" experiments/phase5c_scale_hierarchy.py --cloud-resume --tag "$TAG" "${EXTRA[@]}"
