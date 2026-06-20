#!/usr/bin/env bash
# Pull checkpoints + logs from RunPod.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
runpod_resolve_ssh || exit 1
mkdir -p "$ROOT/checkpoints" "$RUNPOD_LOG_DIR"
$RUNPOD_SSH "tar czf - -C $RUNPOD_REMOTE checkpoints logs/runpod 2>/dev/null || tar czf - -C $RUNPOD_REMOTE checkpoints" | tar xzf - -C "$ROOT" 2>/dev/null || \
  $RUNPOD_SSH "tar czf - -C $RUNPOD_REMOTE checkpoints" | tar xzf - -C "$ROOT"
echo "Pulled checkpoints/ from pod"
