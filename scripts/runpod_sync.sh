#!/usr/bin/env bash
# Sync essential code only (fast — excludes 20GB+ checkpoints/venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
runpod_wait_ssh || exit 1
eval "$RUNPOD_SSH" "mkdir -p $RUNPOD_REMOTE"
echo "Syncing code to $RUNPOD_REMOTE ..."
TAR=/tmp/tiny-gen-net-sync.tar.gz
COPYFILE_DISABLE=1 tar czf "$TAR" -C "$ROOT" experiments src data scripts requirements.txt requirements-cloud.txt demo_generate_model.py
eval "$RUNPOD_SSH" "tar xzf - -C $RUNPOD_REMOTE --no-same-owner --warning=no-unknown-keyword 2>/dev/null" < "$TAR"
rm -f "$TAR"
eval "$RUNPOD_SSH" "test -f $RUNPOD_REMOTE/experiments/phase5a_nano_text.py" && echo "SYNC_OK"
echo "Sync complete"
