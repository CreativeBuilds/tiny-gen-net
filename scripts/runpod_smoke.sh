#!/usr/bin/env bash
# Quick GPU validation on cloud (~15-30 min, RTX 4090 ~$0.10-0.15).
# Usage: ./scripts/runpod_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export RUNPOD_GPU=cheap
export RUNPOD_CLOUD_TYPE=COMMUNITY
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
runpod_check_cli
echo "Cloud smoke: RTX 4090 community (~\$0.34/hr)"
if [[ ! -f "$ROOT/.runpod_pod_id" ]]; then
  "$ROOT/scripts/runpod_launch.sh" "tiny-gen-smoke"
fi
RUNPOD_PULL=1 RUNPOD_STOP=1 "$ROOT/scripts/runpod_train.sh" --cloud-smoke --tag phase5a_cloud_smoke
