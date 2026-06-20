#!/usr/bin/env bash
# Stop or terminate RunPod pod.
# Usage: ./scripts/runpod_stop.sh
#        RUNPOD_TERMINATE=1 ./scripts/runpod_stop.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
if [[ "${RUNPOD_TERMINATE:-0}" == "1" ]]; then runpod_terminate
else runpod_stop; fi
