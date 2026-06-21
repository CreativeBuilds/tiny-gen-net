#!/usr/bin/env bash
# RunPod train. First arg may be experiment script path.
#   ./scripts/runpod_train.sh --cloud-first --tag phase5a_v1
#   ./scripts/runpod_train.sh experiments/phase5b_hierarchical_progressive.py --cloud-resume --tag phase5b_v1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
EXTRA="${*:-}"
runpod_check_cli
runpod_budget_note
runpod_sync
runpod_remote_train $EXTRA
[[ "${RUNPOD_PULL:-0}" == "1" ]] && runpod_pull
[[ "${RUNPOD_STOP:-0}" == "1" ]] && runpod_stop
echo "Done. Pull: RUNPOD_PULL=1 ./scripts/runpod_train.sh (or ./scripts/runpod_pull.sh)"
echo "Stop pod: ./scripts/runpod_stop.sh  |  Delete: RUNPOD_TERMINATE=1 ./scripts/runpod_stop.sh"
