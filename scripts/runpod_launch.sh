#!/usr/bin/env bash
# Launch RunPod GPU pod. Tries COMMUNITY then SECURE; optional GPU fallbacks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"

NAME="${1:-tiny-gen-phase5a}"
WAIT="${RUNPOD_WAIT:-1}"
runpod_check_cli
runpod_budget_note

try_create() {
  local gpu="$1" cloud="$2"
  echo "Trying gpu=$gpu cloud=$cloud ..." >&2
  local out rc=0
  out="$(runpodctl pod create --name "$NAME" --template-id "$RUNPOD_TEMPLATE" --gpu-id "$gpu" --gpu-count 1 \
    --volume-in-gb "$RUNPOD_VOLUME_GB" --cloud-type "$cloud" 2>&1)" || rc=$?
  if [[ $rc -eq 0 ]] && echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'id' in d" 2>/dev/null; then
    echo "$out"; return 0
  fi
  echo "$out" | tail -1 >&2
  return 1
}

OUT=""
for spec in "fast:COMMUNITY" "fast:SECURE" "balanced:SECURE" "balanced:COMMUNITY"; do
  tier="${spec%%:*}"; cloud="${spec##*:}"
  GPU_ID="$(RUNPOD_GPU=$tier runpod_gpu_id)"
  OUT="$(try_create "$GPU_ID" "$cloud")" && break
done
[[ -z "$OUT" ]] && { echo "All launch attempts failed"; exit 1; }

POD_ID="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$OUT")"
echo "$POD_ID" > "$ROOT/.runpod_pod_id"
echo "Pod created: $POD_ID (saved to .runpod_pod_id)"
if [[ "$WAIT" == "1" ]]; then
  runpod_wait_running "$POD_ID" "${RUNPOD_WAIT_SECS:-900}" || true
  runpod_wait_ssh "$POD_ID" 300 || true
fi
echo "Next: RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_train.sh --cloud-first --tag phase5a_v1"
