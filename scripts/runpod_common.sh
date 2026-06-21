#!/usr/bin/env bash
# Shared RunPod helpers for tiny-gen-net.
set -euo pipefail

RUNPOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNPOD_REMOTE="${RUNPOD_REMOTE:-/workspace/tiny-gen-net}"
RUNPOD_LOG_DIR="${RUNPOD_LOG_DIR:-$RUNPOD_ROOT/logs/runpod}"
RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-COMMUNITY}"
RUNPOD_TEMPLATE="${RUNPOD_TEMPLATE:-runpod-torch-v21}"
RUNPOD_VOLUME_GB="${RUNPOD_VOLUME_GB:-50}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"

runpod_pod_id() {
  cat "$RUNPOD_ROOT/.runpod_pod_id" 2>/dev/null || true
}

runpod_gpu_id() {
  local tier="${RUNPOD_GPU:-fast}"
  case "$tier" in
    fast|h100|H100) echo "NVIDIA H100 80GB HBM3" ;;
    balanced|a100|A100) echo "NVIDIA A100 80GB PCIe" ;;
    cheap|4090|smoke) echo "NVIDIA GeForce RTX 4090" ;;
    *) echo "$tier" ;;
  esac
}

runpod_gpu_rate() {
  local tier="${RUNPOD_GPU:-fast}"
  case "$tier" in
    fast|h100|H100) echo "2.69" ;;
    balanced|a100|A100) echo "1.19" ;;
    cheap|4090|smoke) echo "0.34" ;;
    *) echo "?" ;;
  esac
}

runpod_budget_note() {
  local rate="$(runpod_gpu_rate)"
  local tier="${RUNPOD_GPU:-fast}"
  echo "GPU tier=$tier (~\$$rate/hr community) | \$300/day ≈ $(python3 -c "print(int(300/float('$rate')))") hrs | cloud=$RUNPOD_CLOUD_TYPE"
}

runpod_check_cli() {
  command -v runpodctl >/dev/null || { echo "runpodctl not found"; exit 1; }
  runpodctl doctor >/dev/null 2>&1 || { echo "runpodctl doctor failed — check API key (~/.runpod/config.toml)"; exit 1; }
}

runpod_resolve_ssh() {
  if [[ -n "${RUNPOD_SSH:-}" ]]; then return 0; fi
  local pod_id="$(runpod_pod_id)"
  [[ -z "$pod_id" ]] && { echo "No .runpod_pod_id — run ./scripts/runpod_launch.sh first"; return 1; }
  local info
  info="$(runpodctl ssh info "$pod_id" -o json 2>/dev/null)" || { echo "SSH not ready for $pod_id — pod still starting?"; return 1; }
  RUNPOD_SSH="$(python3 -c "
import json,sys
d=json.load(sys.stdin)
for k in ('ssh_command','command','sshCommand','ssh'):
    if d.get(k): print(d[k]); sys.exit(0)
ports=d.get('ports') or d.get('runtime',{}).get('ports') or []
ip,port=d.get('ip'),d.get('port')
if ip and port: print(f'ssh root@{ip} -p {port} -i $RUNPOD_SSH_KEY'); sys.exit(0)
for p in ports:
    if p.get('privatePort')==22 or p.get('type')=='tcp':
        ip=p.get('ip') or p.get('host')
        port=p.get('publicPort') or p.get('port')
        if ip and port: print(f'ssh root@{ip} -p {port} -i $RUNPOD_SSH_KEY'); sys.exit(0)
" <<< "$info")"
  [[ -z "$RUNPOD_SSH" ]] && { echo "Could not parse SSH from runpodctl ssh info"; return 1; }
  export RUNPOD_SSH
  echo "SSH resolved: $RUNPOD_SSH"
}

runpod_wait_running() {
  local pod_id="${1:-$(runpod_pod_id)}"
  local max_wait="${2:-600}"
  local elapsed=0
  echo "Waiting for pod $pod_id to reach RUNNING (max ${max_wait}s)..."
  while [[ $elapsed -lt $max_wait ]]; do
    local pod_status
    pod_status="$(runpodctl pod get "$pod_id" -o json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('desiredStatus',''))" 2>/dev/null || true)"
    if [[ "$pod_status" == "RUNNING" ]]; then echo "Pod RUNNING (${elapsed}s)"; runpod_resolve_ssh || true; return 0; fi
    sleep 10; elapsed=$((elapsed + 10))
  done
  echo "Timeout waiting for RUNNING"; return 1
}

runpod_wait_ssh() {
  local pod_id="${1:-$(runpod_pod_id)}"
  local max_wait="${2:-300}"
  local elapsed=0
  echo "Waiting for SSH on $pod_id (max ${max_wait}s)..."
  while [[ $elapsed -lt $max_wait ]]; do
    local info cmd
    info="$(runpodctl ssh info "$pod_id" -o json 2>/dev/null || true)"
    cmd="$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('ssh_command',''))" "$info" 2>/dev/null || true)"
    if [[ -n "$cmd" ]] && eval "$cmd" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
      export RUNPOD_SSH="$cmd"; echo "SSH ready (${elapsed}s): $RUNPOD_SSH"; return 0
    fi
    sleep 10; elapsed=$((elapsed + 10))
  done
  echo "SSH timeout"; return 1
}

runpod_verify_sync() {
  runpod_resolve_ssh || return 1
  eval "$RUNPOD_SSH" "test -f $RUNPOD_REMOTE/experiments/phase5a_nano_text.py && test -f $RUNPOD_REMOTE/data/shakespeare.py && echo SYNC_OK"
}

runpod_sync() {
  runpod_wait_ssh || return 1
  "$RUNPOD_ROOT/scripts/runpod_sync.sh"
}

runpod_pull() {
  runpod_resolve_ssh || return 1
  "$RUNPOD_ROOT/scripts/runpod_pull.sh"
}

runpod_stop() {
  local pod_id="${1:-$(runpod_pod_id)}"
  [[ -z "$pod_id" ]] && { echo "No pod id"; return 1; }
  echo "Stopping pod $pod_id (billing ends when stopped)..."
  runpodctl pod stop "$pod_id"
}

runpod_terminate() {
  local pod_id="${1:-$(runpod_pod_id)}"
  [[ -z "$pod_id" ]] && { echo "No pod id"; return 1; }
  echo "Terminating pod $pod_id..."
  runpodctl pod delete "$pod_id"
  rm -f "$RUNPOD_ROOT/.runpod_pod_id"
}

runpod_remote_setup() {
  runpod_resolve_ssh || return 1
  eval "$RUNPOD_SSH" bash -s <<EOF
set -euo pipefail
cd $RUNPOD_REMOTE
pip3 install -q -r requirements-cloud.txt
python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable (torch={torch.__version__}, cuda={torch.version.cuda})'; print(f'GPU OK: {torch.cuda.get_device_name(0)} torch={torch.__version__}')"
EOF
}

runpod_remote_train() {
  local script="experiments/phase5a_nano_text.py"
  if [[ "${1:-}" == experiments/*.py ]]; then script="$1"; shift; fi
  local extra="${*:-}"
  local pod_id="$(runpod_pod_id)"
  [[ -z "$pod_id" ]] && { echo "No pod id"; return 1; }
  runpod_resolve_ssh || return 1
  local ssh_cmd
  ssh_cmd="$(runpodctl ssh info "$pod_id" -o json | python3 -c "import json,sys; print(json.load(sys.stdin).get('ssh_command',''))")"
  [[ -n "$ssh_cmd" ]] || { echo "Could not resolve ssh_command for $pod_id"; return 1; }
  mkdir -p "$RUNPOD_LOG_DIR"
  local log="$RUNPOD_LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
  echo "Training on pod (log: $log) script=$script ..."
  tee "$log" < <(
    eval "$ssh_cmd" bash -s <<EOF
set -euo pipefail
cd $RUNPOD_REMOTE
pip3 install -q -r requirements-cloud.txt
python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable (torch={torch.__version__})'; print(f'GPU OK: {torch.cuda.get_device_name(0)} torch={torch.__version__}')"
export PYTHONUNBUFFERED=1
exec python3 -u $script $extra
EOF
  )
}
