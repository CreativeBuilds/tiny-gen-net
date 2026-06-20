#!/usr/bin/env bash
# Show recommended GPUs + budget math for $300/day.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/runpod_common.sh
source "$ROOT/scripts/runpod_common.sh"
cat <<EOF
RunPod GPU tiers for tiny-gen-net Phase 5a (1M–10M params, Shakespeare)

Tier       gpuId                              ~\$/hr (community)  \$300/day buys   Use case
─────────────────────────────────────────────────────────────────────────────────────────────
fast       NVIDIA H100 80GB HBM3 (H100 SXM)   ~\$2.69            ~111 hrs       FULL runs (default)
balanced   NVIDIA A100 80GB PCIe              ~\$1.19            ~252 hrs       good speed/$
cheap      NVIDIA GeForce RTX 4090            ~\$0.34            ~882 hrs       cloud smoke only

Phase 5a full run estimate: 1–3 hrs on H100 (~\$3–8/run). Well within \$300/day.

Env vars:
  RUNPOD_GPU=fast|balanced|cheap   (default: fast)
  RUNPOD_CLOUD_TYPE=COMMUNITY      (default, ~50% cheaper than SECURE)
  RUNPOD_STOP=1                    stop pod after train
  RUNPOD_PULL=1                    pull checkpoints after train

Commands:
  ./scripts/runpod_launch.sh                    # H100 SXM
  ./scripts/runpod_smoke.sh                     # RTX 4090 quick cloud check
  RUNPOD_GPU=balanced ./scripts/runpod_launch.sh
EOF
echo ""
runpodctl gpu list -o json 2>/dev/null | python3 -c "
import json,sys
want={'NVIDIA H100 80GB HBM3','NVIDIA H100 PCIe','NVIDIA A100 80GB PCIe','NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 5090'}
for g in json.load(sys.stdin):
    if g.get('gpuId') in want:
        print(f\"  {g['displayName']:12} avail={g['available']} stock={g.get('stockStatus','?')} id={g['gpuId']}\")
" 2>/dev/null || echo "(run runpodctl gpu list for live availability)"
