# Project State — tiny-gen-net

*Last updated: 2026-06-19*

## Current Phase

**Phase 5a — real text + sentence embeddings + nanoGPT scale (cloud-first)**

Pipeline implemented: Tiny Shakespeare char corpus, frozen `all-MiniLM-L6-v2` task embeddings, `NanoTransformer` (1M–10M params), sub-chunk weight hypernet, RunPod scripts. **Full training runs on RunPod only** — local `--smoke` is a ~2 min wiring check.

## Phase 5a Status

| Item | Status |
|------|--------|
| `data/shakespeare.py` | ✅ Tiny Shakespeare download + char tokenizer |
| `sentence_embedder.py` | ✅ Frozen MiniLM (384-d) |
| `nano_transformer.py` + `nano_spec.py` | ✅ 96 valid specs, presets smoke/1m/3m/8m |
| Sub-chunk weight gen | ✅ (required for 1M+ params; slow on MPS) |
| RunPod scripts | ✅ `runpod_launch/train/smoke/pull/stop/gpus.sh` — **default H100 SXM** |
| Local smoke (wiring) | ✅ ~2 min |
| **Cloud full run** | ✅ 2026-06-20 — task-nano **Δ +0.73** vs random (Shakespeare) |

### Early smoke metrics (incomplete training, 3 collect × 150 steps)

From accidental long local run (`metrics_phase5a_smoke.json` — **not** representative; weight-gen under-trained):

| Init | Zero Δ vs random | Notes |
|------|------------------|-------|
| Task-nano | **+0.43** | Best signal despite tiny train |
| Nano JEPA | +0.35 | Sentence-conditioned JEPA |
| Nano cond | +0.09 | Arch-only cond |
| Random | 4.23 | Shakespeare val CE |

**Do not compare directly to Phase 4b** (synthetic char, ~100k tx). Phase 5a Shakespeare random baseline ≈4.2 CE.

## Recommended Workflow

```bash
# Local: wiring only (~10 sec)
python experiments/phase5a_nano_text.py --smoke

# Cloud: full Phase 5a on H100 SXM (~$3–8/run, default)
./scripts/runpod_launch.sh
RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_train.sh --tag phase5a_v1

# Cloud smoke on RTX 4090 (~$0.10–0.15)
./scripts/runpod_smoke.sh

# GPU tiers + budget: ./scripts/runpod_gpus.sh
```

## Phase 4b (previous best on synthetic tx)

Task-tx ref init zero=**0.31**, FT100=**0.28**, steering match=**0.73**.

## Next Steps

1. **Run full Phase 5a on RunPod** (nano_1m/3m/8m presets, 2500+ collect steps)
2. Compare task-nano vs Phase 4b on Shakespeare (not synthetic)
3. JEPA/hybrid at nano scale once ref weights collected on GPU
4. Hierarchy (Phase 5b) after cloud baseline established

## Artifacts

- `checkpoints/arch_gen/metrics_phase5a_smoke.json` (partial local run)
- `checkpoints/weights/nano_multi_v1/weights_raw.pt` (3 smoke models)
- `scripts/runpod_*.sh`
- Phase 4b: `task_tx_phase4b_v2.pt`
