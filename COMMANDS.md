# Commands — tiny-gen-net

All commands assume repo root and activated venv:

```bash
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
```

---

## Phase 5a — Cloud training (RunPod) ☁️

**Full Phase 5a is too slow locally. Use RunPod.** Budget guide: **$300/day** easily covers many full runs.

### GPU tiers (default: fastest)

| Tier | GPU | ~$/hr (community) | $300/day buys | When |
|------|-----|-------------------|---------------|------|
| **fast** (default) | H100 SXM | ~$2.69 | ~111 hrs | Full Phase 5a (~1–3 hrs, ~$3–8/run) |
| balanced | A100 80GB PCIe | ~$1.19 | ~252 hrs | Good speed/$ if H100 unavailable |
| cheap | RTX 4090 | ~$0.34 | ~882 hrs | Cloud smoke only (~30 min) |

```bash
./scripts/runpod_gpus.sh          # show tiers + live availability
```

### Full run (H100, recommended)

```bash
./scripts/runpod_launch.sh                          # H100 SXM, COMMUNITY cloud
./scripts/runpod_train.sh --tag phase5a_v1          # sync + train
RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_train.sh --tag phase5a_v1  # pull + stop
```

SSH is auto-resolved from `.runpod_pod_id` via `runpodctl ssh info`.

### Cloud smoke (RTX 4090, ~$0.10–0.15)

```bash
./scripts/runpod_smoke.sh
```

### Local wiring only (~10 sec)

```bash
python experiments/phase5a_nano_text.py --smoke
```

### Env vars

```bash
export RUNPOD_GPU=fast          # fast | balanced | cheap | raw gpuId
export RUNPOD_CLOUD_TYPE=COMMUNITY
export RUNPOD_PULL=1              # pull checkpoints after train
export RUNPOD_STOP=1              # stop pod after train (stop billing)
```

---

Defined in `.cursor/commands/` — type `/` in Cursor chat:

| Command | Description |
|---------|-------------|
| `/continue` | Main workflow: read `STATE.md`, implement next increment, update docs |
| `/status` | Read-only summary of phase, metrics, next steps |
| `/run-phase0` | Run baseline training + weight collection |
| `/run-phase1` | Run diffusion training + eval vs random |
| `/run-phase2-hybrid` | Eval JEPA+diffusion hybrid vs baselines |
| `/log-experiment` | Record results in `EXPERIMENTS.md` and `STATE.md` |

Skills (detailed workflows): `.cursor/skills/continue/`, `.cursor/skills/run-experiment/`

Rules (always applied): `.cursor/rules/project-philosophy.mdc`, `.cursor/rules/living-docs.mdc`

Agent reference: [AGENTS.md](AGENTS.md)

---

## MVP Demo

Generate and evaluate models in seconds (requires refine checkpoints):

```bash
python demo_generate_model.py --task "medium balanced char predictor" --num 5
python demo_generate_model.py --list_examples
python demo_generate_model.py --diversity low --num 3 --ft_steps 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | `medium balanced char predictor` | Natural language task (Phase 4) |
| `--list_examples` | — | Print example task strings |
| `--task_ckpt` | `task_cond_phase4_v2.pt` | Task-conditioned checkpoint |
| `--diversity` | `high` | `low`=ref arch, `medium`=random grid, `high`=task or arch generator |
| `--num` | 5 | Models to generate |
| `--ft_steps` | — | e.g. `100` for brief fine-tune eval |
| `--noise_scale` | 0.5 | Latent noise for cond generator (0.25 often best) |

First-time setup: `python experiments/phase3_conditioned_refine.py` then `python experiments/phase4_task_cond_refine.py`

---

## Phase 4 — Task conditioning

```bash
# v2 refine (recommended)
python experiments/phase4_task_cond_refine.py --num_tasks 25 --match_weight 0.5
python demo_generate_model.py --task "deep reasoning capacity char predictor" --num 3

# v1 baseline
python experiments/phase4_task_cond.py --num_tasks 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_tasks` | 25 (refine) / 10 (v1) | Eval task descriptions |
| `--match_weight` | 0.5 | ArchMatchHead aux loss weight (refine only) |
| `--arch_train_steps` | 1200 (refine) / 800 (v1) | TaskArchGenerator training |
| `--weight_train_steps` | 2000 | TaskCondWeightGenerator training |

Outputs: `checkpoints/arch_gen/task_cond_phase4_v2.pt`, `metrics_phase4_v2.json`, `checkpoints/plots/phase4_refine/`

---

## Phase 4b — Tiny transformer

```bash
python experiments/phase4b_tiny_transformer.py --num_collect 30
python experiments/phase4b_tx_refine.py --skip_ref_collect   # steering + fixed-ref init
python demo_generate_model.py --arch transformer --task "large wide capacity transformer char predictor" --num 3
```

Outputs: `task_tx_phase4b_v2.pt`, `metrics_phase4b_v2.json`, `checkpoints/plots/phase4b_refine/`

---

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
# Apple Silicon: PyTorch wheels include MPS; scripts auto-select cuda > mps > cpu
python -c "from src.utils.device import get_device; print(get_device())"
```

---

## Phase 0 — Baselines & Weight Collection

### Train a single tiny model

```bash
python scripts/train_tiny_model.py --seed 42 --steps 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--steps` | 500 | Training steps |
| `--hidden` | 64 | Hidden dimension |
| `--lr` | 1e-3 | Learning rate |
| `--out` | `checkpoints/models/` | Output directory |

### Batch baseline training + plots

```bash
python experiments/phase0_train_baselines.py --num_seeds 10 --steps 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_seeds` | 10 | Number of random seeds |
| `--steps` | 500 | Steps per model |
| `--plot_dir` | `checkpoints/plots/` | Where to save figures |

### Collect weight dataset (for diffusion)

```bash
python scripts/collect_weights.py --num_models 50 --steps 500 --out checkpoints/weights/phase0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_models` | 50 | Models to train and collect |
| `--steps` | 500 | Training steps each |
| `--seed_start` | 0 | First seed (uses seed_start, seed_start+1, ...) |
| `--out` | `checkpoints/weights/phase0` | Output dir (`weights_raw.pt`, `weights.pt`, `meta.json`) |
| `--norm` | `perdim` | Normalization for `weights.pt`: `global`, `layer`, or `perdim` |

Output files:
- `weights.pt` — `(N, D)` normalized weight tensor
- `meta.json` — mean, std, shapes, hyperparameters

---

## Phase 1 — Normalization ablation

Compare `global`, `layer`, and `perdim` normalization on the same raw weights:

```bash
python scripts/collect_weights.py --num_models 100 --norm global
python experiments/phase1_norm_ablation.py --diffusion_steps 2000
```

Outputs: `checkpoints/plots/phase1_norm_ablation/ablation_summary.json` + per-mode plots.

## Phase 2 — Weight Diffusion vs JEPA

Compare minimal JEPA against global-norm diffusion baseline:

```bash
python experiments/phase2_jepa_vs_diffusion.py --train_steps 2000 --num_samples 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--train_steps` | 2000 | JEPA training steps |
| `--diffusion_ckpt` | `checkpoints/diffusion/weight_denoiser_norm_global.pt` | Frozen diffusion baseline |
| `--latent_dim` | 64 | JEPA latent size per chunk |
| `--plot_dir` | `checkpoints/plots/phase2_jepa` | Output plots |

Outputs: `checkpoints/jepa/metrics_phase2_global.json`, comparison plots.

### Phase 2 hybrid — JEPA init + diffusion refine

Prerequisite: Phase 2 JEPA checkpoint + global-norm diffusion checkpoint.

```bash
python experiments/phase2_hybrid.py --refine_t 50 --num_samples 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--jepa_ckpt` | `checkpoints/jepa/weight_jepa_phase2_global.pt` | Trained JEPA |
| `--diffusion_ckpt` | `checkpoints/diffusion/weight_denoiser_norm_global.pt` | Trained denoiser |
| `--refine_t` | 50 | DDPM start timestep for JEPA→diffusion refine |
| `--num_samples` | 30 | Eval sample count |
| `--ft_steps` | `100,200` | Fine-tune step counts |
| `--plot_dir` | `checkpoints/plots/phase2_hybrid` | Output plots |
| `--tag` | `hybrid_t50` | Run tag for metrics JSON |

Outputs: `checkpoints/hybrid/metrics_{tag}.json`, comparison + delta plots.

### Phase 2 hierarchical JEPA (H-JEPA)

Train 2-level JEPA and compare vs flat JEPA + hybrids:

```bash
python experiments/phase2_hierarchical_jepa.py --train_steps 2000 --num_samples 30
```

Re-run eval only (skip training; shows progress bars during slow CPU eval):

```bash
python experiments/phase2_hierarchical_jepa.py --eval_only --num_samples 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--train_steps` | 2000 | H-JEPA training steps (skipped with `--eval_only`) |
| `--eval_only` | off | Load checkpoint and run sampling+eval only |
| `--hjepa_ckpt` | `checkpoints/jepa/weight_hjepa_{tag}.pt` | H-JEPA checkpoint path |
| `--flat_jepa_ckpt` | `checkpoints/jepa/weight_jepa_phase2_global.pt` | Flat JEPA baseline |
| `--refine_t` | 50 | Hybrid diffusion refine timestep |
| `--plot_dir` | `checkpoints/plots/phase2_hjepa` | Output plots |

Outputs: `checkpoints/jepa/weight_hjepa_{tag}.pt`, `metrics_{tag}.json`, comparison plots.

---

## Phase 3 — Architecture generation + weight init

End-to-end: generate variable arch → hybrid/random weights → eval vs fixed baselines.

```bash
python experiments/phase3_arch_gen.py --arch_train_steps 500 --num_samples 30 --ft_steps 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--arch_train_steps` | 500 | GRU arch generator training steps |
| `--jepa_ckpt` | `checkpoints/jepa/weight_jepa_phase2_global.pt` | Flat JEPA (ref arch only) |
| `--diffusion_ckpt` | `checkpoints/diffusion/weight_denoiser_norm_global.pt` | Diffusion for hybrid refine |
| `--refine_t` | 50 | Hybrid diffusion start timestep |
| `--num_samples` | 30 | Generated architectures to eval |
| `--ft_steps` | `100` | Fine-tune step counts |
| `--plot_dir` | `checkpoints/plots/phase3_arch` | Output plots |
| `--tag` | `phase3_v1` | Run tag |

Outputs: `checkpoints/arch_gen/metrics_{tag}.json`, `{tag}_sample_archs.txt`, comparison plots.

### Phase 3 conditioned weights — arch embed + hypernet

Train `CondWeightGenerator` on multi-arch weights; eval vs unconditioned Phase 3.

```bash
python scripts/collect_multi_arch_weights.py --num_per_spec 10   # once
python experiments/phase3_conditioned.py --cond_train_steps 2000 --num_samples 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--multi_arch` | `checkpoints/weights/multi_arch/weights_raw.pt` | Multi-arch training weights |
| `--cond_train_steps` | 2000 | Conditioned weight generator training |
| `--skip_collect` | off | Skip auto-collect if weights missing |
| `--plot_dir` | `checkpoints/plots/phase3_conditioned` | Output plots |

Outputs: `cond_weights_{tag}.pt`, `metrics_{tag}.json`, similarity scatter plot.

### Phase 3 refine — expanded diversity stress test

```bash
python scripts/collect_multi_arch_weights.py --num_per_spec 4 --out checkpoints/weights/multi_arch_v2
python experiments/phase3_conditioned_refine.py --num_gen 80 --num_eval 50 --skip_collect
```

Expanded grid: 7 hidden sizes, depth 1–3, optional skip. Includes noise-scale sweep plot.

---

```bash
python experiments/phase1_weight_diffusion.py \
  --weights checkpoints/weights/phase0/weights_raw.pt \
  --norm global \
  --diffusion_steps 2000 \
  --num_samples 30 \
  --ft_steps 100,200 \
  --tag phase1_global
```

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | required | Path to `weights.pt` or `weights_raw.pt` |
| `--norm` | — | Required with `weights_raw.pt`: `global`, `layer`, or `perdim` |
| `--diffusion_steps` | 2000 | Denoiser training steps |
| `--timesteps` | 200 | DDPM diffusion timesteps |
| `--num_samples` | 30 | Weight samples to generate for eval |
| `--num_collected_eval` | 20 | Held-out collected weights to eval |
| `--ft_steps` | `100,200` | Comma-separated fine-tune step counts (zero-shot always) |
| `--tag` | `phase1_full` | Run tag for plot/checkpoint names |
| `--seed` | 42 | Reproducibility seed |
| `--plot_dir` | `checkpoints/plots/` | Plot output |

---

## Utilities (Python)

```python
# Inspect weight collection
from src.utils.weights import load_weight_collection
w, meta = load_weight_collection("checkpoints/weights/phase0/weights.pt")

# Plot losses from a log file
from src.utils.viz import plot_loss_curve
plot_loss_curve([1.0, 0.5, 0.3], out_path="checkpoints/plots/loss.png")
```

---

## Typical Workflow

```bash
# Full Phase 0 → 1 pipeline
python experiments/phase0_train_baselines.py
python scripts/collect_weights.py --num_models 50
python experiments/phase1_weight_diffusion.py --weights checkpoints/weights/phase0/weights.pt
```

Then update `EXPERIMENTS.md` and `STATE.md` with metrics from the printed summary and saved plots.

---

## Checkpoints Layout

```
checkpoints/
├── models/           # individual TinyMLP checkpoints
├── weights/phase0/   # collected weight tensors + meta
├── diffusion/        # trained denoiser state
├── jepa/             # trained JEPA + phase2 metrics
├── hybrid/           # hybrid eval metrics JSON
└── plots/            # PNG figures from experiments
```

Large `.pt` files are gitignored; keep `checkpoints/.gitkeep`.
