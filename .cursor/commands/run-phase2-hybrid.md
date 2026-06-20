Execute **Phase 2 hybrid** eval: JEPA init + diffusion refine vs pure baselines. Follow `experiments/phase2_hybrid.py`.

Prerequisite: `checkpoints/jepa/weight_jepa_phase2_global.pt` and `checkpoints/diffusion/weight_denoiser_norm_global.pt`.

```bash
python experiments/phase2_hybrid.py --refine_t 50 --num_samples 30
```

After running: log to `EXPERIMENTS.md`, update `STATE.md`, report hybrid vs JEPA vs diffusion on zero-shot and FT100/200.
