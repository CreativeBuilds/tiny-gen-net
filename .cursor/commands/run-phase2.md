Execute **Phase 2** JEPA vs diffusion comparison. Follow `experiments/phase2_jepa_vs_diffusion.py`.

Prerequisite: `checkpoints/weights/phase0/weights_raw.pt` and `checkpoints/diffusion/weight_denoiser_norm_global.pt`.

```bash
python experiments/phase2_jepa_vs_diffusion.py --train_steps 2000 --num_samples 30
```

After running: log to `EXPERIMENTS.md`, update `STATE.md`, report JEPA vs diffusion metrics on zero-shot and FT100/200.
