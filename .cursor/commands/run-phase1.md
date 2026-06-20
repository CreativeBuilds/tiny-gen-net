Run **Phase 1** weight diffusion experiment. Follow `.cursor/skills/run-experiment/SKILL.md` (Phase 1 section).

Prerequisite: `checkpoints/weights/phase0/weights.pt` must exist. If missing, run `/run-phase0` first.

```bash
python experiments/phase1_weight_diffusion.py \
  --weights checkpoints/weights/phase0/weights.pt \
  --diffusion_steps 2000 \
  --num_samples 30 \
  --ft_steps 100,200 \
  --num_samples 20 \
  --eval_steps 100
```

Use smoke config (500 diffusion steps, 5 samples) only if the user says "smoke" or "quick".

After running: log to `EXPERIMENTS.md`, update `STATE.md`, report generated vs random metrics and plot paths.
