Run **Phase 0** baselines and weight collection. Follow `.cursor/skills/run-experiment/SKILL.md` (Phase 0 section).

```bash
python experiments/phase0_train_baselines.py --num_seeds 10 --steps 500
python scripts/collect_weights.py --num_models 50 --steps 500 --out checkpoints/weights/phase0
```

Use smoke config (3 seeds, 10 models, 200 steps) only if the user says "smoke" or "quick".

After running: log to `EXPERIMENTS.md`, update `STATE.md`, report metrics and plot paths.
