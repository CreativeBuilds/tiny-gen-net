Execute **Phase 2 hierarchical JEPA** train+eval. Follow `experiments/phase2_hierarchical_jepa.py`.

```bash
python experiments/phase2_hierarchical_jepa.py --train_steps 2000 --num_samples 30
```

If checkpoint exists and you only need eval (faster, with progress bars):

```bash
python experiments/phase2_hierarchical_jepa.py --eval_only --num_samples 30
```

After running: log to `EXPERIMENTS.md`, update `STATE.md`.
