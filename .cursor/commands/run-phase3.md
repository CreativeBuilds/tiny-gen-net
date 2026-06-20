Execute **Phase 3** architecture generation + weight init. Follow `experiments/phase3_arch_gen.py`.

Prerequisite: Phase 2 JEPA + diffusion checkpoints.

```bash
python experiments/phase3_arch_gen.py --arch_train_steps 500 --num_samples 30 --ft_steps 100
```

After running: log to `EXPERIMENTS.md`, update `STATE.md`, report gen vs ref hybrid metrics.
