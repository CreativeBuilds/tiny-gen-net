Run the **MVP demo** — generate variable tiny MLPs with conditioned weights.

```bash
python demo_generate_model.py --task char_pred --diversity high --num 5
```

Requires `checkpoints/arch_gen/cond_weights_phase3_refine_v1.pt`. If missing, run `/run-phase3-refine` first.
