# MVP Learnings — tiny-gen-net

*Summary after Phase 0–3 (2026-06-19). See [EXPERIMENTS.md](EXPERIMENTS.md) for raw metrics.*

## What Worked Well

### Architecture-conditioned weight generation (Phase 3 breakthrough)
The hypernet-style `CondWeightGenerator` (arch embed + latent → flat weights) was the key missing link. Once every topology got tailored inits instead of random fallback, aggregate zero-shot jumped from Δ +0.01 to **+0.37** vs random on diverse 50-sample eval.

### Flat JEPA for zero-shot (Phase 2)
Layer-chunk JEPA with Gaussian latent sampling was the **first method to beat random zero-shot** on fixed architecture (+0.24–0.26 Δ). Simple, fast, interpretable.

### Hybrid JEPA → diffusion refine (Phase 2)
Img2img-style refinement at `refine_t=50` balanced zero-shot (near JEPA) and FT (better than pure JEPA). Best **fixed-arch** recipe when brief fine-tuning is allowed.

### Global weight normalization (Phase 1 ablation)
`global` norm beat `layer` and `perdim` for diffusion zero-shot and FT on our ~9k-param MLP. **Always use global** unless ablating.

### Staged pipeline (Karpathy-style)
Fixing architecture first, then weights, then variable arch, then conditioning — each phase had a clear pass/fail metric. Negative results (H-JEPA) were as valuable as positive ones.

## What Was Harder Than Expected

### Architecture validity + token parsing
Autoregressive arch generators emit invalid token sequences; robust `ArchSpec.from_tokens` + fallback to `random_spec` was required. Expanded vocab (skip, depth 3) needed defensive parsing.

### Variable-size weight vectors
JEPA/diffusion trained on fixed 9288-dim vectors cannot directly sample for other topologies. Needed multi-arch collection + per-spec hypernet head (trim to `num_parameters()`).

### Normalization sensitivity
Per-dim normalization destroyed diffusion zero-shot (loss ~13 vs random ~2). Global norm was essential — easy to miss without ablation.

### Hierarchical JEPA at tiny scale
2-level H-JEPA did not beat flat JEPA — likely too little data, too simple a task, or insufficient hierarchy depth. Useful as scaffolding idea, not as weight generator.

### Silent long eval phases
Post-training CPU/MPS eval of 30–50 models × FT steps runs minutes with no output unless tqdm progress is added.

## Key Insights for Scaling

1. **Conditioning is non-optional for variable arch** — generative weights must know topology.
2. **Two-stage beats joint (for now)** — arch tokens then weights is easier to debug than one stream.
3. **Hybrid composes specialists** — JEPA (zero-shot) + diffusion (FT) + cond hypernet (variable arch) are complementary, not redundant.
4. **Hierarchy may help structure, not sampling** — H-JEPA global `h` could return for Phase 4 task/arch conditioning before deeper hierarchy for weights.
5. **Always compare to random** — every metric needs the same baseline; Δ vs random is the honest signal.

## Limitations (Current MVP)

- **Tiny scale only** — ~2.6k–105k params, 8-char synthetic task, seconds to train.
- **Cond generator is MSE hypernet** — not a principled weight manifold model; may not generalize far OOD.
- **No task conditioning** — cannot yet "describe task → get model."
- **Checkpoints are local** — large `.pt` files gitignored; demo requires running refine once.

## Recommended Reading Order for New Contributors

1. [README.md](README.md) — MVP summary + demo
2. [STATE.md](STATE.md) — current status
3. [EXPERIMENTS.md](EXPERIMENTS.md) — full experiment log
4. [COMMANDS.md](COMMANDS.md) — reproduce any phase
