# Development Rules — tiny-gen-net

These rules govern every change to this repository. When in doubt, choose **provability and iteration speed** over scale or elegance.

## Philosophy

1. **Karpathy-style minimalism** — Single-file or very few files per phase. Code should be readable, hackable, and educational. Prefer clarity over abstraction.
2. **Prove before scaling** — Every phase must have measurable success signals (loss curves, comparisons to baselines) before moving on.
3. **Start absurdly small** — Tiny models (few thousand to ~50k params max in early phases), synthetic or small datasets, fast experiments.
4. **Visualize aggressively** — Plot losses, weight distributions, generated samples. If you can't see it, you can't debug it.
5. **Reproducible by default** — Fixed seeds, logged hyperparameters, dated experiment entries in `EXPERIMENTS.md`.
6. **Never over-engineer early** — Add complexity only after the previous step works and is verified.

## Code Standards

- **PyTorch** for all model code. Keep external dependencies minimal (`requirements.txt`).
- **Imports at top of file** — No inline imports unless circular dependency is documented.
- **Exhaustive switches** — Use `never` checks in default cases for discriminated unions.
- **Comments** — Explain *why*, not *what*. Non-obvious math and design decisions get comments; obvious code does not.
- **Early returns** — Prefer guard clauses over deep nesting.
- **Scope** — Minimal diffs. Don't refactor unrelated code in the same change.

## Experiment Protocol

1. Run experiment with logged config and seed.
2. Record results in `EXPERIMENTS.md` (date, phase, metrics, outcome, notes).
3. Update `STATE.md` with current phase, what exists, open questions, next steps.
4. Save artifacts to `checkpoints/` with descriptive names.

## The `/continue` Protocol

When continuing work in a new session:

1. Read `STATE.md`, `EXPERIMENTS.md`, `README.md`, and relevant source files.
2. Review experiment outcomes and open questions.
3. Propose and implement **one meaningful increment** aligned with the phased roadmap.
4. Update `STATE.md` and `EXPERIMENTS.md`.
5. Keep changes minimal, documented, and verifiable.

## What NOT To Do (Early Phases)

- Don't add JEPA, architecture generation, or conditioning until Phase 1 weight diffusion shows a clear signal.
- Don't scale model size or dataset before baselines are solid.
- Don't introduce heavy frameworks (Hydra, W&B, etc.) until simple logging is insufficient.
- Don't commit secrets, large binary checkpoints, or generated plots (see `.gitignore`).
