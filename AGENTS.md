# AGENTS.md — tiny-gen-net

Instructions for AI agents working in this repository.

## Project

**tiny-gen-net** — incremental research on generative models (diffusion → JEPA → architecture) for neural network weights. Karpathy-style: tiny models, synthetic data, provable metrics.

## Before any work

1. Read `STATE.md` (current phase, next steps)
2. Read latest entries in `EXPERIMENTS.md`
3. Respect `.cursor/rules/` especially `project-philosophy.mdc`

## Primary workflows

| Command | Skill | Purpose |
|---------|-------|---------|
| `/continue` | `.cursor/skills/continue/SKILL.md` | Resume research — one increment + doc updates |
| `/status` | — | Read-only project status summary |
| `/run-phase0` | `.cursor/skills/run-experiment/SKILL.md` | Baselines + weight collection |
| `/run-phase1` | `.cursor/skills/run-experiment/SKILL.md` | Diffusion train + eval vs random |
| `/log-experiment` | — | Append results to living docs |

## Subagents

| Agent | When to use |
|-------|-------------|
| `experiment-analyst` | Interpret results after runs; suggest next step (analysis only) |

## Phase gates

- **Phase 0** — TinyMLP trains; weights collected to `checkpoints/weights/phase0/`
- **Phase 1** — Diffusion beats random init → document in `EXPERIMENTS.md` before Phase 2
- **Phase 2+** — `src/jepa/`, `src/arch_gen/` are placeholders until prior phase succeeds

## Key paths

```
src/models/tiny_mlp.py          # ~9k param fixed architecture
src/diffusion/simple_diffusion.py
src/utils/{weights,train_loop,viz}.py
experiments/phase0_train_baselines.py
experiments/phase1_weight_diffusion.py
scripts/collect_weights.py
```

## After every experiment

Update `EXPERIMENTS.md` and `STATE.md`. Save plots to `checkpoints/plots/`.

## Human docs

- `README.md` — overview + quick start
- `RULES.md`, `VISION.md`, `COMMANDS.md` — philosophy, roadmap, CLI
