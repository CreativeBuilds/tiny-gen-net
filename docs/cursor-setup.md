# Cursor setup — tiny-gen-net

This folder configures Cursor IDE for the research workflow.

## Layout

```
.cursor/
├── commands/           # Slash commands (type / in chat)
│   ├── continue.md     # /continue — main resume workflow
│   ├── status.md       # /status — read-only project summary
│   ├── run-phase0.md   # /run-phase0
│   ├── run-phase1.md   # /run-phase1
│   └── log-experiment.md
├── skills/             # Detailed workflows for commands
│   ├── continue/SKILL.md
│   └── run-experiment/SKILL.md
├── rules/              # Always-on and file-scoped agent rules
│   ├── project-philosophy.mdc   # alwaysApply
│   ├── living-docs.mdc          # alwaysApply
│   ├── python-pytorch.mdc       # **/*.py
│   └── experiments-protocol.mdc # experiments/**, scripts/**
├── agents/             # Subagents
│   └── experiment-analyst.md
└── settings.json
```

## Quick reference

| You want to… | Use |
|--------------|-----|
| Resume after a break | `/continue` |
| See where we are | `/status` |
| Run Phase 0 pipeline | `/run-phase0` |
| Run Phase 1 pipeline | `/run-phase1` |
| Log results manually | `/log-experiment` |
| Interpret last run | Delegate to `experiment-analyst` subagent |

## Adding new commands

1. Create `.cursor/commands/my-command.md` with a short instruction body
2. If non-trivial, add `.cursor/skills/my-command/SKILL.md` with full workflow
3. Document in `COMMANDS.md` and `AGENTS.md`

## Rules vs RULES.md

- `RULES.md` — human-readable project philosophy
- `.cursor/rules/*.mdc` — machine-applied agent constraints in Cursor

Keep them aligned when philosophy changes.
