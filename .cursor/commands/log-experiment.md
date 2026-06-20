Log an experiment result to living docs. The user may paste metrics or point to a completed run.

1. Read latest `EXPERIMENTS.md` template and append a new dated entry
2. Update `STATE.md` (Key Results, Open Questions, Immediate Next Steps)
3. Confirm artifact paths mentioned exist under `checkpoints/`

If metrics are missing, ask the user or read from `checkpoints/logs/runs.jsonl` and script stdout if available.

Do not invent metrics — only log what was measured.
