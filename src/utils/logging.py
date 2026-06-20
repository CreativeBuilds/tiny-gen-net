"""Simple experiment logging to JSON."""

import json
from datetime import datetime, timezone
from pathlib import Path


def log_run(out_dir: str | Path, name: str, metrics: dict, config: dict | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
        "metrics": metrics,
    }
    log_path = out_dir / "runs.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
