"""Writes one JSON line per request so predictions can be monitored later."""

import json
from datetime import datetime, timezone

from app.config import MODEL_VERSION, get_log_path


def log_prediction(endpoint, inputs, outputs):
    """Append a structured record (input, output, timestamp) to the log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model_version": MODEL_VERSION,
        "input": inputs,
        "output": outputs,
    }
    log_path = get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as file:
        file.write(json.dumps(record) + "\n")
