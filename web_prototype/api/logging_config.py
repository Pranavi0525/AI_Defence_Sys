"""
api/logging_config.py
=======================
Structured (JSON) logging for the API process. Deliberately stdlib-only
(no new dependency) -- a JSON formatter is ~20 lines and this repo's
requirements.txt is already carrying enough.

What gets logged: request_id, method, path, status_code, latency_ms, and
for /api/score specifically: decision, risk_score, model_version. What does
NOT get logged: request bodies, individual transaction amounts, customer_id,
or any other field from the incoming trace -- see app.py's middleware,
which logs only the fields listed above, never req.body().
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Anything passed via logging's `extra={...}` kwarg lands as plain
        # attributes on the record -- surface those too.
        reserved = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)))
        for key, value in vars(record).items():
            if key not in reserved and key != "message":
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
