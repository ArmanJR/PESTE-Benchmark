"""Structured, context-aware standard-library logging."""

import datetime as dt
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_RESERVED = frozenset(logging.makeLogRecord({}).__dict__)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record without leaking process environment."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(
            timespec="milliseconds"
        )
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", log_path: Path | None = None) -> None:
    """Configure process logging exactly once."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)
    root.setLevel(level.upper())


def log_context(**values: str | int) -> Mapping[str, str | int]:
    """Build a typed logging ``extra`` mapping."""
    return values
