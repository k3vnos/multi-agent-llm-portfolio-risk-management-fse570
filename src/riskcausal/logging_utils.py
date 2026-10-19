"""Logging: human-readable console + JSON-lines file per run."""

from __future__ import annotations

import json
import logging
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
             "msg": record.getMessage()}
        )


def setup_logging(level: str = "INFO", logfile: Path | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib", "pymc", "pytensor"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
