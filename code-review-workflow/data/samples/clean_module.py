"""Well-behaved helpers: should produce zero findings."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("config missing: %s", path)
        return {}


def total(values: list[float], start: float = 0.0) -> float:
    return start + sum(values)
