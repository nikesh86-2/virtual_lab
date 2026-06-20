from __future__ import annotations

import logging
import os


log = logging.getLogger("virtual_lab")


def getenv_int(name: str, default: int, min_value: int | None = None) -> int:
    """
    Safely parse an integer environment variable.
    """
    raw = os.getenv(name, str(default))

    try:
        value = int(raw)
    except Exception:
        log.warning("Invalid integer for %s=%r; using default %d", name, raw, default)
        value = default

    if min_value is not None:
        value = max(min_value, value)

    return value


def getenv_float(name: str, default: float) -> float:
    """
    Safely parse a float environment variable.
    """
    raw = os.getenv(name, str(default))

    try:
        return float(raw)
    except Exception:
        log.warning("Invalid float for %s=%r; using default %.3f", name, raw, default)
        return default


def getenv_bool(name: str, default: bool = False) -> bool:
    """
    Safely parse a boolean-like environment variable.
    """
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}