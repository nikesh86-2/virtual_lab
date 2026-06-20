from __future__ import annotations

import json
import logging

from VLAB2.orchestration.state_schema import LabState, safe_jsonable


log = logging.getLogger("virtual_lab")


def save_checkpoint(state: LabState, path: str = "lab_checkpoint.json") -> None:
    """
    Persist current state so a killed job can be resumed.

    Non-fatal: failures are logged but do not interrupt execution.
    """
    try:
        with open(path, "w") as f:
            json.dump(safe_jsonable(dict(state)), f, indent=2)

        log.debug("Checkpoint saved → %s", path)

    except Exception as e:
        log.warning("Failed to save checkpoint: %s", e)