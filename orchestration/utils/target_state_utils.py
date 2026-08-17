"""
target_state_utils.py
====================

Utilities for managing target state: latest batch vs. best validated.

Enforces monotonic best-target promotion, deterministic quality ranking,
and consistent state transitions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


TARGET_STATUS_RANK = {
    "failed_target": 0,
    "unvalidated_target": 1,
    "partial_success_target": 2,
    "accepted_target": 3,
}


def target_status_rank(status: str | None) -> int:
    """
    Return numeric rank for a target status string.

    Higher rank = better/stronger status.
    """
    return TARGET_STATUS_RANK.get(str(status or ""), -1)


def target_quality_key(record: dict[str, Any] | None) -> tuple:
    """
    Return a deterministic quality key for comparing target evaluation records.

    Sorts by:
    1. Status rank (failed < unvalidated < partial < accepted)
    2. Clean interface count (more is better)
    3. Steric clash count (fewer is better)
    4. Aggregate quality score (higher is better)
    5. Best HDOCK score (more negative is better, so negate)
    6. Iteration (earlier records win in exact tie)

    Returns a comparable tuple where higher means better.
    """
    if not record:
        return (-1, -1, -1, float("-inf"), float("-inf"), 0)

    status_rank = target_status_rank(record.get("status"))
    clean_count = int(record.get("clean_interface_count", 0) or 0)
    clash_count = int(record.get("steric_clash_count", 0) or 0)
    quality = float(record.get("aggregate_quality_score", 0.0) or 0.0)

    best_hdock = record.get("best_hdock_relative_score")
    hdock_rank = (
        -float(best_hdock)
        if best_hdock is not None
        else float("-inf")
    )

    iteration = int(record.get("iteration", 0) or 0)

    return (
        status_rank,
        clean_count,
        -clash_count,
        quality,
        hdock_rank,
        -iteration,  # Tie-breaker: earlier record wins
    )


def is_validated_target_record(record: dict[str, Any] | None) -> bool:
    """
    Check if a record represents a validated (accepted) target.

    A valid record must have:
    - accepted=True
    - interface_validated=True
    - status="accepted_target"
    - docking_valid_count >= docking_required_count
    - clean_interface_count >= clean_interface_required_count
    """
    if not record:
        return False

    return bool(
        record.get("accepted")
        and record.get("interface_validated")
        and record.get("status") == "accepted_target"
        and int(record.get("docking_valid_count", 0) or 0)
        >= int(record.get("docking_required_count", 0) or 0)
        and int(record.get("clean_interface_count", 0) or 0)
        >= int(record.get("clean_interface_required_count", 0) or 0)
    )


def should_promote_best_target(
    candidate: dict[str, Any] | None,
    incumbent: dict[str, Any] | None,
) -> bool:
    """
    Decide if candidate should replace incumbent as best validated target.

    Rules:
    - Only validated candidates can be promoted
    - If incumbent is not validated, any validated candidate wins
    - If both are validated, higher quality_key wins
    - If quality_keys are equal, incumbent is retained (no unnecessary change)
    """
    if not is_validated_target_record(candidate):
        return False

    if not is_validated_target_record(incumbent):
        return True

    return target_quality_key(candidate) > target_quality_key(incumbent)


def make_batch_id(payload: dict) -> str:
    """
    Generate deterministic batch identity hash.

    Uses canonical JSON representation of batch signature.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_binding_result_id(result: dict) -> str:
    """
    Generate stable identifier for a binding result row.

    Based on target, sequence, cache_key, and HDOCK run ID.
    """
    identity = {
        "target_pdb": str(result.get("target_pdb") or "").upper(),
        "sequence": str(result.get("sequence") or "").upper(),
        "cache_key": str(result.get("cache_key") or ""),
        "hdock_run_id": str(result.get("hdock_run_id") or ""),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def calculate_target_quality_score(
    clean_count: int,
    clash_count: int,
    valid_count: int,
    interface_quality_scores: list[float],
) -> float:
    """
    Calculate deterministic aggregate quality score for a target.

    Formula:
    - 2.0 * clean_interface_count (strong positive weight)
    - 1.0 * steric_clash_count (strong negative weight)
    + 0.25 * docking_valid_count (weak positive weight)
    + mean(interface_quality_scores) (average per-pose quality)

    Do not include Vina scores in this calculation.
    """
    mean_interface = (
        sum(interface_quality_scores) / len(interface_quality_scores)
        if interface_quality_scores
        else 0.0
    )

    return (
        2.0 * float(clean_count)
        - 1.0 * float(clash_count)
        + 0.25 * float(valid_count)
        + mean_interface
    )
