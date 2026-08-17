from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from VLAB2.orchestration.state_schema import LabState


log = logging.getLogger("virtual_lab")


# ---------------------------------------------------------------------------
# Environment and parsing helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalise_target_id(value: Any) -> str:
    """Normalise a PDB ID or PDB path into an uppercase target key."""
    text = str(value or "").strip()

    if not text:
        return ""

    try:
        text = Path(text).stem
    except Exception:
        pass

    return text.upper()


def _recommendation_line(critique: str) -> str:
    for line in (critique or "").splitlines():
        if line.strip().upper().startswith("RECOMMENDATION:"):
            return line.strip()

    return ""


def _recommendation_value(critique: str) -> str:
    """Extract the normalised recommendation value from Skeptic critique."""
    line = _recommendation_line(critique)

    if not line:
        return ""

    return line.split(":", 1)[1].strip().upper()


def _skeptic_recommends_accept(critique: str) -> bool:
    """Strict ACCEPT parser that does not match DO_NOT_ACCEPT."""
    return _recommendation_value(critique) == "ACCEPT"


def _skeptic_recommends_revise(critique: str) -> bool:
    """Treat redesign, refinement, and evidence requests as revision-like."""
    value = _recommendation_value(critique)

    return value in {
        "REVISE",
        "REVISE_HYPOTHESIS",
        "REDESIGN",
        "REDESIGN_SEQUENCES",
        "REDESIGN_OR_REFINE_POSES",
        "REFINE_POSES",
        "REVISE_OR_REFINE",
        "MORE_MD",
        "MORE_DOCKING",
        "MORE_INTERFACE_VALIDATION",
    }


# ---------------------------------------------------------------------------
# Conservation and target gates
# ---------------------------------------------------------------------------


def _conservation_fitness(state: LabState) -> float:
    conservation = state.get("conservation_signal", {}) or {}

    if not isinstance(conservation, dict):
        return 0.0

    try:
        value = float(conservation.get("conservation_fitness", 0.0) or 0.0)
    except Exception:
        return 0.0

    return value if math.isfinite(value) else 0.0


def _conservation_allows_convergence(state: LabState) -> bool:
    """Optionally require both a quality-passed MSA and minimum fitness."""
    require_conservation = _env_bool(
        "VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE",
        default=False,
    )

    if not require_conservation:
        return True

    conservation = state.get("conservation_signal", {}) or {}
    quality_passed = state.get("bioinfo_quality_passed")

    if quality_passed is None and isinstance(conservation, dict):
        quality_passed = conservation.get("quality_passed")

    if quality_passed is not True:
        reasons = (
            state.get("bioinfo_quality_reasons")
            or (
                conservation.get("quality_reasons", [])
                if isinstance(conservation, dict)
                else []
            )
        )
        log.info(
            "Not converging: bioinformatics quality did not pass. reasons=%s",
            reasons,
        )
        return False

    fitness = _conservation_fitness(state)
    min_fitness = _env_float("VLAB_MIN_CONSERVATION_FOR_ACCEPT", 0.2)

    if fitness >= min_fitness:
        return True

    log.info(
        "Not converging: conservation fitness %.3f is below required %.3f.",
        fitness,
        min_fitness,
    )
    return False


def _target_allows_convergence(state: LabState) -> bool:
    """Require a final accepted target unless explicitly configured otherwise."""
    require_accepted = _env_bool(
        "VLAB_REQUIRE_ACCEPTED_TARGET_FOR_CONVERGENCE",
        default=True,
    )
    target_id = _normalise_target_id(
        state.get("target_pdb_id") or state.get("target_pdb")
    )

    if not require_accepted:
        if target_id:
            return True

        log.info("Not converging: no current target is available.")
        return False

    # Use best validated target for acceptance criteria, fall back to latest
    best_target = state.get("best_validated_target_evaluation")
    latest_target = state.get("latest_target_evaluation")
    target_record = best_target or latest_target

    status = target_record.get("status") if target_record else state.get("target_status")
    status_reason = target_record.get("status_reason") if target_record else state.get("target_status_reason")

    if target_id and status == "accepted_target":
        return True

    log.info(
        "Not converging: target=%s status=%s reason=%s; accepted_target is required.",
        target_id or None,
        status,
        status_reason,
    )
    return False


# ---------------------------------------------------------------------------
# Docking convergence helpers
# ---------------------------------------------------------------------------


def _validated_docking_results(state: LabState) -> list[dict]:
    """
    Return docking-valid, non-rejected rows for the current target.

    By default, each returned row must also have passed interface analysis and
    must not contain a steric clash.
    """
    results = state.get("binding_results", []) or []
    current_target = _normalise_target_id(
        state.get("target_pdb_id") or state.get("target_pdb")
    )
    min_valid_hdock_score = _env_float("VLAB_MIN_VALID_HDOCK_SCORE", -30.0)
    require_interface = _env_bool(
        "VLAB_REQUIRE_INTERFACE_FOR_CONVERGENCE",
        default=True,
    )
    require_current_target = _env_bool(
        "VLAB_REQUIRE_CURRENT_TARGET_FOR_CONVERGENCE",
        default=True,
    )

    valid: list[dict] = []

    for row in results:
        if not isinstance(row, dict):
            continue

        if row.get("binding_mode") == "rejected_docking":
            continue

        if not (
            row.get("dock_valid") is True
            or row.get("vina_valid") is True
            or row.get("valid") is True
        ):
            continue

        row_target = _normalise_target_id(
            row.get("target_pdb_id") or row.get("target_pdb")
        )

        if require_current_target:
            if not current_target:
                continue
            if row_target and row_target != current_target:
                continue
        elif current_target and row_target and row_target != current_target:
            continue

        dock_score = row.get("dock_score", row.get("hdock_score"))

        if dock_score is not None:
            try:
                dock_score_value = float(dock_score)
            except Exception:
                continue

            if not math.isfinite(dock_score_value):
                continue

            if dock_score_value > min_valid_hdock_score:
                continue

        if require_interface:
            if row.get("interface_contacts_valid") is False:
                continue
            if row.get("interface_passed") is not True:
                continue
            if row.get("interface_steric_clash") is True:
                continue

        valid.append(row)

    return valid


def _binding_scores(valid: list[dict]) -> list[float]:
    """Extract finite binding rank scores from validated rows."""
    scores: list[float] = []

    for row in valid:
        value = row.get("binding_rank_score", row.get("dg"))

        if value is None:
            value = row.get("dock_score", row.get("hdock_score"))

        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if not math.isfinite(value):
            continue

        scores.append(value)

    return scores


def _log_no_validated_docking_diagnostics(
    state: LabState,
    valid: list[dict],
) -> None:
    """Explain why raw docking rows failed convergence validation."""
    if valid:
        return

    raw_results = state.get("binding_results", []) or []
    current_target = _normalise_target_id(
        state.get("target_pdb_id") or state.get("target_pdb")
    )

    dock_valid_count = sum(
        1
        for row in raw_results
        if isinstance(row, dict)
        and (
            row.get("dock_valid") is True
            or row.get("vina_valid") is True
            or row.get("valid") is True
        )
    )
    interface_pass_count = sum(
        1
        for row in raw_results
        if isinstance(row, dict)
        and row.get("dock_valid") is True
        and row.get("interface_passed") is True
        and row.get("interface_steric_clash") is not True
    )
    clash_count = sum(
        1
        for row in raw_results
        if isinstance(row, dict)
        and row.get("dock_valid") is True
        and row.get("interface_steric_clash") is True
    )
    current_target_dock_valid_count = sum(
        1
        for row in raw_results
        if isinstance(row, dict)
        and row.get("dock_valid") is True
        and current_target
        and _normalise_target_id(
            row.get("target_pdb_id") or row.get("target_pdb")
        ) == current_target
    )

    if dock_valid_count:
        log.info(
            "No validated docking results for convergence. "
            "dock_valid=%d current_target_dock_valid=%d interface_pass=%d "
            "steric_clash=%d current_target=%s target_status=%s",
            dock_valid_count,
            current_target_dock_valid_count,
            interface_pass_count,
            clash_count,
            current_target or None,
            state.get("target_status"),
        )


# ---------------------------------------------------------------------------
# Main post-Skeptic router
# ---------------------------------------------------------------------------


def should_continue(state: LabState) -> str:
    """LangGraph conditional router after Skeptic."""
    critique = state.get("critique", "") or ""
    max_iter = int(state.get("max_iterations", 3) or 3)
    current_iter = int(state.get("iterations", 0) or 0)

    # Never evaluate stale docking evidence as convergence after a structural
    # failure. The PI may perform a controlled redesign until max iterations.
    if state.get("structural_status") == "failed":
        if current_iter >= max_iter:
            log.error(
                "Ending after structural failure at max_iterations=%d: %s",
                max_iter,
                state.get("structural_error"),
            )
            return "end"

        log.warning(
            "Continuing to controlled redesign after structural failure: %s",
            state.get("structural_error"),
        )
        return "continue"

    # Hard iteration limit takes precedence over further redesign requests.
    if current_iter >= max_iter:
        log.info("Ending: reached max_iterations=%s", max_iter)
        return "end"

    # A known fold failure forces redesign while iterations remain.
    if (
        state.get("fold_thresholds_passed") is False
        and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
    ):
        log.info("Continuing: RNA failed fold thresholds.")
        return "continue"

    min_valid_for_convergence = _env_int(
        "VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE",
        _env_int("VLAB_MIN_VALID_DOCKINGS_PER_TARGET", 2),
    )
    min_clean_for_convergence = _env_int(
        "VLAB_MIN_CLEAN_INTERFACES_FOR_CONVERGENCE",
        min_valid_for_convergence,
    )
    required_count = max(
        1,
        min_valid_for_convergence,
        min_clean_for_convergence,
    )
    allow_converge_on_revise = _env_bool(
        "VLAB_ALLOW_CONVERGENCE_ON_REVISE",
        default=False,
    )

    valid = _validated_docking_results(state)
    _log_no_validated_docking_diagnostics(state, valid)

    if len(valid) >= required_count:
        scores = _binding_scores(valid)

        if len(scores) < required_count:
            log.info(
                "Not converging: validated_rows=%d finite_scores=%d required=%d.",
                len(valid),
                len(scores),
                required_count,
            )
        else:
            spread = max(scores) - min(scores)
            best_score = min(scores)
            accept_binding = _env_float("VLAB_ACCEPT_BINDING_SCORE", -50.0)
            convergence_spread = _env_float(
                "VLAB_CONVERGENCE_SPREAD_THRESHOLD",
                2.0,
            )
            skeptic_says_revise = _skeptic_recommends_revise(critique)

            converged = (
                spread <= convergence_spread
                and best_score <= accept_binding
                and state.get("fold_thresholds_passed") is True
                and _target_allows_convergence(state)
                and _conservation_allows_convergence(state)
            )

            if converged:
                if skeptic_says_revise and not allow_converge_on_revise:
                    log.info(
                        "Not converging despite passing physical metrics because "
                        "Skeptic recommends %s.",
                        _recommendation_value(critique),
                    )
                    return "continue"

                log.info(
                    "Converged: target=%s clean_rows=%d spread=%.3f "
                    "best_score=%.3f.",
                    _normalise_target_id(
                        state.get("target_pdb_id") or state.get("target_pdb")
                    ),
                    len(valid),
                    spread,
                    best_score,
                )
                return "end"

            if spread <= convergence_spread:
                log.info(
                    "Not ending despite small spread %.3f because binding, fold, "
                    "conservation, interface, target, or Skeptic thresholds are "
                    "not sufficient.",
                    spread,
                )
    else:
        log.info(
            "Not converging: clean validated docking rows=%d required=%d.",
            len(valid),
            required_count,
        )

    # Skeptic ACCEPT remains subordinate to objective validation gates.
    if _skeptic_recommends_accept(critique):
        scores = _binding_scores(valid)

        if (
            len(valid) >= required_count
            and len(scores) >= required_count
            and state.get("fold_thresholds_passed") is True
            and _target_allows_convergence(state)
            and _conservation_allows_convergence(state)
        ):
            log.info(
                "Skeptic accepted the hypothesis with sufficient validated "
                "docking and interface evidence; terminating loop."
            )
            return "end"

        log.info(
            "Ignoring Skeptic ACCEPT because validated docking, fold, interface, "
            "target, or conservation criteria are insufficient."
        )

    return "continue"


# ---------------------------------------------------------------------------
# Post-protein inhibitor router
# ---------------------------------------------------------------------------


def _partial_success_target_id(state: LabState) -> str:
    """Extract the first available partial-success target ID."""
    for item in state.get("partial_success_targets", []) or []:
        if isinstance(item, dict):
            candidate = item.get("target_pdb") or item.get("pdb_id")
        else:
            candidate = item

        target_id = _normalise_target_id(candidate)
        if target_id:
            return target_id

    return ""


def _inhibitor_should_run(state: LabState) -> str:
    """Route from protein evaluation to inhibitor screening or Skeptic."""
    if not _env_bool("VLAB_INHIBITOR_ENABLED", default=False):
        log.info("Inhibitor screening disabled (VLAB_INHIBITOR_ENABLED != 1)")
        return "skeptic"

    # Use best validated target for routing, fall back to latest
    best_target = state.get("best_validated_target_evaluation")
    latest_target = state.get("latest_target_evaluation")
    target_record = best_target or latest_target
    
    target_status = target_record.get("status") if target_record else state.get("target_status")
    target_id = _normalise_target_id(
        state.get("target_pdb_id") or state.get("target_pdb")
    )

    if target_status == "failed_target":
        target_id = ""

    if target_status == "partial_success_target":
        allow_partial = _env_bool(
            "VLAB_ALLOW_INHIBITOR_ON_PARTIAL_TARGET",
            default=True,
        )

        if not allow_partial:
            log.info(
                "Skipping inhibitor: partial-success target screening is disabled."
            )
            return "skeptic"

        target_id = target_id or _partial_success_target_id(state)
        log.info(
            "Using partial-success target for exploratory inhibitor screening: %s",
            target_id or None,
        )

    if not target_id:
        target_id = _partial_success_target_id(state)

        if target_id:
            if not _env_bool(
                "VLAB_ALLOW_INHIBITOR_ON_PARTIAL_TARGET",
                default=True,
            ):
                log.info(
                    "Skipping inhibitor: only a partial-success target is available."
                )
                return "skeptic"

            log.info(
                "Using remembered partial-success target for exploratory "
                "inhibitor screening: %s",
                target_id,
            )

    if not target_id:
        log.info(
            "Skipping inhibitor: no accepted or partial-success target is available."
        )
        return "skeptic"

    interface = state.get("interface_contacts") or {}
    interface_residues = interface.get("interface_residues") or []

    if not interface_residues:
        log.info(
            "Skipping inhibitor: no interface residues in state; "
            "interface_contacts=%s",
            {
                key: interface.get(key)
                for key in (
                    "interface_contacts_valid",
                    "contacts_valid",
                    "interface_residues",
                )
            },
        )
        return "skeptic"

    clean_rows = []
    current_iteration = state.get("docking_iteration", 0)

    for row in state.get("binding_results", []) or []:
        if not isinstance(row, dict):
            continue

        row_target = _normalise_target_id(
            row.get("target_pdb_id") or row.get("target_pdb")
        )

        if row_target and row_target != target_id:
            continue

        # Only count clean rows from the current docking iteration
        row_iteration = row.get("docking_iteration")
        if row_iteration is not None and int(row_iteration) != current_iteration:
            continue

        if (
            row.get("dock_valid") is True
            and row.get("interface_passed") is True
            and row.get("interface_steric_clash") is not True
        ):
            clean_rows.append(row)

    if not clean_rows:
        log.info(
            "Skipping inhibitor: no clean RNA-protein interface row is "
            "available for target %s.",
            target_id,
        )
        return "skeptic"

    log.info(
        "Inhibitor screening enabled: target=%s status=%s clean_rows=%d.",
        target_id,
        target_status,
        len(clean_rows),
    )
    return "inhibitor"


# ---------------------------------------------------------------------------
# Post-structural router
# ---------------------------------------------------------------------------


def after_structural(state: LabState) -> str:
    """
    LangGraph conditional router after Structural.

    If structural design or folding failed, skip MD and protein evaluation
    and return directly to the PI for controlled redesign.  This prevents
    downstream agents from processing invalid sequences.
    """
    status = state.get("structural_status")

    if status == "failed":
        log.warning(
            "Skipping MD/protein evaluation after structural failure: %s",
            state.get("structural_error"),
        )
        return "pi"

    log.info(
        "Structural succeeded (status=%s); proceeding to MD evaluation.",
        status,
    )
    return "md"
