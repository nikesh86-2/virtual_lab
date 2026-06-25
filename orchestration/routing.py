from __future__ import annotations

import logging
import os

from VLAB2.orchestration.state_schema import LabState


log = logging.getLogger("virtual_lab")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _recommendation_line(critique: str) -> str:
    for line in (critique or "").splitlines():
        if line.startswith("RECOMMENDATION:"):
            return line.strip()

    return ""


def _recommendation_value(critique: str) -> str:
    """
    Extract normalized recommendation value from Skeptic critique.

    Examples:
      RECOMMENDATION: ACCEPT
      RECOMMENDATION: REVISE_HYPOTHESIS
      RECOMMENDATION: REDESIGN_OR_REFINE_POSES
    """
    line = _recommendation_line(critique)

    if not line:
        return ""

    return line.replace("RECOMMENDATION:", "").strip().upper()


def _skeptic_recommends_accept(critique: str) -> bool:
    """
    Strict ACCEPT parser.

    Avoids treating strings like DO_NOT_ACCEPT as acceptance.
    """
    value = _recommendation_value(critique)
    return value == "ACCEPT"


def _skeptic_recommends_revise(critique: str) -> bool:
    """
    Treat redesign/refinement recommendations as revision-like.

    This blocks premature convergence when Skeptic explicitly asks for more work.
    """
    value = _recommendation_value(critique)

    return value in {
        "REVISE",
        "REVISE_HYPOTHESIS",
        "REDESIGN",
        "REDESIGN_SEQUENCES",
        "REDESIGN_OR_REFINE_POSES",
        "REFINE_POSES",
        "REVISE_OR_REFINE",
    }


def _conservation_fitness(state: LabState) -> float:
    conservation = state.get("conservation_signal", {}) or {}

    if not isinstance(conservation, dict):
        return 0.0

    try:
        return float(conservation.get("conservation_fitness", 0.0) or 0.0)
    except Exception:
        return 0.0


def _validated_docking_results(state: LabState) -> list[dict]:
    """
    Keep only docking-valid, non-rejected results for the current accepted target.

    Optional interface gate:
      VLAB_REQUIRE_INTERFACE_FOR_CONVERGENCE=1 by default

    When enabled, docking results must also:
      - have valid interface-contact analysis, if the field is present
      - pass interface validation
      - have no steric clash

    Optional current-target gate:
      VLAB_REQUIRE_CURRENT_TARGET_FOR_CONVERGENCE=1 by default

    When enabled, convergence is blocked if state["target_pdb"] is missing.
    This prevents attempted/failed-target rows from accidentally counting as
    convergence evidence.
    """
    results = state.get("binding_results", []) or []
    current_target = state.get("target_pdb")
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

    for r in results:
        if not isinstance(r, dict):
            continue

        if not r.get("valid"):
            continue

        # If current_target is None, do not accidentally validate old/attempted
        # target rows unless explicitly allowed.
        if require_current_target:
            if not current_target:
                continue

            if r.get("target_pdb") != current_target:
                continue

        elif current_target and r.get("target_pdb") != current_target:
            continue

        if not (r.get("dock_valid") or r.get("vina_valid")):
            continue

        if r.get("binding_mode") == "rejected_docking":
            continue

        dock_score = r.get("dock_score", r.get("hdock_score"))

        if dock_score is not None:
            try:
                if float(dock_score) > min_valid_hdock_score:
                    continue
            except Exception:
                continue

        if require_interface:
            # If interface analysis explicitly failed, do not count this pose.
            if r.get("interface_contacts_valid") is False:
                continue

            # Require positive interface pass.
            if r.get("interface_passed") is not True:
                continue

            # Never count sterically clashing poses as validated convergence data.
            if r.get("interface_steric_clash") is True:
                continue

        valid.append(r)

    return valid


def _binding_scores(valid: list[dict]) -> list[float]:
    scores: list[float] = []

    for r in valid:
        value = r.get("binding_rank_score", r.get("dg"))

        if value is None:
            continue

        try:
            scores.append(float(value))
        except Exception:
            continue

    return scores


def _log_no_validated_docking_diagnostics(state: LabState, valid: list[dict]) -> None:
    """
    Log a useful diagnostic when raw docking exists but nothing qualifies for
    convergence after target/interface filtering.
    """
    if valid:
        return

    raw_results = state.get("binding_results", []) or []

    dock_valid_count = sum(
        1
        for r in raw_results
        if isinstance(r, dict) and r.get("dock_valid")
    )

    interface_pass_count = sum(
        1
        for r in raw_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("interface_passed")
        and not r.get("interface_steric_clash")
    )

    clash_count = sum(
        1
        for r in raw_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and r.get("interface_steric_clash")
    )

    current_target = state.get("target_pdb")

    current_target_dock_valid_count = sum(
        1
        for r in raw_results
        if isinstance(r, dict)
        and r.get("dock_valid")
        and current_target
        and r.get("target_pdb") == current_target
    )

    if dock_valid_count:
        log.info(
            "No validated docking results for convergence. "
            "dock_valid=%d current_target_dock_valid=%d interface_pass=%d "
            "steric_clash=%d current_target=%s",
            dock_valid_count,
            current_target_dock_valid_count,
            interface_pass_count,
            clash_count,
            current_target,
        )


def _conservation_allows_convergence(state: LabState) -> bool:
    """
    Optionally require a minimum conservation signal before accepting docking
    convergence.

    Defaults:
      VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE=0

    Set this to 1 for stricter scientific runs.
    """
    require_conservation = _env_bool(
        "VLAB_REQUIRE_CONSERVATION_FOR_CONVERGENCE",
        default=False,
    )

    if not require_conservation:
        return True

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


def should_continue(state: LabState) -> str:
    """
    LangGraph conditional router after Skeptic.

    Returns:
      "continue" -> loop back to PI
      "end"      -> terminate workflow

    Convergence policy:
      - Hard stops at max_iterations.
      - Fold failure forces redesign until max_iterations.
      - Docking convergence requires enough validated docking results.
      - Validated docking can require interface-clean poses by default.
      - Optional conservation gate can block premature docking-only convergence.
      - Skeptic REVISE can block premature convergence unless explicitly allowed.
    """
    critique = state.get("critique", "") or ""
    max_iter = state.get("max_iterations", 3)
    current_iter = state.get("iterations", 0)

    min_valid_for_convergence = _env_int(
        "VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE",
        _env_int("VLAB_MIN_VALID_DOCKINGS_PER_TARGET", 3),
    )

    allow_converge_on_revise = _env_bool(
        "VLAB_ALLOW_CONVERGENCE_ON_REVISE",
        default=False,
    )

    # Fold failure should force redesign until max iterations.
    if (
        state.get("fold_thresholds_passed") is False
        and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
    ):
        if current_iter < max_iter:
            log.info("Continuing: RNA failed fold thresholds.")
            return "continue"

    if current_iter >= max_iter:
        log.info("Ending: reached max_iterations=%s", max_iter)
        return "end"

    valid = _validated_docking_results(state)
    _log_no_validated_docking_diagnostics(state, valid)

    # ------------------------------------------------------------------
    # Docking-based convergence
    # ------------------------------------------------------------------
    if len(valid) >= min_valid_for_convergence:
        scores = _binding_scores(valid)

        if scores:
            try:
                spread = max(scores) - min(scores)
                best_score = min(scores)

                accept_binding = _env_float("VLAB_ACCEPT_BINDING_SCORE", -50.0)
                convergence_spread = _env_float(
                    "VLAB_CONVERGENCE_SPREAD_THRESHOLD",
                    2.0,
                )

                skeptic_says_revise = _skeptic_recommends_revise(critique)

                if (
                    spread < convergence_spread
                    and best_score < accept_binding
                    and state.get("fold_thresholds_passed") is not False
                    and _conservation_allows_convergence(state)
                ):
                    if skeptic_says_revise and not allow_converge_on_revise:
                        log.info(
                            "Not converging despite spread %.3f and best score %.3f "
                            "because Skeptic recommends revision. Set "
                            "VLAB_ALLOW_CONVERGENCE_ON_REVISE=1 to override.",
                            spread,
                            best_score,
                        )
                        return "continue"

                    log.info(
                        "Converged: spread %.3f and best score %.3f pass thresholds.",
                        spread,
                        best_score,
                    )
                    return "end"

                if spread < convergence_spread:
                    log.info(
                        "Not ending despite small spread %.3f because binding, fold, "
                        "conservation, interface, target, or skeptic thresholds are "
                        "not sufficient.",
                        spread,
                    )

            except Exception as e:
                log.warning("Could not evaluate convergence scores: %s", e)

    # ------------------------------------------------------------------
    # Skeptic ACCEPT route
    # ------------------------------------------------------------------
    if _skeptic_recommends_accept(critique):
        if (
            len(valid) >= min_valid_for_convergence
            and state.get("fold_thresholds_passed") is not False
            and _conservation_allows_convergence(state)
        ):
            log.info(
                "Skeptic accepted hypothesis with sufficient validated docking — "
                "terminating loop."
            )
            return "end"

        log.info(
            "Ignoring Skeptic ACCEPT because validated docking, fold, interface, "
            "target, or conservation criteria are insufficient."
        )

    return "continue"