from __future__ import annotations

import logging
import os

from VLAB2.orchestration.state_schema import LabState
from VLAB2.orchestration.utils.docking_utils import safe_binding_rank


log = logging.getLogger("virtual_lab")


def should_continue(state: LabState) -> str:
    """
    LangGraph conditional router after Skeptic.

    Returns:
      "continue" -> loop back to PI
      "end"      -> terminate workflow
    """
    
    min_valid_for_convergence = int(
        os.getenv(
            "VLAB_MIN_VALID_DOCKINGS_FOR_CONVERGENCE",
            os.getenv("VLAB_MIN_VALID_DOCKINGS_PER_TARGET", "3"),
        ))

    critique = state.get("critique", "") or ""
    max_iter = state.get("max_iterations", 3)
    current_iter = state.get("iterations", 0)

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

    results = state.get("binding_results", []) or []
    current_target = state.get("target_pdb")

    min_valid_hdock_score = float(os.getenv("VLAB_MIN_VALID_HDOCK_SCORE", "-30"))

    valid = []

    for r in results:
        if not isinstance(r, dict):
            continue

        if not r.get("valid"):
            continue

        if current_target and r.get("target_pdb") != current_target:
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

        valid.append(r)

    if len(valid) >= min_valid_for_convergence:
        scores = [
            r.get("binding_rank_score", r.get("dg"))
            for r in valid
            if r.get("binding_rank_score", r.get("dg")) is not None
        ]

        if scores:
            try:
                scores = [float(x) for x in scores]
                spread = max(scores) - min(scores)
                best_score = min(scores)

                accept_binding = float(
                    os.getenv("VLAB_ACCEPT_BINDING_SCORE", "-50")
                )

                if (
                    spread < float(os.getenv("VLAB_CONVERGENCE_SPREAD_THRESHOLD", "2.0"))
                    and best_score < accept_binding
                    and state.get("fold_thresholds_passed") is not False
                ):
                    log.info(
                        "Converged: spread %.3f and best score %.3f pass thresholds.",
                        spread,
                        best_score,
                    )
                    return "end"

                if spread < 2.0:
                    log.info(
                        "Not ending despite small spread %.3f because binding/fold "
                        "thresholds are not sufficient.",
                        spread,
                    )

            except Exception as e:
                log.warning("Could not evaluate convergence scores: %s", e)

    for line in critique.splitlines():
        if line.startswith("RECOMMENDATION:") and "ACCEPT" in line:
            if len(valid) >= min_valid_for_convergence and state.get("fold_thresholds_passed") is not False:
                log.info("Skeptic accepted hypothesis with sufficient validated docking — terminating loop.")
                return "end"

            log.info(
                "Ignoring Skeptic ACCEPT because validated docking/fold criteria are insufficient."
            )
    return "continue"