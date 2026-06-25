from __future__ import annotations

import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage

from VLAB2.optimisation.final_rna_design_system import run_system
from VLAB2.utils.pareto_analysis import analyse_pareto

from VLAB2.orchestration.failure_memory import FailureMemory
from VLAB2.orchestration.literature_memory import LiteratureMemory
from VLAB2.orchestration.llm import get_llm
from VLAB2.orchestration.state_schema import LabState

from VLAB2.orchestration.utils.env_utils import getenv_float
from VLAB2.orchestration.utils.fold_utils import motif_summary_from_state
from VLAB2.orchestration.utils.hypothesis_utils import (
    critique_signal_summary_for_pi,
    fallback_qualitative_hypothesis,
    hypothesis_contains_bad_metric_threshold,
)
from VLAB2.orchestration.utils.objective_utils import adjust_weights
from VLAB2.orchestration.utils.sequence_utils import (
    clean_rna,
    dedupe_rna_sequences,
)
from VLAB2.orchestration.utils.text_utils import (
    clean_llm_output,
    truncate_str,
)


log = logging.getLogger("virtual_lab")

__all__ = ["pi_agent"]


# ---------------------------------------------------------------------------
# Critique / feedback parsing helpers
# ---------------------------------------------------------------------------

def _parse_best_binding_score_from_critique(critique: str) -> float | None:
    """
    Extract best_binding_score from Skeptic critique.

    Supports:
        best_binding_score=-71.14
        best_binding_score = -71.14
    """
    if not critique:
        return None

    try:
        m = re.search(
            r"best_binding_score\s*=\s*([-+]?\d+(?:\.\d+)?)",
            critique,
            flags=re.IGNORECASE,
        )

        if m:
            return float(m.group(1))

    except Exception:
        return None

    return None


def _parse_n_valid_from_critique(critique: str) -> int | None:
    """
    Extract n=<count> from Skeptic ENERGY line if present.
    """
    if not critique:
        return None

    try:
        m = re.search(
            r"\bn\s*=\s*(\d+)",
            critique,
            flags=re.IGNORECASE,
        )

        if m:
            return int(m.group(1))

    except Exception:
        return None

    return None


def _parse_score_spread_from_critique(critique: str) -> float | None:
    """
    Extract score spread from Skeptic critique.

    Supports:
        spread=68.99
        spread = 68.99
    """
    if not critique:
        return None

    try:
        m = re.search(
            r"spread\s*=\s*([-+]?\d+(?:\.\d+)?)",
            critique,
            flags=re.IGNORECASE,
        )

        if m:
            return float(m.group(1))

    except Exception:
        return None

    return None


def _normalise_failed_target_records(records: list) -> list[dict]:
    """
    Normalise failed_target_pdbs to list[dict] while supporting legacy list[str].

    Output shape:
      {
        "pdb_id": "2GE7",
        "reason": "only_one_clean_pose",
        "metadata": {...}
      }
    """
    out: dict[str, dict] = {}

    for item in records or []:
        if isinstance(item, dict):
            pdb = str(item.get("pdb_id", "") or "").strip().upper()

            if not pdb:
                continue

            out[pdb] = {
                "pdb_id": pdb,
                "reason": item.get("reason", "unknown"),
                "metadata": item.get("metadata", {}),
            }

        elif item:
            pdb = str(item).strip().upper()

            if pdb:
                out[pdb] = {
                    "pdb_id": pdb,
                    "reason": "unknown",
                    "metadata": {},
                }

    return list(out.values())


# ---------------------------------------------------------------------------
# Target reset logic
# ---------------------------------------------------------------------------

def _maybe_reset_target_on_high_spread(state: LabState) -> str | None:
    """
    Conservatively reset target_pdb when Skeptic reports problematic docking spread.

    Behaviour:
      - Do not reset a target solely because spread is moderate/high.
      - Keep a target if the best HDOCK-relative score is already favourable.
      - Reset only when spread is high AND best binding is weak.
      - Optionally reset on extreme spread regardless of best score.
      - Do not mark reset targets as failed unless configured.

    Env controls:
      VLAB_TARGET_RESET_ON_HIGH_SPREAD=1
      VLAB_TARGET_RESET_REQUIRE_WEAK_BINDING=1
      VLAB_ACCEPT_BINDING_SCORE=-50
      VLAB_MAX_ACCEPT_SCORE_SPREAD=10.0
      VLAB_TARGET_RESET_EXTREME_SPREAD_MULTIPLIER=2.0
      VLAB_TARGET_MARK_RESET_AS_FAILED=0
      VLAB_TARGET_RESET_MIN_VALID_N=3
    """
    target_pdb = state.get("target_pdb")
    critique = state.get("critique", "") or ""

    if not target_pdb:
        return target_pdb

    if os.getenv("VLAB_TARGET_RESET_ON_HIGH_SPREAD", "1").strip() != "1":
        return target_pdb

    score_spread = _parse_score_spread_from_critique(critique)
    best_score = _parse_best_binding_score_from_critique(critique)
    n_valid = _parse_n_valid_from_critique(critique)

    max_accept_spread = getenv_float("VLAB_MAX_ACCEPT_SCORE_SPREAD", 10.0)
    accept_binding_score = getenv_float("VLAB_ACCEPT_BINDING_SCORE", -50.0)
    extreme_multiplier = getenv_float(
        "VLAB_TARGET_RESET_EXTREME_SPREAD_MULTIPLIER",
        2.0,
    )

    require_weak_binding = (
        os.getenv("VLAB_TARGET_RESET_REQUIRE_WEAK_BINDING", "1").strip() == "1"
    )

    mark_reset_as_failed = (
        os.getenv("VLAB_TARGET_MARK_RESET_AS_FAILED", "0").strip() == "1"
    )

    min_valid_n = int(os.getenv("VLAB_TARGET_RESET_MIN_VALID_N", "3"))

    if score_spread is None:
        return target_pdb

    if n_valid is not None and n_valid < min_valid_n:
        log.info(
            "Not resetting target_pdb=%s because n_valid=%s < %s.",
            target_pdb,
            n_valid,
            min_valid_n,
        )
        return target_pdb

    high_spread = score_spread > max_accept_spread
    extreme_spread = score_spread > (max_accept_spread * extreme_multiplier)

    if not high_spread:
        return target_pdb

    binding_is_favourable = (
        best_score is not None
        and best_score < accept_binding_score
    )

    if require_weak_binding and binding_is_favourable and not extreme_spread:
        log.info(
            "Keeping target_pdb=%s despite spread %.3f because best_binding_score %.3f "
            "passes accept threshold %.3f. Redesigning sequences against same target.",
            target_pdb,
            score_spread,
            best_score,
            accept_binding_score,
        )
        return target_pdb

    log.info(
        "Resetting target_pdb=%s because spread %.3f is high%s. best_binding_score=%s, "
        "accept_threshold=%.3f.",
        target_pdb,
        score_spread,
        " and extreme" if extreme_spread else "",
        best_score,
        accept_binding_score,
    )

    if mark_reset_as_failed:
        failed_targets = _normalise_failed_target_records(
            list(state.get("failed_target_pdbs", []) or [])
        )

        failed_targets.append(
            {
                "pdb_id": str(target_pdb).upper(),
                "reason": "reset_high_spread",
                "metadata": {
                    "score_spread": score_spread,
                    "best_binding_score": best_score,
                    "accept_binding_score": accept_binding_score,
                    "max_accept_spread": max_accept_spread,
                },
            }
        )

        state["failed_target_pdbs"] = _normalise_failed_target_records(failed_targets)

    else:
        log.info(
            "Target %s was reset but not marked failed because "
            "VLAB_TARGET_MARK_RESET_AS_FAILED=0.",
            target_pdb,
        )

    state["target_pdb"] = None

    return None


# ---------------------------------------------------------------------------
# Joint physics feedback
# ---------------------------------------------------------------------------

def _compute_joint_physics_feedback(state: LabState) -> dict:
    """
    Combine docking, interface, and MD signals into unified PI feedback.

    Returns heuristic signals in [0, 1]:

      binding_signal:
        Uses binding_rank_score/dg. More negative docking/ranking scores are better.

      stability_signal:
        Uses MD min_energy. More negative values are treated as more stable.

      fluctuation_signal:
        Uses MD energy_fluctuation. Lower fluctuation is better.

      interface_signal:
        Uses interface quality, clean-interface count, and clash penalties.

    These are optimisation heuristics, not physical free-energy estimates.
    """
    # ------------------------------------------------------------------
    # Docking / binding signal
    # ------------------------------------------------------------------
    binding_results = state.get("binding_results", []) or []
    dock_scores = []

    for r in binding_results:
        if not isinstance(r, dict):
            continue

        if not (r.get("dock_valid") or r.get("vina_valid")):
            continue

        if r.get("binding_mode") == "rejected_docking":
            continue

        raw_hdock = r.get("dock_score", r.get("hdock_score"))

        if raw_hdock is not None:
            try:
                raw_hdock = float(raw_hdock)
            except Exception:
                raw_hdock = None

        min_valid_hdock_score = getenv_float("VLAB_MIN_VALID_HDOCK_SCORE", -30.0)

        if raw_hdock is not None and raw_hdock > min_valid_hdock_score:
            continue

        value = r.get("binding_rank_score", r.get("dg"))

        if value is None:
            continue

        try:
            dock_scores.append(float(value))
        except Exception:
            pass

    if dock_scores:
        best_dock = min(dock_scores)

        # More negative = better. HDOCK-relative/rank scale heuristic.
        binding_signal = min(1.0, max(0.0, abs(best_dock) / 100.0))
        score_spread = max(dock_scores) - min(dock_scores)

    else:
        best_dock = None
        binding_signal = 0.0
        score_spread = None

    # ------------------------------------------------------------------
    # MD stability / fluctuation signal
    # ------------------------------------------------------------------
    md_results = state.get("md_results", []) or []

    min_energies = []
    mean_energies = []
    fluctuations = []

    for r in md_results:
        if not isinstance(r, dict):
            continue

        res = r.get("result", {}) or {}

        if not isinstance(res, dict):
            continue

        if not res.get("valid"):
            continue

        if res.get("min_energy") is not None:
            try:
                min_energies.append(float(res["min_energy"]))
            except Exception:
                pass

        if res.get("mean_energy") is not None:
            try:
                mean_energies.append(float(res["mean_energy"]))
            except Exception:
                pass

        if res.get("energy_fluctuation") is not None:
            try:
                fluctuations.append(float(res["energy_fluctuation"]))
            except Exception:
                pass

    if min_energies:
        best_min_energy = min(min_energies)
        stability_signal = min(1.0, max(0.0, abs(best_min_energy) / 50.0))
    else:
        best_min_energy = None
        stability_signal = 0.0

    if fluctuations:
        mean_fluctuation = sum(fluctuations) / max(1, len(fluctuations))
        fluctuation_signal = 1.0 / (1.0 + mean_fluctuation)
    else:
        mean_fluctuation = None
        fluctuation_signal = 0.0

    # ------------------------------------------------------------------
    # Interface signal
    # ------------------------------------------------------------------
    interface_scores = []
    clean_interface_count = 0
    clash_count = 0

    for r in binding_results:
        if not isinstance(r, dict):
            continue

        if not r.get("dock_valid"):
            continue

        if r.get("interface_quality_score") is not None:
            try:
                interface_scores.append(float(r["interface_quality_score"]))
            except Exception:
                pass

        if (
            r.get("interface_passed") is True
            and r.get("interface_steric_clash") is not True
        ):
            clean_interface_count += 1

        if r.get("interface_steric_clash") is True:
            clash_count += 1

    if interface_scores:
        interface_signal = max(0.0, min(1.0, max(interface_scores)))
    else:
        interface_signal = 0.0

    if clean_interface_count:
        interface_signal = max(interface_signal, 0.8)

    if clash_count and not clean_interface_count:
        interface_signal *= 0.5

    return {
        "has_binding": bool(dock_scores),
        "has_md": bool(min_energies or fluctuations),
        "binding_signal": binding_signal,
        "stability_signal": stability_signal,
        "fluctuation_signal": fluctuation_signal,
        "best_binding_score": best_dock,
        "binding_score_spread": score_spread,
        "best_md_min_energy": best_min_energy,
        "mean_md_fluctuation": mean_fluctuation,
        "interface_signal": interface_signal,
        "clean_interface_count": clean_interface_count,
        "interface_clash_count": clash_count,
    }


# ---------------------------------------------------------------------------
# Hypothesis refinement
# ---------------------------------------------------------------------------

def _refine_hypothesis_from_critique(
    state: LabState,
    critique: str,
) -> str | None:
    """
    Use the LLM to rewrite the hypothesis based on qualitative Skeptic signals.

    Exact metric thresholds are deliberately stripped/blocked to avoid the PI
    copying previous exact score spread/best score values into the hypothesis.
    """
    if not critique:
        return None

    try:
        from VLAB2.orchestration.skeptic_parser import parse_skeptic_output

        _ = parse_skeptic_output(critique)

        llm = get_llm(temperature=0.0)

        recommendation_line = ""

        for line in critique.splitlines():
            if line.startswith("RECOMMENDATION:"):
                recommendation_line = line.replace("RECOMMENDATION:", "").strip()

        qualitative_critique = critique_signal_summary_for_pi(critique)

        prompt = [
            SystemMessage(
                content=(
                    "You are a computational biophysics Principal Investigator.\n\n"
                    "Rewrite the hypothesis as a single falsifiable qualitative prediction "
                    "grounded in the evidence below.\n\n"
                    "Output exactly one sentence.\n"
                    "Use HDOCK-relative score terminology when binding data are from HDOCK.\n"
                    "Do not call HDOCK scores kcal/mol.\n"
                    "Do not include exact numeric cutoffs or previous metric values.\n"
                    "Do not write inequalities such as '<', '>', 'less than 9.793', "
                    "or 'lower than -71.865'.\n"
                    "Prefer comparative language such as stronger, weaker, less variable, "
                    "more conserved, more stable, or improved.\n"
                    "Do not use may, could, potentially, or might."
                )
            ),
            HumanMessage(
                content=(
                    f"Previous hypothesis:\n"
                    f"{truncate_str(state.get('hypothesis', ''), 600)}\n\n"
                    f"Skeptic recommendation: {recommendation_line or 'N/A'}\n\n"
                    f"Qualitative critique signals, with exact metric thresholds removed:\n"
                    f"{qualitative_critique}\n\n"
                    f"Literature-derived target/motif policy:\n"
                    f"{truncate_str(state.get('literature_policy_text', ''), 800)}\n\n"
                    f"Write a qualitative comparative hypothesis. "
                    f"Do not include exact numeric cutoffs, exact score-spread values, "
                    f"or exact previous best_binding_score values."
                )
            ),
        ]

        resp = llm.invoke(prompt)
        candidate = clean_llm_output(resp.content)

        if candidate:
            if hypothesis_contains_bad_metric_threshold(candidate):
                log.warning(
                    "PI hypothesis copied exact metric thresholds; replacing with qualitative fallback."
                )
                candidate = fallback_qualitative_hypothesis(state)

            log.info("Refined hypothesis: %s", truncate_str(candidate, 200))
            return candidate

    except Exception as e:
        log.warning("Hypothesis refinement failed: %s", e)

    return None


# ---------------------------------------------------------------------------
# Objective construction
# ---------------------------------------------------------------------------

def _build_combined_objectives(
    state: LabState,
    parsed: dict,
    extra_objectives: dict,
    failure_weights: dict,
    conservation_fitness: float,
    critique: str,
    score_spread: float | None,
    joint_feedback: dict,
) -> dict:
    """
    Build and normalise combined optimisation objectives for NSGA-II.

    Preserves original pressure sources:
      - conservation
      - Skeptic/objective injection
      - fold failure pressure
      - high docking spread pressure
      - FailureMemory pressures

    Adds joint MD+docking/interface feedback:
      - weak binding -> binding + diversity pressure
      - good binding but unstable MD -> structure + thermo pressure
      - high fluctuation -> structure pressure
      - clean partial interface -> binding + structure pressure
      - interface clash -> structure + diversity pressure
    """
    combined_objectives = {
        "conservation": conservation_fitness,
    }

    # Skeptic/objective-injector output
    if critique:
        combined_objectives.update(extra_objectives or {})

    # ------------------------------------------------------------------
    # Fold/structure pressure if current fold failed
    # ------------------------------------------------------------------
    if (
        state.get("fold_thresholds_passed") is False
        and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
    ):
        combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 1.0
        combined_objectives["thermo"] = combined_objectives.get("thermo", 0.0) + 0.6
        combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.3

    # ------------------------------------------------------------------
    # Interface feedback
    # ------------------------------------------------------------------
    interface_signal = float(joint_feedback.get("interface_signal", 0.0) or 0.0)
    clean_interface_count = int(joint_feedback.get("clean_interface_count", 0) or 0)
    interface_clash_count = int(joint_feedback.get("interface_clash_count", 0) or 0)

    if clean_interface_count > 0:
        combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.3
        combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.3

    if interface_clash_count > 0:
        combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.4
        combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.2

    if interface_signal > 0.8 and not state.get("target_pdb"):
        combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.2
        combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.2

    # ------------------------------------------------------------------
    # Docking spread feedback from Skeptic critique
    # ------------------------------------------------------------------
    max_accept_spread = getenv_float("VLAB_MAX_ACCEPT_SCORE_SPREAD", 10.0)

    if score_spread is not None and score_spread > max_accept_spread:
        log.info(
            "High binding score spread %.3f detected; increasing diversity/structure pressure.",
            score_spread,
        )

        combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.8
        combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.5
        combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.2

    # ------------------------------------------------------------------
    # FailureMemory pressure
    # ------------------------------------------------------------------
    for k, v in failure_weights.items():
        if k == "binding_pressure":
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + v

        elif k == "structure_pressure":
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + v

        elif k == "diversity_pressure":
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + v

        elif k == "convergence_pressure":
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + v

        elif k == "interface_pressure":
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.5 * v
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.5 * v

        elif k == "clash_avoidance_pressure":
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.7 * v
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.3 * v

        elif k == "target_specific_exploitation":
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.2
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.2

        elif k == "target_reuse_pressure":
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.15
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.10

        elif k.startswith("concern"):
            if "entropy" in k or "structure" in k:
                combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + v

    # ------------------------------------------------------------------
    # Joint docking + MD adaptive control
    # ------------------------------------------------------------------
    binding = float(joint_feedback.get("binding_signal", 0.0) or 0.0)
    stability = float(joint_feedback.get("stability_signal", 0.0) or 0.0)
    fluct = float(joint_feedback.get("fluctuation_signal", 0.0) or 0.0)

    binding_spread = joint_feedback.get("binding_score_spread")
    has_binding = bool(joint_feedback.get("has_binding"))
    has_md = bool(joint_feedback.get("has_md"))

    try:
        binding_spread = float(binding_spread)
    except Exception:
        binding_spread = None

    max_target_score_spread = getenv_float("VLAB_MAX_TARGET_SCORE_SPREAD", 25.0)

    if has_binding and binding > 0.5 and binding_spread is not None:
        if binding_spread > max_target_score_spread:
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.8
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.7
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 0.3

    if has_binding:
        if binding < 0.3:
            combined_objectives["binding"] = combined_objectives.get("binding", 0.0) + 1.0
            combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.4

        if has_md and binding > 0.5 and stability < 0.3:
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.8
            combined_objectives["thermo"] = combined_objectives.get("thermo", 0.0) + 0.6

    if has_md:
        if fluct < 0.4:
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.7

        if stability < 0.3:
            combined_objectives["thermo"] = combined_objectives.get("thermo", 0.0) + 0.5
            combined_objectives["structure"] = combined_objectives.get("structure", 0.0) + 0.4

    if has_binding and has_md and binding > 0.7 and stability > 0.7 and fluct > 0.6:
        combined_objectives["diversity"] = combined_objectives.get("diversity", 0.0) + 0.6

    # ------------------------------------------------------------------
    # Normalise
    # ------------------------------------------------------------------
    total = sum(v for v in combined_objectives.values() if v is not None)

    if total > 0:
        combined_objectives = {
            k: (v / total if v is not None else 0.0)
            for k, v in combined_objectives.items()
        }

    combined_objectives["diversity"] = max(
        combined_objectives.get("diversity", 0.0),
        0.05,
    )

    return combined_objectives


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _select_top_sequences_from_population(population, analysis: dict) -> list[str]:
    """
    Decode Pareto-selected candidates from an NSGA-II population.
    """
    from VLAB2.optimisation.final_rna_design_system import decode_sequence as _decode

    best_idx = analysis["best_idx"]
    extreme_idxs = analysis["extremes"]
    selected_indices = set([best_idx] + extreme_idxs)

    top_sequences = []
    seen = set()

    for i, ind in enumerate(population):
        if i in selected_indices:
            seq = clean_rna(_decode(ind.X))

            if seq and seq not in seen:
                seen.add(seq)
                top_sequences.append(seq)

    if not top_sequences and population:
        fallback = clean_rna(_decode(population[0].X))
        top_sequences = [fallback] if fallback else []

    return top_sequences


def _cleanup_mutation_bias(state: LabState) -> dict:
    """
    Normalise and bound adaptive mutation bias weights.
    """
    new_bias = dict(
        state.get("_run_system_final_weights", state.get("mutation_bias", {})) or {}
    )

    new_bias["conservation"] = max(new_bias.get("conservation", 0.0), 0.1)

    if "thermo" in new_bias and new_bias["thermo"] > 0.8:
        new_bias["thermo"] *= 0.9

    if new_bias:
        required = ["thermo", "binding", "conservation", "structure", "diversity"]

        for k in required:
            new_bias[k] = new_bias.get(k, 0.0)

        for k in list(new_bias.keys()):
            new_bias[k] = max(float(new_bias[k]), 0.05)

        for k in list(new_bias.keys()):
            if new_bias[k] > 0.75:
                new_bias[k] = 0.75

        temp = 0.7 + 0.3 * (
            state.get("iterations", 0) / max(1, state.get("max_iterations", 3))
        )

        new_bias = {k: v ** temp for k, v in new_bias.items()}

        total = sum(new_bias.values())

        if total > 0:
            new_bias = {k: v / total for k, v in new_bias.items()}

    return new_bias


def _capture_optimisation_step_if_available(
    state: LabState,
    population,
    top_sequences: list[str],
    new_bias: dict,
    target_pdb: str | None,
    score_spread: float | None,
    combined_objectives: dict,
    joint_feedback: dict,
):
    """
    Capture optimisation data if a data_collector is available in state.

    This avoids relying on a global data_collector in modular layout.
    """
    data_collector = state.get("data_collector")

    if data_collector is None:
        return

    try:
        from VLAB2.optimisation.final_rna_design_system import decode_sequence as _decode

        summary_pop = data_collector.summarise_population(population, _decode)

        data_collector.capture_optimisation_step(
            population_summary=summary_pop,
            selected_sequences=top_sequences,
            mutation_bias=new_bias,
            metadata={
                "target_pdb": target_pdb,
                "iteration": state.get("iterations", 0),
                "binding_units": state.get("binding_units", "hdock_relative_score"),
                "score_spread": score_spread,
                "combined_objectives": combined_objectives,
                "joint_feedback": joint_feedback,
                "literature_motif_hints": state.get("literature_motif_hints", []),
                "literature_target_hints": state.get("literature_target_hints", []),
            },
        )

    except Exception as log_err:
        log.warning("Failed to log optimisation step: %s", log_err)


def _fallback_result(
    state: LabState,
    topic: str,
    sequences: list[str],
    status: str,
    summary: str,
) -> dict:
    fallback_seq = sequences[0] if sequences else ""

    return {
        "pi_summary": summary,
        "optimisation_status": status,
        "designed_sequences": [fallback_seq] if fallback_seq else [],
        "target_sequence": state.get("target_sequence") or fallback_seq or None,
        "structural_candidates": state.get("structural_candidates", []),
        "target_pdb": state.get("target_pdb"),
        "failed_target_pdbs": state.get("failed_target_pdbs", []),
        "mutation_bias": state.get("mutation_bias", {}),
        "hypothesis": topic,
        "iterations": state.get("iterations", 0) + 1,
        "literature_motif_hints": state.get("literature_motif_hints", []),
        "literature_target_hints": state.get("literature_target_hints", []),
        "literature_policy_text": state.get("literature_policy_text", ""),
    }


# ---------------------------------------------------------------------------
# Main PI agent
# ---------------------------------------------------------------------------

def pi_agent(state: LabState) -> dict:
    """
    Adaptive PI agent.

    Responsibilities:
      - optionally reset high-spread PDB targets
      - refine hypothesis qualitatively from Skeptic critique
      - update objective pressures from:
          * Skeptic parser
          * objective injector
          * failure memory
          * literature memory
          * fold thresholds
          * docking score spread
          * joint MD+docking/interface feedback
      - run NSGA-II optimisation
      - select Pareto/extreme sequences
      - return updated hypothesis, sequences, mutation bias, and iteration count
    """
    log.info("--- PI AGENT (ADAPTIVE MULTI-OBJECTIVE) ---")

    wrappers = state.get("wrappers", {})
    target_pdb = _maybe_reset_target_on_high_spread(state)

    sequences = dedupe_rna_sequences(state.get("designed_sequences", []))
    critique = state.get("critique", "") or ""

    if not wrappers:
        return {
            "pi_summary": "No wrappers available.",
            "optimisation_status": "no_wrappers",
            "iterations": state.get("iterations", 0) + 1,
            "target_pdb": target_pdb,
            "failed_target_pdbs": state.get("failed_target_pdbs", []),
            "literature_motif_hints": state.get("literature_motif_hints", []),
            "literature_target_hints": state.get("literature_target_hints", []),
            "literature_policy_text": state.get("literature_policy_text", ""),
        }

    if not target_pdb and sequences:
        log.info(
            "No protein target yet — running sequence-only optimisation without docking."
        )

    if not sequences:
        return {
            "pi_summary": "Bootstrap: waiting for structural design.",
            "optimisation_status": "bootstrap",
            "iterations": state.get("iterations", 0),
            "target_pdb": target_pdb,
            "failed_target_pdbs": state.get("failed_target_pdbs", []),
            "literature_motif_hints": state.get("literature_motif_hints", []),
            "literature_target_hints": state.get("literature_target_hints", []),
            "literature_policy_text": state.get("literature_policy_text", ""),
        }

    try:
        log.info("Running NSGA-II optimisation (final_rna_design_system)...")

        # ------------------------------------------------------------------
        # Failure memory
        # ------------------------------------------------------------------
        memory_model = FailureMemory()

        for past in state.get("failure_memory", []):
            memory_model.update(past)

        failure_weights = memory_model.compute_failure_weights()
        log.info("Failure memory weights: %s", failure_weights)

        # ------------------------------------------------------------------
        # Literature memory
        # ------------------------------------------------------------------
        try:
            literature_model = LiteratureMemory()
            literature_motif_hints = literature_model.get_motif_hints(limit=8)
            literature_target_hints = literature_model.get_target_policy_hints(limit=8)
            literature_policy_text = literature_model.build_target_policy_text()
        except Exception as e:
            log.warning("Failed to load LiteratureMemory in PI agent: %s", e)
            literature_motif_hints = []
            literature_target_hints = []
            literature_policy_text = ""

        state["literature_motif_hints"] = literature_motif_hints
        state["literature_target_hints"] = literature_target_hints
        state["literature_policy_text"] = literature_policy_text

        log.info(
            "Literature memory hints | targets=%s motifs=%s",
            literature_target_hints,
            literature_motif_hints,
        )

        parsed = {
            "dg_best": None,
            "missing_controls": [],
            "concerns": [],
        }

        extra_objectives = {}

        # ------------------------------------------------------------------
        # Parse Skeptic critique and generate extra objectives
        # ------------------------------------------------------------------
        if critique:
            from VLAB2.orchestration.objective_injector import generate_objectives
            from VLAB2.orchestration.skeptic_parser import parse_skeptic_output

            parsed = parse_skeptic_output(critique)
            extra_objectives = generate_objectives(parsed)

        # ------------------------------------------------------------------
        # Qualitative hypothesis refinement
        # ------------------------------------------------------------------
        refined_hypothesis = _refine_hypothesis_from_critique(state, critique)

        topic = (
            refined_hypothesis
            or state.get("hypothesis")
            or state.get("topic_description")
            or state.get("research_topic", "RNA secondary structure stability")
        )

        # ------------------------------------------------------------------
        # Conservation state
        # ------------------------------------------------------------------
        conservation = state.get("conservation_signal", {}) or {}
        conservation_fitness = conservation.get("conservation_fitness", 0)

        state["mutation_bias"] = adjust_weights(parsed, state.get("mutation_bias", {}))
        state["conservation_fitness"] = conservation_fitness
        state["conserved_regions"] = conservation.get("conserved_regions", [])

        # ------------------------------------------------------------------
        # Docking spread + joint MD/docking/interface physics feedback
        # ------------------------------------------------------------------
        score_spread = _parse_score_spread_from_critique(critique)
        joint_feedback = _compute_joint_physics_feedback(state)

        log.info(
            "Joint physics feedback | binding=%.3f stability=%.3f fluct=%.3f "
            "interface=%.3f clean=%s clash=%s | best_binding=%s spread=%s "
            "best_md_min=%s md_fluct=%s",
            joint_feedback.get("binding_signal", 0.0),
            joint_feedback.get("stability_signal", 0.0),
            joint_feedback.get("fluctuation_signal", 0.0),
            joint_feedback.get("interface_signal", 0.0),
            joint_feedback.get("clean_interface_count", 0),
            joint_feedback.get("interface_clash_count", 0),
            joint_feedback.get("best_binding_score"),
            joint_feedback.get("binding_score_spread"),
            joint_feedback.get("best_md_min_energy"),
            joint_feedback.get("mean_md_fluctuation"),
        )

        combined_objectives = _build_combined_objectives(
            state=state,
            parsed=parsed,
            extra_objectives=extra_objectives,
            failure_weights=failure_weights,
            conservation_fitness=conservation_fitness,
            critique=critique,
            score_spread=score_spread,
            joint_feedback=joint_feedback,
        )

        # ------------------------------------------------------------------
        # Run optimiser
        # ------------------------------------------------------------------
        population = run_system(
            topic=topic,
            target_pdb=target_pdb,
            state=state,
            extra_objectives=combined_objectives,
        )

        if population is None or not hasattr(population, "__len__") or len(population) == 0:
            log.warning("Empty population returned - using fallback")

            return _fallback_result(
                state=state,
                topic=topic,
                sequences=sequences,
                status="empty_population",
                summary="No viable NSGA-II population produced; using fallback sequence.",
            )

        from VLAB2.optimisation.final_rna_design_system import decode_sequence as _decode

        analysis = analyse_pareto(population, _decode)
        top_sequences = _select_top_sequences_from_population(population, analysis)

        if not top_sequences:
            return _fallback_result(
                state=state,
                topic=topic,
                sequences=sequences,
                status="empty_population",
                summary="No viable sequences produced.",
            )

        # ------------------------------------------------------------------
        # Conservation-dependent exploration/exploitation
        # ------------------------------------------------------------------
        if conservation_fitness > 0.7:
            top_sequences = top_sequences[:3]
        elif conservation_fitness < 0.2:
            log.warning("Low conservation — forcing exploitation phase")
            top_sequences = top_sequences[:2]

        top_sequences = dedupe_rna_sequences(top_sequences)

        log.info("Top sequences selected by PI: %s", top_sequences)

        # ------------------------------------------------------------------
        # Mutation bias cleanup
        # ------------------------------------------------------------------
        new_bias = _cleanup_mutation_bias(state)
        selected_motifs = state.get("_run_system_selected_motifs", [])
        min_fold_thresholds = state.get("_run_system_min_fold_thresholds", {})

        # ------------------------------------------------------------------
        # Training/optimisation capture
        # ------------------------------------------------------------------
        _capture_optimisation_step_if_available(
            state=state,
            population=population,
            top_sequences=top_sequences,
            new_bias=new_bias,
            target_pdb=target_pdb,
            score_spread=score_spread,
            combined_objectives=combined_objectives,
            joint_feedback=joint_feedback,
        )

        summary = (
            f"NSGA-II optimisation complete.\n"
            f"Target PDB: {target_pdb}\n"
            f"Selected {len(top_sequences)} sequences.\n"
            f"Conservation fitness: {float(conservation_fitness):.3f}\n"
            f"Selected motifs: {motif_summary_from_state(state)}\n"
            f"Min fold thresholds: {min_fold_thresholds or 'default'}\n"
            f"Binding units: {state.get('binding_units', 'hdock_relative_score')}\n"
            f"Adaptive mutation: {'enabled' if new_bias else 'none'}\n"
            f"Score spread feedback: {score_spread if score_spread is not None else 'N/A'}\n"
            f"Joint feedback: "
            f"binding={joint_feedback.get('binding_signal', 0.0):.3f}, "
            f"stability={joint_feedback.get('stability_signal', 0.0):.3f}, "
            f"fluctuation={joint_feedback.get('fluctuation_signal', 0.0):.3f}, "
            f"interface={joint_feedback.get('interface_signal', 0.0):.3f}, "
            f"clean_interface={joint_feedback.get('clean_interface_count', 0)}, "
            f"clash={joint_feedback.get('interface_clash_count', 0)}\n"
            f"Literature target hints: "
            f"{', '.join(literature_target_hints) if literature_target_hints else 'N/A'}\n"
            f"Literature motif hints: "
            f"{', '.join(literature_motif_hints) if literature_motif_hints else 'N/A'}"
        )

        target_sequence = state.get("target_sequence")

        if not target_sequence and top_sequences:
            target_sequence = top_sequences[0]

        return {
            "pi_summary": summary,
            "optimisation_status": "adaptive_pareto_optimised",
            "designed_sequences": top_sequences,
            "target_sequence": target_sequence,
            "structural_candidates": state.get("structural_candidates", []),
            "target_pdb": target_pdb,
            "failed_target_pdbs": state.get("failed_target_pdbs", []),
            "mutation_bias": new_bias,
            "hypothesis": topic,
            "iterations": state.get("iterations", 0) + 1,
            "_run_system_selected_motifs": selected_motifs,
            "_run_system_min_fold_thresholds": min_fold_thresholds,
            "joint_physics_feedback": joint_feedback,
            "literature_motif_hints": literature_motif_hints,
            "literature_target_hints": literature_target_hints,
            "literature_policy_text": literature_policy_text,
        }

    except Exception as e:
        log.exception("PI optimisation failed")

        fallback_seq = sequences[0] if sequences else ""

        return {
            "pi_summary": f"Fallback due to: {e}",
            "optimisation_status": "fallback",
            "designed_sequences": [fallback_seq] if fallback_seq else [],
            "target_sequence": state.get("target_sequence") or fallback_seq or None,
            "structural_candidates": state.get("structural_candidates", []),
            "target_pdb": target_pdb,
            "failed_target_pdbs": state.get("failed_target_pdbs", []),
            "mutation_bias": state.get("mutation_bias", {}),
            "hypothesis": state.get("hypothesis", ""),
            "iterations": state.get("iterations", 0) + 1,
            "literature_motif_hints": state.get("literature_motif_hints", []),
            "literature_target_hints": state.get("literature_target_hints", []),
            "literature_policy_text": state.get("literature_policy_text", ""),
        }