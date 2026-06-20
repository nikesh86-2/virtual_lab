from __future__ import annotations

import logging
import os
import statistics

from VLAB2.core.gpu_manager import clear_gpu
from VLAB2.orchestration.state_schema import (
    LabState,
    add_conversation_entry,
    record_stage_output,
)

from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.env_utils import getenv_int
from VLAB2.orchestration.utils.sequence_utils import dedupe_rna_sequences
from VLAB2.orchestration.utils.text_utils import truncate_str


log = logging.getLogger("virtual_lab")

__all__ = ["md_agent"]


# ------------------------------------------------------------
# ✅ Energy extraction helper
# ------------------------------------------------------------
def _extract_md_metrics(output: dict) -> dict:
    """
    Standardise MD outputs into consistent fields.

    Supports:
      - direct values from wrapper
      - trajectory energy lists (if present)
    """
    if not isinstance(output, dict):
        return {}

    result = dict(output)

    # Direct fields (preferred)
    min_energy = result.get("min_energy")
    mean_energy = result.get("mean_energy")
    fluct = result.get("energy_fluctuation")

    # If missing, try to infer from energy trajectory
    energy_trace = result.get("energy_trajectory") or result.get("energies")

    if energy_trace and isinstance(energy_trace, list) and len(energy_trace) > 1:
        try:
            vals = [float(x) for x in energy_trace if x is not None]

            if vals:
                min_energy = min_energy if min_energy is not None else min(vals)
                mean_energy = mean_energy if mean_energy is not None else statistics.mean(vals)

                if fluct is None:
                    fluct = max(vals) - min(vals)
        except Exception:
            pass

    result["min_energy"] = min_energy
    result["mean_energy"] = mean_energy
    result["energy_fluctuation"] = fluct

    return result


# ------------------------------------------------------------
# MAIN AGENT
# ------------------------------------------------------------
def md_agent(state: LabState) -> dict:
    log.info("--- MD AGENT: Evaluating Structural Stability ---")

    try:
        eval_top_n = getenv_int("VLAB_AGENT_EVAL_TOP_N", 3, min_value=1)

        # ------------------------------------------------------------
        # Sequence selection
        # ------------------------------------------------------------
        sequences = []

        if state.get("target_sequence"):
            sequences.append(state["target_sequence"])

        sequences.extend(state.get("designed_sequences", []) or [])

        for c in state.get("structural_candidates", []) or []:
            if isinstance(c, dict) and c.get("sequence"):
                sequences.append(c["sequence"])

        sequences = dedupe_rna_sequences(sequences)[:eval_top_n]

        # ------------------------------------------------------------
        # Fold gate
        # ------------------------------------------------------------
        if (
            state.get("fold_thresholds_passed") is False
            and os.getenv("VLAB_RNA_ENFORCE_MIN_FOLD", "1").strip() == "1"
        ):
            candidates = state.get("structural_candidates", []) or []

            any_passed = any(
                isinstance(c, dict)
                and isinstance(c.get("fold_quality"), dict)
                and c["fold_quality"].get("passed")
                for c in candidates
            )

            if not any_passed:
                reasons = state.get("fold_threshold_reasons", []) or []

                summary = "MD skipped due to fold failure: " + "; ".join(reasons)

                result = {
                    "md_analysis": summary,
                    "md_results": [],
                }

                save_checkpoint({**state, **result})
                return result

        if not sequences:
            return {"md_analysis": "No sequences", "md_results": []}

        log.info("MD evaluating: %s", sequences)

        md = state["wrappers"]["md"]

        # ------------------------------------------------------------
        # ✅ Caching layer
        # ------------------------------------------------------------
        cache = state.setdefault("_md_cache", {})

        results = []
        valid_results = []
        failed_sequences = []

        # ------------------------------------------------------------
        # Run MD
        # ------------------------------------------------------------
        for seq in sequences:

            if seq in cache:
                log.info("Using cached MD result for %s", seq[:20])
                output = cache[seq]

            else:
                try:
                    output = md.run_md(seq)
                except Exception as e:
                    log.exception("MD failed for %s", seq[:20])
                    output = {"valid": False, "error": str(e)}

                finally:
                    clear_gpu()

                # ✅ store in cache
                cache[seq] = output

            # ✅ extract metrics
            output = _extract_md_metrics(output)

            entry = {"sequence": seq, "result": output}
            results.append(entry)

            if output and output.get("valid"):
                valid_results.append(entry)
            else:
                failed_sequences.append(
                    {"sequence": seq, "error": output.get("error")}
                )

        # ------------------------------------------------------------
        # All failed
        # ------------------------------------------------------------
        if not valid_results:
            summary = "All MD simulations failed."

            return {
                "md_analysis": summary,
                "md_results": results,
                "failed_sequences": failed_sequences,
            }

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        lines = []

        for r in valid_results[:5]:
            res = r["result"]

            lines.append(
                f"{r['sequence'][:20]}... | "
                f"minE={res.get('min_energy')} | "
                f"meanE={res.get('mean_energy')} | "
                f"fluct={res.get('energy_fluctuation')}"
            )

        summary = "\n".join(lines)

        result = {
            "md_analysis": summary,
            "md_results": valid_results,
            "failed_sequences": failed_sequences,
            "_md_cache": cache,  # ✅ persist cache across iterations
        }

        save_checkpoint({**state, **result})
        return result

    except Exception as e:
        log.exception("MD AGENT ERROR")

        return {
            "md_analysis": f"MD failed: {e}",
            "md_results": [],
        }

    finally:
        clear_gpu()