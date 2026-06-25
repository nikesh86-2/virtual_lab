"""
failure_memory.py

Persistent memory for recurring failure modes and partial successes.

Tracks:
- sequence failures
- motif failures
- energy/docking failures
- structural failures
- interface failures
- target failures with reasons
- partial-success targets/sequences

Used by:
- Skeptic Agent
- PI Agent
- Protein Agent target selection
- postrun training data builder
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pdb_id(x: Any) -> str:
    return str(x or "").strip().upper()


def _clean_rna(seq: Any) -> str:
    if not isinstance(seq, str):
        return ""
    return "".join(c for c in seq.upper().replace("T", "U") if c in "ACGU")


def _has_clean_interface(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and row.get("dock_valid") is True
        and row.get("interface_passed") is True
        and row.get("interface_steric_clash") is not True
    )


class FailureMemory:
    """
    Backward-compatible persistent memory model.

    Existing callers can still use:
      - update(parsed)
      - compute_failure_weights()
      - adjust_score(seq, score)

    New callers can use:
      - ingest_run_state(state)
      - record_target_failure(pdb_id, reason)
      - get_partial_success_targets()
      - get_hard_failed_targets()
    """

    HARD_TARGET_FAILURES = {
        "too_large",
        "implausible_hdock_score",
        "missing_receptor",
        "download_failed",
        "hdock_timeout",
        "hdock_exception",
        "receptor_unavailable",
    }

    SOFT_TARGET_FAILURES = {
        "only_one_clean_pose",
        "interface_partial",
        "score_spread",
        "weak_hdock_score",
        "no_clean_interface",
        "insufficient_dock_valid",
    }

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("VLAB_FAILURE_MEMORY_PATH", "failure_memory.json")

        self.memory: Dict[str, Any] = {
            "schema_version": "failure_memory.v2",
            "sequence_failures": [],
            "motif_failures": [],
            "energy_failures": [],
            "structural_failures": [],
            "interface_failures": [],
            "target_failures": [],
            "partial_success_targets": [],
            "partial_success_sequences": [],
            "successful_pose_features": [],
            "run_summaries": [],
        }

        self._load()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        defaults = {
            "schema_version": "failure_memory.v2",
            "sequence_failures": [],
            "motif_failures": [],
            "energy_failures": [],
            "structural_failures": [],
            "interface_failures": [],
            "target_failures": [],
            "partial_success_targets": [],
            "partial_success_sequences": [],
            "successful_pose_features": [],
            "run_summaries": [],
        }

        for k, v in defaults.items():
            self.memory.setdefault(k, v)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict):
                self.memory.update(loaded)

        except Exception:
            # Never break the lab because the memory file is corrupt.
            pass

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # INTERNAL DEDUPE
    # ------------------------------------------------------------------
    def _dedupe_by_field(self, collection: str, field: str) -> None:
        rows = self.memory.get(collection, []) or []
        seen = {}

        for row in rows:
            if not isinstance(row, dict):
                continue

            key = str(row.get(field, "") or "").strip().upper()
            if not key:
                continue

            # latest record wins
            seen[key] = row

        self.memory[collection] = list(seen.values())

    # ------------------------------------------------------------------
    # RECORD METHODS
    # ------------------------------------------------------------------
    def record_sequence_failure(
        self,
        seq: str,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        seq = _clean_rna(seq)
        if not seq:
            return

        self.memory["sequence_failures"].append(
            {
                "sequence": seq,
                "reason": reason,
                "metadata": metadata or {},
                "timestamp": _now_iso(),
            }
        )

    def record_energy_failure(
        self,
        best_dg: float | None = None,
        spread: float | None = None,
        best_score: float | None = None,
        units: str = "hdock_relative_score",
        metadata: Optional[dict] = None,
    ) -> None:
        # Backward compatibility: old callers pass best_dg.
        score = best_score if best_score is not None else best_dg

        if score is None:
            return

        self.memory["energy_failures"].append(
            {
                "best_score": _safe_float(score),
                "best_dg": _safe_float(score),  # compatibility alias
                "spread": _safe_float(spread),
                "units": units,
                "metadata": metadata or {},
                "timestamp": _now_iso(),
            }
        )

    def record_structural_failure(
        self,
        stability: str,
        mfe: float | None = None,
        pair_density: float | None = None,
        mfe_per_nt: float | None = None,
        sequence: str | None = None,
    ) -> None:
        self.memory["structural_failures"].append(
            {
                "sequence": _clean_rna(sequence),
                "stability": stability,
                "mfe": _safe_float(mfe),
                "pair_density": _safe_float(pair_density),
                "mfe_per_nt": _safe_float(mfe_per_nt),
                "timestamp": _now_iso(),
            }
        )

    def record_motif_failure(self, motif_desc: str) -> None:
        motif_desc = str(motif_desc or "").strip()
        if not motif_desc:
            return

        self.memory["motif_failures"].append(
            {
                "description": motif_desc,
                "timestamp": _now_iso(),
            }
        )

    def record_interface_failure(
        self,
        row: dict,
        reason: str | None = None,
    ) -> None:
        if not isinstance(row, dict):
            return

        self.memory["interface_failures"].append(
            {
                "sequence": _clean_rna(row.get("sequence")),
                "target_pdb": _pdb_id(row.get("target_pdb")),
                "reason": reason or row.get("pose_training_label") or "interface_failure",
                "interface_passed": row.get("interface_passed"),
                "interface_steric_clash": row.get("interface_steric_clash"),
                "interface_clash_severity": row.get("interface_clash_severity"),
                "interface_quality_score": row.get("interface_quality_score"),
                "interface_min_distance_A": row.get("interface_min_distance_A"),
                "dock_score": row.get("dock_score"),
                "timestamp": _now_iso(),
            }
        )

    def record_target_failure(
        self,
        pdb_id: str,
        reason: str,
        metadata: Optional[dict] = None,
    ) -> None:
        pdb_id = _pdb_id(pdb_id)
        if not pdb_id:
            return

        self.memory["target_failures"].append(
            {
                "pdb_id": pdb_id,
                "reason": reason or "unknown",
                "metadata": metadata or {},
                "timestamp": _now_iso(),
            }
        )

        self._dedupe_by_field("target_failures", "pdb_id")

    def record_partial_success_target(
        self,
        pdb_id: str,
        sequence: str | None = None,
        row: Optional[dict] = None,
    ) -> None:
        pdb_id = _pdb_id(pdb_id)
        if not pdb_id:
            return

        row = row or {}
        seq = _clean_rna(sequence or row.get("sequence"))

        self.memory["partial_success_targets"].append(
            {
                "pdb_id": pdb_id,
                "sequence": seq,
                "dock_score": row.get("dock_score"),
                "interface_quality_score": row.get("interface_quality_score"),
                "interface_min_distance_A": row.get("interface_min_distance_A"),
                "interface_basic_residue_contacts": row.get("interface_basic_residue_contacts"),
                "interface_rna_span_covered": row.get("interface_rna_span_covered"),
                "timestamp": _now_iso(),
            }
        )

        self._dedupe_by_field("partial_success_targets", "pdb_id")

        if seq:
            self.memory["partial_success_sequences"].append(
                {
                    "sequence": seq,
                    "target_pdb": pdb_id,
                    "timestamp": _now_iso(),
                }
            )

            self._dedupe_by_field("partial_success_sequences", "sequence")

    # ------------------------------------------------------------------
    # INGEST METHODS
    # ------------------------------------------------------------------
    def ingest_skeptic_output(
        self,
        critique: str,
        sequences: List[str] | None = None,
    ) -> None:
        """
        Parse structured Skeptic output and update memory.

        Supports:
          ENERGY: best_binding_score=-71.1, spread=4.2, n=2, source=HDOCK
          ENERGY: ΔG_best=-10.0, spread=2.0
        """
        if not critique:
            return

        best_score = None
        spread = None

        for line in critique.splitlines():
            line = line.strip()

            if line.startswith("ENERGY:"):
                m_best = re.search(
                    r"(?:best_binding_score|best_dg|dg_best|ΔG_best)\s*=\s*(-?\d+(?:\.\d+)?)",
                    line,
                    flags=re.IGNORECASE,
                )
                m_spread = re.search(
                    r"spread\s*=\s*(-?\d+(?:\.\d+)?)",
                    line,
                    flags=re.IGNORECASE,
                )

                if m_best:
                    best_score = _safe_float(m_best.group(1))
                if m_spread:
                    spread = _safe_float(m_spread.group(1))

            if re.match(r"^\d+\.", line):
                reason = re.sub(r"^\d+\.\s*", "", line).strip()

                if sequences:
                    for seq in sequences:
                        self.record_sequence_failure(seq, reason)
                else:
                    self.record_motif_failure(reason)

            if line.startswith("MISSING_CONTROLS:"):
                self.record_motif_failure(line)

        self.record_energy_failure(best_score=best_score, spread=spread)
        self.save()

    def ingest_run_state(self, state: dict) -> None:
        """
        Ingest final/intermediate state with interface-aware labels.
        This is the main bridge from docking results to memory.
        """
        if not isinstance(state, dict):
            return

        binding_results = state.get("binding_results", []) or []
        accepted_target = state.get("target_pdb")

        clean_rows = []
        clash_rows = []

        for row in binding_results:
            if not isinstance(row, dict):
                continue

            if _has_clean_interface(row):
                clean_rows.append(row)

                self.memory["successful_pose_features"].append(
                    {
                        "sequence": _clean_rna(row.get("sequence")),
                        "target_pdb": _pdb_id(row.get("target_pdb")),
                        "dock_score": row.get("dock_score"),
                        "binding_rank_score": row.get("binding_rank_score"),
                        "interface_quality_score": row.get("interface_quality_score"),
                        "interface_min_distance_A": row.get("interface_min_distance_A"),
                        "interface_basic_residue_contacts": row.get("interface_basic_residue_contacts"),
                        "interface_rna_span_covered": row.get("interface_rna_span_covered"),
                        "timestamp": _now_iso(),
                    }
                )

                if accepted_target is None:
                    self.record_partial_success_target(
                        row.get("target_pdb"),
                        sequence=row.get("sequence"),
                        row=row,
                    )

            elif row.get("dock_valid"):
                if row.get("interface_steric_clash"):
                    clash_rows.append(row)
                    self.record_interface_failure(row, reason="steric_clash")

        # Ingest failed targets. Supports list[str] and list[dict].
        for item in state.get("failed_target_pdbs", []) or []:
            if isinstance(item, dict):
                self.record_target_failure(
                    item.get("pdb_id"),
                    item.get("reason", "unknown"),
                    metadata=item,
                )
            elif item:
                self.record_target_failure(str(item), "unknown")

        feedback = state.get("joint_physics_feedback", {}) or {}
        self.record_energy_failure(
            best_score=feedback.get("best_binding_score"),
            spread=feedback.get("binding_score_spread"),
            units=state.get("binding_units", "hdock_relative_score"),
            metadata={"source": "joint_physics_feedback"},
        )

        if accepted_target:
            run_label = "successful_target_selected"
        elif clean_rows:
            run_label = "partial_success_clean_pose_no_target"
        else:
            run_label = "failed_no_valid_target"

        conservation = state.get("conservation_signal", {}) or {}

        self.memory["run_summaries"].append(
            {
                "research_topic": state.get("research_topic"),
                "target_pdb": accepted_target,
                "run_label": run_label,
                "clean_pose_count": len(clean_rows),
                "clash_pose_count": len(clash_rows),
                "conservation_fitness": conservation.get("conservation_fitness"),
                "timestamp": _now_iso(),
            }
        )

        self.save()

    # ------------------------------------------------------------------
    # QUERY METHODS
    # ------------------------------------------------------------------
    def get_hard_failed_targets(self) -> set[str]:
        out = set()

        for row in self.memory.get("target_failures", []) or []:
            if not isinstance(row, dict):
                continue

            if row.get("reason") in self.HARD_TARGET_FAILURES:
                pdb = _pdb_id(row.get("pdb_id"))
                if pdb:
                    out.add(pdb)

        return out

    def get_soft_failed_targets(self) -> set[str]:
        out = set()

        for row in self.memory.get("target_failures", []) or []:
            if not isinstance(row, dict):
                continue

            if row.get("reason") in self.SOFT_TARGET_FAILURES:
                pdb = _pdb_id(row.get("pdb_id"))
                if pdb:
                    out.add(pdb)

        return out

    def get_partial_success_targets(self) -> list[str]:
        hard_failed = self.get_hard_failed_targets()
        out = []

        for row in self.memory.get("partial_success_targets", []) or []:
            if not isinstance(row, dict):
                continue

            pdb = _pdb_id(row.get("pdb_id"))

            if pdb and pdb not in hard_failed:
                out.append(pdb)

        return list(dict.fromkeys(out))

    def get_partial_success_sequences(self) -> list[str]:
        out = []

        for row in self.memory.get("partial_success_sequences", []) or []:
            if isinstance(row, dict):
                seq = _clean_rna(row.get("sequence"))
                if seq:
                    out.append(seq)

        return list(dict.fromkeys(out))

    def get_failure_penalty(self, seq: str) -> float:
        seq = _clean_rna(seq)
        if not seq:
            return 0.0

        penalty = 0.0

        for item in self.memory.get("sequence_failures", []) or []:
            failed_seq = _clean_rna(item.get("sequence"))

            if not failed_seq:
                continue

            n = min(len(seq), len(failed_seq))
            if n == 0:
                continue

            matches = sum(1 for a, b in zip(seq[:n], failed_seq[:n]) if a == b)
            similarity = matches / max(1, n)

            if similarity > 0.7:
                penalty += 0.3

        return min(penalty, 1.0)

    def get_energy_bias(self) -> float:
        values = []

        for e in self.memory.get("energy_failures", []) or []:
            if not isinstance(e, dict):
                continue

            v = e.get("best_score", e.get("best_dg"))
            v = _safe_float(v)

            if v is not None:
                values.append(v)

        if not values:
            return 0.0

        avg = sum(values) / len(values)

        # For HDOCK-relative scores, weak/absent binding often trends near zero.
        if avg > -30:
            return 0.2

        return 0.0

    # ------------------------------------------------------------------
    # BACKWARD-COMPAT UPDATE
    # ------------------------------------------------------------------
    def update(self, parsed: Dict) -> None:
        """
        Accept parsed skeptic output and convert into memory entries.
        """
        if not parsed:
            return

        self.record_energy_failure(
            best_score=parsed.get("best_binding_score", parsed.get("dg_best")),
            spread=parsed.get("spread"),
        )

        for concern in parsed.get("concerns", []) or []:
            self.record_motif_failure(str(concern))

        if parsed.get("missing_controls"):
            self.record_structural_failure(
                stability="unknown",
                mfe=None,
            )

        self.save()

    def compute_failure_weights(self) -> Dict[str, float]:
        """
        Convert persistent memory into objective bias weights for PI.
        """
        weights: Dict[str, float] = {}

        energy_failures = self.memory.get("energy_failures", []) or []
        structural_failures = self.memory.get("structural_failures", []) or []
        motif_failures = self.memory.get("motif_failures", []) or []
        interface_failures = self.memory.get("interface_failures", []) or []
        partial_success = self.memory.get("partial_success_targets", []) or []

        if len(energy_failures) > 3:
            weights["binding_pressure"] = 0.2

        if len(structural_failures) > 3:
            weights["structure_pressure"] = 0.2

        if len(motif_failures) > 5:
            weights["diversity_pressure"] = 0.2

        if len(interface_failures) > 2:
            weights["interface_pressure"] = 0.35
            weights["clash_avoidance_pressure"] = 0.30
            weights["structure_pressure"] = max(weights.get("structure_pressure", 0.0), 0.2)

        if partial_success:
            weights["interface_pressure"] = max(weights.get("interface_pressure", 0.0), 0.35)
            weights["target_specific_exploitation"] = 1.0

        spreads = []
        for e in energy_failures:
            if isinstance(e, dict):
                s = _safe_float(e.get("spread"))
                if s is not None:
                    spreads.append(s)

        if spreads and sum(spreads) / len(spreads) < 3:
            weights["convergence_pressure"] = 0.3

        return weights

    # ------------------------------------------------------------------
    # APPLY TO SCORING
