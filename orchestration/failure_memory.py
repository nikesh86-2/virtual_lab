"""
failure_memory.py

Persistent memory for recurring failure modes, target outcomes, and partial
successes across Virtual Lab iterations and runs.

Tracks:
- sequence failures
- motif failures
- energy/docking failures
- structural failures
- interface failures
- target failures with reasons
- partial-success targets/sequences
- successful targets
- successful pose/interface features

Used by:
- Skeptic Agent
- PI Agent
- Protein Agent target selection
- postrun training data builder

Design goals:
- Backward compatible with older FailureMemory callers.
- Never break the lab if the memory file is missing/corrupt.
- Distinguish hard target failures from soft/partial target failures.
- Preserve partial successes so future runs can exploit promising targets.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def get_successful_target_records(self) -> list:
    """
    Returning the ranked records.
    """
    return sorted(
        rows,
    key=lambda r: (
        -(int(r.get("best_clean_pose_count") or 0)),
        -(int(r.get("success_count") or 0)),
        -(int(r.get("best_binding_result_count") or 0)),
    ),
)

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default

        return float(x)

    except Exception:
        return default


from pathlib import Path

def _pdb_id(x):
    text = str(x or "").strip()

    if not text:
        return ""

    try:
        text = Path(text).stem
    except Exception:
        pass

    return text.upper()


def _clean_rna(seq: Any) -> str:
    if not isinstance(seq, str):
        return ""

    return "".join(
        c for c in seq.upper().replace("T", "U")
        if c in "ACGU"
    )


def _has_clean_interface(row: dict) -> bool:
    """
    Pose-level clean-interface predicate.

    A clean interface requires:
      - docking was valid
      - interface analysis passed
      - no steric clash
    """
    return (
        isinstance(row, dict)
        and row.get("dock_valid") is True
        and row.get("interface_passed") is True
        and row.get("interface_steric_clash") is not True
    )


def _is_dock_valid(row: dict) -> bool:
    return (
        isinstance(row, dict)
        and (
            row.get("dock_valid") is True
            or row.get("vina_valid") is True
        )
        and row.get("binding_mode") != "rejected_docking"
    )

CLASH_SEVERITY_PENALTIES = {
    "none": 0.0,
    "unknown": 0.10,
    "borderline": 0.25,
    "moderate": 0.60,
    "severe": 1.00,
}

INTERFACE_SIMILARITY_THRESHOLD = float(
    os.getenv("VLAB_INTERFACE_MEMORY_MIN_SIMILARITY", "0.72")
)

INTERFACE_SIMILARITY_POWER = float(
    os.getenv("VLAB_INTERFACE_MEMORY_SIMILARITY_POWER", "3.0")
)

def _normalise_fraction(value: Any, default: float = 0.0) -> float:
    """
    Normalise a possibly percentage-like value to [0, 1].

    Examples:
      0.72 -> 0.72
      72.0 -> 0.72
    """
    parsed = _safe_float(value, default)

    if parsed is None:
        return default

    if parsed > 1.0:
        parsed /= 100.0

    return max(0.0, min(1.0, parsed))


def _sequence_similarity(a: Any, b: Any) -> float:
    """
    Positional similarity for equal or near-equal length RNA sequences.

    A length mismatch is explicitly penalised.
    """
    a = _clean_rna(a)
    b = _clean_rna(b)

    if not a or not b:
        return 0.0

    overlap = min(len(a), len(b))

    if overlap <= 0:
        return 0.0

    matches = sum(
        1 for x, y in zip(a[:overlap], b[:overlap])
        if x == y
    )

    positional_similarity = matches / overlap
    length_similarity = overlap / max(len(a), len(b))

    return positional_similarity * length_similarity


def _interface_evidence_score(row: dict) -> tuple[float, dict]:
    """
    Convert one measured interface row into a signed optimisation score.

    Positive:
      clean, interface-valid, good quality, useful RNA-span coverage.

    Negative:
      steric clash, moderate/severe clash, interface rejection.

    Missing interface evidence is neutral rather than assumed clean.
    """
    if not isinstance(row, dict):
        return 0.0, {
            "known": False,
            "reason": "invalid_row",
        }

    dock_valid = _is_dock_valid(row)

    if not dock_valid:
        return 0.0, {
            "known": False,
            "reason": "dock_not_valid",
        }

    interface_passed = row.get("interface_passed")
    steric_clash = row.get("interface_steric_clash") is True

    severity = str(
        row.get("interface_clash_severity")
        or ("moderate" if steric_clash else "none")
    ).strip().lower()

    severity_penalty = CLASH_SEVERITY_PENALTIES.get(
        severity,
        CLASH_SEVERITY_PENALTIES["unknown"],
    )

    quality = _normalise_fraction(
        row.get("interface_quality_score"),
        default=0.0,
    )

    span = _normalise_fraction(
        row.get("interface_rna_span_covered"),
        default=0.0,
    )

    clean = _has_clean_interface(row)

    # Positive evidence.
    score = (
        0.55 * quality
        + 0.20 * span
        + (0.35 if interface_passed is True else 0.0)
        + (0.25 if clean else 0.0)
    )

    # Negative evidence.
    if interface_passed is False:
        score -= 0.35

    if steric_clash:
        score -= 0.55

    score -= 0.75 * severity_penalty

    # Keep objective in a controlled signed range.
    score = max(-1.5, min(1.5, score))

    return score, {
        "known": True,
        "clean": clean,
        "dock_valid": dock_valid,
        "interface_passed": interface_passed,
        "steric_clash": steric_clash,
        "clash_severity": severity,
        "clash_severity_penalty": severity_penalty,
        "quality": quality,
        "rna_span_covered": span,
        "raw_interface_score": score,
    }

# ---------------------------------------------------------------------------
# FailureMemory
# ---------------------------------------------------------------------------

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
      - get_successful_targets()
    """

    # Hard failures should usually be excluded from future target selection.
    HARD_TARGET_FAILURES = {
        "too_large",
        "implausible_hdock_score",
        "missing_receptor",
        "download_failed",
        "hdock_timeout",
        "hdock_exception",
        "receptor_unavailable",
        "protein_wrapper_failed",
    }

    # Soft failures may still be useful for future exploitation/refinement.
    SOFT_TARGET_FAILURES = {
        "only_one_clean_pose",
        "interface_partial",
        "score_spread",
        "weak_hdock_score",
        "no_clean_interface",
        "insufficient_dock_valid",
        "no_docking_valid_results",
        "no_proxy_valid_results",
        "unknown",
    }

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv(
            "VLAB_FAILURE_MEMORY_PATH",
            "failure_memory.json",
        )

        self.memory: Dict[str, Any] = {
            "schema_version": "failure_memory.v3",
            "sequence_failures": [],
            "motif_failures": [],
            "energy_failures": [],
            "structural_failures": [],
            "interface_failures": [],
            "target_failures": [],
            "partial_success_targets": [],
            "partial_success_sequences": [],
            "successful_targets": [],
            "successful_pose_features": [],
            "run_summaries": [],
        }

        self._load()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        defaults = {
            "schema_version": "failure_memory.v3",
            "sequence_failures": [],
            "motif_failures": [],
            "energy_failures": [],
            "structural_failures": [],
            "interface_failures": [],
            "target_failures": [],
            "partial_success_targets": [],
            "partial_success_sequences": [],
            "successful_targets": [],
            "successful_pose_features": [],
            "run_summaries": [],
        }
        for key, value in defaults.items():
            self.memory.setdefault(key, value)

        for row in self.memory.get("successful_targets", []):
            if not isinstance(row, dict):
                continue

            row.setdefault(
                "best_clean_pose_count",
                int(row.get("clean_pose_count", 0) or 0),
            )

            row.setdefault(
                "best_binding_result_count",
                int(row.get("binding_result_count", 0) or 0),
            )

            row.setdefault(
                "latest_clean_pose_count",
                int(row.get("clean_pose_count", 0) or 0),
            )

            row.setdefault(
                "latest_binding_result_count",
                int(row.get("binding_result_count", 0) or 0),
            )

            row.setdefault("success_count", 1)



    def _load(self) -> None:
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)

            if isinstance(loaded, dict):
                self.memory.update(loaded)

        except Exception:
            # Never break the lab because the memory file is corrupt.
            pass

    def save(self) -> None:
        try:
            self._compact_memory()

            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self.memory, handle, indent=2, ensure_ascii=False)

        except Exception:
            # Memory persistence must be non-fatal.
            pass

    # ------------------------------------------------------------------
    # Internal dedupe/compaction
    # ------------------------------------------------------------------

    def _dedupe_by_field(self, collection: str, field: str) -> None:
        """
        Dedupe dict records by a single field.

        Latest record for each key wins.
        """
        rows = self.memory.get(collection, []) or []
        seen: dict[str, dict] = {}

        for row in rows:
            if not isinstance(row, dict):
                continue

            key = str(row.get(field, "") or "").strip().upper()

            if not key:
                continue

            seen[key] = row

        self.memory[collection] = list(seen.values())

    def _dedupe_by_composite_key(
        self,
        collection: str,
        fields: list[str],
    ) -> None:
        """
        Dedupe dict records by a tuple of fields.

        Latest record for each composite key wins.
        """
        rows = self.memory.get(collection, []) or []
        seen: dict[tuple, dict] = {}

        for row in rows:
            if not isinstance(row, dict):
                continue

            key = tuple(str(row.get(field, "") or "").strip().upper() for field in fields)

            if not any(key):
                continue

            seen[key] = row

        self.memory[collection] = list(seen.values())

    def _cap_collection(self, collection: str, limit: int) -> None:
        rows = self.memory.get(collection, []) or []

        if isinstance(rows, list) and len(rows) > limit:
            self.memory[collection] = rows[-limit:]

    def _compact_memory(self) -> None:
        """
        Keep memory useful and bounded.
        """
        self._dedupe_by_composite_key(
            "interface_failures",
            ["target_pdb", "sequence", "reason"],
        )
        self._dedupe_by_composite_key(
            "successful_pose_features",
            ["target_pdb", "sequence"],
        )
        self._dedupe_by_composite_key(
            "target_failures",
            ["pdb_id", "reason"],
        )
        self._dedupe_by_composite_key(
            "partial_success_targets",
            ["pdb_id", "sequence"],
        )
        self._dedupe_by_composite_key("partial_success_sequences", ["sequence", "target_pdb"])

        self._cap_collection("sequence_failures", 500)
        self._cap_collection("motif_failures", 300)
        self._cap_collection("energy_failures", 300)
        self._cap_collection("structural_failures", 300)
        self._cap_collection("interface_failures", 300)
        self._cap_collection("successful_pose_features", 300)
        self._cap_collection("run_summaries", 200)

    # ------------------------------------------------------------------
    # Record methods
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
        """
        Record docking/ranking-score failure or spread issue.

        best_dg is kept for backward compatibility, but HDOCK values are not
        physical binding free energies.
        """
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

        interface_score, interface_details = _interface_evidence_score(row)

        self.memory["interface_failures"].append(
            {
                "sequence": _clean_rna(row.get("sequence")),
                "target_pdb": _pdb_id(
                    row.get("target_pdb")
                    or row.get("target_pdb_id")
                ),
                "reason": (
                    reason
                    or row.get("pose_training_label")
                    or "interface_failure"
                ),
                "dock_valid": row.get("dock_valid"),
                "interface_passed": row.get("interface_passed"),
                "interface_steric_clash": row.get(
                    "interface_steric_clash"
                ),
                "interface_clash_severity": row.get(
                    "interface_clash_severity"
                ),
                "interface_quality_score": row.get(
                    "interface_quality_score"
                ),
                "interface_min_distance_A": row.get(
                    "interface_min_distance_A"
                ),
                "interface_basic_residue_contacts": row.get(
                    "interface_basic_residue_contacts"
                ),
                "interface_rna_span_covered": row.get(
                    "interface_rna_span_covered"
                ),
                "interface_residues": row.get("interface_residues"),
                "pose_training_label": row.get("pose_training_label"),
                "dock_score": row.get("dock_score"),
                "memory_interface_score": interface_score,
                "memory_interface_details": interface_details,
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

        self._dedupe_by_composite_key("target_failures", ["pdb_id", "reason"])

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
                "binding_rank_score": row.get("binding_rank_score"),
                "interface_quality_score": row.get("interface_quality_score"),
                "interface_min_distance_A": row.get("interface_min_distance_A"),
                "interface_basic_residue_contacts": row.get("interface_basic_residue_contacts"),
                "interface_rna_span_covered": row.get("interface_rna_span_covered"),
                "timestamp": _now_iso(),
            }
        )

        self._dedupe_by_composite_key("partial_success_targets", ["pdb_id", "sequence"])

        if seq:
            self.memory["partial_success_sequences"].append(
                {
                    "sequence": seq,
                    "target_pdb": pdb_id,
                    "timestamp": _now_iso(),
                }
            )

            self._dedupe_by_composite_key("partial_success_sequences", ["sequence", "target_pdb"])

    def record_successful_target(
        self,
        pdb_id: str,
        clean_pose_count: int,
        binding_result_count: int,
        metadata: Optional[dict] = None,
    ) -> None:

        pdb_id = _pdb_id(pdb_id)

        if not pdb_id:
            return

        rows = self.memory.setdefault(
            "successful_targets",
            [],
        )

        existing = None

        for row in rows:
            if (
                isinstance(row, dict)
                and _pdb_id(row.get("pdb_id")) == pdb_id
            ):
                existing = row
                break

        clean_pose_count = int(clean_pose_count or 0)
        binding_result_count = int(binding_result_count or 0)

        if existing is None:

            rows.append(
                {
                    "pdb_id": pdb_id,

                    # best-ever observations
                    "best_clean_pose_count": clean_pose_count,
                    "best_binding_result_count": binding_result_count,

                    # latest run
                    "latest_clean_pose_count": clean_pose_count,
                    "latest_binding_result_count": binding_result_count,

                    # reproducibility
                    "success_count": 1,

                    "metadata": metadata or {},
                    "timestamp": _now_iso(),
                }
            )

            return

        # Upgrade old records automatically.

        if "best_clean_pose_count" not in existing:
            existing["best_clean_pose_count"] = int(
                existing.get("clean_pose_count", 0)
            )

        if "best_binding_result_count" not in existing:
            existing["best_binding_result_count"] = int(
                existing.get("binding_result_count", 0)
            )

        if "success_count" not in existing:
            existing["success_count"] = 1

        # Update statistics.

        existing["best_clean_pose_count"] = max(
            int(existing.get("best_clean_pose_count", 0)),
            clean_pose_count,
        )

        existing["best_binding_result_count"] = max(
            int(existing.get("best_binding_result_count", 0)),
            binding_result_count,
        )

        existing["latest_clean_pose_count"] = clean_pose_count

        existing["latest_binding_result_count"] = (
            binding_result_count
        )

        existing["success_count"] = (
            int(existing.get("success_count", 0))
            + 1
        )

        if metadata:
            existing["metadata"] = metadata

        existing["timestamp"] = _now_iso()
    # ------------------------------------------------------------------
    # Ingest methods
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
        Ingest final/intermediate run state with interface-aware labels.

        This is the main bridge from docking results to persistent memory.
        Records latest target outcomes but only promotes best_validated targets.
        """
        if not isinstance(state, dict):
            return

        binding_results = state.get("binding_results", []) or []
        
        # Use best validated target for success memory if available, else use latest
        best_target = state.get("best_validated_target_evaluation")
        latest_target = state.get("latest_target_evaluation")
        target_record = best_target or latest_target
        
        if target_record and isinstance(target_record, dict):
            accepted_target = target_record.get("target_pdb")
        else:
            # Fall back to legacy field
            accepted_target = state.get("target_pdb")
        
        accepted_target_norm = _pdb_id(accepted_target)

        clean_rows: list[dict] = []
        clash_rows: list[dict] = []

        # ------------------------------------------------------------
        # Pose-level signals
        # ------------------------------------------------------------
        for row in binding_results:
            if not isinstance(row, dict):
                continue

            row_target = _pdb_id(row.get("target_pdb"))

            if _has_clean_interface(row):
                clean_rows.append(row)
                interface_score, interface_details = _interface_evidence_score(row)

                self.memory["successful_pose_features"].append(
                    {
                        "sequence": _clean_rna(row.get("sequence")),
                        "target_pdb": row_target,
                        "dock_valid": row.get("dock_valid"),
                        "dock_score": row.get("dock_score"),
                        "binding_rank_score": row.get(
                            "binding_rank_score"
                        ),
                        "interface_passed": row.get(
                            "interface_passed"
                        ),
                        "interface_steric_clash": row.get(
                            "interface_steric_clash"
                        ),
                        "interface_clash_severity": row.get(
                            "interface_clash_severity"
                        ),
                        "interface_quality_score": row.get(
                            "interface_quality_score"
                        ),
                        "interface_min_distance_A": row.get(
                            "interface_min_distance_A"
                        ),
                        "interface_basic_residue_contacts": row.get(
                            "interface_basic_residue_contacts"
                        ),
                        "interface_rna_span_covered": row.get(
                            "interface_rna_span_covered"
                        ),
                        "interface_residues": row.get(
                            "interface_residues"
                        ),
                        "pose_training_label": row.get(
                            "pose_training_label"
                        ),
                        "memory_interface_score": interface_score,
                        "memory_interface_details": interface_details,
                        "timestamp": _now_iso(),
                    }
                )

                # Partial success if no accepted target OR this clean pose belongs
                # to a non-accepted/competing target.
                if not accepted_target_norm or row_target != accepted_target_norm:
                    self.record_partial_success_target(
                        row_target,
                        sequence=row.get("sequence"),
                        row=row,
                    )
            else:
                if _is_dock_valid(row):
                    if row.get("interface_steric_clash") is True:
                        clash_rows.append(row)
                        reason = "steric_clash"
                    elif row.get("interface_passed") is False:
                        reason = "interface_failed"
                    else:
                        reason = "interface_not_clean"

                    self.record_interface_failure(
                        row,
                        reason=reason,
                    )

        # ------------------------------------------------------------
        # Explicit failed targets from state
        # Supports legacy list[str] and new list[dict].
        # ------------------------------------------------------------
        for item in state.get("failed_target_pdbs", []) or []:
            if isinstance(item, dict):
                self.record_target_failure(
                    item.get("pdb_id"),
                    item.get("reason", "unknown"),
                    metadata=item,
                )
            elif item:
                self.record_target_failure(str(item), "unknown")

        # ------------------------------------------------------------
        # Derived target-level outcomes from interface results.
        # This ensures target failure reasons exist even if the protein agent
        # only stored raw binding rows.
        # ------------------------------------------------------------
        target_to_rows: dict[str, list[dict]] = {}

        for row in binding_results:
            if not isinstance(row, dict):
                continue

            pdb = _pdb_id(row.get("target_pdb"))

            if not pdb:
                continue

            target_to_rows.setdefault(pdb, []).append(row)

        for pdb, rows in target_to_rows.items():
            dock_valid_rows = [r for r in rows if _is_dock_valid(r)]
            clean_count = sum(1 for r in rows if _has_clean_interface(r))

            if not dock_valid_rows:
                continue

            # Do not mark the accepted target as failed if it exists.
            if accepted_target_norm and pdb == accepted_target_norm:
                continue

            if clean_count == 0:
                self.record_target_failure(
                    pdb,
                    "no_clean_interface",
                    {
                        "clean_count": 0,
                        "dock_valid_count": len(dock_valid_rows),
                        "row_count": len(rows),
                    },
                )

            elif clean_count == 1:
                self.record_target_failure(
                    pdb,
                    "only_one_clean_pose",
                    {
                        "clean_count": 1,
                        "dock_valid_count": len(dock_valid_rows),
                        "row_count": len(rows),
                    },
                )

        # ------------------------------------------------------------
        # Accepted target success memory
        # ------------------------------------------------------------
        if accepted_target_norm:
            accepted_rows = [
                r for r in binding_results
                if isinstance(r, dict)
                and _pdb_id(r.get("target_pdb")) == accepted_target_norm
            ]

            accepted_clean_count = sum(
                1 for r in accepted_rows
                if _has_clean_interface(r)
            )

            self.record_successful_target(
                accepted_target_norm,
                clean_pose_count=accepted_clean_count,
                binding_result_count=len(accepted_rows),
                metadata={
                    "research_topic": state.get("research_topic"),
                    "target_selection_reason": state.get("target_pdb_selection_reason"),
                },
            )

        # ------------------------------------------------------------
        # Energy / joint physics memory
        # ------------------------------------------------------------
        feedback = state.get("joint_physics_feedback", {}) or {}

        self.record_energy_failure(
            best_score=feedback.get("best_binding_score"),
            spread=feedback.get("binding_score_spread"),
            units=state.get("binding_units", "hdock_relative_score"),
            metadata={"source": "joint_physics_feedback"},
        )

        # ------------------------------------------------------------
        # Run summary
        # ------------------------------------------------------------
        if accepted_target_norm:
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

        self._compact_memory()
        self.save()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_hard_failed_targets(self) -> set[str]:
        out: set[str] = set()

        for row in self.memory.get("target_failures", []) or []:
            if not isinstance(row, dict):
                continue

            if row.get("reason") in self.HARD_TARGET_FAILURES:
                pdb = _pdb_id(row.get("pdb_id"))

                if pdb:
                    out.add(pdb)

        return out

    def get_soft_failed_targets(self) -> set[str]:
        out: set[str] = set()

        for row in self.memory.get("target_failures", []) or []:
            if not isinstance(row, dict):
                continue

            if row.get("reason") in self.SOFT_TARGET_FAILURES:
                pdb = _pdb_id(row.get("pdb_id"))

                if pdb:
                    out.add(pdb)

        return out

    def get_successful_targets(self) -> list[str]:
        hard_failed = self.get_hard_failed_targets()
        rows = self.memory.get("successful_targets", []) or []

        rows = sorted(
            rows,
            key=lambda r: (
                -(int(r.get("best_clean_pose_count") or 0)),
                -(int(r.get("success_count") or 0)),
                -(int(r.get("best_binding_result_count") or 0)),
            ),
        )

        out = [_pdb_id(r.get("pdb_id")) for r in rows if (_pdb_id(r.get("pdb_id")) and _pdb_id(r.get("pdb_id")) not in hard_failed)]

        return list(dict.fromkeys(out))

    def get_partial_success_targets(self) -> list[str]:
        """
        Return partial-success targets sorted by interface quality.

        Hard-failed targets are excluded.
        """
        hard_failed = self.get_hard_failed_targets()
        rows = self.memory.get("partial_success_targets", []) or []

        usable_rows = [
            r for r in rows
            if isinstance(r, dict)
            and _pdb_id(r.get("pdb_id"))
            and _pdb_id(r.get("pdb_id")) not in hard_failed
        ]

        usable_rows = sorted(
            usable_rows,
            key=lambda r: (
                -float(r.get("interface_quality_score") or 0.0),
                float(r.get("interface_min_distance_A") or 999.0),
                -float(r.get("interface_rna_span_covered") or 0.0),
            ),
        )

        out = [
            _pdb_id(r.get("pdb_id"))
            for r in usable_rows
            if _pdb_id(r.get("pdb_id"))
        ]

        return list(dict.fromkeys(out))

    def get_partial_success_sequences(self) -> list[str]:
        out: list[str] = []

        for row in self.memory.get("partial_success_sequences", []) or []:
            if isinstance(row, dict):
                seq = _clean_rna(row.get("sequence"))

                if seq:
                    out.append(seq)

        return list(dict.fromkeys(out))

    def get_failure_penalty(self, seq: str) -> float:
        """
        Returns a penalty scalar in [0, 1] based on similarity to past
        sequence-level failures.
        """
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

            matches = sum(
                1 for a, b in zip(seq[:n], failed_seq[:n])
                if a == b
            )
            similarity = matches / max(1, n)

            if similarity > 0.7:
                penalty += 0.3

        return min(penalty, 1.0)

    def get_energy_bias(self) -> float:
        """
        Returns scalar pressure for increasing binding exploration.

        HDOCK-relative scores are not physical free energies. This simply
        detects whether recent best scores are weak/near zero.
        """
        values: list[float] = []

        for item in self.memory.get("energy_failures", []) or []:
            if not isinstance(item, dict):
                continue

            value = item.get("best_score", item.get("best_dg"))
            value = _safe_float(value)

            if value is not None:
                values.append(value)

        if not values:
            return 0.0

        avg = sum(values) / len(values)

        if avg > -20:
            return 0.30

        if avg > -50:
            return 0.15

        return 0.0

    # ------------------------------------------------------------------
    # Backward-compatible update
    # ------------------------------------------------------------------

    def update(self, parsed: Dict) -> None:
        """
        Accept parsed Skeptic output and convert into memory entries.

        Expected parser keys may include:
          - best_binding_score
          - dg_best
          - spread
          - concerns
          - missing_controls
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

    def get_interface_evidence(
    self,
    target_pdb: str | None = None,
    ) -> list:
        """
        Return persistent pose-level interface evidence.

        If target_pdb is supplied, prefer evidence specific to that target.
        """
        target = _pdb_id(target_pdb)

        rows: list[dict] = []

        for collection in (
            "successful_pose_features",
            "interface_failures",
        ):
            for row in self.memory.get(collection, []) or []:
                if not isinstance(row, dict):
                    continue

                seq = _clean_rna(row.get("sequence"))
                row_target = _pdb_id(
                    row.get("target_pdb")
                    or row.get("pdb_id")
                )

                if not seq:
                    continue

                if target and row_target != target:
                    continue

                score, details = _interface_evidence_score(row)

                enriched = dict(row)
                enriched["sequence"] = seq
                enriched["target_pdb"] = row_target
                enriched["memory_interface_score"] = score
                enriched["memory_interface_details"] = details
                enriched["memory_source"] = collection

                rows.append(enriched)

        return rows

    def merge_state_memory(self, state: dict) -> None:
        """
        Merge in-memory LabState information into the persistent model.
        """

        if not isinstance(state, dict):
            return

        for item in state.get("partial_success_sequences", []) or []:
            seq = _clean_rna(item)

            if seq:
                self.memory["partial_success_sequences"].append(
                    {
                        "sequence": seq,
                        "target_pdb": "",
                        "timestamp": _now_iso(),
                    }
                )

        for item in state.get("target_failure_records", []) or []:
            if (
                isinstance(item, dict)
                and item.get("pdb_id")
            ):
                self.memory["target_failures"].append(dict(item))

        self._compact_memory()

    def merge_from_state(self, state: dict) -> None:
        """
        New higher-level state ingestion entrypoint.

        Consolidates:
        - partial success sequences
        - target failures
        - binding/interface evidence
        - accepted target outcomes

        Safe to call repeatedly.
        """
        if not isinstance(state, dict):
            return

        self.merge_state_memory(state)

        try:
            self.ingest_run_state(state)
        except Exception:
            pass

        self.save()

    def get_interface_prior(
        self,
        seq: str,
        target_pdb: str | None = None,
        min_similarity: float = INTERFACE_SIMILARITY_THRESHOLD,
    ) -> dict:
        """
        Obtain a target-specific interface prior for a sequence.

        Exact measured evidence is preferred. For unseen mutants, nearby sequence
        evidence is similarity-discounted. Unknown candidates remain neutral.
        """
        seq = _clean_rna(seq)

        if not seq:
            return {
                "known": False,
                "score": 0.0,
                "reason": "empty_sequence",
            }

        evidence = self.get_interface_evidence(target_pdb)

        if not evidence:
            return {
                "known": False,
                "score": 0.0,
                "reason": "no_target_interface_evidence",
            }

        weighted_score = 0.0
        total_weight = 0.0
        matches: list[dict] = []

        for row in evidence:
            observed_seq = _clean_rna(row.get("sequence"))

            if not observed_seq:
                continue

            similarity = _sequence_similarity(seq, observed_seq)

            if similarity < min_similarity:
                continue

            raw_score = _safe_float(
                row.get("memory_interface_score"),
                0.0,
            ) or 0.0

            weight = similarity ** INTERFACE_SIMILARITY_POWER

            # Exact observations should dominate inferred neighbours.
            if similarity >= 0.999999:
                weight *= 4.0

            weighted_score += weight * raw_score
            total_weight += weight

            matches.append(
                {
                    "observed_sequence": observed_seq,
                    "similarity": similarity,
                    "weight": weight,
                    "interface_score": raw_score,
                    "source": row.get("memory_source"),
                    "target_pdb": row.get("target_pdb"),
                    "steric_clash": row.get(
                        "interface_steric_clash"
                    ),
                    "clash_severity": row.get(
                        "interface_clash_severity"
                    ),
                    "interface_quality_score": row.get(
                        "interface_quality_score"
                    ),
                }
            )

        if total_weight <= 0.0:
            return {
                "known": False,
                "score": 0.0,
                "reason": "no_similar_interface_evidence",
            }

        prior_score = weighted_score / total_weight
        best_similarity = max(
            m["similarity"]
            for m in matches
        )

        # Additional confidence discount for inferred—not exact—candidates.
        confidence = best_similarity ** INTERFACE_SIMILARITY_POWER

        if best_similarity < 0.999999:
            prior_score *= confidence

        return {
            "known": True,
            "score": max(-1.5, min(1.5, prior_score)),
            "confidence": confidence,
            "best_similarity": best_similarity,
            "exact": best_similarity >= 0.999999,
            "target_pdb": _pdb_id(target_pdb),
            "matches": sorted(
                matches,
                key=lambda x: x["weight"],
                reverse=True,
            )[:5],
        }

    # ------------------------------------------------------------------
    # Objective weights
    # ------------------------------------------------------------------

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
        successful_targets = self.memory.get("successful_targets", []) or []

        if len(energy_failures) > 3:
            weights["binding_pressure"] = 0.2

        if len(structural_failures) > 3:
            weights["structure_pressure"] = 0.2

        if len(motif_failures) > 5:
            weights["diversity_pressure"] = 0.2

        clash_failures = [
            row
            for row in interface_failures
            if isinstance(row, dict)
            and row.get("interface_steric_clash") is True
        ]

        severe_or_moderate = [
            row
            for row in clash_failures
            if str(
                row.get("interface_clash_severity") or ""
            ).lower() in {"moderate", "severe"}
        ]

        if interface_failures:
            weights["interface_pressure"] = min(
                0.75,
                0.25 + 0.05 * len(interface_failures),
            )

        if clash_failures:
            weights["clash_avoidance_pressure"] = min(
                0.85,
                0.30 + 0.08 * len(clash_failures),
            )

        if severe_or_moderate:
            weights["clash_avoidance_pressure"] = max(
                weights.get("clash_avoidance_pressure", 0.0),
                0.65,
            )

        if interface_failures:
            weights["structure_pressure"] = max(
                weights.get("structure_pressure", 0.0),
                0.2,
            )

        if partial_success:
            weights["interface_pressure"] = max(
                weights.get("interface_pressure", 0.0),
                0.55,
            )
            weights["clash_avoidance_pressure"] = max(
                weights.get("clash_avoidance_pressure", 0.0),
                0.50,
            )
            partial_targets = {
                _pdb_id(row.get("pdb_id"))
                for row in partial_success
                if isinstance(row, dict)
                and _pdb_id(row.get("pdb_id"))
            }

            weights["target_specific_exploitation"] = min(
                1.0,
                0.2 + 0.1 * len(partial_targets),
            )

        if successful_targets:
            weights["target_reuse_pressure"] = 0.25
            weights["interface_pressure"] = max(
                weights.get("interface_pressure", 0.0),
                0.25,
            )

        spreads = []

        for item in energy_failures:
            if not isinstance(item, dict):
                continue

            spread = _safe_float(item.get("spread"))

            if spread is not None:
                spreads.append(spread)

        if spreads and sum(spreads) / len(spreads) < 3:
            weights["convergence_pressure"] = 0.3

        energy_bias = self.get_energy_bias()

        if energy_bias > 0:
            weights["binding_pressure"] = max(
                weights.get("binding_pressure", 0.0),
                energy_bias,
            )

        return weights

    # ------------------------------------------------------------------
    # Apply to scoring
    # ------------------------------------------------------------------

    def adjust_score(self, seq: str, score: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust a candidate score dict using persistent failure memory.

        This is intentionally lightweight because it runs inside optimisation.
        """
        score = dict(score)
        penalty = self.get_failure_penalty(seq)

        if penalty > 0:
            for key, value in list(score.items()):
                try:
                    if isinstance(value, (int, float)):
                        score[key] = float(value) * (1 - penalty)
                except Exception:
                    pass

        weights = self.compute_failure_weights()

        if "binding" in score:
            score["binding"] = float(score.get("binding", 0.0)) + weights.get(
                "binding_pressure",
                0.0,
            )

        if "structure" in score:
            score["structure"] = float(score.get("structure", 0.0)) + weights.get(
                "structure_pressure",
                0.0,
            )

        if "interface" in score:
            interface_value = float(
                score.get("interface", 0.0)
            )

            interface_known = bool(
                score.get("interface_evidence_known")
                or (
                    isinstance(score.get("interface_evidence"), dict)
                    and score["interface_evidence"].get("known")
                )
            )

            if interface_known:
                interface_pressure = weights.get(
                    "interface_pressure",
                    0.0,
                )
                clash_pressure = weights.get(
                    "clash_avoidance_pressure",
                    0.0,
                )

                if interface_value >= 0.0:
                    # Amplify genuinely favourable measured/prior evidence.
                    interface_value *= 1.0 + interface_pressure
                else:
                    # Make negative clash evidence more costly.
                    interface_value *= 1.0 + clash_pressure

            # Unknown candidates stay neutral instead of receiving a free bonus.
            score["interface"] = interface_value

        return score