"""
failure_memory.py

Tracks recurring failure modes across iterations and runs.
Used to:
- prevent repeated bad designs
- bias optimisation away from known failure regions
- provide learning signal for adaptive agents

Integrates with:
- Skeptic Agent (primary signal)
- PI Agent (feedback loop)
- TrainingDataCollector (optional persistence)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List


class FailureMemory:
    def __init__(self, path: str = "failure_memory.json"):
        self.path = path
        self.memory = {
            "sequence_failures": [],    # list of (sequence, reason)
            "motif_failures": [],       # recurring motif-level issues
            "energy_failures": [],      # weak binding / bad energy ranges
            "structural_failures": [],  # unstable folds
        }
        self._load()

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.memory = json.load(f)
            except Exception:
                pass

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # RECORD FAILURES
    # ------------------------------------------------------------------
    def record_sequence_failure(self, seq: str, reason: str):
        self.memory["sequence_failures"].append({
            "sequence": seq,
            "reason": reason
        })

    def record_energy_failure(self, best_dg: float | None, spread: float | None):
        if best_dg is None:
            return

        self.memory["energy_failures"].append({
            "best_dg": best_dg,
            "spread": spread
        })

    def record_structural_failure(self, stability: str, mfe: float | None):
        self.memory["structural_failures"].append({
            "stability": stability,
            "mfe": mfe
        })

    def record_motif_failure(self, motif_desc: str):
        if motif_desc:
            self.memory["motif_failures"].append(motif_desc)

    # ------------------------------------------------------------------
    # PARSE SKEPTIC OUTPUT
    # ------------------------------------------------------------------
    def ingest_skeptic_output(self, critique: str, sequences: List[str] | None = None):
        """
        Parse structured Skeptic output and update failure memory.
        """
        if not critique:
            return

        best_dg = None
        spread = None

        for line in critique.splitlines():
            line = line.strip()

            # ENERGY parsing
            if line.startswith("ENERGY:"):
                try:
                    parts = line.split(",")
                    for p in parts:
                        if "ΔG_best" in p:
                            best_dg = float(p.split("=")[1].strip())
                        if "spread" in p:
                            spread = float(p.split("=")[1].strip())
                except Exception:
                    pass

            # CONCERNS parsing
            if line.startswith("1.") or line.startswith("2."):
                reason = line[2:].strip()
                if sequences:
                    for seq in sequences:
                        self.record_sequence_failure(seq, reason)

            # Missing controls often imply structural weakness
            if line.startswith("MISSING_CONTROLS:"):
                self.record_motif_failure(line)

        self.record_energy_failure(best_dg, spread)

        self.save()

    # ------------------------------------------------------------------
    # QUERY MEMORY
    # ------------------------------------------------------------------
    def get_failure_penalty(self, seq: str) -> float:
        """
        Returns a penalty scalar (0–1) based on similarity to past failures.
        """
        penalty = 0.0

        for item in self.memory["sequence_failures"]:
            failed_seq = item["sequence"]

            if not failed_seq:
                continue

            # crude similarity: % identical positions
            matches = sum(1 for a, b in zip(seq, failed_seq) if a == b)
            similarity = matches / max(1, len(seq))

            if similarity > 0.7:
                penalty += 0.3

        return min(penalty, 1.0)

    def get_energy_bias(self) -> float:
        """
        Returns bias suggesting direction of optimisation
        (e.g., if most failures have weak ΔG)
        """
        values = [e["best_dg"] for e in self.memory["energy_failures"] if e["best_dg"]]

        if not values:
            return 0.0

        avg = sum(values) / len(values)

        # If binding is weak → push optimisation harder
        if avg > -5:
            return 0.2  # increase binding pressure

        return 0.0
        
    # ------------------------------------------------------------------
    # BACKWARD COMPATIBILITY (for orchestrator)
    # ------------------------------------------------------------------

    def update(self, parsed: Dict):
        """
        Accept parsed skeptic output and convert into memory entries.
        """
        if not parsed:
            return

        # Energy
        self.record_energy_failure(
            parsed.get("dg_best"),
            parsed.get("spread")
        )

        # Concerns → sequence/motif failures
        for c in parsed.get("concerns", []):
            self.record_motif_failure(c)

        # Missing controls → structural gaps
        if parsed.get("missing_controls"):
            self.record_structural_failure(
                stability="unknown",
                mfe=None
            )


    def compute_failure_weights(self) -> Dict:
        """
        Convert memory into objective bias weights for PI.
        """
        weights = {}

        # Energy failures → increase binding pressure
        if len(self.memory.get("energy_failures", [])) > 3:
            weights["binding_pressure"] = 0.2

        # Structural failures → improve structure objective
        if len(self.memory.get("structural_failures", [])) > 3:
            weights["structure_pressure"] = 0.2

        # Motif failures → diversity/exploration
        if len(self.memory.get("motif_failures", [])) > 5:
            weights["diversity_pressure"] = 0.2

        # Convergence detection
        spreads = [e.get("spread") for e in self.memory.get("energy_failures", []) if e.get("spread")]
        if spreads and sum(spreads) / len(spreads) < 3:
            weights["convergence_pressure"] = 0.3

        return weights

    # ------------------------------------------------------------------
    # APPLY TO SCORING
    # ------------------------------------------------------------------
    def adjust_score(self, seq: str, score: Dict) -> Dict:
        penalty = self.get_failure_penalty(seq)

        if penalty > 0:
            for k in score:
                score[k] *= (1 - penalty)

        bias = self.get_energy_bias()
        if bias > 0:
            score["binding"] += bias

        return score
