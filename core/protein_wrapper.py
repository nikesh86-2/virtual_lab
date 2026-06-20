"""
protein_wrapper.py (UPGRADED)

Features:
- fast vs full evaluation modes
- structured outputs (dg, confidence, method)
- internal caching
- safe error handling
- orchestrator + optimiser compatible
"""

from __future__ import annotations

import random
from typing import List, Dict, Tuple


class ProteinWrapper:
    def __init__(self, simrna=None):
        """
        simrna: optional dependency (MD / docking backend)
        """
        self.simrna = simrna

        # ✅ internal cache (huge speedup)
        self._cache: Dict[Tuple[str, str], Dict] = {}

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def evaluate_sequences(
        self,
        pdb_id: str,
        sequences: List[str],
        fast: bool = True,
    ) -> List[Dict]:
        """
        Evaluate RNA sequences against a protein target

        Returns list of dict:
        {
            "sequence": str,
            "valid": bool,
            "dg": float,
            "confidence": float,
            "method": str
        }
        """

        results = []

        for seq in sequences:
            key = (pdb_id, seq)

            # ✅ cache check
            if key in self._cache:
                results.append(self._cache[key])
                continue

            try:
                if fast:
                    res = self._fast_evaluate(pdb_id, seq)
                else:
                    res = self._full_evaluate(pdb_id, seq)

            except Exception as e:
                res = {
                    "sequence": seq,
                    "valid": False,
                    "dg": None,
                    "confidence": 0.0,
                    "method": "error",
                    "error": str(e),
                }

            # ✅ store in cache
            self._cache[key] = res
            results.append(res)

        return results

    # ------------------------------------------------------------------
    # FAST MODE (used in optimisation loop)
    # ------------------------------------------------------------------
    def _fast_evaluate(self, pdb_id: str, seq: str) -> Dict:
        """
        Fast approximate binding evaluation

        This SHOULD be fast (used repeatedly during optimisation)
        """

        # --- simple heuristic features ---
        length = len(seq)
        gc_content = (seq.count("G") + seq.count("C")) / max(length, 1)

        # pseudo-binding estimation
        # (replace with real coarse docking if available)
        dg = -10.0 * gc_content - random.uniform(0, 2)

        confidence = 0.5 + 0.5 * gc_content  # proxy confidence

        return {
            "sequence": seq,
            "valid": True,
            "dg": dg,
            "confidence": confidence,
            "method": "fast_heuristic",
        }

    # ------------------------------------------------------------------
    # FULL MODE (optional, more accurate)
    # ------------------------------------------------------------------
    def _full_evaluate(self, pdb_id: str, seq: str) -> Dict:
        """
        More expensive evaluation using MD/docking backend if available
        """

        if not self.simrna:
            # fallback to fast
            return self._fast_evaluate(pdb_id, seq)

        try:
            result = self.simrna.run_binding(pdb_id, seq)

            dg = float(result.get("dg", 0.0))

            return {
                "sequence": seq,
                "valid": result.get("valid", True),
                "dg": dg,
                "confidence": result.get("confidence", 0.9),
                "method": "full_simulation",
            }

        except Exception as e:
            return {
                "sequence": seq,
                "valid": False,
                "dg": None,
                "confidence": 0.0,
                "method": "full_failed",
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # CACHE CONTROL (optional utility)
    # ------------------------------------------------------------------
    def clear_cache(self):
        self._cache = {}