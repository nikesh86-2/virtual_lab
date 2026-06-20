from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import Dict, Optional

log = logging.getLogger("virtual_lab.vienna")


class ViennaRNAWrapper:

    def __init__(self, rnafold_bin: str = "RNAfold", rnaalifold_bin: str = "RNAalifold"):
        self.rnafold_bin = os.getenv("RNAFOLD_BIN", rnafold_bin)
        self.rnaalifold_bin = os.getenv("RNAALIFOLD_BIN", rnaalifold_bin)
        self._cache = {}

        self.min_pair_density = float(os.getenv("VLAB_RNA_MIN_ACCEPT_PAIR_DENSITY", "0.24"))
        self.min_mfe_per_nt = float(os.getenv("VLAB_RNA_MIN_ACCEPT_MFE_PER_NT", "-0.08"))

    # ------------------------------------------------------------------
    def run_rnafold(self, sequence: str, timeout: int = 60) -> Dict:
        sequence = (sequence or "").strip().upper().replace("T", "U")

        if not self._is_valid_rna(sequence):
            return self._fail_rnafold(sequence, "Invalid RNA sequence")

        if sequence in self._cache:
            return self._cache[sequence]

        with tempfile.TemporaryDirectory() as tmpdir:
            seq_file = os.path.join(tmpdir, "input.fa")

            try:
                with open(seq_file, "w") as f:
                    f.write(f">seq\n{sequence}\n")

                proc = subprocess.run(
                    [self.rnafold_bin, "--noPS", seq_file],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if proc.returncode != 0:
                    return self._fail_rnafold(sequence, proc.stderr)

                output = proc.stdout.strip()

                structure, mfe = self._parse_rnafold_output(output)

                if structure is None:
                    log.warning("Vienna parse failed, forcing fallback")
                    return {
                        "sequence": sequence,
                        "structure": None,
                        "mfe": None,
                        "mfe_per_nt": None,
                        "pair_density": 0.0,
                        "valid": False,
                        "passes_min_fold_thresholds": False,
                        "threshold_reasons": ["parse_fail"],
                        "error": "parse_fail",
                    }

                pair_density = self._pair_density(structure)
                mfe_per_nt = mfe / max(1, len(sequence)) if mfe is not None else None
                threshold = self._fold_threshold_status(pair_density, mfe_per_nt)

                result = {
                    "sequence": sequence,
                    "structure": structure,
                    "mfe": round(mfe, 3) if mfe is not None else None,
                    "mfe_per_nt": round(mfe_per_nt, 5) if mfe_per_nt is not None else None,
                    "pair_density": pair_density,
                    "valid": True,
                    "passes_min_fold_thresholds": threshold["passed"],
                    "threshold_reasons": threshold["reasons"],
                    "error": None,
                }

                self._cache[sequence] = result
                return result

            except Exception as e:
                return self._fail_rnafold(sequence, str(e))

    # ------------------------------------------------------------------
    def _pair_density(self, structure: Optional[str]) -> float:
        if not structure:
            return 0.0

        paired = sum(1 for c in structure if c in "()[]{}<>")
        return paired / (2.0 * max(len(structure), 1))

    # ------------------------------------------------------------------
    def _parse_rnafold_output(self, output: str):
        lines = output.split("\n")

        for line in lines:
            m = re.search(r"([().\[\]{}<>]+)\s+\(\s*(-?\d+(?:\.\d+)?)\s*\)", line)
            if m:
                return m.group(1), float(m.group(2))

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            first = parts[0]
            if set(first) <= set(".()[]{}<>"):
                mfe = None
                m = re.search(r"-?\d+(?:\.\d+)?", line)
                if m:
                    mfe = float(m.group(0))
                return first, mfe

        log.warning("Vienna failed to parse output:\n%s", output[:300])
        return None, None

    # ------------------------------------------------------------------
    def _fold_threshold_status(self, pair_density: Optional[float], mfe_per_nt: Optional[float]) -> dict:
        reasons = []

        if pair_density is None:
            reasons.append("missing_pair_density")
        elif pair_density < self.min_pair_density:
            reasons.append(
                f"pair_density_below_min:{pair_density:.3f}<{self.min_pair_density:.3f}"
            )

        if mfe_per_nt is None:
            reasons.append("missing_mfe")
        elif mfe_per_nt > self.min_mfe_per_nt:
            reasons.append(
                f"mfe_per_nt_not_negative_enough:{mfe_per_nt:.3f}>{self.min_mfe_per_nt:.3f}"
            )

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    def score_mfe(self, mfe: Optional[float], length: int) -> float:
        if mfe is None:
            return 0.0
        norm = abs(mfe) / max(length, 1)
        return min(norm / 5.0, 1.0)

    # ------------------------------------------------------------------
    def _is_valid_rna(self, sequence: str) -> bool:
        return bool(sequence) and all(c in "ACGU" for c in sequence)

    # ------------------------------------------------------------------
    def _fail_rnafold(self, sequence: str, error: str):
        return {
            "sequence": sequence,
            "structure": None,
            "mfe": None,
            "mfe_per_nt": None,
            "pair_density": 0.0,
            "valid": False,
            "passes_min_fold_thresholds": False,
            "threshold_reasons": [error or "rnafold_failed"],
            "error": error,
        }