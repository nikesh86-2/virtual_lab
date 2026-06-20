"""
viennarna_wrapper.py

Wrapper for ViennaRNA command-line tools.

ELI5:
This file asks:
- "What secondary structure does this RNA probably fold into?"
- "How stable is that fold?"
- "If I have multiple aligned RNAs, what conserved structure do they share?"

It uses:
- RNAfold     -> single-sequence folding
- RNAalifold  -> consensus folding from an alignment
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Optional

log = logging.getLogger("virtual_lab.viennarna")


class ViennaRNAWrapper:
    """
    Lightweight wrapper around ViennaRNA command-line tools.
    """

    def __init__(self):
        self.rnafold_bin = shutil.which("RNAfold")
        self.rnaalifold_bin = shutil.which("RNAalifold")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_rnafold(self, sequence: str, timeout: int = 30) -> Dict:
        """
        Run RNAfold on a single RNA sequence.

        Returns:
            {
                "sequence": ...,
                "valid": True/False,
                "error": None or message,
                "structure": str or None,
                "mfe": float or None,
                "raw_output": str,
            }

        Notes:
        - mfe = minimum free energy
        - lower (more negative) mfe is generally more stable
        """
        sequence = (sequence or "").strip().upper().replace("T", "U")

        if not self.rnafold_bin:
            return self._fail(sequence, "RNAfold binary not found")

        if not self._is_valid_rna(sequence):
            return self._fail(sequence, "Invalid RNA sequence")

        log.info("Running RNAfold on %d nt sequence (timeout=%ss)...", len(sequence), timeout)

        try:
            proc = subprocess.run(
                [self.rnafold_bin],
                input=sequence + "\n",
                text=True,
                capture_output=True,
                timeout=timeout,
            )

            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "").strip()
                return self._fail(sequence, msg[:500] or "RNAfold failed")

            raw = (proc.stdout or "").strip()

            structure, mfe = self._parse_rnafold_output(raw)

            if structure is None or mfe is None:
                return self._fail(sequence, "Could not parse RNAfold output", raw_output=raw)

            log.info("RNAfold completed: %s ( %.2f )", structure, mfe)

            return {
                "sequence": sequence,
                "valid": True,
                "error": None,
                "structure": structure,
                "mfe": float(mfe),
                "raw_output": raw,
            }

        except subprocess.TimeoutExpired as e:
            return self._fail(sequence, f"Timeout after {e.timeout}s")

        except Exception as e:
            return self._fail(sequence, str(e))

    def run_rnaalifold(self, msa_text: str, timeout: int = 60) -> Dict:
        """
        Run RNAalifold on a multiple sequence alignment in FASTA-like format.

        Input:
            msa_text: aligned sequences as text

        Returns:
            {
                "valid": True/False,
                "error": None or message,
                "structure": str or None,
                "mfe": float or None,
                "raw_output": str,
            }

        Notes:
        - This assumes you already prepared an alignment externally.
        - RNAalifold expects aligned sequences of equal length.
        """
        msa_text = (msa_text or "").strip()

        if not self.rnaalifold_bin:
            return self._fail_msa("RNAalifold binary not found")

        if not msa_text or ">" not in msa_text:
            return self._fail_msa("Invalid or empty MSA input")

        log.info("Running RNAalifold on MSA (timeout=%ss)...", timeout)

        # RNAalifold is easiest to run using a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            msa_file = os.path.join(tmpdir, "input.aln")

            try:
                with open(msa_file, "w") as f:
                    f.write(msa_text + "\n")

                proc = subprocess.run(
                    [self.rnaalifold_bin, msa_file],
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    cwd=tmpdir,
                )

                if proc.returncode != 0:
                    msg = (proc.stderr or proc.stdout or "").strip()
                    return self._fail_msa(msg[:500] or "RNAalifold failed")

                raw = (proc.stdout or "").strip()

                structure, mfe = self._parse_rnaalifold_output(raw)

                if structure is None or mfe is None:
                    return self._fail_msa("Could not parse RNAalifold output", raw_output=raw)

                log.info("RNAalifold completed: %s ( %.2f )", structure, mfe)

                return {
                    "valid": True,
                    "error": None,
                    "structure": structure,
                    "mfe": float(mfe),
                    "raw_output": raw,
                }

            except subprocess.TimeoutExpired as e:
                return self._fail_msa(f"Timeout after {e.timeout}s")

            except Exception as e:
                return self._fail_msa(str(e))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _is_valid_rna(self, sequence: str) -> bool:
        if not sequence or len(sequence) < 5:
            return False

        if any(c not in "ACGU" for c in sequence):
            return False

        return True

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    def _parse_rnafold_output(self, raw: str) -> tuple[Optional[str], Optional[float]]:
        """
        Typical RNAfold output looks roughly like:

            GGGAAAUCC
            (((...))) (-3.40)

        Sometimes extra spacing exists, so parse defensively.
        """
        if not raw:
            return None, None

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) < 2:
            return None, None

        # Usually the last line contains structure + energy
        candidate = lines[-1]

        structure = self._extract_dot_bracket(candidate)
        mfe = self._extract_energy(candidate)

        return structure, mfe

    def _parse_rnaalifold_output(self, raw: str) -> tuple[Optional[str], Optional[float]]:
        """
        Typical RNAalifold output often includes a consensus line followed by:
            (((...))) (-5.60 = ...)
        or similar

        We just extract:
        - first valid dot-bracket string
        - first energy value in parentheses
        """
        if not raw:
            return None, None

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return None, None

        # Find the first line that looks like a structure line
        for line in lines:
            structure = self._extract_dot_bracket(line)
            mfe = self._extract_energy(line)

            if structure is not None and mfe is not None:
                return structure, mfe

        return None, None

    def _extract_dot_bracket(self, text: str) -> Optional[str]:
        """
        Extract a dot-bracket-like structure string.
        """
        # Accept dot-bracket and some extended notation characters
        match = re.search(r"([().\[\]{}<>]+)", text)
        if not match:
            return None

        structure = match.group(1).strip()

        # Sanity check
        if len(structure) < 3:
            return None

        return structure

    def _extract_energy(self, text: str) -> Optional[float]:
        """
        Extract the first float inside parentheses.
        Example:
            '(((...))) (-3.40)'
            '(((...))) (-5.60 = -4.20 + -1.40)'
        """
        match = re.search(r"\(\s*(-?\d+(?:\.\d+)?)", text)
        if not match:
            return None

        try:
            return float(match.group(1))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Consistent failure helpers
    # ------------------------------------------------------------------
    def _fail(self, sequence: str, error: str, raw_output: str = "") -> Dict:
        return {
            "sequence": sequence,
            "valid": False,
            "error": error,
            "structure": None,
            "mfe": None,
            "raw_output": raw_output,
        }

    def _fail_msa(self, error: str, raw_output: str = "") -> Dict:
        return {
            "valid": False,
            "error": error,
            "structure": None,
            "mfe": None,
            "raw_output": raw_output,
        }