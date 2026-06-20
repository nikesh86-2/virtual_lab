"""
md_wrapper.py (FINAL + STRUCTURE-ENABLED)

SimRNA wrapper with:
- energy stats
- structural metrics
- ✅ RNA PDB extraction (for docking)
- ✅ persistent structure storage
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import shutil
import hashlib
from typing import Dict, List

from VLAB2.core.gpu_manager import set_gpu_for_agent

log = logging.getLogger("virtual_lab.md")


class MDWrapper:

    def __init__(self, simrna_bin: str = "SimRNA_bin", data_dir: str = "data"):
        self.simrna_bin = os.path.abspath(simrna_bin)
        self.data_dir = os.path.abspath(data_dir)
        self.device = set_gpu_for_agent(prefer=1)
        # ✅ persistent structure output directory
        self.output_dir = "/tmp/rna_structures"
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def run_md(self, sequence: str, iterations: int = 1000) -> Dict:

        sequence = (sequence or "").strip().upper().replace("T", "U")

        if not os.path.exists(self.simrna_bin):
            return self._fail(sequence, "SimRNA binary missing")

        if not os.path.isdir(self.data_dir):
            return self._fail(sequence, "SimRNA data dir missing")

        if not self._is_valid_rna(sequence):
            return self._fail(sequence, "Invalid RNA")

        with tempfile.TemporaryDirectory() as tmpdir:

            seq_file = os.path.join(tmpdir, "input.seq")
            data_link = os.path.join(tmpdir, "data")

            try:
                # ---------------------------
                # WRITE INPUT
                # ---------------------------
                with open(seq_file, "w") as f:
                    f.write(sequence + "\n")

                if not os.path.exists(data_link):
                    os.symlink(self.data_dir, data_link)

                # ---------------------------
                # RUN SIMRNA
                # ---------------------------
                proc = subprocess.run(
                    [self.simrna_bin, "-s", seq_file, "-n", str(iterations)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if proc.returncode != 0:
                    return self._fail(sequence, proc.stderr)

                # ---------------------------
                # PARSE RESULTS
                # ---------------------------
                energies = self._parse_trafl(os.path.join(tmpdir, "input.seq.trafl"))

                if not energies:
                    return self._fail(sequence, "No energies")

                mean_e = sum(energies) / len(energies)
                min_e = min(energies)
                max_e = max(energies)
                fluct = max_e - min_e

                base_pairs = self._count_bonds(os.path.join(tmpdir, "input.seq.bonds"))

                structure = self._read_structure(
                    os.path.join(tmpdir, "input.seq-000001.ss_detected")
                )

                # ---------------------------
                # ✅ EXTRACT RNA PDB (CRITICAL FIX)
                # ---------------------------
                rna_pdb = self._extract_pdb(tmpdir, sequence)

                # ---------------------------
                # METRICS
                # ---------------------------
                stability_index = abs(min_e) / (1 + fluct)
                compactness = base_pairs / len(sequence)

                return {
                    "sequence": sequence,
                    "valid": True,
                    "error": None,

                    "mean_energy": round(mean_e, 3),
                    "min_energy": round(min_e, 3),
                    "energy_fluctuation": round(fluct, 3),

                    "base_pairs": base_pairs,
                    "secondary_structure": structure,

                    "stability_index": round(stability_index, 3),
                    "compactness": round(compactness, 3),

                    # ✅ NEW: structure path for docking
                    "rna_pdb": rna_pdb,
                }

            except Exception as e:
                return self._fail(sequence, str(e))

    # ------------------------------------------------------------------
    def _extract_pdb(self, tmpdir: str, sequence: str) -> str | None:
        """
        Extract a PDB structure from SimRNA output and persist it.
        """

        # SimRNA often outputs multiple structures
        candidates = [
            f for f in os.listdir(tmpdir)
            if f.endswith(".pdb")
        ]

        if not candidates:
            return None

        # ✅ choose first (safe baseline)
        chosen = os.path.join(tmpdir, sorted(candidates)[0])

        # ✅ create stable filename
        hash_id = hashlib.md5(sequence.encode()).hexdigest()[:10]
        final_path = os.path.join(self.output_dir, f"{hash_id}.pdb")

        try:
            shutil.copy(chosen, final_path)
            return final_path
        except Exception as e:
            log.warning("Failed to persist RNA PDB: %s", e)
            return None

    # ------------------------------------------------------------------
    def _parse_trafl(self, path: str) -> List[float]:
        energies = []

        if not os.path.exists(path):
            return energies

        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    energies.append(float(parts[-2]))
                except:
                    continue

        return energies

    def _count_bonds(self, path: str) -> int:
        if not os.path.exists(path):
            return 0

        with open(path) as f:
            return sum(1 for line in f if line.strip())

    def _read_structure(self, path: str):
        if not os.path.exists(path):
            return None

        with open(path) as f:
            lines = f.readlines()

        if len(lines) > 2:
            return lines[2].strip()

        return None

    def _is_valid_rna(self, seq: str) -> bool:
        return seq and all(c in "ACGU" for c in seq)

    def _fail(self, seq, err):
        return {
            "sequence": seq,
            "valid": False,
            "error": err,
            "mean_energy": None,
            "min_energy": None,
            "energy_fluctuation": None,
            "base_pairs": 0,
            "secondary_structure": None,
            "stability_index": None,
            "compactness": None,

            # ✅ ensure downstream safety
            "rna_pdb": None,
        }
