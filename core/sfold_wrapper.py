from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional
from VLAB2.core.gpu_manager import set_gpu_for_agent


log = logging.getLogger("virtual_lab.sfold")


class SFoldWrapper:

    def __init__(self, sfold_bin: str = "sfold", debug: bool = False):
        self.debug = debug
        self.device = set_gpu_for_agent(prefer=0)

        if os.path.isabs(sfold_bin) or os.path.exists(sfold_bin):
            self.sfold_bin = os.path.abspath(sfold_bin)
        else:
            self.sfold_bin = shutil.which(sfold_bin)

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------
    def run_sfold(self, sequence: str, timeout: int = 300) -> Dict:

        sequence = (sequence or "").strip().upper().replace("T", "U")

        if not self.sfold_bin:
            return self._fail(sequence, "SFold binary not found")

        if not self._is_valid_rna(sequence):
            return self._fail(sequence, "Invalid RNA sequence")

        with tempfile.TemporaryDirectory() as tmpdir:

            seq_file = os.path.join(tmpdir, "input.fa")

            try:
                # ---------------------------
                # WRITE INPUT
                # ---------------------------
                with open(seq_file, "w") as f:
                    f.write(">query\n")
                    f.write(sequence + "\n")

                # ---------------------------
                # RUN SFOLD
                # ---------------------------
                proc = subprocess.run(
                    [self.sfold_bin, seq_file],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if proc.returncode != 0:
                    return self._fail(sequence, proc.stderr)

                raw_output = proc.stdout or ""

                # ---------------------------
                # FIND OUTPUT DIR (ROBUST)
                # ---------------------------
                output_dir = os.path.join(tmpdir, "output")
                if not os.path.exists(output_dir):
                    output_dir = tmpdir  # fallback

                # ---------------------------
                # DEBUG FILE LISTING
                # ---------------------------
                if self.debug:
                    for root, _, files in os.walk(tmpdir):
                        for name in files:
                            path = os.path.join(root, name)
                            log.debug("FILE: %s (%d bytes)", path, os.path.getsize(path))

                # ---------------------------
                # ✅ STRUCTURE EXTRACTION
                # ---------------------------
                structures = self._extract_structures_from_dir(output_dir)

                # ✅ safe cluster extraction (no crash if missing)
                cluster_structures = self._safe_extract_cluster_structures(output_dir)
                clusters = self._safe_extract_clusters(output_dir)

                # ---------------------------
                # ✅ FAIL → FALLBACK
                # ---------------------------
                if not structures:
                    return self._fallback(sequence, raw_output)

                # ---------------------------
                # METRICS
                # ---------------------------
                pair_density = self._estimate_pair_density(structures, len(sequence)) or 0.0
                mfe = self._extract_mfe(output_dir)
                bpp_score = self._extract_bpp(output_dir)

                # ---------------------------
                # CLUSTER STATS (SAFE)
                # ---------------------------
                energy_mean = None
                prob_mean = None
                weighted_energy = None
                entropy = None

                if clusters:
                    energies = [c["energy"] for c in clusters if c.get("energy") is not None]
                    probs = [c["prob"] for c in clusters if c.get("prob") is not None]

                    if energies:
                        energy_mean = sum(energies) / len(energies)

                    if probs:
                        prob_mean = sum(probs) / len(probs)

                    if energies and probs and len(energies) == len(probs):
                        weighted_energy = sum(p * e for p, e in zip(probs, energies))

                    if probs:
                        import math
                        entropy = -sum(p * math.log(p + 1e-12) for p in probs)

                return {
                    "sequence": sequence,
                    "valid": True,
                    "error": None,

                    "pair_density": round(pair_density, 4),
                    "structure_count": len(structures),
                    "cluster_count": len(clusters),

                    "cluster_energy_mean": energy_mean,
                    "cluster_prob_mean": prob_mean,
                    "cluster_weighted_energy": weighted_energy,
                    "ensemble_entropy": entropy,

                    "mfe": mfe,
                    "bpp_mean": bpp_score,

                    "structures": structures,
                    "cluster_structures": cluster_structures,

                    "files_generated": len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0,
                    "raw_output": raw_output,

                    "summary": self._build_summary(structures, pair_density, energy_mean),
                }

            except Exception as e:
                return self._fail(sequence, str(e))

    # ------------------------------------------------------------------
    # STRUCTURE EXTRACTION
    # ------------------------------------------------------------------
    def _extract_structures_from_dir(self, directory: str) -> List[str]:

        if not os.path.exists(directory):
            return []

        structures = []
        candidates = ["ecentroid.ct", "centroid.ct", "sample_1.ct"]

        for fname in candidates:
            path = os.path.join(directory, fname)
            if not os.path.exists(path):
                continue

            try:
                structures += self._parse_ct_file(path)
            except Exception as e:
                log.debug("CT parse failed for %s: %s", fname, e)

        return structures

    def _safe_extract_cluster_structures(self, directory: str) -> List[str]:
        try:
            return self._extract_cluster_structures(directory)
        except Exception:
            return []

    def _safe_extract_clusters(self, directory: str) -> List[Dict]:
        try:
            return self._extract_clusters(directory)
        except Exception:
            return []

    def _extract_cluster_structures(self, directory: str) -> List[str]:

        cluster_dir = os.path.join(directory, "clusters")

        if not os.path.exists(cluster_dir):
            return []

        structures = []

        for name in os.listdir(cluster_dir):
            if not name.endswith(".ct"):
                continue

            path = os.path.join(cluster_dir, name)

            try:
                structures += self._parse_ct_file(path)
            except Exception:
                continue

        return structures

    def _parse_ct_file(self, path: str) -> List[str]:

        with open(path) as f:
            lines = f.readlines()

        pairs = {}
        max_i = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            if not parts[0].isdigit() or not parts[4].isdigit():
                continue

            i = int(parts[0])
            j = int(parts[4])

            max_i = max(max_i, i)

            if j > 0:
                pairs[i] = j

        if max_i == 0:
            return []

        structure = ["." for _ in range(max_i)]

        for i, j in pairs.items():
            if i < j:
                structure[i - 1] = "("
                structure[j - 1] = ")"

        return ["".join(structure)]

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _extract_clusters(self, directory: str) -> List[Dict]:

        clusters = []
        fpath = os.path.join(directory, "sclass.out")

        if not os.path.exists(fpath):
            return clusters

        with open(fpath) as f:
            for line in f:
                m = re.search(r"([0-9]*\.[0-9]+)\s+(-?\d+\.\d+)", line)
                if m:
                    clusters.append({
                        "prob": float(m.group(1)),
                        "energy": float(m.group(2))
                    })

        return clusters

    def _extract_bpp(self, directory: str) -> Optional[float]:

        path = os.path.join(directory, "bp.out")
        if not os.path.exists(path):
            return None

        probs = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        probs.append(float(parts[2]))
                    except:
                        pass

        return sum(probs) / len(probs) if probs else None

    def _estimate_pair_density(self, structures: List[str], seq_len: int) -> Optional[float]:

        if not structures:
            return None

        return sum(
            sum(c in "()" for c in s) / seq_len
            for s in structures
        ) / len(structures)

    def _extract_mfe(self, directory: str) -> Optional[float]:

        path = os.path.join(directory, "fe.out")

        if not os.path.exists(path):
            return None

        with open(path) as f:
            for line in f:
                m = re.search(r"-?\d+\.\d+", line)
                if m:
                    return float(m.group())

        return None


    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _is_valid_rna(self, seq: str) -> bool:
        if not seq:
            return False

        seq = seq.upper().replace("T", "U")

        # optionally allow N but reject others
        allowed = {"A", "U", "G", "C", "N"}
        return all(base in allowed for base in seq)


    # ------------------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------------------
    def _fallback(self, sequence: str, raw_output: str):

        return {
            "sequence": sequence,
            "valid": False,
            "error": "No structures parsed",

            "pair_density": 0.0,
            "structure_count": 1,
            "cluster_count": 0,

            "mfe": 0.0,
            "bpp_mean": 0.0,

            "cluster_energy_mean": 0.0,
            "cluster_prob_mean": 0.0,
            "cluster_weighted_energy": 0.0,
            "ensemble_entropy": 0.0,

            "structures": ["." * len(sequence)],
            "cluster_structures": [],

            "files_generated": 0,
            "raw_output": raw_output,

            "summary": "Fallback flat structure",
        }

    # ------------------------------------------------------------------
    #BUILD SUMMARY
    # ------------------------------------------------------------------

    def _build_summary(self, structures, density, energy):

        density_str = f"{density:.3f}" if density is not None else "NA"
        energy_str = f"{energy:.3f}" if energy is not None else "NA"

        return f"{len(structures)} structures | density≈{density_str} | energy≈{energy_str}"

    def _fail(self, sequence, error):

        return {
            "sequence": sequence,
            "valid": False,
            "error": error,

            "pair_density": 0.0,
            "structure_count": 0,
            "cluster_count": 0,

            "files_generated": 0,
            "raw_output": "",
            "summary": None,
        }