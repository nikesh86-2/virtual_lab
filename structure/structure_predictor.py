"""
Structure Predictor with Fallback

ELI5:
If we can't find a protein structure,
we try to PREDICT one instead.
"""

import os
import logging

log = logging.getLogger("structure_predictor")


class StructurePredictor:

    def __init__(self, pdb_dir="cache/pdb_cache"):
        self.pdb_dir = pdb_dir
        os.makedirs(pdb_dir, exist_ok=True)

    def get_structure(self, pdb_id: str, sequence: str = None):
        """
        Try:
        1. Local PDB
        2. Download PDB
        3. Predict structure (fallback)
        """

        local = self._check_local(pdb_id)
        if local:
            return local

        fetched = self._fetch_pdb(pdb_id)
        if fetched:
            return fetched

        # 🔥 FALLBACK → predict
        log.warning("Falling back to structure prediction...")

        predicted = self._predict_structure(sequence)
        return predicted

    # ------------------------------
    def _check_local(self, pdb_id):
        path = os.path.join(self.pdb_dir, pdb_id + ".pdb")
        return path if os.path.exists(path) else None

    # ------------------------------
    def _fetch_pdb(self, pdb_id):
        try:
            from Bio.PDB import PDBList

            pdbl = PDBList()
            path = pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_dir)

            return path if os.path.exists(path) else None

        except Exception as e:
            log.warning("PDB fetch failed: %s", e)
            return None

    # ------------------------------
    def _predict_structure(self, sequence):
        """
        SUPER SIMPLE placeholder prediction

        ✅ you can swap later with:
        - ESMFold
        - AlphaFold
        """

        if not sequence:
            return None

        out = os.path.join(self.pdb_dir, "predicted.pdb")

        try:
            with open(out, "w") as f:
                for i, aa in enumerate(sequence):
                    f.write(
                        f"ATOM  {i:5d} CA  ALA A{i:4d}    "
                        f"{i*1.5:8.3f}{0:8.3f}{0:8.3f}\n"
                    )

            return out

        except Exception as e:
            log.error("Prediction failed: %s", e)
            return None