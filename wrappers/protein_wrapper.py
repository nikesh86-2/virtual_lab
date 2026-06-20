from __future__ import annotations

import logging
import os
from typing import List, Optional

log = logging.getLogger("protein_wrapper")


class ProteinWrapper:
    """
    Main RNA-protein scoring class.

    Public method:
        evaluate_sequences(pdb_id, sequences)

    Optional:
        self.target_sequence can be set externally if you want structure prediction
        fallback to use a protein sequence later.
    """

    def __init__(self, pdb_dir: str = "pdb_cache", simrna=None):
        self.pdb_dir = pdb_dir
        self.simrna = simrna
        self.target_sequence: Optional[str] = None

        os.makedirs(self.pdb_dir, exist_ok=True)
        os.makedirs("rna_cache", exist_ok=True)
        os.makedirs("complex_cache", exist_ok=True)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def evaluate_sequences(self, pdb_id: str, sequences: List[str]) -> List[dict]:
        """
        Evaluate a list of RNA sequences against one protein target.

        Returns a ranked list of:
            {
                "sequence": ...,
                "dg": ...,
                "dock_score": ...,
                "electro": ...,
                "valid": True/False,
                "rank": 1,2,3...
            }
        """
        pdb_path = self._fetch_or_predict_structure(pdb_id)
        if not pdb_path:
            log.warning("Could not obtain structure for target %s", pdb_id)
            return []

        protein_atoms, residue_names = self._load_protein_atoms(pdb_path)

        if protein_atoms.size == 0:
            log.warning("Protein structure contained no readable atoms: %s", pdb_path)
            return []

