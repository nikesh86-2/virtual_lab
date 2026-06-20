from typing import Dict
import logging

log = logging.getLogger("virtual_lab.docking")

# ----------------------------------------------------------------------
# RNA–PROTEIN DOCKING MODEL (HPC-SAFE HYBRID)
# ----------------------------------------------------------------------

class ProteinRNADocking:

    def __init__(self):
    
        pass

    # ------------------------------------------------------------------
    def score_binding(self, seq: str, sfold_res: Dict, md_res: Dict) -> Dict:
        """
        Hybrid docking score using:
        - accessibility
        - structure stability
        - 3D compactness
        - motif interaction

        Returns:
            {
                "binding_score": float
                "interface_score": float
                "accessibility": float
            }
        """

        # ---------------------------
        # 1. Accessibility (critical for binding)
        # ---------------------------
        structures = sfold_res.get("structures", [""])
        s = structures[0] if structures else ""

        accessible = s.count(".") / max(1, len(s))

        # ---------------------------
        # 2. Structural rigidity penalty
        # ---------------------------
        bpp = sfold_res.get("bpp_mean") or 0.0
        rigidity_penalty = bpp

        # ---------------------------
        # 3. 3D compactness (SimRNA)
        # ---------------------------
        compactness = md_res.get("compactness") or 0.0

        # ---------------------------
        # 4. Motif interaction potential
        # ---------------------------
        motif_score = self._motif_binding(seq)

        # ---------------------------
        # 5. Energy contribution
        # ---------------------------
        min_e = md_res.get("min_energy")
        energy_term = 0.0

        if min_e is not None:
            energy_term = -min_e / 50.0  # normalize

        # ---------------------------
        # FINAL DOCKING SCORE
        # ---------------------------
        binding_score = (
            0.35 * accessible +
            0.25 * motif_score +
            0.20 * compactness +
            0.20 * energy_term -
            0.25 * rigidity_penalty
        )

        interface_score = 0.5 * accessible + 0.5 * motif_score

        return {
            "binding_score": max(0.0, binding_score),
            "interface_score": interface_score,
            "accessibility": accessible,
        }

    # ------------------------------------------------------------------
    def _motif_binding(self, seq: str) -> float:
        """
        Motifs commonly involved in protein binding.
        """
        motifs = ["GGG", "GAG", "AGG", "UGG", "CGG"]

        score = 0.0
        for m in motifs:
            if m in seq:
                score += 0.2

        return min(score, 1.0)
