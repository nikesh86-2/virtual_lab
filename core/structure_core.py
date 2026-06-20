"""
structure_core.py

Unified structural scoring using:
- SFold (ensemble)
- ViennaRNA (thermodynamic)

Returns a SINGLE robust structural score
"""

from typing import Dict


def compute_structural_score(seq: str, wrappers: Dict) -> Dict:
    sfold = wrappers["sfold"]
    vienna = wrappers["vienna"]

    # --- SFold ---
    sf_res = sfold.run_sfold(seq)
    sf_pd = sf_res.get("pair_density") or 0.0

    # --- Vienna ---
    vr = vienna.run_rnafold(seq)
    mfe_raw = vr.get("mfe")
    mfe_score = vienna.score_mfe(mfe_raw, len(seq))
    vr_pd = vr.get("pair_density", 0.0)

    # ------------------------------------------------------------------
    # COMBINE SIGNALS
    # ------------------------------------------------------------------

    # weighted combination
    structure_score = (
        0.4 * sf_pd +
        0.3 * vr_pd +
        0.3 * mfe_score
    )

    return {
        "structure_score": structure_score,
        "sfold_pair_density": sf_pd,
        "vienna_pair_density": vr_pd,
        "mfe_score": mfe_score,
    }