from __future__ import annotations

import os


def _first_non_none(*values):
    """
    Return first value that is not None.

    Important: do NOT use `or` for fold metrics because 0.0 is a valid
    scientific value and should not be skipped.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def extract_fold_quality_from_outputs(
    sfold_output: dict | None = None,
    vienna_output: dict | None = None,
    alifold_output: dict | None = None,
) -> dict:
    """
    Normalise folding quality fields from SFold/Vienna/RNAalifold outputs.

    This preserves the monolith behaviour of combining metrics from all
    available fold engines, while fixing the 0.0-vs-None bug.
    """
    sfold_output = sfold_output or {}
    vienna_output = vienna_output or {}
    alifold_output = alifold_output or {}

    pair_density_values = []
    mfe_values = []
    mfe_per_nt_values = []
    reasons = []

    for label, obj in [
        ("sfold", sfold_output),
        ("vienna", vienna_output),
        ("alifold", alifold_output),
    ]:
        if not isinstance(obj, dict):
            continue

        pd = _to_float_or_none(obj.get("pair_density"))
        mfe = _to_float_or_none(obj.get("mfe"))
        mfe_per_nt = _to_float_or_none(obj.get("mfe_per_nt"))

        if pd is not None:
            pair_density_values.append(pd)

        if mfe is not None:
            mfe_values.append(mfe)

        if mfe_per_nt is not None:
            mfe_per_nt_values.append(mfe_per_nt)

        if obj.get("threshold_reasons"):
            reasons.extend(
                [f"{label}:{x}" for x in obj.get("threshold_reasons", [])]
            )

        if obj.get("passes_min_fold_thresholds") is False:
            reasons.append(f"{label}:failed_min_fold_thresholds")

    best_pair_density = max(pair_density_values) if pair_density_values else None
    best_mfe = min(mfe_values) if mfe_values else None
    best_mfe_per_nt = min(mfe_per_nt_values) if mfe_per_nt_values else None

    try:
        min_pd = float(os.getenv("VLAB_RNA_MIN_ACCEPT_PAIR_DENSITY", "0.24"))
    except Exception:
        min_pd = 0.24

    try:
        min_mfe_per_nt = float(os.getenv("VLAB_RNA_MIN_ACCEPT_MFE_PER_NT", "-0.08"))
    except Exception:
        min_mfe_per_nt = -0.08

    passed = True

    if best_pair_density is None:
        passed = False
        reasons.append("missing_pair_density")
    elif best_pair_density < min_pd:
        passed = False
        reasons.append(
            f"pair_density_below_min:{best_pair_density:.3f}<{min_pd:.3f}"
        )

    if best_mfe_per_nt is None:
        passed = False
        reasons.append("missing_mfe_per_nt")
    elif best_mfe_per_nt > min_mfe_per_nt:
        passed = False
        reasons.append(
            f"mfe_per_nt_not_negative_enough:{best_mfe_per_nt:.3f}>{min_mfe_per_nt:.3f}"
        )

    return {
        "passed": passed,
        "best_pair_density": best_pair_density,
        "best_mfe": best_mfe,
        "best_mfe_per_nt": best_mfe_per_nt,
        "min_pair_density": min_pd,
        "min_mfe_per_nt": min_mfe_per_nt,
        "reasons": list(dict.fromkeys(reasons)),
    }


def motif_summary_from_state(state: dict) -> str:
    """
    Build a short motif summary from conservation and optimiser state.
    Handles both the new coordinate schema and legacy start/end fields.
    """
    motifs = []

    def _format_motif(item: dict) -> str | None:
        """Format a motif dict into a human-readable location string."""
        if not isinstance(item, dict):
            return None

        name = str(item.get("motif") or item.get("selected_motif") or "unknown")
        mapping_status = item.get("mapping_status")
        coordinate_system = item.get("coordinate_system")

        # Try new coordinate schema first
        if mapping_status == "mapped" and coordinate_system == "sequence_zero_based_half_open":
            start = item.get("sequence_start")
            end = item.get("sequence_end")
            if start is not None and end is not None:
                return f"{name}@sequence[{start},{end})"

        msa_start = item.get("msa_start")
        msa_end = item.get("msa_end")
        if msa_start is not None and msa_end is not None:
            return f"{name}@alignment[{msa_start},{msa_end})"

        # Fall back to legacy fields
        start = item.get("start") or item.get("selected_motif_start")
        end = item.get("end") or item.get("selected_motif_end")
        if start is not None and end is not None:
            return f"{name}@{start}-{end}"

        return f"{name}@unmapped"

    conservation_signal = state.get("conservation_signal", {}) or {}

    for item in conservation_signal.get("selected_motifs", []) or []:
        formatted = _format_motif(item)
        if formatted:
            motifs.append(formatted)
        elif isinstance(item, str):
            motifs.append(item)

    for item in state.get("_run_system_selected_motifs", []) or []:
        formatted = _format_motif(item)
        if formatted:
            motifs.append(formatted)
        elif isinstance(item, str):
            motifs.append(item)

    motifs = list(dict.fromkeys(motifs))

    return ", ".join(motifs[:8]) if motifs else "None"