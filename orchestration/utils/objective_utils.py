from __future__ import annotations


def adjust_weights(parsed, weights):
    """
    Adaptive objective-weight update from Skeptic output.

    HDOCK mode:
      parsed['dg_best'] is treated as a relative docking/rank score,
      not kcal/mol.
    """
    new = dict(weights or {})

    for k in ["binding", "kinetic", "thermo", "structure"]:
        new[k] = new.get(k, 1.0)

    dg_best = parsed.get("dg_best") if isinstance(parsed, dict) else None

    if dg_best is not None:
        try:
            dg_best = float(dg_best)

            # HDOCK/rank-score heuristic:
            # less negative / weakly favourable scores increase binding pressure.
            if dg_best > -50:
                new["binding"] *= 1.3

        except Exception:
            pass

    missing_controls = (
        parsed.get("missing_controls", [])
        if isinstance(parsed, dict)
        else []
    )

    if isinstance(missing_controls, str):
        missing_controls_text = missing_controls.lower()
    else:
        missing_controls_text = " ".join(map(str, missing_controls)).lower()

    if "md trajectory" in missing_controls_text or "md" in missing_controls_text:
        new["kinetic"] *= 1.2

    total = sum(new.values()) or 1.0

    return {k: v / total for k, v in new.items()}
