
import re

def parse_skeptic_output(text: str) -> dict:
    result = {
        "support": None,
        "dg_best": None,
        "spread": None,
        "n": None,
        "source": None,
        "concerns": [],
        "missing_controls": [],
        "recommendation": None,
        "reason": None,
    }

    if not text:
        return result

    # ---- SUPPORT ----
    m = re.search(r"SUPPORT:\s*([A-Z]+)", text, re.IGNORECASE)
    if m:
        result["support"] = m.group(1).upper()

    # ---- ENERGY (robust: works across line breaks) ----
    energy_block = re.search(r"ENERGY:\s*(.+?)(?:\n|CONCERNS:)", text, re.DOTALL)
    if energy_block:
        e = energy_block.group(1)

        dg = re.search(r"ΔG[_ ]?best\s*=\s*([-0-9.]+)", e)
        sp = re.search(r"spread\s*=\s*([-0-9.]+)", e)
        n = re.search(r"n\s*=\s*(\d+)", e)
        src = re.search(r"source\s*=\s*(\w+)", e)

        if dg:
            result["dg_best"] = float(dg.group(1))
        if sp:
            result["spread"] = float(sp.group(1))
        if n:
            result["n"] = int(n.group(1))
        if src:
            result["source"] = src.group(1)

    # ---- CONCERNS (strictly from CONCERNS block only) ----
    concerns_block = re.search(r"CONCERNS:(.+?)MISSING_CONTROLS:", text, re.DOTALL)
    if concerns_block:
        lines = re.findall(r"\d+\.\s*(.+)", concerns_block.group(1))
        result["concerns"] = [l.strip() for l in lines if l.strip()][:3]

    # ---- MISSING CONTROLS ----
    m = re.search(r"MISSING_CONTROLS:\s*(.+)", text)
    if m:
        result["missing_controls"] = [
            x.strip() for x in m.group(1).split(",") if x.strip()
        ]

    # ---- RECOMMENDATION ----
    m = re.search(r"RECOMMENDATION:\s*([A-Z_]+)", text)
    if m:
        result["recommendation"] = m.group(1)

    # ---- REASON ----
    m = re.search(r"REASON:\s*(.+)", text)
    if m:
        result["reason"] = m.group(1).strip()

    # ---- SAFETY NORMALISATION ----
    result["concerns"] = result.get("concerns") or []
    result["missing_controls"] = result.get("missing_controls") or []

    return result
