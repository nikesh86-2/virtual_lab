from __future__ import annotations

import csv
import json
import re
import time

from VLAB2.orchestration.state_schema import LabState, safe_jsonable


def safe_binding_rank(r: dict):
    """
    Sort helper for canonical binding rank score.

    Lower is better.
    """
    value = r.get("binding_rank_score", r.get("dg"))

    if value is None:
        return float("inf")

    try:
        return float(value)
    except Exception:
        return float("inf")


def safe_dg(r: dict):
    """
    Legacy compatibility sorter.

    NOTE:
    In HDOCK mode, r['dg'] is a compatibility alias for binding_rank_score,
    not a physical kcal/mol free energy.
    """
    value = r.get("dg")

    if value is None:
        return float("inf")

    try:
        return float(value)
    except Exception:
        return float("inf")


def parse_pdb_candidates(text: str) -> list[str]:
    """
    Extract PDB IDs from noisy LLM output.
    """
    candidates = []
    blacklist = {"NONE", "NULL", "N/A"}

    if not text:
        return []

    for line in str(text).splitlines():
        line = line.strip().upper()
        matches = re.findall(r"\b[0-9][A-Z0-9]{3}\b", line)

        for m in matches:
            if m not in blacklist:
                candidates.append(m)

    return list(dict.fromkeys(candidates))


def _summary_jsonable(obj):
    """
    Conservative JSON converter for docking summary exports.

    Unlike state_schema.safe_jsonable, this should never collapse a valid row
    into None simply because one field is unusual.
    """
    from pathlib import Path

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _summary_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_summary_jsonable(v) for v in obj]

    if hasattr(obj, "model_dump"):
        try:
            return _summary_jsonable(obj.model_dump())
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return _summary_jsonable(obj.dict())
        except Exception:
            pass

    return repr(obj)

def export_docking_outputs(state: dict) -> dict:
    """
    Export docking summary JSON/CSV/Markdown files.

    Returns paths:
      docking_summary_json
      docking_summary_csv
      docking_summary_md
    """
    import csv
    import json
    import logging
    import time
    from pathlib import Path

    log = logging.getLogger("virtual_lab")

    binding_results = state.get("binding_results", []) or []

    # Create the docking_summaries directory if it doesn't exist
    output_dir = Path("/scratch/fbsnpat/bot/VLAB2/docking_summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = int(time.time())
    json_path = output_dir / f"docking_summary_{stamp}.json"
    csv_path = output_dir / f"docking_summary_{stamp}.csv"
    md_path = output_dir / f"docking_summary_{stamp}.md"

    rows = []

    for r in binding_results:
        if not isinstance(r, dict):
            continue

        linked_md = r.get("linked_md") or {}

        if not isinstance(linked_md, dict):
            linked_md = {}

        row = {
            "rank": r.get("rank"),
            "sequence": r.get("sequence"),
            "target_pdb": r.get("target_pdb") or state.get("target_pdb"),
            "valid": r.get("valid"),
            "binding_mode": r.get("binding_mode"),
            "binding_units": r.get("binding_units") or state.get("binding_units"),
            "binding_rank_score": r.get("binding_rank_score", r.get("dg")),
            "dg": r.get("dg"),
            "proxy_dg": r.get("proxy_dg"),
            "dock_score": r.get("dock_score", r.get("hdock_score")),
            "hdock_score": r.get("hdock_score", r.get("dock_score")),
            "dock_valid": r.get("dock_valid"),
            "dock_method": r.get("dock_method"),
            "dock_error": r.get("dock_error"),
            "dock_output_file": r.get("dock_output_file"),
            "dock_complex_file": r.get("dock_complex_file"),
            "vina_energy": r.get("vina_energy"),
            "vina_valid": r.get("vina_valid"),
            "vina_method": r.get("vina_method"),
            "vina_error": r.get("vina_error"),
            "md_min_energy": linked_md.get("min_energy"),
            "md_mean_energy": linked_md.get("mean_energy"),
            "md_energy_fluctuation": linked_md.get("energy_fluctuation"),
            "rna_pdb": linked_md.get("rna_pdb"),

            # Visual snapshot output.
            "docking_snapshot_png": r.get("docking_snapshot_png"),
            "docking_snapshot_valid": r.get("docking_snapshot_valid"),
            "docking_snapshot_error": r.get("docking_snapshot_error"),

            # Interface contact metrics.
            "interface_contacts_valid": r.get("interface_contacts_valid"),
            "interface_contacts_error": r.get("interface_contacts_error"),
            "interface_contact_csv": r.get("interface_contact_csv"),
            "interface_contact_json": r.get("interface_contact_json"),
            "interface_contact_cutoff_A": r.get("interface_contact_cutoff_A"),
            "interface_atom_contacts": r.get("interface_atom_contacts"),
            "interface_residue_contacts": r.get("interface_residue_contacts"),
            "interface_protein_contact_residues": r.get(
                "interface_protein_contact_residues"
            ),
            "interface_rna_contact_residues": r.get(
                "interface_rna_contact_residues"
            ),
            "interface_basic_residue_contacts": r.get(
                "interface_basic_residue_contacts"
            ),
            "interface_basic_contact_fraction": r.get(
                "interface_basic_contact_fraction"
            ),
            "interface_min_distance_A": r.get("interface_min_distance_A"),
            "interface_quality_score": r.get("interface_quality_score"),
            "interface_passed": r.get("interface_passed"),
        }

        rows.append(row)

    safe_rows = _summary_jsonable(rows)

    if safe_rows is None:
        safe_rows = []

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_rows, f, indent=2)
        f.write("\n")

    # CSV
    fieldnames = [
        "rank",
        "sequence",
        "target_pdb",
        "valid",
        "binding_mode",
        "binding_units",
        "binding_rank_score",
        "dg",
        "proxy_dg",
        "dock_score",
        "hdock_score",
        "dock_valid",
        "dock_method",
        "dock_error",
        "dock_output_file",
        "dock_complex_file",
        "vina_energy",
        "vina_valid",
        "vina_method",
        "vina_error",
        "md_min_energy",
        "md_mean_energy",
        "md_energy_fluctuation",
        "rna_pdb",

        # Visuals.
        "docking_snapshot_png",
        "docking_snapshot_valid",
        "docking_snapshot_error",

        # Interface metrics.
        "interface_contacts_valid",
        "interface_contacts_error",
        "interface_contact_csv",
        "interface_contact_json",
        "interface_contact_cutoff_A",
        "interface_atom_contacts",
        "interface_residue_contacts",
        "interface_protein_contact_residues",
        "interface_rna_contact_residues",
        "interface_basic_residue_contacts",
        "interface_basic_contact_fraction",
        "interface_min_distance_A",
        "interface_quality_score",
        "interface_passed",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    # Markdown
    lines = [
        "# Docking Summary",
        "",
        f"Target PDB: `{state.get('target_pdb')}`",
        f"Binding units: `{state.get('binding_units', 'hdock_relative_score')}`",
        "",
        "| Rank | Sequence | Target | Dock score | Rank score | Valid | Mode | Method | Interface contacts | Basic contacts | Min dist Å | Interface score | Interface passed | Snapshot | Contacts CSV |",
        "|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]

    for row in rows:
        seq = row.get("sequence") or ""
        seq_short = seq[:20] + "..." if len(seq) > 20 else seq

        snapshot = row.get("docking_snapshot_png") or ""
        contacts_csv = row.get("interface_contact_csv") or ""

        # Keep Markdown cells compact but useful.
        snapshot_cell = f"[PNG]({snapshot})" if snapshot else ""
        contacts_cell = f"[CSV]({contacts_csv})" if contacts_csv else ""

        lines.append(
            "| {rank} | `{seq}` | {target} | {dock_score} | {rank_score} | "
            "{valid} | {mode} | {method} | {contacts} | {basic} | {min_dist} | "
            "{iface_score} | {iface_passed} | {snapshot} | {contacts_csv} |".format(
                rank=row.get("rank"),
                seq=seq_short,
                target=row.get("target_pdb"),
                dock_score=row.get("dock_score"),
                rank_score=row.get("binding_rank_score"),
                valid=row.get("dock_valid"),
                mode=row.get("binding_mode"),
                method=row.get("dock_method"),
                contacts=row.get("interface_residue_contacts"),
                basic=row.get("interface_basic_residue_contacts"),
                min_dist=row.get("interface_min_distance_A"),
                iface_score=row.get("interface_quality_score"),
                iface_passed=row.get("interface_passed"),
                snapshot=snapshot_cell,
                contacts_csv=contacts_cell,
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log.info(
        "Docking summary exports written: json=%s csv=%s md=%s rows=%d",
        json_path,
        csv_path,
        md_path,
        len(rows),
    )

    return {
        "docking_summary_json": str(json_path),
        "docking_summary_csv": str(csv_path),
        "docking_summary_md": str(md_path),
    }