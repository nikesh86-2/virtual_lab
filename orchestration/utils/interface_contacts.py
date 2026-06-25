from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


log = logging.getLogger("virtual_lab")


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

BASIC_RESIDUES = {"ARG", "LYS", "HIS"}

RNA_RESIDUES = {
    "A", "C", "G", "U",
    "RA", "RC", "RG", "RU",
    "ADE", "CYT", "GUA", "URA",
}


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except Exception:
        return None


def _safe_int(text: str) -> int | None:
    try:
        cleaned = "".join(ch for ch in str(text) if ch.isdigit() or ch == "-")
        if cleaned in {"", "-"}:
            return None
        return int(cleaned)
    except Exception:
        return None


def _atom_element(atom_name: str, element_field: str) -> str:
    element = (element_field or "").strip()

    if element:
        return element.upper()

    atom_name = atom_name.strip()

    if not atom_name:
        return ""

    atom_name = atom_name.lstrip("0123456789")

    return atom_name[:1].upper()


def _parse_pdb_atoms(path: str) -> list[dict]:
    """
    Minimal PDB parser sufficient for protein/RNA contact analysis.
    """
    atoms: list[dict] = []

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            x = _safe_float(line[30:38])
            y = _safe_float(line[38:46])
            z = _safe_float(line[46:54])

            if x is None or y is None or z is None:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "?"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            element = _atom_element(atom_name, line[76:78] if len(line) >= 78 else "")

            residue_id = f"{chain}:{resname}:{resseq}{icode}"

            if resname in PROTEIN_RESIDUES:
                moltype = "protein"
            elif resname in RNA_RESIDUES:
                moltype = "rna"
            else:
                moltype = "other"

            atoms.append(
                {
                    "atom_name": atom_name,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "element": element,
                    "x": x,
                    "y": y,
                    "z": z,
                    "residue_id": residue_id,
                    "moltype": moltype,
                }
            )

    return atoms


def _dist(a: dict, b: dict) -> float:
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _cell_key(atom: dict, cell_size: float) -> tuple[int, int, int]:
    return (
        int(math.floor(atom["x"] / cell_size)),
        int(math.floor(atom["y"] / cell_size)),
        int(math.floor(atom["z"] / cell_size)),
    )


def _neighbour_cells(key: tuple[int, int, int]):
    x, y, z = key

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def _compute_contact_entropy(rows: list[dict]) -> tuple[float, float]:
    """
    Shannon entropy of contact distribution over RNA residues.

    Returns:
      raw_entropy
      normalized_entropy in [0, 1]
    """
    if not rows:
        return 0.0, 0.0

    counts: Counter[str] = Counter()

    for row in rows:
        rna_id = row.get("rna_residue_id")
        weight = row.get("atom_contact_count", 1) or 1

        if rna_id:
            counts[str(rna_id)] += int(weight)

    if not counts:
        return 0.0, 0.0

    total = sum(counts.values())

    entropy = 0.0

    for value in counts.values():
        p = value / total
        entropy -= p * math.log(p + 1e-12)

    max_entropy = math.log(max(len(counts), 2))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(entropy, 4), round(normalized, 4)


def _count_rna_contact_clusters(rows: list[dict], gap_threshold: int = 3) -> int:
    """
    Count rough RNA contact clusters using residue-index gaps.

    Example:
      contacted RNA residues: 1,2,3,10,11 -> 2 clusters
    """
    positions: set[int] = set()

    for row in rows:
        pos = _safe_int(row.get("rna_resseq", ""))

        if pos is not None:
            positions.add(pos)

    if not positions:
        return 0

    ordered = sorted(positions)

    clusters = 1
    prev = ordered[0]

    for pos in ordered[1:]:
        if abs(pos - prev) > gap_threshold:
            clusters += 1
        prev = pos

    return clusters


def _cluster_quality(cluster_count: int) -> float:
    """
    Heuristic cluster quality.

    1-3 clusters are generally plausible.
    Too many clusters suggests diffuse/noisy contacts.
    """
    if cluster_count <= 0:
        return 0.0
    if cluster_count <= 3:
        return 1.0
    if cluster_count <= 5:
        return 0.65
    return 0.35


def _score_interface(
    residue_contacts: int,
    basic_contacts: int,
    min_distance: float | None,
    min_residue_contacts: int,
    min_basic_contacts: int,
    contact_entropy_normalized: float,
    rna_span_covered: float,
    cluster_count: int,
    min_rna_span: float,
) -> tuple[float, bool, bool]:
    """
    Return:
      interface_quality_score in [0, 1]
      interface_passed
      steric_clash_flag

    Heuristic only:
      - rewards broad protein/RNA interface
      - rewards basic residue involvement
      - rewards RNA span coverage
      - rewards non-overconcentrated contact distribution
      - penalises diffuse/noisy cluster count
      - strongly penalises unrealistically short atom distances
    """
    residue_component = min(residue_contacts / max(min_residue_contacts, 1), 1.0)
    basic_component = min(basic_contacts / max(min_basic_contacts, 1), 1.0)
    span_component = min(rna_span_covered / max(min_rna_span, 1e-6), 1.0)
    entropy_component = max(0.0, min(float(contact_entropy_normalized), 1.0))
    cluster_component = _cluster_quality(cluster_count)

    steric_clash = False
    severe_clash = False

    if min_distance is None:
        distance_component = 0.0

    elif min_distance < 1.2:
        distance_component = 0.0
        steric_clash = True
        severe_clash = True

    elif min_distance < 1.8:
        distance_component = 0.25
        steric_clash = True

    elif min_distance <= 3.5:
        distance_component = 1.0

    elif min_distance <= 5.0:
        distance_component = 0.75

    elif min_distance <= 7.0:
        distance_component = 0.35

    else:
        distance_component = 0.0

    score = (
        0.25 * residue_component
        + 0.18 * basic_component
        + 0.22 * distance_component
        + 0.15 * span_component
        + 0.12 * entropy_component
        + 0.08 * cluster_component
    )

    if severe_clash:
        score *= 0.20
    elif steric_clash:
        score *= 0.45

    passed = (
        residue_contacts >= min_residue_contacts
        and basic_contacts >= min_basic_contacts
        and rna_span_covered >= min_rna_span
        and not steric_clash
    )

    return round(score, 4), passed, steric_clash


def extract_protein_rna_contacts(
    complex_pdb: str,
    output_csv: str,
    output_json: str | None = None,
    cutoff: float = 5.0,
    min_residue_contacts: int = 5,
    min_basic_contacts: int = 1,
    min_rna_span: float = 0.15,
) -> dict:
    """
    Extract protein-RNA interface contacts from a docked complex PDB.

    Produces residue-pair contacts within cutoff Angstrom.

    Returns summary dict.
    """
    complex_path = Path(complex_pdb)

    if not complex_path.exists():
        return {
            "valid": False,
            "error": f"complex_pdb missing: {complex_pdb}",
            "contact_csv": None,
            "contact_json": None,
        }

    atoms = _parse_pdb_atoms(str(complex_path))

    protein_atoms = [a for a in atoms if a["moltype"] == "protein"]
    rna_atoms = [a for a in atoms if a["moltype"] == "rna"]

    if not protein_atoms or not rna_atoms:
        return {
            "valid": False,
            "error": (
                f"Could not identify protein/RNA atoms: "
                f"protein_atoms={len(protein_atoms)} rna_atoms={len(rna_atoms)}"
            ),
            "protein_atom_count": len(protein_atoms),
            "rna_atom_count": len(rna_atoms),
            "contact_csv": None,
            "contact_json": None,
        }

    rna_residue_set = {a["residue_id"] for a in rna_atoms}

    grid: dict[tuple[int, int, int], list[dict]] = defaultdict(list)

    for atom in protein_atoms:
        grid[_cell_key(atom, cutoff)].append(atom)

    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    atom_contact_count = 0
    min_distance: float | None = None

    for rna_atom in rna_atoms:
        key = _cell_key(rna_atom, cutoff)

        for neighbour_key in _neighbour_cells(key):
            for protein_atom in grid.get(neighbour_key, []):
                d = _dist(rna_atom, protein_atom)

                if d > cutoff:
                    continue

                atom_contact_count += 1

                if min_distance is None or d < min_distance:
                    min_distance = d

                pair_key = (protein_atom["residue_id"], rna_atom["residue_id"])

                if pair_key not in pair_stats:
                    pair_stats[pair_key] = {
                        "protein_chain": protein_atom["chain"],
                        "protein_resname": protein_atom["resname"],
                        "protein_resseq": protein_atom["resseq"],
                        "protein_residue_id": protein_atom["residue_id"],
                        "rna_chain": rna_atom["chain"],
                        "rna_resname": rna_atom["resname"],
                        "rna_resseq": rna_atom["resseq"],
                        "rna_residue_id": rna_atom["residue_id"],
                        "min_distance_A": d,
                        "atom_contact_count": 1,
                        "is_basic_residue": protein_atom["resname"] in BASIC_RESIDUES,
                    }
                else:
                    pair_stats[pair_key]["atom_contact_count"] += 1

                    if d < pair_stats[pair_key]["min_distance_A"]:
                        pair_stats[pair_key]["min_distance_A"] = d

    rows = sorted(
        pair_stats.values(),
        key=lambda r: (
            r["min_distance_A"],
            r["protein_residue_id"],
            r["rna_residue_id"],
        ),
    )

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "protein_chain",
        "protein_resname",
        "protein_resseq",
        "protein_residue_id",
        "rna_chain",
        "rna_resname",
        "rna_resseq",
        "rna_residue_id",
        "min_distance_A",
        "atom_contact_count",
        "is_basic_residue",
    ]

    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    protein_contact_residues = {r["protein_residue_id"] for r in rows}
    rna_contact_residues = {r["rna_residue_id"] for r in rows}

    basic_contact_residues = {
        r["protein_residue_id"]
        for r in rows
        if r["is_basic_residue"]
    }

    residue_contact_count = len(rows)
    basic_contact_count = len(basic_contact_residues)

    basic_fraction = (
        basic_contact_count / len(protein_contact_residues)
        if protein_contact_residues
        else 0.0
    )

    contact_entropy, contact_entropy_norm = _compute_contact_entropy(rows)
    cluster_count = _count_rna_contact_clusters(rows)

    rna_total_residues = len(rna_residue_set)

    if rna_total_residues > 0:
        rna_span_covered = len(rna_contact_residues) / rna_total_residues
    else:
        rna_span_covered = 0.0

    interface_score, interface_passed, steric_clash = _score_interface(
        residue_contacts=residue_contact_count,
        basic_contacts=basic_contact_count,
        min_distance=min_distance,
        min_residue_contacts=min_residue_contacts,
        min_basic_contacts=min_basic_contacts,
        contact_entropy_normalized=contact_entropy_norm,
        rna_span_covered=rna_span_covered,
        cluster_count=cluster_count,
        min_rna_span=min_rna_span,
    )

    summary = {
        "valid": True,
        "error": None,
        "complex_pdb": str(complex_path),
        "contact_csv": str(output_csv_path),
        "contact_json": str(output_json) if output_json else None,
        "cutoff_A": cutoff,
        "protein_atom_count": len(protein_atoms),
        "rna_atom_count": len(rna_atoms),
        "atom_contact_count": atom_contact_count,
        "residue_contact_count": residue_contact_count,
        "protein_contact_residue_count": len(protein_contact_residues),
        "rna_contact_residue_count": len(rna_contact_residues),
        "basic_residue_contact_count": basic_contact_count,
        "basic_contact_fraction": round(basic_fraction, 4),
        "min_distance_A": round(min_distance, 3) if min_distance is not None else None,
        "interface_quality_score": interface_score,
        "interface_passed": interface_passed,
        "interface_steric_clash": steric_clash,
        "interface_contact_entropy": contact_entropy,
        "interface_contact_entropy_normalized": contact_entropy_norm,
        "interface_rna_span_covered": round(rna_span_covered, 4),
        "interface_cluster_count": cluster_count,
        "min_required_residue_contacts": min_residue_contacts,
        "min_required_basic_contacts": min_basic_contacts,
        "min_required_rna_span": min_rna_span,
    }

    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "contacts": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return summary


def analyse_interface_contacts_for_results(state: dict) -> dict:
    """
    Add protein/RNA interface contact metrics to each docking-valid binding result.

    Controlled by:
      VLAB_ANALYSE_INTERFACE_CONTACTS=1
    """
    if os.getenv("VLAB_ANALYSE_INTERFACE_CONTACTS", "1").strip() != "1":
        return {}

    cutoff = float(os.getenv("VLAB_INTERFACE_CONTACT_CUTOFF", "5.0"))
    min_residue_contacts = int(os.getenv("VLAB_MIN_INTERFACE_RESIDUE_CONTACTS", "5"))
    min_basic_contacts = int(os.getenv("VLAB_MIN_INTERFACE_BASIC_CONTACTS", "1"))
    min_rna_span = float(os.getenv("VLAB_MIN_INTERFACE_RNA_SPAN", "0.15"))

    out_dir = Path(
        os.getenv(
            "VLAB_INTERFACE_CONTACT_DIR",
            "output_data/docking_contacts",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_results = []
    contact_files = []

    for idx, r in enumerate(state.get("binding_results", []) or []):
        if not isinstance(r, dict):
            updated_results.append(r)
            continue

        copied = dict(r)

        complex_file = (
            copied.get("dock_complex_file")
            or copied.get("complex_file")
            or copied.get("dock_complex")
        )

        if not copied.get("dock_valid") or not complex_file:
            updated_results.append(copied)
            continue

        target = copied.get("target_pdb") or state.get("target_pdb") or "target"
        rank = copied.get("rank", idx + 1)
        seq = copied.get("sequence", "")
        seq_tag = seq[:12] if seq else f"rank{rank}"

        base = out_dir / f"{target}_rank{rank}_{seq_tag}"

        csv_path = str(base.with_suffix(".contacts.csv"))
        json_path = str(base.with_suffix(".contacts.json"))

        summary = extract_protein_rna_contacts(
            complex_pdb=complex_file,
            output_csv=csv_path,
            output_json=json_path,
            cutoff=cutoff,
            min_residue_contacts=min_residue_contacts,
            min_basic_contacts=min_basic_contacts,
            min_rna_span=min_rna_span,
        )

        copied["interface_contacts_valid"] = summary.get("valid", False)
        copied["interface_contacts_error"] = summary.get("error")
        copied["interface_contact_csv"] = summary.get("contact_csv")
        copied["interface_contact_json"] = summary.get("contact_json")
        copied["interface_contact_cutoff_A"] = summary.get("cutoff_A")
        copied["interface_atom_contacts"] = summary.get("atom_contact_count")
        copied["interface_residue_contacts"] = summary.get("residue_contact_count")
        copied["interface_protein_contact_residues"] = summary.get(
            "protein_contact_residue_count"
        )
        copied["interface_rna_contact_residues"] = summary.get(
            "rna_contact_residue_count"
        )
        copied["interface_basic_residue_contacts"] = summary.get(
            "basic_residue_contact_count"
        )
        copied["interface_basic_contact_fraction"] = summary.get(
            "basic_contact_fraction"
        )
        copied["interface_min_distance_A"] = summary.get("min_distance_A")
        copied["interface_quality_score"] = summary.get("interface_quality_score")
        copied["interface_passed"] = summary.get("interface_passed")
        copied["interface_steric_clash"] = summary.get("interface_steric_clash")
        copied["interface_contact_entropy"] = summary.get("interface_contact_entropy")
        copied["interface_contact_entropy_normalized"] = summary.get(
            "interface_contact_entropy_normalized"
        )
        copied["interface_rna_span_covered"] = summary.get(
            "interface_rna_span_covered"
        )
        copied["interface_cluster_count"] = summary.get("interface_cluster_count")

        if summary.get("valid"):
            contact_files.append(csv_path)
            contact_files.append(json_path)

        updated_results.append(copied)

    log.info(
        "Analysed interface contacts for %d docking results.",
        len(contact_files) // 2,
    )

    return {
        "binding_results": updated_results,
        "interface_contact_files": contact_files,
    }