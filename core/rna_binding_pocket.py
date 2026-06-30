"""
rna_binding_pocket.py

RNA binding pocket identification from protein-RNA interface contacts.

Used by the inhibitor_agent to define the search box for small molecule
(AutoDock Vina) and peptide (HDOCK) docking into the same pocket that
the RNA occupies.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser


log = logging.getLogger("virtual_lab.rna_binding_pocket")


# ---------------------------------------------------------------------------
# Pocket extraction from interface contacts
# ---------------------------------------------------------------------------

def extract_pocket_residues_from_interface(
    interface_contacts: dict,
) -> list[dict]:
    """
    Extract protein pocket residues from RNA-protein interface contact data.

    The interface_contacts dict is produced by the protein_agent's interface
    contact analysis and has the structure:
        {
            "interface_residues": [
                {"residue_id": "A:123", "chain": "A", "residue_number": 123,
                 "contact_count": 8, "atom_count": 5, ...},
                ...
            ],
            "interface_quality_score": 0.85,
            ...
        }

    Returns:
        List of residue dicts with at least:
            chain, residue_number, residue_id, contact_count
    """
    if not isinstance(interface_contacts, dict):
        log.warning("interface_contacts is not a dict: %s", type(interface_contacts))
        return []

    residues = interface_contacts.get("interface_residues", [])

    if not residues:
        log.warning("No interface_residues found in interface_contacts")
        return []

    # Filter to residues with meaningful contact counts
    min_contacts = int(os.getenv("VLAB_MIN_INTERFACE_RESIDUE_CONTACTS", "5"))
    pocket_residues = []

    for r in residues:
        if not isinstance(r, dict):
            continue

        contact_count = r.get("contact_count", 0)
        try:
            contact_count = int(contact_count)
        except (ValueError, TypeError):
            contact_count = 0

        if contact_count >= min_contacts:
            pocket_residues.append(r)

    if not pocket_residues:
        log.warning(
            "No pocket residues passed min_contacts=%d filter; "
            "falling back to all interface residues",
            min_contacts,
        )
        pocket_residues = residues

    log.info(
        "Extracted %d pocket residues from %d interface residues",
        len(pocket_residues),
        len(residues),
    )

    return pocket_residues


# ---------------------------------------------------------------------------
# Pocket centroid computation
# ---------------------------------------------------------------------------

def compute_pocket_centroid(
    protein_pdb: str,
    residue_ids: list[str],
) -> tuple[float, float, float] | None:
    """
    Compute the geometric centroid of a set of protein residues.

    Args:
        protein_pdb: Path to the protein PDB file.
        residue_ids: List of residue identifiers like "A:123" or ["A", 123].

    Returns:
        (cx, cy, cz) centroid coordinates, or None if computation fails.
    """
    if not protein_pdb or not os.path.exists(protein_pdb):
        log.warning("Protein PDB not found: %s", protein_pdb)
        return None

    if not residue_ids:
        log.warning("No residue IDs provided for centroid computation")
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_pdb)
    except Exception as e:
        log.warning("Failed to parse protein PDB for centroid: %s", e)
        return None

    # Parse residue IDs into (chain_id, res_number) pairs
    parsed_ids = []
    for rid in residue_ids:
        if isinstance(rid, dict):
            chain = rid.get("chain", "")
            res_num = rid.get("residue_number")
            if chain and res_num is not None:
                parsed_ids.append((chain, int(res_num)))
        elif isinstance(rid, str):
            parts = rid.replace(":", " ").split()
            if len(parts) >= 2:
                parsed_ids.append((parts[0], int(parts[1])))

    if not parsed_ids:
        log.warning("Could not parse any residue IDs from: %s", residue_ids[:5])
        return None

    # Collect CA coordinates for target residues
    coords = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                res_id = residue.get_id()[1]
                hetflag = residue.get_id()[0]
                if hetflag != " ":
                    continue  # skip heteroatoms
                if (chain_id, res_id) in parsed_ids:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_coord())

    if not coords:
        log.warning(
            "No CA coordinates found for any of the %d target residues",
            len(parsed_ids),
        )
        return None

    coords_array = np.array(coords)
    centroid = coords_array.mean(axis=0)

    log.info(
        "Pocket centroid: (%.2f, %.2f, %.2f) from %d residues",
        centroid[0],
        centroid[1],
        centroid[2],
        len(coords),
    )

    return tuple(centroid)


# ---------------------------------------------------------------------------
# Pocket size computation
# ---------------------------------------------------------------------------

def compute_pocket_size(
    protein_pdb: str,
    residue_ids: list[str],
    padding: float = 8.0,
) -> tuple[float, float, float] | None:
    """
    Compute the bounding box size for a set of pocket residues.

    Args:
        protein_pdb: Path to the protein PDB file.
        residue_ids: List of residue identifiers.
        padding: Extra space (Angstrom) to add around the pocket.

    Returns:
        (sx, sy, sz) box dimensions, or None if computation fails.
    """
    if not protein_pdb or not os.path.exists(protein_pdb):
        log.warning("Protein PDB not found for pocket size: %s", protein_pdb)
        return None

    if not residue_ids:
        log.warning("No residue IDs provided for pocket size computation")
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_pdb)
    except Exception as e:
        log.warning("Failed to parse protein PDB for pocket size: %s", e)
        return None

    parsed_ids = []
    for rid in residue_ids:
        if isinstance(rid, dict):
            chain = rid.get("chain", "")
            res_num = rid.get("residue_number")
            if chain and res_num is not None:
                parsed_ids.append((chain, int(res_num)))
        elif isinstance(rid, str):
            parts = rid.replace(":", " ").split()
            if len(parts) >= 2:
                parsed_ids.append((parts[0], int(parts[1])))

    if not parsed_ids:
        return None

    all_coords = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                res_id = residue.get_id()[1]
                hetflag = residue.get_id()[0]
                if hetflag != " ":
                    continue
                if (chain_id, res_id) in parsed_ids:
                    for atom in residue:
                        all_coords.append(atom.get_coord())

    if not all_coords:
        log.warning("No coordinates found for pocket residues")
        return None

    coords_array = np.array(all_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    size = (max_coords - min_coords) + padding

    log.info(
        "Pocket size: (%.1f, %.1f, %.1f) with padding %.1f",
        size[0],
        size[1],
        size[2],
        padding,
    )

    return tuple(size)


# ---------------------------------------------------------------------------
# Combined pocket definition
# ---------------------------------------------------------------------------

def define_docking_box(
    protein_pdb: str,
    interface_contacts: dict,
    padding: float | None = None,
) -> dict:
    """
    One-stop function: extract pocket residues from interface contacts,
    compute centroid and size for a docking search box.

    Args:
        protein_pdb: Path to the protein PDB file.
        interface_contacts: Interface contact dict from protein_agent.
        padding: Box padding in Angstrom. Falls back to
                 VLAB_INHIBITOR_BOX_PADDING env var, then 8.0.

    Returns:
        {
            "pocket_residues": [...],
            "center": (cx, cy, cz) or None,
            "size": (sx, sy, sz) or None,
            "n_pocket_residues": int,
            "error": str or None,
        }
    """
    if padding is None:
        padding = float(os.getenv("VLAB_INHIBITOR_BOX_PADDING", "8.0"))

    pocket_residues = extract_pocket_residues_from_interface(interface_contacts)

    if not pocket_residues:
        return {
            "pocket_residues": [],
            "center": None,
            "size": None,
            "n_pocket_residues": 0,
            "error": "No pocket residues extracted from interface contacts",
        }

    center = compute_pocket_centroid(protein_pdb, pocket_residues)
    size = compute_pocket_size(protein_pdb, pocket_residues, padding=padding)

    if center is None or size is None:
        return {
            "pocket_residues": pocket_residues,
            "center": None,
            "size": None,
            "n_pocket_residues": len(pocket_residues),
            "error": "Failed to compute centroid or size from pocket residues",
        }

    return {
        "pocket_residues": pocket_residues,
        "center": center,
        "size": size,
        "n_pocket_residues": len(pocket_residues),
        "error": None,
    }
