"""
inhibitor_docking.py

Main docking execution module for the VLAB2 inhibitor screening pipeline.

Coordinates:
1. Small-molecule docking via AutoDock Vina
2. Peptide docking via HDOCK protein-protein mode
3. True inhibitor-vs-RNA pocket overlap comparison

Design decisions:
- Vina for small molecules.
- HDOCK for peptides.
- Direct PDBQT heavy-atom parsing for Vina poses.
- Atom-level inhibitor-vs-RNA interface overlap after receptor alignment.
- Legacy CA-RMSD proxy retained only as fallback.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from Bio import PDB
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False

from VLAB2.core.vina_wrapper import VinaDocking
from VLAB2.core.hdock_wrapper import HDockDocking


log = logging.getLogger("virtual_lab.inhibitor_docking")


# ── Env defaults ──────────────────────────────────────────────────────────────

VINA_TIMEOUT = int(os.getenv("VINA_TIMEOUT", "600"))
VINA_EXHAUSTIVENESS = int(os.getenv("VLAB_VINA_EXHAUSTIVENESS", "8"))
HDOCK_TIMEOUT = int(os.getenv("HDOCK_TIMEOUT", "1200"))
HDOCK_N_MODELS = int(os.getenv("HDOCK_N_MODELS", "10"))
REQUIRE_VINA = os.getenv("VLAB_REQUIRE_VINA", "0").lower() in {"1", "true", "yes"}

DEFAULT_CACHE_DIR = os.getenv(
    "VLAB_INHIBITOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "vlab_inhibitor_cache"),
)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# SMALL MOLECULE DOCKING
# ─────────────────────────────────────────────────────────────────────────────

def dock_small_molecules(
    receptor_pdbqt: str,
    ligands_dir: str,
    center: tuple,
    size: tuple,
    exhaustiveness: int = VINA_EXHAUSTIVENESS,
    vina_path: Optional[str] = None,
    max_ligands: int | None = None,
) -> list:
    """
    Dock prepared small-molecule PDBQT files in ligands_dir against receptor.

    Defensive behaviour:
      - skips receptor files
      - skips already docked outputs
      - caps ligand count using max_ligands or VLAB_INHIBITOR_MAX_SMALL_MOLECULES

    Returns:
        List of docking result dictionaries.
    """
    vina = VinaDocking(vina_path=vina_path)
    results: list[dict] = []

    ligands_path = Path(ligands_dir)
    if not ligands_path.is_dir():
        log.error("Ligands directory not found: %s", ligands_path)
        return []

    try:
        receptor_resolved = Path(receptor_pdbqt).resolve()
    except Exception:
        receptor_resolved = None

    ligand_files: list[Path] = []

    for p in sorted(ligands_path.glob("*.pdbqt")):
        try:
            if receptor_resolved is not None and p.resolve() == receptor_resolved:
                continue
        except OSError:
            pass

        name = p.name.lower()

        if name.endswith("_receptor.pdbqt"):
            continue
        if name.endswith("_docked.pdbqt"):
            continue
        if "_docked" in name:
            continue

        ligand_files.append(p)

    if not ligand_files:
        log.warning("No PDBQT files found in %s", ligands_path)
        return []

    if max_ligands is None:
        max_ligands = _env_int("VLAB_INHIBITOR_MAX_SMALL_MOLECULES", 10)

    if max_ligands and max_ligands > 0 and len(ligand_files) > max_ligands:
        log.warning(
            "Capping small-molecule docking inputs from %d to %d files in %s",
            len(ligand_files),
            max_ligands,
            ligands_path,
        )
        ligand_files = ligand_files[:max_ligands]

    log.info(
        "[INHIBITOR DOCK] Docking %d small molecules with Vina (exhaustiveness=%d)",
        len(ligand_files),
        exhaustiveness,
    )

    for ligand_path in ligand_files:
        log.info("[VINA LIGAND] %s", ligand_path.name)

        result = vina.dock(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=str(ligand_path),
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
        )

        method = result.get("method", "vina")
        binding_energy = result.get("binding_energy")

        row = {
            "name": ligand_path.stem,
            "ligand_name": ligand_path.stem,
            "_inhibitor_name": ligand_path.stem,
            "ligand_type": "small_molecule",
            "ligand_path": str(ligand_path),

            "binding_energy": binding_energy,
            "score": binding_energy,
            "binding_units": "vina_kcal_mol",
            "binding_energy_is_physical": True,

            "method": method,
            "dock_method": method,
            "valid": result.get("valid", False),

            "output_file": result.get("output_file"),
            "docked_pdbqt": result.get("output_file"),
            "error": result.get("error"),
        }

        results.append(row)

        if result.get("valid") and binding_energy is not None:
            log.info("  → %s: %.3f kcal/mol", ligand_path.stem, binding_energy)
        else:
            log.warning("  → %s: FAILED (%s)", ligand_path.stem, result.get("error"))

    return results


def rank_small_molecules(results: list) -> list:
    """
    Sort small-molecule docking results by Vina energy ascending.
    """
    valid = [
        r for r in results
        if isinstance(r, dict)
        and r.get("valid")
        and r.get("binding_energy") is not None
    ]
    invalid = [r for r in results if r not in valid]

    valid_sorted = sorted(valid, key=lambda r: r["binding_energy"])
    return valid_sorted + invalid


# ─────────────────────────────────────────────────────────────────────────────
# PEPTIDE DOCKING
# ─────────────────────────────────────────────────────────────────────────────

def _is_clean_input_peptide_pdb(path: Path) -> bool:
    """
    Identify clean peptide input PDB files while excluding generated complexes.

    Accepts:
      clean_peptide_<hash>_<sequence>.pdb
      clean_peptide_selected_<idx>_<sequence>.pdb

    Excludes:
      *_hdock_complex*
      *_docked*
      *_complex*
    """
    if not path.is_file():
        return False

    name = path.name

    lowered = name.lower()
    if "_hdock_complex" in lowered:
        return False
    if "_docked" in lowered:
        return False
    if "_complex" in lowered:
        return False

    if not name.startswith("clean_peptide_") or not name.endswith(".pdb"):
        return False

    return bool(
        re.fullmatch(
            r"clean_peptide_(?:[0-9A-Fa-f]+|selected_\d+)_[A-Za-z]+\.pdb",
            name,
        )
    )


def dock_peptides(
    receptor_pdb: str,
    peptide_pdbs_dir: str,
    n_models: int = HDOCK_N_MODELS,
    hdock_path: Optional[str] = None,
    createpl_path: Optional[str] = None,
) -> list:
    """
    Dock prepared peptide PDB files against the receptor protein with HDOCK.
    """
    hdock = HDockDocking(
        hdock_path=hdock_path,
        createpl_path=createpl_path,
    )

    results: list[dict] = []
    pdbs_dir = Path(peptide_pdbs_dir)

    if not pdbs_dir.is_dir():
        log.error("Peptide PDB directory not found: %s", pdbs_dir)
        return []

    peptide_files = [
        p for p in sorted(pdbs_dir.glob("clean_peptide_*.pdb"))
        if _is_clean_input_peptide_pdb(p)
    ]

    if not peptide_files:
        log.warning("No peptide PDB files found in %s", pdbs_dir)
        return []

    log.info(
        "[INHIBITOR DOCK] Docking %d peptides with HDOCK (n_models=%d)",
        len(peptide_files),
        n_models,
    )

    for peptide_pdb in peptide_files:
        log.info("[HDOCK PEPTIDE] %s", peptide_pdb.name)

        result = hdock.dock(
            receptor_pdb=receptor_pdb,
            ligand_pdb=str(peptide_pdb),
            n_models=n_models,
        )

        hdock_score = _extract_hdock_score(result)

        complex_pdb = (
            result.get("complex_pdb")
            or result.get("output_complex")
            or result.get("complex_file")
            or result.get("dock_complex_file")
        )

        seq = _extract_sequence_from_filename(peptide_pdb.name)
        method = result.get("method", "hdock")

        row = {
            "sequence": seq,
            "peptide_sequence": seq,
            "ligand_type": "peptide",
            "peptide_pdb": str(peptide_pdb),

            "score": hdock_score,
            "dock_score": hdock_score,
            "hdock_score": hdock_score,

            "binding_units": "hdock_relative_score",
            "binding_energy_is_physical": False,

            "method": method,
            "dock_method": method,
            "valid": result.get("valid", False),
            "dock_valid": result.get("valid", False),

            "complex_pdb": complex_pdb,
            "complex_file": complex_pdb,
            "dock_complex_file": complex_pdb,

            "output_file": result.get("output_file"),
            "dock_output_file": result.get("dock_output_file") or result.get("output_file"),
            "error": result.get("error"),
        }

        results.append(row)

        if hdock_score is not None:
            log.info(
                "  → %s: HDOCK relative score = %.2f",
                peptide_pdb.name,
                hdock_score,
            )
        else:
            log.warning("  → %s: FAILED (%s)", peptide_pdb.name, result.get("error"))

    return results


def _extract_hdock_score(result: dict) -> Optional:

    """
    Extract best HDOCK score from a docking result dict.
    """
    if not isinstance(result, dict):
        return None

    if not result.get("valid"):
        return None

    for key in ("best_score", "score", "hdock_score", "dock_score"):
        if key not in result:
            continue

        score = result.get(key)

        if score is None:
            continue

        try:
            return float(score)
        except (TypeError, ValueError):
            continue

    return None


def _extract_sequence_from_filename(filename: str) -> str:
    """
    Extract peptide sequence from clean_peptide_<tag>_<seq>.pdb.
    """
    name = Path(filename).stem
    parts = name.split("_")

    if len(parts) >= 4 and parts[0] == "clean" and parts[1] == "peptide":
        return parts[-1]

    if len(parts) >= 3:
        return parts[-1]

    return name


def rank_peptides(results: list) -> list:
    """
    Sort peptide docking results by HDOCK score ascending.
    """
    valid = [
        r for r in results
        if isinstance(r, dict)
        and r.get("valid")
        and r.get("hdock_score") is not None
    ]
    invalid = [r for r in results if r not in valid]

    valid_sorted = sorted(valid, key=lambda r: r["hdock_score"])
    return valid_sorted + invalid


# ─────────────────────────────────────────────────────────────────────────────
# PATH COERCION
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_pose_path(item: Any) -> str | None:
    """
    Convert pose item into a filesystem path.

    Supports:
      - path strings
      - pathlib paths
      - dict rows from binding_results
      - {"path": "..."} wrappers
    """
    if isinstance(item, (str, os.PathLike)):
        return str(item)

    if isinstance(item, dict):
        path = (
            item.get("path")
            or item.get("dock_complex_file")
            or item.get("complex_file")
            or item.get("complex_pdb")
            or item.get("dock_complex_pdb")
            or item.get("output_complex")
        )
        return str(path) if path else None

    return None


def _coerce_existing_pose_paths(items: list | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in items or []:
        path = _coerce_pose_path(item)

        if not path:
            continue

        if path in seen:
            continue

        if os.path.exists(path):
            seen.add(path)
            out.append(path)
        else:
            log.debug("Skipping missing pose path: %s", path)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ATOM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
}

_NUCLEIC = {
    "A", "U", "G", "C", "I",
    "DA", "DT", "DG", "DC", "DI",
    "ADE", "URA", "GUA", "CYT",
}


def _atom_is_hydrogen(atom) -> bool:
    try:
        element = (atom.element or "").strip().upper()
        if element == "H":
            return True
    except Exception:
        pass

    try:
        name = atom.get_name().strip().upper()
        return name.startswith("H")
    except Exception:
        return False


def _is_protein_residue(residue) -> bool:
    try:
        return residue.get_resname().strip().upper() in _STANDARD_AA
    except Exception:
        return False


def _is_rna_or_dna_residue(residue) -> bool:
    try:
        return residue.get_resname().strip().upper() in _NUCLEIC
    except Exception:
        return False


def _coords_from_residue_filter(structure, residue_filter) -> np.ndarray:
    coords: list[np.ndarray] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not residue_filter(residue):
                    continue

                for atom in residue:
                    if _atom_is_hydrogen(atom):
                        continue

                    coords.append(atom.get_coord())

    if not coords:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(coords, dtype=float)


def _get_protein_ca_coords(structure) -> np.ndarray:
    coords: list[np.ndarray] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not _is_protein_residue(residue):
                    continue

                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())

    if not coords:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(coords, dtype=float)


def _pairwise_min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    For every point in a, compute minimum distance to any point in b.
    """
    if a.size == 0 or b.size == 0:
        return np.asarray([], dtype=float)

    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    return np.sqrt(np.min(d2, axis=1))


def _superpose_mobile_to_reference(
    mobile_coords: np.ndarray,
    mobile_ref_ca: np.ndarray,
    reference_ref_ca: np.ndarray,
) -> np.ndarray:
    """
    Superpose mobile_coords by aligning mobile_ref_ca onto reference_ref_ca.
    """
    if mobile_coords.size == 0:
        return mobile_coords

    if mobile_ref_ca.size == 0 or reference_ref_ca.size == 0:
        return mobile_coords

    n = min(len(mobile_ref_ca), len(reference_ref_ca))

    if n < 3:
        return mobile_coords

    try:
        from Bio.SVDSuperimposer import SVDSuperimposer

        mobile_ref = mobile_ref_ca[:n]
        reference_ref = reference_ref_ca[:n]

        sup = SVDSuperimposer()
        sup.set(reference_ref, mobile_ref)
        sup.run()

        rot, tran = sup.get_rotran()

        return np.dot(mobile_coords, rot) + tran

    except Exception as e:
        log.debug("Superposition failed; returning unaligned coords: %s", e)
        return mobile_coords


# ─────────────────────────────────────────────────────────────────────────────
# PDBQT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_pdbqt_heavy_atom_coords(path: str) -> np.ndarray:
    """
    Parse heavy-atom coordinates from a Vina/AutoDock PDBQT file.

    Coordinates use PDB-style columns:
      x: 30:38
      y: 38:46
      z: 46:54

    Hydrogens excluded by atom name and AutoDock atom type heuristics.
    """
    coords: list[tuple[float, float, float]] = []

    if not path or not os.path.exists(path):
        return np.zeros((0, 3), dtype=float)

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            try:
                atom_name = line[12:16].strip().upper()

                # AutoDock type is usually at the end.
                autodock_type = ""
                if len(line) > 77:
                    tail = line[77:].strip().split()
                    if tail:
                        autodock_type = tail[0].upper()

                if atom_name.startswith("H"):
                    continue
                if autodock_type.startswith("H"):
                    continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                coords.append((x, y, z))

            except Exception:
                continue

    if not coords:
        return np.zeros((0, 3), dtype=float)

    return np.asarray(coords, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# RNA INTERFACE / LIGAND EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rna_interface_from_complex(
    rna_complex_pdb: str,
    interface_cutoff_A: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract receptor protein CA coords and RNA interface atoms.

    RNA interface atoms are nucleic-acid heavy atoms within interface_cutoff_A
    of protein heavy atoms.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("rna_complex", rna_complex_pdb)

    protein_ca = _get_protein_ca_coords(structure)
    protein_atoms = _coords_from_residue_filter(structure, _is_protein_residue)
    nucleic_atoms = _coords_from_residue_filter(structure, _is_rna_or_dna_residue)

    if protein_atoms.size == 0 or nucleic_atoms.size == 0:
        return protein_ca, np.zeros((0, 3), dtype=float)

    min_d = _pairwise_min_distances(nucleic_atoms, protein_atoms)
    interface_atoms = nucleic_atoms[min_d <= interface_cutoff_A]

    return protein_ca, interface_atoms


def _extract_peptide_ligand_coords_from_complex(
    peptide_complex_pdb: str,
    receptor_chain_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract receptor CA coords and peptide ligand heavy atoms.

    Default convention:
      receptor chain = A
      peptide ligand = protein chain(s) not A

    If your HDOCK output uses other receptor chains, pass receptor_chain_ids.
    """
    if receptor_chain_ids is None:
        receptor_chain_ids = {"A"}

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("peptide_complex", peptide_complex_pdb)

    receptor_ca: list[np.ndarray] = []
    peptide_atoms: list[np.ndarray] = []

    for model in structure:
        for chain in model:
            chain_id = chain.id

            for residue in chain:
                if not _is_protein_residue(residue):
                    continue

                if chain_id in receptor_chain_ids:
                    if "CA" in residue:
                        receptor_ca.append(residue["CA"].get_coord())
                else:
                    for atom in residue:
                        if not _atom_is_hydrogen(atom):
                            peptide_atoms.append(atom.get_coord())

    return (
        np.asarray(receptor_ca, dtype=float) if receptor_ca else np.zeros((0, 3), dtype=float),
        np.asarray(peptide_atoms, dtype=float) if peptide_atoms else np.zeros((0, 3), dtype=float),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRUE ATOM-LEVEL OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def _compute_atom_overlap_metrics(
    inhibitor_coords: np.ndarray,
    rna_interface_coords: np.ndarray,
    overlap_cutoff_A: float = 4.0,
) -> dict:
    """
    Compute symmetric atom-level overlap between inhibitor atoms and RNA interface atoms.
    """
    if inhibitor_coords.size == 0:
        return {
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "inhibitor_overlap_fraction": 0.0,
            "rna_interface_coverage_fraction": 0.0,
            "min_distance_A": None,
            "n_inhibitor_atoms": 0,
            "n_rna_interface_atoms": int(len(rna_interface_coords)),
            "n_inhibitor_atoms_near_rna": 0,
            "n_rna_interface_atoms_near_inhibitor": 0,
            "overlap_cutoff_A": overlap_cutoff_A,
            "comparable": False,
            "error": "No inhibitor atoms",
        }

    if rna_interface_coords.size == 0:
        return {
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "inhibitor_overlap_fraction": 0.0,
            "rna_interface_coverage_fraction": 0.0,
            "min_distance_A": None,
            "n_inhibitor_atoms": int(len(inhibitor_coords)),
            "n_rna_interface_atoms": 0,
            "n_inhibitor_atoms_near_rna": 0,
            "n_rna_interface_atoms_near_inhibitor": 0,
            "overlap_cutoff_A": overlap_cutoff_A,
            "comparable": False,
            "error": "No RNA interface atoms",
        }

    d_inhib_to_rna = _pairwise_min_distances(inhibitor_coords, rna_interface_coords)
    d_rna_to_inhib = _pairwise_min_distances(rna_interface_coords, inhibitor_coords)

    n_inhib_near = int(np.sum(d_inhib_to_rna <= overlap_cutoff_A))
    n_rna_near = int(np.sum(d_rna_to_inhib <= overlap_cutoff_A))

    inhibitor_fraction = n_inhib_near / max(1, len(inhibitor_coords))
    rna_fraction = n_rna_near / max(1, len(rna_interface_coords))
    overlap_score = 0.5 * inhibitor_fraction + 0.5 * rna_fraction

    return {
        "overlap_score": float(overlap_score),
        "overlap_percent": float(overlap_score * 100.0),
        "inhibitor_overlap_fraction": float(inhibitor_fraction),
        "rna_interface_coverage_fraction": float(rna_fraction),
        "min_distance_A": float(np.min(d_inhib_to_rna)) if len(d_inhib_to_rna) else None,
        "n_inhibitor_atoms": int(len(inhibitor_coords)),
        "n_rna_interface_atoms": int(len(rna_interface_coords)),
        "n_inhibitor_atoms_near_rna": n_inhib_near,
        "n_rna_interface_atoms_near_inhibitor": n_rna_near,
        "overlap_cutoff_A": overlap_cutoff_A,
        "comparable": True,
        "error": None,
    }


def compare_inhibitor_with_rna_interface_overlap(
    inhibitor_pose_file: str,
    rna_poses: list,
    ligand_type: str,
    receptor_chain_ids: set[str] | None = None,
    interface_cutoff_A: float = 5.0,
    overlap_cutoff_A: float = 4.0,
) -> dict:
    """
    True atom-level inhibitor/RNA pocket overlap.

    Supports:
      ligand_type="peptide"
        inhibitor_pose_file is protein-peptide HDOCK complex PDB.
        Receptor CA atoms are aligned to each RNA complex receptor, then peptide
        ligand atoms are compared against RNA interface atoms.

      ligand_type="small_molecule"
        inhibitor_pose_file is Vina PDBQT pose.
        Assumes Vina receptor and RNA receptor share the same coordinate frame.
    """
    if not _HAS_BIOPYTHON:
        return {
            "method": "atom_interface_overlap",
            "ligand_type": ligand_type,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "comparable": False,
            "error": "Bio.PDB not available",
        }

    if not inhibitor_pose_file or not os.path.exists(inhibitor_pose_file):
        return {
            "method": "atom_interface_overlap",
            "ligand_type": ligand_type,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "comparable": False,
            "error": f"Inhibitor pose file not found: {inhibitor_pose_file}",
        }

    rna_pose_paths = _coerce_existing_pose_paths(rna_poses)

    if not rna_pose_paths:
        return {
            "method": "atom_interface_overlap",
            "ligand_type": ligand_type,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "comparable": False,
            "error": "No RNA pose paths provided",
            "n_comparable_poses": 0,
        }

    best: dict | None = None
    all_results: list[dict] = []

    for rna_pose in rna_pose_paths:
        try:
            rna_receptor_ca, rna_interface_atoms = _extract_rna_interface_from_complex(
                rna_pose,
                interface_cutoff_A=interface_cutoff_A,
            )

            if ligand_type == "peptide":
                inhib_receptor_ca, inhibitor_atoms = _extract_peptide_ligand_coords_from_complex(
                    inhibitor_pose_file,
                    receptor_chain_ids=receptor_chain_ids,
                )

                inhibitor_atoms = _superpose_mobile_to_reference(
                    mobile_coords=inhibitor_atoms,
                    mobile_ref_ca=inhib_receptor_ca,
                    reference_ref_ca=rna_receptor_ca,
                )

            elif ligand_type == "small_molecule":
                inhibitor_atoms = _parse_pdbqt_heavy_atom_coords(inhibitor_pose_file)

            else:
                raise ValueError(f"Unsupported ligand_type: {ligand_type}")

            metrics = _compute_atom_overlap_metrics(
                inhibitor_coords=inhibitor_atoms,
                rna_interface_coords=rna_interface_atoms,
                overlap_cutoff_A=overlap_cutoff_A,
            )

            metrics.update(
                {
                    "method": "atom_interface_overlap",
                    "ligand_type": ligand_type,
                    "rna_pose": rna_pose,
                    "inhibitor_pose_file": inhibitor_pose_file,
                    "interface_cutoff_A": interface_cutoff_A,
                }
            )

            all_results.append(metrics)

            if best is None or metrics.get("overlap_score", 0.0) > best.get("overlap_score", 0.0):
                best = metrics

        except Exception as e:
            log.debug(
                "Atom interface overlap failed for inhibitor=%s rna_pose=%s: %s",
                inhibitor_pose_file,
                rna_pose,
                e,
            )
            continue

    if best is None:
        return {
            "method": "atom_interface_overlap",
            "ligand_type": ligand_type,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "comparable": False,
            "error": "No comparable inhibitor/RNA interface pose pairs",
            "n_comparable_poses": 0,
        }

    best["n_comparable_poses"] = len(all_results)
    best["all_pose_overlap_scores"] = [
        {
            "rna_pose": r.get("rna_pose"),
            "overlap_score": r.get("overlap_score"),
            "overlap_percent": r.get("overlap_percent"),
            "min_distance_A": r.get("min_distance_A"),
            "comparable": r.get("comparable"),
            "error": r.get("error"),
        }
        for r in all_results
    ]

    return best


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY CA-RMSD FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def compare_inhibitor_with_rna_pose(
    inhibitor_complex_pdb: str,
    rna_poses: list,
    cutoff_rms: float = 5.0,
) -> dict:
    """
    Legacy coarse inhibitor/RNA comparison.

    Uses CA-atom RMSD between complexes after CA-based superposition.
    This is retained only as fallback; atom-interface overlap is preferred.
    """
    if not _HAS_BIOPYTHON:
        return {
            "method": "legacy_ca_rmsd_proxy",
            "min_rmsd": None,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": "Bio.PDB not available",
        }

    rna_pose_paths = _coerce_existing_pose_paths(rna_poses)

    if not rna_pose_paths:
        return {
            "method": "legacy_ca_rmsd_proxy",
            "min_rmsd": None,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": "No RNA poses provided",
        }

    if not inhibitor_complex_pdb or not os.path.exists(inhibitor_complex_pdb):
        return {
            "method": "legacy_ca_rmsd_proxy",
            "min_rmsd": None,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": f"Inhibitor complex PDB not found: {inhibitor_complex_pdb}",
        }

    try:
        parser = PDB.PDBParser(QUIET=True)
        inhibitor_structure = parser.get_structure("inhibitor", inhibitor_complex_pdb)

        rmsd_values: list[float] = []

        for rna_pose in rna_pose_paths:
            try:
                rna_structure = parser.get_structure("rna", rna_pose)
                rmsd = _compute_ca_rmsd(inhibitor_structure, rna_structure)
                if rmsd is not None:
                    rmsd_values.append(rmsd)
            except Exception as e:
                log.debug("Could not compare with RNA pose %s: %s", rna_pose, e)
                continue

        if not rmsd_values:
            return {
                "method": "legacy_ca_rmsd_proxy",
                "min_rmsd": None,
                "overlap_score": 0.0,
                "overlap_percent": 0.0,
                "n_comparable_poses": 0,
                "comparable": False,
                "error": "Could not compute RMSD for any pose pair",
            }

        min_rmsd = min(rmsd_values)
        overlap_score = max(0.0, 1.0 - min_rmsd / cutoff_rms)

        return {
            "method": "legacy_ca_rmsd_proxy",
            "min_rmsd": float(min_rmsd),
            "overlap_score": float(overlap_score),
            "overlap_percent": float(overlap_score * 100.0),
            "n_comparable_poses": len(rmsd_values),
            "comparable": True,
            "error": None,
        }

    except Exception as e:
        log.exception("Pose comparison failed")
        return {
            "method": "legacy_ca_rmsd_proxy",
            "min_rmsd": None,
            "overlap_score": 0.0,
            "overlap_percent": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": str(e),
        }


def _compute_ca_rmsd(structure_a, structure_b) -> Optional:
    """
    Compute CA-atom RMSD after superposition.

    structure_b is mobile and is aligned onto structure_a.
    """
    try:
        from Bio.SVDSuperimposer import SVDSuperimposer

        ca_a = _get_ca_atoms(structure_a)
        ca_b = _get_ca_atoms(structure_b)

        if len(ca_a) < 3 or len(ca_b) < 3:
            return None

        n = min(len(ca_a), len(ca_b))
        coords_a = np.asarray([a.get_coord() for a in ca_a[:n]], dtype=float)
        coords_b = np.asarray([b.get_coord() for b in ca_b[:n]], dtype=float)

        sup = SVDSuperimposer()
        sup.set(coords_a, coords_b)
        sup.run()

        return float(sup.get_rms())

    except Exception:
        return None


def _get_ca_atoms(structure) -> list:
    """
    Extract all protein CA atoms from a structure.
    """
    atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not _is_protein_residue(residue):
                    continue

                if "CA" in residue:
                    atoms.append(residue["CA"])

    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED INHIBITOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_inhibitor_screen(
    receptor_pdbqt: str | None,
    receptor_pdb: str,
    small_molecule_dir: str,
    peptide_pdb_dir: str,
    docking_box: dict,
    rna_poses: Optional[list] = None,
    small_molecule_results: Optional[list] = None,
    peptide_results: Optional[list] = None,
    max_small_molecules: int | None = None,
) -> dict:
    """
    Run complete inhibitor screen.

    Returns ranked small molecules/peptides plus true inhibitor-vs-RNA overlap
    comparisons where possible.
    """
    center = docking_box.get("center") or (0.0, 0.0, 0.0)
    size = docking_box.get("size") or (20.0, 20.0, 20.0)

    # ── Small molecules ──────────────────────────────────────────────────────

    if small_molecule_results is not None:
        sm_results = small_molecule_results

    elif (
        receptor_pdbqt
        and os.path.exists(receptor_pdbqt)
        and os.path.isdir(small_molecule_dir)
    ):
        sm_results = dock_small_molecules(
            receptor_pdbqt=receptor_pdbqt,
            ligands_dir=small_molecule_dir,
            center=center,
            size=size,
            max_ligands=max_small_molecules,
        )

    else:
        if not receptor_pdbqt:
            log.warning("Skipping small-molecule docking: receptor_pdbqt is missing")
        elif not os.path.exists(receptor_pdbqt):
            log.warning("Skipping small-molecule docking: receptor_pdbqt not found: %s", receptor_pdbqt)
        elif not os.path.isdir(small_molecule_dir):
            log.warning("Skipping small-molecule docking: small_molecule_dir not found: %s", small_molecule_dir)

        sm_results = []

    sm_ranked = rank_small_molecules(sm_results)

    # ── Peptides ─────────────────────────────────────────────────────────────

    if peptide_results is not None:
        pep_results = peptide_results

    elif os.path.isdir(peptide_pdb_dir):
        pep_results = dock_peptides(
            receptor_pdb=receptor_pdb,
            peptide_pdbs_dir=peptide_pdb_dir,
        )

    else:
        log.warning("Skipping peptide docking: peptide_pdb_dir not found: %s", peptide_pdb_dir)
        pep_results = []

    pep_ranked = rank_peptides(pep_results)

    # ── Pose comparison ───────────────────────────────────────────────────────

    comparison = None
    small_molecule_comparison = None
    peptide_comparison = None

    valid_rna_poses = _coerce_existing_pose_paths(rna_poses or [])

    if valid_rna_poses:
        best_sm = sm_ranked[0] if sm_ranked and sm_ranked[0].get("valid") else None
        best_pep = pep_ranked[0] if pep_ranked and pep_ranked[0].get("valid") else None

        if best_sm and best_sm.get("output_file"):
            small_molecule_comparison = compare_inhibitor_with_rna_interface_overlap(
                inhibitor_pose_file=best_sm["output_file"],
                rna_poses=valid_rna_poses,
                ligand_type="small_molecule",
            )

        if best_pep and best_pep.get("complex_pdb"):
            peptide_comparison = compare_inhibitor_with_rna_interface_overlap(
                inhibitor_pose_file=best_pep["complex_pdb"],
                rna_poses=valid_rna_poses,
                ligand_type="peptide",
            )

        comparable = [
            c for c in (small_molecule_comparison, peptide_comparison)
            if isinstance(c, dict) and c.get("comparable")
        ]

        if comparable:
            comparison = max(
                comparable,
                key=lambda c: c.get("overlap_score", 0.0),
            )

        elif best_pep and best_pep.get("complex_pdb"):
            comparison = compare_inhibitor_with_rna_pose(
                best_pep["complex_pdb"],
                valid_rna_poses,
            )

    # ── Summary ───────────────────────────────────────────────────────────────

    n_valid_sm = sum(1 for r in sm_ranked if isinstance(r, dict) and r.get("valid"))
    n_valid_pep = sum(1 for r in pep_ranked if isinstance(r, dict) and r.get("valid"))

    best_sm_data = sm_ranked[0] if sm_ranked else None
    best_pep_data = pep_ranked[0] if pep_ranked else None

    summary_parts: list[str] = []

    if (
        best_sm_data
        and best_sm_data.get("valid")
        and best_sm_data.get("binding_energy") is not None
    ):
        summary_parts.append(
            f"Best small molecule: {best_sm_data['ligand_name']} "
            f"({best_sm_data['binding_energy']:.2f} kcal/mol, Vina estimate)"
        )

    if (
        best_pep_data
        and best_pep_data.get("valid")
        and best_pep_data.get("hdock_score") is not None
    ):
        summary_parts.append(
            f"Best peptide: {best_pep_data['peptide_sequence']} "
            f"(HDOCK relative score: {best_pep_data['hdock_score']})"
        )

    if comparison and comparison.get("comparable"):
        method = comparison.get("method")

        if method == "atom_interface_overlap":
            min_d = comparison.get("min_distance_A")
            ligand_type = comparison.get("ligand_type", "inhibitor")
            overlap_percent = comparison.get("overlap_percent", 0.0)

            if min_d is not None:
                summary_parts.append(
                    f"RNA pocket overlap: {overlap_percent:.1f}% "
                    f"({ligand_type}, min distance: {min_d:.1f} Å)"
                )
            else:
                summary_parts.append(
                    f"RNA pocket overlap: {overlap_percent:.1f}% ({ligand_type})"
                )

        else:
            summary_parts.append(
                f"RNA overlap score: {comparison['overlap_score']:.2f} "
                f"(legacy RMSD proxy: {comparison.get('min_rmsd', 0.0):.1f} Å)"
            )

    summary = "; ".join(summary_parts) if summary_parts else "No valid inhibitor results"

    return {
        "small_molecules": sm_ranked,
        "peptides": pep_ranked,
        "best_small_molecule": best_sm_data,
        "best_peptide": best_pep_data,

        "comparison": comparison,
        "small_molecule_comparison": small_molecule_comparison,
        "peptide_comparison": peptide_comparison,

        "summary": summary,
        "n_small_molecules": len(sm_ranked),
        "n_peptides": len(pep_ranked),
        "n_valid_small_molecules": n_valid_sm,
        "n_valid_peptides": n_valid_pep,
    }