"""
inhibitor_docking.py

Main docking execution module for the VLAB2 inhibitor screening pipeline.

Coordinates:
1. Small-molecule docking via AutoDock Vina
2. Peptide docking via HDOCK (protein-protein mode)
3. Pose comparison between inhibitor and RNA-binding poses

Design decisions:
- Vina for small molecules (validated, already in env)
- HDOCK for peptides (same binary as protein-RNA, protein-protein mode)
- Bio.PDB for RMSD-based pose overlap comparison
- Parallel execution where possible
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

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
) -> list:
    """
    Dock all prepared small-molecule PDBQT files in ligands_dir against
    the receptor using AutoDock Vina.

    Args:
        receptor_pdbqt: Path to prepared receptor PDBQT file.
        ligands_dir: Directory containing ligand PDBQT files.
        center: Vina search box center (x, y, z) in Å.
        size: Vina search box dimensions (sx, sy, sz) in Å.
        exhaustiveness: Vina exhaustiveness parameter.
        vina_path: Optional path to vina binary.

    Returns:
        List of result dicts, one per ligand. Each dict has keys:
            - ligand_name (str): Base name of the ligand file.
            - ligand_path (str): Full path to the ligand PDBQT.
            - binding_energy (float or None): Vina binding energy (kcal/mol).
            - binding_units (str): "vina_kcal_mol".
            - binding_energy_is_physical (bool): True for Vina kcal/mol estimate.
            - method (str): "vina" or "vina_failed".
            - dock_method (str): Same as method, for downstream schema consistency.
            - valid (bool): Whether docking succeeded.
            - output_file (str or None): Path to docked PDBQT output.
            - error (str or None): Error message if failed.
    """
    vina = VinaDocking(vina_path=vina_path)
    results = []

    ligands_dir = Path(ligands_dir)
    if not ligands_dir.is_dir():
        log.error("Ligands directory not found: %s", ligands_dir)
        return []

    receptor_resolved = Path(receptor_pdbqt).resolve()

    ligand_files = []
    for p in sorted(ligands_dir.glob("*.pdbqt")):
        try:
            if p.resolve() == receptor_resolved:
                continue
        except OSError:
            pass

        if p.name.endswith("_receptor.pdbqt"):
            continue
        if p.name.endswith("_docked.pdbqt"):
            continue

        ligand_files.append(p)

    if not ligand_files:
        log.warning("No PDBQT files found in %s", ligands_dir)
        return []

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
            log.info(
                "  → %s: %.3f kcal/mol",
                ligand_path.stem,
                binding_energy,
            )
        else:
            log.warning("  → %s: FAILED (%s)", ligand_path.stem, result.get("error"))

    return results


def rank_small_molecules(results: list) -> list:
    """
    Sort small-molecule docking results by binding energy (ascending).

    Args:
        results: List of dock_small_molecules result dicts.

    Returns:
        Sorted list (best binder first). Invalid results are placed at the end.
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
    Strictly identify peptide input PDB files produced by peptide preparation.

    Expected format:
        clean_peptide_<hash>_<sequence>.pdb

    Excludes generated HDOCK complexes and other derived files so that
    HDOCK output is never recursively re-docked as a peptide ligand.
    """
    if not path.is_file():
        return False

    name = path.name

    if "_hdock_complex" in name:
        return False
    if "_docked" in name:
        return False
    if "_complex" in name:
        return False

    return bool(
        re.fullmatch(
            r"clean_peptide_[0-9A-Fa-f]+_[A-Za-z]+\.pdb",
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
    Dock all prepared peptide PDB files in peptide_pdbs_dir against the
    receptor protein using HDOCK in protein-protein mode.

    HDOCK is called once per peptide (receptor + peptide as ligand).
    The top model is extracted and scored.

    Args:
        receptor_pdb: Path to the receptor (protein) PDB file.
        peptide_pdbs_dir: Directory containing cleaned peptide PDB files.
        n_models: Number of models to generate per docking run.
        hdock_path: Optional path to hdock binary.
        createpl_path: Optional path to createpl binary.

    Returns:
        List of result dicts, one per peptide. Each dict has keys:
            - sequence (str): Peptide sequence.
            - peptide_sequence (str): Peptide sequence.
            - peptide_pdb (str): Path to the peptide PDB file.
            - score (float or None): HDOCK-relative score.
            - hdock_score (float or None): HDOCK-relative score.
            - dock_score (float or None): HDOCK-relative score.
            - binding_units (str): "hdock_relative_score".
            - binding_energy_is_physical (bool): False.
            - method (str): "hdock" or "hdock_failed".
            - dock_method (str): Same as method, for downstream schema consistency.
            - valid (bool): Whether docking succeeded.
            - complex_pdb (str or None): Path to the top complex PDB.
            - complex_file (str or None): Alias for complex_pdb.
            - dock_complex_file (str or None): Alias for complex_pdb.
            - error (str or None): Error message if failed.
    """
    hdock = HDockDocking(
        hdock_path=hdock_path,
        createpl_path=createpl_path,
    )

    results = []

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
    """Extract the best HDOCK score from a docking result dict."""
    if not isinstance(result, dict):
        return None

    if not result.get("valid"):
        return None

    # Avoid `or` chaining because valid scores could theoretically be 0.0.
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
    """Extract the peptide sequence from a peptide PDB filename."""
    name = Path(filename).stem  # e.g. clean_peptide_a1b2c3d4_RRM
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "clean" and parts[1] == "peptide":
        return parts[-1]
    if len(parts) >= 3:
        return parts[-1]
    return name


def rank_peptides(results: list) -> list:
    """
    Sort peptide docking results by HDOCK score (ascending = better).

    Args:
        results: List of dock_peptides result dicts.

    Returns:
        Sorted list (best score first). Failed results at the end.
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
# POSE COMPARISON: INHIBITOR vs RNA-BINDING POSES
# ─────────────────────────────────────────────────────────────────────────────

def compare_inhibitor_with_rna_pose(
    inhibitor_complex_pdb: str,
    rna_poses: list,
    cutoff_rms: float = 5.0,
) -> dict:
    """
    Compare an inhibitor binding pose with the RNA-protein docking poses
    to assess whether the inhibitor occupies the same space as the RNA.

    Current method:
        Uses CA-atom RMSD between complexes after CA-based superposition.
        This is a coarse geometric proxy, not a rigorous ligand/RNA pocket
        overlap metric.

    Args:
        inhibitor_complex_pdb: Path to the inhibitor-protein complex PDB.
        rna_poses: List of RNA-protein complex PDB paths from HDOCK.
        cutoff_rms: RMSD cutoff (Å). Poses within this distance are
                    considered to overlap (lower = more overlap).

    Returns:
        dict with keys:
            - min_rmsd (float or None): Minimum RMSD to any RNA pose.
            - overlap_score (float): 0-1 score (1 = full overlap, 0 = no overlap).
                                     Computed as max(0, 1 - min_rmsd / cutoff_rms).
            - n_comparable_poses (int): Number of RNA poses compared.
            - comparable (bool): Whether comparison was possible.
            - error (str or None): Error message if comparison failed.
    """
    if not _HAS_BIOPYTHON:
        return {
            "min_rmsd": None,
            "overlap_score": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": "Bio.PDB not available",
        }

    if not rna_poses:
        return {
            "min_rmsd": None,
            "overlap_score": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": "No RNA poses provided",
        }

    if not os.path.exists(inhibitor_complex_pdb):
        return {
            "min_rmsd": None,
            "overlap_score": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": f"Inhibitor complex PDB not found: {inhibitor_complex_pdb}",
        }

    try:
        parser = PDB.PDBParser(QUIET=True)
        inhibitor_structure = parser.get_structure("inhibitor", inhibitor_complex_pdb)

        rmsd_values = []
        for rna_pose in rna_poses:
            if not os.path.exists(rna_pose):
                continue

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
                "min_rmsd": None,
                "overlap_score": 0.0,
                "n_comparable_poses": 0,
                "comparable": False,
                "error": "Could not compute RMSD for any pose pair",
            }

        min_rmsd = min(rmsd_values)
        overlap_score = max(0.0, 1.0 - min_rmsd / cutoff_rms)

        return {
            "min_rmsd": float(min_rmsd),
            "overlap_score": float(overlap_score),
            "n_comparable_poses": len(rmsd_values),
            "comparable": True,
            "error": None,
        }

    except Exception as e:
        log.exception("Pose comparison failed")
        return {
            "min_rmsd": None,
            "overlap_score": 0.0,
            "n_comparable_poses": 0,
            "comparable": False,
            "error": str(e),
        }


def _compute_ca_rmsd(structure_a, structure_b) -> Optional:
    """
    Compute CA-atom RMSD between two structures after CA-based superposition.

    Returns RMSD in Å, or None if computation fails.
    """
    try:
        from Bio.SVDSuperimposer import SVDSuperimposer

        ca_a = _get_ca_atoms(structure_a)
        ca_b = _get_ca_atoms(structure_b)

        if len(ca_a) < 3 or len(ca_b) < 3:
            return None

        n = min(len(ca_a), len(ca_b))
        ca_a = ca_a[:n]
        ca_b = ca_b[:n]

        coords_a = np.array([a.get_coord() for a in ca_a])
        coords_b = np.array([a.get_coord() for a in ca_b])

        superimposer = SVDSuperimposer(coords_a, coords_b)
        superimposer.run(coords_a, coords_b)

        rmsd = superimposer.get_rms()
        return float(rmsd)

    except Exception:
        return None


def _get_ca_atoms(structure) -> list:
    """Extract all CA atoms from a structure."""
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
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
) -> dict:
    """
    Run the complete inhibitor screening pipeline.

    This function can be called with pre-computed results (for re-ranking
    or re-comparison) or will compute from scratch if results are not provided.

    Args:
        receptor_pdbqt: Path to receptor PDBQT (for Vina).
        receptor_pdb: Path to receptor PDB (for HDOCK).
        small_molecule_dir: Directory with prepared small-molecule PDBQTs.
        peptide_pdb_dir: Directory with cleaned peptide PDBs.
        docking_box: Docking box dict from define_docking_box().
        rna_poses: Optional list of RNA-protein complex PDB paths for comparison.
        small_molecule_results: Optional pre-computed small-molecule results.
        peptide_results: Optional pre-computed peptide results.

    Returns:
        dict with keys:
            - small_molecules (list): Ranked small-molecule results.
            - peptides (list): Ranked peptide results.
            - best_small_molecule (dict or None): Top small-molecule result.
            - best_peptide (dict or None): Top peptide result.
            - comparison (dict or None): Inhibitor vs RNA pose comparison.
            - summary (str): Human-readable summary.
            - n_small_molecules (int): Number of small molecules screened.
            - n_peptides (int): Number of peptides screened.
            - n_valid_small_molecules (int): Number with valid Vina scores.
            - n_valid_peptides (int): Number with valid HDOCK scores.
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
        )
    else:
        if not receptor_pdbqt:
            log.warning("Skipping small-molecule docking: receptor_pdbqt is missing")
        elif not os.path.exists(receptor_pdbqt):
            log.warning(
                "Skipping small-molecule docking: receptor_pdbqt not found: %s",
                receptor_pdbqt,
            )
        elif not os.path.isdir(small_molecule_dir):
            log.warning(
                "Skipping small-molecule docking: small_molecule_dir not found: %s",
                small_molecule_dir,
            )

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
        log.warning(
            "Skipping peptide docking: peptide_pdb_dir not found: %s",
            peptide_pdb_dir,
        )
        pep_results = []

    pep_ranked = rank_peptides(pep_results)

    # ── Pose comparison ───────────────────────────────────────────────────────

    comparison = None
    if rna_poses:
        best_pep = pep_ranked[0] if pep_ranked and pep_ranked[0].get("valid") else None

        if best_pep and best_pep.get("complex_pdb"):
            comparison = compare_inhibitor_with_rna_pose(
                best_pep["complex_pdb"],
                rna_poses,
            )

    # ── Summary ───────────────────────────────────────────────────────────────

    n_valid_sm = sum(1 for r in sm_ranked if isinstance(r, dict) and r.get("valid"))
    n_valid_pep = sum(1 for r in pep_ranked if isinstance(r, dict) and r.get("valid"))

    best_sm_data = sm_ranked[0] if sm_ranked else None
    best_pep_data = pep_ranked[0] if pep_ranked else None

    summary_parts = []

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
        summary_parts.append(
            f"RNA overlap score: {comparison['overlap_score']:.2f} "
            f"(RMSD proxy: {comparison['min_rmsd']:.1f} Å)"
        )

    summary = "; ".join(summary_parts) if summary_parts else "No valid inhibitor results"

    return {
        "small_molecules": sm_ranked,
        "peptides": pep_ranked,
        "best_small_molecule": best_sm_data,
        "best_peptide": best_pep_data,
        "comparison": comparison,
        "summary": summary,
        "n_small_molecules": len(sm_ranked),
        "n_peptides": len(pep_ranked),
        "n_valid_small_molecules": n_valid_sm,
        "n_valid_peptides": n_valid_pep,
    }