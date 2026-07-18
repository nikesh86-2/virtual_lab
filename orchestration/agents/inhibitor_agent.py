"""
inhibitor_agent.py — Small-molecule & peptide inhibitor screening agent.

Screens candidate inhibitors against the RNA-binding pocket identified from
the protein_agent's interface contacts. Coordinates:
  1. Pocket definition (RNA-binding site centroid + box dimensions)
  2. Small-molecule fetching from PubChem + AutoDock Vina docking
  3. Peptide design/fetching + HDOCK protein-protein docking
  4. Pose comparison with RNA-protein complex
  5. Snapshot rendering via PyMOL

State inputs
------------
  target_pdb          : str  — path or PDB ID for protein target
  target_pdb_path     : str  — path to validated protein PDB, preferred if available
  interface_contacts  : dict — from protein_agent
  rna_sequence        : str  — RNA sequence, optional
  rna_structure       : str  — secondary structure, optional
  inhibitor_enabled   : bool — informational state flag

State outputs
-------------
  inhibitor_small_molecules                 : list[dict]
  inhibitor_peptides                        : list[dict]
  inhibitor_analysis                        : str
  inhibitor_summary                         : str
  inhibitor_binding_site_overlap            : float  # percent, legacy-compatible
  inhibitor_binding_site_overlap_score      : float  # 0-1
  inhibitor_pose_comparison                 : dict
  inhibitor_small_molecule_comparison       : dict | None
  inhibitor_peptide_comparison              : dict | None
  inhibitor_docking_box                     : dict
  inhibitor_snapshot_paths                  : list[str]
  stage_outputs                             : list[dict]
  conversation_history                      : list[dict]
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from VLAB2.core.gpu_manager import clear_gpu
from VLAB2.core.inhibitor_docking import run_inhibitor_screen
from VLAB2.core.peptide_prep import (
    design_inhibitor_peptides,
    get_known_antiviral_peptides,
    prepare_peptides,
)
from VLAB2.core.protein_prep import ensure_protein_pdb
from VLAB2.core.rna_binding_pocket import define_docking_box
from VLAB2.core.small_molecule_prep import (
    fetch_and_prepare_compounds,
    fetch_known_rna_binding_inhibitors,
    prepare_receptor_pdbqt,
)
from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.docking_visuals import (
    render_inhibitor_snapshots_for_results,
)
from VLAB2.orchestration.utils.text_utils import _safe_file_tag


log = logging.getLogger("virtual_lab")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, "").strip().lower()

    if val in ("1", "true", "yes", "on"):
        return True

    if val in ("0", "false", "no", "off"):
        return False

    return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _is_valid_pdb(path: str | None) -> bool:
    if not path:
        return False

    p = Path(path)
    return p.exists() and p.stat().st_size > 100


def _looks_like_pdb_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    raw = value.strip()
    return len(raw) == 4 and raw.isalnum()


def _empty_inhibitor_update(
    analysis: str,
    summary: str | None = None,
    enabled: bool = True,
) -> dict:
    """
    Return a complete inhibitor-update payload for disabled/skipped/error paths.
    Keeps downstream manifest/export code from seeing missing keys.
    """
    summary = summary or analysis

    return {
        "inhibitor_enabled": enabled,
        "inhibitor_small_molecules": [],
        "inhibitor_peptides": [],
        "inhibitor_analysis": analysis,
        "inhibitor_summary": summary,
        "inhibitor_binding_site_overlap": 0.0,
        "inhibitor_binding_site_overlap_score": 0.0,
        "inhibitor_pose_comparison": {},
        "inhibitor_small_molecule_comparison": None,
        "inhibitor_peptide_comparison": None,
        "inhibitor_docking_box": {},
        "inhibitor_snapshot_paths": [],
        "stage_outputs": [],
        "conversation_history": [],
    }


def _rna_pose_paths_from_state(state: dict) -> list[str]:
    """
    Extract RNA-protein complex PDB paths from state.

    Supports:
      - binding_results rows
      - direct path strings
      - accidental dict records in rna_poses-like fields
    """
    out: list[str] = []
    seen: set[str] = set()

    candidates: list[Any] = []

    for r in state.get("binding_results", []) or []:
        candidates.append(r)

    for key in (
        "rna_poses",
        "rna_pose_files",
        "rna_complexes",
        "docking_snapshot_files",
    ):
        for item in state.get(key, []) or []:
            candidates.append(item)

    for item in candidates:
        path = None

        if isinstance(item, (str, os.PathLike)):
            path = str(item)

        elif isinstance(item, dict):
            path = (
                item.get("dock_complex_file")
                or item.get("complex_file")
                or item.get("complex_pdb")
                or item.get("dock_complex_pdb")
                or item.get("output_complex")
            )

        if not path:
            continue

        path = str(path)

        if path in seen:
            continue

        if Path(path).exists():
            seen.add(path)
            out.append(path)
        else:
            log.debug("Skipping missing RNA pose path: %s", path)

    return out


def _resolve_target_pdb_from_state(state: dict) -> tuple[str | None, str | None]:
    """
    Resolve target PDB path and PDB ID from state.

    Returns:
      (target_pdb_path, target_pdb_id)
    """
    raw_target = (
        state.get("target_pdb_path")
        or state.get("target_pdb_file")
        or state.get("receptor_pdb")
        or state.get("target_pdb")
    )

    target_pdb_id = (
        state.get("target_pdb_id")
        or state.get("target_pdb")
    )

    target_pdb = raw_target

    if _is_valid_pdb(target_pdb):
        if _looks_like_pdb_id(str(target_pdb_id)):
            return str(target_pdb), str(target_pdb_id).strip().upper()

        return str(target_pdb), None

    pdb_id = None

    if _looks_like_pdb_id(raw_target):
        pdb_id = str(raw_target).strip().upper()

    elif _looks_like_pdb_id(target_pdb_id):
        pdb_id = str(target_pdb_id).strip().upper()

    if not pdb_id:
        partial_targets = state.get("partial_success_targets", []) or []
        resolved_partial = state.get("resolved_partial_success_targets", []) or []

        candidates = partial_targets or resolved_partial

        if candidates:
            first = candidates[0]

            if isinstance(first, dict):
                candidate = first.get("target_pdb") or first.get("pdb_id")
            else:
                candidate = str(first).strip().upper()

            if _looks_like_pdb_id(candidate):
                pdb_id = str(candidate).strip().upper()

    if pdb_id:
        try:
            log.info("inhibitor_agent: resolving PDB ID to file: %s", pdb_id)
            target_pdb = ensure_protein_pdb(pdb_id)
            log.info("inhibitor_agent: resolved target_pdb=%s", target_pdb)

        except Exception as e:
            log.warning("inhibitor_agent: failed to resolve PDB ID %s: %s", pdb_id, e)
            target_pdb = None

    if _is_valid_pdb(target_pdb):
        return str(target_pdb), pdb_id

    return None, pdb_id


def _reset_selected_dir(path: str | Path, suffixes: tuple[str, ...]) -> None:
    """
    Fully clear selected per-run input directory.

    This directory should contain only files intentionally selected for the
    current inhibitor run. Removing the whole directory prevents stale ligand
    or peptide files from previous runs inflating counts.
    """
    p = Path(path)

    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

    p.mkdir(parents=True, exist_ok=True)


def _compound_pdbqt_path(compound: dict) -> str | None:
    """
    Recover prepared ligand PDBQT path from compound metadata.
    """
    if not isinstance(compound, dict):
        return None

    for key in (
        "pdbqt_path",
        "ligand_pdbqt",
        "pdbqt_file",
        "pdbqt",
        "prepared_pdbqt",
        "output_file",
        "ligand_path",
    ):
        value = compound.get(key)

        if value and Path(str(value)).exists():
            return str(value)

    return None


def _copy_selected_ligands(
    compounds: list[dict],
    selected_dir: str | Path,
) -> list[str]:
    """
    Copy only this run's selected ligand PDBQT files into an isolated directory.

    This prevents old cached ligands from being redocked by dock_small_molecules(),
    which scans all *.pdbqt files in the directory it is given.
    """
    selected_path = Path(selected_dir)
    _reset_selected_dir(selected_path, suffixes=(".pdbqt",))

    copied: list[dict] = []
    seen_dest: set[str] = set()

    for idx, compound in enumerate(compounds or []):
        if not isinstance(compound, dict):
            continue

        src = _compound_pdbqt_path(compound)

        if not src:
            log.warning(
                "Skipping selected ligand with no PDBQT path: name=%s cid=%s",
                compound.get("name") or compound.get("compound_name"),
                compound.get("cid") or compound.get("pubchem_cid"),
            )
            continue

        src_path = Path(src)

        if not src_path.exists():
            continue

        name = (
            compound.get("name")
            or compound.get("compound_name")
            or compound.get("display_name")
            or src_path.stem
            or f"ligand_{idx}"
        )

        safe_name = _safe_file_tag(str(name))
        dest = selected_path / f"{safe_name}.pdbqt"

        if str(dest) in seen_dest:
            dest = selected_path / f"{safe_name}_{idx}.pdbqt"

        seen_dest.add(str(dest))

        try:
            if src_path.resolve() != dest.resolve():
                shutil.copy2(src_path, dest)

        except FileNotFoundError:
            continue

        except Exception as e:
            log.warning("Could not copy selected ligand %s -> %s: %s", src_path, dest, e)
            continue

        row = dict(compound)
        row["selected_pdbqt"] = str(dest)
        row["pdbqt_path"] = str(dest)

        copied.append(row)

    return copied


def _copy_selected_peptides(
    prepared_peptides: list[dict],
    selected_dir: str | Path,
) -> list[str]:
    """
    Copy only this run's selected clean peptide PDB files into an isolated directory.

    This prevents old clean peptide files or generated HDOCK complexes from being
    redocked in later iterations.
    """
    selected_path = Path(selected_dir)
    _reset_selected_dir(selected_path, suffixes=(".pdb",))

    copied: list[dict] = []
    seen_seq: set[str] = set()

    for idx, pep in enumerate(prepared_peptides or []):
        if not isinstance(pep, dict):
            continue

        clean_pdb = pep.get("clean_pdb")
        seq = str(pep.get("sequence") or "").strip()

        if not seq:
            log.warning("Skipping peptide with missing sequence: %s", pep)
            continue

        if seq in seen_seq:
            continue

        seen_seq.add(seq)

        if not clean_pdb or not Path(str(clean_pdb)).exists():
            log.warning(
                "Skipping peptide with missing clean PDB: seq=%s clean_pdb=%s error=%s",
                seq,
                clean_pdb,
                pep.get("error"),
            )
            continue

        src_path = Path(str(clean_pdb))
        dest = selected_path / src_path.name

        if not dest.name.startswith("clean_peptide_"):
            safe_seq = _safe_file_tag(seq)
            dest = selected_path / f"clean_peptide_selected_{idx}_{safe_seq}.pdb"

        try:
            if src_path.resolve() != dest.resolve():
                shutil.copy2(src_path, dest)

        except Exception as e:
            log.warning("Could not copy selected peptide %s -> %s: %s", src_path, dest, e)
            continue

        row = dict(pep)
        row["selected_clean_pdb"] = str(dest)
        row["clean_pdb"] = str(dest)

        copied.append(row)

    return copied


def _dedupe_compounds(compounds: list[dict], max_count: int) -> list[str]:
    """
    Deduplicate compounds robustly.

    Priority:
      1. SMILES, if present
      2. PubChem CID, if present
      3. normalised compound name
    """
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for c in compounds or []:
        if not isinstance(c, dict):
            continue

        smiles = str(c.get("smiles") or "").strip()
        cid = str(c.get("cid") or c.get("pubchem_cid") or "").strip()
        name = str(
            c.get("name")
            or c.get("compound_name")
            or c.get("display_name")
            or ""
        ).strip().lower()

        if smiles:
            key = ("smiles", smiles)

        elif cid:
            key = ("cid", cid)

        elif name:
            key = ("name", name)

        else:
            continue

        if key in seen:
            continue

        seen.add(key)
        out.append(c)

        if len(out) >= max_count:
            break

    return out


def _dedupe_peptide_specs(peptides: list[dict], max_count: int) -> list[str]:
    """
    Deduplicate peptide specs by uppercase sequence.
    """
    out: list[dict] = []
    seen: set[str] = set()

    for p in peptides or []:
        if not isinstance(p, dict):
            continue

        seq = str(p.get("sequence") or "").strip().upper()

        if not seq or seq in seen:
            continue

        seen.add(seq)

        row = dict(p)
        row["sequence"] = seq
        out.append(row)

        if len(out) >= max_count:
            break

    return out


def _enrich_inhibitor_rows(
    rows: list[dict],
    target_pdb_id: str | None,
    target_pdb_path: str | None,
    target_tag: str,
    ligand_type: str,
) -> list[str]:
    """
    Add consistent target and ligand-type metadata to inhibitor result rows.
    """
    enriched: list[dict] = []

    for r in rows or []:
        if not isinstance(r, dict):
            enriched.append(r)
            continue

        row = dict(r)

        row["target_pdb"] = target_pdb_id or target_tag
        row["target_pdb_id"] = target_pdb_id or target_tag
        row["target_pdb_path"] = target_pdb_path
        row["target_tag"] = target_tag
        row["ligand_type"] = ligand_type

        if ligand_type == "small_molecule":
            row["binding_units"] = row.get("binding_units") or "vina_kcal_mol"
            row["binding_energy_is_physical"] = bool(
                row.get("binding_energy_is_physical", True)
            )

        elif ligand_type == "peptide":
            row["binding_units"] = row.get("binding_units") or "hdock_relative_score"
            row["binding_energy_is_physical"] = False
            row["score"] = row.get("score", row.get("hdock_score"))
            row["dock_score"] = row.get("dock_score", row.get("hdock_score"))
            row["hdock_score"] = row.get("hdock_score", row.get("score"))

        enriched.append(row)

    return enriched


def _comparison_overlap_score(comparison: Any) -> float:
    """
    Extract overlap score in 0-1 scale from a comparison dict.
    """
    if not isinstance(comparison, dict):
        return 0.0

    if comparison.get("overlap_score") is not None:
        try:
            return float(comparison.get("overlap_score") or 0.0)
        except Exception:
            return 0.0

    if comparison.get("overlap_percent") is not None:
        try:
            return float(comparison.get("overlap_percent") or 0.0) / 100.0
        except Exception:
            return 0.0

    return 0.0


def _comparison_overlap_percent(comparison: Any) -> float:
    """
    Extract overlap in percent from a comparison dict.
    """
    if not isinstance(comparison, dict):
        return 0.0

    if comparison.get("overlap_percent") is not None:
        try:
            return float(comparison.get("overlap_percent") or 0.0)
        except Exception:
            return 0.0

    if comparison.get("overlap_score") is not None:
        try:
            return float(comparison.get("overlap_score") or 0.0) * 100.0
        except Exception:
            return 0.0

    return 0.0


def record_stage_output(
    state: dict,
    stage: str,
    content: str,
    summary: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Build a stage-output record matching the canonical LabState shape.

    Includes both:
      - agent/stage
      - output/content

    This prevents final reports from labelling the inhibitor agent as UNKNOWN.
    """
    content = "" if content is None else str(content)

    return {
        "agent": stage,
        "stage": stage,
        "summary": summary or content[:200],
        "output": content,
        "content": content,
        "metadata": metadata or {},
    }


def add_conversation_entry(
    state: dict,
    role: str,
    content: str,
    agent: str,
) -> dict:
    """
    Build a conversation-history entry matching protein_agent's format.
    """
    return {
        "role": role,
        "content": content,
        "agent": agent,
    }


# ---------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------

def inhibitor_agent(state: dict) -> dict:
    """
    Screen small-molecule and peptide inhibitors against the RNA-binding pocket.

    Returns a dict of state updates for the LangGraph pipeline.
    """

    if not _env_bool("VLAB_INHIBITOR_ENABLED", default=True):
        log.info("Inhibitor screening disabled (VLAB_INHIBITOR_ENABLED != 1)")
        return _empty_inhibitor_update(
            analysis="Inhibitor screening disabled.",
            summary="Inhibitor screening disabled.",
            enabled=False,
        )

    result: dict[str, Any] = {}

    try:
        interface_contacts = state.get("interface_contacts")
        raw_target = state.get("target_pdb_path") or state.get("target_pdb")

        log.warning(
            "INHIBITOR_AGENT_ENTERED raw_target_pdb=%s interface_keys=%s",
            raw_target,
            list((interface_contacts or {}).keys()),
        )

        target_pdb, target_pdb_id = _resolve_target_pdb_from_state(state)

        log.warning(
            "INHIBITOR_TARGET_RESOLVED raw=%s path=%s pdb_id=%s valid=%s",
            state.get("target_pdb"),
            target_pdb,
            target_pdb_id,
            _is_valid_pdb(target_pdb),
        )

        if not _is_valid_pdb(target_pdb):
            log.warning(
                "inhibitor_agent: target_pdb not available after resolution. raw=%s resolved=%s",
                state.get("target_pdb"),
                target_pdb,
            )

            return _empty_inhibitor_update(
                analysis="Skipped: no validated protein target.",
                summary="Skipped: no validated protein target.",
                enabled=True,
            )

        if not interface_contacts:
            log.warning("inhibitor_agent: interface_contacts not available, skipping")
            return _empty_inhibitor_update(
                analysis="Skipped: no RNA-protein interface contacts.",
                summary="Skipped: no RNA-protein interface contacts.",
                enabled=True,
            )

        log.info("inhibitor_agent: starting inhibitor screening for %s", target_pdb)

        inhibitor_cache_dir = os.getenv(
            "VLAB_INHIBITOR_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "vlab_inhibitor_cache"),
        )

        target_tag = _safe_file_tag(
            target_pdb_id
            or state.get("target_pdb_id")
            or state.get("target_pdb")
            or target_pdb
        )

        run_root_dir = Path(inhibitor_cache_dir) / target_tag
        run_ligand_dir = run_root_dir / "ligands"
        run_peptide_dir = run_root_dir / "peptides"
        selected_ligand_dir = run_root_dir / "selected_ligands"
        selected_peptide_dir = run_root_dir / "selected_peptides"

        run_ligand_dir.mkdir(parents=True, exist_ok=True)
        run_peptide_dir.mkdir(parents=True, exist_ok=True)
        selected_ligand_dir.mkdir(parents=True, exist_ok=True)
        selected_peptide_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Step 1: Define docking box from RNA-binding pocket
        # ------------------------------------------------------------
        box_result = define_docking_box(target_pdb, interface_contacts)

        if box_result.get("error"):
            log.warning("inhibitor_agent: pocket definition error: %s", box_result["error"])

        docking_box = {
            "center": box_result.get("center"),
            "size": box_result.get("size"),
            "n_pocket_residues": box_result.get("n_pocket_residues", 0),
            "pocket_residues": box_result.get("pocket_residues", []),
        }

        result["inhibitor_docking_box"] = docking_box

        # ------------------------------------------------------------
        # Step 2: Fetch & prepare small molecules
        # ------------------------------------------------------------
        max_small_mols = _env_int("VLAB_INHIBITOR_MAX_SMALL_MOLECULES", 10)

        known_sm = fetch_known_rna_binding_inhibitors(
            max_compounds=max_small_mols,
            output_dir=str(run_ligand_dir),
            target_tag=target_tag,
        )

        log.info(
            "inhibitor_agent: %d known RNA-binding compounds available",
            len(known_sm or []),
        )

        pubchem_queries = os.getenv("VLAB_PUBCHEM_QUERIES", "").strip()
        pubchem_compounds = []

        if pubchem_queries:
            try:
                pubchem_compounds = fetch_and_prepare_compounds(
                    pubchem_queries.split("|"),
                    max_per_query=max_small_mols,
                    output_dir=str(run_ligand_dir),
                    target_tag=target_tag,
                )

                log.info(
                    "inhibitor_agent: fetched %d PubChem compounds",
                    len(pubchem_compounds or []),
                )

            except Exception as e:
                log.warning("inhibitor_agent: PubChem fetch failed: %s", e)

        small_mol_list = _dedupe_compounds(
            (known_sm or []) + (pubchem_compounds or []),
            max_count=max_small_mols,
        )

        selected_small_mols = _copy_selected_ligands(
            small_mol_list,
            selected_ligand_dir,
        )

        selected_ligand_files = sorted(Path(selected_ligand_dir).glob("*.pdbqt"))

        log.warning(
            "INHIBITOR_SELECTED_LIGANDS count=%d max=%d dir=%s files=%s",
            len(selected_ligand_files),
            max_small_mols,
            selected_ligand_dir,
            [p.name for p in selected_ligand_files],
        )

        log.info(
            "inhibitor_agent: %d small molecules selected for docking",
            len(selected_small_mols),
        )

        # ------------------------------------------------------------
        # Step 3: Prepare peptides
        # ------------------------------------------------------------
        max_peptides = _env_int("VLAB_INHIBITOR_MAX_PEPTIDES", 5)

        known_pep = get_known_antiviral_peptides()
        log.info(
            "inhibitor_agent: %d known antiviral peptides available",
            len(known_pep or []),
        )

        llm_designed_peptides = []

        try:
            llm = state.get("llm") or state.get("language_model")

            llm_designed_peptides = design_inhibitor_peptides(
                target_pdb=target_pdb,
                pocket_residues=docking_box.get("pocket_residues", []),
                llm=llm,
                max_peptides=min(max_peptides, 3),
            )

            log.info(
                "inhibitor_agent: LLM designed %d peptide candidates",
                len(llm_designed_peptides or []),
            )

        except Exception as e:
            log.warning("inhibitor_agent: peptide design failed: %s", e)

        peptide_specs = _dedupe_peptide_specs(
            (known_pep or []) + (llm_designed_peptides or []),
            max_count=max_peptides,
        )

        peptide_list: list[dict] = []

        if peptide_specs:
            try:
                prepared_peptides = prepare_peptides(
                    peptide_specs,
                    output_dir=str(run_peptide_dir),
                    target_tag=target_tag,
                )

                log.info(
                    "inhibitor_agent: %d peptides prepared for HDOCK",
                    len(prepared_peptides or []),
                )

                valid_peptide_list = [
                    p for p in prepared_peptides or []
                    if p.get("clean_pdb") and os.path.exists(p["clean_pdb"])
                ]

                for p in prepared_peptides or []:
                    if not p.get("clean_pdb"):
                        log.warning(
                            "Peptide prep failed or produced no clean PDB: seq=%s pdb=%s error=%s",
                            p.get("sequence"),
                            p.get("pdb_path"),
                            p.get("error"),
                        )

                for p in valid_peptide_list:
                    log.info(
                        "Valid peptide ready for docking: seq=%s clean_pdb=%s",
                        p.get("sequence"),
                        p.get("clean_pdb"),
                    )

                peptide_list = _copy_selected_peptides(
                    valid_peptide_list,
                    selected_peptide_dir,
                )

                log.info(
                    "inhibitor_agent: %d valid peptides selected for HDOCK",
                    len(peptide_list),
                )

            except Exception as e:
                log.warning("inhibitor_agent: peptide preparation failed: %s", e)

        # ------------------------------------------------------------
        # Step 4: Run combined inhibitor screen
        # ------------------------------------------------------------
        rna_poses = _rna_pose_paths_from_state(state)

        receptor_pdbqt = prepare_receptor_pdbqt(
            receptor_pdb=target_pdb,
            output_dir=inhibitor_cache_dir,
        )

        if receptor_pdbqt:
            log.info("inhibitor_agent: receptor PDBQT ready: %s", receptor_pdbqt)
        else:
            log.warning(
                "inhibitor_agent: receptor PDBQT unavailable; small-molecule Vina docking will be skipped"
            )

        screen_result = run_inhibitor_screen(
            receptor_pdbqt=receptor_pdbqt,
            receptor_pdb=target_pdb,
            small_molecule_dir=str(selected_ligand_dir),
            peptide_pdb_dir=str(selected_peptide_dir),
            docking_box=docking_box,
            rna_poses=rna_poses,
            max_small_molecules=max_small_mols,
            vina_seed=_env_int("VLAB_VINA_SEED", 1),
            force_vina_redock=_env_bool("VLAB_VINA_FORCE_REDOCK", False),
        )

        small_mol_results = screen_result.get("small_molecules", []) or []
        peptide_results = screen_result.get("peptides", []) or []

        small_mol_results = _enrich_inhibitor_rows(
            small_mol_results,
            target_pdb_id=target_pdb_id,
            target_pdb_path=target_pdb,
            target_tag=target_tag,
            ligand_type="small_molecule",
        )

        peptide_results = _enrich_inhibitor_rows(
            peptide_results,
            target_pdb_id=target_pdb_id,
            target_pdb_path=target_pdb,
            target_tag=target_tag,
            ligand_type="peptide",
        )

        comparison = screen_result.get("comparison") or {}
        small_molecule_comparison = screen_result.get("small_molecule_comparison")
        peptide_comparison = screen_result.get("peptide_comparison")

        binding_site_overlap_score = _comparison_overlap_score(comparison)
        binding_site_overlap_percent = _comparison_overlap_percent(comparison)

        analysis_text = screen_result.get("summary", "")

        result["inhibitor_small_molecules"] = small_mol_results
        result["inhibitor_peptides"] = peptide_results
        result["inhibitor_best_small_molecule"] = screen_result.get("best_small_molecule")
        result["inhibitor_best_peptide"] = screen_result.get("best_peptide")
        result["inhibitor_pose_comparison"] = comparison
        result["inhibitor_small_molecule_comparison"] = small_molecule_comparison
        result["inhibitor_peptide_comparison"] = peptide_comparison
        result["inhibitor_binding_site_overlap"] = binding_site_overlap_percent
        result["inhibitor_binding_site_overlap_score"] = binding_site_overlap_score
        result["inhibitor_analysis"] = analysis_text
        result["inhibitor_vina_seed"] = screen_result.get("vina_seed")
        result["inhibitor_vina_cache_hits"] = screen_result.get("vina_cache_hits", 0)
        result["inhibitor_vina_cache_misses"] = screen_result.get("vina_cache_misses", 0)
        result["inhibitor_vina_cache_enabled"] = screen_result.get(
            "vina_cache_enabled", False
        )

        # ------------------------------------------------------------
        # Step 5: Build summary
        # ------------------------------------------------------------
        n_valid_sm = sum(
            1 for r in small_mol_results
            if isinstance(r, dict) and r.get("valid")
        )
        n_valid_pep = sum(
            1 for r in peptide_results
            if isinstance(r, dict) and r.get("valid")
        )

        summary_parts: list[str] = []

        if selected_small_mols or small_mol_results:
            summary_parts.append(
                f"{n_valid_sm}/{len(small_mol_results)} small molecules docked successfully."
            )

        if peptide_list or peptide_results:
            summary_parts.append(
                f"{n_valid_pep}/{len(peptide_results)} peptides docked successfully."
            )

        if binding_site_overlap_percent > 0:
            summary_parts.append(
                f"Binding-site overlap with RNA interface: {binding_site_overlap_percent:.1f}%."
            )

        if peptide_results:
            summary_parts.append(
                "Peptide HDOCK scores are exploratory relative docking ranks, "
                "not physical binding free energies."
            )

        if not summary_parts:
            summary_parts.append("No inhibitors docked successfully.")

        if small_mol_results:
            summary_parts.append(
                f"Vina cache: {screen_result.get('vina_cache_hits', 0)} hit(s), "
                f"{screen_result.get('vina_cache_misses', 0)} miss(es), "
                f"seed={screen_result.get('vina_seed')}."
            )

        inhibitor_summary = " ".join(summary_parts)
        result["inhibitor_summary"] = inhibitor_summary

        # ------------------------------------------------------------
        # Step 6: Render PyMOL snapshots
        # ------------------------------------------------------------
        try:
            snapshot_state = {
                **state,
                **result,
                "target_pdb": target_pdb_id or target_tag,
                "target_pdb_path": target_pdb,
                "vina_seed": screen_result.get("vina_seed"),
                "vina_cache_hits": screen_result.get("vina_cache_hits", 0),
                "vina_cache_misses": screen_result.get("vina_cache_misses", 0),
                "vina_cache_enabled": screen_result.get("vina_cache_enabled", False),
                "target_pdb_id": target_pdb_id or target_tag,
            }

            visual_paths = render_inhibitor_snapshots_for_results(snapshot_state)

            if visual_paths:
                result.update(visual_paths)

        except Exception as e:
            log.warning("inhibitor_agent: snapshot rendering failed: %s", e)

        # ------------------------------------------------------------
        # Step 7: Build stage output & conversation history
        # ------------------------------------------------------------
        stage_output = record_stage_output(
            state,
            "inhibitor",
            analysis_text or inhibitor_summary,
            summary=(
                f"Inhibitor screen: {n_valid_sm} small molecules, "
                f"{n_valid_pep} peptides"
            ),
            metadata={
                "target_pdb": target_pdb_id or target_tag,
                "target_pdb_path": target_pdb,
                "n_small_molecules": len(small_mol_results),
                "n_selected_small_molecules": len(selected_small_mols),
                "n_peptides": len(peptide_results),
                "n_selected_peptides": len(peptide_list),
                "n_valid_small_molecules": n_valid_sm,
                "n_valid_peptides": n_valid_pep,
                "binding_site_overlap": binding_site_overlap_percent,
                "binding_site_overlap_score": binding_site_overlap_score,
                "inhibitor_pose_comparison": comparison,
                "inhibitor_small_molecule_comparison": small_molecule_comparison,
                "inhibitor_peptide_comparison": peptide_comparison,
                "docking_box": docking_box,
                "small_molecule_binding_units": "vina_kcal_mol",
                "peptide_binding_units": "hdock_relative_score",
                "peptide_binding_energy_is_physical": False,
                "peptide_score_caveat": (
                    "Peptide HDOCK scores are exploratory relative docking ranks, "
                    "not physical binding free energies."
                ),
            },
        )

        conv_entry = add_conversation_entry(
            state,
            "assistant",
            f"[Inhibitor Screen] {inhibitor_summary}\n\n{analysis_text}",
            "inhibitor",
        )

        result["stage_outputs"] = [stage_output]
        result["conversation_history"] = [conv_entry]
        result["inhibitor_enabled"] = True

        # ------------------------------------------------------------
        # Checkpoint and return
        # ------------------------------------------------------------
        save_checkpoint({**state, **result})
        clear_gpu()

        return result

    except Exception as e:
        log.exception("INHIBITOR AGENT ERROR")
        clear_gpu()

        return _empty_inhibitor_update(
            analysis=f"Inhibitor agent error: {e}",
            summary=f"Inhibitor agent error: {e}",
            enabled=True,
        )