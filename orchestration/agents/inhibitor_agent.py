"""
inhibitor_agent.py — Small-molecule & peptide inhibitor screening agent.

Screens candidate inhibitors against the RNA-binding pocket identified from
the protein_agent's interface contacts. Coordinates:
  1. Pocket definition (RNA-binding site centroid + box dimensions)
  2. Small-molecule fetching from PubChem + AutoDock Vina docking
  3. Peptide design/fetching + HDOCK protein-protein docking
  4. Pose comparison with RNA-protein complex (CA-RMSD)
  5. Snapshot rendering via PyMOL

State inputs
------------
  target_pdb          : str  — path to validated protein PDB
  interface_contacts  : dict — from protein_agent
  rna_sequence        : str  — RNA sequence (optional)
  rna_structure       : str  — secondary structure (optional)
  inhibitor_enabled   : bool — gate; skip entirely if False

State outputs
-------------
  inhibitor_small_molecules     : list[dict]
  inhibitor_peptides            : list[dict]
  inhibitor_analysis            : str
  inhibitor_summary              : str
  inhibitor_binding_site_overlap : float
  inhibitor_docking_box          : dict
  inhibitor_snapshot_paths       : list[str]
  stage_outputs                 : list[dict]
  conversation_history           : list[dict]
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from VLAB2.core.inhibitor_docking import run_inhibitor_screen
from VLAB2.core.peptide_prep import (
    design_inhibitor_peptides,
    get_known_antiviral_peptides,
    prepare_peptides,
)
from VLAB2.core.rna_binding_pocket import define_docking_box
from VLAB2.core.small_molecule_prep import (
    fetch_and_prepare_compounds,
    fetch_known_rna_binding_inhibitors,
)
from VLAB2.orchestration.utils.checkpointing import save_checkpoint
from VLAB2.orchestration.utils.docking_visuals import (
    render_inhibitor_snapshots_for_results,
)
from VLAB2.utils.gpu import clear_gpu

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


def _find_executable(name: str) -> str | None:
    path = shutil.which(name)
    return path if path else None


def _is_valid_pdb(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.stat().st_size > 100


def record_stage_output(
    state: dict,
    stage: str,
    content: str,
    summary: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a stage-output record matching protein_agent's format."""
    return {
        "stage": stage,
        "content": content,
        "summary": summary or content[:200],
        "metadata": metadata or {},
    }


def add_conversation_entry(
    state: dict,
    role: str,
    content: str,
    agent: str,
) -> dict:
    """Build a conversation-history entry matching protein_agent's format."""
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

    # ------------------------------------------------------------
    # Gate: inhibitor screening must be enabled
    # ------------------------------------------------------------
    if not _env_bool("VLAB_INHIBITOR_ENABLED", default=True):
        log.info("Inhibitor screening disabled (VLAB_INHIBITOR_ENABLED != 1)")
        return {
            "inhibitor_enabled": False,
            "inhibitor_small_molecules": [],
            "inhibitor_peptides": [],
            "inhibitor_analysis": "Inhibitor screening disabled.",
            "inhibitor_summary": "Inhibitor screening disabled.",
            "inhibitor_binding_site_overlap": 0.0,
            "inhibitor_docking_box": {},
            "inhibitor_snapshot_paths": [],
            "stage_outputs": [],
            "conversation_history": [],
        }

    result: dict[str, Any] = {}

    try:
        # ------------------------------------------------------------
        # Validate required inputs
        # ------------------------------------------------------------
        target_pdb = state.get("target_pdb")
        interface_contacts = state.get("interface_contacts")

        if not _is_valid_pdb(target_pdb):
            log.warning("inhibitor_agent: target_pdb not available, skipping")
            return {
                "inhibitor_enabled": True,
                "inhibitor_small_molecules": [],
                "inhibitor_peptides": [],
                "inhibitor_analysis": "Skipped: no validated protein target.",
                "inhibitor_summary": "Skipped: no validated protein target.",
                "inhibitor_binding_site_overlap": 0.0,
                "inhibitor_docking_box": {},
                "inhibitor_snapshot_paths": [],
                "stage_outputs": [],
                "conversation_history": [],
            }

        if not interface_contacts:
            log.warning("inhibitor_agent: interface_contacts not available, skipping")
            return {
                "inhibitor_enabled": True,
                "inhibitor_small_molecules": [],
                "inhibitor_peptides": [],
                "inhibitor_analysis": "Skipped: no RNA-protein interface contacts.",
                "inhibitor_summary": "Skipped: no RNA-protein interface contacts.",
                "inhibitor_binding_site_overlap": 0.0,
                "inhibitor_docking_box": {},
                "inhibitor_snapshot_paths": [],
                "stage_outputs": [],
                "conversation_history": [],
            }

        log.info("inhibitor_agent: starting inhibitor screening for %s", target_pdb)

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

        # Start with curated known RNA-binding / antiviral compounds
        known_sm = fetch_known_rna_binding_inhibitors()
        log.info(
            "inhibitor_agent: %d known RNA-binding compounds available",
            len(known_sm),
        )

        # Optionally fetch from PubChem (user can pass compound names via env var)
        pubchem_queries = os.getenv("VLAB_PUBCHEM_QUERIES", "").strip()
        pubchem_compounds = []
        if pubchem_queries:
            try:
                pubchem_compounds = fetch_and_prepare_compounds(
                    pubchem_queries.split("|"),
                    max_results=max_small_mols,
                )
                log.info(
                    "inhibitor_agent: fetched %d PubChem compounds",
                    len(pubchem_compounds),
                )
            except Exception as e:
                log.warning("inhibitor_agent: PubChem fetch failed: %s", e)

        # Combine, deduplicate by SMILES, cap at max
        all_sm = {c.get("smiles", ""): c for c in known_sm + pubchem_compounds if c.get("smiles")}
        small_mol_list = list(all_sm.values())[:max_small_mols]

        log.info(
            "inhibitor_agent: %d small molecules for docking",
            len(small_mol_list),
        )

        # ------------------------------------------------------------
        # Step 3: Prepare peptides
        # ------------------------------------------------------------
        max_peptides = _env_int("VLAB_INHIBITOR_MAX_PEPTIDES", 5)

        # Known antiviral peptides
        known_pep = get_known_antiviral_peptides()
        log.info(
            "inhibitor_agent: %d known antiviral peptides available",
            len(known_pep) if (known_pep := known_pep) else 0,
        )

        # LLM-designed peptides (if LLM available)
        llm_designed_peptides = []
        try:
            llm_designed_peptides = design_inhibitor_peptides(
                pocket_residues=docking_box.get("pocket_residues", []),
                rna_sequence=state.get("rna_sequence"),
                rna_structure=state.get("rna_structure"),
                n_peptides=min(max_peptides, 3),
            )
            log.info(
                "inhibitor_agent: LLM designed %d peptide candidates",
                len(llm_designed_peptides),
            )
        except Exception as e:
            log.warning("inhibitor_agent: peptide design failed: %s", e)

        # Prepare all peptides for HDOCK
        all_pep_seqs = []
        for p in (known_pep or []) + llm_designed_peptides:
            seq = p.get("sequence", "")
            if seq and seq not in all_pep_seqs:
                all_pep_seqs.append(seq)

        peptide_list = []
        if all_pep_seqs:
            try:
                peptide_list = prepare_peptides(all_pep_seqs[:max_peptides])
                log.info(
                    "inhibitor_agent: %d peptides prepared for HDOCK",
                    len(peptide_list),
                )
            except Exception as e:
                log.warning("inhibitor_agent: peptide preparation failed: %s", e)

        # ------------------------------------------------------------
        # Step 4: Run combined inhibitor screen
        # ------------------------------------------------------------
        screen_result = run_inhibitor_screen(
            target_pdb=target_pdb,
            receptor_pdb=target_pdb,
            small_molecules=small_mol_list,
            peptides=peptide_list,
            docking_box=docking_box,
            interface_contacts=interface_contacts,
        )

        small_mol_results = screen_result.get("small_molecule_results", [])
        peptide_results = screen_result.get("peptide_results", [])
        binding_site_overlap = screen_result.get("binding_site_overlap", 0.0)
        analysis_text = screen_result.get("analysis", "")

        result["inhibitor_small_molecules"] = small_mol_results
        result["inhibitor_peptides"] = peptide_results
        result["inhibitor_binding_site_overlap"] = binding_site_overlap
        result["inhibitor_analysis"] = analysis_text

        # ------------------------------------------------------------
        # Step 5: Build summary
        # ------------------------------------------------------------
        n_valid_sm = sum(1 for r in small_mol_results if r.get("valid"))
        n_valid_pep = sum(1 for r in peptide_results if r.get("valid"))

        summary_parts = []
        if small_mol_results:
            summary_parts.append(
                f"{n_valid_sm}/{len(small_mol_results)} small molecules docked successfully."
            )
        if peptide_list:
            summary_parts.append(
                f"{n_valid_pep}/{len(peptide_results)} peptides docked successfully."
            )
        if binding_site_overlap > 0:
            summary_parts.append(
                f"Binding-site overlap with RNA interface: {binding_site_overlap:.1f}%."
            )

        if not summary_parts:
            summary_parts.append("No inhibitors docked successfully.")

        inhibitor_summary = " ".join(summary_parts)
        result["inhibitor_summary"] = inhibitor_summary

        # ------------------------------------------------------------
        # Step 6: Render PyMOL snapshots
        # ------------------------------------------------------------
        try:
            snapshot_state = {
                **state,
                **result,
                "target_pdb": target_pdb,
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
            summary=f"Inhibitor screen: {n_valid_sm} small molecules, {n_valid_pep} peptides",
            metadata={
                "n_small_molecules": len(small_mol_results),
                "n_peptides": len(peptide_results),
                "n_valid_small_molecules": n_valid_sm,
                "n_valid_peptides": n_valid_pep,
                "binding_site_overlap": binding_site_overlap,
                "docking_box": docking_box,
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

        return {
            "inhibitor_enabled": True,
            "inhibitor_small_molecules": [],
            "inhibitor_peptides": [],
            "inhibitor_analysis": f"Inhibitor agent error: {e}",
            "inhibitor_summary": f"Inhibitor agent error: {e}",
            "inhibitor_binding_site_overlap": 0.0,
            "inhibitor_docking_box": {},
            "inhibitor_snapshot_paths": [],
            "stage_outputs": [],
            "conversation_history": [],
        }