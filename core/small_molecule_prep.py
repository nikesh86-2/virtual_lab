"""
small_molecule_prep.py

Utilities for fetching small-molecule compounds from PubChem and preparing
them as PDBQT files for AutoDock Vina docking against the RNA-binding pocket.

Key functions
-------------
fetch_pubchem_compounds(query, max_compounds)
    Search PubChem by name / substructure and return metadata + SMILES.

prepare_small_molecule_pdbqt(smiles, output_dir)
    Convert a SMILES string → PDBQT via OpenBabel.

fetch_known_rna_binding_inhibitors()
    Return a curated list of known RNA-binding-pocket or antiviral small
    molecules (ribavirin, remdesivir, tilorone analogues, etc.) whose
    structures are fetched from PubChem on demand.

cache_key(smiles)
    Deterministic cache filename derived from the SMILES hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

from VLAB2.core.rna_prep import resolve_obabel

log = logging.getLogger("virtual_lab.small_molecule_prep")

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

DEFAULT_OBABEL_BIN = (
    "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/obabel"
)

INHIBITOR_CACHE_DIR = os.getenv(
    "VLAB_INHIBITOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "vlab_inhibitor_cache"),
)
os.makedirs(INHIBITOR_CACHE_DIR, exist_ok=True)

VLAB_INHIBITOR_MAX_SMALL_MOLECULES = int(
    os.getenv("VLAB_INHIBITOR_MAX_SMALL_MOLECULES", "10")
)

# ---------------------------------------------------------------------------
# PubChem REST helpers
# ---------------------------------------------------------------------------

PUBCCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCCHEM_VIEW = f"{PUBCCHEM_REST}/compound"
PUBCCHEM_SEARCH = f"{PUBCCHEM_REST}/compound/name"
PUBCCHEM_PROPERTY = "IsomericSMILES,CanonicalSMILES,Title,MolecularFormula,MolecularWeight"

PUBCCHEM_TIMEOUT = 20  # seconds per request


def _pubchem_get(url: str, params: dict | None = None) -> dict:
    """GET a PubChem REST endpoint and return parsed JSON."""
    try:
        resp = requests.get(url, params=params, timeout=PUBCCHEM_TIMEOUT, headers={
            "Accept": "application/json",
        })
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("PubChem request failed for %s: %s", url, exc)
        return {}


def _smiles_to_cache_key(smiles: str) -> str:
    """Deterministic cache filename from SMILES."""
    h = hashlib.sha256(smiles.encode()).hexdigest()[:16]
    return f"smi_{h}"


# ---------------------------------------------------------------------------
# Compound fetching
# ---------------------------------------------------------------------------

def fetch_pubchem_compounds(
    query: str,
    max_compounds: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search PubChem by compound name and return metadata.

    Parameters
    ----------
    query : str
        Compound name (e.g. "ribavirin", "remdesivir").
    max_compounds : int, optional
        Cap on number of results. Defaults to VLAB_INHIBITOR_MAX_SMALL_MOLECULES.

    Returns
    -------
    list[dict]
        Each entry contains: ``cid``, ``name``, ``smiles``, ``formula``,
        ``mw``, ``pdbqt_path`` (set later by prepare_small_molecule_pdbqt).
    """
    if max_compounds is None:
        max_compounds = VLAB_INHIBITOR_MAX_SMALL_MOLECULES

    url = f"{PUBCCHEM_SEARCH}/{requests.utils.quote(query)}/property/{PUBCCHEM_PROPERTY}/JSON"
    data = _pubchem_get(url)
    if not data:
        log.warning("No PubChem results for query '%s'", query)
        return []

    props = data.get("PropertyTable", {}).get("Properties", [])
    results = []
    for entry in props[:max_compounds]:
        cid = entry.get("CID")
        smiles = entry.get("IsomericSMILES") or entry.get("CanonicalSMILES", "")
        results.append({
            "cid": cid,
            "name": entry.get("Title") or query,
            "smiles": smiles,
            "formula": entry.get("MolecularFormula", ""),
            "mw": entry.get("MolecularWeight", 0.0),
            "pdbqt_path": None,
        })
        log.debug("  PubChem CID %s: %s (MW=%.2f)", cid, entry.get("Title", ""), entry.get("MolecularWeight", 0))

    return results


def fetch_compound_by_cid(cid: int | str) -> dict[str, Any] | None:
    """Fetch a single compound by PubChem CID."""
    url = f"{PUBCCHEM_VIEW}/cid/{cid}/property/{PUBCCHEM_PROPERTY}/JSON"
    data = _pubchem_get(url)
    if not data:
        return None
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    entry = props[0]
    return {
        "cid": entry.get("CID"),
        "name": entry.get("Title") or f"CID{cid}",
        "smiles": entry.get("IsomericSMILES") or entry.get("CanonicalSMILES", ""),
        "formula": entry.get("MolecularFormula", ""),
        "mw": entry.get("MolecularWeight", 0.0),
        "pdbqt_path": None,
    }


# ---------------------------------------------------------------------------
# PDBQT preparation via OpenBabel
# ---------------------------------------------------------------------------

def prepare_small_molecule_pdbqt(
    smiles: str,
    output_dir: str | None = None,
    name: str | None = None,
    add_hydrogens: bool = True,
) -> str | None:
    """
    Convert a SMILES string to a PDBQT file using OpenBabel.

    Parameters
    ----------
    smiles : str
        Isomeric or canonical SMILES for the compound.
    output_dir : str, optional
        Directory to write the PDBQT file. Defaults to INHIBITOR_CACHE_DIR.
    name : str, optional
        Stem for the output file. Auto-derived from SMILES hash if omitted.
    add_hydrogens : bool
        Whether to add hydrogens via OpenBabel (default True).

    Returns
    -------
    str
        Absolute path to the generated PDBQT file, or None on failure.
    """
    if output_dir is None:
        output_dir = INHIBITOR_CACHE_DIR
    os.makedirs(output_dir, exist_ok=True)

    cache_key = _smiles_to_cache_key(smiles)
    if name:
        stem = name.replace(" ", "_").replace("/", "_")[:40]
    else:
        stem = cache_key

    pdbqt_path = os.path.join(output_dir, f"{stem}.pdbqt")

    # Return cached file if it already exists
    if os.path.exists(pdbqt_path):
        log.info("Using cached PDBQT for %s: %s", stem, pdbqt_path)
        return pdbqt_path

    # Write SMILES to a temp file for obabel stdin
    tmp_smi = os.path.join(output_dir, f"{stem}.smi")
    with open(tmp_smi, "w") as fh:
        fh.write(smiles)

    obabel_bin = resolve_obabel()

    # Build obabel command: SMILES → PDB → PDBQT
    cmd = [
        obabel_bin,
        "-ismi", tmp_smi,
        "-opdb",
        "--gen3d",          # generate 3D coordinates
    ]
    if add_hydrogens:
        cmd += ["--hydrogens"]   # add hydrogens

    cmd += ["-O", pdbqt_path]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        log.debug("obabel stdout:\n%s", result.stdout)
        if result.stderr:
            log.debug("obabel stderr:\n%s", result.stderr)
    except subprocess.CalledProcessError as exc:
        log.error("obabel failed for %s: %s\nstderr: %s", stem, exc, exc.stderr)
        return None
    except subprocess.TimeoutExpired:
        log.error("obabel timed out for %s", stem)
        return None
    finally:
        # Clean up temp SMILES file
        if os.path.exists(tmp_smi):
            os.remove(tmp_smi)

    if not os.path.exists(pdbqt_path):
        log.error("obabel produced no output file for %s", stem)
        return None

    log.info("Prepared PDBQT: %s", pdbqt_path)
    return pdbqt_path


# ---------------------------------------------------------------------------
# Curated list of known RNA-binding / antiviral small molecules
# ---------------------------------------------------------------------------

# Each entry: (display_name, pubchem_query_or_cid)
#   - If int/str digits → treated as PubChem CID
#   - Otherwise        → treated as a name search query
_KNOWN_INHIBITORS: list[tuple[str, str | int]] = [
    # Nucleoside analogues targeting viral RNA polymerases / RNA pockets
    ("Ribavirin", "Ribavirin"),
    ("Remdesivir", "Remdesivir"),
    ("Favipiravir", "Favipiravir"),
    ("Molnupiravir", "Molnupiravir"),
    ("Galidesivir", "Galidesivir"),
    # Viral entry / fusion inhibitors (known RNA-binding or cationic)
    ("Tilorone", "Tilorone"),
    ("Amiodarone", "Amiodarone"),
    # Broad-spectrum host-directed antivirals with known RNA interaction
    ("Niclosamide", "Niclosamide"),
    ("Mefloquine", "Mefloquine"),
    # Topoisomerase / nucleic-acid binders that occupy RNA pockets
    ("Doxorubicin", "Doxorubicin"),
    ("Etoposide", "Etoposide"),
    # Cationic amphiphiles that disrupt RNA-protein interactions
    ("Chloroquine", "Chloroquine"),
    ("Hydroxychloroquine", "Hydroxychloroquine"),
    # HDAC inhibitors with reported RNA-binding activity
    ("Panobinostat", "Panobinostat"),
    ("Romidepsin", "Romidepsin"),
    # Natural-product RNA binders
    ("Berbamine", "Berbamine"),
    ("Matrine", "Matrine"),
    # Broad antiviral / host-targeting
    ("Ivermectin", "Ivermectin"),
    ("Azithromycin", "Azithromycin"),
    # Additional well-characterised RNA-protein interface binders
    ("Suramin", "Suramin"),
    ("Clemizole", "Clemizole"),
]


def fetch_known_rna_binding_inhibitors(
    max_compounds: int | None = None,
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch a curated list of known RNA-binding / antiviral small molecules
    from PubChem and prepare them as PDBQT files.

    Parameters
    ----------
    max_compounds : int, optional
        Maximum total compounds to return. Defaults to
        VLAB_INHIBITOR_MAX_SMALL_MOLECULES.
    output_dir : str, optional
        Directory for PDBQT files. Defaults to INHIBITOR_CACHE_DIR.

    Returns
    -------
    list[dict]
        List of compound dicts (same schema as fetch_pubchem_compounds) with
        ``pdbqt_path`` populated.
    """
    if max_compounds is None:
        max_compounds = VLAB_INHIBITOR_MAX_SMALL_MOLECULES

    if output_dir is None:
        output_dir = INHIBITOR_CACHE_DIR

    compounds: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()

    for display_name, query in _KNOWN_INHIBITORS[:max_compounds]:
        # Resolve CID vs name query
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            comp = fetch_compound_by_cid(int(query))
        else:
            hits = fetch_pubchem_compounds(query, max_compounds=3)
            if not hits:
                log.warning("Could not fetch PubChem entry for '%s' (%s)", display_name, query)
                continue
            # Pick the first hit that matches the query reasonably
            comp = hits[0]

        if not comp or not comp.get("smiles"):
            log.warning("No SMILES for '%s', skipping", display_name)
            continue

        # Deduplicate by exact SMILES
        if comp["smiles"] in seen_smiles:
            log.debug("Duplicate SMILES for '%s', skipping", display_name)
            continue
        seen_smiles.add(comp["smiles"])

        # Prepare PDBQT
        pdbqt_path = prepare_small_molecule_pdbqt(
            comp["smiles"],
            output_dir=output_dir,
            name=display_name,
        )
        comp["pdbqt_path"] = pdbqt_path
        comp["display_name"] = display_name

        if pdbqt_path:
            compounds.append(comp)
            log.info("Prepared known inhibitor: %s (CID=%s, MW=%.2f, PDBQT=%s)",
                     display_name, comp.get("cid"), comp.get("mw", 0), pdbqt_path)
        else:
            log.warning("Failed to prepare PDBQT for '%s', skipping", display_name)

    log.info("fetch_known_rna_binding_inhibitors: %d compounds prepared", len(compounds))
    return compounds


# ---------------------------------------------------------------------------
# Convenience: fetch + prepare in one call
# ---------------------------------------------------------------------------

def fetch_and_prepare_compounds(
    queries: list[str],
    max_per_query: int = 3,
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Given a list of compound name queries, fetch from PubChem and prepare
    PDBQT files for each.

    Parameters
    ----------
    queries : list[str]
        List of compound names to search PubChem for.
    max_per_query : int
        Max hits per query.
    output_dir : str, optional
        Output directory for PDBQT files.

    Returns
    -------
    list[dict]
        Compound dicts with ``pdbqt_path`` populated.
    """
    if output_dir is None:
        output_dir = INHIBITOR_CACHE_DIR

    compounds: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()

    for q in queries:
        hits = fetch_pubchem_compounds(q, max_compounds=max_per_query)
        for comp in hits:
            smiles = comp.get("smiles", "")
            if not smiles or smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            pdbqt = prepare_small_molecule_pdbqt(smiles, output_dir=output_dir, name=comp.get("name"))
            comp["pdbqt_path"] = pdbqt
            if pdbqt:
                compounds.append(comp)

    return compounds