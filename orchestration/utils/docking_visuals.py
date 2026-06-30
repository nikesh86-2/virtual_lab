from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path


log = logging.getLogger("virtual_lab")


# ---------------------------------------------------------------------
# Utility: find executables
# ---------------------------------------------------------------------
def _find_executable(name: str) -> str | None:
    path = shutil.which(name)
    return path if path else None


# ---------------------------------------------------------------------
# Single snapshot renderer (PyMOL headless)
# ---------------------------------------------------------------------
def render_docking_snapshot_pymol(
    complex_pdb: str,
    output_png: str,
    title: str | None = None,
    width: int = 1600,
    height: int = 1200,
) -> dict:
    """
    Render a docking complex to PNG using PyMOL (headless).

    Returns dict:
      valid: bool
      image_file: str | None
      method: "pymol"
      error: str | None
    """

    pymol_bin = (
        os.getenv("PYMOL_BIN")
        or _find_executable("pymol")
        or _find_executable("pymol-open-source")
    )

    if not pymol_bin:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": "PyMOL not found",
        }

    complex_path = Path(complex_pdb)

    if not complex_path.exists():
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": f"Missing PDB: {complex_pdb}",
        }

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pml_path = output_path.with_suffix(".pml")

    title_text = title or complex_path.stem

    pml = f"""
reinitialize
load {complex_path.as_posix()}, complex

hide everything
bg_color white

set ray_opaque_background, off
set antialias, 2
set depth_cue, 0
set orthoscopic, on

# protein
select protein_part, polymer.protein
show cartoon, protein_part
color slate, protein_part

# RNA
select rna_part, polymer.nucleic
show sticks, rna_part
color orange, rna_part

# interface highlight
select interface_protein, protein_part within 5 of rna_part
show surface, interface_protein
color marine, interface_protein
set transparency, 0.4, interface_protein

select interface_rna, rna_part within 5 of protein_part
show spheres, interface_rna
color tv_orange, interface_rna
set sphere_scale, 0.35, interface_rna

orient
zoom all, 8
turn x, 20
turn y, -25

pseudoatom title_label, pos=[0,0,0], label="{title_text}"
hide spheres, title_label
set label_color, black
set label_size, 18

ray {width}, {height}
png {output_path.as_posix()}, dpi=220

quit
"""

    pml_path.write_text(pml, encoding="utf-8")

    try:
        proc = subprocess.run(
            [pymol_bin, "-cq", str(pml_path)],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VLAB_RENDER_TIMEOUT", "180")),
        )

        if proc.returncode != 0:
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": proc.stderr[-1000:],
            }

        if not output_path.exists():
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": "PNG not generated",
            }

        return {
            "valid": True,
            "image_file": str(output_path),
            "method": "pymol",
            "error": None,
        }

    except Exception as e:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": str(e),
        }


# ---------------------------------------------------------------------
# Batch rendering for pipeline
# ---------------------------------------------------------------------
def render_docking_snapshots_for_results(state: dict) -> dict:
    """
    Generate snapshots for all docking-valid results.
    """

    if os.getenv("VLAB_RENDER_DOCKING_SNAPSHOTS", "0").strip() != "1":
        return {}

    out_dir = Path(
        os.getenv(
            "VLAB_DOCKING_SNAPSHOT_DIR",
            "output_data/docking_snapshots",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_results = []
    image_files = []

    for idx, r in enumerate(state.get("binding_results", []) or []):
        if not isinstance(r, dict):
            updated_results.append(r)
            continue

        copied = dict(r)

        complex_file = (
            copied.get("dock_complex_file")
            or copied.get("complex_file")
        )

        if not copied.get("dock_valid") or not complex_file:
            updated_results.append(copied)
            continue

        target = copied.get("target_pdb") or "target"
        seq = copied.get("sequence", "")
        rank = copied.get("rank", idx + 1)
        score = copied.get("dock_score")

        seq_tag = seq[:12] if seq else f"rank{rank}"

        png = out_dir / f"{target}_rank{rank}_{seq_tag}_score{score}.png"

        title = f"{target} | rank {rank} | score {score}"

        res = render_docking_snapshot_pymol(
            complex_pdb=complex_file,
            output_png=str(png),
            title=title,
        )

        copied["docking_snapshot_valid"] = res.get("valid")
        copied["docking_snapshot_png"] = res.get("image_file")
        copied["docking_snapshot_error"] = res.get("error")

        if res.get("valid"):
            image_files.append(res["image_file"])

        updated_results.append(copied)

    log.info("Rendered %d docking snapshots", len(image_files))

    return {
        "binding_results": updated_results,
        "docking_snapshot_files": image_files,
    }


# ---------------------------------------------------------------------
# Small-molecule snapshot renderer (PyMOL headless)
# ---------------------------------------------------------------------
def render_small_molecule_snapshot_pymol(
    complex_pdb: str,
    output_png: str,
    title: str | None = None,
    width: int = 1600,
    height: int = 1200,
) -> dict:
    """
    Render a protein + small-molecule complex to PNG using PyMOL (headless).

    Coloring scheme:
      protein   : cartoon / slate
      small mol : sticks / yellow
      interface : surface / marine / transparent

    Returns dict:
      valid: bool
      image_file: str | None
      method: "pymol"
      error: str | None
    """

    pymol_bin = (
        os.getenv("PYMOL_BIN")
        or _find_executable("pymol")
        or _find_executable("pymol-open-source")
    )

    if not pymol_bin:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": "PyMOL not found",
        }

    complex_path = Path(complex_pdb)

    if not complex_path.exists():
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": f"Missing PDB: {complex_pdb}",
        }

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pml_path = output_path.with_suffix(".pml")

    title_text = title or complex_path.stem

    pml = f"""
reinitialize
load {complex_path.as_posix()}, complex

hide everything
bg_color white

set ray_opaque_background, off
set antialias, 2
set depth_cue, 0
set orthoscopic, on

# protein
select protein_part, polymer.protein
show cartoon, protein_part
color slate, protein_part

# small molecule (non-polymer, non-water)
select small_mol, not polymer and not resn HOH
show sticks, small_mol
color yellow, small_mol

# interface highlight
select interface_protein, protein_part within 5 of small_mol
show surface, interface_protein
color marine, interface_protein
set transparency, 0.4, interface_protein

orient
zoom all, 10
turn x, 20
turn y, -25

pseudoatom title_label, pos=[0,0,0], label="{title_text}"
hide spheres, title_label
set label_color, black
set label_size, 18

ray {width}, {height}
png {output_path.as_posix()}, dpi=220

quit
"""

    pml_path.write_text(pml, encoding="utf-8")

    try:
        proc = subprocess.run(
            [pymol_bin, "-cq", str(pml_path)],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VLAB_RENDER_TIMEOUT", "180")),
        )

        if proc.returncode != 0:
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": proc.stderr[-1000:],
            }

        if not output_path.exists():
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": "PNG not generated",
            }

        return {
            "valid": True,
            "image_file": str(output_path),
            "method": "pymol",
            "error": None,
        }

    except Exception as e:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": str(e),
        }


# ---------------------------------------------------------------------
# Peptide snapshot renderer (PyMOL headless)
# ---------------------------------------------------------------------
def render_peptide_snapshot_pymol(
    complex_pdb: str,
    output_png: str,
    title: str | None = None,
    width: int = 1600,
    height: int = 1200,
) -> dict:
    """
    Render a protein + peptide complex to PNG using PyMOL (headless).

    Coloring scheme:
      protein  : cartoon / slate
      peptide  : cartoon + sticks / magenta
      interface: surface / marine / transparent

    Returns dict:
      valid: bool
      image_file: str | None
      method: "pymol"
      error: str | None
    """

    pymol_bin = (
        os.getenv("PYMOL_BIN")
        or _find_executable("pymol")
        or _find_executable("pymol-open-source")
    )

    if not pymol_bin:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": "PyMOL not found",
        }

    complex_path = Path(complex_pdb)

    if not complex_path.exists():
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": f"Missing PDB: {complex_pdb}",
        }

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pml_path = output_path.with_suffix(".pml")

    title_text = title or complex_path.stem

    pml = f"""
reinitialize
load {complex_path.as_posix()}, complex

hide everything
bg_color white

set ray_opaque_background, off
set antialias, 2
set depth_cue, 0
set orthoscopic, on

# protein
select protein_part, polymer.protein
show cartoon, protein_part
color slate, protein_part

# peptide (second protein chain or all protein within 5 of first)
select peptide_part, polymer.protein and not chain A
show cartoon, peptide_part
color magenta, peptide_part
show sticks, peptide_part

# interface highlight
select interface_protein, protein_part within 5 of peptide_part
show surface, interface_protein
color marine, interface_protein
set transparency, 0.4, interface_protein

orient
zoom all, 10
turn x, 20
turn y, -25

pseudoatom title_label, pos=[0,0,0], label="{title_text}"
hide spheres, title_label
set label_color, black
set label_size, 18

ray {width}, {height}
png {output_path.as_posix()}, dpi=220

quit
"""

    pml_path.write_text(pml, encoding="utf-8")

    try:
        proc = subprocess.run(
            [pymol_bin, "-cq", str(pml_path)],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("VLAB_RENDER_TIMEOUT", "180")),
        )

        if proc.returncode != 0:
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": proc.stderr[-1000:],
            }

        if not output_path.exists():
            return {
                "valid": False,
                "image_file": None,
                "method": "pymol",
                "error": "PNG not generated",
            }

        return {
            "valid": True,
            "image_file": str(output_path),
            "method": "pymol",
            "error": None,
        }

    except Exception as e:
        return {
            "valid": False,
            "image_file": None,
            "method": "pymol",
            "error": str(e),
        }


# ---------------------------------------------------------------------
# Batch rendering for inhibitor pipeline
# ---------------------------------------------------------------------
def render_inhibitor_snapshots_for_results(state: dict) -> dict:
    """
    Generate PyMOL snapshots for all inhibitor docking results.

    Small molecules: yellow sticks
    Peptides       : magenta cartoon + sticks

    Returns dict:
      inhibitor_snapshot_paths : list[str]
    """

    if os.getenv("VLAB_RENDER_DOCKING_SNAPSHOTS", "0").strip() != "1":
        return {}

    out_dir = Path(
        os.getenv(
            "VLAB_DOCKING_SNAPSHOT_DIR",
            "output_data/docking_snapshots",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = []

    # ------------------------------------------------------------
    # Small-molecule snapshots
    # ------------------------------------------------------------
    for idx, r in enumerate(state.get("inhibitor_small_molecules") or []):
        if not isinstance(r, dict):
            continue

        complex_file = r.get("output_file") or r.get("complex_pdb")
        if not r.get("valid") or not complex_file:
            continue

        name = r.get("name", r.get("compound_name", f"sm_{idx}")).replace(" ", "_")
        score = r.get("binding_energy", r.get("score", "na"))

        png = out_dir / f"sm_{name}_score{score}.png"

        res = render_small_molecule_snapshot_pymol(
            complex_pdb=complex_file,
            output_png=str(png),
            title=f"SM: {name} | score {score}",
        )

        r["snapshot_valid"] = res.get("valid")
        r["snapshot_png"] = res.get("image_file")

        if res.get("valid"):
            image_files.append(res["image_file"])

    # ------------------------------------------------------------
    # Peptide snapshots
    # ------------------------------------------------------------
    for idx, r in enumerate(state.get("inhibitor_peptides") or []):
        if not isinstance(r, dict):
            continue

        complex_file = r.get("complex_pdb") or r.get("output_complex")
        if not r.get("valid") or not complex_file:
            continue

        seq = r.get("sequence", f"pep_{idx}")[:12]
        score = r.get("score", r.get("hdock_score", "na"))

        png = out_dir / f"pep_{seq}_score{score}.png"

        res = render_peptide_snapshot_pymol(
            complex_pdb=complex_file,
            output_png=str(png),
            title=f"Peptide: {seq} | score {score}",
        )

        r["snapshot_valid"] = res.get("valid")
        r["snapshot_png"] = res.get("image_file")

        if res.get("valid"):
            image_files.append(res["image_file"])

    log.info("Rendered %d inhibitor snapshots", len(image_files))

    return {"inhibitor_snapshot_paths": image_files}