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