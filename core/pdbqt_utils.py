"""
pdbqt_utils.py

Legacy PDB -> PDBQT conversion helper using MGLTools.

This module is retained for backward compatibility with older Vina-based
workflows. The HDOCKlite migration does NOT require this module because
HDOCKlite consumes receptor/ligand PDB files directly.
"""

from __future__ import annotations

import os
import subprocess
import logging
from typing import Optional

log = logging.getLogger("virtual_lab.pdbqt_utils")


MGLTOOLS_ENV = os.getenv(
    "MGLTOOLS_ENV",
    "/users/fbsnpat/.conda/envs/mgltools-env",
)

PYTHONSH = os.path.join(MGLTOOLS_ENV, "bin", "pythonsh")
PREP_LIGAND = os.path.join(MGLTOOLS_ENV, "bin", "prepare_ligand4.py")


def _check_mgltools_ligand() -> bool:
    ok = True

    if not os.path.exists(PYTHONSH):
        log.warning("MGLTools pythonsh not found: %s", PYTHONSH)
        ok = False

    if not os.path.exists(PREP_LIGAND):
        log.warning("MGLTools prepare_ligand4.py not found: %s", PREP_LIGAND)
        ok = False

    return ok


def convert_to_pdbqt(input_pdb: str, output_pdbqt: str) -> Optional[str]:
    """
    Convert PDB -> PDBQT using MGLTools prepare_ligand4.py.

    This is retained for legacy Vina workflows only. HDOCKlite does not need
    PDBQT conversion.
    """

    if not input_pdb or not os.path.exists(input_pdb):
        log.warning("Input PDB missing for PDBQT conversion: %s", input_pdb)
        return None

    if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 1000:
        return output_pdbqt

    if not _check_mgltools_ligand():
        if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 1000:
            return output_pdbqt
        return None

    cmd = [
        PYTHONSH,
        PREP_LIGAND,
        "-l",
        input_pdb,
        "-o",
        output_pdbqt,
        "-A",
        "hydrogens",
    ]

    log.info("Converting ligand PDB -> PDBQT with MGLTools: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("MGLTOOLS_LIGAND_TIMEOUT", "120")),
        )

        if result.stdout:
            log.debug(
                "prepare_ligand4.py stdout: %s",
                result.stdout.decode(errors="ignore"),
            )

        if result.stderr:
            log.debug(
                "prepare_ligand4.py stderr: %s",
                result.stderr.decode(errors="ignore"),
            )

        if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 1000:
            return output_pdbqt

        log.warning("PDBQT output missing or too small: %s", output_pdbqt)
        return None

    except subprocess.CalledProcessError as e:
        log.warning(
            "prepare_ligand4.py failed with return code %s",
            e.returncode,
        )

        if e.stderr:
            log.warning("stderr: %s", e.stderr.decode(errors="ignore"))

    except subprocess.TimeoutExpired:
        log.warning("prepare_ligand4.py timed out for %s", input_pdb)

    except Exception as e:
        log.warning("PDBQT conversion failed for %s: %s", input_pdb, e)

    if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 1000:
        return output_pdbqt

    return None


# Alias for backward compatibility.
pdb_to_pdbqt = convert_to_pdbqt

