"""
protein_prep.py

Protein receptor preparation utilities.

HDOCKlite migration notes:
- HDOCKlite uses raw receptor PDB files directly.
- Vina legacy workflows still require receptor PDBQT files.
- Therefore prepare_protein() now always ensures the raw PDB exists, then
  optionally prepares/returns the PDBQT for backward compatibility.

Primary HDOCK path:
    ensure_protein_pdb("2HW8") -> /path/to/pdb_cache/2hw8.pdb

Legacy Vina path:
    prepare_protein("2HW8") -> /path/to/pdb_cache/2hw8.pdbqt or None

mmCIF support:
- If the legacy .pdb download fails, this module downloads .cif from RCSB.
- It then converts mmCIF to PDB using Bio.PDB when available.
- This helps with newer/large entries that are mmCIF-only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.request
from typing import Optional


log = logging.getLogger("virtual_lab.protein_prep")


MGLTOOLS_ENV = os.getenv(
    "MGLTOOLS_ENV",
    "/users/fbsnpat/.conda/envs/mgltools-env",
)

PYTHONSH = os.path.join(MGLTOOLS_ENV, "bin", "pythonsh")
PREP_RECEPTOR = os.path.join(MGLTOOLS_ENV, "bin", "prepare_receptor4.py")

CACHE_DIR = os.getenv(
    "PROTEIN_CACHE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pdb_cache")),
)

os.makedirs(CACHE_DIR, exist_ok=True)


def _normalise_pdb_id(pdb_id: str) -> Optional[str]:
    if not pdb_id:
        return None

    pdb_id = str(pdb_id).strip().lower()

    if len(pdb_id) != 4 or not pdb_id[0].isdigit():
        log.warning("Invalid PDB ID: %s", pdb_id)
        return None

    return pdb_id


def _pdb_path(pdb_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{pdb_id.lower()}.pdb")


def _cif_path(pdb_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{pdb_id.lower()}.cif")


def _pdbqt_path(pdb_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{pdb_id.lower()}.pdbqt")


def _file_ok(path: str, min_size: int = 1000) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > min_size


def _download_url(url: str, output_path: str, label: str) -> bool:
    """
    Download a file using urllib first, with wget fallback.

    Returns True if output_path exists and is non-trivial.
    """
    tmp_path = f"{output_path}.tmp"

    log.info("Downloading %s to %s", label, output_path)

    try:
        with urllib.request.urlopen(
            url,
            timeout=int(os.getenv("PDB_DOWNLOAD_TIMEOUT", "60")),
        ) as response:
            data = response.read()

        if not data or len(data) < 1000:
            log.warning("Downloaded data too small for %s", label)
            return False

        with open(tmp_path, "wb") as handle:
            handle.write(data)

        os.replace(tmp_path, output_path)
        return _file_ok(output_path)

    except Exception as e:
        log.warning("urllib download failed for %s: %s", label, e)

    try:
        result = subprocess.run(
            [
                "wget",
                url,
                "-O",
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("PDB_DOWNLOAD_TIMEOUT", "120")),
        )

        if result.stderr:
            log.debug("wget stderr for %s: %s", label, result.stderr.decode(errors="ignore"))

        if _file_ok(tmp_path):
            os.replace(tmp_path, output_path)
            return True

        log.warning("wget output missing or too small for %s", label)
        return False

    except Exception as e:
        log.warning("wget download failed for %s: %s", label, e)

    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    return False


def _download_pdb_rcsb(pdb_id: str, pdb_path: str) -> bool:
    """
    Download a legacy PDB file from RCSB.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    return _download_url(url, pdb_path, f"PDB {pdb_id.upper()}")


def _download_cif_rcsb(pdb_id: str, cif_path: str) -> bool:
    """
    Download an mmCIF file from RCSB.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    return _download_url(url, cif_path, f"mmCIF {pdb_id.upper()}")


def _convert_cif_to_pdb_biopython(cif_path: str, pdb_path: str, pdb_id: str) -> bool:
    """
    Convert mmCIF to PDB using Bio.PDB.

    Notes:
      - PDB format has legacy limitations for very large structures.
      - For HDOCKlite, a simplified ATOM/HETATM PDB is usually sufficient.
      - If Bio.PDB fails due to PDB format constraints, fallback writer is used.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBIO

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id.upper(), cif_path)

        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_path)

        if _file_ok(pdb_path):
            log.info("Converted mmCIF to PDB using Bio.PDB: %s", pdb_path)
            return True

        log.warning("Bio.PDB conversion produced missing/small PDB: %s", pdb_path)
        return False

    except Exception as e:
        log.warning("Bio.PDB mmCIF->PDB conversion failed for %s: %s", pdb_id.upper(), e)
        return False


def _convert_cif_to_pdb_fallback(cif_path: str, pdb_path: str, pdb_id: str) -> bool:
    """
    Minimal mmCIF to PDB fallback writer.

    This is intentionally conservative and writes ATOM/HETATM-like records from
    common _atom_site columns.

    It is not a full mmCIF implementation, but works for many standard RCSB
    entries when Bio.PDB is unavailable or fails.
    """
    try:
        with open(cif_path, errors="ignore") as f:
            lines = f.readlines()

        atom_headers = []
        atom_rows = []
        in_loop = False
        collecting_atom = False

        for raw in lines:
            line = raw.rstrip("\n")

            if line.strip() == "loop_":
                in_loop = True
                atom_headers = []
                atom_rows = []
                collecting_atom = False
                continue

            if in_loop and line.startswith("_atom_site."):
                atom_headers.append(line.strip())
                collecting_atom = True
                continue

            if collecting_atom:
                stripped = line.strip()

                if not stripped:
                    continue

                if stripped.startswith("#"):
                    break

                if stripped.startswith("_"):
                    break

                if stripped.startswith("ATOM") or stripped.startswith("HETATM"):
                    atom_rows.append(stripped)

        if not atom_headers or not atom_rows:
            log.warning(
                "Fallback mmCIF parser found no atom_site rows for %s",
                pdb_id.upper(),
            )
            return False

        header_to_idx = {
            h.replace("_atom_site.", ""): i
            for i, h in enumerate(atom_headers)
        }

        def idx(*names):
            for name in names:
                if name in header_to_idx:
                    return header_to_idx[name]
            return None

        i_group = idx("group_PDB")
        i_id = idx("id")
        i_atom = idx("label_atom_id", "auth_atom_id")
        i_alt = idx("label_alt_id")
        i_res = idx("label_comp_id", "auth_comp_id")
        i_chain = idx("auth_asym_id", "label_asym_id")
        i_seq = idx("auth_seq_id", "label_seq_id")
        i_x = idx("Cartn_x")
        i_y = idx("Cartn_y")
        i_z = idx("Cartn_z")
        i_occ = idx("occupancy")
        i_b = idx("B_iso_or_equiv")
        i_elem = idx("type_symbol")

        required = [i_group, i_atom, i_res, i_chain, i_seq, i_x, i_y, i_z]

        if any(x is None for x in required):
            log.warning(
                "Fallback mmCIF parser missing required atom_site columns for %s",
                pdb_id.upper(),
            )
            return False

        def clean_token(tok: str) -> str:
            tok = tok.strip()
            if tok in {".", "?"}:
                return ""
            return tok.strip("'").strip('"')

        wrote = 0

        with open(pdb_path, "w") as out:
            out.write(f"HEADER    CONVERTED FROM MMCIF {pdb_id.upper()}\n")

            for n, row in enumerate(atom_rows, start=1):
                # Most RCSB atom_site rows are whitespace separated.
                # This fallback will not perfectly handle quoted tokens with spaces,
                # but atom rows usually do not require those for core coordinates.
                parts = row.split()

                try:
                    record = clean_token(parts[i_group])[:6]
                    serial = (
                        int(float(clean_token(parts[i_id])))
                        if i_id is not None and clean_token(parts[i_id])
                        else n
                    )
                    atom = clean_token(parts[i_atom])[:4]
                    alt = clean_token(parts[i_alt])[:1] if i_alt is not None else ""
                    res = clean_token(parts[i_res])[:3]
                    chain = clean_token(parts[i_chain])[:1] or "A"
                    seq = int(float(clean_token(parts[i_seq])))
                    x = float(clean_token(parts[i_x]))
                    y = float(clean_token(parts[i_y]))
                    z = float(clean_token(parts[i_z]))

                    occ = (
                        float(clean_token(parts[i_occ]))
                        if i_occ is not None and clean_token(parts[i_occ])
                        else 1.00
                    )

                    bfac = (
                        float(clean_token(parts[i_b]))
                        if i_b is not None and clean_token(parts[i_b])
                        else 0.00
                    )

                    elem = (
                        clean_token(parts[i_elem])[:2].rjust(2)
                        if i_elem is not None and clean_token(parts[i_elem])
                        else atom[0].rjust(2)
                    )

                    if record not in {"ATOM", "HETATM"}:
                        continue

                    out.write(
                        f"{record:<6}{serial:>5d} "
                        f"{atom:<4}{alt:1}{res:>3} {chain:1}"
                        f"{seq:>4d}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                        f"{occ:>6.2f}{bfac:>6.2f}          "
                        f"{elem:>2}\n"
                    )

                    wrote += 1

                except Exception:
                    continue

            out.write("END\n")

        if wrote > 0 and _file_ok(pdb_path):
            log.info(
                "Converted mmCIF to PDB using fallback writer: %s atoms=%d",
                pdb_path,
                wrote,
            )
            return True

        log.warning(
            "Fallback mmCIF conversion wrote no valid atoms for %s",
            pdb_id.upper(),
        )
        return False

    except Exception as e:
        log.warning(
            "Fallback mmCIF->PDB conversion failed for %s: %s",
            pdb_id.upper(),
            e,
        )
        return False
    
def _convert_cif_to_pdb(cif_path: str, pdb_path: str, pdb_id: str) -> bool:
    """
    Convert mmCIF to PDB.

    Preference:
      1. Bio.PDB
      2. Minimal fallback writer
    """
    if _convert_cif_to_pdb_biopython(cif_path, pdb_path, pdb_id):
        return True

    return _convert_cif_to_pdb_fallback(cif_path, pdb_path, pdb_id)


def ensure_protein_cif(pdb_id: str) -> Optional[str]:
    """
    Ensure mmCIF receptor file is present in CACHE_DIR.

    Returns:
        Path to cached mmCIF, or None.
    """
    pdb_id = _normalise_pdb_id(pdb_id)

    if not pdb_id:
        return None

    cif_path = _cif_path(pdb_id)

    if _file_ok(cif_path):
        return cif_path

    if _download_cif_rcsb(pdb_id, cif_path) and _file_ok(cif_path):
        return cif_path

    log.warning("Could not ensure receptor mmCIF for %s", pdb_id.upper())
    return None


def ensure_protein_pdb(pdb_id: str) -> Optional[str]:
    """
    Ensure raw receptor PDB is present in CACHE_DIR.

    Canonical HDOCKlite preparation function.

    Resolution order:
      1. Existing cached .pdb
      2. Download legacy .pdb from RCSB
      3. Existing/downloaded .cif converted to .pdb

    Returns:
        Path to cached raw PDB, or None.
    """
    pdb_id = _normalise_pdb_id(pdb_id)

    if not pdb_id:
        return None

    pdb_path = _pdb_path(pdb_id)

    if _file_ok(pdb_path):
        return pdb_path

    # First try classic PDB.
    if _download_pdb_rcsb(pdb_id, pdb_path) and _file_ok(pdb_path):
        return pdb_path

    # Fallback to mmCIF.
    cif_path = ensure_protein_cif(pdb_id)

    if cif_path and _convert_cif_to_pdb(cif_path, pdb_path, pdb_id):
        if _file_ok(pdb_path):
            return pdb_path

    log.warning("Could not ensure receptor PDB for %s", pdb_id.upper())
    return None


def _check_mgltools() -> bool:
    """
    Validate isolated MGLTools executables for legacy PDBQT receptor prep.
    """
    ok = True

    if not os.path.exists(PYTHONSH):
        log.warning("MGLTools pythonsh not found: %s", PYTHONSH)
        ok = False

    if not os.path.exists(PREP_RECEPTOR):
        log.warning("MGLTools prepare_receptor4.py not found: %s", PREP_RECEPTOR)
        ok = False

    return ok


def _clean_receptor_pdb_for_mgltools(input_pdb: str, output_pdb: str) -> Optional[str]:
    """
    Create a minimal receptor PDB for MGLTools.

    Keeps only ATOM records and drops waters/ligands. This is safer for
    prepare_receptor4.py while leaving the original cached raw PDB untouched.
    """
    try:
        wrote = False

        with open(input_pdb, errors="ignore") as fin, open(output_pdb, "w") as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    fout.write(line)
                    wrote = True

            if wrote:
                fout.write("END\n")

        if wrote and _file_ok(output_pdb, min_size=500):
            return output_pdb

        return None

    except Exception as e:
        log.warning("Failed to clean receptor PDB %s: %s", input_pdb, e)
        return None


def prepare_protein(pdb_id: str) -> Optional[str]:
    """
    Legacy-compatible receptor preparation.

    Ensures the raw PDB exists, then prepares receptor PDBQT using MGLTools.

    Returns:
        Path to cached receptor PDBQT, or None if PDBQT preparation failed.

    Important:
        HDOCKlite callers should use ensure_protein_pdb() directly.
        This function remains for legacy Vina code paths.
    """
    pdb_id_norm = _normalise_pdb_id(pdb_id)

    if not pdb_id_norm:
        return None

    pdb_path = ensure_protein_pdb(pdb_id_norm)

    if not pdb_path:
        return None

    pdbqt_path = _pdbqt_path(pdb_id_norm)

    if _file_ok(pdbqt_path):
        return pdbqt_path

    if not _check_mgltools():
        log.warning(
            "MGLTools unavailable; raw PDB exists for HDOCK but PDBQT cannot be prepared: %s",
            pdb_path,
        )
        return None

    clean_pdb = os.path.join(CACHE_DIR, f"{pdb_id_norm}_receptor_clean.pdb")
    clean_pdb = _clean_receptor_pdb_for_mgltools(pdb_path, clean_pdb) or pdb_path

    cmd = [
        PYTHONSH,
        PREP_RECEPTOR,
        "-r",
        clean_pdb,
        "-o",
        pdbqt_path,
    ]

    log.info("Preparing receptor PDBQT with isolated MGLTools: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(os.getenv("MGLTOOLS_RECEPTOR_TIMEOUT", "120")),
        )

        if result.stdout:
            log.debug(
                "prepare_receptor4.py stdout: %s",
                result.stdout.decode(errors="ignore"),
            )

        if result.stderr:
            log.debug(
                "prepare_receptor4.py stderr: %s",
                result.stderr.decode(errors="ignore"),
            )

        if _file_ok(pdbqt_path):
            return pdbqt_path

        log.warning("PDBQT was not created or is too small: %s", pdbqt_path)
        return None

    except subprocess.CalledProcessError as e:
        log.warning(
            "Protein prep command failed for %s with return code %s",
            pdb_id_norm,
            e.returncode,
        )

        if e.stderr:
            log.warning("stderr: %s", e.stderr.decode(errors="ignore"))

    except subprocess.TimeoutExpired:
        log.warning("Protein prep timed out for %s", pdb_id_norm)

    except Exception as e:
        log.warning("Protein prep failed for %s: %s", pdb_id_norm, e)

    if _file_ok(pdbqt_path):
        return pdbqt_path

    return None