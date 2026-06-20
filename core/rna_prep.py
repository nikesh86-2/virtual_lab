"""
rna_prep.py

RNA preparation utilities.

HDOCKlite migration notes:
- HDOCKlite consumes RNA PDB directly.
- prepare_rna_pdb_for_hdock() provides a clean PDB for HDOCKlite.
- prepare_rna_ligand() remains for legacy Vina/PDBQT workflows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import logging
import hashlib
from pathlib import Path

log = logging.getLogger("virtual_lab.rna_prep")


RNA_PREP_DEBUG_DIR = os.getenv("RNA_PREP_DEBUG_DIR")

DEFAULT_OBABEL_BIN = (
    "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/obabel"
)

RNA_HDOCK_CACHE_DIR = os.getenv(
    "RNA_HDOCK_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "vlab_rna_hdock_cache"),
)

os.makedirs(RNA_HDOCK_CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# OpenBabel resolution
# ---------------------------------------------------------------------------

def resolve_obabel() -> str:
    candidates = [
        os.getenv("OBABEL_BIN"),
        DEFAULT_OBABEL_BIN,
        shutil.which("obabel"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            log.info("Using OpenBabel executable: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Could not find OpenBabel executable 'obabel'. "
        "Set OBABEL_BIN or install conda-forge::openbabel into the main env."
    )


def log_obabel_diagnostics() -> None:
    try:
        obabel_bin = resolve_obabel()

        version = subprocess.run(
            [obabel_bin, "-V"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )

        log.info("[OBABEL DIAG] Version return code: %s", version.returncode)

        if version.stdout:
            log.info("[OBABEL DIAG] Version STDOUT:\n%s", version.stdout.strip())

        if version.stderr:
            log.warning("[OBABEL DIAG] Version STDERR:\n%s", version.stderr.strip())

        formats = subprocess.run(
            [obabel_bin, "-L", "formats"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )

        formats_text = f"{formats.stdout}\n{formats.stderr}"

        log.info("[OBABEL DIAG] Has PDB format: %s", "pdb" in formats_text.lower())
        log.info("[OBABEL DIAG] Has PDBQT format: %s", "pdbqt" in formats_text.lower())

    except Exception as e:
        log.warning("[OBABEL DIAG] Failed to run diagnostics: %s", e)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _file_info(path: str | Path) -> dict:
    path = Path(path)

    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "is_file": path.is_file() if path.exists() else None,
    }


def _count_pdb_records(path: str | Path) -> dict:
    path = Path(path)

    counts = {
        "ATOM": 0,
        "HETATM": 0,
        "MODEL": 0,
        "TER": 0,
        "END": 0,
        "OTHER": 0,
        "TOTAL_LINES": 0,
    }

    if not path.exists():
        return counts

    with path.open(errors="ignore") as handle:
        for line in handle:
            counts["TOTAL_LINES"] += 1

            if line.startswith("ATOM"):
                counts["ATOM"] += 1
            elif line.startswith("HETATM"):
                counts["HETATM"] += 1
            elif line.startswith("MODEL"):
                counts["MODEL"] += 1
            elif line.startswith("TER"):
                counts["TER"] += 1
            elif line.startswith("END"):
                counts["END"] += 1
            else:
                counts["OTHER"] += 1

    return counts


def _preview_file(path: str | Path, n: int = 10) -> str:
    path = Path(path)

    if not path.exists():
        return "<missing>"

    lines = []

    try:
        with path.open(errors="ignore") as handle:
            for i, line in enumerate(handle):
                if i >= n:
                    break
                lines.append(line.rstrip("\n"))
    except Exception as e:
        return f"<could not preview file: {e}>"

    return "\n".join(lines) if lines else "<empty>"


def _copy_debug_file(src: str | Path, label: str) -> None:
    if not RNA_PREP_DEBUG_DIR:
        return

    src = Path(src)

    if not src.exists():
        return

    try:
        debug_dir = Path(RNA_PREP_DEBUG_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_label = label.replace("/", "_").replace(" ", "_")
        dst = debug_dir / safe_label

        shutil.copy2(src, dst)
        log.info("Copied RNA prep debug file: %s -> %s", src, dst)

    except Exception as e:
        log.warning("Could not copy debug file %s: %s", src, e)


def _validate_nonempty_output(path: str | Path, step_name: str) -> None:
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"[OBABEL:{step_name}] Expected output missing: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"[OBABEL:{step_name}] Expected output is empty: {path}")


def _hash_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# PDB atom/element repair
# ---------------------------------------------------------------------------

def _infer_element_from_pdb_atom_line(line: str) -> str:
    atom_name = line[12:16].strip()

    if not atom_name:
        return ""

    cleaned = atom_name.replace("'", "").replace("*", "").strip()

    if not cleaned:
        return ""

    for ch in cleaned:
        if ch.isalpha():
            first = ch.upper()
            break
    else:
        return ""

    if first in {"C", "N", "O", "P", "H", "S"}:
        return first

    return first


def _ensure_pdb_element_column(line: str) -> str:
    line = line.rstrip("\n")

    if len(line) < 78:
        line = line.ljust(78)

    element = line[76:78].strip()

    if not element:
        inferred = _infer_element_from_pdb_atom_line(line)

        if inferred:
            line = line[:76] + inferred.rjust(2) + line[78:]

    return line


# ---------------------------------------------------------------------------
# PDB cleaning
# ---------------------------------------------------------------------------

def clean_pdb_for_obabel(input_pdb: str | Path, output_pdb: str | Path) -> None:
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)

    with input_pdb.open(errors="ignore") as f_in, output_pdb.open("w") as f_out:
        wrote_any = False

        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                repaired = _ensure_pdb_element_column(line)
                f_out.write(repaired.rstrip("\n") + "\n")
                wrote_any = True

        if wrote_any:
            f_out.write("END\n")


def prepare_rna_pdb_for_hdock(rna_pdb: str) -> str | None:
    """
    Prepare a clean RNA PDB for HDOCKlite.

    HDOCKlite accepts PDB directly. This function:
      - validates input
      - keeps ATOM/HETATM only
      - repairs missing element columns
      - caches the cleaned PDB by input file hash

    Returns:
        Path to cleaned RNA PDB, or None.
    """

    try:
        if not rna_pdb or not os.path.exists(rna_pdb):
            log.warning("[RNA HDOCK PREP] RNA PDB missing: %s", rna_pdb)
            return None

        counts = _count_pdb_records(rna_pdb)

        if counts["ATOM"] + counts["HETATM"] == 0:
            log.warning("[RNA HDOCK PREP] RNA PDB has no coordinate records: %s", rna_pdb)
            return None

        key = _hash_file(rna_pdb)
        out_pdb = os.path.join(RNA_HDOCK_CACHE_DIR, f"rna_hdock_{key}.pdb")

        if os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 100:
            return out_pdb

        clean_pdb_for_obabel(rna_pdb, out_pdb)

        out_counts = _count_pdb_records(out_pdb)

        if out_counts["ATOM"] + out_counts["HETATM"] == 0:
            log.warning("[RNA HDOCK PREP] Cleaned RNA PDB has no coordinate records: %s", out_pdb)
            return None

        _copy_debug_file(out_pdb, f"rna_hdock_{key}.pdb")

        log.info("[RNA HDOCK PREP] Prepared RNA PDB for HDOCK: %s", out_pdb)
        return out_pdb

    except Exception as e:
        log.exception("[RNA HDOCK PREP] Failed for %s: %s", rna_pdb, e)
        return None


# ---------------------------------------------------------------------------
# OpenBabel execution helpers
# ---------------------------------------------------------------------------

def run_obabel(
    args: list[str],
    timeout: int = 60,
    expected_output: str | Path | None = None,
    step_name: str = "obabel",
) -> subprocess.CompletedProcess:
    obabel_bin = resolve_obabel()
    cmd = [obabel_bin, *map(str, args)]

    log.info("[OBABEL:%s] Command: %s", step_name, " ".join(cmd))
    log.info("[OBABEL:%s] CWD: %s", step_name, os.getcwd())

    if expected_output is not None:
        log.info(
            "[OBABEL:%s] Expected output before run: %s",
            step_name,
            _file_info(expected_output),
        )

    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    log.info("[OBABEL:%s] Return code: %s", step_name, result.returncode)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if stdout:
        log.info("[OBABEL:%s] STDOUT:\n%s", step_name, stdout)
    else:
        log.info("[OBABEL:%s] STDOUT: <empty>", step_name)

    if stderr:
        log.warning("[OBABEL:%s] STDERR:\n%s", step_name, stderr)
    else:
        log.info("[OBABEL:%s] STDERR: <empty>", step_name)

    if expected_output is not None:
        log.info(
            "[OBABEL:%s] Expected output after run: %s",
            step_name,
            _file_info(expected_output),
        )

        if Path(expected_output).exists():
            log.debug(
                "[OBABEL:%s] Output preview:\n%s",
                step_name,
                _preview_file(expected_output, n=10),
            )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    if expected_output is not None:
        _validate_nonempty_output(expected_output, step_name)

    return result


def run_obabel_variants(
    variants: list[list[str]],
    expected_output: str | Path,
    timeout: int,
    step_name: str,
) -> subprocess.CompletedProcess:
    expected_output = Path(expected_output)
    last_error: Exception | None = None

    for i, args in enumerate(variants, start=1):
        variant_name = f"{step_name}_variant_{i}"

        try:
            if expected_output.exists():
                expected_output.unlink()
        except Exception as e:
            log.warning(
                "[OBABEL:%s] Could not remove stale output %s: %s",
                variant_name,
                expected_output,
                e,
            )

        try:
            result = run_obabel(
                args=args,
                timeout=timeout,
                expected_output=expected_output,
                step_name=variant_name,
            )

            log.info("[OBABEL:%s] Variant succeeded.", variant_name)
            return result

        except Exception as e:
            last_error = e
            log.warning("[OBABEL:%s] Variant failed: %s", variant_name, e)

    raise RuntimeError(
        f"All OpenBabel variants failed for {step_name}. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Legacy Vina ligand prep
# ---------------------------------------------------------------------------

def prepare_rna_ligand(rna_pdb: str) -> str | None:
    """
    Legacy RNA PDB -> PDBQT preparation for Vina.

    HDOCKlite callers should use prepare_rna_pdb_for_hdock().
    """

    try:
        log.info("[RNA PREP] Starting legacy PDBQT ligand preparation for: %s", rna_pdb)
        log.info("[RNA PREP] Input file info: %s", _file_info(rna_pdb))

        log_obabel_diagnostics()

        if not rna_pdb or not os.path.exists(rna_pdb):
            log.warning("[RNA PREP] RNA PDB missing: %s", rna_pdb)
            return None

        input_counts = _count_pdb_records(rna_pdb)
        log.info("[RNA PREP] Input PDB record counts: %s", input_counts)

        if input_counts["ATOM"] + input_counts["HETATM"] == 0:
            log.warning("[RNA PREP] Input RNA PDB has no ATOM/HETATM records: %s", rna_pdb)
            return None

        tmp_dir = tempfile.mkdtemp(prefix="rna_ligand_")
        log.info("[RNA PREP] Temporary ligand directory: %s", tmp_dir)

        cleaned_pdb = os.path.join(tmp_dir, "rna_clean.pdb")
        hydrogenated_pdb = os.path.join(tmp_dir, "rna_h.pdb")
        pdbqt_path = os.path.join(tmp_dir, "rna.pdbqt")

        clean_pdb_for_obabel(rna_pdb, cleaned_pdb)

        _copy_debug_file(cleaned_pdb, "rna_clean.pdb")

        if not os.path.exists(cleaned_pdb) or os.path.getsize(cleaned_pdb) == 0:
            log.warning("[RNA PREP] Cleaned RNA PDB is empty: %s", rna_pdb)
            return None

        run_obabel_variants(
            variants=[
                ["-i", "pdb", cleaned_pdb, "-o", "pdb", "-O", hydrogenated_pdb, "-h"],
                ["-ipdb", cleaned_pdb, "-opdb", "-O", hydrogenated_pdb, "-h"],
                [cleaned_pdb, "-O", hydrogenated_pdb, "-h"],
            ],
            timeout=60,
            expected_output=hydrogenated_pdb,
            step_name="hydrogenate_pdb",
        )

        _copy_debug_file(hydrogenated_pdb, "rna_h.pdb")

        run_obabel_variants(
            variants=[
                [
                    "-i",
                    "pdb",
                    hydrogenated_pdb,
                    "-o",
                    "pdbqt",
                    "-O",
                    pdbqt_path,
                    "--partialcharge",
                    "gasteiger",
                    "-xr",
                ],
                [
                    "-ipdb",
                    hydrogenated_pdb,
                    "-opdbqt",
                    "-O",
                    pdbqt_path,
                    "--partialcharge",
                    "gasteiger",
                    "-xr",
                ],
                [
                    hydrogenated_pdb,
                    "-O",
                    pdbqt_path,
                    "--partialcharge",
                    "gasteiger",
                    "-xr",
                ],
            ],
            timeout=60,
            expected_output=pdbqt_path,
            step_name="pdb_to_pdbqt",
        )

        _copy_debug_file(pdbqt_path, "rna.pdbqt")

        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            log.info("[RNA PREP] Prepared RNA ligand PDBQT: %s", pdbqt_path)
            return pdbqt_path

        log.warning("[RNA PREP] OpenBabel did not produce a valid PDBQT: %s", pdbqt_path)
        return None

    except subprocess.CalledProcessError as e:
        log.warning(
            "[RNA PREP] OpenBabel command failed for %s\n"
            "Return code: %s\n"
            "Command: %s\n"
            "STDOUT: %s\n"
            "STDERR: %s",
            rna_pdb,
            e.returncode,
            " ".join(map(str, e.cmd)) if getattr(e, "cmd", None) else "<unknown>",
            e.output,
            e.stderr,
        )
        return None

    except subprocess.TimeoutExpired as e:
        log.warning(
            "[RNA PREP] OpenBabel command timed out for %s\n"
            "Command: %s\n"
            "STDOUT: %s\n"
            "STDERR: %s",
            rna_pdb,
            " ".join(map(str, e.cmd)) if getattr(e, "cmd", None) else "<unknown>",
            getattr(e, "stdout", None),
            getattr(e, "stderr", None),
        )
        return None

    except Exception as e:
        log.exception("[RNA PREP] RNA ligand prep failed for %s: %s", rna_pdb, e)
        return None
