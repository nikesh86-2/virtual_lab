"""
hdock_wrapper.py

HDOCKlite wrapper for protein-RNA / protein-protein docking.

Keeps current working createpl path:
    createpl hdock.out top_models.pdb -nmax N -complex -models

Adds:
- stronger PDB validation
- debug preservation
- safer output discovery
- legacy Vina-compatible fields
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


try:
    from VLAB2.core.gpu_manager import set_gpu_for_agent
except Exception:
    def set_gpu_for_agent(prefer=1):
        return None


log = logging.getLogger("virtual_lab.hdock_wrapper")

DEFAULT_HDOCK_BIN = os.getenv("HDOCK_BIN", "hdock")
DEFAULT_CREATEPL_BIN = os.getenv("CREATEPL_BIN", "createpl")
DEFAULT_HDOCK_DEBUG_DIR = "/mnt/scratch/fbsnpat/bot/VLAB2/output_data/debug_hdock"

ERROR_TOKENS = [
    "parameters input error",
    "input error",
    "cannot open",
    "no such file",
    "segmentation fault",
    "fatal",
    "error:",
]


class HDockDocking:
    def __init__(
        self,
        hdock_path: Optional[str] = None,
        createpl_path: Optional[str] = None,
    ):
        self.device = set_gpu_for_agent(prefer=1)

        self.models = self._getenv_int("HDOCK_N_MODELS", 10, min_value=1)
        self.timeout = self._getenv_int("HDOCK_TIMEOUT", 1200, min_value=1)
        self.createpl_timeout = self._getenv_int("CREATEPL_TIMEOUT", 300, min_value=1)

        self.hdock_path = (
            hdock_path
            or os.getenv("HDOCK_BIN")
            or shutil.which("hdock")
            or DEFAULT_HDOCK_BIN
        )

        self.createpl_path = (
            createpl_path
            or os.getenv("CREATEPL_BIN")
            or shutil.which("createpl")
            or DEFAULT_CREATEPL_BIN
        )

        self.debug_dir = os.getenv("HDOCK_DEBUG_DIR", DEFAULT_HDOCK_DEBUG_DIR)
        self.preserve_debug = os.getenv("HDOCK_PRESERVE_DEBUG", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.min_atoms = self._getenv_int("HDOCK_MIN_PDB_ATOMS", 3, min_value=1)

        log.info("HDockDocking initialised with hdock binary: %s", self.hdock_path)
        log.info("HDockDocking initialised with createpl binary: %s", self.createpl_path)
        log.info("HDockDocking configured HDOCK_TIMEOUT=%s", self.timeout)
        log.info("HDockDocking configured HDOCK_N_MODELS=%s", self.models)

    def dock(
        self,
        receptor_pdb: str,
        ligand_pdb: str,
        n_models: Optional[int] = None,
    ) -> dict:
        """
        Run HDOCK and create complex models.

        Args:
            receptor_pdb:
                Raw receptor PDB path.
            ligand_pdb:
                Raw ligand/RNA PDB path.
            n_models:
                Optional override for number of createpl models. If None,
                uses HDOCK_N_MODELS from environment captured at wrapper init.
        """
        timeout = self._getenv_int("HDOCK_TIMEOUT", self.timeout, min_value=1)

        try:
            n_models = int(n_models if n_models is not None else self.models)
        except Exception:
            n_models = self.models

        n_models = max(1, n_models)

        if not self._binary_exists(self.hdock_path):
            return self._fallback(f"HDOCK binary missing: {self.hdock_path}")

        if not self._binary_exists(self.createpl_path):
            return self._fallback(f"createpl binary missing: {self.createpl_path}")

        receptor_pdb = str(receptor_pdb)
        ligand_pdb = str(ligand_pdb)

        if not self._valid_pdb_file(receptor_pdb):
            return self._fallback(f"Invalid receptor PDB: {receptor_pdb}")

        if not self._valid_pdb_file(ligand_pdb):
            return self._fallback(f"Invalid ligand PDB: {ligand_pdb}")

        ligand_base = Path(ligand_pdb).with_suffix("")
        createpl_timeout = self._getenv_int(
            "CREATEPL_TIMEOUT",
            self.createpl_timeout,
            min_value=1,
        )

        run_id = self._run_id(
            receptor_pdb,
            ligand_pdb,
            n_models=n_models,
            timeout=timeout,
            createpl_timeout=createpl_timeout,
        )

        output_out = str(
            ligand_base.parent / f"{ligand_base.name}_{run_id}_hdock.out"
        )
        output_complex = str(
            ligand_base.parent / f"{ligand_base.name}_{run_id}_hdock_complex.pdb"
        )
        failure_meta = {
                        "hdock_run_id": run_id,
                        "receptor_pdb_input": receptor_pdb,
                        "ligand_pdb_input": ligand_pdb,
                        "n_models": n_models,
                    }

        log.info(
            "[HDOCK RUN] run_id=%s receptor=%s ligand=%s output=%s complex=%s",
            run_id,
            receptor_pdb,
            ligand_pdb,
            output_out,
            output_complex,
        )


        try:
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)

                shutil.copy2(receptor_pdb, work / "receptor.pdb")
                shutil.copy2(ligand_pdb, work / "ligand.pdb")

                cmd = [
                    self.hdock_path,
                    "receptor.pdb",
                    "ligand.pdb",
                    "-out",
                    "hdock.out",
                ]

                log.info("[HDOCK CMD] %s", " ".join(cmd))
                log.info("[HDOCK CONFIG] timeout=%s n_models=%s", timeout, n_models)

                try:
                    result = subprocess.run(
                        cmd,
                        cwd=work,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired as e:
                    self._maybe_preserve_debug(work, run_id)
                    return self._fallback(
                        f"HDOCK timed out after {timeout}s",
                        stdout=e.stdout or "",
                        stderr=e.stderr or "",
                        **failure_meta,
                    )

                stdout = result.stdout or ""
                stderr = result.stderr or ""

                if result.returncode != 0 or self._has_error(stdout, stderr):
                    self._maybe_preserve_debug(work, run_id)
                    return self._fallback(
                        f"HDOCK failed rc={result.returncode}: {self._trim(stderr or stdout)}",
                        stdout=stdout,
                        stderr=stderr,
                        **failure_meta
                    )

                out_file = self._discover_hdock_output(work)

                if not out_file:
                    self._maybe_preserve_debug(work, run_id)
                    return self._fallback(
                        "No hdock.out generated",
                        stdout=stdout,
                        stderr=stderr,
                        **failure_meta
                    )

                shutil.copy2(out_file, output_out)

                score = self._parse_hdock_score(output_out)

                if score is None:
                    self._maybe_preserve_debug(work, run_id)
                    return self._fallback(
                    "Score parse failed",
                    stdout=stdout,
                    stderr=stderr,
                    **failure_meta,
                )


                complex_status = self._create_complex(
                    work_dir=work,
                    hdock_out=out_file,
                    n_models=n_models,
                    createpl_timeout=createpl_timeout,
                )


                final_complex = None

                if complex_status.get("valid"):
                    try:
                        shutil.copy2(complex_status["output_file"], output_complex)
                        final_complex = output_complex
                    except Exception as e:
                        complex_status["copy_error"] = str(e)

                self._maybe_preserve_debug(work, run_id)

                return {
                    "dock_score": score,
                    "hdock_score": score,
                    "binding_energy": score,
                    "binding_energy_is_physical": False,
                    "binding_units": "hdock_relative_score",
                    "valid": True,
                    "dock_valid": True,
                    "method": "hdock",
                    "dock_method": "hdock",
                    "output_file": output_out,
                    "dock_output_file": output_out,
                    "complex_file": final_complex,
                    "dock_complex_file": final_complex,
                    "complex_status": complex_status,

                    "hdock_run_id": run_id,
                    "receptor_pdb_input": receptor_pdb,
                    "ligand_pdb_input": ligand_pdb,
                    "n_models": n_models,

                    "vina_energy": score,
                    "vina_valid": True,
                    "vina_method": "hdock",
                    "vina_error": None,
                }

        except subprocess.TimeoutExpired:
            return self._fallback(f"HDOCK timed out after {timeout}s")

        except Exception as e:
            return self._fallback(str(e))

    def _create_complex(
        self,
        work_dir: Path,
        hdock_out: Path,
        n_models: int,
        createpl_timeout: Optional[int] = None,
    ) -> dict:
        if not self._binary_exists(self.createpl_path):
            return {"valid": False, "error": "createpl missing"}

        output_name = "top_models.pdb"
        before = {str(p) for p in work_dir.glob("*.pdb")}

        commands = [
            [
                self.createpl_path,
                hdock_out.name,
                output_name,
                "-nmax",
                str(n_models),
                "-complex",
                "-models",
            ],
            [
                self.createpl_path,
                hdock_out.name,
                output_name,
                "-nmax",
                str(n_models),
                "-complex",
            ],
            [
                self.createpl_path,
                hdock_out.name,
                output_name,
                str(n_models),
            ],
            [
                self.createpl_path,
                hdock_out.name,
                output_name,
            ],
        ]

        last = None

        for cmd in commands:
            log.info("[CREATEPL CMD] %s", " ".join(cmd))

            try:
                res = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=createpl_timeout if createpl_timeout is not None else self.createpl_timeout,
                )

            except Exception as e:
                last = {
                    "valid": False,
                    "error": str(e),
                    "cmd": " ".join(cmd),
                }
                continue

            stdout = res.stdout or ""
            stderr = res.stderr or ""
            pdb = self._discover_complex_output(work_dir, before)

            if self._valid_createpl(stdout, stderr, res.returncode, pdb):
                return {
                    "valid": True,
                    "output_file": str(pdb),
                    "cmd": " ".join(cmd),
                    "stdout": self._trim(stdout),
                    "stderr": self._trim(stderr),
                    "n_models": n_models,
                }

            last = {
                "valid": False,
                "stdout": self._trim(stdout),
                "stderr": self._trim(stderr),
                "cmd": " ".join(cmd),
                "returncode": res.returncode,
                "n_models": n_models,
            }

        return last or {"valid": False, "error": "createpl failed", "n_models": n_models}

    def _discover_complex_output(
        self,
        work_dir: Path,
        before: set,
    ) -> Optional[Path]:
        expected = work_dir / "top_models.pdb"

        if self._valid_pdb_file(expected):
            return expected

        patterns = [
            "model_*.pdb",
            "model*.pdb",
            "*complex*.pdb",
            "*.pdb",
        ]

        for pattern in patterns:
            for p in sorted(work_dir.glob(pattern)):
                if str(p) in before:
                    continue

                if self._valid_pdb_file(p):
                    return p

        return None

    @staticmethod
    def _discover_hdock_output(work_dir: Path) -> Optional[Path]:
        for name in ["hdock.out", "Hdock.out", "HDOCK.out"]:
            p = work_dir / name

            if p.exists() and p.stat().st_size > 0:
                return p

        for p in sorted(work_dir.glob("*.out")):
            if p.exists() and p.stat().st_size > 0:
                return p

        return None

    def _valid_createpl(
        self,
        stdout: str,
        stderr: str,
        rc: int,
        pdb: Optional[Path],
    ) -> bool:
        if self._has_error(stdout, stderr):
            return False

        if rc != 0:
            return False

        if not pdb:
            return False

        return self._valid_pdb_file(pdb)

    def _valid_pdb_file(self, path_like) -> bool:
        try:
            path = Path(path_like)

            if not path.exists() or path.stat().st_size == 0:
                return False

            atom_count = 0

            with path.open("r", errors="ignore") as handle:
                for line in handle:
                    if line.startswith(("ATOM", "HETATM")):
                        atom_count += 1

                        if atom_count >= self.min_atoms:
                            return True

            return False

        except Exception:
            return False

    @staticmethod
    def _has_error(stdout: str, stderr: str) -> bool:
        text = ((stdout or "") + "\n" + (stderr or "")).lower()
        return any(token in text for token in ERROR_TOKENS)

    @staticmethod
    def _parse_hdock_score(path_like) -> Optional[float]:
        """
        observed in VLAB2:    Parse HDOCK-relative score from HDOCK output.
            a b c x y z score ligand_rmsd cluster
        where score is the 7th numeric column, index 6.
        """

        text = Path(path_like).read_text(errors="ignore")

        # Format 1: explicit rank followed by score.
        patterns = [
            r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+",
            r"^\s*MODEL\s+1\b.*?score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            r"score\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            r"energy\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    pass

        # Format 2: HDOCK numeric table.
        # Example:
        # 2.96663 2.89884 4.31038 22.733 -23.529 -7.133 -61.46 54.21 1.00
        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            # Skip obvious headers.
            if not re.match(r"^[\s\-+]?\d", stripped):
                continue

            parts = stripped.split()

            # Need at least 9 numeric columns for the HDOCK pose table.
            if len(parts) < 9:
                continue

            try:
                vals = [float(x) for x in parts[:9]]
            except Exception:
                continue

            score = vals[6]

            # HDOCK scores should usually be negative for meaningful poses.
            # Avoid parsing grid/progress lines accidentally.
            if score < -1.0:
                return score

        return None

    def _fallback(
        self,
        error: str,
        stdout: str = "",
        stderr: str = "",
        **metadata,
    ) -> dict:
        result = {
            "valid": False,
            "dock_valid": False,
            "error": error,
            "dock_error": error,
            "dock_score": None,
            "hdock_score": None,
            "binding_energy": None,
            "binding_energy_is_physical": False,
            "binding_units": "hdock_relative_score",
            "method": "hdock",
            "dock_method": "hdock",
            "stdout": self._trim(stdout),
            "stderr": self._trim(stderr),

            # Legacy Vina compatibility fields.
            "vina_energy": None,
            "vina_valid": False,
            "vina_method": "hdock",
            "vina_error": error,
        }

        result.update(metadata)
        return result

    @staticmethod
    def _binary_exists(path_like) -> bool:
        if not path_like:
            return False

        return os.path.exists(str(path_like)) or shutil.which(str(path_like)) is not None

    @staticmethod
    def _trim(text: str, n: int = 2000) -> str:
        text = text or ""

        if len(text) <= n:
            return text

        return text[:n] + "...[truncated]"

    @staticmethod
    def _file_sha1(path_like: str | Path) -> str:
        h = hashlib.sha1()
        path = Path(path_like)

        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)

        return h.hexdigest()


    def _run_id(
        self,
        receptor_pdb: str,
        ligand_pdb: str,
        n_models: Optional[int] = None,
        timeout: Optional[int] = None,
        createpl_timeout: Optional[int] = None,
    ) -> str:
        h = hashlib.sha1()

        try:
            receptor_hash = self._file_sha1(receptor_pdb)
        except Exception:
            receptor_hash = str(receptor_pdb)

        try:
            ligand_hash = self._file_sha1(ligand_pdb)
        except Exception:
            ligand_hash = str(ligand_pdb)

        payload = "|".join(
            [
                "hdock_v2",
                receptor_hash,
                ligand_hash,
                str(n_models if n_models is not None else self.models),
                str(timeout if timeout is not None else self.timeout),
                str(createpl_timeout if createpl_timeout is not None else self.createpl_timeout),
                str(self.hdock_path),
                str(self.createpl_path),
            ]
        )

        h.update(payload.encode("utf-8"))
        return h.hexdigest()[:16]

    def _maybe_preserve_debug(
        self,
        work_dir: Path,
        run_id: str,
    ) -> Optional[str]:
        if not self.preserve_debug:
            return None

        try:
            dest = Path(self.debug_dir) / f"hdock_{run_id}"
            dest.mkdir(parents=True, exist_ok=True)

            for p in work_dir.glob("*"):
                if p.is_file():
                    shutil.copy2(p, dest / p.name)

            log.info("[HDOCK DEBUG] preserved run files in %s", dest)
            return str(dest)

        except Exception as e:
            log.warning("[HDOCK DEBUG] failed to preserve files: %s", e)
            return None

    @staticmethod
    def _getenv_int(
        name: str,
        default: int,
        min_value: Optional[int] = None,
    ) -> int:
        raw = os.getenv(name, str(default))

        try:
            value = int(raw)
        except Exception:
            log.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
            value = int(default)

        if min_value is not None:
            value = max(min_value, value)

        return value