import os
import re
import shutil
import subprocess
import logging
from pathlib import Path

from VLAB2.core.gpu_manager import set_gpu_for_agent

log = logging.getLogger("virtual_lab.vina_wrapper")

DEFAULT_VINA_BIN = "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/vina"

DEFAULT_VINA_DEBUG_DIR = (
    "/mnt/scratch/fbsnpat/bot/VLAB2/output_data/debug_vina"
)


class VinaDocking:
    def __init__(self, vina_path: str = None):
        self.device = set_gpu_for_agent(prefer=1)

        self.vina_path = (
            vina_path
            or os.getenv("VINA_BIN")
            or DEFAULT_VINA_BIN
            or shutil.which("vina")
            or shutil.which("autodock-vina")
            or "vina"
        )

        if self.vina_path == DEFAULT_VINA_BIN and not os.path.exists(self.vina_path):
            self.vina_path = (
                shutil.which("vina")
                or shutil.which("autodock-vina")
                or "vina"
            )

        self.debug_dir = os.getenv("VINA_DEBUG_DIR", DEFAULT_VINA_DEBUG_DIR)

        log.info("VinaDocking initialised with binary: %s", self.vina_path)

    # --------------------------------------------------
    # MAIN DOCK FUNCTION
    # --------------------------------------------------

    def dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: tuple,
        size: tuple,
        exhaustiveness: int = 8,
    ) -> dict:

        receptor_pdbqt = str(receptor_pdbqt)
        ligand_pdbqt = str(ligand_pdbqt)

        exhaustiveness = int(os.getenv("VINA_EXHAUSTIVENESS", str(exhaustiveness)))
        timeout = int(os.getenv("VINA_TIMEOUT", "600"))

        log.info("[VINA INPUT] receptor info: %s", self._file_info(receptor_pdbqt))
        log.info("[VINA INPUT] ligand info: %s", self._file_info(ligand_pdbqt))
        log.info("[VINA INPUT] receptor counts: %s", self._count_pdbqt_records(receptor_pdbqt))
        log.info("[VINA INPUT] ligand counts: %s", self._count_pdbqt_records(ligand_pdbqt))

        # ---------- VALIDATION ----------
        if not os.path.exists(self.vina_path):
            return self._fail("Vina binary missing")

        if not os.access(self.vina_path, os.X_OK):
            return self._fail("Vina binary not executable")

        if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
            return self._fail("Missing PDBQT files")

        # ---------- OUTPUT PATHS ----------
        base, ext = os.path.splitext(ligand_pdbqt)
        output_pdbqt = f"{base}_docked{ext}"

        # ---------- COMMAND (NO --LOG) ----------
        cmd = [
            self.vina_path,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(float(center[0])),
            "--center_y", str(float(center[1])),
            "--center_z", str(float(center[2])),
            "--size_x", str(float(size[0])),
            "--size_y", str(float(size[1])),
            "--size_z", str(float(size[2])),
            "--exhaustiveness", str(exhaustiveness),
            "--out", output_pdbqt,
        ]

        log.info("[VINA CMD] %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            log.info("[VINA RETURN] %s", result.returncode)

            if stdout.strip():
                log.info("[VINA STDOUT]\n%s", stdout.strip())
            if stderr.strip():
                log.warning("[VINA STDERR]\n%s", stderr.strip())

            log.info("[VINA OUT FILE] %s", self._file_info(output_pdbqt))

            # Save debug artefacts
            self._copy_debug_file(receptor_pdbqt, "receptor.pdbqt")
            self._copy_debug_file(ligand_pdbqt, "ligand.pdbqt")
            self._copy_debug_file(output_pdbqt, "docked.pdbqt")

            # ---------- FAILURE ----------
            if result.returncode != 0:
                return self._simulate_fallback(
                    ligand_pdbqt,
                    error=stderr.strip() or "Vina failed",
                    stdout=stdout,
                    stderr=stderr,
                    output_file=output_pdbqt,
                    returncode=result.returncode,
                )

            # ---------- SUCCESS → PARSE ENERGY ----------
            combined = "\n".join([stdout, stderr])
            energy = self._parse_vina_energy(combined)

            if energy is not None:
                log.info("[VINA SUCCESS] binding_energy = %.3f kcal/mol", energy)

                return {
                    "binding_energy": energy,
                    "method": "vina",
                    "valid": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": result.returncode,
                    "output_file": output_pdbqt,
                }

            return self._simulate_fallback(
                ligand_pdbqt,
                error="Could not parse energy from stdout",
                stdout=stdout,
                stderr=stderr,
                output_file=output_pdbqt,
                returncode=result.returncode,
            )

        except subprocess.TimeoutExpired as e:
            return self._simulate_fallback(
                ligand_pdbqt,
                error=f"Timeout after {timeout}s",
                stdout=getattr(e, "stdout", ""),
                stderr=getattr(e, "stderr", ""),
                output_file=output_pdbqt,
            )

        except Exception as e:
            log.exception("Unexpected Vina error")
            return self._simulate_fallback(
                ligand_pdbqt,
                error=str(e),
                output_file=output_pdbqt,
            )

    # --------------------------------------------------
    # PARSER
    # --------------------------------------------------

    @staticmethod
    def _parse_vina_energy(text: str):
        """
        Extracts best binding energy from stdout table
        """
        if not text:
            return None

        match = re.search(
            r"^\s*1\s+([-+]?\d*\.\d+|[-+]?\d+)",
            text,
            re.MULTILINE,
        )

        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _fail(self, msg):
        log.warning(msg)
        return self._simulate_fallback("", error=msg)

    def _simulate_fallback(
        self,
        ligand_pdbqt,
        error=None,
        stdout=None,
        stderr=None,
        output_file=None,
        returncode=None,
    ):
        return {
            "binding_energy": None,
            "method": "vina_failed",
            "valid": False,
            "warning": "Vina execution failed; no physical docking score produced",
            "error": error,
            "stdout": stdout,
            "stderr": stderr,
            "output_file": output_file,
            "returncode": returncode,
        }

    @staticmethod
    def _file_info(path):
        path = Path(path)
        return {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
        }

    @staticmethod
    def _count_pdbqt_records(path):
        path = Path(path)

        counts = {"ATOM": 0, "BRANCH": 0, "ROOT": 0, "TORSDOF": 0}

        if not path.exists():
            return counts

        with path.open(errors="ignore") as f:
            for line in f:
                for k in counts:
                    if line.startswith(k):
                        counts[k] += 1

        return counts

    def _copy_debug_file(self, src, label):
        if not self.debug_dir:
            return

        src = Path(src)
        if not src.exists():
            return

        try:
            debug_dir = Path(self.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

            dst = debug_dir / label
            shutil.copy2(src, dst)

        except Exception as e:
            log.warning("Debug copy failed: %s", e)