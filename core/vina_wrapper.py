from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from VLAB2.core.gpu_manager import set_gpu_for_agent


log = logging.getLogger("virtual_lab.vina_wrapper")

DEFAULT_VINA_BIN = "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/vina"
DEFAULT_VINA_DEBUG_DIR = "/mnt/scratch/fbsnpat/bot/VLAB2/output_data/debug_vina"
DEFAULT_VINA_CACHE_DIR = "/mnt/scratch/fbsnpat/bot/VLAB2/output_data/vina_result_cache"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


# Phase 1.4: Vina cache schema version
VINA_CACHE_SCHEMA_VERSION = "vina-cache-v3"


class VinaDocking:
    """AutoDock Vina wrapper with deterministic seeds and persistent result caching."""

    def __init__(
        self,
        vina_path: str | None = None,
        cache_dir: str | None = None,
    ):
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
            self.vina_path = shutil.which("vina") or shutil.which("autodock-vina") or "vina"

        self.debug_dir = os.getenv("VINA_DEBUG_DIR", DEFAULT_VINA_DEBUG_DIR)
        self.cache_enabled = _env_bool("VLAB_VINA_CACHE_ENABLED", True)
        self.cache_dir = Path(
            cache_dir
            or os.getenv("VLAB_VINA_RESULT_CACHE_DIR")
            or DEFAULT_VINA_CACHE_DIR
        )
        self.cache_version = os.getenv("VLAB_VINA_CACHE_VERSION", VINA_CACHE_SCHEMA_VERSION)

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "VinaDocking initialised binary=%s cache_enabled=%s cache_dir=%s",
            self.vina_path,
            self.cache_enabled,
            self.cache_dir,
        )

    def dock(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: tuple,
        size: tuple,
        exhaustiveness: int = 8,
        seed: int | None = None,
        force_redock: bool | None = None,
        ligand_smiles_sha256: str | None = None,
        ligand_pdbqt_sha256: str | None = None,
        manifest_version: str | None = None,
        record_version: int | None = None,
    ) -> dict:
        receptor_pdbqt = str(receptor_pdbqt)
        ligand_pdbqt = str(ligand_pdbqt)
        exhaustiveness = _env_int("VLAB_VINA_EXHAUSTIVENESS", exhaustiveness)
        exhaustiveness = _env_int("VINA_EXHAUSTIVENESS", exhaustiveness)
        timeout = _env_int("VINA_TIMEOUT", 600)
        seed = _env_int("VLAB_VINA_SEED", 1) if seed is None else int(seed)
        force_redock = (
            _env_bool("VLAB_VINA_FORCE_REDOCK", False)
            if force_redock is None
            else bool(force_redock)
        )

        center = tuple(float(value) for value in center)
        size = tuple(float(value) for value in size)

        log.info("[VINA INPUT] receptor info: %s", self._file_info(receptor_pdbqt))
        log.info("[VINA INPUT] ligand info: %s", self._file_info(ligand_pdbqt))
        log.info("[VINA INPUT] receptor counts: %s", self._count_pdbqt_records(receptor_pdbqt))
        log.info("[VINA INPUT] ligand counts: %s", self._count_pdbqt_records(ligand_pdbqt))

        if not os.path.exists(self.vina_path):
            return self._fail("Vina binary missing", seed=seed)
        if not os.access(self.vina_path, os.X_OK):
            return self._fail("Vina binary not executable", seed=seed)
        if not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
            return self._fail("Missing PDBQT files", seed=seed)

        base, ext = os.path.splitext(ligand_pdbqt)
        output_pdbqt = f"{base}_docked{ext}"

        cache_key, cache_payload = self._build_cache_key(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            seed=seed,
            ligand_smiles_sha256=ligand_smiles_sha256,
            ligand_pdbqt_sha256=ligand_pdbqt_sha256,
            manifest_version=manifest_version,
            record_version=record_version,
        )

        if self.cache_enabled and not force_redock:
            cached = self._load_cached_result(cache_key, output_pdbqt)
            if cached is not None:
                log.info(
                    "[VINA CACHE HIT] ligand=%s key=%s seed=%s energy=%s",
                    Path(ligand_pdbqt).name,
                    cache_key[:16],
                    seed,
                    cached.get("binding_energy"),
                )
                return cached

        log.info(
            "[VINA CACHE MISS] ligand=%s key=%s seed=%s force_redock=%s",
            Path(ligand_pdbqt).name,
            cache_key[:16],
            seed,
            force_redock,
        )

        cmd = [
            self.vina_path,
            "--receptor", receptor_pdbqt,
            "--ligand", ligand_pdbqt,
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--seed", str(seed),
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

            self._copy_debug_file(receptor_pdbqt, "receptor.pdbqt")
            self._copy_debug_file(ligand_pdbqt, "ligand.pdbqt")
            self._copy_debug_file(output_pdbqt, "docked.pdbqt")

            if result.returncode != 0:
                return self._simulate_fallback(
                    ligand_pdbqt,
                    error=stderr.strip() or "Vina failed",
                    stdout=stdout,
                    stderr=stderr,
                    output_file=output_pdbqt,
                    returncode=result.returncode,
                    seed=seed,
                    cache_key=cache_key,
                )

            energy = self._parse_vina_energy("\n".join([stdout, stderr]))

            if energy is None:
                return self._simulate_fallback(
                    ligand_pdbqt,
                    error="Could not parse energy from stdout",
                    stdout=stdout,
                    stderr=stderr,
                    output_file=output_pdbqt,
                    returncode=result.returncode,
                    seed=seed,
                    cache_key=cache_key,
                )

            success = {
                "binding_energy": energy,
                "method": "vina",
                "valid": True,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "output_file": output_pdbqt,
                "seed": seed,
                "exhaustiveness": exhaustiveness,
                "cache_key": cache_key,
                "cache_hit": False,
                "cache_version": self.cache_version,
                "center": list(center),
                "size": list(size),
            }

            log.info("[VINA SUCCESS] binding_energy = %.3f kcal/mol seed=%s", energy, seed)

            if self.cache_enabled and os.path.exists(output_pdbqt):
                self._save_cached_result(
                    cache_key=cache_key,
                    result=success,
                    output_pdbqt=output_pdbqt,
                    cache_payload=cache_payload,
                )

            return success

        except subprocess.TimeoutExpired as exc:
            return self._simulate_fallback(
                ligand_pdbqt,
                error=f"Timeout after {timeout}s",
                stdout=getattr(exc, "stdout", ""),
                stderr=getattr(exc, "stderr", ""),
                output_file=output_pdbqt,
                seed=seed,
                cache_key=cache_key,
            )
        except Exception as exc:
            log.exception("Unexpected Vina error")
            return self._simulate_fallback(
                ligand_pdbqt,
                error=str(exc),
                output_file=output_pdbqt,
                seed=seed,
                cache_key=cache_key,
            )

    def _build_cache_key(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: tuple,
        size: tuple,
        exhaustiveness: int,
        seed: int,
        ligand_smiles_sha256: str | None = None,
        ligand_pdbqt_sha256: str | None = None,
        manifest_version: str | None = None,
        record_version: int | None = None,
    ) -> tuple[str, dict]:
        """
        Phase 1.4: Build structure-aware cache key including ligand identity hashes.

        The cache key now includes:
        - receptor PDBQT SHA-256
        - ligand PDBQT SHA-256
        - ligand SMILES SHA-256 (if available)
        - manifest version and record version (if available)
        This ensures that changing a ligand structure or manifest version
        produces a different cache key, preventing stale cache reuse.
        """
        payload = {
            "cache_schema": VINA_CACHE_SCHEMA_VERSION,
            "cache_version": self.cache_version,
            "vina_binary": str(Path(self.vina_path).resolve()),
            "vina_binary_fingerprint": self._file_fingerprint(self.vina_path),
            "receptor_fingerprint": self._file_fingerprint(receptor_pdbqt),
            "ligand_fingerprint": self._file_fingerprint(ligand_pdbqt),
            "ligand_pdbqt_sha256": ligand_pdbqt_sha256,
            "ligand_smiles_sha256": ligand_smiles_sha256,
            "manifest_version": manifest_version,
            "record_version": record_version,
            "center": [round(float(value), 6) for value in center],
            "size": [round(float(value), 6) for value in size],
            "exhaustiveness": int(exhaustiveness),
            "seed": int(seed),
            "scoring_function": "vina",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), payload

    def _cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        entry_dir = self.cache_dir / cache_key[:2] / cache_key
        return entry_dir / "result.json", entry_dir / "docked.pdbqt"

    def _load_cached_result(self, cache_key: str, requested_output: str) -> dict | None:
        metadata_path, cached_pose = self._cache_paths(cache_key)

        if not metadata_path.exists() or not cached_pose.exists():
            return None

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            result = payload.get("result") or {}

            if not result.get("valid") or result.get("binding_energy") is None:
                return None

            Path(requested_output).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_pose, requested_output)
            result = dict(result)
            result["output_file"] = requested_output
            result["cache_hit"] = True
            result["cache_key"] = cache_key
            result["cache_metadata_file"] = str(metadata_path)
            result["cached_pose_file"] = str(cached_pose)
            return result
        except Exception as exc:
            log.warning("[VINA CACHE INVALID] key=%s error=%s", cache_key[:16], exc)
            return None

    def _save_cached_result(
        self,
        cache_key: str,
        result: dict,
        output_pdbqt: str,
        cache_payload: dict,
    ) -> None:
        metadata_path, cached_pose = self._cache_paths(cache_key)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_pose = cached_pose.with_suffix(".tmp")

        try:
            shutil.copy2(output_pdbqt, tmp_pose)
            os.replace(tmp_pose, cached_pose)
            serialisable_result = dict(result)
            serialisable_result["output_file"] = str(cached_pose)
            _atomic_write_json(
                metadata_path,
                {
                    "cache_key": cache_key,
                    "inputs": cache_payload,
                    "result": serialisable_result,
                },
            )
            log.info("[VINA CACHE SAVE] key=%s path=%s", cache_key[:16], metadata_path)
        except Exception as exc:
            log.warning("[VINA CACHE SAVE FAILED] key=%s error=%s", cache_key[:16], exc)
        finally:
            if tmp_pose.exists():
                tmp_pose.unlink()

    @staticmethod
    def _file_fingerprint(path: str) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {"path": str(file_path), "exists": False}

        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "sha256": digest.hexdigest(),
        }

    @staticmethod
    def _parse_vina_energy(text: str) -> float | None:
        if not text:
            return None

        match = re.search(
            r"^\s*1\s+([-+]?\d*\.\d+|[-+]?\d+)",
            text,
            re.MULTILINE,
        )
        if not match:
            return None

        try:
            return float(match.group(1))
        except Exception:
            return None

    def _fail(self, message: str, seed: int | None = None) -> dict:
        log.warning(message)
        return self._simulate_fallback("", error=message, seed=seed)

    def _simulate_fallback(
        self,
        ligand_pdbqt: str,
        error: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        output_file: str | None = None,
        returncode: int | None = None,
        seed: int | None = None,
        cache_key: str | None = None,
    ) -> dict:
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
            "seed": seed,
            "cache_key": cache_key,
            "cache_hit": False,
        }

    @staticmethod
    def _file_info(path: str) -> dict:
        file_path = Path(path)
        return {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else None,
        }

    @staticmethod
    def _count_pdbqt_records(path: str) -> dict:
        file_path = Path(path)
        counts = {"ATOM": 0, "BRANCH": 0, "ROOT": 0, "TORSDOF": 0}
        if not file_path.exists():
            return counts

        with file_path.open(errors="ignore") as handle:
            for line in handle:
                for key in counts:
                    if line.startswith(key):
                        counts[key] += 1
        return counts

    def _copy_debug_file(self, src: str, label: str) -> None:
        if not self.debug_dir:
            return

        source = Path(src)
        if not source.exists():
            return

        try:
            debug_dir = Path(self.debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, debug_dir / label)
        except Exception as exc:
            log.warning("Debug copy failed: %s", exc)
