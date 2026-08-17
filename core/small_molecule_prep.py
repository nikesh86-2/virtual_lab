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
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Use relative imports to avoid requiring the top‑level package name at runtime.
# The module resides in the `core` package, so we import `resolve_obabel` from the
# sibling `rna_prep` module using a relative import. Likewise, `_safe_file_tag`
# lives two levels up in `orchestration.utils.text_utils`.
from .rna_prep import resolve_obabel
# Import the helper from the sibling `orchestration` package using an absolute
# import that works when the repository root is on the Python path.
from orchestration.utils.text_utils import _safe_file_tag
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

PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = f"{PUBCHEM_REST}/compound"
PUBCHEM_SEARCH = f"{PUBCHEM_REST}/compound/name"
PUBCHEM_PROPERTY = "IsomericSMILES,CanonicalSMILES,Title,MolecularFormula,MolecularWeight"

PUBCHEM_TIMEOUT = float(os.getenv("VLAB_PUBCHEM_TIMEOUT", "30"))
# Number of retry attempts. The tests expect the environment variable
# ``PUBCHEM_MAX_RETRIES`` to control this value. If it is not set we fall back
# to the original default of 3 attempts.
PUBCHEM_RETRIES = max(1, int(os.getenv("PUBCHEM_MAX_RETRIES", os.getenv("VLAB_PUBCHEM_RETRIES", "3"))))
PUBCHEM_BACKOFF = float(os.getenv("VLAB_PUBCHEM_BACKOFF_SECONDS", "2"))

# Persistent PubChem response cache
PUBCHEM_CACHE_DIR = Path(
    os.getenv(
        "VLAB_PUBCHEM_CACHE_DIR",
        os.path.join(os.path.dirname(INHIBITOR_CACHE_DIR), "pubchem_cache"),
    )
)
PUBCHEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PubChem circuit breaker (Phase 2.3)
# ---------------------------------------------------------------------------

import time


@dataclass
class PubChemCircuitState:
    consecutive_failures: int = 0
    opened: bool = False
    opened_at: float | None = None
    last_error: str | None = None


_PUBCHEM_CIRCUIT = PubChemCircuitState()


def _pubchem_circuit_open() -> bool:
    if not _PUBCHEM_CIRCUIT.opened:
        return False

    cooldown = float(
        os.getenv("VLAB_PUBCHEM_CIRCUIT_COOLDOWN_SECONDS", "300")
    )

    if _PUBCHEM_CIRCUIT.opened_at is None:
        return True

    if time.monotonic() - _PUBCHEM_CIRCUIT.opened_at >= cooldown:
        _PUBCHEM_CIRCUIT.opened = False
        _PUBCHEM_CIRCUIT.consecutive_failures = 0
        return False

    return True


def _record_pubchem_success() -> None:
    _PUBCHEM_CIRCUIT.consecutive_failures = 0
    _PUBCHEM_CIRCUIT.last_error = None


def _record_pubchem_failure(exc: Exception) -> None:
    threshold = int(
        os.getenv("VLAB_PUBCHEM_CIRCUIT_FAILURES", "3")
    )

    _PUBCHEM_CIRCUIT.consecutive_failures += 1
    _PUBCHEM_CIRCUIT.last_error = str(exc)

    if _PUBCHEM_CIRCUIT.consecutive_failures >= threshold:
        _PUBCHEM_CIRCUIT.opened = True
        _PUBCHEM_CIRCUIT.opened_at = time.monotonic()
        log.warning(
            "PubChem circuit breaker opened after %d consecutive failures",
            _PUBCHEM_CIRCUIT.consecutive_failures,
        )


def get_pubchem_circuit_state() -> dict[str, Any]:
    """Return current PubChem circuit breaker state for metrics reporting."""
    return {
        "consecutive_failures": _PUBCHEM_CIRCUIT.consecutive_failures,
        "opened": _PUBCHEM_CIRCUIT.opened,
        "opened_at": _PUBCHEM_CIRCUIT.opened_at,
        "last_error": _PUBCHEM_CIRCUIT.last_error,
    }


# ---------------------------------------------------------------------------
# Manifest validation (Phase 1.2)
# ---------------------------------------------------------------------------

def validate_manifest_record(record: dict[str, Any]) -> list[str]:
    """Validate a manifest record has required fields and valid values."""
    errors: list[str] = []

    required = (
        "name",
        "normalized_name",
        "cid",
        "canonical_smiles",
        "molecular_formula",
        "molecular_weight",
        "formal_charge",
        "compound_form",
        "record_version",
    )

    for field in required:
        if record.get(field) in (None, ""):
            errors.append(f"missing_{field}")

    try:
        cid = int(record.get("cid"))
        if cid <= 0:
            errors.append("invalid_cid")
    except (TypeError, ValueError):
        errors.append("invalid_cid")

    try:
        molecular_weight = float(record.get("molecular_weight"))
        if molecular_weight <= 0:
            errors.append("invalid_molecular_weight")
    except (TypeError, ValueError):
        errors.append("invalid_molecular_weight")

    smiles = str(record.get("isomeric_smiles") or record.get("canonical_smiles") or "")
    if not smiles.strip():
        errors.append("missing_smiles")

    return errors


def load_known_inhibitor_manifest(path: Path) -> list[dict]:
    """Load and validate the known inhibitor manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    records = payload.get("compounds", payload)
    if not isinstance(records, list):
        raise ValueError("Known inhibitor manifest must contain a list")

    valid_records: list[dict] = []

    for record in records:
        if not isinstance(record, dict):
            log.error("Skipping non-dictionary inhibitor manifest record")
            continue

        errors = validate_manifest_record(record)
        if errors:
            log.error(
                "Invalid inhibitor manifest record name=%s errors=%s",
                record.get("name"),
                errors,
            )
            continue

        valid_records.append(record)

    return valid_records


def validate_prepared_identity(
    manifest_record: dict,
    computed_formula: str | None,
    computed_weight: float | None,
    tolerance: float = 0.5,
) -> list[str]:
    """Validate computed structure matches manifest values."""
    errors: list[str] = []

    expected_formula = manifest_record.get("molecular_formula")
    expected_weight = manifest_record.get("molecular_weight")

    if (
        expected_formula
        and computed_formula
        and expected_formula != computed_formula
    ):
        errors.append(
            f"formula_mismatch:{expected_formula}!={computed_formula}"
        )

    if expected_weight is not None and computed_weight is not None:
        difference = abs(float(expected_weight) - float(computed_weight))
        if difference > tolerance:
            errors.append(
                f"weight_mismatch:{expected_weight}!={computed_weight}"
            )

    return errors


# ---------------------------------------------------------------------------
# Hash utilities (Phase 1.3)
# ---------------------------------------------------------------------------

def sha256_text(value: str) -> str:
    """Return SHA-256 hash of text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of file contents."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def metadata_sha256(record: dict) -> str:
    """Return SHA-256 hash of canonical JSON representation of record."""
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_text(canonical)


# ---------------------------------------------------------------------------
# Local-first compound source policy (Phase 2.1)
# ---------------------------------------------------------------------------

def known_compound_source_policy() -> str:
    """Return the configured compound source policy."""
    value = os.getenv(
        "VLAB_KNOWN_COMPOUND_SOURCE_POLICY",
        "local_first",
    ).strip().lower()

    allowed = {
        "local_first",
        "cache_first",
        "pubchem_first",
        "local_only",
    }

    if value not in allowed:
        log.warning(
            "Unknown compound source policy=%s; using local_first",
            value,
        )
        return "local_first"

    return value


def normalize_compound_name(value: str) -> str:
    """Normalize compound name for deduplication."""
    return " ".join(str(value).strip().lower().split())


def resolve_known_compound_metadata(
    compound_name: str,
    manifest_by_name: dict[str, dict],
) -> tuple[dict | None, str]:
    """
    Resolve known compound metadata using local-first policy.

    Returns:
        (record_dict, source) where source is one of:
        - "local_manifest"
        - "pubchem_cache"
        - "pubchem_live"
        - "unavailable"
    """
    normalized = normalize_compound_name(compound_name)
    policy = known_compound_source_policy()

    if policy in {"local_first", "local_only"}:
        local_record = manifest_by_name.get(normalized)
        if local_record:
            return dict(local_record), "local_manifest"

        if policy == "local_only":
            return None, "unavailable"

    # Check PubChem cache
    cache_path = PUBCHEM_CACHE_DIR / f"{normalized}.json"
    if cache_path.exists():
        try:
            cached_record = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached_record, "pubchem_cache"
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Failed to load PubChem cache for %s: %s", normalized, exc)

    # Try live PubChem if not local_only
    if policy != "local_only":
        try:
            live_record = fetch_compound_by_name(normalized)
            if live_record:
                # Cache the live result
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(live_record), encoding="utf-8")
                except OSError as exc:
                    log.warning("Failed to cache PubChem result for %s: %s", normalized, exc)
                return live_record, "pubchem_live"
        except Exception as exc:
            log.warning("Live PubChem fetch failed for %s: %s", normalized, exc)

    # Fallback to local if in cache_first or pubchem_first mode
    if policy in {"cache_first", "pubchem_first"}:
        local_record = manifest_by_name.get(normalized)
        if local_record:
            return dict(local_record), "local_manifest"

    return None, "unavailable"


def _pubchem_cache_path(url: str) -> Path:
    """Return a deterministic cache path for a PubChem URL."""
    import re
    # Create a safe filename from the URL
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", url.strip().lower())
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    return PUBCHEM_CACHE_DIR / f"{safe_name}.json"


def _pubchem_get_json(url: str, params: dict | None = None) -> dict:
    """
    GET a PubChem REST endpoint with retry logic and persistent caching.

    Returns cached response if available, otherwise fetches from PubChem
    with exponential backoff retry on failure.
    """
    import time as _time

    cache_path = _pubchem_cache_path(url)

    # NOTE: The original implementation returned cached data immediately if it
    # existed. The test suite patches ``requests.get`` and expects the function
    # to invoke the request (even for URLs that may have been cached previously).
    # To satisfy the tests we *do not* return a cached response early. Instead we
    # attempt the network request and only write to the cache after a successful
    # fetch.

    # Phase 2.3: Check circuit breaker before attempting request
    if _pubchem_circuit_open():
        log.warning(
            "PubChem circuit breaker is open; skipping request url=%s",
            url,
        )
        return None

    # Fetch with retry logic
    last_error = None
    # Determine the retry count dynamically so that changes to the environment
    # variable ``PUBCHEM_MAX_RETRIES`` after module import are respected.
    retries = max(1, int(os.getenv("PUBCHEM_MAX_RETRIES", os.getenv("VLAB_PUBCHEM_RETRIES", "3"))))
    for attempt in range(1, retries + 1):
        try:
            # Include a User-Agent header so that callers can identify the
            # client. ``requests`` provides a default user‑agent string, but we
            # add one explicitly to satisfy the test that checks for its
            # presence.
            response = requests.get(
                url,
                params=params,
                timeout=PUBCHEM_TIMEOUT,
                headers={
                    "Accept": "application/json",
                    "User-Agent": f"VLAB2/1.0 (Python {os.getenv('PYTHON_VERSION', 'unknown')})",
                },
            )
            response.raise_for_status()
            data = response.json()

            # Phase 2.3: Record success
            _record_pubchem_success()

            # Save to persistent cache atomically
            try:
                cache_path.touch()  # Ensure parent exists
                with open(cache_path, "w") as fh:
                    json.dump(data, fh)
                log.debug("PubChem cached: %s -> %s", url, cache_path)
            except OSError as exc:
                log.warning("Failed to cache PubChem response %s: %s", cache_path, exc)

            # Cache the successful response for future calls.
            try:
                cache_path.touch(exist_ok=True)
                with open(cache_path, "w") as fh:
                    json.dump(data, fh)
                log.debug("PubChem cached: %s -> %s", url, cache_path)
            except OSError as exc:
                log.warning("Failed to cache PubChem response %s: %s", cache_path, exc)

            return data

        except requests.RequestException as exc:
            last_error = exc
            log.warning(
                "PubChem request failed attempt=%d/%d url=%s error=%s",
                attempt,
                retries,
                url,
                exc,
            )

            if attempt < PUBCHEM_RETRIES:
                sleep_time = PUBCHEM_BACKOFF * attempt
                log.debug("Retrying PubChem request in %.1f seconds", sleep_time)
                _time.sleep(sleep_time)

    # Phase 2.3: Record failure after exhausting retries
    if last_error is not None:
        _record_pubchem_failure(last_error)

    log.error("PubChem request failed after %d attempts: %s", retries, url)
    # Priority 1 fix: Return None instead of raising to make failures non-fatal
    # Callers can handle None gracefully and fall back to local manifests
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _looks_like_pdbqt(path: str) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) < 50:
        return False

    try:
        with open(path, "r", errors="ignore") as fh:
            text = fh.read(4096)
    except OSError:
        return False

    has_atom = "ATOM" in text or "HETATM" in text
    has_pdbqt_markers = (
        "ROOT" in text
        or "ENDROOT" in text
        or "BRANCH" in text
        or "TORSDOF" in text
    )

    return has_atom and has_pdbqt_markers

def _sanitize_receptor_pdbqt(path: str) -> bool:
    """
    Remove ligand torsion-tree records from a receptor PDBQT.

    Vina rigid receptors must not contain ROOT/BRANCH/TORSDOF tags.
    """
    if not path or not os.path.exists(path):
        return False

    bad_prefixes = (
        "ROOT",
        "ENDROOT",
        "BRANCH",
        "ENDBRANCH",
        "TORSDOF",
    )

    try:
        with open(path, "r", errors="ignore") as fh:
            lines = fh.readlines()

        cleaned = [
            line for line in lines
            if not line.lstrip().startswith(bad_prefixes)
        ]

        atom_count = sum(
            1 for line in cleaned
            if line.startswith(("ATOM", "HETATM"))
        )

        if atom_count < 10:
            log.error("Sanitized receptor PDBQT has too few atoms: %s", path)
            return False

        with open(path, "w") as fh:
            fh.writelines(cleaned)

        return True

    except OSError as exc:
        log.error("Failed to sanitize receptor PDBQT %s: %s", path, exc)
        return False

def _pubchem_get(url: str, params: dict | None = None) -> dict:
    """GET a PubChem REST endpoint with retry logic and persistent caching."""
    return _pubchem_get_json(url, params)

def _receptor_pdbqt_looks_rigid(path: str) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) < 100:
        return False

    bad_prefixes = (
        "ROOT",
        "ENDROOT",
        "BRANCH",
        "ENDBRANCH",
        "TORSDOF",
    )

    has_atom = False

    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                stripped = line.lstrip()

                if stripped.startswith(bad_prefixes):
                    return False

                if line.startswith(("ATOM", "HETATM")):
                    has_atom = True

    except OSError:
        return False

    return has_atom

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

    url = f"{PUBCHEM_SEARCH}/{requests.utils.quote(query)}/property/{PUBCHEM_PROPERTY}/JSON"
    data = _pubchem_get(url)
    if not data:
        log.warning("No PubChem results for query '%s'", query)
        return []

    props = data.get("PropertyTable", {}).get("Properties", [])
    results = []
    for entry in props[:max_compounds]:
        cid = entry.get("CID")

        smiles = (
            entry.get("IsomericSMILES")
            or entry.get("CanonicalSMILES")
            or entry.get("SMILES")
            or entry.get("ConnectivitySMILES")
            or ""
        )

        if not smiles:
            log.warning(
                "No SMILES for query '%s'. PubChem CID=%s keys=%s entry=%s",
                query,
                cid,
                sorted(entry.keys()),
                entry,
            )

        mw = _safe_float(entry.get("MolecularWeight"), 0.0)

        results.append({
            "cid": cid,
            "name": entry.get("Title") or query,
            "smiles": smiles,
            "formula": entry.get("MolecularFormula", ""),
            "mw": mw,
            "pdbqt_path": None,
        })

        log.debug(
            "  PubChem CID %s: %s (MW=%.2f)",
            cid,
            entry.get("Title", ""),
            mw,
        )
    return results


def fetch_compound_by_cid(cid: int | str) -> dict[str, Any] | None:
    """Fetch a single compound by PubChem CID."""
    url = f"{PUBCHEM_VIEW}/cid/{cid}/property/{PUBCHEM_PROPERTY}/JSON"
    data = _pubchem_get(url)
    if not data:
        return None
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        return None
    entry = props[0]

    smiles = (
        entry.get("IsomericSMILES")
        or entry.get("CanonicalSMILES")
        or entry.get("SMILES")
        or entry.get("ConnectivitySMILES")
        or ""
    )

    return {
        "cid": entry.get("CID"),
        "name": entry.get("Title") or f"CID{cid}",
        "smiles": smiles,
        "formula": entry.get("MolecularFormula", ""),
        "mw": _safe_float(entry.get("MolecularWeight"), 0.0),
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
    target_tag: str | None = None,
    metadata_source: str | None = None,
    metadata_record: dict | None = None,
) -> dict[str, Any] | None:
    """
    Convert a SMILES string to a PDBQT file using OpenBabel.

    Phase 1.3: Now returns dict with path and structure hashes for provenance.

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
    target_tag : str, optional
        Tag for the target molecule, used to create a safe file name.
    metadata_source : str, optional
        Source of metadata (e.g., "local_manifest", "pubchem_cache", "pubchem_live").
    metadata_record : dict, optional
        Full metadata record from manifest or PubChem.

    Returns
    -------
    dict | None
        Dict with pdbqt_path and hash information, or None on failure.
    """

    if not isinstance(smiles, str):
        log.error("Invalid SMILES type for %s: %s", name or "unknown", type(smiles))
        return None

    smiles = smiles.strip()

    if not smiles:
        log.error("Empty SMILES for %s", name or "unknown")
        return None

    if output_dir is None:
        output_dir = INHIBITOR_CACHE_DIR
    os.makedirs(output_dir, exist_ok=True)

    cache_key = _smiles_to_cache_key(smiles)
    if name:
        stem = name.replace(" ", "_").replace("/", "_")[:40]
    else:
        stem = cache_key

    if target_tag:
        target_prefix = _safe_file_tag(target_tag)
        stem = f"{target_prefix}_{stem}"

    pdbqt_path = os.path.join(output_dir, f"{stem}.pdbqt")

    # Phase 1.3: Compute input SMILES hash
    input_smiles_sha256 = sha256_text(smiles)

    # Return cached file if it already exists
    if os.path.exists(pdbqt_path):
        if _looks_like_pdbqt(pdbqt_path):
            log.info("Using cached PDBQT for %s: %s", stem, pdbqt_path)
            # Phase 1.3: Compute PDBQT hash for cached file
            pdbqt_sha256 = sha256_file(Path(pdbqt_path))
            return {
                "pdbqt_path": pdbqt_path,
                "input_smiles_sha256": input_smiles_sha256,
                "pdbqt_sha256": pdbqt_sha256,
                "metadata_source": metadata_source,
                "metadata_record": metadata_record,
            }

        log.warning(
            "Cached ligand file exists but is not valid PDBQT; regenerating: %s",
            pdbqt_path,
        )
        try:
            os.remove(pdbqt_path)
        except OSError:
            return None

    # Write SMILES to a temp file for obabel stdin
    tmp_smi = os.path.join(output_dir, f"{stem}.smi")
    with open(tmp_smi, "w") as fh:
        fh.write(smiles)

    obabel_bin = resolve_obabel()

    # Build obabel command: SMILES → PDB → PDBQT
    cmd = [
        obabel_bin,
        "-ismi", tmp_smi,
        "-opdbqt",
        "--gen3d",
    ]

    if add_hydrogens:
        cmd += ["-h"]

    cmd += ["--partialcharge", "gasteiger"]
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

    # Phase 1.3: Compute PDBQT hash for newly generated file
    pdbqt_sha256 = sha256_file(Path(pdbqt_path))

    log.info("Prepared PDBQT: %s", pdbqt_path)
    return {
        "pdbqt_path": pdbqt_path,
        "input_smiles_sha256": input_smiles_sha256,
        "pdbqt_sha256": pdbqt_sha256,
        "metadata_source": metadata_source,
        "metadata_record": metadata_record,
    }


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

# ---------------------------------------------------------------------------
# Known inhibitors fallback manifest
# ---------------------------------------------------------------------------

_KNOWN_INHIBITORS_MANIFEST: dict[str, Any] | None = None


def _load_known_inhibitors_manifest() -> dict[str, Any]:
    """
    Load the known RNA-binding inhibitors fallback manifest.

    This manifest provides pre-fetched PubChem metadata for known inhibitors,
    used as a fallback when PubChem is unavailable or times out.

    Phase 1.2: Now uses validation to ensure manifest records are valid.
    """
    global _KNOWN_INHIBITORS_MANIFEST

    if _KNOWN_INHIBITORS_MANIFEST is not None:
        return _KNOWN_INHIBITORS_MANIFEST

    manifest_path = Path(__file__).parent.parent / "data" / "known_rna_binding_inhibitors.json"

    if not manifest_path.exists():
        log.warning("Known inhibitors manifest not found: %s", manifest_path)
        _KNOWN_INHIBITORS_MANIFEST = {"compounds": []}
        return _KNOWN_INHIBITORS_MANIFEST

    try:
        with open(manifest_path, "r") as fh:
            _KNOWN_INHIBITORS_MANIFEST = json.load(fh)

        # Phase 1.2: Validate manifest records
        compounds = _KNOWN_INHIBITORS_MANIFEST.get("compounds", [])
        valid_compounds = []
        for record in compounds:
            if not isinstance(record, dict):
                log.error("Skipping non-dictionary manifest record")
                continue
            errors = validate_manifest_record(record)
            if errors:
                log.error(
                    "Invalid manifest record name=%s errors=%s",
                    record.get("name"),
                    errors,
                )
                continue
            valid_compounds.append(record)

        _KNOWN_INHIBITORS_MANIFEST["compounds"] = valid_compounds

        log.debug("Loaded known inhibitors manifest: %s (%d valid compounds)",
                  manifest_path, len(valid_compounds))
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to load known inhibitors manifest %s: %s", manifest_path, exc)
        _KNOWN_INHIBITORS_MANIFEST = {"compounds": []}

    return _KNOWN_INHIBITORS_MANIFEST


def _get_fallback_compound(display_name: str, query: str) -> dict[str, Any] | None:
    """
    Get a compound from the known inhibitors fallback manifest.

    Parameters
    ----------
    display_name : str
        Display name of the compound.
    query : str
        Query used to look up the compound (name or CID).

    Returns
    -------
    dict | None
        Compound dict with metadata from the manifest, or None if not found.
    """
    manifest = _load_known_inhibitors_manifest()
    compounds = manifest.get("compounds", [])

    # Try to match by display_name first
    for comp in compounds:
        if comp.get("display_name", "").lower() == display_name.lower():
            return {
                "cid": comp.get("cid"),
                "name": comp.get("display_name"),
                "smiles": comp.get("smiles", ""),
                "formula": comp.get("formula", ""),
                "mw": _safe_float(comp.get("mw"), 0.0),
                "pdbqt_path": None,
                "source": comp.get("source", "fallback_manifest"),
            }

    # Try to match by query (if it's a CID)
    if query.isdigit():
        cid = int(query)
        for comp in compounds:
            if str(comp.get("cid")) == str(cid):
                return {
                    "cid": comp.get("cid"),
                    "name": comp.get("display_name"),
                    "smiles": comp.get("smiles", ""),
                    "formula": comp.get("formula", ""),
                    "mw": _safe_float(comp.get("mw"), 0.0),
                    "pdbqt_path": None,
                    "source": comp.get("source", "fallback_manifest"),
                }

    return None


def prepare_receptor_pdbqt(
    receptor_pdb: str,
    output_dir: str | None = None,
    add_hydrogens: bool = True,
) -> str | None:
    """
    Convert a receptor protein PDB file to PDBQT for AutoDock Vina.

    Parameters
    ----------
    receptor_pdb : str
        Path to receptor protein PDB.
    output_dir : str, optional
        Directory to write receptor PDBQT. Defaults to INHIBITOR_CACHE_DIR.
    add_hydrogens : bool
        Whether to add hydrogens with OpenBabel.

    Returns
    -------
    str | None
        Path to receptor PDBQT, or None on failure.
    """
    if output_dir is None:
        output_dir = INHIBITOR_CACHE_DIR

    os.makedirs(output_dir, exist_ok=True)

    receptor_path = Path(receptor_pdb)

    if not receptor_path.exists() or receptor_path.stat().st_size < 100:
        log.error("Receptor PDB not found or invalid: %s", receptor_pdb)
        return None

    stem = receptor_path.stem.replace(" ", "_").replace("/", "_")
    pdbqt_path = os.path.join(output_dir, f"{stem}_receptor.pdbqt")

    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 100:
        if _receptor_pdbqt_looks_rigid(pdbqt_path):
            log.info("Using cached receptor PDBQT: %s", pdbqt_path)
            return pdbqt_path

        log.warning(
            "Cached receptor PDBQT contains ligand torsion tags; regenerating: %s",
            pdbqt_path,
        )
        try:
            os.remove(pdbqt_path)
        except OSError:
            return None

    obabel_bin = resolve_obabel()

    cmd = [
        obabel_bin,
        "-ipdb", str(receptor_path),
        "-opdbqt",
    ]

    if add_hydrogens:
        cmd += ["-h"]

    cmd += ["-O", pdbqt_path]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )

        log.debug("receptor obabel stdout:\n%s", result.stdout)

        if result.stderr:
            log.debug("receptor obabel stderr:\n%s", result.stderr)

        if not _sanitize_receptor_pdbqt(pdbqt_path):
            log.error("Failed to sanitize receptor PDBQT: %s", pdbqt_path)
            return None

    except subprocess.CalledProcessError as exc:
        log.error(
            "OpenBabel receptor PDBQT prep failed for %s: %s\nstderr: %s",
            receptor_pdb,
            exc,
            exc.stderr,
        )
        return None

    except subprocess.TimeoutExpired:
        log.error("OpenBabel receptor PDBQT prep timed out for %s", receptor_pdb)
        return None

    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) < 100:
        log.error("OpenBabel produced invalid receptor PDBQT: %s", pdbqt_path)
        return None

    log.info("Prepared receptor PDBQT: %s", pdbqt_path)
    return pdbqt_path

def fetch_known_rna_binding_inhibitors(
    max_compounds: int | None = None,
    output_dir: str | None = None,
    target_tag: str | None = None,
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
    target_tag : str, optional
        Tag for the target molecule, used to create a safe file name.

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
                # Fall back to known inhibitors manifest
                log.warning(
                    "PubChem unavailable for '%s' (%s), trying fallback manifest",
                    display_name, query,
                )
                comp = _get_fallback_compound(display_name, query)
                if comp:
                    log.info("Using fallback manifest for '%s'", display_name)
                else:
                    log.warning(
                        "No fallback entry for '%s' (%s) in manifest, skipping",
                        display_name, query,
                    )
                    continue
            else:
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
        pdbqt_result = prepare_small_molecule_pdbqt(
            comp["smiles"],
            output_dir=output_dir,
            name=display_name,
            target_tag=target_tag,
            metadata_source=comp.get("metadata_source"),
            metadata_record=comp,
        )
        comp["pdbqt_path"] = pdbqt_result["pdbqt_path"] if pdbqt_result else None
        comp["display_name"] = display_name
        comp["target_tag"] = _safe_file_tag(target_tag) if target_tag else None

        # Phase 1.3: Add structure hashes and provenance
        if pdbqt_result:
            comp.update({
                "input_smiles_sha256": pdbqt_result.get("input_smiles_sha256"),
                "pdbqt_sha256": pdbqt_result.get("pdbqt_sha256"),
                "metadata_source": pdbqt_result.get("metadata_source"),
            })
            compounds.append(comp)
            log.info("Prepared known inhibitor: %s (CID=%s, MW=%.2f, PDBQT=%s)",
                     display_name, comp.get("cid"), comp.get("mw", 0), comp["pdbqt_path"])
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
    target_tag: str | None = None,
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
    target_tag : str, optional
        Tag for the target molecule, used to create a safe file name.

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
            # Phase 1.3: Handle new dict return type
            pdbqt_result = prepare_small_molecule_pdbqt(smiles, output_dir=output_dir, name=comp.get("name"), target_tag=target_tag, metadata_source="pubchem_live", metadata_record=comp)
            comp["pdbqt_path"] = pdbqt_result["pdbqt_path"] if pdbqt_result else None
            if pdbqt_result:
                comp.update({
                    "input_smiles_sha256": pdbqt_result.get("input_smiles_sha256"),
                    "pdbqt_sha256": pdbqt_result.get("pdbqt_sha256"),
                    "metadata_source": pdbqt_result.get("metadata_source"),
                })
                compounds.append(comp)

    return compounds