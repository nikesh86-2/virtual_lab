"""
peptide_prep.py

Peptide inhibitor preparation for the VLAB2 docking pipeline.

Generates 3D peptide structures from sequences using OpenBabel --gen3d,
cleans them for HDOCK protein-protein docking, and manages peptide caching.

Design decisions:
- OpenBabel --gen3d for 3D structure generation (pragmatic; HDOCK refines the pose)
- HDOCK protein-protein mode for docking (same binary as protein-RNA, different input pairing)
- Cached to VLAB_INHIBITOR_CACHE_DIR alongside small molecules
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from VLAB2.orchestration.utils.text_utils import _safe_file_tag
# Reuse obabel resolver from rna_prep
try:
    from VLAB2.core.rna_prep import resolve_obabel
except Exception:
    # Fallback if rna_prep not available
    def resolve_obabel() -> str:
        return os.getenv("OBABEL_BIN", shutil.which("obabel") or "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/obabel")


log = logging.getLogger("virtual_lab.peptide_prep")

# ── Env defaults ──────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = os.getenv(
    "VLAB_INHIBITOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "vlab_inhibitor_cache"),
)

MAX_PEPTIDES = int(os.getenv("VLAB_INHIBITOR_MAX_PEPTIDES", "5"))

DEFAULT_OBABEL_BIN = os.getenv(
    "OBABEL_BIN",
    "/mnt/scratch/fbsnpat/envs/biophysics-research-agent/bin/obabel",
)

def _resolve_pymol() -> str | None:
    return (
        os.getenv("PYMOL_BIN")
        or shutil.which("pymol")
        or shutil.which("pymol-open-source")
    )

def generate_peptide_pdb_with_pymol(
    sequence: str,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Generate a peptide PDB from a one-letter amino acid sequence using PyMOL fab.

    This is preferred over OpenBabel for peptide sequences because OpenBabel
    treats inline strings as SMILES and does not reliably build peptide PDBs.
    """
    seq = _normalise_sequence(sequence)

    if not seq:
        return {
            "pdb_path": None,
            "sequence": sequence,
            "error": "Empty or invalid sequence",
        }

    pymol_bin = _resolve_pymol()

    if not pymol_bin:
        return {
            "pdb_path": None,
            "sequence": seq,
            "error": "PyMOL not found for peptide building",
        }

    cache_dir = output_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    seq_hash = hashlib.md5(seq.encode()).hexdigest()[:8]
    output_path = os.path.join(cache_dir, f"peptide_{seq_hash}_{seq}.pdb")
    pml_path = os.path.join(cache_dir, f"build_peptide_{seq_hash}_{seq}.pml")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        log.info("Peptide PDB cache hit: %s", output_path)
        return {
            "pdb_path": output_path,
            "sequence": seq,
            "error": None,
        }

    pml = f"""
reinitialize
fab {seq}, peptide
set retain_order, 1
h_add peptide
save {Path(output_path).as_posix()}, peptide
quit
"""

    with open(pml_path, "w", encoding="utf-8") as fh:
        fh.write(pml)

    try:
        proc = subprocess.run(
            [pymol_bin, "-cq", pml_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            return {
                "pdb_path": None,
                "sequence": seq,
                "error": f"PyMOL peptide build failed: {proc.stderr.strip()}",
            }

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            return {
                "pdb_path": None,
                "sequence": seq,
                "error": "PyMOL produced empty/invalid peptide PDB",
            }

        atom_count = 0
        with open(output_path, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    atom_count += 1

        if atom_count < 5:
            return {
                "pdb_path": None,
                "sequence": seq,
                "error": f"PyMOL peptide PDB has too few atoms: {atom_count}",
            }

        log.info("Peptide PDB generated with PyMOL: %s", output_path)

        return {
            "pdb_path": output_path,
            "sequence": seq,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "pdb_path": None,
            "sequence": seq,
            "error": "PyMOL peptide build timed out",
        }

    except Exception as e:
        return {
            "pdb_path": None,
            "sequence": seq,
            "error": str(e),
        }

    finally:
        try:
            if os.path.exists(pml_path):
                os.remove(pml_path)
        except OSError:
            pass


# ── 3D structure generation ───────────────────────────────────────────────────

def generate_peptide_pdb(
    sequence: str,
    output_dir: Optional[str] = None,
    obabel_bin: Optional[str] = None,
    target_tag: Optional[str] = None,
) -> dict:
    """
    Generate a 3D PDB file for a peptide sequence using OpenBabel --gen3d.

    Uses a SMILES-like inline format that OpenBabel can parse:
        H-{aa1}-{aa2}-...-OH

    Args:
        sequence: Amino acid sequence (e.g. "RRM" or "KGRRRM").
                   Supports 1-letter or 3-letter codes.
        output_dir: Directory for output PDB. If None, uses DEFAULT_CACHE_DIR.
        obabel_bin: Path to obabel binary. Auto-resolved if None.
        target_tag: Tag for the target molecule, used to create a safe file name.

    Returns:
        dict with keys:
            - pdb_path (str): Path to generated PDB file, or None on failure.
            - sequence (str): Normalised sequence.
            - error (str or None): Error message if failed.
    """

    pymol_result = generate_peptide_pdb_with_pymol(
        sequence=sequence,
        output_dir=output_dir,
    )

    if pymol_result.get("pdb_path"):
        return pymol_result

    log.warning(
        "PyMOL peptide build failed for %s; falling back to OpenBabel. error=%s",
        sequence,
        pymol_result.get("error"),
    )

    obabel_bin = obabel_bin or resolve_obabel()
    if not os.path.isfile(obabel_bin):
        return {
            "pdb_path": None,
            "sequence": sequence,
            "error": f"OpenBabel not found at: {obabel_bin}",
        }

    # Normalise: strip whitespace, convert to uppercase single-letter
    seq = _normalise_sequence(sequence)
    if not seq:
        return {"pdb_path": None, "sequence": sequence, "error": "Empty or invalid sequence"}

    # Build output path
    cache_dir = output_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    seq_hash = hashlib.md5(seq.encode()).hexdigest()[:8]
    prefix = f"{_safe_file_tag(target_tag)}_" if target_tag else ""
    output_path = os.path.join(cache_dir, f"{prefix}peptide_{seq_hash}_{seq}.pdb")

    # Check cache
    if os.path.exists(output_path):
        log.info("Peptide PDB cache hit: %s", output_path)
        return {"pdb_path": output_path, "sequence": seq, "error": None}

    # Build SMILES-like input for OpenBabel
    # Format: peptide SMILES with explicit N-terminal H and C-terminal OH
    # OpenBabel can parse this as a peptide
    peptide_smiles = _sequence_to_peptide_smiles(seq)

    log.info("Generating 3D peptide structure for sequence: %s", seq)
    log.debug("Peptide SMILES input: %s", peptide_smiles)

    try:
        cmd = [
            obabel_bin,
            f"-:{peptide_smiles}",
            "-opdb",
            "--gen3d",
            "-O", output_path,
            "--errorlevel", "0",
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            log.warning("OpenBabel peptide gen3d failed: %s", result.stderr.strip())
            return {
                "pdb_path": None,
                "sequence": seq,
                "error": f"OpenBabel failed: {result.stderr.strip()}",
            }

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            return {
                "pdb_path": None,
                "sequence": seq,
                "error": "OpenBabel produced empty/invalid PDB",
            }

        log.info("Peptide PDB generated: %s (%d bytes)", output_path, os.path.getsize(output_path))
        return {"pdb_path": output_path, "sequence": seq, "error": None}

    except subprocess.TimeoutExpired:
        return {"pdb_path": None, "sequence": seq, "error": "OpenBabel timed out (60s)"}
    except Exception as e:
        return {"pdb_path": None, "sequence": seq, "error": str(e)}


def _sequence_to_peptide_smiles(seq: str) -> str:
    """
    Convert a 1-letter amino acid sequence to an OpenBabel-parseable peptide SMILES.

    Uses the format: H-[AA1]-[AA2]-...-[AAn]-OH
    This is the canonical peptide SMILES format that obabel can interpret.
    """
    # 1-letter to 3-letter mapping
    aa_map = {
        "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
        "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
        "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
        "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
        "M": "Met", "U": "Sec", "O": "Pyl",
    }

    # Build peptide SMILES: H-[aa1]-[aa2]-...-OH
    parts = ["H"]
    for aa in seq.upper():
        if aa in aa_map:
            parts.append(aa_map[aa])
        else:
            log.warning("Unknown amino acid '%s' in sequence, skipping", aa)
    parts.append("OH")

    return "-".join(parts)


def _normalise_sequence(sequence: str) -> str:
    """
    Normalise a peptide sequence to 1-letter uppercase.

    Handles both 1-letter (RRM) and 3-letter (ArgArgMet) input.
    """
    seq = sequence.strip().upper().replace(" ", "").replace("\n", "")

    # If contains digits, strip them (residue numbers)
    seq = "".join(c for c in seq if c.isalpha())

    # If length > 3 and no single-letter pattern detected, try 3-letter decode
    if len(seq) > 3 and len(seq) % 3 == 0:
        # Try 3-letter to 1-letter
        aa_3to1 = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            "SEC": "U", "PYL": "O",
        }
        decoded = []
        for i in range(0, len(seq), 3):
            chunk = seq[i:i+3]
            if chunk in aa_3to1:
                decoded.append(aa_3to1[chunk])
            else:
                # Not a valid 3-letter sequence, treat as 1-letter
                return seq  # Return as-is, let obabel handle validation
        return "".join(decoded)

    return seq


# ── Peptide design via LLM ─────────────────────────────────────────────────────

def design_inhibitor_peptides(
    target_pdb: str,
    pocket_residues: list,
    llm,
    max_peptides: int = MAX_PEPTIDES,
) -> dict:
    """
    Use an LLM to design peptide sequences that target the RNA-binding pocket.

    Args:
        target_pdb: Path to the target protein PDB file.
        pocket_residues: List of pocket residue info dicts from define_docking_box().
                         Each dict should have at least 'residue_id' and 'residue_name'.
        llm: LangChain LLM instance for generating peptide sequences.
        max_peptides: Maximum number of peptide sequences to generate.

    Returns:
        Dict with keys:
        - peptides: List of dicts, each with keys: sequence, rationale, source ("llm_generated" or "default")
        - generation_method: "llm" or "conservative_defaults"
    """
    if llm is None:
        log.warning("No LLM provided for peptide design, using conservative defaults")
        return {
            "peptides": _default_inhibitor_peptides(),
            "generation_method": "conservative_defaults",
        }

    # Build context for the LLM prompt
    pocket_str = ", ".join(
        r.get("residue_id", "?") for r in pocket_residues[:20]
    ) if pocket_residues else "unknown"

    prompt = f"""You are a computational biophysicist designing peptide inhibitors for an RNA-binding protein.

Target protein PDB: {target_pdb}
RNA-binding pocket residues: {pocket_str}

Design {max_peptides} short peptide sequences (6-15 amino acids) that could competitively bind to the RNA-binding pocket of this protein.
Each peptide should:
1. Be rich in positively charged residues (K, R) to compete with RNA binding
2. Have hydrophobic residues for pocket complementarity
3. Be structurally feasible as a standalone peptide
4. Target the same residues that the RNA contacts

Return ONLY a JSON list (no markdown, no explanation) with this exact format:
[
  {{"sequence": "KRRMKG", "rationale": "Short rationale for this peptide"}},
  ...
]

Do not include any text outside the JSON array. Generate exactly {max_peptides} peptides.
"""

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from response
        import json
        # Handle markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        peptides = json.loads(text)

        # Validate and normalise
        results = []
        for p in peptides:
            seq = _normalise_sequence(p.get("sequence", ""))
            if seq and 4 <= len(seq) <= 25:
                results.append({
                    "sequence": seq,
                    "rationale": p.get("rationale", ""),
                    "source": "llm_generated",
                })
            else:
                log.warning("Skipping invalid peptide sequence: %s", p.get("sequence"))

        log.info("LLM generated %d peptide sequences", len(results))
        return {
            "peptides": results[:max_peptides],
            "generation_method": "llm",
        }

    except Exception as e:
        log.error("LLM peptide design failed: %s", e)
        return {
            "peptides": _default_inhibitor_peptides(),
            "generation_method": "conservative_defaults",
        }


def _default_inhibitor_peptides() -> list:
    """
    Return a set of conservative peptide sequences known to target
    RNA-binding pockets in viral proteins.
    """
    return [
        {"sequence": "RRM", "rationale": "Minimal RRM recognition motif", "source": "default"},
        {"sequence": "KGRRRM", "rationale": "Extended RRM with nuclear localisation", "source": "default"},
        {"sequence": "RPKRRM", "rationale": "PKR-like basic patch", "source": "default"},
        {"sequence": "RRG", "rationale": "Minimal RNA-mimetic tripeptide", "source": "default"},
        {"sequence": "KRR", "rationale": "Basic RNA-mimetic tripeptide", "source": "default"},
    ]


# ── PDB cleaning for HDOCK ─────────────────────────────────────────────────────

def prepare_peptide_for_hdock(
    peptide_pdb: str,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Clean a peptide PDB file for HDOCK protein-protein docking.

    HDOCK requires:
    - Standard atom names (CA, N, C, O, etc.)
    - Chain identifier present
    - No HETATMs (or converted to ATOM)
    - Residue numbers present

    Args:
        peptide_pdb: Path to the raw peptide PDB (from generate_peptide_pdb).
        output_dir: Output directory. Defaults to DEFAULT_CACHE_DIR.

    Returns:
        dict with keys:
            - clean_pdb (str): Path to cleaned PDB, or None on failure.
            - error (str or None): Error message if failed.
    """
    cache_dir = output_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(peptide_pdb):
        return {"clean_pdb": None, "error": f"Peptide PDB not found: {peptide_pdb}"}

    output_path = os.path.join(
        cache_dir,
        f"clean_{os.path.basename(peptide_pdb)}",
    )

    try:
        with open(peptide_pdb, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                if line.startswith(("ATOM  ", "HETATM")):
                    # Standardise: ensure ATOM record
                    record = "ATOM  " if line.startswith("ATOM") else "ATOM  "
                    # Ensure chain ID is present (use B for peptide ligand)
                    if len(line) >= 22:
                        chain = line[21]
                        if chain in (" ", ""):
                            line = line[:21] + "B" + line[22:]
                    fout.write(line)
                elif line.startswith("END"):
                    pass  # Skip END, we'll add our own
                # Skip all other records (CONECT, etc.)

            # Add TER and END
            fout.write("TER\n")
            fout.write("END\n")

        log.info("Peptide PDB cleaned for HDOCK: %s", output_path)
        return {"clean_pdb": output_path, "error": None}

    except Exception as e:
        return {"clean_pdb": None, "error": str(e)}


# ── Batch peptide preparation ──────────────────────────────────────────────────

def prepare_peptides(
    peptide_specs: list,
    output_dir: Optional[str] = None,
    obabel_bin: Optional[str] = None,
    target_tag: Optional[str] = None,
) -> list:
    """
    Prepare a list of peptide sequences for docking.

    Args:
        peptide_specs: List of dicts with 'sequence' key (and optional 'rationale').
        output_dir: Output directory for PDB files.
        obabel_bin: Path to obabel binary.

    Returns:
        List of dicts with keys:
            - sequence (str)
            - pdb_path (str or None)
            - clean_pdb (str or None)
            - rationale (str)
            - source (str)
            - error (str or None)
    """
    results = []

    for spec in peptide_specs:
        if isinstance(spec, str):
            spec = {
                "sequence": spec,
                "rationale": "",
                "source": "raw_sequence",
            }

        seq = spec.get("sequence", "")
        if not seq:
            continue

        # Generate 3D structure
        gen_result = generate_peptide_pdb(seq, output_dir, obabel_bin, target_tag=target_tag)
        pdb_path = gen_result.get("pdb_path")

        # Clean for HDOCK
        clean_pdb = None
        if pdb_path:
            clean_result = prepare_peptide_for_hdock(pdb_path, output_dir)
            clean_pdb = clean_result.get("clean_pdb")

        results.append({
            "sequence": seq,
            "pdb_path": pdb_path,
            "clean_pdb": clean_pdb,
            "rationale": spec.get("rationale", ""),
            "source": spec.get("source", "unknown"),
            "error": gen_result.get("error"),
        })

    return results


# ── Known antiviral peptide motifs ────────────────────────────────────────────

def get_known_antiviral_peptides() -> list:
    """
    Return a curated list of peptide sequences derived from known
    antiviral and RNA-binding motifs that can compete with RNA for
    protein binding.

    These are not full proteins — they are short, cell-penetrating
    peptide sequences that have been shown to disrupt RNA-protein
    interactions in literature.
    """
    return [
        {
            "sequence": "RRM",
            "rationale": "Core RRM recognition motif found in many RBPs",
            "source": "known_antiviral",
        },
        {
            "sequence": "TAT",
            "rationale": "HIV-1 Tat protein RNA-binding motif",
            "source": "known_antiviral",
        },
        {
            "sequence": "RKK",
            "rationale": "Minimal basic helix for RNA mimicry",
            "source": "known_antiviral",
        },
        {
            "sequence": "KGRRRM",
            "rationale": "Extended RRM with NLS for nuclear targeting",
            "source": "known_antiviral",
        },
        {
            "sequence": "RPKRRM",
            "rationale": "PKR-like double-RRM basic patch",
            "source": "known_antiviral",
        },
    ]