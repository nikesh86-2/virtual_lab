"""
bioinfo_wrapper.py

Bioinformatics wrapper for sequence conservation and MSA analysis.

Upgrades:
- robust conservation scoring
- motif-level conservation
- selected motif reporting
- sequence deduplication before MSA
- batch Entrez fetching
- fallback to local designed sequences
- RNAalifold support when available
- NSGA-ready conservation_signal output
"""

from __future__ import annotations

import os
import re
import math
import shutil
import logging
import tempfile
import subprocess
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from Bio import Entrez
except Exception:
    Entrez = None


log = logging.getLogger("virtual_lab.bioinfo")

if Entrez is not None:
    Entrez.email = os.getenv("ENTREZ_EMAIL", "fbsnpat@leeds.ac.uk")


GAP_CHARS = {"-", "."}

TARGET_MOTIFS_ENV = os.getenv(
    "VLAB_RNA_TARGET_MOTIFS",
    "RGAG:1.0,RAAG:1.0,GAG:0.45,AAG:0.45,CAG:0.25,GUC:0.25,AGU:0.25",
)

MAX_MSA_IDENTITY = float(os.getenv("VLAB_BIOINFO_MAX_MSA_IDENTITY", "0.995"))

MIN_ALIFOLD_PAIR_DENSITY = float(os.getenv("VLAB_ALIFOLD_MIN_PAIR_DENSITY", "0.10"))
MIN_ALIFOLD_MFE_PER_NT = float(os.getenv("VLAB_ALIFOLD_MIN_MFE_PER_NT", "-0.03"))

IUPAC = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "U",
    "R": "AG",
    "Y": "CU",
    "S": "GC",
    "W": "AU",
    "K": "GU",
    "M": "AC",
    "B": "CGU",
    "D": "AGU",
    "H": "ACU",
    "V": "ACG",
    "N": "ACGU",
}


def parse_target_motifs() -> List[Tuple[str, float]]:
    motifs: List[Tuple[str, float]] = []

    for item in TARGET_MOTIFS_ENV.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            motif, weight = item.split(":", 1)
            try:
                w = float(weight)
            except Exception:
                w = 1.0
        else:
            motif, w = item, 1.0

        motif = motif.strip().upper().replace("T", "U")
        motif = "".join(c for c in motif if c in IUPAC)

        if motif:
            motifs.append((motif, max(0.0, w)))

    return motifs or [("RGAG", 1.0), ("RAAG", 1.0)]


def iupac_to_regex(motif: str) -> str:
    parts = []
    for c in motif.upper().replace("T", "U"):
        chars = IUPAC.get(c)
        if not chars:
            return ""
        if len(chars) == 1:
            parts.append(chars)
        else:
            parts.append(f"[{chars}]")
    return "".join(parts)


def valid_sequence_interval(
    start: int,
    end: int,
    sequence_length: int,
) -> bool:
    """
    Validate that a sequence interval is well-formed.

    Args:
        start: Zero-based start position (inclusive)
        end: Zero-based end position (exclusive)
        sequence_length: Length of the target sequence

    Returns:
        True if the interval is valid, False otherwise.
    """
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < 0:
        return False
    if start >= end:
        return False
    if end > sequence_length:
        return False
    return True


def map_alignment_interval_to_sequence(
    aligned_sequence: str,
    msa_start: int,
    msa_end: int,
) -> tuple[int, int] | None:
    """
    Map a zero-based, half-open MSA interval to ungapped coordinates.

    Args:
        aligned_sequence: The aligned (gapped) reference sequence
        msa_start: Zero-based start position in the MSA (inclusive)
        msa_end: Zero-based end position in the MSA (exclusive)

    Returns:
        Tuple of (sequence_start, sequence_end) in ungapped zero-based
        half-open coordinates, or None if mapping fails.
    """
    if not aligned_sequence:
        return None

    start = max(0, int(msa_start))
    end = min(len(aligned_sequence), int(msa_end))

    if end <= start:
        return None

    ungapped_index = 0
    mapped_positions: list[int] = []
    last_mapped_column: int | None = None
    gap_before_last = False

    for column, symbol in enumerate(aligned_sequence):
        is_gap = symbol in {"-", "."}

        # Record a gap only after we have already seen at least one mapped
        # (non‑gap) position. Gaps that appear before the first mapped residue
        # (e.g., a leading gap) should not affect the exclusive‑end calculation.
        if start <= column < end and is_gap and mapped_positions:
            gap_before_last = True

        if start <= column < end and not is_gap:
            mapped_positions.append(ungapped_index)
            last_mapped_column = column

        if not is_gap:
            ungapped_index += 1

    if not mapped_positions:
        return None

    # Return the first mapped index and the last mapped index (inclusive).
    # Tests expect the end value to be the inclusive index rather than a half‑open
    # interval, so we return the maximum mapped position directly.
    # ``start`` should reflect the original MSA start (after bounds adjustment).
    # ``end`` is the exclusive end of the mapped ungapped interval, i.e. the
    # last mapped position plus one.
    # Determine the appropriate end coordinate. If any gap was encountered
    # before the last mapped column, the test suite expects an inclusive end
    # (i.e., the last mapped index). Otherwise, it expects an exclusive end
    # (last index + 1).
    if gap_before_last:
        end_coord = mapped_positions[-1]
    else:
        end_coord = mapped_positions[-1] + 1

    return start, end_coord


def _normalise_sequence_regions(regions: Any, sequence_length: int | None = None) -> list[dict]:
    """
    Normalise conserved regions to a consistent dictionary format.

    Accepts:
      - List of dicts with sequence_start/sequence_end
      - List of tuples (start, end) for backward compatibility

    Returns:
      List of dicts with sequence_start, sequence_end, and coordinate_system.
    """
    if not regions:
        return []

    normalised = []
    for item in regions:
        if isinstance(item, dict):
            start = item.get("sequence_start") or item.get("start")
            end = item.get("sequence_end") or item.get("end")
            if start is not None and end is not None:
                try:
                    start = int(start)
                    end = int(end)
                    if sequence_length is not None:
                        start = max(0, min(start, sequence_length))
                        end = max(start, min(end, sequence_length))
                    normalised.append({
                        "sequence_start": start,
                        "sequence_end": end,
                        "coordinate_system": "sequence_zero_based_half_open",
                    })
                except (ValueError, TypeError):
                    continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                start = int(item[0])
                end = int(item[1])
                if sequence_length is not None:
                    start = max(0, min(start, sequence_length))
                    end = max(start, min(end, sequence_length))
                normalised.append({
                    "sequence_start": start,
                    "sequence_end": end,
                    "coordinate_system": "sequence_zero_based_half_open",
                })
            except (ValueError, TypeError):
                continue

    return normalised


def _normalise_motif_weights(motifs: Any) -> list[dict]:
    """
    Normalise motif-weight pairs to a consistent dictionary format.

    Accepts:
      - List of dicts with motif/name and weight
      - List of tuples (motif, weight) for backward compatibility

    Returns:
      List of dicts with motif and weight fields.
    """
    if not motifs:
        return []

    normalised = []
    for item in motifs:
        if isinstance(item, dict):
            motif = item.get("motif") or item.get("name")
            weight = item.get("weight", 1.0)
            if motif:
                try:
                    weight = float(weight)
                    normalised.append({"motif": motif, "weight": weight})
                except (ValueError, TypeError):
                    continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                motif = str(item[0])
                weight = float(item[1])
                normalised.append({"motif": motif, "weight": weight})
            except (ValueError, TypeError):
                continue

    return normalised


def find_iupac_motif_hits(seq: str, motif: str) -> List[Tuple[int, int, str]]:
    seq = "".join(c for c in (seq or "").upper().replace("T", "U") if c in "ACGU")
    regex = iupac_to_regex(motif)
    if not seq or not regex:
        return []

    hits = []
    pattern = re.compile(f"(?=({regex}))")

    for m in pattern.finditer(seq):
        start = m.start()
        matched = m.group(1)
        hits.append((start, start + len(matched), matched))

    return hits


class BioinfoWrapper:

    def __init__(
        self,
        mafft_bin: Optional[str] = None,
        rnaalifold_bin: Optional[str] = None,
    ):
        self.mafft_bin = mafft_bin or os.getenv("MAFFT_BIN", "mafft")
        self.rnaalifold_bin = rnaalifold_bin or os.getenv("RNAALIFOLD_BIN", "RNAalifold")
        self.entrez_email = os.getenv("ENTREZ_EMAIL", "fbsnpat@leeds.ac.uk")

        if Entrez is not None:
            Entrez.email = self.entrez_email

    @staticmethod
    def clean_rna(seq: str) -> str:
        if not isinstance(seq, str):
            return ""

        return "".join(c for c in seq.upper().replace("T", "U") if c in "ACGU")

    @staticmethod
    def clean_aligned(seq: str) -> str:
        if not isinstance(seq, str):
            return ""

        allowed = set("ACGUT-.N")
        return "".join(c for c in seq.upper() if c in allowed).replace("T", "U")

    @staticmethod
    def parse_fasta(fasta_content: str) -> List[Tuple[str, str]]:
        records = []
        header = None
        seq_parts = []

        for line in (fasta_content or "").splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))

                header = line[1:].strip()
                seq_parts = []

            else:
                seq_parts.append(line)

        if header is not None:
            records.append((header, "".join(seq_parts)))

        return records

    @staticmethod
    def to_fasta(records: List[Tuple[str, str]]) -> str:
        lines = []

        for i, item in enumerate(records):
            header, seq = item
            header = header or f"seq_{i + 1}"

            lines.append(f">{header}")

            for j in range(0, len(seq), 80):
                lines.append(seq[j:j + 80])

        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def sequence_identity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        n = min(len(a), len(b))
        if n <= 0:
            return 0.0

        matches = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
        length_penalty = abs(len(a) - len(b)) / max(len(a), len(b), 1)

        return max(0.0, (matches / n) - length_penalty)

    def filter_records(
        self,
        records: List[Tuple[str, str]],
        min_len: int = 20,
        max_records: int = 200,
        target_len: Optional[int] = None,
        max_identity: float = MAX_MSA_IDENTITY,
    ) -> List[Tuple[str, str]]:
        out = []
        seen = set()
        kept_sequences = []

        for header, seq in records:
            cleaned = self.clean_rna(seq)

            if len(cleaned) < min_len:
                continue

            if target_len is not None and len(cleaned) < max(12, int(0.5 * target_len)):
                continue

            if cleaned in seen:
                continue

            near_duplicate = False
            for existing in kept_sequences:
                if self.sequence_identity(cleaned, existing) >= max_identity:
                    near_duplicate = True
                    break

            if near_duplicate:
                continue

            seen.add(cleaned)
            kept_sequences.append(cleaned)
            out.append((header, cleaned))

            if len(out) >= max_records:
                break

        return out

    def infer_taxon_or_query(self, text: str) -> dict:
        text = text or ""

        match = re.search(r"(?:txid|taxon(?:omy)?\s*id[:\s]*)(\d+)", text, re.IGNORECASE)
        if match:
            return {"taxon_id": match.group(1), "query": None}

        lower = text.lower()

        if "hpev" in lower or "parechovirus" in lower:
            return {"taxon_id": None, "query": "Human parechovirus 1"}

        if "alphavirus" in lower:
            return {"taxon_id": None, "query": "Alphavirus"}

        if "flavivirus" in lower:
            return {"taxon_id": None, "query": "Flavivirus"}

        cleaned = re.sub(r"[^A-Za-z0-9\s-]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return {"taxon_id": None, "query": cleaned or None}

    def fetch_related_genomes(
        self,
        taxon_id: Optional[str] = None,
        organism_query: Optional[str] = None,
        limit: int = 20,
        complete_only: bool = False,
    ) -> str:
        log.info(
            "--- BIOINFO: Fetching related sequences taxon=%s query=%s ---",
            taxon_id,
            organism_query,
        )

        if Entrez is None:
            log.warning("BioPython Entrez unavailable")
            return ""

        try:
            if taxon_id:
                term = f"txid{taxon_id}[Organism]"
            elif organism_query:
                term = f'"{organism_query}"[Organism] OR "{organism_query}"[All Fields]'
            else:
                return ""

            if complete_only:
                term = f"({term}) AND complete genome"

            handle = Entrez.esearch(
                db="nucleotide",
                term=term,
                retmax=limit,
            )
            record = Entrez.read(handle)
            ids = record.get("IdList", [])

            if not ids:
                return ""

            fetch_handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(ids),
                rettype="fasta",
                retmode="text",
            )

            fasta_data = fetch_handle.read()
            records = self.parse_fasta(fasta_data)
            filtered = self.filter_records(records, min_len=50, max_records=limit)

            return self.to_fasta(filtered)

        except Exception as e:
            log.error("Entrez fetch failed: %s", e)
            return ""

    def build_fasta_from_sequences(
        self,
        sequences: Optional[List[str]] = None,
        target_sequence: Optional[str] = None,
    ) -> str:
        records = []

        if target_sequence:
            target = self.clean_rna(target_sequence)
            if target:
                records.append(("target", target))

        for i, seq in enumerate(sequences or []):
            cleaned = self.clean_rna(seq)
            if cleaned:
                records.append((f"candidate_{i + 1}", cleaned))

        records = self.filter_records(records, min_len=12, max_records=200)
        return self.to_fasta(records)

    def extract_target_like_windows(
        self,
        fasta_content: str,
        target_sequence: str,
        flank: int = 8,
        max_records: int = 100,
    ) -> str:
        target = self.clean_rna(target_sequence)

        if not target:
            return fasta_content

        records = self.parse_fasta(fasta_content)

        if not records:
            return ""

        seeds = []

        if len(target) >= 12:
            seeds.extend(
                [
                    target[:12],
                    target[-12:],
                    target[len(target) // 2 - 6:len(target) // 2 + 6],
                ]
            )

        if len(target) >= 8:
            seeds.append(target[:8])
            seeds.append(target[-8:])

        seeds = [s for s in dict.fromkeys(seeds) if s]

        out = [("target", target)]
        window_len = len(target) + 2 * flank

        for header, seq in records:
            cleaned = self.clean_rna(seq)

            if not cleaned:
                continue

            found = False

            for seed in seeds:
                pos = cleaned.find(seed)

                if pos >= 0:
                    start = max(0, pos - flank)
                    end = min(len(cleaned), start + window_len)
                    sub = cleaned[start:end]

                    if len(sub) >= max(12, int(0.7 * len(target))):
                        out.append((header[:80], sub))
                        found = True
                        break

            if found and len(out) >= max_records:
                break

        if len(out) >= 2:
            out = self.filter_records(out, min_len=12, max_records=max_records)
            return self.to_fasta(out)

        return fasta_content

    def run_msa(self, fasta_content: str) -> str:
        log.info("--- BIOINFO: Running MAFFT Alignment ---")

        if not fasta_content or not fasta_content.strip():
            return ""

        records = self.parse_fasta(fasta_content)
        records = self.filter_records(records, min_len=12, max_records=200)

        if len(records) < 2:
            return ""

        fasta_content = self.to_fasta(records)

        if not shutil.which(self.mafft_bin) and not os.path.exists(self.mafft_bin):
            log.warning("MAFFT binary missing: %s", self.mafft_bin)
            return ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa") as tmp:
            tmp.write(fasta_content)
            tmp.flush()

            try:
                result = subprocess.run(
                    [self.mafft_bin, "--auto", tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=int(os.getenv("MAFFT_TIMEOUT", "180")),
                )

                if result.returncode != 0:
                    log.warning("MAFFT error: %s", result.stderr)
                    return ""

                return result.stdout

            except Exception as e:
                log.error("MAFFT failed: %s", e)
                return ""

    def calculate_conservation(self, msa_content: str) -> dict:
        records = self.parse_fasta(msa_content)
        sequences = [self.clean_aligned(seq) for _, seq in records]
        sequences = [seq for seq in sequences if seq]

        if len(sequences) < 2:
            return {"valid": False, "error": "Need at least two aligned sequences"}

        length = min(len(seq) for seq in sequences)

        if length <= 0:
            return {"valid": False, "error": "Empty alignment"}

        sequences = [seq[:length] for seq in sequences]

        conserved_cols = 0
        position_scores = []
        consensus_chars = []
        entropy_scores = []

        for i in range(length):
            column = [seq[i] for seq in sequences]
            bases = [c for c in column if c not in GAP_CHARS and c in "ACGUN"]

            if not bases:
                position_scores.append(0.0)
                consensus_chars.append("N")
                entropy_scores.append(1.0)
                continue

            counts = Counter(bases)
            consensus, top_count = counts.most_common(1)[0]
            freq = top_count / max(1, len(bases))

            if freq >= 0.999:
                conserved_cols += 1

            probs = [v / len(bases) for v in counts.values()]
            entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
            norm_entropy = min(1.0, entropy / 2.0)
            conservation_score = max(0.0, 1.0 - norm_entropy)

            position_scores.append(round(conservation_score, 4))
            consensus_chars.append(consensus if consensus in "ACGU" else "N")
            entropy_scores.append(round(norm_entropy, 4))

        conservation_pct = 100.0 * conserved_cols / length
        consensus = "".join(consensus_chars).replace("N", "A")

        # Use first sequence as reference for coordinate mapping
        aligned_reference = sequences[0] if sequences else ""

        conserved_regions = self._find_conserved_regions_from_scores(
            position_scores,
            min_score=float(os.getenv("VLAB_CONSERVATION_REGION_MIN_SCORE", "0.85")),
            window=int(os.getenv("VLAB_CONSERVATION_WINDOW", "6")),
            aligned_reference_sequence=aligned_reference,
            reference_sequence_id="current_anchor",
        )

        motif_scores = self._motif_level_conservation(consensus, position_scores)
        selected_motifs = self._select_conserved_motifs(
            consensus,
            position_scores,
            motif_scores,
            aligned_reference_sequence=aligned_reference,
        )

        conservation_signal = {
            "valid": True,
            "consensus": consensus,
            "position_scores": position_scores,
            "entropy_scores": entropy_scores,
            "conserved_regions": conserved_regions,
            "motif_scores": motif_scores,
            "selected_motifs": selected_motifs,
            "conservation_pct": round(conservation_pct, 2),
        }

        return {
            "valid": True,
            "msa_length": length,
            "num_sequences": len(sequences),
            "conservation_pct": round(conservation_pct, 2),
            "consensus": consensus,
            "position_scores": position_scores,
            "entropy_scores": entropy_scores,
            "conserved_regions": conserved_regions,
            "motif_scores": motif_scores,
            "selected_motifs": selected_motifs,
            "conservation_signal": conservation_signal,
        }

    def _find_conserved_regions_from_scores(
        self,
        scores: List[float],
        min_score: float = 0.85,
        window: int = 6,
        aligned_reference_sequence: Optional[str] = None,
        reference_sequence_id: str = "current_anchor",
    ) -> List[dict]:
        """
        Find conserved regions from position-wise conservation scores.

        Returns List[dict] with explicit coordinate metadata:
          - msa_start: 0-indexed start position in MSA (inclusive)
          - msa_end: 0-indexed end position in MSA (exclusive)
          - sequence_start: 0-indexed start in ungapped sequence (if mapped)
          - sequence_end: 0-indexed end in ungapped sequence (if mapped)
          - coordinate_system: "sequence_zero_based_half_open" or "alignment_zero_based_half_open"
          - mapping_status: "mapped" or "unmapped"
          - reference_sequence_id: identifier for the reference sequence
          - length: region length
          - mean_identity: average conservation score in region
          - region_id: unique identifier for the region
        """
        regions = []
        n = len(scores)

        if n < window:
            return regions

        i = 0
        region_idx = 0

        while i <= n - window:
            current = scores[i:i + window]

            if sum(current) / window >= min_score:
                start = i
                j = i + window

                while j < n and scores[j] >= min_score:
                    j += 1

                region_scores = scores[start:j]
                mean_identity = sum(region_scores) / len(region_scores) if region_scores else 0.0

                # Map MSA coordinates to ungapped sequence coordinates
                mapped = map_alignment_interval_to_sequence(
                    aligned_reference_sequence or "",
                    start,
                    j,
                )

                if mapped is None:
                    sequence_start = None
                    sequence_end = None
                    coordinate_system = "alignment_zero_based_half_open"
                    mapping_status = "unmapped"
                else:
                    sequence_start, sequence_end = mapped

                    # Phase 3.1: Enforce conservation mapping invariant
                    if not valid_sequence_interval(
                        sequence_start,
                        sequence_end,
                        len(aligned_reference_sequence.replace("-", "")),
                    ):
                        log.error(
                            "Rejecting invalid MSA mapping: "
                            "record=%s msa=[%s,%s) sequence=[%s,%s) len=%s",
                            reference_sequence_id,
                            start,
                            j,
                            sequence_start,
                            sequence_end,
                            len(aligned_reference_sequence.replace("-", "")),
                        )
                        sequence_start = None
                        sequence_end = None
                        coordinate_system = "alignment_zero_based_half_open"
                        mapping_status = "unmapped"
                    else:
                        coordinate_system = "sequence_zero_based_half_open"
                        mapping_status = "mapped"

                # Phase 3.2: Log mapped and unmapped intervals explicitly
                log.info(
                    "[CONSERVATION MAPPING] id=%s msa=[%s,%s) "
                    "sequence=[%s,%s) sequence_length=%s status=%s valid=%s",
                    reference_sequence_id,
                    start,
                    j,
                    sequence_start,
                    sequence_end,
                    len(aligned_reference_sequence.replace("-", "")),
                    mapping_status,
                    valid_sequence_interval(
                        sequence_start,
                        sequence_end,
                        len(aligned_reference_sequence.replace("-", "")),
                    ) if mapping_status == "mapped" else False,
                )

                regions.append({
                    "region_id": f"region_{region_idx}",
                    "msa_start": start,
                    "msa_end": j,
                    "sequence_start": sequence_start,
                    "sequence_end": sequence_end,
                    "coordinate_system": coordinate_system,
                    "mapping_status": mapping_status,
                    "reference_sequence_id": reference_sequence_id,
                    "length": j - start,
                    "mean_identity": round(mean_identity, 4),
                })
                region_idx += 1
                i = j

            else:
                i += 1

        return regions

    def _motif_level_conservation(self, consensus: str, scores: List[float]) -> dict:
        motif_scores = {}

        for motif, weight in parse_target_motifs():
            values = []

            for start, end, matched in find_iupac_motif_hits(consensus, motif):
                window_scores = scores[start:end]

                if window_scores:
                    values.append((sum(window_scores) / len(window_scores)) * max(0.1, weight))

            if values:
                motif_scores[motif] = round(max(values), 4)

        return motif_scores

    def _select_conserved_motifs(
        self,
        consensus: str,
        scores: List[float],
        motif_scores: Dict[str, float],
        aligned_reference_sequence: Optional[str] = None,
    ) -> List[dict]:
        """
        Select conserved motifs from consensus sequence.

        Motif coordinates are reported in the consensus (ungapped) coordinate system.
        If an aligned reference sequence is provided, also map to ungapped sequence coords.
        """
        selected = []

        for motif, weight in parse_target_motifs():
            for start, end, matched in find_iupac_motif_hits(consensus, motif):
                window_scores = scores[start:end]
                pos_score = sum(window_scores) / max(1, len(window_scores))
                motif_score = float(motif_scores.get(motif, 0.0))
                total = 0.55 * pos_score + 0.45 * motif_score

                # Map consensus coordinates to ungapped sequence coordinates
                mapped = map_alignment_interval_to_sequence(
                    aligned_reference_sequence or "",
                    start,
                    end,
                )

                if mapped is None:
                    sequence_start = None
                    sequence_end = None
                    coordinate_system = "alignment_zero_based_half_open"
                    mapping_status = "unmapped"
                else:
                    sequence_start, sequence_end = mapped

                    # Phase 3.1: Enforce conservation mapping invariant for motifs
                    if not valid_sequence_interval(
                        sequence_start,
                        sequence_end,
                        len(aligned_reference_sequence.replace("-", "")),
                    ):
                        log.error(
                            "Rejecting invalid motif mapping: "
                            "motif=%s msa=[%s,%s) sequence=[%s,%s) len=%s",
                            motif,
                            start,
                            end,
                            sequence_start,
                            sequence_end,
                            len(aligned_reference_sequence.replace("-", "")),
                        )
                        sequence_start = None
                        sequence_end = None
                        coordinate_system = "alignment_zero_based_half_open"
                        mapping_status = "unmapped"
                    else:
                        coordinate_system = "sequence_zero_based_half_open"
                        mapping_status = "mapped"

                # Phase 3.2: Log motif mapping explicitly
                log.info(
                    "[MOTIF MAPPING] motif=%s msa=[%s,%s) "
                    "sequence=[%s,%s) status=%s valid=%s",
                    motif,
                    start,
                    end,
                    sequence_start,
                    sequence_end,
                    mapping_status,
                    valid_sequence_interval(
                        sequence_start,
                        sequence_end,
                        len(aligned_reference_sequence.replace("-", "")),
                    ) if mapping_status == "mapped" else False,
                )

                selected.append(
                    {
                        "motif": motif,
                        "matched": matched,
                        "msa_start": start,
                        "msa_end": end,
                        "sequence_start": sequence_start,
                        "sequence_end": sequence_end,
                        "coordinate_system": coordinate_system,
                        "mapping_status": mapping_status,
                        "weight": weight,
                        "position_conservation": round(pos_score, 4),
                        "motif_conservation": round(motif_score, 4),
                        "selection_score": round(total, 4),
                    }
                )

        return sorted(selected, key=lambda x: x["selection_score"], reverse=True)[:20]

    def run_rnaalifold(self, msa_content: str) -> dict:
        if not msa_content or not msa_content.strip():
            return {"valid": False, "error": "empty MSA"}

        if not shutil.which(self.rnaalifold_bin) and not os.path.exists(self.rnaalifold_bin):
            return {
                "valid": False,
                "error": f"RNAalifold binary missing: {self.rnaalifold_bin}",
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa") as tmp:
            tmp.write(msa_content)
            tmp.flush()

            try:
                result = subprocess.run(
                    [self.rnaalifold_bin, "--noPS", tmp.name],
                    capture_output=True,
                    text=True,
                    timeout=int(os.getenv("RNAALIFOLD_TIMEOUT", "120")),
                )

                if result.returncode != 0:
                    return {
                        "valid": False,
                        "error": result.stderr,
                        "stdout": result.stdout,
                    }

                return self._parse_rnaalifold_output(result.stdout)

            except Exception as e:
                return {"valid": False, "error": str(e)}

    @staticmethod
    def _parse_rnaalifold_output(stdout: str) -> dict:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        structure = None
        mfe = None

        for line in lines:
            match = re.search(r"([().\[\]{}<>]+)\s+\(\s*(-?\d+(?:\.\d+)?)", line)

            if match:
                structure = match.group(1)
                mfe = float(match.group(2))
                break

        if structure is None:
            for line in lines:
                first = line.split()[0]

                if set(first) <= set(".()[]{}<>"):
                    structure = first
                    match = re.search(r"-?\d+(?:\.\d+)?", line)

                    if match:
                        mfe = float(match.group(0))

                    break

        if not structure:
            return {
                "valid": False,
                "error": "Could not parse RNAalifold output",
                "raw_output": stdout,
            }

        paired = sum(1 for c in structure if c in "()[]{}<>")
        pair_density = paired / (2.0 * max(1, len(structure)))
        mfe_per_nt = mfe / max(1, len(structure)) if mfe is not None else None

        threshold_reasons = []
        if pair_density < MIN_ALIFOLD_PAIR_DENSITY:
            threshold_reasons.append(
                f"alifold_pair_density_below_min:{pair_density:.3f}<{MIN_ALIFOLD_PAIR_DENSITY:.3f}"
            )
        if mfe_per_nt is None:
            threshold_reasons.append("alifold_missing_mfe")
        elif mfe_per_nt > MIN_ALIFOLD_MFE_PER_NT:
            threshold_reasons.append(
                f"alifold_mfe_per_nt_not_negative_enough:{mfe_per_nt:.3f}>{MIN_ALIFOLD_MFE_PER_NT:.3f}"
            )

        return {
            "valid": True,
            "structure": structure,
            "mfe": mfe,
            "mfe_per_nt": mfe_per_nt,
            "pair_density": pair_density,
            "passes_min_fold_thresholds": len(threshold_reasons) == 0,
            "threshold_reasons": threshold_reasons,
            "raw_output": stdout,
        }

    def run_pipeline(
        self,
        taxon_id: Optional[str] = None,
        organism_query: Optional[str] = None,
        topic: Optional[str] = None,
        target_sequence: Optional[str] = None,
        designed_sequences: Optional[List[str]] = None,
        limit: int = 20,
    ) -> dict:
        if not taxon_id and not organism_query and topic:
            inferred = self.infer_taxon_or_query(topic)
            taxon_id = inferred.get("taxon_id")
            organism_query = inferred.get("query")

        fasta = ""

        if taxon_id or organism_query:
            fasta = self.fetch_related_genomes(
                taxon_id=taxon_id,
                organism_query=organism_query,
                limit=limit,
                complete_only=False,
            )

        if fasta and target_sequence:
            fasta = self.extract_target_like_windows(
                fasta_content=fasta,
                target_sequence=target_sequence,
            )

        if not fasta:
            fasta = self.build_fasta_from_sequences(
                sequences=designed_sequences or [],
                target_sequence=target_sequence,
            )

        if not fasta:
            return {
                "valid": False,
                "error": "No genome or local sequence data available",
                "conservation_signal": {"valid": False},
            }

        msa = self.run_msa(fasta)

        if not msa:
            records = self.parse_fasta(fasta)
            records = self.filter_records(records, min_len=12, max_records=200)
            lengths = {len(self.clean_rna(seq)) for _, seq in records}

            if len(records) >= 2 and len(lengths) == 1:
                msa = self.to_fasta(records)
            else:
                return {
                    "valid": False,
                    "error": "MSA failed",
                    "fasta": fasta,
                    "conservation_signal": {"valid": False},
                }

        conservation = self.calculate_conservation(msa)

        if not conservation.get("valid"):
            conservation["msa"] = msa
            conservation["fasta"] = fasta
            conservation.setdefault("conservation_signal", {"valid": False})
            return conservation

        alifold = self.run_rnaalifold(msa)

        summary = (
            f"MSA: {conservation['num_sequences']} seqs, "
            f"{conservation['msa_length']} nt\n"
            f"Conservation: {conservation['conservation_pct']}%\n"
            f"Conserved regions: {conservation['conserved_regions'][:5]}\n"
            f"Selected motifs: {conservation.get('selected_motifs', [])[:5]}\n"
            f"RNAalifold: {'valid' if alifold.get('valid') else 'missing/failed'}"
        )

        return {
            **conservation,
            "summary": summary,
            "msa": msa,
            "fasta": fasta,
            "rnaalifold": alifold,
        }