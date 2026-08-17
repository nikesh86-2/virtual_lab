from __future__ import annotations

import logging
import os
import random
from typing import Optional


log = logging.getLogger("virtual_lab")


def clean_rna(seq: str) -> str:
    """
    Normalise to uppercase RNA and remove invalid bases.
    """
    if not isinstance(seq, str):
        return ""

    return "".join(
        c for c in seq.strip().upper().replace("T", "U")
        if c in "ACGU"
    )


def valid_sequence_interval(
    start: int,
    end: int,
    sequence_length: int,
) -> bool:
    """
    Validate that a sequence interval is well-formed: 0 <= start < end <= sequence_length.
    """
    try:
        start = int(start)
        end = int(end)
        sequence_length = int(sequence_length)
    except (TypeError, ValueError):
        return False

    return 0 <= start < end <= sequence_length


# ---------------------------------------------------------------------------
# MSA-to-sequence coordinate mapping (Priority 2 fix)
# ---------------------------------------------------------------------------

def alignment_boundary_to_sequence_offset(
    aligned_reference: str,
    boundary: int,
) -> int | None:
    """
    Convert an alignment column boundary to a sequence offset.

    For half-open interval [msa_start, msa_end), counts non-gap residues
    before each boundary position in the aligned reference.

    Args:
        aligned_reference: The aligned reference sequence (may contain gaps)
        boundary: The alignment column position (0-based)

    Returns:
        The sequence offset (number of non-gap residues before boundary),
        or None if boundary is out of range.
    """
    if boundary < 0 or boundary > len(aligned_reference):
        return None

    return sum(
        1
        for character in aligned_reference[:boundary]
        if character not in {"-", "."}
    )


def map_alignment_interval_to_sequence(
    aligned_reference: str,
    msa_start: int,
    msa_end: int,
) -> tuple[int, int] | None:
    """
    Map a half-open alignment interval to a sequence interval.

    Correctly handles gaps by using boundary-based counting, not
    position mapping.

    Args:
        aligned_reference: Aligned reference sequence (may contain gaps)
        msa_start: Alignment start column (0-based, inclusive)
        msa_end: Alignment end column (0-based, exclusive)

    Returns:
        Tuple of (sequence_start, sequence_end) as half-open interval,
        or None if interval is invalid.

    Examples:
        >>> map_alignment_interval_to_sequence("AACUGAGUCC", 4, 7)
        (4, 7)  # No gaps, so alignment and sequence intervals match

        >>> map_alignment_interval_to_sequence("AA-CUGA-GUCC", 3, 7)
        (2, 6)  # Gap before boundary 3, so sequence start is lower

    """
    if not isinstance(msa_start, int) or not isinstance(msa_end, int):
        return None

    if not 0 <= msa_start < msa_end <= len(aligned_reference):
        return None

    sequence_start = alignment_boundary_to_sequence_offset(
        aligned_reference,
        msa_start,
    )
    sequence_end = alignment_boundary_to_sequence_offset(
        aligned_reference,
        msa_end,
    )

    if sequence_start is None or sequence_end is None:
        return None

    if sequence_start >= sequence_end:
        return None

    return sequence_start, sequence_end


def validate_motif_mapping(
    matched_motif: str,
    sequence_start: int | None,
    sequence_end: int | None,
    reference_sequence: str,
) -> tuple[bool, str | None]:
    """
    Validate that a mapped motif interval is correct.

    Checks:
    - Both boundaries are defined
    - Interval is within reference bounds
    - Mapped length matches motif length
    - Mapped sequence equals matched sequence exactly

    Args:
        matched_motif: The concrete matched motif sequence (not a pattern)
        sequence_start: Mapped sequence interval start (0-based)
        sequence_end: Mapped sequence interval end (0-based, exclusive)
        reference_sequence: The ungapped reference sequence

    Returns:
        Tuple of (is_valid, error_reason)
        If is_valid, error_reason is None.

    Examples:
        >>> validate_motif_mapping("GAG", 5, 8, "AAAAAGAGCCCC")
        (True, None)

        >>> validate_motif_mapping("GAG", 5, 7, "AAAAAGAGCCCC")
        (False, "mapped_length_mismatch:expected=3,observed=2")
    """
    if sequence_start is None or sequence_end is None:
        return False, "missing_sequence_interval"

    if not 0 <= sequence_start < sequence_end <= len(reference_sequence):
        return False, "interval_out_of_bounds"

    expected_length = len(matched_motif)
    observed_length = sequence_end - sequence_start

    if observed_length != expected_length:
        return False, (
            "mapped_length_mismatch:"
            f"expected={expected_length},observed={observed_length}"
        )

    mapped_sequence = reference_sequence[sequence_start:sequence_end]

    if mapped_sequence != matched_motif:
        return False, (
            "mapped_sequence_mismatch:"
            f"expected={matched_motif},observed={mapped_sequence}"
        )

    return True, None



def hamming_distance(a: str, b: str) -> int:
    """
    Hamming distance with length mismatch treated as max length.
    """
    if len(a) != len(b):
        return max(len(a), len(b))

    return sum(x != y for x, y in zip(a, b))


def dedupe_rna_sequences(
    sequences: list[str],
    max_hamming: Optional[int] = None,
) -> list[str]:
    """
    Exact and optional near-duplicate RNA sequence deduplication.

    Env controls:
      VLAB_RNA_DEDUP_NEAR_DUPLICATES=1
      VLAB_RNA_NEAR_DUP_MAX_HAMMING=2
    """
    if max_hamming is None:
        try:
            max_hamming = int(os.getenv("VLAB_RNA_NEAR_DUP_MAX_HAMMING", "2"))
        except Exception:
            max_hamming = 2

    use_near = os.getenv("VLAB_RNA_DEDUP_NEAR_DUPLICATES", "1").strip() == "1"

    out: list[str] = []
    seen = set()

    for seq in sequences or []:
        s = clean_rna(seq)

        if not s:
            continue

        if s in seen:
            continue

        if use_near:
            too_close = False

            for existing in out:
                if (
                    len(existing) == len(s)
                    and hamming_distance(existing, s) <= max_hamming
                ):
                    too_close = True
                    break

            if too_close:
                continue

        seen.add(s)
        out.append(s)

    return out


def validate_sequence(seq: str) -> tuple[bool, str]:
    """
    Validate an RNA candidate for basic sequence sanity.
    """
    seq = clean_rna(seq)

    if not seq or len(seq) < 12:
        return False, "Sequence too short"

    max_repeat = 1
    current_repeat = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1

    try:
        max_homopolymer = int(os.getenv("VLAB_RNA_MAX_HOMOPOLYMER", "4"))
    except Exception:
        max_homopolymer = 4

    if max_repeat > max_homopolymer:
        return False, f"Too many repeated nucleotides ({max_repeat})"

    if any(c not in "ACGU" for c in seq):
        return False, "Sequence contains invalid characters"

    if len(set(seq)) < 3:
        return False, "Sequence uses fewer than three nucleotide types"

    gc = (seq.count("G") + seq.count("C")) / len(seq)

    if gc < 0.30 or gc > 0.80:
        return False, f"GC content out of range: {gc:.0%} (require 30–80%)"

    return True, "OK"


def mutate_sequence(
    seq: str,
    num_mutations: int = 2,
    locked_positions: Optional[set[int]] = None,
) -> str:
    """
    Mutate an RNA sequence while preserving locked positions.
    """
    seq_list = list(clean_rna(seq))
    locked_positions = locked_positions or set()

    mutable_positions = [
        i for i in range(len(seq_list))
        if i not in locked_positions
    ]

    if not mutable_positions:
        return "".join(seq_list)

    for _ in range(num_mutations):
        i = random.choice(mutable_positions)
        choices = [n for n in "ACGU" if n != seq_list[i]]

        if choices:
            seq_list[i] = random.choice(choices)

    return "".join(seq_list)