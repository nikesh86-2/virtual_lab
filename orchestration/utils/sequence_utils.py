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