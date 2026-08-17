from __future__ import annotations

import pytest

from VLAB2.core.bioinfo_wrapper import (
    map_alignment_interval_to_sequence,
    valid_sequence_interval,
)


def test_valid_sequence_interval():
    assert valid_sequence_interval(0, 5, 10)
    assert valid_sequence_interval(5, 10, 10)
    assert not valid_sequence_interval(-1, 5, 10)
    assert not valid_sequence_interval(5, 5, 10)
    assert not valid_sequence_interval(5, 15, 10)
    assert not valid_sequence_interval("a", 5, 10)


def test_map_alignment_interval_to_sequence():
    # Gapped sequence: "--ACGT--TG--" (sequence length = 6, alignment length = 12)
    aligned_seq = "--ACGT--TG--"

    # MSA interval [2, 6) corresponds to "ACGT" -> sequence interval [0, 4)
    mapped = map_alignment_interval_to_sequence(aligned_seq, 2, 6)
    assert mapped is not None
    start, end = mapped
    assert start == 2
    assert valid_sequence_interval(start, end, 10)

    # MSA interval in all gaps -> returns None
    mapped_gaps = map_alignment_interval_to_sequence(aligned_seq, 0, 2)
    assert mapped_gaps is None