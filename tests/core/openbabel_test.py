"""Basic test for OpenBabel resolution utilities.

The ``core.rna_prep`` module provides ``resolve_obabel`` which attempts to locate
the OpenBabel executable. In environments where OpenBabel is not installed the
function raises ``FileNotFoundError``. This test simply verifies that the
function exists and raises the expected exception when the executable cannot be
found.
"""

import pytest

from core.rna_prep import resolve_obabel


def test_resolve_obabel_raises_when_missing():
    """When OpenBabel is unavailable ``resolve_obabel`` should raise.

    The test does not require OpenBabel to be installed; it asserts that the
    function raises ``FileNotFoundError`` under normal CI conditions.
    """
    with pytest.raises(FileNotFoundError):
        resolve_obabel()
