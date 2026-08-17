from __future__ import annotations

import pytest

from VLAB2.core.rna_binding_pocket import (
    constrain_vina_box,
    minimum_box_for_points,
    points_inside_box,
)


def test_points_inside_box():
    center = (0.0, 0.0, 0.0)
    size = (10.0, 10.0, 10.0)

    # Points inside box
    inside_points = [
        (0.0, 0.0, 0.0),
        (4.5, -4.5, 4.0),
        (-5.0, 5.0, -5.0),
    ]
    all_in, outside = points_inside_box(inside_points, center, size)
    assert all_in
    assert len(outside) == 0

    # Points outside box
    mixed_points = [
        (0.0, 0.0, 0.0),
        (6.0, 0.0, 0.0),  # outside on X axis
        (0.0, -5.5, 0.0), # outside on Y axis
    ]
    all_in, outside = points_inside_box(mixed_points, center, size)
    assert not all_in
    assert outside == [1, 2]


def test_minimum_box_for_points():
    points = [
        (-10.0, 5.0, 0.0),
        (10.0, -5.0, 20.0),
    ]
    center, size = minimum_box_for_points(points, padding=4.0)

    assert center == (0.0, 0.0, 10.0)
    assert size == (24.0, 14.0, 24.0)

    all_in, outside = points_inside_box(points, center, size)
    assert all_in


def test_constrain_vina_box():
    raw_size = (40.0, 30.0, 30.0)  # Volume = 36000 > 27000, X = 40 > 32
    constrained = constrain_vina_box(raw_size, max_volume=27000.0, max_dimension=32.0, min_dimension=18.0)

    assert all(dim <= 32.0 for dim in constrained)
    assert all(dim >= 18.0 for dim in constrained)
    vol = constrained[0] * constrained[1] * constrained[2]
    assert vol <= 27000.0 + 1e-3