"""
pocket_geometry.py
==================

Vina pocket-box geometry utilities for Priority 4: Docking box validity.

Implements:
- Minimum enclosing box calculation
- Box coverage evaluation
- Pocket splitting for oversized boxes
"""

from __future__ import annotations

from typing import Any

import numpy as np


def minimum_enclosing_box(
    points: Any,
    padding: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute the minimum enclosing box for a set of 3D points.

    Args:
        points: Array-like of shape (N, 3) containing 3D coordinates
        padding: Extra spacing on all sides (default 0.0)

    Returns:
        Tuple of (center, size) where:
        - center is (x, y, z) tuple
        - size is (dx, dy, dz) tuple for half-widths

    Raises:
        ValueError: If points array is invalid
    """
    xyz = np.asarray(points, dtype=float)

    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(
            f"Pocket points must be a non-empty N x 3 array; got shape {xyz.shape}"
        )

    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    center = (lower + upper) / 2.0
    size = upper - lower + 2.0 * float(padding)

    return tuple(center.tolist()), tuple(size.tolist())


def evaluate_box_coverage(
    points: Any,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Evaluate whether a box completely contains a set of points.

    Args:
        points: Array-like of shape (N, 3) containing 3D coordinates
        center: Box center (x, y, z)
        size: Box half-width in each axis (dx, dy, dz)
        tolerance: Numeric tolerance for inclusion test (default 1e-6 Å)

    Returns:
        Dictionary with keys:
        - contains_all: bool, True if all points are inside
        - coverage_fraction: float in [0, 1]
        - covered_point_count: int
        - outside_point_count: int
        - outside_point_indices: list[int]
        - total_point_count: int
    """
    xyz = np.asarray(points, dtype=float)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Points must be shape (N, 3); got {xyz.shape}")

    center_arr = np.asarray(center, dtype=float)
    half_size = np.asarray(size, dtype=float) / 2.0

    # Check if each point is within the box
    inside = np.all(
        np.abs(xyz - center_arr) <= half_size + tolerance,
        axis=1,
    )

    outside_indices = np.flatnonzero(~inside).tolist()
    covered = int(inside.sum())
    total = int(len(inside))

    return {
        "contains_all": covered == total,
        "coverage_fraction": covered / total if total else 0.0,
        "covered_point_count": covered,
        "outside_point_count": len(outside_indices),
        "outside_point_indices": outside_indices,
        "total_point_count": total,
    }


def split_points_on_longest_axis(points: Any) -> tuple[Any, Any]:
    """
    Recursively split points along their longest axis.

    Used for handling oversized pockets that exceed box constraints.

    Args:
        points: Array-like of shape (N, 3)

    Returns:
        Tuple of (first_half, second_half) as numpy arrays

    Raises:
        ValueError: If split produces empty partition
    """
    xyz = np.asarray(points, dtype=float)

    if xyz.shape[0] < 2:
        raise ValueError("Cannot split fewer than 2 points")

    # Find axis with largest extent
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    axis = int(np.argmax(extents))

    # Split at median along that axis
    midpoint = float(np.median(xyz[:, axis]))

    first = xyz[xyz[:, axis] <= midpoint]
    second = xyz[xyz[:, axis] > midpoint]

    if len(first) == 0 or len(second) == 0:
        raise ValueError(
            f"Unable to split points on axis {axis} at {midpoint}; "
            f"got {len(first)} and {len(second)} partitions"
        )

    return first, second


def recursively_partition_pocket(
    points: Any,
    max_axis_size: float = 32.0,
    max_volume: float | None = None,
) -> list[np.ndarray]:
    """
    Recursively partition oversized pockets into smaller subpockets.

    Each sub-pocket satisfies:
    - All axes <= max_axis_size
    - All points 100% contained
    - (Optional) Volume <= max_volume

    Args:
        points: Array-like of shape (N, 3)
        max_axis_size: Maximum allowed axis extent (default 32.0 Å)
        max_volume: Optional maximum box volume (default None = no limit)

    Returns:
        List of sub-pocket arrays, each satisfying size constraints
    """
    points = np.asarray(points, dtype=float)

    # Check if current pocket fits
    extents = points.max(axis=0) - points.min(axis=0)
    oversized_axis = np.max(extents)

    volume = np.prod(extents) if max_volume else 0.0
    oversized_volume = volume > max_volume if max_volume else False

    if oversized_axis <= max_axis_size and not oversized_volume:
        # This pocket is valid
        return [points]

    # Need to split
    try:
        first, second = split_points_on_longest_axis(points)
    except ValueError:
        # Cannot split further, return as-is (will be marked partial)
        return [points]

    # Recursively partition both halves
    result = []
    for partition in [first, second]:
        result.extend(
            recursively_partition_pocket(
                partition,
                max_axis_size=max_axis_size,
                max_volume=max_volume,
            )
        )

    return result
