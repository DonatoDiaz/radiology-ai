"""Shared shape measurements for CT regions.

Feret diameter, perimeter and thickness all appear in more than one module, and
each has a trap that is easy to get wrong on a binary mask:

* a **4-neighbour edge count** over-states a smooth boundary by 4/pi, so the
  Crofton factor is applied or a 12 mm cortex comes out at 9.5 mm;
* a **Feret diameter measured on the eroded core** is short by one voxel on
  every side, which is 2 mm on a 2 mm grid -- enough to miss a 30 mm threshold;
* a **bounding box** is not a diameter for anything that is not a box, which
  is why the maximum chord is computed on the convex hull.

Keeping these in one place means the traps are documented once.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial.qhull import QhullError

CROFTON_FACTOR = np.pi / 4.0


def perimeter_mm(mask2d: np.ndarray, spacing_yx: tuple[float, float]) -> float:
    """Perimeter of a 2D mask, by Crofton on a 4-neighbour edge count.

    A digital disk of radius r counts about 8r edges against a true 2*pi*r, so
    27% too much; the Crofton factor pi/4 removes that staircase bias. What is
    left is plain discretization, a few percent and shrinking as the object
    grows. Never divides by zero.
    """
    mask = np.asarray(mask2d).astype(bool)
    if not mask.any():
        return 0.0
    pad = np.pad(mask, 1)
    edges = int(np.count_nonzero(pad[1:-1, 1:-1] != pad[:-2, 1:-1]))
    edges += int(np.count_nonzero(pad[1:-1, 1:-1] != pad[1:-1, :-2]))
    return float(edges * CROFTON_FACTOR) * float(spacing_yx[0])


def feret_diameter_mm(mask2d: np.ndarray, spacing_yx: tuple[float, float]) -> float:
    """Longest chord through a 2D mask — the diameter read off a slice.

    Taken on the convex hull of the boundary, which has the same diameter as
    the mask but a few hundred points where the mask has tens of thousands.
    The boundary is used rather than the eroded core because erosion shortens
    every chord by a voxel, which is 2 mm on a 2 mm grid.
    """
    mask = np.asarray(mask2d).astype(bool)
    if not mask.any():
        return 0.0
    boundary = mask & ~_erode(mask)
    if not boundary.any():
        boundary = mask
    yy, xx = np.nonzero(boundary)
    pts = np.stack([yy * float(spacing_yx[0]), xx * float(spacing_yx[1])], axis=1)
    if len(pts) < 2:
        return 0.0
    try:
        hull = pts[ConvexHull(pts).vertices]
    except QhullError:  # degenerate input: collinear or fewer than 3 points
        hull = pts
    return float(pdist(hull).max())


def equivalent_diameter_mm(area_mm2: float) -> float:
    """Diameter of the circle with the same area."""
    return float(2.0 * np.sqrt(max(float(area_mm2), 0.0) / np.pi))


def hydraulic_thickness_mm(mask2d: np.ndarray, spacing_yx: tuple[float, float]) -> float:
    """Wall thickness as twice the hydraulic radius, 2A/P.

    A rim of thickness t has area about P*t/2, so 2A/P lands on t. It is an
    estimate and not a measurement, and it is only meaningful where the mask is
    a filled object rather than a wall-only segmentation.
    """
    mask = np.asarray(mask2d).astype(bool)
    if not mask.any():
        return 0.0
    area = float(mask.sum()) * float(spacing_yx[0]) * float(spacing_yx[1])
    per = perimeter_mm(mask, spacing_yx)
    if per <= 0.0:
        return 0.0
    return 2.0 * area / per


def extent_mm(mask3d: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    """Bounding box of a mask in mm, as (z, y, x)."""
    zz, yy, xx = np.nonzero(np.asarray(mask3d))
    if zz.size == 0:
        return (0.0, 0.0, 0.0)
    sp = [float(s) for s in spacing]
    return (
        (zz.max() - zz.min() + 1) * sp[0],
        (yy.max() - yy.min() + 1) * sp[1],
        (xx.max() - xx.min() + 1) * sp[2],
    )


def short_axis_mm(mask3d: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Short-axis diameter, measured in the axial plane.

    The standard reads a node's largest diameter in the plane, so the two
    in-plane extents are reported and the smaller one is the short axis; the
    craniocaudal extent is deliberately ignored.
    """
    _, dy, dx = extent_mm(mask3d, spacing)
    return float(min(dy, dx))


def volume_ml(mask3d: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Volume of a mask in mL, spacing in mm."""
    n = float(np.asarray(mask3d).sum())
    return n * float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0


def _erode(mask2d: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_erosion(mask2d, np.ones((3, 3), bool))
