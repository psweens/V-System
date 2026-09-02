"""
Voxelisation of a vessel network into a binary volume.

The network arrives as a (4, N) array whose rows are x, y, z and diameter,
with all-NaN columns separating branches. Consecutive non-NaN columns are
connected by tapered capsules: every voxel whose centre lies within the
linearly interpolated radius of the segment is set. This is the binary
analogue of convolving the centreline with a diameter-scaled spherical kernel
(Galarreta-Valverde 2012, section 3.4.4) and is rotation invariant, so a vessel
has the same calibre whatever its orientation.

Before rasterisation the network is mapped into the volume. The default
mapping is a single isotropic scale factor chosen so that the network, plus its
largest radius and an explicit voxel margin, fits the volume; the diameters are
scaled by the same factor, so bifurcation angles and the length-to-diameter
ratio survive into the image. Two alternatives exist: `voxel_size` maps grammar
units to voxels at a fixed physical resolution (the network is centred and
clipped), and `stretch` reproduces the historical behaviour of scaling every
axis independently to fill the volume with unscaled diameters. The isotropic
fit can also be told to ignore some axes (`clip_axes`), so that a network
fills a slab's two wide axes and is clipped along its depth, as a real imaging
slab would clip it.
"""

import numpy as np


FITS = ("isotropic", "voxel_size", "stretch")


def fit_to_volume(data, tVol, fit="isotropic", voxel_size=None, margin=1.0, clip_axes=()):
    """
    Maps network coordinates into voxel coordinates.

    Args:
        data (ndarray): (4, N) array of x, y, z, diameter; NaN columns separate branches.
        tVol (sequence): volume shape (nx, ny, nz).
        fit (str): "isotropic" (default), "voxel_size" or "stretch".
        voxel_size (float): grammar units per voxel, required when fit == "voxel_size".
        margin (float): empty border, in voxels, kept around the network for the
            fitted modes.
        clip_axes (sequence of int): axes (0, 1, 2) left out of the isotropic
            fit; the network is centred on them and clipped by the rasteriser.

    Returns:
        tuple: (points, radii) with points a (3, N) array in voxel coordinates
        (NaN preserved) and radii an (N,) array in voxels.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 4:
        raise ValueError("data must be a (4, N) array of x, y, z, diameter")
    shape = np.asarray(tVol, dtype=float)
    if shape.shape != (3,) or np.any(shape < 3):
        raise ValueError("tVol must be three dimensions of at least 3 voxels")
    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")

    xyz = data[:3]
    diam = data[3]
    if not np.any(~np.isnan(xyz[0])):
        raise ValueError("the network has no points")
    lo = np.nanmin(xyz, axis=1)
    hi = np.nanmax(xyz, axis=1)
    extent = hi - lo
    rmax = float(np.nanmax(diam)) / 2.0

    if fit == "isotropic":
        available = shape - 2.0 * margin
        keep = [a for a in range(3) if a not in tuple(clip_axes)]
        if not keep:
            raise ValueError("clip_axes cannot exclude every axis")
        s = float(np.min((available / (extent + 2.0 * rmax))[keep]))
        scale = np.full(3, s)
        radius_scale = s
    elif fit == "voxel_size":
        if voxel_size is None or voxel_size <= 0:
            raise ValueError("voxel_size must be a positive number of grammar units per voxel")
        s = 1.0 / float(voxel_size)
        scale = np.full(3, s)
        radius_scale = s
    else:  # stretch: legacy per-axis fill, diameters already in voxels
        scale = (shape - 2.0 * (margin + rmax)) / np.maximum(extent, 1e-12)
        radius_scale = 1.0
    if np.any(scale <= 0):
        raise ValueError("the volume is too small for the network's largest radius and margin")

    offset = (shape - scale * extent) / 2.0
    points = (xyz - lo[:, None]) * scale[:, None] + offset[:, None]
    radii = diam / 2.0 * radius_scale
    return points, radii


def rasterise_capsule(volume, p0, p1, r0, r1):
    """
    Sets every voxel whose centre lies within the tapered capsule from p0
    (radius r0) to p1 (radius r1). Coordinates are voxel indices; the capsule
    is clipped to the volume.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    shape = np.asarray(volume.shape)
    rmax = max(r0, r1)
    lo = np.maximum(np.floor(np.minimum(p0, p1) - rmax).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(p0, p1) + rmax).astype(int), shape - 1)
    if np.any(hi < lo):
        return
    grid = np.stack(np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1], axis=-1).astype(float)
    axis = p1 - p0
    length2 = float(axis @ axis)
    if length2 == 0.0:
        t = np.zeros(grid.shape[:-1])
    else:
        t = np.clip(((grid - p0) @ axis) / length2, 0.0, 1.0)
    closest = p0 + t[..., None] * axis
    dist2 = np.sum((grid - closest) ** 2, axis=-1)
    radius = r0 + t * (r1 - r0)
    inside = dist2 <= radius * radius
    volume[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1][inside] = 1


def rasterise_segments(points, radii, tVol):
    """
    Rasterises a polyline network. Consecutive non-NaN columns of `points`
    are joined by capsules; a NaN column breaks the chain.

    Returns:
        ndarray: uint8 volume of shape tVol with 1 inside vessels.
    """
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    volume = np.zeros(tuple(int(v) for v in tVol), dtype=np.uint8)
    previous = None
    for i in range(points.shape[1]):
        p = points[:, i]
        if np.isnan(p[0]):
            previous = None
            continue
        if previous is not None:
            q, rq = previous
            if not np.array_equal(p, q):
                rasterise_capsule(volume, q, p, rq, radii[i])
        previous = (p, radii[i])
    return volume


def process_network(data, tVol, fit="isotropic", voxel_size=None, margin=1.0, clip_axes=()):
    """
    Maps a network into the volume and rasterises it.

    Args:
        data (ndarray): (4, N) array of x, y, z, diameter with NaN separators.
        tVol (sequence): volume shape (nx, ny, nz).
        fit, voxel_size, margin, clip_axes: see fit_to_volume.

    Returns:
        ndarray: uint8 volume with 1 inside vessels and 0 elsewhere.
    """
    points, radii = fit_to_volume(data, tVol, fit=fit, voxel_size=voxel_size, margin=margin,
                                  clip_axes=clip_axes)
    return rasterise_segments(points, radii, tVol)
