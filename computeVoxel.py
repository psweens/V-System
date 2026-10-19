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

A capsule sets only the voxels whose centres it contains, so a vessel thinner
than a voxel contains no centre along much of its length and rasterises as a
dotted line, breaking a connected tree into fragments. Every segment is
therefore also drawn as the face-connected chain of voxels it passes through,
which renders a sub-voxel vessel one voxel wide instead of dotted and leaves
everything else untouched: a voxel on the chain lies within sqrt(3)/2 of the
centreline, so for a radius of 0.866 voxels or more the chain is already
inside the capsule. Face connectivity matters because it is the weakest
connectivity a downstream tool assumes -- a six-connected label or flood fill
splits a chain that touches only at edges or corners. Pass connect=False for
the bare capsule rasterisation.

Containing voxel centres aliases at small calibres. How many centres a vessel
about two voxels across contains swings with its centre line's position
against the grid, so such a vessel renders beaded, alternating between about
one voxel and 2 x 2 across, and its area depends on its sub-voxel offset.
Smoothing the capsule's coverage before thresholding it does not help: on a
straight vessel the result is still the voxel centres inside a smooth tube.
rasterise_segments(..., antialias=True) instead gives each vessel, in every
lattice plane across it, as many voxels as its cross-section covers, rounded
by error diffusion along the vessel, and keeps vessels thinner than 2.45
voxels free of 2 x 2 x 2 cubes, so they keep their calibre along their length
at every orientation and offset without constrictions (rule and guarantees in
_rasterise_antialiased). It is off by default, and the default draws what 3.9
drew.

Coordinates and diameters arrive in grammar units. The pipeline treats one
grammar unit as one micrometre, so `voxel_size` is a physical voxel size in
micrometres per voxel; see the Units section of the README.
"""

import math

import numpy as np


FITS = ("isotropic", "voxel_size", "stretch")

# Axis names accepted wherever an axis may be named rather than indexed.
AXES = {"x": 0, "y": 1, "z": 2}


def normalise_axes(axes):
    """
    Normalises an axis selection to a sorted tuple of distinct indices.

    Accepts indices (0, 1, 2) and names ("x", "y", "z"), in any mixture, and
    so also a bare string: "z" and "xy" are iterated character by character.
    Booleans are rejected rather than read as the indices 0 and 1, so that a
    boolean mask -- the natural numpy way to write "clip z" -- cannot be
    misread as "clip x and y".

    Raises:
        ValueError: on an axis that is neither a name nor an index in range,
            and on a selection that is not iterable.
    """
    if isinstance(axes, (int, float)) or axes is None:
        raise ValueError(
            f"clip_axes must be a sequence of axes, got {axes!r}; write (2,) or \"z\"")
    out = set()
    for item in axes:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(
                f"unknown axis {item!r}; use x, y, z or the indices 0, 1, 2, "
                "not a boolean mask")
        if isinstance(item, str):
            name = item.strip().lower()
            if name not in AXES:
                raise ValueError(
                    f"unknown axis {item!r}; use x, y, z or the indices 0, 1, 2")
            out.add(AXES[name])
            continue
        try:
            index = int(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"unknown axis {item!r}; use x, y, z or the indices 0, 1, 2") from None
        if index != item or index not in (0, 1, 2):
            raise ValueError(
                f"unknown axis {item!r}; use x, y, z or the indices 0, 1, 2")
        out.add(index)
    return tuple(sorted(out))


def fit_to_volume(data, tVol, fit="isotropic", voxel_size=None, margin=1.0, clip_axes=()):
    """
    Maps network coordinates into voxel coordinates.

    Args:
        data (ndarray): (4, N) array of x, y, z, diameter; NaN columns separate branches.
        tVol (sequence): volume shape (nx, ny, nz).
        fit (str): "isotropic" (default), "voxel_size" or "stretch".
        voxel_size (float): grammar units (micrometres) per voxel, required
            when fit == "voxel_size".
        margin (float): empty border, in voxels, kept around the network for the
            fitted modes.
        clip_axes (sequence): axes left out of the isotropic fit, as names
            ("x", "y", "z") or indices (0, 1, 2); the network is centred on
            them and clipped by the rasteriser. Only the isotropic fit reads
            this: "voxel_size" and "stretch" ignore it. The default excludes no
            axis, so the whole network is fitted; the command line and
            main.generate_network default to clipping z instead, because their
            default volume is an imaging slab. An unknown axis raises rather
            than being ignored.

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
    clip = normalise_axes(clip_axes)

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
        keep = [a for a in range(3) if a not in clip]
        if not keep:
            raise ValueError("clip_axes cannot exclude every axis")
        s = float(np.min((available / (extent + 2.0 * rmax))[keep]))
        scale = np.full(3, s)
        radius_scale = s
    elif fit == "voxel_size":
        if voxel_size is None or voxel_size <= 0:
            raise ValueError(
                "voxel_size must be a positive number of grammar units (micrometres) per voxel")
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


def rasterise_line(volume, p0, p1):
    """
    Sets every voxel the segment from p0 to p1 passes through, so that a
    vessel too thin to contain a voxel centre still renders as an unbroken
    face-connected path. Coordinates are voxel indices, voxel i spanning
    [i - 0.5, i + 0.5); voxels outside the volume are dropped.

    This is the voxel traversal of Amanatides and Woo (1987): from the voxel
    containing p0 the walk steps across whichever face the segment crosses
    next, one axis at a time, until it reaches the voxel containing p1.
    Consecutive voxels therefore share a face, which is what a six-connected
    flood fill, label or morphological operation needs; a chain touching only
    at edges or corners falls apart under them. Every voxel visited is one the
    segment passes through, so its centre lies within sqrt(3)/2 of the
    centreline and the capsule already contains it once the radius reaches
    0.866 voxels.
    """
    shape = np.asarray(volume.shape)
    line = _line_voxels(p0, p1)
    inside = np.all((line >= 0) & (line < shape[:, None]), axis=0)
    if not np.any(inside):
        return
    line = line[:, inside]
    volume[line[0], line[1], line[2]] = 1


def _line_voxels(p0, p1):
    """The voxels rasterise_line sets, as a (3, n) integer array in walk order, unclipped."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    cur = np.floor(p0 + 0.5).astype(int)
    end = np.floor(p1 + 0.5).astype(int)
    step = np.sign(direction).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        # parameter at which the segment next crosses a face on each axis,
        # and the parameter distance between successive crossings
        t_next = np.where(direction != 0.0, (cur + 0.5 * step - p0) / direction, np.inf)
        t_step = np.where(direction != 0.0, np.abs(1.0 / direction), np.inf)
    visited = [cur.copy()]
    # exactly one face crossing per unit of L1 distance between the end voxels;
    # confining the choice to axes still short of the end makes the walk land
    # on the end voxel whatever rounding does to a crossing at the boundary
    for _ in range(int(np.abs(end - cur).sum())):
        axis = int(np.argmin(np.where(cur != end, t_next, np.inf)))
        cur[axis] += step[axis]
        t_next[axis] += t_step[axis]
        visited.append(cur.copy())
    return np.array(visited).T


def rasterise_segments(points, radii, tVol, connect=True, antialias=False):
    """
    Rasterises a polyline network. Consecutive non-NaN columns of `points`
    are joined by capsules; a NaN column breaks the chain.

    Args:
        points (ndarray): (3, N) array of voxel coordinates, NaN separated.
        radii (ndarray): (N,) radii in voxels.
        tVol (sequence): volume shape (nx, ny, nz).
        connect (bool): also draw each segment as the face-connected chain of
            voxels it passes through, so that a vessel thinner than a voxel
            renders one voxel wide rather than as a dotted line and the
            rendered tree is as connected as the geometry, under six- as well
            as 26-connectivity. False gives the bare capsule rasterisation,
            which drops sub-voxel vessels in and out along their length.
        antialias (bool): give each vessel, in every lattice plane across it,
            as many voxels as its cross-section covers, instead of every
            voxel whose centre the capsule contains (rule in
            _rasterise_antialiased). The number of centres a thin capsule
            contains swings with the centre line's position against the grid,
            so by default a vessel about two voxels across renders beaded,
            alternating between about one voxel and 2 x 2 across; anti-aliased,
            it keeps its calibre along its length at every orientation and
            sub-voxel offset, without constrictions. The chain is still
            unioned in when `connect` is true. False, the default, draws
            exactly what 3.9 draws.

    Returns:
        ndarray: uint8 volume of shape tVol with 1 inside vessels.
    """
    if not isinstance(antialias, (bool, np.bool_)):
        raise ValueError(f"antialias must be True or False, got {antialias!r}")
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if antialias:
        return _rasterise_antialiased(points, radii, tuple(int(v) for v in tVol), bool(connect))
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
                if connect:
                    rasterise_line(volume, q, p)
        previous = (p, radii[i])
    return volume


# The anti-aliased mode (rasterise_segments(..., antialias=True)). These values
# are part of the rule stated in _rasterise_antialiased: changing one changes
# the volumes the mode draws and what its tests establish.
_AA_CARRY = 0.5                 # share of a plane's rounding remainder carried into the next plane
_AA_HYSTERESIS = 0.1            # a run keeps its axis k while |u_k| >= max|u| - this
_AA_CUBE_FREE = 2.45            # planes of a smaller mean diameter complete no 2 x 2 x 2 cube
_AA_SUPPORT = 0.5               # a voxel is drawn only within r_eff + this of the centre line
_AA_SEPARATION = math.sqrt(3.0)  # capsules farther apart than this never touch, as by default
_AA_CULL = 8.0                  # segments this far beyond their reach outside the volume are skipped
_AA_CHUNK = 1 << 19             # candidate voxels, or voxel-segment pairs, evaluated at a time
_AA_CELL = 8                    # cell side of the segment grid behind the clearance test
_AA_SLAB = 32                   # x planes examined at a time by _fill_enclosed


def _rasterise_antialiased(points, radii, shape, connect):
    """
    Anti-aliased rasterisation of a polyline network, as
    rasterise_segments(points, radii, shape, connect, antialias=True).

    The default rule sets a voxel when its centre lies within the tapered
    capsule. This rule draws the same capsules, polyline by polyline, but
    decides in each lattice plane across a vessel how many voxels the plane
    gets, from the capsule's cross-section, and then which:

    1. Segments are read as by default: one between each two consecutive
       distinct points of a run of non-NaN columns, from the last repeat of
       the first point to the first repeat of the second, with their radii.
       A segment whose bounding box, grown by its larger radius plus
       1 + _AA_CULL voxels, misses the volume is skipped, splitting its
       polyline there.
    2. Runs. Each segment of a polyline has a dominant axis k, kept from the
       segment before, with the sign of u_k, while |u_k| >= max|u| - 0.1 and
       the sign holds, and otherwise argmax |u|. A run is a maximal stretch
       of segments with the same axis and sign.
    3. Interior and band. A voxel centre x within r + 0.5 of a segment has,
       against it, the distance |x - c| to its closest point c at parameter
       t in [0, 1], the radius r there (linear along the segment), the score
       s = |x - c| - r and the ramp clip(0.5 + r_eff - |x - c|, 0, 1), with
       r_eff = sqrt(max(r^2 - 1/12, 0)): summed over the lattice, the ramp
       gives the capsule's volume. A voxel whose ramp is 1 against some
       segment of the polyline, half a voxel or more inside r_eff, is drawn.
       Any other whose ramp is positive against some segment, or that lies
       within sqrt(3)/2 of one, is a candidate, owned by the segment of
       lowest score among those (the earlier on a tie).
    4. Counts. A slice is the candidates owned by one run that lie in one
       lattice plane x_k = c, and slices are ordered along the polyline. A
       slice's target T is the sum of their ramps; with the interior it
       makes the capsules' volume: pi r^2 / |u_k| per plane of a straight
       tube, and the right share at ends, bends and tapers. The slice gets
       N = floor(T + e + 0.5) of its candidates, where e = 0.5 (T' + e' - N')
       for the slice before and 0 at the polyline's start. Carrying half the
       rounding remainder rather than all of it gives a vessel whose planes
       each ask for n + 0.2 voxels n in every plane rather than n + 1 in
       every fifth.
    5. Eligibility. With `connect`, the voxels of the polyline's
       face-connected chains (rasterise_line) are drawn and count first in
       their slices. A candidate owned by an end (t = 0 on the first
       segment, 1 on the last) is eligible only inside the capsule, s <= 0;
       any other only within r_eff + 0.5 of the centre line. One outside the
       capsule, s > 0, is eligible only if every capsule whose surface lies
       more than sqrt(3) from its owner's (measured at the two axes' closest
       points, which never under-estimates the gap) is more than sqrt(3) +
       max(0.5, sqrt(3)/2 - r_min) away from it, r_min the capsule's smaller
       radius.
    6. Selection. A slice takes its eligible candidates by score, then
       position. In a slice whose candidates' mean diameter is below 2.45
       they are taken one at a time, skipping a candidate that would
       complete a 2 x 2 x 2 cube with the polyline's voxels already drawn
       (its interior, its chains, its larger slices and its earlier ones)
       and, with `connect`, one with no face neighbour among them; a voxel
       inside the volume counts only neighbours inside it. A larger slice
       takes the first N.
    7. The polylines are unioned and clipped to the volume, and every unset
       voxel whose six face neighbours are set is then set.

    Every voxel drawn is a chain voxel, a voxel less than half a voxel
    outside its capsule, or a voxel enclosed by its six neighbours, which
    joins nothing that was apart under 26-connectivity. A voxel outside the
    capsules is drawn only where step 5 allows it, so two capsules more than
    sqrt(3) voxels apart surface to surface render touching only where the
    default renders them touching (through the chain of a vessel thinner
    than sqrt(3) voxels). Below 2.45 voxels across, a segment's interior
    holds no 2 x 2 x 2 cube (its voxels lie within 0.69 of the centre line,
    and a cube's centres cannot all lie within 0.707 of a line) and the
    selection completes none: a thin vessel holds a cube only where its
    chain and interior alone form one, which those of a smooth vessel do
    not, or where it meets another polyline, so no 2 x 2 x 2 opening pinches
    it into necks. The result depends on the order of points along a
    polyline, not on the order of the polylines, and involves no random
    numbers.
    """
    volume = np.zeros(shape, dtype=np.uint8)
    a, b, ra, rb, starts = _aa_segments(points, radii, shape)
    if len(a):
        clearance = _Clearance(a, b, ra, rb)
        bounds = np.append(starts, len(a)).tolist()
        for first, last in zip(bounds[:-1], bounds[1:]):
            _aa_polyline(volume, a[first:last], b[first:last], ra[first:last], rb[first:last],
                         first, connect, clearance)
    _fill_enclosed(volume)
    return volume


def _aa_segments(points, radii, shape):
    """
    Step 1 of _rasterise_antialiased: the segments as arrays a, b (m, 3) and
    ra, rb (m,), and the index of each polyline's first segment.
    """
    n = points.shape[1]
    valid = ~np.isnan(points[0])
    repeat = np.zeros(n, dtype=bool)
    if n > 1:
        repeat[1:] = valid[1:] & valid[:-1] & np.all(points[:, 1:] == points[:, :-1], axis=0)
    first = np.flatnonzero(valid & ~repeat)      # the first column of each run of repeats
    end = first[first > 0]
    end = end[valid[end - 1]]                    # a segment ends where a new point follows a valid column,
    start = end - 1                              # which is the last repeat of the point it starts at
    a = np.ascontiguousarray(points[:, start].T)
    b = np.ascontiguousarray(points[:, end].T)
    ra, rb = radii[start], radii[end]
    run = np.cumsum(~valid)[end]
    reach = np.maximum(ra, rb) + 1.0 + _AA_CULL
    with np.errstate(invalid="ignore"):
        keep = (np.all(np.isfinite(a) & np.isfinite(b), axis=1) & np.isfinite(ra) & np.isfinite(rb)
                & np.all(np.maximum(a, b) + reach[:, None] >= -0.5, axis=1)
                & np.all(np.minimum(a, b) - reach[:, None] <= np.asarray(shape) - 0.5, axis=1))
    index = np.flatnonzero(keep)
    starts = np.ones(len(index), dtype=bool)
    starts[1:] = (np.diff(index) != 1) | (run[index][1:] != run[index][:-1])
    return a[index], b[index], ra[index], rb[index], np.flatnonzero(starts)


def _box_points(lo, hi):
    """
    The integer points of the boxes [lo_i, hi_i] (inclusive, (m, 3)), box by
    box, and the number in each box.
    """
    extent = hi - lo + 1
    sizes = np.prod(extent, axis=1)
    offset = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    rest, z = np.divmod(offset, np.repeat(extent[:, 2], sizes))
    x, y = np.divmod(rest, np.repeat(extent[:, 1], sizes))
    points = np.empty((len(offset), 3), dtype=np.int64)
    points[:, 0] = x + np.repeat(lo[:, 0], sizes)
    points[:, 1] = y + np.repeat(lo[:, 1], sizes)
    points[:, 2] = z + np.repeat(lo[:, 2], sizes)
    return points, sizes


def _keys(points, origin, strides):
    """Integer keys of points (n, 3) in a frame of the given origin and strides."""
    return ((points[:, 0] - origin[0]) * strides[0] + (points[:, 1] - origin[1]) * strides[1]
            + (points[:, 2] - origin[2]) * strides[2])


def _chunks(sizes, limit):
    """Consecutive ranges [i, j) of `sizes` summing to at most `limit` (or a single item)."""
    total = np.cumsum(sizes)
    i = 0
    while i < len(sizes):
        base = total[i - 1] if i else 0
        j = max(int(np.searchsorted(total, base + limit, side="right")), i + 1)
        yield i, j
        i = j


def _aa_runs(u):
    """Step 2 of _rasterise_antialiased: each segment's dominant axis, its sign and its run."""
    axis = np.empty(len(u), dtype=np.int64)
    sign = np.empty(len(u), dtype=np.int64)
    run = np.empty(len(u), dtype=np.int64)
    k, s, r = -1, 0, -1
    for j, v in enumerate(u.tolist()):
        size = [abs(c) for c in v]
        top = max(size)
        if k < 0 or size[k] < top - _AA_HYSTERESIS or (v[k] > 0) != (s > 0):
            k = size.index(top)
            s = 1 if v[k] > 0 else -1
            r += 1
        axis[j], sign[j], run[j] = k, s, r
    return axis, sign, run


def _aa_counts(target):
    """Step 4 of _rasterise_antialiased: the voxels each slice gets, in slice order."""
    counts = np.empty(len(target), dtype=np.int64)
    carry = 0.0
    for i, value in enumerate(target.tolist()):
        wanted = value + carry
        count = math.floor(wanted + 0.5)
        counts[i] = count
        carry = _AA_CARRY * (wanted - count)
    return counts


def _best(key, score, *columns):
    """Each key once, with the entry of lowest score (the first on a tie), keys ascending."""
    order = np.lexsort((score, key))
    order = order[np.concatenate([[True], key[order][1:] != key[order][:-1]])] if len(order) else order
    return (key[order], score[order]) + tuple(c[order] for c in columns)


# the seven other voxels of each of the eight 2 x 2 x 2 cubes holding a voxel
_CUBES = [[(cx + i, cy + j, cz + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
           if (cx + i, cy + j, cz + k) != (0, 0, 0)]
          for cx in (-1, 0) for cy in (-1, 0) for cz in (-1, 0)]


def _aa_polyline(volume, a, b, ra, rb, first, connect, clearance):
    """Steps 2-6 of _rasterise_antialiased for one polyline; `first` is its first segment's index."""
    shape = volume.shape
    m = len(a)
    d = b - a
    length2 = np.einsum("ij,ij->i", d, d)
    axis, sign, run = _aa_runs(d / np.sqrt(length2)[:, None])

    # the voxel centres within reach of each segment (r + 0.5, or sqrt(3)/2 for
    # the chain), and a frame holding them and their neighbours, for integer keys
    reach = np.maximum(np.maximum(ra, rb) + _AA_SUPPORT, math.sqrt(3.0) / 2.0)[:, None]
    lo = np.ceil(np.minimum(a, b) - reach).astype(np.int64)
    hi = np.floor(np.maximum(a, b) + reach).astype(np.int64)
    origin = lo.min(axis=0) - 1
    dims = hi.max(axis=0) - origin + 2
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)

    # step 3: the interior is drawn at once; every band candidate once, with its owner
    interior = np.zeros(int(np.prod(dims)), dtype=bool)
    found = []
    for i, j in _chunks(np.prod(hi - lo + 1, axis=1), _AA_CHUNK):
        g, n = _box_points(lo[i:j], hi[i:j])
        seg = np.repeat(np.arange(i, j), n)
        step = np.repeat(d[i:j], n, axis=0)
        w = g - np.repeat(a[i:j], n, axis=0)
        t = np.clip(np.einsum("ij,ij->i", w, step) / np.repeat(length2[i:j], n), 0.0, 1.0)
        w -= t[:, None] * step
        dist = np.sqrt(np.einsum("ij,ij->i", w, w))
        r = np.repeat(ra[i:j], n) + t * np.repeat(rb[i:j] - ra[i:j], n)
        ramp = 0.5 + np.sqrt(np.maximum(r * r - 1.0 / 12.0, 0.0)) - dist
        inner = ramp >= 1.0
        if np.any(inner):
            interior[_keys(g[inner], origin, strides)] = True
            _draw(volume, g[inner])
        near = ~inner & ((ramp > 0.0) | (dist <= math.sqrt(3.0) / 2.0))
        found.append(_best(_keys(g[near], origin, strides), dist[near] - r[near], seg[near], t[near], dist[near], r[near]))
    key, score, seg, t, dist, r = _best(*(np.concatenate(c) for c in zip(*found)))
    band = ~interior[key]
    key, score, seg, t, dist, r = key[band], score[band], seg[band], t[band], dist[band], r[band]

    if connect:
        chain = np.concatenate([_line_voxels(a[i], b[i]) for i in range(m)], axis=1).T
        _draw(volume, chain)
        chain = np.unique(_keys(chain, origin, strides))
        at = np.minimum(np.searchsorted(chain, key), len(chain) - 1)
        is_chain = chain[at] == key
    else:
        chain = np.zeros(0, dtype=np.int64)
        is_chain = np.zeros(len(key), dtype=bool)
    if not len(key):
        return
    g = np.stack([key // strides[0], key // strides[1] % dims[1], key % dims[2]], axis=1) + origin

    # step 4: slices and their counts
    coord = g[np.arange(len(g)), axis[seg]] * sign[seg]
    coord -= coord.min()
    slices, sid = np.unique(run[seg] * (int(coord.max()) + 1) + coord, return_inverse=True)
    sid = sid.ravel()
    ns = len(slices)
    reff = np.sqrt(np.maximum(r * r - 1.0 / 12.0, 0.0))
    want = _aa_counts(np.bincount(sid, weights=np.clip(0.5 + reff - dist, 0.0, 1.0), minlength=ns))
    thin = 2.0 * np.bincount(sid, weights=r, minlength=ns) < _AA_CUBE_FREE * np.bincount(sid, minlength=ns)

    # step 5: eligibility
    end = ((seg == 0) & (t <= 0.0)) | ((seg == m - 1) & (t >= 1.0))
    eligible = np.where(end, score <= 0.0, dist < reff + _AA_SUPPORT) | is_chain
    outside = np.flatnonzero(eligible & ~is_chain & (score > 0.0))
    if len(outside):
        eligible[outside] = clearance.allows(g[outside], first + seg[outside])

    # step 6: the larger slices take their first voxels, the thin ones avoid cubes
    order = np.lexsort((key, score, ~is_chain, sid))
    order = order[eligible[order]]
    osid = sid[order]
    rank = np.arange(len(order)) - np.searchsorted(osid, np.arange(ns))[osid]
    thick = order[(rank < want[osid]) & ~thin[osid]]
    _draw(volume, g[thick])
    if np.any(thin):
        drawn = set(chain.tolist())
        drawn.update(key[thick].tolist())
        drawn.update(np.flatnonzero(interior).tolist())
        _draw(volume, g[_aa_thin(order[thin[osid]], key, g, is_chain, sid, want, drawn, strides, shape, connect)])


def _draw(volume, voxels):
    """Sets the voxels (n, 3) that lie inside the volume."""
    voxels = voxels[np.all((voxels >= 0) & (voxels < np.asarray(volume.shape)), axis=1)]
    volume[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1


def _aa_thin(order, key, g, is_chain, sid, want, drawn, strides, shape, connect):
    """
    Step 6 for the slices below _AA_CUBE_FREE, taking candidates `order` in
    rank order: the indices of the voxels they add. `drawn` holds the keys
    of the polyline's voxels already drawn and gains those added.
    """
    sx, sy, sz = (int(s) for s in strides)
    cubes = [[i * sx + j * sy + k * sz for i, j, k in cube] for cube in _CUBES]
    faces = [-sx, sx, -sy, sy, -sz, sz]
    cand = g[order]
    inside = np.all((cand >= 0) & (cand < np.asarray(shape)), axis=1)
    # bit f set: the neighbour across face f may count (always, for a voxel outside the volume)
    bits = np.zeros(len(order), dtype=np.int64)
    for f in range(6):
        nb = cand[:, f // 2] + (1 if f % 2 else -1)
        bits |= ((~inside | ((nb >= 0) & (nb < shape[f // 2]))).astype(np.int64)) << f
    added = []
    index = order.tolist()
    slice_of = sid[order].tolist()
    keys = key[order].tolist()
    chained = is_chain[order].tolist()
    masks = bits.tolist()
    counts = want.tolist()
    taken, current = 0, -1
    for i, x in enumerate(keys):
        if slice_of[i] != current:
            current, taken = slice_of[i], 0
        if taken >= counts[current]:
            continue
        if chained[i]:
            taken += 1
            continue
        if connect and not any(masks[i] >> f & 1 and x + faces[f] in drawn for f in range(6)):
            continue
        if any(all(x + o in drawn for o in cube) for cube in cubes):
            continue
        drawn.add(x)
        added.append(index[i])
        taken += 1
    return np.array(added, dtype=np.int64)


class _Clearance:
    """
    The segments of a network on a grid, for the clearance test of step 5 of
    _rasterise_antialiased.
    """

    def __init__(self, a, b, ra, rb):
        self.a, self.ra, self.rb = a, ra, rb
        self.d = b - a
        self.length2 = np.einsum("ij,ij->i", self.d, self.d)
        # how far outside its capsule a voxel drawn for the segment may lie (the
        # band, or the chain of a segment thinner than sqrt(3)/2), plus sqrt(3)
        self.clear = _AA_SEPARATION + np.maximum(_AA_SUPPORT, math.sqrt(3.0) / 2.0 - np.minimum(ra, rb))
        reach = (np.maximum(ra, rb) + self.clear)[:, None]
        lo = np.floor((np.minimum(a, b) - reach) / _AA_CELL).astype(np.int64)
        hi = np.floor((np.maximum(a, b) + reach) / _AA_CELL).astype(np.int64)
        self.origin = lo.min(axis=0)
        self.dims = hi.max(axis=0) - self.origin + 1
        self.strides = np.array([self.dims[1] * self.dims[2], self.dims[2], 1], dtype=np.int64)
        cells, n = _box_points(lo, hi)
        segments = np.repeat(np.arange(len(lo)), n)
        keys = _keys(cells, self.origin, self.strides)
        order = np.argsort(keys, kind="stable")
        self.keys, self.segments = keys[order], segments[order]

    def allows(self, x, owner):
        """
        For voxels x (n, 3) outside the capsules of segments `owner`: whether
        every capsule more than _AA_SEPARATION from the owner's lies farther
        than its clearance from the voxel.
        """
        allowed = np.ones(len(x), dtype=bool)
        cell = np.floor_divide(x, _AA_CELL)
        listed = np.flatnonzero(np.all((cell >= self.origin) & (cell < self.origin + self.dims), axis=1))
        key = _keys(cell[listed], self.origin, self.strides)
        left = np.searchsorted(self.keys, key, side="left")
        count = np.searchsorted(self.keys, key, side="right") - left
        for i, j in _chunks(count, _AA_CHUNK):
            n = count[i:j]
            voxel = np.repeat(listed[i:j], n)
            m = self.segments[np.repeat(left[i:j], n) + np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)]
            w = x[voxel] - self.a[m]
            t = np.clip(np.einsum("ij,ij->i", w, self.d[m]) / self.length2[m], 0.0, 1.0)
            w -= t[:, None] * self.d[m]
            near = np.sqrt(np.einsum("ij,ij->i", w, w)) - (self.ra[m] + t * (self.rb[m] - self.ra[m])) <= self.clear[m]
            voxel, m = voxel[near], m[near]
            allowed[voxel[self.gap(owner[voxel], m) > _AA_SEPARATION]] = False
        return allowed

    def gap(self, i, j):
        """
        The surface gap of capsules i and j at their axes' closest points,
        which is never less than the true gap.
        """
        d1, d2 = self.d[i], self.d[j]
        w = self.a[i] - self.a[j]
        aa, ee = self.length2[i], self.length2[j]
        bb = np.einsum("ij,ij->i", d1, d2)
        c = np.einsum("ij,ij->i", d1, w)
        f = np.einsum("ij,ij->i", d2, w)
        denom = aa * ee - bb * bb
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(denom > 1e-12 * aa * ee, np.clip((bb * f - c * ee) / denom, 0.0, 1.0), 0.0)
        t = (bb * s + f) / ee
        s = np.where(t < 0.0, np.clip(-c / aa, 0.0, 1.0), np.where(t > 1.0, np.clip((bb - c) / aa, 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
        w += s[:, None] * d1 - t[:, None] * d2
        return (np.sqrt(np.einsum("ij,ij->i", w, w)) - (self.ra[i] + s * (self.rb[i] - self.ra[i]))
                - (self.ra[j] + t * (self.rb[j] - self.ra[j])))


def _fill_enclosed(volume):
    """Sets every unset voxel whose six face neighbours are set (step 7 of _rasterise_antialiased)."""
    nx = volume.shape[0]
    if min(volume.shape) < 3:
        return
    for x0 in range(1, nx - 1, _AA_SLAB):
        x1 = min(x0 + _AA_SLAB, nx - 1)
        v = volume[x0 - 1:x1 + 1].astype(bool)
        enclosed = (~v[1:-1, 1:-1, 1:-1] & v[:-2, 1:-1, 1:-1] & v[2:, 1:-1, 1:-1]
                    & v[1:-1, :-2, 1:-1] & v[1:-1, 2:, 1:-1] & v[1:-1, 1:-1, :-2] & v[1:-1, 1:-1, 2:])
        volume[x0:x1, 1:-1, 1:-1][enclosed] = 1


def process_network(data, tVol, fit="isotropic", voxel_size=None, margin=1.0, clip_axes=(),
                    connect=True, antialias=False):
    """
    Maps a network into the volume and rasterises it.

    Args:
        data (ndarray): (4, N) array of x, y, z, diameter with NaN separators,
            as returned by main.generate_network or read back from a saved
            centreline with main.load_network.
        tVol (sequence): volume shape (nx, ny, nz).
        fit, voxel_size, margin, clip_axes: see fit_to_volume. Note that
            clip_axes defaults to no axis here, while the command line and
            main.generate_network default to clipping z.
        connect (bool): see rasterise_segments. Left true, a vessel too thin to
            contain a voxel centre still renders as an unbroken one-voxel path.
        antialias (bool): see rasterise_segments. Left false, the volume is
            the one 3.9 draws.

    Returns:
        ndarray: uint8 volume with 1 inside vessels and 0 elsewhere.

    A network clipped by the volume can still render as several pieces: a branch
    that leaves a slab and re-enters it is two vessels in the image, as it would
    be in a real acquisition. `clip_axes` and the volume shape control that;
    `connect` does not.
    """
    points, radii = fit_to_volume(data, tVol, fit=fit, voxel_size=voxel_size, margin=margin,
                                  clip_axes=clip_axes)
    return rasterise_segments(points, radii, tVol, connect=connect, antialias=antialias)
