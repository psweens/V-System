#!/usr/bin/env python3
"""
Geometric and topological descriptors of a centreline network.

A synthetic network is only useful as training data if its local geometry
matches the vessels it stands in for: calibre, curvature, how often a vessel
ends, how junctions are arranged, which way vessels run, how densely they
fill the tissue and whether any pass through each other. None of that can be
read off a generator's parameters; a persistence length or a collision margin
promises a behaviour, and this module measures whether the network delivers
it. The same numbers, computed by the same rules on the skeleton of a real
image, then say how far the synthetic bank is from the tissue it imitates, so
the definitions here are written to be reproduced elsewhere and `describe`'s
docstring is the reference for every quantity.

    python describe.py output/Lnet_i9_s3.npz
    python describe.py output/*.npz --margin 1 --compact
    python describe.py network.npz --volume 512 512 140

prints the descriptors as JSON, one object per archive, keyed by path when
several archives are given. An archive needs only `nodes`; a graph stored in
`edges` is used when present and rebuilt from the coordinates otherwise, and
the metadata record supplies the collision margin, the growth volume and the
event counters when it is there.

Every quantity is a plain number or None, never NaN, so the result survives a
JSON round trip. Coordinates and diameters are in grammar units, which the
pipeline takes to be micrometres (see the Units section of the README); the
quantities suffixed _mm assume that.

The clearance search is the one measurement whose cost is not linear in the
network: every pair of points closer than the sum of their radii plus a
margin must be found. The points are kept in spatial indexes from
`spatial.py`, one per octave of radius, so that a capillary point is compared
with the capillaries near it and with the few arterioles within an arteriole's
reach, never with every point within the largest radius in the network. The
search then widens geometrically from the margin until it has checked a pair,
so the minimum is exact whenever it can be found at bounded cost and is
otherwise reported as a lower bound.

A network grown along a preferred direction, or within a plane, records the
frame it was steered by in its metadata, and the descriptors after
"per_tree" measure the network against that frame and against the calibre
class of its edges: how well the vessels follow the axis, whether the trees
run along it in the sense the generator chose, how the capillaries are
spaced across it and how length and volume divide between capillaries and
larger vessels. The frame-relative quantities are None for an archive
without a frame, and the class-dependent ones None without a reference
diameter; the older keys read as they always did.

The descriptors after "transverse_spacing" read the network as a graph of
vessel segments: how many vessels meet at a junction and at what angles,
how many segments the shortest loop through a segment or a junction takes,
and how much a vessel's calibre varies along it. Loops are searched from a
bounded sample of segments and junctions, each search bounded in depth and
in the vertices it reaches, so that they cost a fraction of a second on a
network of a million points. On request, the distance from the tissue to
the nearest vessel wall is also measured on a grid over the volume, exactly
and by the voxeliser's own rule for what lies inside a vessel.
"""
import argparse
import copy
import heapq
import json
import math
import sys

import numpy as np

import graph
from collisions import KIN_REACH
from graph import DEFAULT_TOL
from spatial import make_index

# Metadata keys copied into "flags": the settings that shape a network's
# geometry, so a table of descriptors can be read against them.
FLAG_KEYS = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
             "anastomose", "anastomosis_fraction", "anastomosis_radius", "anastomose_mode",
             "family", "seed", "iterations", "d0", "d_min")

# Orientation histograms: the polar angle to the z axis folded to 0..90
# degrees in bins of ten, the azimuth of the xy projection over 0..180 degrees
# in bins of thirty.
POLAR_BIN_DEG = 10.0
POLAR_BINS = 9
AZIMUTH_BIN_DEG = 30.0
AZIMUTH_BINS = 6

# Query points handled per pass of the clearance search, which bounds the
# memory a pass needs to a few million pairs.
_CHUNK = 16384

# Stored points are indexed in classes an octave of radius wide, so a query
# reaches only as far as the largest radius in the class it is searching. The
# smallest vessels of a deep tree can be a few thousandths of the root's
# radius, and a class narrower than this many octaves below the largest
# radius saves nothing more.
_MAX_RADIUS_CLASSES = 12

# The clearance search stops widening once a stage has gathered more pairs
# (excused ones included) than its budget without checking one: the next
# stage would cost more again and the network evidently has no two vessels
# within that reach of each other, so a lower bound is the honest answer.
# The budget grows with the network, because most of what a stage gathers is
# each point's neighbours along its own polyline, a fixed few dozen per point
# whatever the reach, and it never falls below a floor that lets a small
# network be searched out to its diagonal.
_STAGE_PAIRS_PER_POINT = 64
_STAGE_PAIRS_FLOOR = 4_000_000

# Calibre classes: an edge thinner than DEFAULT_CLASS_BOUND reference
# diameters is capillary-class, a thicker one larger-class, and the calibre
# shares are reported below each of CALIBRE_MULTIPLES reference diameters.
DEFAULT_CLASS_BOUND = 2.0
CALIBRE_MULTIPLES = (1.5, 2.0, 3.0)

# Transverse spacing is measured on planes perpendicular to the frame axis
# spread over the central part of the capillaries' extent along it, so that
# the planes stay clear of the ends, where vessels are still arriving.
TRANSVERSE_PLANES = 9
_TRANSVERSE_FIRST, _TRANSVERSE_STEP = 0.1, 0.1

# The concentration of a Watson or an axial Fisher distribution is found by
# bisection. Beyond this concentration both are a single direction at any
# precision the mean cosines reach, and the mean is reported as outside the
# range rather than as a number that no longer means anything.
_CONCENTRATION_MAX = 200.0
_BISECTION_STEPS = 200

# Gauss-Legendre rule on [0, 1] for the Watson moment; 256 points resolve
# the integrand's peak at the largest concentration searched.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(256)
_WATSON_U = (_GAUSS_NODES + 1.0) / 2.0
_WATSON_WEIGHT = _GAUSS_WEIGHTS / 2.0

# Pairs measured per pass of the nearest-neighbour search on a plane, which
# bounds the memory a pass needs.
_NEIGHBOUR_PAIRS = 2_000_000

# Loops are searched from at most LOOP_SAMPLES segments and as many
# junctions, taken at a regular stride through them. A search follows paths
# of at most LOOP_DEPTH segments (in total over the two ends of a segment,
# along each branch from a junction) and reaches at most LOOP_VERTICES
# vertices, so that its cost is bounded whatever the size of the network.
LOOP_SAMPLES = 2000
LOOP_DEPTH = 16
LOOP_VERTICES = 4096

# A branch angle is taken between the directions from a junction to the
# points this many of its diameters along each vessel, or halfway along a
# vessel shorter than twice that: far enough out that the vessels have left
# the junction, near enough that they have not yet turned away from it.
BRANCH_ARC_DIAMETERS = 2.0

# The tissue distance is sampled on a grid of at most this many points; a
# finer spacing than that allows is raised and recorded.
EVD_MAX_POINTS = 64 ** 3

# A pass of the exact tissue-distance search handles at most _EVD_CHUNK grid
# points, and fewer when they would hold more than _EVD_PAIRS pairs of a
# point and a cell or piece of vessel at once, which bounds the memory a
# pass needs whatever the geometry (a single point may hold more, up to the
# number of pieces). The finest cells of its search tree are halved from the
# grid spacing until they hold at most _EVD_CELL_PIECES vessel pieces on
# average, but no further than _EVD_HALVINGS times or than
# 2 ** _EVD_LEVEL_BITS cells along an axis, so that cell keys fit in 64 bits.
_EVD_CHUNK = 4096
_EVD_PAIRS = 1 << 20
_EVD_CELL_PIECES = 8
_EVD_HALVINGS = 10
_EVD_LEVEL_BITS = 20

# Vessels are cut into pieces no longer than the grid spacing for that
# search, and into longer ones when there would otherwise be more than this
# many pieces, which bounds the memory the search needs.
_EVD_PIECES = 1 << 22

# Bounds in the tissue-distance search are compared with this much room,
# relative to the size of the coordinates, so that rounding can never prune
# the vessel that is nearest.
_EVD_SLACK = 1e-9


def _as_nodes(nodes):
    """Returns `nodes` as a (4, N) float array; empty input is (4, 0)."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size == 0:
        return np.empty((4, 0))
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError(f"nodes must be a (4, N) array of x, y, z and diameter, got {nodes.shape}")
    return nodes


def _as_edges(edges, n_columns):
    """Returns `edges` as a (2, E) int64 array of column indices into `nodes`."""
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"edges must be a (2, E) array of column indices, got {edges.shape}")
    edges = edges.astype(np.int64, copy=False)
    if edges.min() < 0 or edges.max() >= n_columns:
        raise ValueError(f"edges refer to columns outside 0..{n_columns - 1}")
    return edges


def _as_metadata(metadata):
    """Accepts the record as a dict, a JSON string or the 0-d array numpy.load returns."""
    if metadata is None:
        return {}
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    if isinstance(metadata, (str, bytes)):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict or a JSON object")
    return metadata


def _json_ready(value):
    """
    Converts numpy scalars and arrays to Python numbers and lists and NaN or
    infinite floats to None, so the result serialises as strict JSON.
    """
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _expand_ranges(low, high):
    """
    Enumerates the half-open integer ranges low[k] .. high[k].

    Returns:
        tuple: (owner, member) int64 arrays, owner the range each entry came
        from and member the integer, in order of k then of member.
    """
    counts = high - low
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(len(low), dtype=np.int64), counts)
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    return owner, np.repeat(low, counts) + (np.arange(total, dtype=np.int64) - run_start)


def _position_in_sorted(sorted_keys, wanted):
    """Position of each of `wanted` in the sorted array `sorted_keys`, or -1 when absent."""
    if sorted_keys.size == 0:
        return np.full(wanted.shape, -1, dtype=np.int64)
    slot = np.minimum(np.searchsorted(sorted_keys, wanted), sorted_keys.size - 1)
    return np.where(sorted_keys[slot] == wanted, slot, -1)


def _weighted_percentile(values, weights, fraction):
    """
    The smallest value whose cumulative weight reaches `fraction` of the total.

    Entries of zero weight cannot be the answer, so a zero-length edge never
    sets a percentile; None when the total weight is zero.
    """
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] if cumulative.size else 0.0
    if not total > 0.0:
        return None
    slot = int(np.searchsorted(cumulative, fraction * total, side="left"))
    return float(values[order][min(slot, values.size - 1)])


def _polyline_geometry(nodes):
    """
    Lays out the points of an archive by polyline.

    Returns:
        dict: "columns" the column index of every finite point in archive
        order, "points" their (n, 3) coordinates, "radii" their radii (half
        the diameter; a missing or negative diameter counts as zero),
        "polyline" the polyline of each point, "starts"/"ends" the point
        range of each polyline, "arc" the arc-length position of each point
        along its polyline and "length" the arc length of each polyline.
    """
    finite = np.isfinite(nodes[:3]).all(axis=0)
    columns = np.flatnonzero(finite)
    n = columns.size
    polyline = graph.polyline_of_column(nodes)[columns]
    points = nodes[:3, columns].T
    radii = np.maximum(np.nan_to_num(nodes[3, columns] / 2.0, nan=0.0), 0.0)
    first = np.ones(n, dtype=bool)
    first[1:] = polyline[1:] != polyline[:-1]
    starts = np.flatnonzero(first)
    ends = np.append(starts[1:], n)
    step = np.zeros(n)
    inside = ~first[1:]
    step[1:][inside] = np.linalg.norm(points[1:][inside] - points[:-1][inside], axis=1)
    cumulative = np.cumsum(step)
    arc = cumulative - cumulative[starts][polyline] if n else cumulative
    length = cumulative[ends - 1] - cumulative[starts] if n else np.zeros(0)
    return {"columns": columns, "points": points, "radii": radii, "polyline": polyline,
            "starts": starts, "ends": ends, "arc": arc, "length": length}


def _representatives(geometry, canonical):
    """
    Restricts a polyline geometry to the columns that represent their vertex
    (canonical[c] == c), so that every point of the network is one point.

    A junction or a tip appears in the archive as several coincident columns
    on different polylines: the parent's last point, each daughter's first,
    and the single-point polylines the interpreter leaves when it restores a
    branch point. Those copies carry no geometry of their own, and judged as
    points of their own short polylines they would lose the junction
    relations of the vessel they sit on. Polyline ids are kept, so a polyline
    reduced to nothing has an empty range.
    """
    n_columns_total = int(canonical.size)
    full_vertex = canonical[geometry["columns"]]
    incidence, first_seen = np.unique(geometry["polyline"] * n_columns_total + full_vertex, return_index=True)
    incidence_arc = geometry["arc"][first_seen]      # where along its polyline each vertex sits
    n_polylines = geometry["starts"].size
    # a polyline that returns to its first vertex is a ring, judged on every
    # column since the closing copy of the first point is dropped below
    closed = np.zeros(n_polylines, dtype=bool)
    spans = geometry["ends"] - geometry["starts"]
    ring = spans > 2
    closed[ring] = (full_vertex[geometry["starts"][ring]] == full_vertex[geometry["ends"][ring] - 1])
    keep = canonical[geometry["columns"]] == geometry["columns"]
    columns = geometry["columns"][keep]
    polyline = geometry["polyline"][keep]
    counts = np.bincount(polyline, minlength=n_polylines)
    ends = np.cumsum(counts)
    starts = ends - counts
    points = geometry["points"][keep]
    n = columns.size
    # arc positions stay those of the full polyline, measured from its first
    # column, so that they agree with the positions recorded for the vertices
    arc = geometry["arc"][keep]
    length = np.zeros(n_polylines)
    filled = counts > 0
    if n:
        length[filled] = arc[ends[filled] - 1] - arc[starts[filled]]
        seam = filled & closed
        if seam.any():
            # the ring's length includes the edge back from its last kept point to its first
            length[seam] += np.linalg.norm(points[ends[seam] - 1] - points[starts[seam]], axis=1)
    closed &= filled & (length > 0.0)
    return {"columns": columns, "points": points, "radii": geometry["radii"][keep], "polyline": polyline,
            "starts": starts, "ends": ends, "arc": arc, "length": length, "filled": filled,
            "closed": closed, "incidence": incidence, "incidence_arc": incidence_arc}


def _edge_geometry(nodes, edges):
    """Lengths and unit tangents of the edges; a zero-length edge has a zero tangent."""
    positions = nodes[:3].T
    vector = positions[edges[1]] - positions[edges[0]]
    length = np.linalg.norm(vector, axis=1)
    tangent = np.zeros_like(vector)
    positive = length > 0.0
    tangent[positive] = vector[positive] / length[positive, None]
    return length, tangent


def _diameter_stats(vertex_diameter, vertices, edges, length):
    """Length-weighted percentiles of the edge diameter and the vertex extremes."""
    stats = {"p50": None, "p90": None, "p99": None, "min": None, "max": None}
    at_vertices = vertex_diameter[vertices]
    at_vertices = at_vertices[np.isfinite(at_vertices)]
    if at_vertices.size:
        stats["min"] = float(at_vertices.min())
        stats["max"] = float(at_vertices.max())
    if edges.shape[1]:
        edge_diameter = (vertex_diameter[edges[0]] + vertex_diameter[edges[1]]) / 2.0
        known = np.isfinite(edge_diameter)
        for key, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
            stats[key] = _weighted_percentile(edge_diameter[known], length[known], fraction)
    return stats


def _segment_stats(nodes, paths):
    """
    Arc/chord ratios of the open segments and the turning-angle curvature of
    consecutive edges, over the segment paths graph.segments returns.
    """
    arc_chord = {"mean": None, "median": None, "p90": None, "max": None,
                 "n_segments": 0, "n_closed": 0}
    curvature = {"mean_abs": None, "n_pairs": 0}
    if not paths:
        return arc_chord, curvature
    positions = nodes[:3].T
    sizes = np.array([len(path) for path in paths], dtype=np.int64)
    flat = np.concatenate(paths)
    segment = np.repeat(np.arange(len(paths), dtype=np.int64), sizes)
    offsets = np.cumsum(sizes) - sizes

    # consecutive entries of one path are its edges, in walking order
    inside = segment[1:] == segment[:-1]
    tail, head = flat[:-1][inside], flat[1:][inside]
    edge_segment = segment[:-1][inside]
    vector = positions[head] - positions[tail]
    length = np.linalg.norm(vector, axis=1)

    first, last = flat[offsets], flat[offsets + sizes - 1]
    arc = np.bincount(edge_segment, weights=length, minlength=len(paths))
    chord = np.linalg.norm(positions[last] - positions[first], axis=1)
    open_segment = (first != last) & (chord > 0.0)
    arc_chord["n_closed"] = int(np.count_nonzero(~open_segment))
    if open_segment.any():
        ratio = arc[open_segment] / chord[open_segment]
        arc_chord.update({"mean": float(ratio.mean()), "median": float(np.percentile(ratio, 50)),
                          "p90": float(np.percentile(ratio, 90)), "max": float(ratio.max()),
                          "n_segments": int(ratio.size)})

    # consecutive edges of one path meet at an interior vertex of the path
    pair = (edge_segment[1:] == edge_segment[:-1]) & (length[1:] > 0.0) & (length[:-1] > 0.0)
    edge_a = np.flatnonzero(pair)
    edge_b = edge_a + 1
    # the vertex where a closed path returns to its start is an interior
    # vertex like any other, so its last edge turns into its first; path p's
    # edges start at offsets[p] - p, one boundary having been dropped per
    # earlier path
    closed = (first == last) & (sizes >= 3)
    if closed.any():
        first_edge = (offsets - np.arange(len(paths), dtype=np.int64))[closed]
        last_edge = first_edge + sizes[closed] - 2
        turning = (length[first_edge] > 0.0) & (length[last_edge] > 0.0)
        edge_a = np.concatenate([edge_a, last_edge[turning]])
        edge_b = np.concatenate([edge_b, first_edge[turning]])
    # atan2 of the cross and dot products is exact for collinear edges and
    # accurate at the small angles a smooth vessel turns through
    if edge_a.size:
        t_a = vector[edge_a] / length[edge_a, None]
        t_b = vector[edge_b] / length[edge_b, None]
        angle = np.arctan2(np.linalg.norm(np.cross(t_a, t_b), axis=1), np.sum(t_a * t_b, axis=1))
        weight = (length[edge_a] + length[edge_b]) / 2.0
        curvature = {"mean_abs": float(angle.sum() / weight.sum()), "n_pairs": int(edge_a.size)}
    return arc_chord, curvature


def _orientation(tangent, length):
    """Length-weighted tangent covariance, its anisotropy and the direction histograms."""
    empty = {"eigenvalues": None, "principal_axis": None, "fractional_anisotropy": None,
             "polar_histogram_deg": None, "azimuth_histogram_deg": None}
    weight = length[length > 0.0]
    if weight.size == 0:
        return empty
    t = tangent[length > 0.0]
    total = weight.sum()
    covariance = (t * weight[:, None]).T @ t / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis = eigenvectors[:, order[0]]
    if axis[np.argmax(np.abs(axis))] < 0.0:
        axis = -axis
    deviation = np.sqrt(np.sum((eigenvalues - 1.0 / 3.0) ** 2))
    anisotropy = math.sqrt(1.5) * deviation / np.sqrt(np.sum(eigenvalues ** 2))

    polar = np.degrees(np.arccos(np.clip(np.abs(t[:, 2]), 0.0, 1.0)))
    polar_hist, _ = np.histogram(polar, bins=POLAR_BINS, range=(0.0, POLAR_BINS * POLAR_BIN_DEG),
                                 weights=weight)
    # unoriented tangents: flip each into the half plane t_y >= 0, with t_x >= 0
    # on the boundary, so the azimuth runs from 0 (x) through 90 (y) to 180.
    # An edge is weighted by the length of its projection onto the xy plane,
    # since a vertical edge has no azimuth and a steep one only a little
    flip = (t[:, 1] < 0.0) | ((t[:, 1] == 0.0) & (t[:, 0] < 0.0))
    planar = np.where(flip[:, None], -t[:, :2], t[:, :2])
    azimuth = np.degrees(np.arctan2(planar[:, 1], planar[:, 0]))
    planar_weight = weight * np.hypot(planar[:, 0], planar[:, 1])
    planar_total = planar_weight.sum()
    azimuth_fraction = None
    if planar_total > 0.0:
        azimuth_hist, _ = np.histogram(azimuth, bins=AZIMUTH_BINS,
                                       range=(0.0, AZIMUTH_BINS * AZIMUTH_BIN_DEG), weights=planar_weight)
        azimuth_fraction = [float(v) for v in azimuth_hist / planar_total]
    return {"eigenvalues": [float(v) for v in eigenvalues],
            "principal_axis": [float(v) for v in axis],
            "fractional_anisotropy": float(anisotropy),
            "polar_histogram_deg": [float(v) for v in polar_hist / total],
            "azimuth_histogram_deg": azimuth_fraction}


def _positive_number(value):
    """True for a finite, positive int or float; a bool is not a number here."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(math.isfinite(value) and value > 0.0)


def _reference_diameter(d_ref, metadata):
    """
    The reference diameter the calibre classes are judged against, with the
    name of where it came from; (None, None) when nothing defines one.

    The generator's d_min is the smallest diameter it draws, so a network
    grown with one has its capillaries at one to two d_min, which is what the
    classes are built on. A record without a usable d_min (growth stopped on
    the iteration count instead) leaves the classes undefined rather than
    taking the thinnest vertex, whose diameter says nothing about where the
    capillaries of that network end.
    """
    if d_ref is not None:
        if not _positive_number(d_ref):
            raise ValueError(f"d_ref must be a positive number, got {d_ref!r}")
        return float(d_ref), "argument"
    if _positive_number(metadata.get("d_min")):
        return float(metadata["d_min"]), "d_min"
    kwargs = metadata.get("grow_kwargs")
    if isinstance(kwargs, dict) and _positive_number(kwargs.get("d_min")):
        return float(kwargs["d_min"]), "grow_kwargs"
    return None, None


def _vector3(value, name):
    """`value` as a (3,) float array of finite entries, else ValueError."""
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return vector


def _unit(vector, name):
    """`vector` scaled to unit length, else ValueError for the zero vector."""
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return vector / norm


def _frame_vector(frame):
    """
    The unit vector a frame record names, as ("axis", a) or ("plane", n);
    None for no frame or a frame of kind "none".
    """
    if frame is None:
        return None
    if not isinstance(frame, dict):
        raise ValueError("frame must be a dict, the frame record of an archive's metadata")
    kind = frame.get("kind", "none")
    if kind == "none":
        return None
    if kind == "axis":
        return "axis", _unit(_vector3(frame.get("axis"), "frame axis"), "frame axis")
    if kind == "plane":
        return "plane", _unit(_vector3(frame.get("normal"), "frame normal"), "frame normal")
    raise ValueError(f"frame kind must be 'none', 'axis' or 'plane', got {kind!r}")


def _watson_moment(concentration):
    """
    <u^2> of the Watson distribution of the given concentration, u the
    cosine to its axis, by quadrature with the exponent shifted by its
    maximum so that no concentration in the search range overflows.
    """
    exponent = concentration * _WATSON_U ** 2 - max(concentration, 0.0)
    density = _WATSON_WEIGHT * np.exp(exponent)
    return float(np.sum(density * _WATSON_U ** 2) / np.sum(density))


def _langevin(concentration):
    """coth K - 1 / K, from its series where the two terms would cancel."""
    if concentration < 1e-6:
        return concentration / 3.0 - concentration ** 3 / 45.0
    return 1.0 / math.tanh(concentration) - 1.0 / concentration


def _bisect_increasing(function, target, low, high):
    """
    The argument in [low, high] at which the increasing `function` reaches
    `target`, by bisection; None when the target lies outside the function's
    range on the interval.
    """
    if not function(low) <= target <= function(high):
        return None
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2.0
        if not low < mid < high:
            break
        if function(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _axial_mean(concentration):
    """<|u|> of the Fisher-axial law exp(K |u|), 1 / (1 - e^-K) - 1 / K, from its series near K = 0."""
    if abs(concentration) < 1e-2:
        return 0.5 + concentration / 12.0 - concentration ** 3 / 720.0 + concentration ** 5 / 30240.0
    return -1.0 / math.expm1(-concentration) - 1.0 / concentration


def _fisher_axial_exact_concentration(mean_abs):
    """The K of exp(K |u|) whose <|u|> is `mean_abs`, or None outside [-_CONCENTRATION_MAX, _CONCENTRATION_MAX]."""
    return _bisect_increasing(_axial_mean, mean_abs, -_CONCENTRATION_MAX, _CONCENTRATION_MAX)


def _watson_concentration(mean_square):
    """The Watson concentration whose <u^2> is `mean_square`, or None outside the searched range."""
    return _bisect_increasing(_watson_moment, mean_square, -_CONCENTRATION_MAX, _CONCENTRATION_MAX)


def _fisher_axial_concentration(mean_abs):
    """The concentration K with coth K - 1 / K equal to `mean_abs`, or None outside (0, _CONCENTRATION_MAX]."""
    # the function rises from 0 at K = 0, which is not a concentration
    if not mean_abs > 0.0:
        return None
    return _bisect_increasing(_langevin, mean_abs, 0.0, _CONCENTRATION_MAX)


def _axis_entry(cosine, weight, d_ref):
    """Alignment of one class of edges to the frame axis, from the signed cosines; None without length."""
    total = float(weight.sum())
    if not total > 0.0:
        return None
    absolute = np.abs(cosine)
    square = cosine * cosine
    angle = np.degrees(np.arccos(np.minimum(absolute, 1.0)))
    mean_abs = float(weight @ absolute) / total
    return {"S": float(weight @ ((3.0 * square - 1.0) / 2.0)) / total,
            "mean_abs_cos": mean_abs,
            "crossing_ratio": 1.0 / mean_abs if mean_abs > 0.0 else None,
            "mean_angle_deg": float(weight @ angle) / total,
            "within_20_deg": float(weight[angle <= 20.0].sum()) / total,
            "within_45_deg": float(weight[angle <= 45.0].sum()) / total,
            "watson_K": _watson_concentration(float(weight @ square) / total),
            "fisher_axial_K": _fisher_axial_concentration(mean_abs),
            "length": total,
            "length_d": total / d_ref if d_ref is not None else None,
            "fisher_axial_K_exact": _fisher_axial_exact_concentration(mean_abs)}


def _plane_entry(cosine, weight, d_ref):
    """How much of one class of edges lies in the frame plane, from the cosines to its normal; None without length."""
    total = float(weight.sum())
    if not total > 0.0:
        return None
    square = cosine * cosine
    return {"in_plane_fraction": float(weight @ (1.0 - square)) / total,
            "S_n": float(weight @ ((3.0 * square - 1.0) / 2.0)) / total,
            "length": total,
            "length_d": total / d_ref if d_ref is not None else None}


def _frame_orientation(vector, tangent, length, classes, d_ref):
    """The axis or plane entry of every class; a class that is undefined (None mask) stays None."""
    kind, direction = vector
    cosine = tangent @ direction
    entry = _axis_entry if kind == "axis" else _plane_entry
    result = {"kind": kind}
    for name, mask in classes.items():
        result[name] = None if mask is None else entry(cosine[mask], length[mask], d_ref)
    return result


def _polar_order(cosine, length, label, senses, classes):
    """
    Length-weighted mean of the signed cosine to the frame axis, each edge
    taken with the recorded sense of the tree its higher column belongs to;
    an edge whose label indexes no sense (a bridge, a separator) is left out.
    """
    senses = np.asarray([] if senses is None else senses, dtype=float).reshape(-1)
    labelled = (label >= 0) & (label < senses.size)
    signed = np.zeros(cosine.size)
    signed[labelled] = senses[label[labelled]] * cosine[labelled]
    result = {}
    for name, mask in classes.items():
        if mask is None:
            result[name] = None
            continue
        inside = mask & labelled
        total = float(length[inside].sum())
        result[name] = float(length[inside] @ signed[inside]) / total if total > 0.0 else None
    return result


def _orientation_shape(tangent, length):
    """_orientation over one class of edges, with the order about its principal axis and its planarity appended."""
    entry = _orientation(tangent, length)
    eigenvalues = entry["eigenvalues"]
    entry["S_max"] = None if eigenvalues is None else (3.0 * eigenvalues[0] - 1.0) / 2.0
    entry["planarity"] = None if eigenvalues is None else 1.0 - 3.0 * eigenvalues[2]
    return entry


def _calibre_shares(edge_diameter, length, d_ref):
    """Shares of length and of volume held by the edges thinner than each multiple of d_ref."""
    known = np.isfinite(edge_diameter)
    diameter, weight = edge_diameter[known], length[known]
    volume = math.pi / 4.0 * diameter ** 2 * weight
    shares = {}
    for name, weights in (("length", weight), ("volume", volume)):
        total = float(weights.sum())
        shares[name] = {f"below_{multiple:g}": float(weights[diameter < multiple * d_ref].sum()) / total
                        if total > 0.0 else None for multiple in CALIBRE_MULTIPLES}
    return shares


def _segment_calibres(nodes, paths, vertex_diameter):
    """
    Length and length-weighted mean diameter of every segment path, in the
    order graph.segments returns them; NaN for a segment whose diameter is
    unknown on an edge of positive length.
    """
    if not paths:
        return np.zeros(0), np.zeros(0)
    positions = nodes[:3].T
    sizes = np.array([len(path) for path in paths], dtype=np.int64)
    flat = np.concatenate(paths)
    segment = np.repeat(np.arange(len(paths), dtype=np.int64), sizes)
    inside = segment[1:] == segment[:-1]
    tail, head = flat[:-1][inside], flat[1:][inside]
    edge_segment = segment[:-1][inside]
    step = np.linalg.norm(positions[head] - positions[tail], axis=1)
    diameter = (vertex_diameter[tail] + vertex_diameter[head]) / 2.0
    # a zero-length edge carries no weight, so its diameter, known or not, does not enter
    weighted = np.where(step > 0.0, step * diameter, 0.0)
    arc = np.bincount(edge_segment, weights=step, minlength=len(paths))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(edge_segment, weights=weighted, minlength=len(paths)) / arc
    return arc, mean


def _length_distribution(values, d_ref):
    """Count, percentiles and coefficient of variation of segment lengths; every statistic None without any."""
    stats = {"count": int(values.size), "median": None, "p10": None, "p90": None, "cv": None,
             "median_d": None, "p10_d": None, "p90_d": None}
    if values.size:
        median, p10, p90 = (float(v) for v in np.percentile(values, [50.0, 10.0, 90.0]))
        mean = float(values.mean())
        stats.update({"median": median, "p10": p10, "p90": p90,
                      "cv": float(values.std()) / mean if values.size >= 2 and mean > 0.0 else None,
                      "median_d": median / d_ref, "p10_d": p10 / d_ref, "p90_d": p90 / d_ref})
    return stats


def _segments_by_class(arc, diameter, threshold, d_ref):
    """Length distributions of the capillary and the larger segments, leaving out those without length or diameter."""
    valid = (arc > 0.0) & np.isfinite(diameter)
    capillary = valid & (diameter < threshold)
    return {"capillary": _length_distribution(arc[capillary], d_ref),
            "larger": _length_distribution(arc[valid & ~capillary], d_ref)}


def _nearest_neighbour_distances(points):
    """Distance from each of the (n, 2) points to the nearest other one: exact, by brute force in chunks."""
    n = points.shape[0]
    out = np.empty(n)
    rows = max(1, _NEIGHBOUR_PAIRS // n)
    for start in range(0, n, rows):
        block = points[start:start + rows]
        squared = ((block[:, 0, None] - points[None, :, 0]) ** 2
                   + (block[:, 1, None] - points[None, :, 1]) ** 2)
        own = np.arange(block.shape[0])
        squared[own, start + own] = np.inf
        out[start:start + rows] = np.sqrt(squared.min(axis=1))
    return out


def _box_corners(origin, basis, extents):
    """
    The eight corners of the box centred on `origin` whose sides run along
    the rows of `basis` with the given extents, and its twelve edges as
    pairs of corner indices.
    """
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    corners = origin + (signs * extents / 2.0) @ basis
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(signs[i] != signs[j]) == 1]
    return corners, pairs


def _section_area(corners, pairs, axis, level, e1, e2):
    """
    Area of the polygon where the plane x . axis = level cuts the box: the
    plane meets the box's edges in the polygon's vertices, which are ordered
    by angle about their centroid in the plane's own coordinates and summed
    by the shoelace formula. A corner on the plane is found from each of its
    edges, and a repeated vertex adds no area.
    """
    s = corners @ axis
    points = []
    for i, j in pairs:
        if s[i] == s[j]:
            if s[i] == level:
                points.extend([corners[i], corners[j]])
            continue
        if (s[i] - level) * (s[j] - level) <= 0.0:
            points.append(corners[i] + (level - s[i]) / (s[j] - s[i]) * (corners[j] - corners[i]))
    if len(points) < 3:
        return 0.0
    points = np.array(points)
    u, v = points @ e1, points @ e2
    order = np.argsort(np.arctan2(v - v.mean(), u - u.mean()), kind="stable")
    u, v = u[order], v[order]
    return 0.5 * abs(float(u @ np.roll(v, -1) - v @ np.roll(u, -1)))


def _transverse_spacing(positions, lower, higher, axis, frame, d_ref, volume, metadata):
    """
    Nearest-neighbour spacing and density of the crossings of the given
    edges (the capillary class, as columns of their lower and higher ends)
    through planes perpendicular to the frame axis; see `describe`.
    """
    if lower.size == 0:
        return None
    x_lo, x_hi = positions[lower], positions[higher]
    s_lo, s_hi = x_lo @ axis, x_hi @ axis
    s_min = float(min(s_lo.min(), s_hi.min()))
    s_max = float(max(s_lo.max(), s_hi.max()))
    if not s_max > s_min:
        return None
    grow = _unit(_vector3(frame.get("grow_direction"), "frame grow_direction"), "frame grow_direction")
    perpendicular = _unit(_vector3(frame.get("grow_perpendicular"), "frame grow_perpendicular"),
                          "frame grow_perpendicular")
    # in-plane axes from the frame itself, so that the result does not depend on the lab axes
    e1 = grow - (grow @ axis) * axis
    if np.linalg.norm(e1) < 1e-9:
        e1 = perpendicular - (perpendicular @ axis) * axis
    e1 = _unit(e1, "the frame's in-plane direction")
    e2 = np.cross(axis, e1)
    levels = s_min + (_TRANSVERSE_FIRST + _TRANSVERSE_STEP * np.arange(TRANSVERSE_PLANES)) * (s_max - s_min)

    crossings = 0
    pooled = 0
    distances = []
    for level in levels:
        crossing = (s_lo < level) != (s_hi < level)
        found = int(np.count_nonzero(crossing))
        crossings += found
        if found < 2:
            continue
        fraction = (level - s_lo[crossing]) / (s_hi[crossing] - s_lo[crossing])
        point = x_lo[crossing] + fraction[:, None] * (x_hi[crossing] - x_lo[crossing])
        distances.append(_nearest_neighbour_distances(np.stack([point @ e1, point @ e2], axis=1)))
        pooled += found

    extents = None
    if volume is not None:
        extents, source = _vector3(volume, "volume"), "argument"
    elif metadata.get("growth_box_um") is not None:
        extents, source = _vector3(metadata["growth_box_um"], "growth_box_um"), "growth_box_um"
    if extents is not None:
        origin = _vector3(frame.get("origin"), "frame origin")
        basis = np.array([_unit(np.cross(grow, perpendicular), "the frame's third axis"), grow, perpendicular])
        corners, pairs = _box_corners(origin, basis, extents)
        area = sum(_section_area(corners, pairs, axis, level, e1, e2) for level in levels)
    else:
        source = "capillary_bounding_box"
        ends = np.concatenate([x_lo, x_hi])
        u, v = ends @ e1, ends @ e2
        area = TRANSVERSE_PLANES * float((u.max() - u.min()) * (v.max() - v.min()))

    result = {"planes": TRANSVERSE_PLANES, "count": pooled, "median": None, "mean": None, "p10": None, "p90": None,
              "median_d": None, "mean_d": None, "p10_d": None, "p90_d": None,
              "crossing_density": None, "crossing_density_d": None, "area_source": source}
    if pooled:
        values = np.concatenate(distances)
        median, p10, p90 = (float(v) for v in np.percentile(values, [50.0, 10.0, 90.0]))
        mean = float(values.mean())
        result.update({"median": median, "mean": mean, "p10": p10, "p90": p90, "median_d": median / d_ref,
                       "mean_d": mean / d_ref, "p10_d": p10 / d_ref, "p90_d": p90 / d_ref})
    if area > 0.0:
        density = crossings / area
        result.update({"crossing_density": density, "crossing_density_d": density * d_ref ** 2})
    return result


def _segment_layout(nodes, paths):
    """
    The segment paths laid end to end.

    Returns:
        dict: "flat" the vertices of every path in turn, "sizes" and
        "offsets" each path's run in it, "segment" the path of each entry,
        "first" and "last" the end vertices of each path, and "cumulative"
        the arc length summed along "flat", which stands still from a path's
        last vertex to the next path's first.
    """
    positions = nodes[:3].T
    sizes = np.array([len(path) for path in paths], dtype=np.int64)
    flat = np.concatenate(paths) if paths else np.zeros(0, dtype=np.int64)
    offsets = np.cumsum(sizes) - sizes
    segment = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    step = np.zeros(flat.size)
    inside = np.flatnonzero(segment[1:] == segment[:-1]) + 1
    step[inside] = np.linalg.norm(positions[flat[inside]] - positions[flat[inside - 1]], axis=1)
    return {"flat": flat, "sizes": sizes, "offsets": offsets, "segment": segment,
            "first": flat[offsets], "last": flat[offsets + sizes - 1], "cumulative": np.cumsum(step)}


def _junctions(degree, vertices, n_segments):
    """Degrees of the vertices of degree three or more, and the segments per junction; see `describe`."""
    junction_degree = degree[vertices][degree[vertices] >= 3]
    count = int(junction_degree.size)
    result = {"count": count, "degree_3": None, "degree_4": None, "degree_5_plus": None,
              "mean_degree": None, "segments_per_junction": None}
    if count:
        result.update({"degree_3": float(np.count_nonzero(junction_degree == 3)) / count,
                       "degree_4": float(np.count_nonzero(junction_degree == 4)) / count,
                       "degree_5_plus": float(np.count_nonzero(junction_degree >= 5)) / count,
                       "mean_degree": float(junction_degree.mean()),
                       "segments_per_junction": n_segments / count})
    return result


def _branch_angles(nodes, layout, degree, vertex_diameter):
    """
    Angles between the three vessels at each vertex of degree three, taken
    between the directions to points a little way along each; see `describe`.
    """
    result = {"count": 0, "skipped": 0, "min": None, "median": None, "max": None}
    positions = nodes[:3].T
    flat, offsets, sizes, cumulative = layout["flat"], layout["offsets"], layout["sizes"], layout["cumulative"]
    # every end of a segment at a vertex of degree three is an arm of it; a
    # closed segment through the vertex gives one arm from each of its ends
    at_first = np.flatnonzero(degree[layout["first"]] == 3)
    at_last = np.flatnonzero(degree[layout["last"]] == 3)
    segment = np.concatenate([at_first, at_last])
    if segment.size == 0:
        return result
    from_first = np.arange(segment.size) < at_first.size
    vertex = np.where(from_first, layout["first"][segment], layout["last"][segment])
    start_arc = cumulative[offsets[segment]]
    end_arc = cumulative[offsets[segment] + sizes[segment] - 1]
    # NaN for a vertex of unknown diameter, which is skipped below
    reach = np.minimum(BRANCH_ARC_DIAMETERS * vertex_diameter[vertex], (end_arc - start_arc) / 2.0)
    target = np.where(from_first, start_arc + reach, end_arc - reach)
    known = np.isfinite(target)
    # the edge of the segment on which the arm ends, found on the cumulative
    # arc, which is flat between segments, so the edge is kept inside the segment
    edge_end = np.clip(np.searchsorted(cumulative, np.where(known, target, 0.0), side="left"),
                       offsets[segment] + 1, offsets[segment] + sizes[segment] - 1)
    tail, head = flat[edge_end - 1], flat[edge_end]
    # an edge of zero length, which only edges a caller supplies can have,
    # gives an arm of no length, skipped below
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.clip((target - cumulative[edge_end - 1]) / (cumulative[edge_end] - cumulative[edge_end - 1]),
                           0.0, 1.0)
    arm = positions[tail] + fraction[:, None] * (positions[head] - positions[tail]) - positions[vertex]
    norm = np.linalg.norm(arm, axis=1)

    order = np.lexsort((from_first, segment, vertex))
    arm, norm, known = arm[order], norm[order], known[order]
    # a vertex of degree three has exactly three segment ends at it
    good = (known & (norm > 0.0)).reshape(-1, 3).all(axis=1)
    result["skipped"] = int(np.count_nonzero(~good))
    unit = arm.reshape(-1, 3, 3)[good] / norm.reshape(-1, 3)[good][:, :, None]
    if unit.shape[0] == 0:
        return result
    angles = np.sort(np.stack([np.degrees(np.arctan2(np.linalg.norm(np.cross(unit[:, i], unit[:, j]), axis=1),
                                                     np.sum(unit[:, i] * unit[:, j], axis=1)))
                               for i, j in ((0, 1), (0, 2), (1, 2))], axis=1), axis=1)
    smallest, middle, largest = (float(np.percentile(angles[:, k], 50.0)) for k in range(3))
    result.update({"count": int(unit.shape[0]), "min": smallest, "median": middle, "max": largest})
    return result


def _two_core(n_vertices, a, b):
    """
    The 2-core of the multigraph whose edges join a[k] and b[k]: what is left
    after repeatedly removing every vertex of degree one or less together
    with its edges. Every cycle lies in it, and so does every shortest path
    between two of its vertices.

    Returns:
        tuple: boolean masks of the vertices and the edges that remain.
    """
    n_edges = a.size
    ends = np.concatenate([a, b])
    order = np.argsort(ends, kind="stable")
    incident = np.concatenate([np.arange(n_edges), np.arange(n_edges)])[order]
    start = np.searchsorted(ends[order], np.arange(n_vertices + 1))
    # a self-loop counts twice, so a vertex that closes a loop on itself stays
    degree = np.bincount(ends, minlength=n_vertices)
    vertex_alive = np.ones(n_vertices, dtype=bool)
    edge_alive = np.ones(n_edges, dtype=bool)
    leaves = np.flatnonzero(degree <= 1)
    # each pass removes the current leaves, so a pendant tree goes in as many
    # passes as it is deep, touching only what it removes
    while leaves.size:
        vertex_alive[leaves] = False
        _, slot = _expand_ranges(start[leaves], start[leaves + 1])
        removed = np.unique(incident[slot])
        removed = removed[edge_alive[removed]]
        edge_alive[removed] = False
        np.subtract.at(degree, a[removed], 1)
        np.subtract.at(degree, b[removed], 1)
        touched = np.unique(np.concatenate([a[removed], b[removed]]))
        leaves = touched[vertex_alive[touched] & (degree[touched] <= 1)]
    return vertex_alive, edge_alive


def _loops_by_segment(a, b, closed, parallel, in_core, start, target, segment, sample):
    """
    Size in segments of the shortest loop through each sampled segment, 0
    when there is none within the caps; see `describe`.

    The search runs from both ends of the segment at once, a whole level at
    a time from the end whose frontier is smaller, over the core's neighbour
    lists `start`, `target` and `segment`. When a level first reaches a
    vertex the other side has reached, the path through it is a shortest
    path: every path of one segment fewer would have been met a level
    earlier, and none is shorter than the levels expanded so far allow.
    """
    n_vertices = len(start) - 1
    # mark[v] = 2 search + side once a search has reached v, level[v] its distance from that side's end
    mark = [-1] * n_vertices
    level = [0] * n_vertices
    sizes = []
    for search, e in enumerate(sample):
        if closed[e]:
            sizes.append(1)
            continue
        if parallel[e]:
            sizes.append(2)
            continue
        if not in_core[e]:
            sizes.append(0)
            continue
        base = 2 * search
        ends = (int(a[e]), int(b[e]))
        mark[ends[0]], mark[ends[1]] = base, base + 1
        level[ends[0]] = level[ends[1]] = 0
        fronts = [[ends[0]], [ends[1]]]
        depth = [0, 0]
        reached = 2
        size = 0
        while not size and fronts[0] and fronts[1] and depth[0] + depth[1] < LOOP_DEPTH:
            side = 0 if len(fronts[0]) <= len(fronts[1]) else 1
            own, other = base + side, base + 1 - side
            following = []
            for x in fronts[side]:
                for slot in range(start[x], start[x + 1]):
                    if segment[slot] == e:
                        continue
                    y = target[slot]
                    if mark[y] == own:
                        continue
                    if mark[y] == other:
                        # the path, of depth + 1 + level segments, and the segment itself
                        size = depth[side] + level[y] + 2
                        break
                    if reached >= LOOP_VERTICES:
                        size = -1
                        break
                    mark[y], level[y] = own, depth[side] + 1
                    reached += 1
                    following.append(y)
                if size:
                    break
            depth[side] += 1
            fronts[side] = following
        sizes.append(max(size, 0))
    return sizes


def _loops_by_node(vertex_in_core, self_loop, start, target, segment, weight, sample):
    """
    The shortest loop through each sampled junction, as (segments, length),
    by length and then by segment count; (0, None) when there is none within
    the caps. See `describe`.

    Dijkstra's search from the junction s labels every vertex it settles with
    the segment at s its path leaves by. A segment between two settled
    vertices of different labels, other than the one either was reached by,
    closes a loop through s. The search stops once no loop it could still
    close is shorter than the best. Such a loop joins a vertex y settled
    from now on, at distance d(y) at least the distance d just popped, to a
    vertex x by a segment of weight w. If x searched on from where it was
    settled, d(y) <= d(x) + w and the loop is at least 2 d long; if x sat
    at the depth cap and did not, the loop is still at least d + d(x) + w,
    w no less than the lightest segment at x.
    """
    n_vertices = len(start) - 1
    lightest = [min((weight[segment[slot]] for slot in range(start[v], start[v + 1])), default=math.inf)
                for v in range(n_vertices)]
    settled_by = [-1] * n_vertices
    distance = [0.0] * n_vertices
    hops = [0] * n_vertices
    label = [0] * n_vertices
    loops = []
    for search, s in enumerate(sample):
        if not vertex_in_core[s]:
            loops.append((0, None))
            continue
        best = (self_loop.get(s, math.inf), 1)
        heap = [(0.0, 0, s, -1, -1)]
        settled = 0
        # the least d(x) + w over the vertices settled at the depth cap
        capped = math.inf
        while heap:
            d, h, x, branch, via = heapq.heappop(heap)
            if settled_by[x] == search:
                continue
            if min(2.0 * d, d + capped) > best[0] or settled >= LOOP_VERTICES:
                break
            settled_by[x] = search
            settled += 1
            distance[x], hops[x], label[x] = d, h, branch
            if h >= LOOP_DEPTH:
                capped = min(capped, d + lightest[x])
            for slot in range(start[x], start[x + 1]):
                e = segment[slot]
                if e == via:
                    continue
                y = target[slot]
                if settled_by[y] == search:
                    if label[y] != branch:
                        candidate = (d + weight[e] + distance[y], h + 1 + hops[y])
                        if candidate < best:
                            best = candidate
                elif h < LOOP_DEPTH and d + weight[e] < best[0]:
                    # a vertex at least as far as the best loop is long closes no shorter one
                    heapq.heappush(heap, (d + weight[e], h + 1, y, e if x == s else branch, e))
        loops.append((best[1], best[0]) if math.isfinite(best[0]) else (0, None))
    return loops


def _loop_summary(sizes):
    """Found count, share without a loop, median, mean and histogram of loop sizes (0 meaning none)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    found = sizes[sizes > 0]
    values, counts = np.unique(found, return_counts=True)
    return {"sampled": int(sizes.size), "found": int(found.size),
            "none_fraction": float(sizes.size - found.size) / sizes.size if sizes.size else None,
            "median": float(np.percentile(found, 50.0)) if found.size else None,
            "mean": float(found.mean()) if found.size else None,
            "histogram": {str(int(v)): int(c) for v, c in zip(values, counts)}}


def _loops(layout, arc, degree, vertices, cycles, total_length, d_ref):
    """The shortest loops through sampled segments and junctions; see `describe`."""
    first, last = layout["first"], layout["last"]
    n_segments = int(first.size)
    # the segment multigraph, its vertices numbered densely in column order
    columns, dense = np.unique(np.concatenate([first, last]), return_inverse=True)
    dense = dense.reshape(-1)
    n_vertices = int(columns.size)
    a, b = dense[:n_segments], dense[n_segments:]
    closed = a == b
    pair = np.minimum(a, b) * max(n_vertices, 1) + np.maximum(a, b)
    _, inverse, multiplicity = np.unique(pair, return_inverse=True, return_counts=True)
    parallel = ~closed & (multiplicity[inverse.reshape(-1)] >= 2)
    vertex_in_core, in_core = _two_core(n_vertices, a, b)
    # neighbour lists of the core in compressed form, sorted by neighbour and
    # then by segment so that every search is a function of the edge list;
    # plain lists, since the searches are Python loops
    kept = np.flatnonzero(in_core & ~closed)
    source = np.concatenate([a[kept], b[kept]])
    target = np.concatenate([b[kept], a[kept]])
    segment = np.concatenate([kept, kept])
    order = np.lexsort((segment, target, source))
    start = np.searchsorted(source[order], np.arange(n_vertices + 1)).tolist()
    target, segment = target[order].tolist(), segment[order].tolist()

    every = max(1, -(-n_segments // LOOP_SAMPLES))
    sizes = _loops_by_segment(a, b, closed, parallel, in_core, start, target, segment,
                              range(0, n_segments, every))
    by_segment = {"segments": n_segments, "every": every}
    by_segment.update(_loop_summary(sizes))
    by_segment.update({"cycles": cycles,
                       "cycles_per_length": cycles / total_length if total_length > 0.0 else None,
                       "cycles_per_length_d": cycles * d_ref / total_length
                       if total_length > 0.0 and d_ref is not None else None})

    junction = vertices[degree[vertices] >= 3]
    every = max(1, -(-junction.size // LOOP_SAMPLES))
    self_loop = {}
    for e in np.flatnonzero(closed).tolist():
        vertex = int(a[e])
        self_loop[vertex] = min(self_loop.get(vertex, math.inf), float(arc[e]))
    found = _loops_by_node(vertex_in_core.tolist(), self_loop, start, target, segment, arc.tolist(),
                           np.searchsorted(columns, junction[::every]).tolist())
    lengths = np.array([length for _, length in found if length is not None])
    by_node = {"junctions": int(junction.size), "every": every}
    by_node.update(_loop_summary([size for size, _ in found]))
    by_node.update({"length_median": None, "length_mean": None, "length_median_d": None, "length_mean_d": None})
    if lengths.size:
        median, mean = float(np.percentile(lengths, 50.0)), float(lengths.mean())
        by_node.update({"length_median": median, "length_mean": mean,
                        "length_median_d": median / d_ref if d_ref is not None else None,
                        "length_mean_d": mean / d_ref if d_ref is not None else None})
    return {"by_segment": by_segment, "by_node": by_node}


def _diameter_variation(layout, vertex_diameter, degree, arc, mean_diameter, threshold):
    """(max - min) / mean of the vertex diameters along each segment, by class; see `describe`."""
    flat, offsets, sizes, segment = layout["flat"], layout["offsets"], layout["sizes"], layout["segment"]
    n_segments = int(sizes.size)
    last = offsets + sizes - 1
    keep = np.ones(flat.size, dtype=bool)
    # the closing repeat of a closed segment, and every end at a junction,
    # whose diameter is the widest vessel's there
    keep[last[layout["first"] == layout["last"]]] = False
    for end in (offsets, last):
        keep[end[degree[flat[end]] >= 3]] = False
    diameter = vertex_diameter[flat]
    keep &= np.isfinite(diameter)
    owner, diameter = segment[keep], diameter[keep]
    count = np.bincount(owner, minlength=n_segments)
    largest = np.full(n_segments, -np.inf)
    smallest = np.full(n_segments, np.inf)
    np.maximum.at(largest, owner, diameter)
    np.minimum.at(smallest, owner, diameter)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(owner, weights=diameter, minlength=n_segments) / count
        variation = (largest - smallest) / mean
    defined = (count >= 2) & (mean > 0.0)

    def summary(members):
        values = variation[members & defined]
        entry = {"count": int(values.size), "excluded": int(np.count_nonzero(members & ~defined)),
                 "median": None, "p90": None}
        if values.size:
            median, p90 = (float(v) for v in np.percentile(values, [50.0, 90.0]))
            entry.update({"median": median, "p90": p90})
        return entry

    result = {"all": summary(np.ones(n_segments, dtype=bool)), "capillary": None, "larger": None}
    if threshold is not None:
        # the classes of "segments_by_class"
        valid = (arc > 0.0) & np.isfinite(mean_diameter)
        capillary = valid & (mean_diameter < threshold)
        result.update({"capillary": summary(capillary), "larger": summary(valid & ~capillary)})
    return result


def _volume_um3(points, r_max, volume, metadata):
    """
    The reference volume for the length density, in cubic micrometres, with
    the name of where it came from; (None, source) when nothing defines one.
    """
    if volume is not None:
        extents = np.asarray(volume, dtype=float).reshape(-1)
        if extents.size != 3 or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
            raise ValueError("volume must be three positive extents in micrometres")
        return float(np.prod(extents)), "argument"
    box = metadata.get("growth_box_um")
    if box is not None:
        return float(np.prod(np.asarray(box, dtype=float))), "growth_box_um"
    shape, voxel = metadata.get("volume"), metadata.get("voxel_size")
    if metadata.get("fit") == "voxel_size" and shape is not None and voxel is not None:
        return float(np.prod(np.asarray(shape, dtype=float)) * float(voxel) ** 3), "voxel_size"
    if len(points) == 0:
        return None, "bounding_box"
    extents = points.max(axis=0) - points.min(axis=0) + 2.0 * r_max
    return float(np.prod(extents)), "bounding_box"


def _tissue_domain(points, r_max, volume, metadata):
    """
    The box the tissue distance is sampled in: the reference box of the
    length density, from the same source, placed as that source places it.

    Returns:
        tuple: (low, high, source), the box's corners and the source's name
        as _volume_um3 gives it.
    """
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    if volume is not None:
        extents = _vector3(volume, "volume")
        return centre - extents / 2.0, centre + extents / 2.0, "argument"
    if metadata.get("growth_box_um") is not None:
        # grow_network's box runs from the origin to its extents
        return np.zeros(3), _vector3(metadata["growth_box_um"], "growth_box_um"), "growth_box_um"
    shape, voxel = metadata.get("volume"), metadata.get("voxel_size")
    if metadata.get("fit") == "voxel_size" and shape is not None and voxel is not None:
        # the voxeliser puts the centre of the network's bounding box at voxel
        # coordinate n / 2 of the n along an axis, voxel i spanning
        # [i - 1/2, i + 1/2), so the field runs from n / 2 + 1/2 voxels below
        # that centre to n / 2 - 1/2 above it
        cells, voxel = _vector3(shape, "volume"), float(voxel)
        return centre - (cells + 1.0) / 2.0 * voxel, centre + (cells - 1.0) / 2.0 * voxel, "voxel_size"
    return low - r_max, high + r_max, "bounding_box"


def _evd_grid(low, high, spacing):
    """
    The tissue-distance grid over the box from `low` to `high`; see `describe`.

    Returns:
        tuple: (h, counts, axes), the spacing used, the number of points along
        each axis and their coordinates along it.
    """
    extents = np.maximum(high - low, 0.0)

    def counts(h):
        # in floating point, so that a tiny spacing cannot overflow an integer
        # and a count past the largest float is infinite
        with np.errstate(over="ignore"):
            return np.maximum(1.0, np.floor(extents / h + 1e-9))

    def fits(h):
        with np.errstate(over="ignore"):
            return np.prod(counts(h)) <= EVD_MAX_POINTS

    h = float(spacing)
    if not fits(h):
        # The grid only gains points as the spacing falls, so along each axis
        # E_i / m fits for every whole m up to some M_i, which bisection over
        # m finds, and the least E_i / m that fits is the least E_i / M_i. It
        # lies above the requested spacing, which does not fit. E_max / 1
        # fits, every axis then holding one point, and no m above
        # (cap + 1) E_i / E_max does, the longest axis then holding more
        # points than the cap.
        longest = float(extents.max())
        least = []
        for extent in extents[extents > 0.0].tolist():
            # E_i / few fits, few 0 standing for an infinite spacing, and E_i / many does not
            few, many = 0, int((EVD_MAX_POINTS + 1) * (extent / longest)) + 2
            while many - few > 1:
                middle = (few + many) // 2
                if fits(extent / middle):
                    few = middle
                else:
                    many = middle
            if few:
                least.append(extent / few)
        h = min(least)
    n = counts(h).astype(np.int64)
    centre = (low + high) / 2.0
    return h, n, [centre[i] + (np.arange(n[i]) - (n[i] - 1) / 2.0) * h for i in range(3)]


def _capsules(nodes):
    """
    The tapered capsules the voxeliser draws (computeVoxel.rasterise_segments):
    one between each two consecutive finite columns of a polyline that are
    not the same point, with radii half the columns' diameters. A missing or
    negative diameter counts as zero, where the voxeliser would draw nothing
    of the capsule or take the radius by its size; see `describe`.

    Returns:
        tuple: (start, end, r0, r1), the (n, 3) end points and the radii.
    """
    finite = np.isfinite(nodes[:3]).all(axis=0)
    first = np.flatnonzero(finite[:-1] & finite[1:])
    first = first[np.any(nodes[:3, first] != nodes[:3, first + 1], axis=0)]
    radii = np.maximum(np.nan_to_num(nodes[3] / 2.0, nan=0.0), 0.0)
    return nodes[:3, first].T, nodes[:3, first + 1].T, radii[first], radii[first + 1]


def _capsule_values(point, start, axis, length2, r0, dr):
    """
    |p - c(t)| - (r0 + t dr) for each row, c(t) = start + t axis with t the
    projection of p on the axis clamped to [0, 1]: the voxeliser's test of a
    point against a capsule, negative or zero inside it. Written element by
    element, so that a pair rounds alike whichever pass evaluates it.
    """
    ox, oy, oz = point[:, 0] - start[:, 0], point[:, 1] - start[:, 1], point[:, 2] - start[:, 2]
    t = np.clip((ox * axis[:, 0] + oy * axis[:, 1] + oz * axis[:, 2]) / length2, 0.0, 1.0)
    ox, oy, oz = ox - t * axis[:, 0], oy - t * axis[:, 1], oz - t * axis[:, 2]
    return np.sqrt(ox * ox + oy * oy + oz * oz) - (r0 + t * dr)


def _morton_keys(cells, bits):
    """The bits of the (n, 3) non-negative integer cells interleaved, x highest, as int64 keys."""
    keys = np.zeros(cells.shape[0], dtype=np.int64)
    for bit in range(bits):
        for axis in range(3):
            keys |= ((cells[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return keys


def _cell_tree(centres, reach, inner, spacing):
    """
    An octree over the vessel pieces of the tissue-distance search.

    The finest cells are cubes laid from the lowest piece centre, their side
    halved from the grid spacing until they hold at most _EVD_CELL_PIECES
    pieces on average. Cells are numbered in Morton order, so the eight
    children of a cell are contiguous and its parent's key is its own
    shifted by three bits; the root is the single cell of the top level.

    Returns:
        dict: "pieces" the piece numbers in cell order and "levels", from the
        finest to the root, each {"half": half the side of its cells,
        "centre": their centres as three arrays, one per axis, "reach" and
        "inner": the largest piece reach and inner radius inside each,
        "below": where each cell's children (pieces at the finest level)
        start in the level below, with the total last}.
    """
    origin = centres.min(axis=0)
    span = float((centres.max(axis=0) - origin).max())
    floor_size = span / (2 ** _EVD_LEVEL_BITS - 1)
    size = max(float(spacing), floor_size)
    cells = np.floor((centres - origin) / size).astype(np.int64)
    for _ in range(_EVD_HALVINGS):
        occupied = np.unique((cells[:, 0] << (2 * _EVD_LEVEL_BITS)) | (cells[:, 1] << _EVD_LEVEL_BITS)
                             | cells[:, 2]).size
        if centres.shape[0] <= _EVD_CELL_PIECES * occupied or size / 2.0 <= floor_size:
            break
        size /= 2.0
        cells = np.floor((centres - origin) / size).astype(np.int64)
    bits = max(1, int(cells.max()).bit_length())
    keys = _morton_keys(cells, bits)
    pieces = np.argsort(keys, kind="stable")
    keys = keys[pieces]
    first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    corners = [cells[pieces[first]]]
    levels = [{"reach": np.maximum.reduceat(reach[pieces], first), "inner": np.maximum.reduceat(inner[pieces], first),
               "below": np.append(first, keys.size)}]
    keys = keys[first]
    for _ in range(bits):
        keys = keys >> 3
        first = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        corners.append(corners[-1][first] >> 1)
        levels.append({"reach": np.maximum.reduceat(levels[-1]["reach"], first),
                       "inner": np.maximum.reduceat(levels[-1]["inner"], first),
                       "below": np.append(first, keys.size)})
        keys = keys[first]
    for level, (cells_at, corner) in enumerate(zip(levels, corners)):
        half = size * 2.0 ** level / 2.0
        cells_at["half"] = half
        cells_at["centre"] = [origin[i] + corner[:, i] * (2.0 * half) + half for i in range(3)]
    return {"pieces": pieces, "levels": levels}


def _run_minimum(index, values, n):
    """The minimum of `values` for each of n indices, inf where absent; `index` is non-decreasing."""
    out = np.full(n, np.inf)
    if index.size:
        first = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        out[index[first]] = np.minimum.reduceat(values, first)
    return out


def _seed_pieces(axes, spacing, centre):
    """
    A piece near each point of the grid with the given axes, for the first
    upper bound of the exact search; not always the nearest, which the
    search does not need.

    Every piece is placed at the grid point nearest its centre, the piece
    nearest that point winning, and sweeps forwards and backwards along each
    axis in turn then hand each point its neighbour's piece whenever that
    one is nearer to it.

    Returns:
        ndarray: (n,) the piece at each grid point, in the grid's order.
    """
    shape = tuple(a.size for a in axes)
    index = np.stack([np.clip(np.rint((centre[:, i] - axes[i][0]) / spacing), 0, shape[i] - 1) for i in range(3)])
    point = np.ravel_multi_index(index.astype(np.int64), shape)
    coordinates = [np.broadcast_to(axes[i].reshape([-1 if k == i else 1 for k in range(3)]), shape)
                   for i in range(3)]
    offset = np.stack([coordinates[i].reshape(-1)[point] - centre[:, i] for i in range(3)], axis=1)
    distance = np.sqrt(offset[:, 0] * offset[:, 0] + offset[:, 1] * offset[:, 1] + offset[:, 2] * offset[:, 2])
    order = np.lexsort((np.arange(point.size), distance, point))
    first = order[np.r_[True, point[order][1:] != point[order][:-1]]]
    label = np.full(shape, -1, dtype=np.int64)
    best = np.full(shape, np.inf)
    label.reshape(-1)[point[first]] = first
    best.reshape(-1)[point[first]] = distance[first]
    for axis in range(3):
        labels, bests = np.moveaxis(label, axis, 0), np.moveaxis(best, axis, 0)
        along = [np.moveaxis(c, axis, 0) for c in coordinates]
        n = shape[axis]
        for q, r in [(q, q - 1) for q in range(1, n)] + [(q, q + 1) for q in range(n - 2, -1, -1)]:
            candidate = labels[r]
            m = centre[np.maximum(candidate, 0)]
            dx, dy, dz = along[0][q] - m[..., 0], along[1][q] - m[..., 1], along[2][q] - m[..., 2]
            nearer = (candidate >= 0) & (np.sqrt(dx * dx + dy * dy + dz * dz) < bests[q])
            labels[q][nearer] = candidate[nearer]
            bests[q][nearer] = np.sqrt(dx * dx + dy * dy + dz * dz)[nearer]
    return label.reshape(-1)


def _tissue_values(axes, start, end, r0, r1, spacing):
    """
    The minimum of _capsule_values over every capsule at each point of the
    grid with the given axes, exactly, in the grid's order.

    Each capsule is cut into pieces no longer than the spacing. A piece with
    centre m and reach R (half its length plus the larger of its capsule's
    radii) bounds its capsule's value from below by |p - m| - R wherever the
    clamped projection falls inside it; and since the projection is the
    nearest point of the axis, |p - m| - r, r the smaller of the capsule's
    radii (its inner radius), bounds the value from above. The pieces sit in
    the cells of an octree, each cell knowing the largest reach and the
    largest inner radius inside it, so the distances from p to a cell's box
    and to its farthest corner bound every piece inside. Each point starts
    from the value of a capsule near it (_seed_pieces), and a pass descends
    the tree for a chunk of grid points, keeping at each level the cells
    whose lower bound does not exceed the least upper bound met so far, and
    evaluates the capsules of the pieces that survive; a chunk that would
    hold more than _EVD_PAIRS pairs is searched again in halves. The capsule
    that holds the minimum always survives, so the result is the minimum
    over all capsules, and it depends neither on the pruning nor on the
    chunks: every value is computed by the same formula.
    """
    x, y, z = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    axis = end - start
    length2 = axis[:, 0] * axis[:, 0] + axis[:, 1] * axis[:, 1] + axis[:, 2] * axis[:, 2]
    length = np.sqrt(length2)
    dr = r1 - r0
    # pieces no longer than the spacing, unless the network is so long that
    # there would be more than _EVD_PIECES of them
    piece_length = max(float(spacing), float(length.sum()) / _EVD_PIECES)
    k = np.maximum(1, np.ceil(length / piece_length)).astype(np.int64)
    owner = np.repeat(np.arange(length.size), k)
    position = np.arange(owner.size) - np.repeat(np.cumsum(k) - k, k)
    centre = start[owner] + ((position + 0.5) / k[owner])[:, None] * axis[owner]
    reach = length[owner] / (2.0 * k[owner]) + np.maximum(r0, r1)[owner]
    inner = np.minimum(r0, r1)[owner]
    tree = _cell_tree(centre, reach, inner, spacing)
    slack = _EVD_SLACK * (max(float(np.abs(grid).max()), float(np.abs(centre).max())) + float(reach.max())
                          + float(spacing))
    seed = owner[_seed_pieces(axes, spacing, centre)]
    values = _capsule_values(grid, start[seed], axis[seed], length2[seed], r0[seed], dr[seed])

    begin, size = 0, _EVD_CHUNK
    while begin < grid.shape[0]:
        n = min(size, grid.shape[0] - begin)
        survivors = _tissue_pass(grid[begin:begin + n], values[begin:begin + n], tree, centre, reach, inner, slack,
                                 _EVD_PAIRS if n > 1 else None)
        if survivors is None:
            # too many pairs: the same points again, half as many at a time
            size = n // 2
            continue
        local, piece, largest = survivors
        capsule = owner[piece]
        found = _capsule_values(grid[begin + local], start[capsule], axis[capsule], length2[capsule], r0[capsule],
                                dr[capsule])
        values[begin:begin + n] = np.minimum(values[begin:begin + n], _run_minimum(local, found, n))
        begin += n
        # a pass that held at most half the budget lets the next take twice the points
        if 2 * largest <= _EVD_PAIRS:
            size = min(_EVD_CHUNK, 2 * size)
    return values


def _tissue_pass(points, values, tree, centre, reach, inner, slack, budget):
    """
    One pass of the search of _tissue_values over the (n, 3) grid `points`,
    whose values so far are `values`.

    Returns:
        tuple or None: (local, piece, largest), the (point, piece) pairs that
        survive, the point numbered within the pass, and the most pairs the
        pass held at once; None, having stopped early, when it would have
        held more than `budget` (None for no limit).
    """
    n = points.shape[0]
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    bound = values.copy()
    # (point, cell) pairs, kept in order of point, so that per-point minima are runs
    local = np.arange(n)
    cell = np.zeros(n, dtype=np.int64)
    largest = n
    for cells in tree["levels"][::-1]:
        # along each axis, |p - c| less half the side is the gap to the
        # cell's box, and |p - c| plus half the side the distance to its
        # farther face
        cx, cy, cz = cells["centre"]
        ux, uy, uz = np.abs(px[local] - cx[cell]), np.abs(py[local] - cy[cell]), np.abs(pz[local] - cz[cell])
        half = cells["half"]
        gx, gy, gz = np.maximum(ux - half, 0.0), np.maximum(uy - half, 0.0), np.maximum(uz - half, 0.0)
        near = np.sqrt(gx * gx + gy * gy + gz * gz)
        ux, uy, uz = ux + half, uy + half, uz + half
        far = np.sqrt(ux * ux + uy * uy + uz * uz)
        bound = np.minimum(bound, _run_minimum(local, far - cells["inner"][cell], n))
        keep = near - cells["reach"][cell] <= bound[local] + slack
        local, cell = local[keep], cell[keep]
        low, high = cells["below"][cell], cells["below"][cell + 1]
        count = int((high - low).sum())
        if budget is not None and count > budget:
            return None
        largest = max(largest, count)
        pair, cell = _expand_ranges(low, high)
        local = local[pair]
    piece = tree["pieces"][cell]
    dx, dy, dz = px[local] - centre[piece, 0], py[local] - centre[piece, 1], pz[local] - centre[piece, 2]
    near = np.sqrt(dx * dx + dy * dy + dz * dz)
    bound = np.minimum(bound, _run_minimum(local, near - inner[piece], n))
    keep = near - reach[piece] <= bound[local] + slack
    return local[keep], piece[keep], largest


def _tissue_distance(nodes, points, r_max, spacing, volume, metadata, d_ref):
    """Distance from the tissue on a grid to the nearest vessel wall; see `describe`."""
    if points.shape[0] == 0:
        return None
    low, high, source = _tissue_domain(points, r_max, volume, metadata)
    h, counts, axes = _evd_grid(low, high, spacing)
    n_points = int(np.prod(counts))
    start, end, r0, r1 = _capsules(nodes)
    result = {"spacing_requested": spacing, "spacing": h, "raised": h != spacing, "shape": counts.tolist(),
              "points": n_points, "domain": {"min": low.tolist(), "max": high.tolist(), "source": source},
              "capsules": int(start.shape[0]), "inside_fraction": 0.0, "outside_points": n_points}
    statistics = ("mean", "median", "p90", "p99", "max")
    result.update({key: None for key in statistics})
    result["spacing_d"] = h / d_ref if d_ref is not None else None
    result.update({key + "_d": None for key in statistics})
    if start.shape[0] == 0:
        return result
    values = _tissue_values(axes, start, end, r0, r1, h)
    outside = values[values > 0.0]
    result.update({"inside_fraction": float(n_points - outside.size) / n_points,
                   "outside_points": int(outside.size)})
    if outside.size:
        median, p90, p99 = (float(v) for v in np.percentile(outside, [50.0, 90.0, 99.0]))
        found = {"mean": float(outside.mean()), "median": median, "p90": p90, "p99": p99,
                 "max": float(outside.max())}
        result.update(found)
        if d_ref is not None:
            result.update({key + "_d": value / d_ref for key, value in found.items()})
    return result


def _junction_neighbourhoods(nodes, geometry, canonical, r_max, margin):
    """
    Finds, for every point, the shared vertices of its polyline that it lies
    close to, with its distance to each: the junction neighbourhoods inside
    which two vessels may legitimately overlap.

    A vertex J is shared when columns of two or more polylines map to it. A
    point P of a polyline is recorded against every shared vertex J of that
    polyline whose arc-length distance along the polyline, arc(P, J), is
    below KIN_REACH (r_P + r_max + margin), which is every J the pair rule of
    _ClearanceScan._excused could excuse P at, whatever the other point's
    radius; the rule itself is applied pair by pair there.

    Returns:
        tuple: (keys, members, distances, offsets, counts) with keys the sorted
        int64 codes point * N + J of every recorded (point, J), members the J
        and distances the |P - J| of each key, and offsets/counts locating
        each point's run in those arrays.
    """
    n_columns = nodes.shape[1]
    n = geometry["columns"].size
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0),
             np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
    if n == 0:
        return empty
    # which polylines meet at which vertex, judged on every column of the
    # archive so that a polyline whose copy of a junction was dropped as a
    # duplicate still counts as passing through it
    incidence = geometry["incidence"]
    inc_polyline, inc_vertex = incidence // n_columns, incidence % n_columns
    inc_arc = geometry["incidence_arc"]
    polylines_at = np.bincount(inc_vertex, minlength=n_columns)
    shared = polylines_at[inc_vertex] >= 2
    if not shared.any():
        return empty
    inc_polyline, inc_vertex, inc_arc = inc_polyline[shared], inc_vertex[shared], inc_arc[shared]

    # every point of a polyline against every shared vertex of that polyline;
    # a polyline meets a handful of vertices, so this is a few passes over n
    owner, point = _expand_ranges(geometry["starts"][inc_polyline], geometry["ends"][inc_polyline])
    junction = inc_vertex[owner]
    reach = KIN_REACH * (geometry["radii"][point] + r_max + margin)
    # distance measured along the polyline, so that a vessel curling back
    # onto its own junction is not excused
    distance = np.abs(geometry["arc"][point] - inc_arc[owner])
    near = distance < reach
    codes = point[near] * n_columns + junction[near]
    keys, first = np.unique(codes, return_index=True)
    distances = distance[near][first]
    counts = np.bincount(keys // n_columns, minlength=n)
    return keys, keys % n_columns, distances, np.cumsum(counts) - counts, counts


def _radius_classes(radii):
    """
    Assigns every point to a class an octave of radius wide, 0 holding the
    largest radii.

    Returns:
        tuple: (level, n_classes) with level an int64 array of class numbers.
    """
    r_max = float(radii.max())
    positive = radii[radii > 0.0]
    if r_max <= 0.0 or positive.size == 0:
        return np.zeros(radii.size, dtype=np.int64), 1
    octaves = int(math.ceil(math.log2(r_max / positive.min()))) + 1
    n_classes = max(1, min(_MAX_RADIUS_CLASSES, octaves))
    with np.errstate(divide="ignore"):
        level = np.floor(np.log2(r_max / np.maximum(radii, np.finfo(float).tiny)))
    return np.clip(level, 0, n_classes - 1).astype(np.int64), n_classes


class _ClearanceScan:
    """
    One network's data for the pairwise clearance search, and the search
    itself at any reach.
    """

    def __init__(self, nodes, geometry, canonical, vertex_diameter, margin, index):
        self.geometry = geometry
        self.margin = margin
        self.index_kind = index
        self.n_columns = nodes.shape[1]
        self.points = geometry["points"]
        self.radii = geometry["radii"]
        self.polyline = geometry["polyline"]
        # a polyline that returns to its first vertex is a ring: arc-length
        # separation along it is measured the shorter way round
        self.closed = geometry["closed"]
        r_max = float(self.radii.max()) if self.radii.size else 0.0
        self.keys, self.members, self.distances, self.offsets, self.counts = _junction_neighbourhoods(
            nodes, geometry, canonical, r_max, margin)
        self.level, self.n_classes = _radius_classes(self.radii)

    def _excused(self, a, b):
        """Boolean mask of the candidate pairs (a, b) that the two exclusion rules skip."""
        same = self.polyline[a] == self.polyline[b]
        separation = np.abs(self.geometry["arc"][a] - self.geometry["arc"][b])
        ring = same & self.closed[self.polyline[a]]
        if ring.any():
            around = self.geometry["length"][self.polyline[a[ring]]] - separation[ring]
            separation[ring] = np.minimum(separation[ring], around)
        skip = same & (separation <= 2.0 * (self.radii[a] + self.radii[b] + self.margin))

        # different polylines: excused when, at some vertex J shared by the two
        # polylines, arc(a, J) + arc(b, J) < KIN_REACH (r_a + r_b + margin);
        # a's recorded vertices are few, so each in turn is looked up among b's
        cross = np.flatnonzero(~same)
        slot = 0
        while cross.size:
            has = self.counts[a[cross]] > slot
            cross = cross[has]
            if cross.size == 0:
                break
            at = self.offsets[a[cross]] + slot
            junction = self.members[at]
            position = _position_in_sorted(self.keys, b[cross] * self.n_columns + junction)
            found = position >= 0
            together = self.distances[at[found]] + self.distances[position[found]]
            reach = KIN_REACH * (self.radii[a[cross[found]]] + self.radii[b[cross[found]]] + self.margin)
            skip[cross[found][together < reach]] = True
            slot += 1
        return skip

    def run(self, reach):
        """
        Checks every pair with clearance below `reach` that the rules do not
        skip.

        Returns:
            dict: "gathered" pairs found before exclusion, "checked" pairs
            after it, "violations" checked pairs with clearance below the
            margin, and "best" the smallest clearance as (clearance, a, b)
            or None; ties go to the pair with the lowest columns.
        """
        stats = {"gathered": 0, "checked": 0, "violations": 0, "best": None}
        # A pair is found from its smaller point: the points of each class are
        # indexed and queried by every point of that class or a smaller one.
        # The reach of such a query is about the class's largest radius, so
        # it covers a few cells of an index whose cells are that size, while
        # searching the other way round would sweep an arteriole's radius
        # through the dense sampling of the capillaries around it.
        for j in range(self.n_classes):
            stored = np.flatnonzero(self.level == j)
            if stored.size == 0:
                continue
            r_class = float(self.radii[stored].max())
            # the class is stored once and queried many times, so its search
            # structure is built straight away rather than after a buffer fills
            index = make_index(self.index_kind, cell_size=max(2.0 * r_class + reach, 1e-9), rebuild_every=0)
            index.add(self.points[stored], self.radii[stored], stored)
            queries = np.flatnonzero(self.level >= j)
            for start in range(0, queries.size, _CHUNK):
                q = queries[start:start + _CHUNK]
                qi, si = index.query(self.points[q], self.radii[q], reach)
                a, b = q[qi], index.tags[si]
                # a pair inside the class is met from both ends; keep one
                keep = (self.level[a] != j) | (a < b)
                self._account(np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep]), stats)
        return stats

    def _account(self, a, b, stats):
        """Applies the exclusions to candidate pairs (a < b) and folds the rest into `stats`."""
        stats["gathered"] += a.size
        if a.size == 0:
            return
        keep = ~self._excused(a, b)
        a, b = a[keep], b[keep]
        if a.size == 0:
            return
        clearance = (np.linalg.norm(self.points[a] - self.points[b], axis=1)
                     - self.radii[a] - self.radii[b])
        stats["checked"] += a.size
        stats["violations"] += int(np.count_nonzero(clearance < self.margin))
        low = np.flatnonzero(clearance == clearance.min())
        low = low[np.lexsort((b[low], a[low]))][0]
        best = stats["best"]
        candidate = (float(clearance[low]), int(a[low]), int(b[low]))
        if best is None or candidate < best:
            stats["best"] = candidate


def clearance(nodes, margin=0.0, tol=DEFAULT_TOL, canonical=None, index="auto"):
    """
    Measures how close the vessels of a network come to each other.

    The clearance of two centreline points i and j is |p_i - p_j| - r_i - r_j,
    the gap between the surfaces of the two vessels there; negative when the
    vessels overlap. It is measured over every pair of points of the archive
    except the pairs that overlap by construction:

    (a) two points of the same polyline whose arc-length separation along it
        is at most 2 (r_i + r_j + margin), consecutive samples of one vessel
        being always inside each other's radius; for a polyline that returns
        to its own first vertex the separation is measured the shorter way
        round the ring;
    (b) two points of different polylines that share a vertex J (a column of
        each polyline maps to J) when arc(p_i, J) + arc(p_j, J) is below
        collisions.KIN_REACH (r_i + r_j + margin), arc(p, J) the arc length
        along p's polyline from p to J: the parent's last samples and the
        daughters' first sit inside each other at every junction, out to a
        distance set by their radii and the bifurcation angles (see
        collisions.py for the bound). Arc length is used rather than straight
        distance so that a vessel curling back onto its own junction from far
        along its path is measured, not excused.

    These are the rules the generator's collision avoidance applies while it
    grows, so a network grown with a margin should show no violation at that
    margin, and the count here is the check. Rule (a) also bounds the minimum
    from above: along a straight vessel the first pair outside the window has
    clearance r_i + r_j + 2 margin, so the minimum reported for a network
    rarely exceeds the diameter of its thinnest vessel plus twice the margin,
    and a value near that is a vessel measured against itself rather than a
    close approach between two vessels.

    Pairs are found with spatial index queries, which return every pair
    whose clearance is below the query's reach. The search starts at a reach
    equal to the margin, so every violation is found, and when that checks no
    pair it widens the reach in doublings of the smallest radius until a pair
    is checked, the reach exceeds the network's diagonal or a stage gathers
    more pairs than it is worth chasing. Within the final reach the check is
    exhaustive, so the minimum over the checked pairs is the minimum over all
    pairs the rules admit whenever a pair was checked at all.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        margin (float): clearance the network is required to keep; a pair
            below it is a violation.
        tol (float): merging distance for the vertices (graph.canonical_columns).
        canonical (ndarray): (N,) result of graph.canonical_columns for
            `nodes`, computed here when omitted.
        index (str): spatial index kind for spatial.make_index: "grid",
            "kdtree" or "auto".

    Returns:
        dict:
            "min": the smallest clearance over the checked pairs, or None when
                none was checked;
            "min_columns": the two column indices of that pair, or None;
            "min_is_lower_bound": True when no pair was checked, meaning that
                every pair the rules admit has clearance of at least
                search_reach_um;
            "search_reach_um": the clearance below which the search was
                exhaustive;
            "violations": checked pairs with clearance below the margin;
            "pairs_checked": pairs found within the reach that the rules did
                not skip;
            "margin": the margin used.
    """
    nodes = _as_nodes(nodes)
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"margin must be a non-negative number, got {margin!r}")
    result = {"min": None, "min_columns": None, "min_is_lower_bound": True,
              "search_reach_um": margin, "violations": 0, "pairs_checked": 0, "margin": margin}
    geometry = _polyline_geometry(nodes)
    if geometry["columns"].size < 2:
        return result
    if canonical is None:
        canonical = graph.canonical_columns(nodes, tol)
    geometry = _representatives(geometry, canonical)
    if geometry["columns"].size < 2:
        return result
    vertex_diameter = graph.vertex_diameter(nodes, canonical)
    scan = _ClearanceScan(nodes, geometry, canonical, vertex_diameter, margin, index)

    positive = geometry["radii"][geometry["radii"] > 0.0]
    step = float(positive.min()) if positive.size else 1.0
    budget = max(_STAGE_PAIRS_FLOOR, _STAGE_PAIRS_PER_POINT * geometry["columns"].size)
    diagonal = float(np.linalg.norm(geometry["points"].max(axis=0) - geometry["points"].min(axis=0)))
    extra = 0.0
    while True:
        reach = margin + extra
        stage = scan.run(reach)
        if stage["checked"] or extra > diagonal or stage["gathered"] > budget:
            break
        extra = step if extra == 0.0 else 2.0 * extra

    result.update({"search_reach_um": reach, "violations": stage["violations"],
                   "pairs_checked": stage["checked"]})
    if stage["best"] is not None:
        smallest, a, b = stage["best"]
        columns = geometry["columns"]
        result.update({"min": smallest, "min_columns": [int(columns[a]), int(columns[b])],
                       "min_is_lower_bound": False})
    return result


def describe(nodes, edges=None, metadata=None, volume=None, margin=None, tol=DEFAULT_TOL, tree=None,
             frame=None, d_ref=None, class_bound=DEFAULT_CLASS_BOUND, evd_spacing=None):
    """
    Measures the geometry and topology of a centreline network.

    The definitions below are the reference for every quantity: a skeleton
    extracted from an image is described by the same numbers when they are
    computed by these rules, so that synthetic and real vasculature can be
    compared term by term.

    Graph. Columns of `nodes` within `tol` of one another (transitively) are
    one vertex, named by its lowest column (graph.canonical_columns);
    consecutive columns of a polyline give an edge between their vertices,
    with zero-length and repeated edges dropped (graph.edges_from_nodes).
    `edges` supplied by the caller, as an archive stores them, are used as
    they are and must be expressed in the vertices that `tol` produces. A
    vertex's position is its column's and its diameter is the largest
    diameter over the columns it gathers (graph.vertex_diameter). The degree
    of a vertex is the number of edges at it.

    Lengths. The length of an edge is the Euclidean distance between its two
    vertices, in micrometres; "total_length_mm" is the sum over all edges
    divided by 1000.

    "diameter_um". The diameter of an edge is the mean of its two vertices'
    diameters. "p50", "p90" and "p99" are length-weighted percentiles: the
    smallest edge diameter d such that the edges of diameter at most d hold
    at least 50 %, 90 % or 99 % of the total length. "min" and "max" are
    over vertex diameters.

    "arc_chord". A segment is a path of edges between two vertices of degree
    other than two, or a whole component of degree-two vertices
    (graph.segments). Its arc is the sum of its edge lengths and its chord
    the distance between its end vertices; the ratio is arc / chord. A
    segment whose ends are the same vertex has no chord and is left out,
    counted in "n_closed"; a segment of a single edge has ratio 1 and is
    included. "mean", "median", "p90" and "max" are over the ratios of the
    open segments, "median" and "p90" interpolating linearly between order
    statistics (numpy.percentile's default); "n_segments" counts them.

    "curvature_per_um". For each pair of consecutive edges (a, b) along a
    segment, with unit tangents t_a, t_b and lengths l_a, l_b, the turning
    angle is angle(t_a, t_b) in radians (0 for collinear edges) and the local
    curvature is kappa = angle / ((l_a + l_b) / 2). "mean_abs" is the mean
    of kappa weighted by (l_a + l_b) / 2, which is the sum of the turning
    angles divided by the sum of those half-lengths, in radians per
    micrometre; "n_pairs" counts the pairs. In a closed segment the vertex
    where the path returns to its start is an interior vertex like any
    other, so the pair of its last and first edges is counted too.

    "tips". "count" is the number of vertices of degree one, a tree's root
    among them; "per_mm" is count / total_length_mm, None without length.

    "degree_histogram". {"d": number of vertices of degree d} for every
    degree present, keys as strings.

    "components" and "cycles". The Betti numbers b0 and b1 = E - V + b0 of
    the graph (graph.betti), where "vertices" is V and "edges" E. An
    isolated vertex is a component.

    "orientation". Each edge has a unit tangent t_e, of either sign, and a
    length l_e. C = sum l_e t_e t_e^T / sum l_e is the length-weighted
    tangent covariance; "eigenvalues" are its eigenvalues in descending
    order (they sum to one), "principal_axis" the unit eigenvector of the
    largest with its largest component made positive, and
    "fractional_anisotropy" is sqrt(3/2) sqrt(sum (lambda_i - 1/3)^2) /
    sqrt(sum lambda_i^2): 0 when length is spread equally over three
    orthogonal directions, 1 for a single direction.
    "polar_histogram_deg" is the fraction of length in each of nine bins of
    ten degrees of the angle acos(|t_z|) between the tangent and the z axis,
    0..90 degrees since tangents are unoriented (bin 0 holds 0..10 and the
    last bin includes 90). "azimuth_histogram_deg" is the fraction of length
    in each of six bins of 30 degrees of the azimuth of the tangent's xy
    projection, atan2(t_y, t_x) after flipping the tangent so that t_y >= 0
    (and t_x >= 0 when t_y = 0), running from 0 along x through 90 along y
    to 180. Each edge is weighted by the length of its xy projection, so an
    edge parallel to z contributes nothing and the fractions sum to one over
    the in-plane length. Both histograms are lists in bin order; every entry
    is None without edges (the azimuth one also without in-plane length).

    "length_density_mm_per_mm3". total_length_mm / volume_mm3. The volume
    is taken, in order of preference, from `volume` (three extents in
    micrometres), from metadata["growth_box_um"] (three extents), from
    metadata["volume"] scaled by metadata["voxel_size"] on every axis when
    metadata["fit"] is "voxel_size", and otherwise from the axis-aligned
    bounding box of the points padded by the largest radius on every side.
    "volume_mm3" and "volume_source" ("argument", "growth_box_um",
    "voxel_size" or "bounding_box") record the choice, and
    "bounding_box_um" the unpadded box as {"min": [x, y, z], "max": ...}.

    "clearance_um". The result of `clearance` at the margin, which defaults
    to metadata["collision_margin"] and then to 0; see that function for the
    definition and the exclusion rules.

    "events" is a copy of metadata["events"], {} when absent, and "flags"
    the entries of FLAG_KEYS present in the metadata.

    Counts. "points" is the number of finite columns, "polylines" the number
    of runs of them, "vertices" and "edges" as above. "per_tree" is None
    unless `tree` is given, and then maps each label found at a finite
    column to {"points", "polylines", "tips", "length_mm"}: a tip belongs
    to the tree of its vertex's column, a polyline to the tree of its first
    column, and the length is the sum of the arc lengths of the tree's
    polylines, which equals the graph's length unless a polyline retraces
    another.

    Frame. A network grown along a preferred direction or within a plane
    records the frame it was steered by in metadata["frame"], which `frame`
    overrides: its kind ("none", "axis" or "plane"), the unit axis a or
    plane normal n, the heading g and perpendicular p of the first tree's
    turtle frame (the generator's defaults are [0, 1, 0] and [0, 0, 1]) and
    the origin o, the centre of the growth box or else the first root, in
    the coordinates of `nodes`. "frame" is a copy of the record, or None.
    Every quantity below that refers to a or n is None without a frame and
    for a frame of kind "none". For them each edge has an oriented unit
    tangent t_e = (x_hi - x_lo) / |x_hi - x_lo| from the vertex of its lower
    column to the vertex of its higher: columns run from root to tip along
    a polyline and a junction is named by its parent's column, so for a
    network the generator wrote the tangent points the way the vessel grew.
    <.> is the mean over the edges of a class weighted by edge length l_e;
    a zero-length edge carries no weight. Lengths are in the archive's unit
    (micrometres for the generator's archives) and a quantity suffixed _d
    is the same in units of d_ref.

    "classes". An edge of diameter d_e (as in "diameter_um") is
    capillary-class when d_e < class_bound * d_ref and larger-class
    otherwise; an edge whose diameter is unknown is in neither class. d_ref
    is the `d_ref` argument, else metadata["d_min"], else
    metadata["grow_kwargs"]["d_min"], whichever is the first finite positive
    number; {"bound", "d_ref", "d_ref_source"} records the choice, the
    source being "argument", "d_min", "grow_kwargs" or None. Without a d_ref
    every quantity that depends on the classes is None: the thinnest vertex
    is never taken as the reference, since its diameter says nothing about
    where the capillaries of a network end.

    "frame_orientation". {"kind", "all", "capillary", "larger"}, the class
    entries None without d_ref and for a class without length. For an axis
    frame, with u = t_e . a: "S" = <(3 u^2 - 1) / 2>, the nematic order
    parameter, 1 for edges along a, -1/2 for edges perpendicular to it and
    0 for isotropic ones; "mean_abs_cos" = <|u|>; "crossing_ratio" =
    1 / <|u|>, the length of vessel per unit of extent covered along a
    (None when <|u|> is 0); "mean_angle_deg" = <acos(min(|u|, 1))> in
    degrees; "within_20_deg" and "within_45_deg" the fraction of length at
    an angle of at most 20 or 45 degrees to a; "watson_K" the concentration
    K of the Watson distribution, density proportional to exp(K u^2) on the
    sphere, that has the measured <u^2>, W(K) = int_0^1 u^2 e^(K u^2) du /
    int_0^1 e^(K u^2) du solved by bisection on [-200, 200] (W by a 256
    point Gauss-Legendre rule), None when <u^2> lies outside
    [W(-200), W(200)]; "fisher_axial_K" the K solving coth K - 1 / K = <|u|>
    by bisection on (0, 200], None when <|u|> lies outside the range of
    that function there, 0 to about 0.995 (an isotropic <|u|> of 1/2
    reads about 1.80: it is the polar Fisher law's <cos theta>, which the
    Fisher-axial law's <|cos theta|> approaches only as K grows); "length"
    the class's total edge length and "length_d" the same in d_ref;
    "fisher_axial_K_exact" the K of the Fisher-axial law itself, density
    proportional to exp(K |u|), whose <|u|> = 1 / (1 - e^-K) - 1 / K equals
    the measured one, solved by bisection on [-200, 200] (0 for isotropic
    tangents, negative when they gather across a), None when <|u|> lies
    outside the range of that function there, about 0.005 to 0.995. For a
    plane frame, with
    v = t_e . n: "in_plane_fraction" = <1 - v^2>, "S_n" = <(3 v^2 - 1) / 2>,
    and "length" and "length_d" as above.

    "polar_order". For an axis frame whose sense is "polar", when `tree` is
    given: p = sum l_e s_k (t_e . a) / sum l_e over the edges whose higher
    column carries a tree label k that indexes frame["tree_senses"], s_k
    being that tree's recorded sense, +1 or -1; 1 when every tree runs
    along a in its own sense and -1 when every tree runs against it. An
    edge whose label indexes no sense (a bridge, label 2 in a generated
    network, or a separator) is left out of both sums. {"all", "capillary",
    "larger"}, the class entries None without d_ref and for a class without
    labelled length; None altogether otherwise.

    "orientation_by_class". {"all", "capillary", "larger"}, each the
    "orientation" entry computed over the class's edges with "S_max" =
    (3 lambda_1 - 1) / 2, the order parameter about the principal axis, and
    "planarity" = 1 - 3 lambda_3, 0 for an isotropic covariance and 1 when
    no length leaves a plane, both None without edges. None without d_ref.

    "calibre_shares". {"length", "volume"}, each holding under "below_1.5",
    "below_2" and "below_3" the share of the total edge length, or of the
    total edge volume sum (pi / 4) d_e^2 l_e, held by the edges with
    d_e < 1.5, 2 or 3 d_ref, over the edges of known diameter; None when
    the total is 0. None without d_ref.

    "segments_by_class". {"capillary", "larger"} over the segments of
    "arc_chord": a segment's length is the sum of its edge lengths and its
    diameter the length-weighted mean of its edge diameters, and it is
    capillary when that diameter is below class_bound * d_ref; a segment of
    zero length or unknown diameter is left out. Each entry holds "count",
    the "median", "p10" and "p90" of the lengths (interpolated linearly as
    for "arc_chord"), "cv" the population standard deviation over the mean
    (None below two segments) and "median_d", "p10_d" and "p90_d" in d_ref;
    every statistic is None when the count is 0. None without d_ref.

    "transverse_spacing". For an axis frame with d_ref, over the
    capillary-class edges of positive length; None otherwise, when there is
    no such edge or when the edges have no extent along a. With s = x . a
    over their end points and [s_min, s_max] its extent, nine planes
    perpendicular to a are placed at c_k = s_min + (0.1 + 0.1 k)
    (s_max - s_min), k = 0..8, through the central 80 % of the extent. An
    edge crosses plane k when (s_lo < c_k) != (s_hi < c_k), at the point
    interpolated linearly along it. In-plane coordinates run along
    e1 = unit(g - (g . a) a), or unit(p - (p . a) a) when g lies along a,
    and e2 = a x e1, so that nothing depends on the lab axes. On each plane
    with at least two crossings the distance from every crossing to its
    nearest other crossing is taken; pooled over the planes these give
    "count", "median", "mean", "p10" and "p90", with "median_d", "mean_d",
    "p10_d" and "p90_d" in d_ref, None when the count is 0.
    "crossing_density" is the number of crossings on the nine planes, all
    of them, over the summed area of the planes' sections through the
    reference box, per square unit, and "crossing_density_d" the same per
    d_ref^2. The box is `volume` ("area_source" "argument") or else
    metadata["growth_box_um"] ("growth_box_um"), three extents along
    (g x p, g, p) centred on o, each section being the polygon where the
    plane cuts it. grow_network builds its box along the x, y and z axes of
    `nodes`, which are (g x p, g, p) for the generator's default turtle
    frame and for a network moved from it by frames.transform_frame, not
    for one grown from another direction. Without either, the box is
    replaced by the rectangle in (e1, e2) bounding the end points of the
    capillary edges, the same on every plane ("capillary_bounding_box"), and
    the density is None when that rectangle has no area. "planes" is 9.

    Topology. The keys below are measured on the segments of "arc_chord"
    and need no frame; only their entries in d_ref need d_ref.

    "junctions". Over the J vertices of degree three or more: "count" J;
    "degree_3", "degree_4" and "degree_5_plus" the fractions of them of
    degree 3, 4 and at least 5; "mean_degree" their mean degree; and
    "segments_per_junction" the number of segments, open, closed and
    pendant alike, divided by J. Every entry but "count" is None when J is 0.

    "branch_angles_deg". At a vertex v of degree three, each of the three
    segment ends there gives an arm: the unit vector from v to the point of
    the segment at arc length s = min(2 d_v, L / 2) from v, d_v the vertex's
    diameter and L the segment's arc, interpolated linearly along its edges
    (a closed segment that starts and ends at v gives an arm from each end).
    The angle between two arms u and w is atan2(|u x w|, u . w) in degrees,
    and each vertex's three angles are sorted. "min", "median" and "max" are
    the medians over the vertices of the smallest, the middle and the
    largest, None when no vertex is measured; "count" counts the vertices
    measured and "skipped" those left out because an arm has no length or
    the diameter is unknown.

    "loops". {"by_segment", "by_node"}, measured on the segment multigraph:
    its vertices are the ends of the segments and its edges the segments,
    each weighted by its arc, a closed segment being a self-loop and two
    segments with the same two ends a pair of parallel edges. Every loop of
    the network lies in the multigraph's 2-core, what remains once vertices
    of degree one have been removed with their segment until none is left,
    and so do the shortest paths between its vertices; the searches run on
    the core, and a segment or junction outside it has no loop. Of the S
    segments, every k-th in the order of graph.segments is sampled, k =
    ceil(S / 2000) and at least 1, and the same for the J junctions in
    column order, so that at most 2000 of each are searched from. A loop's
    size is its number of segments; "median", "mean" and "histogram"
    ({"n": count}) are over the sizes of the sampled items' loops, the
    median and mean None when there is none; "every" is k, "sampled" the
    number sampled, "found" the number with a loop and "none_fraction" the
    share without one (None when nothing is sampled).
    "by_segment": the shortest loop through a segment has size 1 when the
    segment is closed, 2 when another segment joins the same two ends, and
    otherwise 1 + the number of segments on a shortest path between its
    ends that does not use it. The path is found by a breadth-first search
    from both ends at once, a whole level at a time from the end whose
    frontier is smaller (the segment's first end on a tie), neighbours in
    column order. Only paths of at most 16 segments are found, so loops of
    at most 17, and a search that would reach more than 4096 vertices of
    the core, its two ends included, before its two sides meet finds none.
    The entry also holds "segments" S, "cycles" the "cycles" b1 of the
    graph, "cycles_per_length" b1 over the total edge length (per unit
    length of the archive) and "cycles_per_length_d" b1 d_ref over the total
    edge length.
    "by_node": from a junction s, a shortest-path search by arc length
    (Dijkstra's, ties going to fewer segments and then to the lower column)
    settles the vertices of the core in order of their distance d from s,
    labelling each with the segment at s that its path leaves by; a vertex
    whose path from s has 16 segments is settled but not searched beyond,
    and at most 4096 vertices are settled. The shortest loop through s is
    the least, by length and then by number of segments, of d(u) + w + d(v)
    over the segments of arc w joining two settled vertices u and v of
    different labels (s's own label differing from every other), other than
    the segment by which u or v was reached, and of w over the closed
    segments at s; a loop through a junction can thus have up to 33
    segments. The entry also holds "junctions" J and the "length_median"
    and "length_mean" of the loops' lengths, with "length_median_d" and
    "length_mean_d" in d_ref.

    "segment_diameter_variation". For each segment, over the diameters of
    its vertices, the first vertex of a closed segment counted once: (max -
    min) / mean, leaving out vertices of unknown diameter and the segment's
    end vertices of degree three or more, whose diameter is that of the
    widest vessel at the junction, so that the step from a parent to its
    daughter does not count as variation along the daughter. A segment with
    fewer than two such vertices, or with a mean of 0, has no value.
    {"all", "capillary", "larger"} over every segment and over those of each
    class of "segments_by_class", each {"count" the segments with a value,
    "excluded" those without, and the "median" and "p90" of the values,
    interpolated linearly as for "arc_chord" and None when the count is 0};
    the class entries are None without d_ref.

    "tissue_distance". None unless `evd_spacing` is given, and for a network
    without points. The tissue is sampled on a grid over the reference box
    of "length_density_mm_per_mm3", placed as its source places it: `volume`
    centred on the centre c of "bounding_box_um"; the voxel_size field as
    the voxeliser places it (computeVoxel.fit_to_volume), voxel i of the n
    along an axis centred at c + (i - n / 2) s for a voxel size s, so that
    the field runs from c - (n + 1) s / 2 to c + (n - 1) s / 2 and a grid of
    spacing s has its points at the voxel centres; metadata["growth_box_um"]
    from 0 to its extents in the coordinates of `nodes`; and the padded
    bounding box where it lies. Along axis i, of extent E_i, the grid has
    n_i = max(1, floor(E_i / h + 1e-9)) points, h apart and centred in the
    box, h being `evd_spacing`; when that would make more than 64^3 points
    in all, h is raised to the least E_i / m (m a whole number) at which
    there are no more. At each grid point p, f(p) is the least over the
    capsules of |p - c(t)| - (r0 + t (r1 - r0)). The capsules are those the
    voxeliser draws (computeVoxel.rasterise_segments), one between each two
    consecutive finite columns c0 and c1 of a polyline that are not the
    same point, with radii r0 and r1 half their diameters,
    c(t) = c0 + t (c1 - c0) and t the projection of p onto the line through
    them clamped to [0, 1]; capsules outside the box count. Where the
    diameters are finite and not negative, a point with f(p) <= 0 lies
    inside a vessel exactly when the voxeliser's capsule test
    (computeVoxel.rasterise_capsule) puts it inside. A missing or negative
    diameter counts here as zero, which is describe's own choice: the
    voxeliser draws nothing of a capsule with a missing diameter and takes
    a negative radius by its size. The minimum is exact, the search
    leaving out only capsules that bounds show cannot hold it, and nothing
    is drawn at random. {"spacing_requested", "spacing" h, "raised" whether
    h was raised, "shape" the number of points along each axis, "points"
    their total, "domain" {"min", "max", "source"} the box and the source's
    name as in "volume_source", "capsules" their number, "inside_fraction"
    the share of points inside, "outside_points" the number outside, "mean",
    "median", "p90", "p99" and "max" of f over the points outside
    (interpolated linearly as for "arc_chord"), and "spacing_d", "mean_d",
    "median_d", "p90_d", "p99_d" and "max_d" the same in d_ref}; the
    statistics of f are None when no point lies outside or there is no
    capsule (every point then being outside), and those in d_ref without
    d_ref.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        edges (ndarray or None): (2, E) column indices of the graph's edges,
            rebuilt from `nodes` when None.
        metadata (dict, str or None): the archive's record, as a dict, a
            JSON string or the 0-d array numpy.load returns.
        volume (sequence or None): three extents in micrometres to measure
            the length density against.
        margin (float or None): collision margin for the clearance check;
            None takes the metadata's collision_margin when the metadata
            says avoid_collisions was on, else 0.
        tol (float): merging distance for the vertices, in micrometres.
        tree (ndarray or None): (N,) per-column tree label, -1 at separators.
        frame (dict or None): the frame record; None takes
            metadata["frame"] when there is one.
        d_ref (float or None): reference diameter for the calibre classes,
            in the units of `nodes`; None takes the metadata's d_min.
        class_bound (float): an edge thinner than class_bound * d_ref is
            capillary-class.
        evd_spacing (float or None): spacing of the grid "tissue_distance"
            is measured on, in the units of `nodes`; None leaves it out.

    Returns:
        dict: JSON-serialisable; numbers are Python int or float, undefined
        quantities None.
    """
    nodes = _as_nodes(nodes)
    n_columns = nodes.shape[1]
    metadata = _as_metadata(metadata)
    if margin is None:
        # a stored margin means something only if the generator enforced it
        margin = (metadata.get("collision_margin") or 0.0) if metadata.get("avoid_collisions") else 0.0
    canonical = graph.canonical_columns(nodes, tol)
    if edges is None:
        edges = graph.edges_from_nodes(nodes, tol, canonical)
    else:
        edges = _as_edges(edges, n_columns)
    if tree is not None:
        tree = np.asarray(tree)
        if tree.shape != (n_columns,):
            raise ValueError(f"tree must hold one label per column, {n_columns}, got {tree.shape}")
    if frame is None:
        frame = metadata.get("frame")
    frame_vector = _frame_vector(frame)
    d_ref, d_ref_source = _reference_diameter(d_ref, metadata)
    if not _positive_number(class_bound):
        raise ValueError(f"class_bound must be a positive number, got {class_bound!r}")
    class_bound = float(class_bound)
    if evd_spacing is not None and not _positive_number(evd_spacing):
        raise ValueError(f"evd_spacing must be a positive number, got {evd_spacing!r}")

    degree = graph.degree(edges, n_columns)
    vertices = np.flatnonzero(canonical == np.arange(n_columns))
    vertex_diameter = graph.vertex_diameter(nodes, canonical)
    topology = graph.betti(edges, canonical)
    geometry = _polyline_geometry(nodes)
    length, tangent = _edge_geometry(nodes, edges)
    total_length_mm = float(length.sum()) / 1000.0
    paths = graph.segments(nodes, edges, canonical)
    arc_chord, curvature = _segment_stats(nodes, paths)

    tip_vertices = vertices[degree[vertices] == 1]
    tips = {"count": int(tip_vertices.size),
            "per_mm": tip_vertices.size / total_length_mm if total_length_mm > 0.0 else None}
    histogram = np.bincount(degree[vertices]) if vertices.size else np.zeros(0, dtype=np.int64)
    degree_histogram = {str(d): int(count) for d, count in enumerate(histogram) if count}

    points = geometry["points"]
    r_max = float(geometry["radii"].max()) if points.shape[0] else 0.0
    volume_um3, volume_source = _volume_um3(points, r_max, volume, metadata)
    volume_mm3 = None if volume_um3 is None else volume_um3 / 1e9
    density = total_length_mm / volume_mm3 if volume_mm3 else None
    bounding_box = None
    if points.shape[0]:
        bounding_box = {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}

    per_tree = None
    if tree is not None:
        per_tree = {}
        column_tree = tree[geometry["columns"]]
        polyline_tree = tree[geometry["columns"][geometry["starts"]]]
        for label in np.unique(column_tree):
            of_polyline = polyline_tree == label
            per_tree[str(int(label))] = {
                "points": int(np.count_nonzero(column_tree == label)),
                "polylines": int(np.count_nonzero(of_polyline)),
                "tips": int(np.count_nonzero(tree[tip_vertices] == label)),
                "length_mm": float(geometry["length"][of_polyline].sum()) / 1000.0}

    # Frame-relative and class descriptors. The tangent of an edge runs from
    # its lower column to its higher one, the direction of growth for a
    # network the generator wrote; _edge_geometry's runs the way the edge is
    # stored, which is the same for edges graph.edges_from_nodes built.
    lower, higher = np.minimum(edges[0], edges[1]), np.maximum(edges[0], edges[1])
    oriented = np.where((edges[0] <= edges[1])[:, None], tangent, -tangent)
    edge_diameter = (vertex_diameter[edges[0]] + vertex_diameter[edges[1]]) / 2.0
    classes = {"all": np.ones(edges.shape[1], dtype=bool), "capillary": None, "larger": None}
    if d_ref is not None:
        known = np.isfinite(edge_diameter)
        classes["capillary"] = known & (edge_diameter < class_bound * d_ref)
        classes["larger"] = known & ~classes["capillary"]
    frame_orientation = polar_order = transverse_spacing = None
    if frame_vector is not None:
        frame_orientation = _frame_orientation(frame_vector, oriented, length, classes, d_ref)
        kind, direction = frame_vector
        if kind == "axis" and frame.get("sense") == "polar" and tree is not None:
            polar_order = _polar_order(oriented @ direction, length, tree[higher].astype(np.int64),
                                       frame.get("tree_senses"), classes)
        if kind == "axis" and d_ref is not None:
            capillary = classes["capillary"] & (length > 0.0)
            transverse_spacing = _transverse_spacing(nodes[:3].T, lower[capillary], higher[capillary], direction,
                                                     frame, d_ref, volume, metadata)
    orientation_by_class = calibre_shares = segments_by_class = None
    arc, mean_diameter = _segment_calibres(nodes, paths, vertex_diameter)
    if d_ref is not None:
        orientation_by_class = {name: _orientation_shape(oriented[mask], length[mask])
                                for name, mask in classes.items()}
        calibre_shares = _calibre_shares(edge_diameter, length, d_ref)
        segments_by_class = _segments_by_class(arc, mean_diameter, class_bound * d_ref, d_ref)

    # Topology descriptors, on the segments of "arc_chord", and the opt-in
    # distance from the tissue to the vessels
    layout = _segment_layout(nodes, paths)
    junctions = _junctions(degree, vertices, len(paths))
    branch_angles = _branch_angles(nodes, layout, degree, vertex_diameter)
    loops = _loops(layout, arc, degree, vertices, topology["cycles"], float(length.sum()), d_ref)
    diameter_variation = _diameter_variation(layout, vertex_diameter, degree, arc, mean_diameter,
                                             None if d_ref is None else class_bound * d_ref)
    tissue_distance = None
    if evd_spacing is not None:
        tissue_distance = _tissue_distance(nodes, points, r_max, float(evd_spacing), volume, metadata, d_ref)

    result = {
        "units": metadata.get("units", "um"),
        "points": int(geometry["columns"].size),
        "polylines": int(geometry["starts"].size),
        "vertices": topology["vertices"],
        "edges": topology["edges"],
        "components": topology["components"],
        "cycles": topology["cycles"],
        "total_length_mm": total_length_mm,
        "diameter_um": _diameter_stats(vertex_diameter, vertices, edges, length),
        "arc_chord": arc_chord,
        "curvature_per_um": curvature,
        "tips": tips,
        "degree_histogram": degree_histogram,
        "orientation": _orientation(tangent, length),
        "length_density_mm_per_mm3": density,
        "volume_mm3": volume_mm3,
        "volume_source": volume_source,
        "bounding_box_um": bounding_box,
        "clearance_um": clearance(nodes, margin=margin, tol=tol, canonical=canonical),
        "events": dict(metadata.get("events") or {}),
        "flags": {key: metadata[key] for key in FLAG_KEYS if key in metadata},
        "per_tree": per_tree,
        "frame": copy.deepcopy(frame),
        "classes": {"bound": class_bound, "d_ref": d_ref, "d_ref_source": d_ref_source},
        "frame_orientation": frame_orientation,
        "polar_order": polar_order,
        "orientation_by_class": orientation_by_class,
        "calibre_shares": calibre_shares,
        "segments_by_class": segments_by_class,
        "transverse_spacing": transverse_spacing,
        "junctions": junctions,
        "branch_angles_deg": branch_angles,
        "loops": loops,
        "segment_diameter_variation": diameter_variation,
        "tissue_distance": tissue_distance,
    }
    return _json_ready(result)


def load_archive(path):
    """
    Reads the arrays `describe` needs from a centreline archive.

    Returns:
        dict: "nodes" the (4, N) float array, "metadata" the decoded record
        or None, and "edges", "node_kind" and "tree" as stored or None when
        the archive predates them.
    """
    with np.load(path, allow_pickle=False) as handle:
        names = set(handle.files)
        if "nodes" not in names:
            raise ValueError(f"{path}: not a centreline archive, it has no 'nodes' array")
        nodes = np.asarray(handle["nodes"], dtype=float)
        metadata = json.loads(handle["metadata"].item()) if "metadata" in names else None
        edges = np.asarray(handle["edges"], dtype=np.int64) if "edges" in names else None
        node_kind = np.asarray(handle["node_kind"], dtype=np.int8) if "node_kind" in names else None
        tree = np.asarray(handle["tree"]) if "tree" in names else None
    return {"nodes": nodes, "metadata": metadata, "edges": edges, "node_kind": node_kind, "tree": tree}


def describe_archive(path, volume=None, margin=None, tol=DEFAULT_TOL, frame=None, d_ref=None,
                     class_bound=DEFAULT_CLASS_BOUND, evd_spacing=None):
    """Describes the network stored in an archive; see `describe`."""
    archive = load_archive(path)
    return describe(archive["nodes"], edges=archive["edges"], metadata=archive["metadata"],
                    volume=volume, margin=margin, tol=tol, tree=archive["tree"],
                    frame=frame, d_ref=d_ref, class_bound=class_bound, evd_spacing=evd_spacing)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the geometry and topology of centreline archives and print the "
                    "descriptors as JSON, one object per archive.")
    parser.add_argument("paths", nargs="+", metavar="ARCHIVE.npz")
    parser.add_argument("--volume", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="extents in micrometres to measure the length density against "
                             "(default: the metadata's growth box, else the bounding box)")
    parser.add_argument("--margin", type=float,
                        help="collision margin for the clearance check in micrometres "
                             "(default: the metadata's collision_margin, else 0)")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"distance within which columns are one vertex (default {DEFAULT_TOL:g})")
    parser.add_argument("--compact", action="store_true", help="print each result on one line")
    args = parser.parse_args(argv)
    if args.margin is not None and args.margin < 0.0:
        parser.error("--margin cannot be negative")

    results = {}
    for path in args.paths:
        results[path] = describe_archive(path, volume=args.volume, margin=args.margin, tol=args.tol)
    output = results[args.paths[0]] if len(args.paths) == 1 else results
    print(json.dumps(output, indent=None if args.compact else 2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
