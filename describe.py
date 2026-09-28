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
"""
import argparse
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


def describe(nodes, edges=None, metadata=None, volume=None, margin=None, tol=DEFAULT_TOL, tree=None):
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

    degree = graph.degree(edges, n_columns)
    vertices = np.flatnonzero(canonical == np.arange(n_columns))
    vertex_diameter = graph.vertex_diameter(nodes, canonical)
    topology = graph.betti(edges, canonical)
    geometry = _polyline_geometry(nodes)
    length, tangent = _edge_geometry(nodes, edges)
    total_length_mm = float(length.sum()) / 1000.0
    arc_chord, curvature = _segment_stats(nodes, graph.segments(nodes, edges, canonical))

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


def describe_archive(path, volume=None, margin=None, tol=DEFAULT_TOL):
    """Describes the network stored in an archive; see `describe`."""
    archive = load_archive(path)
    return describe(archive["nodes"], edges=archive["edges"], metadata=archive["metadata"],
                    volume=volume, margin=margin, tol=tol, tree=archive["tree"])


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
