"""
Joining separately grown networks inside a field of view.

A volume assembled from many independently grown networks is a forest: every
network is its own connected component and ends in free tips everywhere,
whereas the vasculature inside a real field of view is essentially one
connected network whose only free ends are the vessels the field's edge cuts.
This module applies the anastomosis rules of anastomosis.py across a forest:
a seeded fraction of the eligible tips each look for a partner, on another
network by default, and are joined to it by a bridge that is a vessel (the
pinned persistent walk of tortuosity.py) routed clear of every input vessel
and every earlier bridge. The caller places the networks; this module only
adds bridges, and returns them separately from the inputs so that they can be
rendered, measured or truncated on their own.

Compared with anastomosis.anastomose, which closes one tree:

- networks are identified by their position in the input sequence (int32),
  never by the int8 `tree` label, so any number of networks can be joined
  and one network's roots are never mistaken for another's tips;
- roots and cut ends are never sources or partners, unless `attach_roots`
  makes the roots inside the box sources (below): a root is the first
  column of each tree (or the `roots` given), a cut end is a degree-one
  vertex that the crop made (its input `node_kind` is not TIP) or that lies
  outside the box or within `boundary_margin` of a face;
- partners lie inside the box, and a bridge whose centreline leaves the box
  is redrawn;
- bridge points are collision obstacles only, never partners, so every
  bridge end is a (network, column) of an input network;
- the selected tips are processed in a permutation drawn right after the
  Bernoulli draws, so the result does not depend on the order of the
  networks beyond the draws themselves;
- a volume budget stops the joining once the bridges placed would exceed it,
  and every outcome is reported per tip rather than as counters alone.

Everything else is the rule of `anastomose`: the search radius is `radius`
tip diameters; partners lie within FORWARD_CONE_DEG of the tip's direction;
candidates are ranked tip before interior point, then by distance, then by
(network, column); the tip's own polyline, junctions, roots, stubs (tips
inside a junction's overlap zone), points inside a junction zone, points
inside the zone of the junction the tip's polyline leaves, and points already
touching the tip are excluded, and so are close kin under policy "any"; the
bridge has diameter min(d_tip, d_partner), is sampled at the median spacing
of the tip's polyline (0.2 d_tip when that is undefined), leaves along the
tip's tangent, arrives along a partner tip's inward tangent or along the
chord into the side of a vessel, carries the walk's curvature at
`persistence` diameters, collides under the rule of collisions.py (strict
r + r + margin against every input vessel and every earlier bridge, excused
only along the tip's and the partner's polylines, plus the check of the
bridge against itself), and its end columns are bitwise copies of the joined
columns so that a graph rebuilt from coordinates closes. A tip used as a
partner stops being a source, and a bridged interior point becomes a
junction.

The walk draws its angles and axes in bulk, one array of normal and one of
uniform draws per bridge, where tortuosity.bridge_path interleaves the two
draws step by step, so a bridge drawn here is not the one bridge_path would
draw from the same generator state; the rule applied to the draws is the
same. Draws come from the caller's `numpy.random.Generator` in a fixed order:
one Bernoulli per eligible tip in (network, column) order, the processing
permutation, then the bridge walks in processing order.

Root attachment. A forest packed into a field of view has a root inside the
field for many of its networks, and every such root is a blunt start: real
vasculature has none, since a vessel's upstream end is a branch off a parent
vessel. With `attach_roots`, an eligible root (a degree-one root vertex
inside the box, clear of the faces by `boundary_margin`, and a tip of the
uncropped network; whether it lies in a junction zone does not matter)
becomes the source of a bridge to a vessel of another network at least as
thick, so that the blunt start becomes a side branch, with the machinery of
the tip bridges. The root's upstream direction is the tip tangent
_tip_tangents computes for it, the unit vector from the first distinct point
of its polyline to the root, which points out of the vessel; the bridge
departs along it and the forward cone is taken about it. A root's partner
is a vertex of degree one or two that the partner mask allows, on another
network whatever the policy, not already touching the root, inside the cone,
within `root_radius` (or `radius`) root diameters, and whose vertex diameter
is at least `root_partner_min_ratio` times the root's; the start-junction
zone (a disc around the root itself) and the kin rule (no same-network
partner is possible) do not apply. Interior points rank before tips, then
distance, then (network, column), so a root prefers a side branch to a
tip-to-tip junction. The bridge has the root's diameter, so the vessel
continues upstream at its own calibre; the step, persistence, redraws,
collision rule and excusals, bitwise end columns, the junction at an
attached interior point and the consumed attached tip are those of a tip
bridge. The root phase comes first: one Bernoulli per eligible root in
(network, column) order, a permutation of the selected roots, then the root
bridges, each an obstacle for every later bridge; then the tip sequence
above, whose draws are made even when the budget was spent during the root
phase. With no eligible root nothing is drawn and the result equals the
option off on the fields of 3.3. Root and tip bridges share
`max_bridge_volume`, and the first k bridges of any result remain a valid
result. Every bridge record carries `source_kind` ("tip" or "root"), the
root bridges are counted in the join_root_* counters rather than in
join_bridges, and the summary reports the blunt roots (eligible roots not
attached) and the free ends (free tips plus blunt roots) before and after.

The per-tip cost is kept low by computing every quantity that is constant
during a run once, as numpy arrays over the columns of all networks: degree,
network, polyline, arc position, the root, stub and junction-zone masks, the
tip tangents and each polyline's starting junction. Candidate filtering is
vectorised over the points a search returns, earlier bridges live in a
growing spatial index rather than in lists, and the collision check queries
one index per octave of vessel radius, so that a capillary-sized bridge is
compared with the capillaries near it and with the few large vessels within a
large vessel's reach, never with every point within the largest radius in the
forest.

Coordinates, diameters, `collision_margin`, `box`, `boundary_margin` and
`tol` are all in the caller's unit, whatever it is; nothing here assumes
micrometres. Python 3.9 compatible; numpy is the only dependency.
"""
import math

import numpy as np

import graph
from anastomosis import BRIDGE, FORWARD_CONE_DEG, _TreeSeparation
from collisions import KIN_REACH

# Tag of the joining stream, for a caller seeding from a run seed as main.py
# does for its stages: numpy.random.default_rng([seed, RNG_STREAMS["join"]]).
# It shares no tag with main.RNG_STREAMS.
RNG_STREAMS = {"join": 3}

POLICIES = ("cross", "any")

# What became of each degree-one vertex of the input, coded as its index here.
# The codes from root_attached on are the outcomes of an eligible root under
# attach_roots; they are appended so that the earlier codes keep their values.
TIP_OUTCOMES = ("root", "cut_end", "stub", "not_selected", "bridged_source", "bridged_partner",
                "no_partner", "collision_failed", "over_budget",
                "root_attached", "root_not_selected", "root_no_partner", "root_collision_failed",
                "root_over_budget")

# Every counter a run reports, so that the zero ones are listed too. The
# outcome counters (join_roots through join_over_budget and the five from
# join_root_attached on) partition join_tips. The counters from
# join_root_eligible on are appended so that the earlier keys keep their
# positions; they stay zero unless attach_roots is on.
EVENT_KEYS = ("join_networks", "join_points", "join_isolated", "join_tips",
              "join_roots", "join_cut_ends", "join_stubs", "join_eligible", "join_selected",
              "join_not_selected", "join_bridges", "join_bridged_partners", "join_no_partner",
              "join_collision_failed", "join_over_budget",
              "join_partner_tips", "join_partner_interior", "join_source_consumed",
              "join_redraws", "join_box_redraws", "join_kin_skipped", "join_behind_skipped",
              "join_components_joined",
              "join_root_eligible", "join_root_selected", "join_root_attached", "join_root_not_selected",
              "join_root_no_partner", "join_root_collision_failed", "join_root_over_budget")

# The outcome counter of each outcome code.
_OUTCOME_COUNTERS = (("root", "join_roots"), ("cut_end", "join_cut_ends"), ("stub", "join_stubs"),
                     ("not_selected", "join_not_selected"), ("bridged_source", "join_bridges"),
                     ("bridged_partner", "join_bridged_partners"), ("no_partner", "join_no_partner"),
                     ("collision_failed", "join_collision_failed"), ("over_budget", "join_over_budget"),
                     ("root_attached", "join_root_attached"), ("root_not_selected", "join_root_not_selected"),
                     ("root_no_partner", "join_root_no_partner"),
                     ("root_collision_failed", "join_root_collision_failed"),
                     ("root_over_budget", "join_root_over_budget"))

# One row per degree-one vertex of the (cropped) input.
TIP_DTYPE = np.dtype([("network", np.int32), ("column", np.int64), ("outcome", np.int8)])

# The `network` id of a bridge column in the merged output.
BRIDGE_NETWORK = -2

# Radius classes an octave wide for the collision indexes, as in describe.py:
# the smallest vessels of a deep tree can be a few thousandths of the root's
# radius, and finer classes than this save nothing more.
_MAX_RADIUS_CLASSES = 12

# Candidate pairs handled per pass when the junction zones are computed.
_ZONE_CHUNK = 1 << 20

# Cell coordinates are packed into one int64 key with 21 bits per axis, as in
# spatial.GridIndex; cells farther apart than 2**21 cells share a key, which
# only adds candidates that the exact distance test then removes.
_KEY_BITS = 21
_KEY_MASK = (1 << _KEY_BITS) - 1

# A grid holding no more points than this is searched by direct comparison,
# which is cheaper than enumerating cells and bounds the work of a query whose
# reach covers many cells of a sparse class (the few largest vessels, say).
_DIRECT_BELOW = 64

# Bridge points tolerated in the pending buffer, searched directly, before
# they are folded into the sorted grid of the earlier bridges: a few bridges'
# worth, so the direct search costs less than the fold it postpones.
_PENDING_LIMIT = 256

_OUTCOME = {name: code for code, name in enumerate(TIP_OUTCOMES)}


def _as_box(box):
    if box is None:
        raise ValueError("box is required: (lo, hi), two points in the caller's unit")
    lo, hi = (np.asarray(b, dtype=float).reshape(-1) for b in box)
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("box must be (lo, hi) with three finite coordinates each")
    if np.any(hi <= lo):
        raise ValueError("box must have hi > lo on every axis")
    return lo, hi


def _as_nodes(nodes):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size == 0:
        return np.empty((4, 0))
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError(f"nodes must be a (4, N) array of x, y, z and diameter, got {nodes.shape}")
    return nodes


def _as_labels(array, n_columns, name, dtype=np.int8):
    if array is None:
        return None
    array = np.asarray(array)
    if array.shape != (n_columns,):
        raise ValueError(f"{name} must hold one label per column, {n_columns}, got {array.shape}")
    return array.astype(dtype, copy=False)


def _unique_sorted_edges(low, high, n_columns):
    """Edges in graph.edges_from_nodes's form: (2, E), low first, sorted, no repeats."""
    proper = low != high
    if not proper.any():
        return np.zeros((2, 0), dtype=np.int64)
    code = np.unique(low[proper].astype(np.int64) * n_columns + high[proper].astype(np.int64))
    return np.stack([code // n_columns, code % n_columns]).astype(np.int64)


def crop_network(network, lo, hi, margin):
    """
    Crops a network to a box, by whole columns.

    A column is kept when either segment it belongs to (to the previous or the
    next column of its polyline) has a radius-padded bounding box that meets
    [lo - margin, hi + margin]; a column with no segment is kept when its own
    padded point does. Kept columns stay in archive order and every dropped
    run, separators included, becomes one NaN column, so a polyline that
    leaves the box and comes back is two polylines and no polyline is split
    into per-segment pairs: the tangent of a tip, the sampling step, the stub
    test and the collision excusal are all per polyline.

    Args:
        network (dict): "nodes" (4, N) with NaN separators and, when present,
            "node_kind", "tree" (per-column labels) and "roots" (root columns).
        lo, hi (sequence): the box, in the unit of `nodes`.
        margin (float): extra reach beyond the box; at least the largest
            vessel radius plus the joining collision margin, so that bridges
            near a face are checked against the vessels just outside it.

    Returns:
        dict: "nodes", "node_kind" and "tree" (when given) over the kept
        columns, "roots" remapped (roots whose column was dropped are
        removed) when given, and "columns", the original column of each kept
        column with -1 at separators. A network with no kept column comes
        back with (4, 0) nodes.
    """
    nodes = _as_nodes(network["nodes"])
    lo, hi = _as_box((lo, hi))
    margin = float(margin)
    if not margin >= 0.0:
        raise ValueError("the crop margin cannot be negative")
    n = nodes.shape[1]
    node_kind = _as_labels(network.get("node_kind"), n, "node_kind")
    tree = _as_labels(network.get("tree"), n, "tree")
    roots = network.get("roots")

    finite = np.isfinite(nodes[:3]).all(axis=0)
    xyz = nodes[:3]
    radius = np.maximum(np.nan_to_num(nodes[3] / 2.0, nan=0.0), 0.0)
    keep = np.zeros(n, dtype=bool)
    outer_lo, outer_hi = lo - margin, hi + margin
    if n:
        linked = finite[:-1] & finite[1:]
        first = np.flatnonzero(linked)
        second = first + 1
        pad = np.maximum(radius[first], radius[second])
        box_lo = np.minimum(xyz[:, first], xyz[:, second]) - pad
        box_hi = np.maximum(xyz[:, first], xyz[:, second]) + pad
        hit = np.all(box_lo <= outer_hi[:, None], axis=0) & np.all(box_hi >= outer_lo[:, None], axis=0)
        keep[first[hit]] = True
        keep[second[hit]] = True
        has_segment = np.zeros(n, dtype=bool)
        has_segment[first] = True
        has_segment[second] = True
        lone = np.flatnonzero(finite & ~has_segment)
        hit = (np.all(xyz[:, lone] - radius[lone] <= outer_hi[:, None], axis=0)
               & np.all(xyz[:, lone] + radius[lone] >= outer_lo[:, None], axis=0))
        keep[lone[hit]] = True

    kept = np.flatnonzero(keep)
    if kept.size == 0:
        layout = np.zeros(0, dtype=np.int64)
    else:
        # a separator wherever columns were dropped: before the first kept
        # column, between two kept columns more than one apart, after the last
        pieces = []
        if kept[0] > 0:
            pieces.append([-1])
        pieces.append([kept[0]])
        for previous, column in zip(kept[:-1], kept[1:]):
            if column - previous > 1:
                pieces.append([-1])
            pieces.append([column])
        if kept[-1] < n - 1:
            pieces.append([-1])
        layout = np.concatenate(pieces).astype(np.int64)
    separator = layout < 0
    out_nodes = np.full((4, layout.size), np.nan)
    out_nodes[:, ~separator] = nodes[:, layout[~separator]]
    result = {"nodes": out_nodes, "columns": layout}
    for name, labels in (("node_kind", node_kind), ("tree", tree)):
        if labels is not None:
            out = np.full(layout.size, -1, dtype=np.int8)
            out[~separator] = labels[layout[~separator]]
            result[name] = out
    if roots is not None:
        roots = np.asarray(roots, dtype=np.int64).reshape(-1)
        position = np.full(n, -1, dtype=np.int64)
        position[layout[~separator]] = np.flatnonzero(~separator)
        remapped = position[roots[(roots >= 0) & (roots < n)]]
        result["roots"] = remapped[remapped >= 0]
    return result


def _radius_classes(radii):
    """Class of each point, an octave of radius wide, 0 holding the largest radii."""
    r_max = float(radii.max()) if radii.size else 0.0
    positive = radii[radii > 0.0]
    if r_max <= 0.0 or positive.size == 0:
        return np.zeros(radii.size, dtype=np.int64), 1
    octaves = int(math.ceil(math.log2(r_max / positive.min()))) + 1
    n_classes = max(1, min(_MAX_RADIUS_CLASSES, octaves))
    with np.errstate(divide="ignore"):
        level = np.floor(np.log2(r_max / np.maximum(radii, np.finfo(float).tiny)))
    return np.clip(level, 0, n_classes - 1).astype(np.int64), n_classes


def _hermite(p0, t0, p1, t1, n_samples):
    u = np.linspace(0.0, 1.0, n_samples)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def _resample_by_arc_length(points, n_points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= 0.0:
        return np.repeat(points[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, 3))
    for k in range(3):
        out[:, k] = np.interp(targets, cumulative, points[:, k])
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def _pinned_walk(tangent, n_steps, h, std, angles, phis):
    """
    The persistent walk of tortuosity.free_walk, from the origin along
    `tangent`, with its rotation angles and axis angles supplied, in scalar
    arithmetic: a few microseconds a step rather than the tens that numpy
    calls on 3-vectors cost.

    Returns:
        ndarray: (n_steps + 1, 3) points of the walk.
    """
    hx, hy, hz = (float(v) for v in tangent)
    trial = (1.0, 0.0, 0.0) if abs(hx) < 0.9 else (0.0, 1.0, 0.0)
    dot = trial[0] * hx + trial[1] * hy + trial[2] * hz
    px, py, pz = trial[0] - dot * hx, trial[1] - dot * hy, trial[2] - dot * hz
    norm = math.sqrt(px * px + py * py + pz * pz)
    px, py, pz = px / norm, py / norm, pz / norm
    x = y = z = 0.0
    out = np.empty((n_steps + 1, 3))
    out[0] = 0.0
    angles = angles.tolist()
    phis = phis.tolist()
    for i in range(n_steps):
        angle = angles[i]
        phi = phis[i]
        # a random axis in the plane perpendicular to the heading
        bx, by, bz = hy * pz - hz * py, hz * px - hx * pz, hx * py - hy * px
        c, s = math.cos(phi), math.sin(phi)
        ax, ay, az = c * px + s * bx, c * py + s * by, c * pz + s * bz
        ca, sa = math.cos(angle), math.sin(angle)
        # Rodrigues rotation of the heading and the perpendicular about it
        k = (ax * hx + ay * hy + az * hz) * (1.0 - ca)
        nx = hx * ca + (ay * hz - az * hy) * sa + ax * k
        ny = hy * ca + (az * hx - ax * hz) * sa + ay * k
        nz = hz * ca + (ax * hy - ay * hx) * sa + az * k
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        hx, hy, hz = nx / norm, ny / norm, nz / norm
        k = (ax * px + ay * py + az * pz) * (1.0 - ca)
        qx = px * ca + (ay * pz - az * py) * sa + ax * k
        qy = py * ca + (az * px - ax * pz) * sa + ay * k
        qz = pz * ca + (ax * py - ay * px) * sa + az * k
        dot = qx * hx + qy * hy + qz * hz
        qx, qy, qz = qx - dot * hx, qy - dot * hy, qz - dot * hz
        norm = math.sqrt(qx * qx + qy * qy + qz * qz)
        px, py, pz = qx / norm, qy / norm, qz / norm
        x += h * hx
        y += h * hy
        z += h * hz
        out[i + 1, 0] = x
        out[i + 1, 1] = y
        out[i + 1, 2] = z
    return out


def bridge_geometry(start, end, step, rng, persistence, diameter, start_tangent=None, end_tangent=None,
                    tangent_scale=0.6):
    """
    Traces a bridge from `start` to `end`: the construction of
    tortuosity.bridge_path (a cubic Hermite curve leaving along
    `start_tangent` and arriving along `end_tangent`, each defaulting to the
    chord, resampled by arc length, plus the deviation of a persistent walk
    pinned to zero at both ends), with the walk's draws taken in bulk, the
    angles first and the axes second.

    Returns:
        ndarray: (m, 3) points, m >= 3, from `start` to `end` inclusive.
    """
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    chord = p1 - p0
    length = float(np.linalg.norm(chord))
    if length <= 0.0:
        raise ValueError("a bridge needs two distinct points")
    if not step > 0.0:
        raise ValueError("step must be positive")
    direction = chord / length
    tangent = direction if start_tangent is None else np.asarray(start_tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    arrival = direction if end_tangent is None else np.asarray(end_tangent, dtype=float)
    arrival = arrival / np.linalg.norm(arrival)
    n_steps = max(2, int(math.ceil(1.05 * length / step)))
    dense = _hermite(p0, tangent * tangent_scale * length, p1, arrival * tangent_scale * length,
                     8 * n_steps + 1)
    base = _resample_by_arc_length(dense, n_steps + 1)
    base[0] = p0
    base[-1] = p1
    arc = float(np.sum(np.linalg.norm(np.diff(base, axis=0), axis=1)))
    h = arc / n_steps
    if not (persistence > 0.0 and diameter > 0.0):
        raise ValueError("persistence and diameter must be positive")
    std = math.sqrt(h / (persistence * diameter))
    angles = rng.normal(0.0, std, n_steps)
    phis = rng.uniform(0.0, 2.0 * math.pi, n_steps)
    walk = _pinned_walk(tangent, n_steps, h, std, angles, phis)
    straight = np.arange(n_steps + 1)[:, None] * h * tangent
    deviation = walk - straight
    weights = (np.arange(n_steps + 1) / n_steps)[:, None]
    out = base + deviation - weights * deviation[-1]
    out[0] = p0
    out[-1] = p1
    return out


def _any_within(q, stored, threshold):
    """Whether any pair has |q - stored| < threshold[stored], one axis at a time, so no (q, s, 3) temporary."""
    d2 = np.zeros((q.shape[0], stored.shape[0]))
    for axis in range(3):
        diff = q[:, axis, None] - stored[None, :, axis]
        np.square(diff, out=diff)
        d2 += diff
    return bool(np.any(d2 < (threshold ** 2)[None, :]))


class _StaticGrid:
    """
    Uniform grid over a fixed set of points, for the two searches the joining
    makes many thousands of times: the points whose spheres collide with a
    set of query spheres, and the points within a radius of one point. It
    answers exactly what spatial.GridIndex answers for the same points (the
    strict r_P + r_Q + margin test, the inclusive distance test), with a
    fraction of the calls per query: the points never change, so there is no
    pending buffer to merge, no validation beyond the caller's and no
    de-duplication, since each covered cell is visited once per query point.
    """

    def __init__(self, points, radii, tags, cell_size):
        self.points = np.ascontiguousarray(points, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.tags = np.asarray(tags, dtype=np.int64)
        self.n = self.points.shape[0]
        self.cell = float(cell_size)
        self.r_max = float(self.radii.max()) if self.n else 0.0
        keys = self._keys(np.floor(self.points / self.cell).astype(np.int64))
        order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[order]
        self.sorted_index = order

    @staticmethod
    def _keys(cells):
        return ((cells[..., 0] & _KEY_MASK)
                | ((cells[..., 1] & _KEY_MASK) << _KEY_BITS)
                | ((cells[..., 2] & _KEY_MASK) << (2 * _KEY_BITS)))

    def _candidates(self, q, reach):
        """(qi, si) covering every stored point within reach[qi] of q[qi]."""
        if self.n <= _DIRECT_BELOW:
            return self._direct(q, reach)
        lo = np.floor((q - reach[:, None]) / self.cell).astype(np.int64)
        hi = np.floor((q + reach[:, None]) / self.cell).astype(np.int64)
        per_axis = hi - lo + 1
        counts = per_axis.prod(axis=1)
        total = int(counts.sum())
        if total > 4 * self.n:
            return self._direct(q, reach)
        owner = np.repeat(np.arange(len(q), dtype=np.int64), counts)
        local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        ny, nz = per_axis[owner, 1], per_axis[owner, 2]
        cells = np.stack([lo[owner, 0] + local // (ny * nz),
                          lo[owner, 1] + (local // nz) % ny,
                          lo[owner, 2] + local % nz], axis=1)
        keys = self._keys(cells)
        left = np.searchsorted(self.sorted_keys, keys, side="left")
        right = np.searchsorted(self.sorted_keys, keys, side="right")
        hits = right - left
        found = int(hits.sum())
        if found == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        entry = np.repeat(np.arange(len(keys), dtype=np.int64), hits)
        slot = np.repeat(left - (np.cumsum(hits) - hits), hits) + np.arange(found, dtype=np.int64)
        return owner[entry], self.sorted_index[slot]

    def _direct(self, q, reach):
        d2 = np.zeros((q.shape[0], self.n))
        for axis in range(3):
            diff = q[:, axis, None] - self.points[None, :, axis]
            np.square(diff, out=diff)
            d2 += diff
        qi, si = np.nonzero(d2 <= (reach ** 2)[:, None])
        return qi.astype(np.int64), si.astype(np.int64)

    @staticmethod
    def _widen(reach, points):
        """A few units in the last place, so rounding cannot drop a pair on the edge of the reach."""
        return reach + 8.0 * np.finfo(float).eps * (np.abs(points).max(axis=1) + reach)

    def collisions(self, q, radius, margin):
        """
        Every (qi, si) with |q[qi] - P[si]| < radius + r[si] + margin, for a
        set of query points sharing one radius.
        """
        if self.n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        reach = np.full(len(q), radius + self.r_max + margin)
        qi, si = self._candidates(q, self._widen(reach, q))
        if qi.size == 0:
            return qi, si
        distance = np.linalg.norm(q[qi] - self.points[si], axis=1)
        keep = distance < radius + self.radii[si] + margin
        return qi[keep], si[keep]

    def within(self, point, radius):
        """Stored indices within `radius` of `point` (inclusive), nearest first, ties by index."""
        if self.n == 0:
            return np.zeros(0, dtype=np.int64)
        q = np.asarray(point, dtype=float)[None, :]
        _, si = self._candidates(q, self._widen(np.array([radius]), q))
        if si.size == 0:
            return si
        distance = np.linalg.norm(self.points[si] - q[0], axis=1)
        keep = distance <= radius
        si, distance = si[keep], distance[keep]
        return si[np.lexsort((si, distance))]


class _GrowingGrid:
    """
    The interior points of the bridges placed so far. New points wait in a
    buffer that is searched directly, and once the buffer holds more than
    _PENDING_LIMIT points they are merged into the sorted keys of the grid in
    one pass, so that the cost of growing the grid stays linear in its size
    per batch rather than per bridge.
    """

    def __init__(self, cell_size):
        self.cell = float(cell_size)
        self.n = 0
        self.r_max = 0.0
        self._points = np.empty((0, 3))
        self._radii = np.empty(0)
        self._sorted_keys = np.empty(0, dtype=np.int64)
        self._sorted_index = np.empty(0, dtype=np.int64)
        self._committed = 0
        self._pending = np.empty((0, 3))
        self._pending_radii = np.empty(0)

    def add(self, points, radius):
        points = np.asarray(points, dtype=float)
        count = points.shape[0]
        if count == 0:
            return
        self._pending = np.concatenate([self._pending, points], axis=0)
        self._pending_radii = np.concatenate([self._pending_radii, np.full(count, float(radius))])
        self.n += count
        self.r_max = max(self.r_max, float(radius))
        if self._pending.shape[0] > _PENDING_LIMIT:
            self._fold()

    def _fold(self):
        """Merges the pending points into the sorted grid in one linear pass."""
        pending = self._pending
        count = pending.shape[0]
        needed = self._committed + count
        if needed > self._points.shape[0]:
            capacity = max(needed, 2 * self._points.shape[0], 4096)
            points = np.empty((capacity, 3))
            radii = np.empty(capacity)
            points[:self._committed] = self._points[:self._committed]
            radii[:self._committed] = self._radii[:self._committed]
            self._points, self._radii = points, radii
        self._points[self._committed:needed] = pending
        self._radii[self._committed:needed] = self._pending_radii
        keys = _StaticGrid._keys(np.floor(pending / self.cell).astype(np.int64))
        order = np.argsort(keys, kind="stable")
        new_keys = keys[order]
        new_index = order + self._committed
        slots = np.searchsorted(self._sorted_keys, new_keys, side="right")
        self._sorted_keys = np.insert(self._sorted_keys, slots, new_keys)
        self._sorted_index = np.insert(self._sorted_index, slots, new_index)
        self._committed = needed
        self._pending = np.empty((0, 3))
        self._pending_radii = np.empty(0)

    def hits(self, q, radius, margin):
        """Whether any stored point collides with a query point of `radius`."""
        if self._pending.shape[0]:
            if _any_within(q, self._pending, radius + self._pending_radii + margin):
                return True
        if self._committed == 0:
            return False
        reach = np.full(len(q), radius + self.r_max + margin)
        reach = _StaticGrid._widen(reach, q)
        lo = np.floor((q - reach[:, None]) / self.cell).astype(np.int64)
        hi = np.floor((q + reach[:, None]) / self.cell).astype(np.int64)
        per_axis = hi - lo + 1
        counts = per_axis.prod(axis=1)
        total = int(counts.sum())
        committed = self._points[:self._committed]
        if total > 4 * self._committed:
            return _any_within(q, committed, radius + self._radii[:self._committed] + margin)
        owner = np.repeat(np.arange(len(q), dtype=np.int64), counts)
        local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        ny, nz = per_axis[owner, 1], per_axis[owner, 2]
        cells = np.stack([lo[owner, 0] + local // (ny * nz),
                          lo[owner, 1] + (local // nz) % ny,
                          lo[owner, 2] + local % nz], axis=1)
        keys = _StaticGrid._keys(cells)
        left = np.searchsorted(self._sorted_keys, keys, side="left")
        right = np.searchsorted(self._sorted_keys, keys, side="right")
        found = right - left
        hit_total = int(found.sum())
        if hit_total == 0:
            return False
        entry = np.repeat(np.arange(len(keys), dtype=np.int64), found)
        slot = np.repeat(left - (np.cumsum(found) - found), found) + np.arange(hit_total, dtype=np.int64)
        qi, si = owner[entry], self._sorted_index[slot]
        distance = np.linalg.norm(q[qi] - committed[si], axis=1)
        return bool(np.any(distance < radius + self._radii[si] + margin))


class _Forest:
    """
    The placed networks laid out as one archive with a separator between
    networks, and every per-column quantity the joining needs.
    """

    def __init__(self, networks, tol, lo, hi, margin, boundary_margin, policy, min_separation,
                 check_roots=False):
        self.n_networks = len(networks)
        pieces, kinds, trees, bases = [], [], [], []
        root_columns = []
        self.graphs = []
        total = 0
        for k, network in enumerate(networks):
            nodes = _as_nodes(network["nodes"])
            n = nodes.shape[1]
            built = graph.build(nodes, tol) if n else {"canonical": np.zeros(0, dtype=np.int64),
                                                        "edges": np.zeros((2, 0), dtype=np.int64),
                                                        "degree": np.zeros(0, dtype=np.int64),
                                                        "node_kind": np.zeros(0, dtype=np.int8)}
            kind = _as_labels(network.get("node_kind"), n, "node_kind")
            if kind is None:
                kind = built["node_kind"]
            tree = _as_labels(network.get("tree"), n, "tree")
            roots = network.get("roots")
            if roots is not None:
                roots = np.asarray(roots, dtype=np.int64).reshape(-1)
                if roots.size and (roots.min() < 0 or roots.max() >= n):
                    raise ValueError(f"network {k}: roots index columns outside 0..{n - 1}")
            else:
                roots = self._default_roots(nodes, tree, built["canonical"])
            bases.append(total)
            self.graphs.append(built)
            pieces.append(nodes)
            kinds.append(kind)
            trees.append(np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8) if tree is None else tree)
            root_columns.append(roots + total)
            total += n
            if k < self.n_networks - 1:
                pieces.append(np.full((4, 1), np.nan))
                kinds.append(np.array([-1], dtype=np.int8))
                trees.append(np.array([-1], dtype=np.int8))
                total += 1
        self.base = np.array(bases, dtype=np.int64)
        self.nodes = np.concatenate(pieces, axis=1) if pieces else np.empty((4, 0))
        self.n = self.nodes.shape[1]
        self.kind_in = np.concatenate(kinds) if kinds else np.zeros(0, dtype=np.int8)
        self.tree = np.concatenate(trees) if trees else np.zeros(0, dtype=np.int8)
        self.xyz = np.ascontiguousarray(self.nodes[:3].T)

        # per column: network, canonical column, degree, vertex diameter, polyline, arc
        self.net = np.full(self.n, -1, dtype=np.int32)
        self.canon = np.full(self.n, -1, dtype=np.int64)
        self.deg = np.zeros(self.n, dtype=np.int32)
        self.diam = np.full(self.n, np.nan)
        edges = []
        components = 0
        self.component = np.full(self.n, -1, dtype=np.int64)
        for k, built in enumerate(self.graphs):
            n_k = pieces[2 * k].shape[1]
            if n_k == 0:
                continue
            start, stop = self.base[k], self.base[k] + n_k
            canonical = built["canonical"]
            finite = canonical >= 0
            self.net[start:stop] = k
            self.canon[start:stop][finite] = canonical[finite] + start
            self.deg[start:stop] = built["degree"]
            self.diam[start:stop] = graph.vertex_diameter(pieces[2 * k], canonical)
            edges.append(built["edges"] + start)
            labels, count = graph.components(built["edges"], canonical)
            vertices = labels >= 0
            self.component[start:stop][vertices] = labels[vertices] + components
            components += count
        self.edges = np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
        self.components_before = components
        self.finite = self.canon >= 0
        self.vertices = np.flatnonzero(self.canon == np.arange(self.n))
        self.pid = graph.polyline_of_column(self.nodes)
        self.arc = np.zeros(self.n)
        finite_columns = np.flatnonzero(self.finite)
        if finite_columns.size:
            step = np.zeros(finite_columns.size)
            same = self.pid[finite_columns[1:]] == self.pid[finite_columns[:-1]]
            step[1:][same] = np.linalg.norm(self.xyz[finite_columns[1:][same]] - self.xyz[finite_columns[:-1][same]],
                                            axis=1)
            cumulative = np.cumsum(step)
            starts = np.ones(finite_columns.size, dtype=bool)
            starts[1:] = ~same
            start_of = np.flatnonzero(starts)
            self.arc[finite_columns] = cumulative - cumulative[start_of][np.cumsum(starts) - 1]
            self.poly_start = finite_columns[start_of]
            self.poly_end = np.append(finite_columns[start_of[1:] - 1], finite_columns[-1]) + 1
        else:
            self.poly_start = np.zeros(0, dtype=np.int64)
            self.poly_end = np.zeros(0, dtype=np.int64)
        self.n_polylines = self.poly_start.size
        # the junction a polyline leaves: the vertex of its first column
        self.start_junction = self.canon[self.poly_start] if self.n_polylines else np.zeros(0, dtype=np.int64)

        self.is_root = np.zeros(self.n, dtype=bool)
        for k, roots in enumerate(root_columns):
            if roots.size:
                canonical = self.canon[roots]
                if check_roots and np.any(canonical < 0):
                    bad = int(roots[np.flatnonzero(canonical < 0)[0]] - self.base[k])
                    raise ValueError(f"network {k}: root column {bad} is a NaN separator")
                self.is_root[canonical] = True
        self.lo, self.hi = lo, hi
        self.margin = margin
        inside = np.all((self.xyz >= lo) & (self.xyz <= hi), axis=1)
        near_face = np.any((self.xyz < lo + boundary_margin) | (self.xyz > hi - boundary_margin), axis=1)
        self.inside = inside
        self.in_zone = self._junction_zones()

        degree = self.deg[self.vertices]
        tips = self.vertices[degree == 1]
        self.tips = tips
        self.isolated = int(np.count_nonzero(degree == 0))
        is_cut = (self.kind_in[tips] != graph.TIP) | ~inside[tips] | near_face[tips]
        self.tip_class = np.full(self.n, -1, dtype=np.int8)
        self.tip_class[tips] = _OUTCOME["not_selected"]        # eligible until classified otherwise
        self.tip_class[tips[self.in_zone[tips]]] = _OUTCOME["stub"]
        self.tip_class[tips[is_cut]] = _OUTCOME["cut_end"]
        roots = tips[self.is_root[tips]]
        self.tip_class[roots] = np.where(inside[roots], _OUTCOME["root"], _OUTCOME["cut_end"])
        self.eligible = tips[self.tip_class[tips] == _OUTCOME["not_selected"]]
        # the degree-one roots, and those that may attach under attach_roots:
        # inside the box, clear of the faces by the test tips use, and tips of
        # the uncropped network; whether one lies in a junction zone does not
        # matter, since the zone is a disc around the root itself
        self.roots = roots
        self.root_eligible = roots[inside[roots] & ~near_face[roots] & (self.kind_in[roots] == graph.TIP)]

        # partners: vertices of degree one or two, never a root, a cut end, a
        # stub or a point inside a junction zone, and inside the box
        partner = ((degree == 1) | (degree == 2)) & ~self.is_root[self.vertices] & inside[self.vertices]
        partner &= ~((degree == 1) & (self.tip_class[self.vertices] != _OUTCOME["not_selected"]))
        partner &= ~((degree == 2) & self.in_zone[self.vertices])
        self.partner_ok = np.zeros(self.n, dtype=bool)
        self.partner_ok[self.vertices[partner]] = True
        self.tangent = self._tip_tangents()

        self.separation = None
        if policy == "any" and min_separation > 1:
            self.separation = []
            for k, built in enumerate(self.graphs):
                n_k = pieces[2 * k].shape[1]
                self.separation.append(_TreeSeparation(pieces[2 * k], built["edges"], built["canonical"])
                                       if n_k else None)

    @staticmethod
    def _default_roots(nodes, tree, canonical):
        """First finite column of each tree label other than BRIDGE, or the first finite column."""
        finite = np.flatnonzero(canonical >= 0)
        if finite.size == 0:
            return np.zeros(0, dtype=np.int64)
        if tree is None:
            return finite[:1]
        roots = []
        for label in np.unique(tree[finite]):
            if label == BRIDGE or label < 0:
                continue
            roots.append(int(finite[np.flatnonzero(tree[finite] == label)[0]]))
        return np.array(roots, dtype=np.int64)

    def _junction_zones(self):
        """
        Whether each vertex lies in the overlap zone of a junction of its own
        polyline: the polyline's start vertex or a vertex of degree three or
        more along it, other than the vertex itself, within
        KIN_REACH (r_P + r_J + margin) of it.
        """
        in_zone = np.zeros(self.n, dtype=bool)
        if self.n_polylines == 0:
            return in_zone
        # (polyline, junction) pairs: the start vertex and every branching vertex
        finite = np.flatnonzero(self.finite)
        branching = finite[self.deg[self.canon[finite]] >= 3]
        pairs = np.unique(np.concatenate([np.arange(self.n_polylines, dtype=np.int64) * self.n + self.start_junction,
                                          self.pid[branching] * self.n + self.canon[branching]]))
        pair_poly, pair_junction = pairs // self.n, pairs % self.n
        # every canonical column of degree one or two of that polyline against each junction
        candidate = self.vertices[(self.deg[self.vertices] == 1) | (self.deg[self.vertices] == 2)]
        poly_of = self.pid[candidate]
        order = np.argsort(poly_of, kind="stable")
        candidate = candidate[order]
        poly_of = poly_of[order]
        first = np.searchsorted(poly_of, pair_poly, side="left")
        last = np.searchsorted(poly_of, pair_poly, side="right")
        counts = last - first
        radius = self.diam / 2.0
        # pairs in passes of a bounded size, so the temporaries stay small on a large forest
        per_pass = max(1, _ZONE_CHUNK // max(1, int(counts.max()) if counts.size else 1))
        for start in range(0, pairs.size, per_pass):
            stop = min(start + per_pass, pairs.size)
            owner = np.repeat(np.arange(start, stop), counts[start:stop])
            if owner.size == 0:
                continue
            offset = np.arange(owner.size) - np.repeat(np.cumsum(counts[start:stop]) - counts[start:stop],
                                                       counts[start:stop])
            point = candidate[first[owner] + offset]
            junction = pair_junction[owner]
            other = point != junction
            point, junction = point[other], junction[other]
            distance = np.linalg.norm(self.xyz[point] - self.xyz[junction], axis=1)
            reach = KIN_REACH * (radius[point] + radius[junction] + self.margin)
            in_zone[point[distance < reach]] = True
        return in_zone

    def _tip_tangents(self):
        """Unit direction each tip points in, away from the last distinct point of its polyline before it."""
        tangent = np.full((self.n, 3), np.nan)
        tips = self.tips
        if tips.size == 0:
            return tangent
        start = self.poly_start[self.pid[tips]]
        end = self.poly_end[self.pid[tips]]
        # the common case: the previous column of the polyline is distinct
        has_previous = tips > start
        previous = np.where(has_previous, tips - 1, tips)
        delta = self.xyz[tips] - self.xyz[previous]
        norm = np.linalg.norm(delta, axis=1)
        quick = has_previous & (norm > 0.0)
        tangent[tips[quick]] = delta[quick] / norm[quick, None]
        for tip, s, e in zip(tips[~quick], start[~quick], end[~quick]):
            here = self.xyz[tip]
            inward = range(tip - 1, s - 1, -1) if tip > s else range(tip + 1, e)
            for other in inward:
                d = here - self.xyz[other]
                length = math.sqrt(float(d @ d))
                if length > 0.0:
                    tangent[tip] = d / length
                    break
        return tangent

    def polyline_spacing(self, column):
        """Median spacing of the polyline of `column`, or None if it has fewer than two points."""
        start, end = self.poly_start[self.pid[column]], self.poly_end[self.pid[column]]
        if end - start < 2:
            return None
        steps = np.linalg.norm(np.diff(self.xyz[start:end], axis=0), axis=1)
        steps = steps[steps > 0.0]
        return float(np.median(steps)) if steps.size else None


class _Obstacles:
    """
    The collision indexes: one grid per octave of vessel radius over the
    input vertices, built once and searched most populous class first so
    that a colliding bridge is rejected after one search as often as
    possible, and one growing grid over the interior points of the bridges
    placed so far.
    """

    def __init__(self, forest, margin, cell_hint):
        self.forest = forest
        self.margin = margin
        self.classes = []
        vertices = forest.vertices
        radii = forest.diam[vertices] / 2.0
        if vertices.size:
            level, n_classes = _radius_classes(radii)
            populated = [j for j in range(n_classes) if np.any(level == j)]
            for j in sorted(populated, key=lambda j: (-int(np.count_nonzero(level == j)), j)):
                members = level == j
                r_class = float(radii[members].max())
                self.classes.append(_StaticGrid(forest.xyz[vertices[members]], radii[members], vertices[members],
                                                max(2.0 * r_class + margin, 1e-9)))
        self.bridges = _GrowingGrid(max(cell_hint, 1e-9))

    def collides(self, path, radius, tip, partner):
        """
        Whether a bridge's interior points collide with an input vessel, an
        earlier bridge or the bridge itself, under the rule of collisions.py:
        a stored point P and a bridge point Q collide when
        |P - Q| < r_P + r_Q + margin unless P lies on the polyline of the tip
        (or of the partner) and arc(P, J) + arc(Q, J) < KIN_REACH
        (r_P + r_Q + margin) for J that end, arc lengths measured along P's
        polyline and along the bridge; two points of the bridge collide when
        their arc-length separation exceeds 2 (2 r_Q + margin) while they lie
        within 2 r_Q + margin of each other.
        """
        forest = self.forest
        margin = self.margin
        interior = path[1:-1]
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        bridge_arc = np.concatenate([[0.0], np.cumsum(steps)])
        from_tip = bridge_arc[1:-1]
        from_partner = bridge_arc[-1] - bridge_arc[1:-1]
        ends = ((forest.pid[tip], forest.arc[tip], from_tip),
                (forest.pid[partner], forest.arc[partner], from_partner))
        for grid in self.classes:
            qi, si = grid.collisions(interior, radius, margin)
            if qi.size == 0:
                continue
            columns = grid.tags[si]
            threshold = grid.radii[si] + radius + margin
            excused = np.zeros(qi.size, dtype=bool)
            for pid, arc_end, along in ends:
                kin = forest.pid[columns] == pid
                if kin.any():
                    together = np.abs(forest.arc[columns[kin]] - arc_end) + along[qi[kin]]
                    excused[np.flatnonzero(kin)[together < KIN_REACH * threshold[kin]]] = True
            if not excused.all():
                return True
        if self.bridges.n and self.bridges.hits(interior, radius, margin):
            return True
        gap = np.abs(from_tip[:, None] - from_tip[None, :])
        close = np.linalg.norm(interior[:, None, :] - interior[None, :, :], axis=2) < 2.0 * radius + margin
        return bool(np.any(close & (gap > 2.0 * (2.0 * radius + margin))))

    def add_bridge(self, path, radius):
        if len(path) > 2:
            self.bridges.add(path[1:-1], radius)


def join_networks(networks, rng, *, fraction, collision_margin, box, boundary_margin, radius=25.0,
                  policy="cross", min_separation=3, persistence=8.0, attempts=10, max_candidates=5,
                  tol=graph.DEFAULT_TOL, max_bridge_volume=None, merged=False, events=None,
                  attach_roots=False, root_fraction=1.0, root_partner_min_ratio=1.0, root_radius=None):
    """
    Bridges a seeded fraction of the free tips of placed networks to partners
    on other networks (or, under policy "any", anywhere far enough along the
    graph), inside a box, and optionally attaches the roots inside the box to
    vessels of other networks at least as thick.

    Args:
        networks (sequence): placed networks in one common frame and unit,
            each a dict with "nodes" ((4, N) x, y, z, diameter with NaN
            separators) and optionally "node_kind" (the kind each column had
            in the uncropped network, from which cut ends are told from
            tips), "tree" (per-column labels) and "roots" (root columns;
            empty means none). A network's identity is its position in the
            sequence. Edges are ignored: each network's graph is rebuilt
            from its own coordinates with `tol`, so networks never merge by
            coincidence.
        rng (numpy.random.Generator): the joining stream, seeded by the caller.
        fraction (float): share of the eligible tips drawn as sources, in [0, 1].
        collision_margin (float): clearance every bridge keeps from every
            input vessel and every earlier bridge, caller units.
        box (tuple): (lo, hi), the field of view in caller units; sources and
            partners lie inside it and a bridge never leaves it.
        boundary_margin (float): a tip within this distance of a face of the
            box is a cut end, caller units.
        radius (float): partner search radius in multiples of the tip diameter.
        policy (str): "cross" allows partners on other networks only; "any"
            also allows the same network at a graph separation of at least
            `min_separation` segments (another component of the same
            network counts as far enough).
        min_separation (int): see `policy`; 3 excludes the parent and sister
            stems, 1 or less excludes nothing.
        persistence (float): persistence of the bridge walk, in bridge diameters.
        attempts (int): redraws of a colliding or box-leaving bridge per candidate.
        max_candidates (int): partners tried per tip before giving up.
        tol (float): coincidence tolerance for each network's graph, caller units.
        max_bridge_volume (float or None): budget for the summed bridge
            volumes pi r^2 x arc, caller units cubed, shared by root and tip
            bridges; once a bridge would exceed it no bridge starts and the
            remaining selected roots and tips are reported as
            root_over_budget and over_budget. None sets no budget.
        merged (bool): also return the inputs and the bridges as one archive.
        events (dict or None): counters incremented in place (EVENT_KEYS).
        attach_roots (bool): let each eligible root (a degree-one root
            vertex inside the box, clear of the faces by `boundary_margin`,
            whose input node_kind is a tip) become the source of a bridge to
            a vessel of another network, as a side branch off that vessel.
            Off by default, in which case roots are never sources.
        root_fraction (float): share of the eligible roots drawn as sources,
            in [0, 1].
        root_partner_min_ratio (float): a root's partner must have a vertex
            diameter of at least this many root diameters; at least 1.
        root_radius (float or None): partner search radius for a root in
            multiples of the root diameter; None uses `radius`.

    Returns:
        dict:
            "bridges": one dict per bridge, in the order placed, with "tip"
                and "partner" as (network, column), "source_kind" ("tip" or
                "root": the source the "tip" field names), "partner_kind"
                ("tip" or "interior"), "geometry" a (4, m) float64 array in
                the caller's frame whose first and last columns are bitwise
                copies of the joined columns (with the bridge diameter),
                "chord", "arc", "diameter", "volume" and "redraws" (attempts
                that failed before this bridge was placed);
            "bridge_nodes": the bridges concatenated with NaN separators, (4, M);
            "tips": a structured array (TIP_DTYPE) with one row per
                degree-one vertex of the input, its "network", "column" and
                "outcome" code into TIP_OUTCOMES, in (network, column) order;
            "events": the counters;
            "summary": components, free in-box tips (neither root nor cut
                end), blunt roots (eligible roots not attached) and free ends
                (both together), each per unit length too, before and after;
            "merged" (when asked for): "nodes" with the bridges appended
                after NaN separators, "edges", "node_kind", "network" (the
                network of each column, BRIDGE_NETWORK on bridges, -1 at
                separators) and "tree" (BRIDGE on bridges).

        The first k bridges of any result are themselves a valid result, so a
        caller may truncate after measuring what it rendered: each bridge was
        checked against the inputs and the bridges before it only.

    Root attachment. With `attach_roots`, the eligible roots are drawn first
    (one Bernoulli per eligible root in (network, column) order, then a
    permutation of the selected ones), and each selected root looks for a
    partner within `root_radius` root diameters on another network, whatever
    `policy` is: a vertex of degree one or two that the partner mask allows,
    not already touching the root, within FORWARD_CONE_DEG of the root's
    upstream direction (the unit vector from the first distinct point of its
    polyline to the root, which points out of the vessel), and whose vertex
    diameter is at least `root_partner_min_ratio` times the root's. Interior
    points rank before tips, then by distance, then by (network, column).
    The bridge has the root's diameter, so the vessel continues upstream at
    its own calibre, leaves along the upstream direction and arrives as a
    tip's bridge does; everything else (step, persistence, redraws, the
    collision rule and its excusals, the bitwise end columns, the junction
    or consumed tip at the partner) is the rule for tips. The tip draws
    follow the root bridges, so a run with the option on and at least one
    eligible root draws differently from a run with it off; with no eligible
    root the option draws nothing and the result is the same on the fields
    of 3.3. An ineligible root (outside the box, near a face, or not a tip
    of the uncropped network) is reported as cut_end, and an eligible one as
    exactly one of root_attached, root_not_selected, root_no_partner,
    root_collision_failed and root_over_budget; the code root never occurs
    with the option on.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction!r}")
    if not radius > 0.0:
        raise ValueError("radius must be positive")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if collision_margin is None or not float(collision_margin) >= 0.0:
        raise ValueError("collision_margin is required and cannot be negative")
    if boundary_margin is None or not float(boundary_margin) >= 0.0:
        raise ValueError("boundary_margin is required and cannot be negative")
    if not persistence > 0.0:
        raise ValueError("persistence must be positive")
    if attempts < 0 or max_candidates < 1:
        raise ValueError("attempts cannot be negative and max_candidates must be at least 1")
    if max_bridge_volume is not None and not max_bridge_volume >= 0.0:
        raise ValueError("max_bridge_volume cannot be negative")
    if not 0.0 <= root_fraction <= 1.0:
        raise ValueError(f"root_fraction must lie in [0, 1], got {root_fraction!r}")
    if not root_partner_min_ratio >= 1.0:
        raise ValueError(f"root_partner_min_ratio must be at least 1, got {root_partner_min_ratio!r}")
    if root_radius is not None and not root_radius > 0.0:
        raise ValueError(f"root_radius must be positive or None, got {root_radius!r}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    lo, hi = _as_box(box)
    margin = float(collision_margin)
    boundary_margin = float(boundary_margin)
    min_separation = int(min_separation)
    attach_roots = bool(attach_roots)
    root_search = float(radius if root_radius is None else root_radius)
    if events is None:
        events = {}
    for key in EVENT_KEYS:
        events.setdefault(key, 0)
    cone = math.cos(math.radians(FORWARD_CONE_DEG))

    forest = _Forest(networks, tol, lo, hi, margin, boundary_margin, policy, min_separation,
                     check_roots=attach_roots)
    n_networks = forest.n_networks
    events["join_networks"] += n_networks
    events["join_points"] += int(np.count_nonzero(forest.finite))
    events["join_isolated"] += forest.isolated
    tips = forest.tips
    outcome = forest.tip_class.copy()
    events["join_tips"] += int(tips.size)
    eligible = forest.eligible
    events["join_eligible"] += int(eligible.size)
    root_eligible = forest.root_eligible
    if attach_roots:
        outcome[forest.roots] = _OUTCOME["cut_end"]
        outcome[root_eligible] = _OUTCOME["root_not_selected"]
        events["join_root_eligible"] += int(root_eligible.size)

    deg = forest.deg
    xyz = forest.xyz
    diam = forest.diam
    pid = forest.pid
    net = forest.net
    partner_columns = np.flatnonzero(forest.partner_ok)
    tip_reach = radius * diam[eligible] if eligible.size else np.zeros(0)
    cell_hint = float(np.median(tip_reach)) if tip_reach.size else 1.0
    partners = _StaticGrid(xyz[partner_columns], np.zeros(partner_columns.size), partner_columns,
                           max(cell_hint, 1e-9))
    bridge_cell = float(np.median(diam[eligible])) + margin if eligible.size else 1.0
    obstacles = _Obstacles(forest, margin, bridge_cell)

    bridges = []
    total_volume = 0.0
    budget_spent = False
    component_parent = np.arange(forest.components_before, dtype=np.int64)

    def find(c):
        while component_parent[c] != c:
            component_parent[c] = component_parent[component_parent[c]]
            c = component_parent[c]
        return c

    def place(source, source_kind, v, d_v, chord, rank, step, departure, bridge_diameter, codes):
        """
        Tries the candidates `v` in the order of `rank`, then distance, then
        column, up to max_candidates, drawing up to attempts + 1 bridges to
        each, and places the first bridge that stays in the box and clear of
        every obstacle: the source's outcome becomes codes["placed"], a
        partner tip is consumed, a partner interior point becomes a junction
        and the components merge. Otherwise the outcome is codes["failed"],
        or codes["over_budget"] when the bridge found would exceed the budget,
        which then stops every later bridge.
        """
        nonlocal total_volume, budget_spent
        ranked = np.lexsort((v, chord, rank))[:max_candidates]
        source_xyz = xyz[source]
        joined = None
        failed = 0
        for at in ranked.tolist():
            partner = int(v[at])
            partner_is_tip = deg[partner] == 1
            d_bridge = bridge_diameter(float(d_v[at]))
            partner_xyz = xyz[partner]
            arrival = None
            if partner_is_tip:
                outward = forest.tangent[partner]
                if np.all(np.isfinite(outward)):
                    arrival = -outward
            for _ in range(attempts + 1):
                path = bridge_geometry(source_xyz, partner_xyz, step, rng, persistence, d_bridge,
                                       start_tangent=departure, end_tangent=arrival)
                if np.any(path < lo) or np.any(path > hi):
                    events["join_box_redraws"] += 1
                    failed += 1
                    continue
                if obstacles.collides(path, d_bridge / 2.0, source, partner):
                    events["join_redraws"] += 1
                    failed += 1
                    continue
                joined = (partner, partner_is_tip, d_bridge, path, float(chord[at]))
                break
            if joined is not None:
                break
        if joined is None:
            outcome[source] = codes["failed"]
            return

        partner, partner_is_tip, d_bridge, path, chord_length = joined
        arc = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        volume = math.pi * (d_bridge / 2.0) ** 2 * arc
        if max_bridge_volume is not None and total_volume + volume > max_bridge_volume:
            budget_spent = True
            outcome[source] = codes["over_budget"]
            return
        total_volume += volume
        rows = np.empty((4, len(path)))
        rows[:3] = path.T
        rows[3] = d_bridge
        rows[:3, 0] = source_xyz                   # bitwise copies, so the graph closes
        rows[:3, -1] = partner_xyz
        deg[source] += 1
        deg[partner] += 1
        outcome[source] = codes["placed"]
        if partner_is_tip:
            outcome[partner] = _OUTCOME["bridged_partner"]
            events["join_partner_tips"] += 1
        else:
            events["join_partner_interior"] += 1
        a, b = find(int(forest.component[source])), find(int(forest.component[partner]))
        if a != b:
            component_parent[max(a, b)] = min(a, b)
            events["join_components_joined"] += 1
        obstacles.add_bridge(path, d_bridge / 2.0)
        source_net, partner_net = int(net[source]), int(net[partner])
        bridges.append({
            "tip": (source_net, int(source - forest.base[source_net])),
            "partner": (partner_net, int(partner - forest.base[partner_net])),
            "source_kind": source_kind,
            "partner_kind": "tip" if partner_is_tip else "interior",
            "geometry": rows, "chord": chord_length, "arc": arc, "diameter": float(d_bridge),
            "volume": volume, "redraws": int(failed),
        })

    # the root phase: one Bernoulli per eligible root in (network, column)
    # order, the permutation, then the root bridges, each an obstacle for
    # every later bridge; nothing is drawn when no root is eligible
    if attach_roots and root_eligible.size:
        root_codes = {"placed": _OUTCOME["root_attached"], "failed": _OUTCOME["root_collision_failed"],
                      "over_budget": _OUTCOME["root_over_budget"]}
        root_picks = rng.random(root_eligible.size) < root_fraction
        selected_roots = root_eligible[root_picks]
        selected_roots = selected_roots[rng.permutation(selected_roots.size)]
        events["join_root_selected"] += int(selected_roots.size)
        for root in selected_roots.tolist():
            if budget_spent:
                outcome[root] = _OUTCOME["root_over_budget"]
                continue
            d_root = float(diam[root])
            root_xyz = xyz[root]
            upstream = forest.tangent[root]
            has_upstream = bool(np.all(np.isfinite(upstream)))

            # partners on other networks only, whatever the policy; no
            # start-junction zone (the root is the start of its polyline) and
            # no kin rule (no same-network partner is possible)
            found = partners.within(root_xyz, root_search * d_root)
            v = partners.tags[found]
            if v.size:
                keep = ((deg[v] == 1) | (deg[v] == 2)) & (net[v] != net[root])
                v = v[keep]
            if v.size:
                d_v = diam[v]
                offset = xyz[v] - root_xyz
                chord = np.linalg.norm(offset, axis=1)
                keep = chord > (d_root + d_v) / 2.0 + margin                     # already touching
                keep &= d_v >= root_partner_min_ratio * d_root                   # at least as thick
                if has_upstream:
                    behind = (offset @ upstream) < cone * chord
                    events["join_behind_skipped"] += int(np.count_nonzero(behind & keep))
                    keep &= ~behind
                v, d_v, chord = v[keep], d_v[keep], chord[keep]
            if v.size == 0:
                outcome[root] = _OUTCOME["root_no_partner"]
                continue
            # a side branch off an interior point before a tip-to-tip junction
            rank = np.where(deg[v] == 2, 0, 1)
            step = forest.polyline_spacing(root) or 0.2 * d_root
            place(root, "root", v, d_v, chord, rank, step, upstream if has_upstream else None,
                  lambda d_partner: d_root, root_codes)

    # the tip phase: one Bernoulli per eligible tip in (network, column)
    # order, then the processing permutation, then the walks
    tip_codes = {"placed": _OUTCOME["bridged_source"], "failed": _OUTCOME["collision_failed"],
                 "over_budget": _OUTCOME["over_budget"]}
    picks = rng.random(eligible.size) < fraction
    selected = eligible[picks]
    order = rng.permutation(selected.size)
    selected = selected[order]
    events["join_selected"] += int(selected.size)

    for tip in selected.tolist():
        if deg[tip] != 1:
            # consumed as a partner since it was drawn; it already has its outcome
            events["join_source_consumed"] += 1
            continue
        if budget_spent:
            outcome[tip] = _OUTCOME["over_budget"]
            continue
        d_tip = float(diam[tip])
        tip_xyz = xyz[tip]
        own = int(pid[tip])
        start_junction = int(forest.start_junction[own])
        tangent = forest.tangent[tip]
        has_tangent = bool(np.all(np.isfinite(tangent)))

        found = partners.within(tip_xyz, radius * d_tip)
        v = partners.tags[found]
        if v.size:
            keep = ((deg[v] == 1) | (deg[v] == 2)) & (v != tip) & (pid[v] != own)
            if policy == "cross":
                keep &= net[v] != net[tip]
            v = v[keep]
        if v.size:
            d_v = diam[v]
            offset = xyz[v] - tip_xyz
            chord = np.linalg.norm(offset, axis=1)
            keep = chord > (d_tip + d_v) / 2.0 + margin                      # already touching
            to_junction = np.linalg.norm(xyz[v] - xyz[start_junction], axis=1)
            keep &= to_junction >= KIN_REACH * ((d_v + diam[start_junction]) / 2.0 + margin)
            if has_tangent:
                behind = (offset @ tangent) < cone * chord
                events["join_behind_skipped"] += int(np.count_nonzero(behind & keep))
                keep &= ~behind
            v, d_v, chord = v[keep], d_v[keep], chord[keep]
        if v.size and policy == "any" and forest.separation is not None:
            same = np.flatnonzero(net[v] == net[tip])
            if same.size:
                separation = forest.separation[int(net[tip])]
                base = int(forest.base[int(net[tip])])
                hops = separation.within(tip - base, min_separation)
                keep = np.ones(v.size, dtype=bool)
                for at in same.tolist():
                    apart = separation.separation(hops, int(v[at]) - base)
                    if apart is not None and apart < min_separation:
                        keep[at] = False
                events["join_kin_skipped"] += int(np.count_nonzero(~keep))
                v, d_v, chord = v[keep], d_v[keep], chord[keep]
        if v.size == 0:
            outcome[tip] = _OUTCOME["no_partner"]
            continue
        rank = np.where(deg[v] == 1, 0, 1)
        step = forest.polyline_spacing(tip) or 0.2 * d_tip
        place(tip, "tip", v, d_v, chord, rank, step, tangent if has_tangent else None,
              lambda d_partner: min(d_tip, d_partner), tip_codes)

    # the outcome counters come from the final table, since a tip that found
    # no partner or failed every bridge may since have received one
    counts = np.bincount(outcome[tips], minlength=len(TIP_OUTCOMES)) if tips.size else np.zeros(len(TIP_OUTCOMES), int)
    for name, key in _OUTCOME_COUNTERS:
        events[key] += int(counts[_OUTCOME[name]])

    table = np.empty(tips.size, dtype=TIP_DTYPE)
    table["network"] = net[tips]
    table["column"] = tips - forest.base[net[tips]] if tips.size else tips
    table["outcome"] = outcome[tips]

    length_before = float(np.sum(np.linalg.norm(xyz[forest.edges[0]] - xyz[forest.edges[1]], axis=1))) \
        if forest.edges.shape[1] else 0.0
    length_after = length_before + sum(b["arc"] for b in bridges)
    root_rows = int(sum(counts[_OUTCOME[name]] for name in TIP_OUTCOMES if name == "root" or name.startswith("root_")))
    free_before = int(tips.size - root_rows - counts[_OUTCOME["cut_end"]])
    free_after = int(free_before - counts[_OUTCOME["bridged_source"]] - counts[_OUTCOME["bridged_partner"]])
    blunt_before = int(root_eligible.size)
    blunt_after = int(blunt_before - counts[_OUTCOME["root_attached"]])

    def per_length(before, after):
        return {"before": before / length_before if length_before > 0.0 else None,
                "after": after / length_after if length_after > 0.0 else None}

    summary = {
        "components": {"before": forest.components_before,
                       "after": forest.components_before - events["join_components_joined"]},
        "length": {"before": length_before, "after": length_after},
        "free_tips": {"before": free_before, "after": free_after},
        "free_tips_per_length": per_length(free_before, free_after),
        "bridge_volume": total_volume,
        "blunt_roots": {"before": blunt_before, "after": blunt_after},
        "blunt_roots_per_length": per_length(blunt_before, blunt_after),
        "free_ends": {"before": free_before + blunt_before, "after": free_after + blunt_after},
        "free_ends_per_length": per_length(free_before + blunt_before, free_after + blunt_after),
    }

    separator = np.full((4, 1), np.nan)
    pieces = []
    for bridge in bridges:
        if pieces:
            pieces.append(separator)
        pieces.append(bridge["geometry"])
    bridge_nodes = np.concatenate(pieces, axis=1) if pieces else np.empty((4, 0))

    result = {"bridges": bridges, "bridge_nodes": bridge_nodes, "tips": table, "events": events,
              "summary": summary}
    if merged:
        result["merged"] = _merge(forest, bridges)
    return result


def _merge(forest, bridges):
    """The inputs and the bridges as one archive with its graph."""
    n = forest.n
    pieces = [forest.nodes]
    network = [forest.net]
    tree = [forest.tree]
    canonical = [forest.canon]
    low, high = [forest.edges[0]], [forest.edges[1]]
    at = n
    for bridge in bridges:
        rows = bridge["geometry"]
        m = rows.shape[1]
        tip_net, tip_col = bridge["tip"]
        partner_net, partner_col = bridge["partner"]
        tip = int(forest.base[tip_net] + tip_col)
        partner = int(forest.base[partner_net] + partner_col)
        if pieces:
            pieces.append(np.full((4, 1), np.nan))
            network.append(np.array([-1], dtype=np.int32))
            tree.append(np.array([-1], dtype=np.int8))
            canonical.append(np.array([-1], dtype=np.int64))
            at += 1
        columns = np.arange(at, at + m, dtype=np.int64)
        vertex = columns.copy()
        vertex[0] = tip
        vertex[-1] = partner
        pieces.append(rows)
        network.append(np.full(m, BRIDGE_NETWORK, dtype=np.int32))
        tree.append(np.full(m, BRIDGE, dtype=np.int8))
        canonical.append(vertex)
        low.append(np.minimum(vertex[:-1], vertex[1:]))
        high.append(np.maximum(vertex[:-1], vertex[1:]))
        at += m
    nodes = np.concatenate(pieces, axis=1)
    canonical = np.concatenate(canonical)
    edges = _unique_sorted_edges(np.concatenate(low), np.concatenate(high), nodes.shape[1])
    return {"nodes": nodes, "edges": edges, "node_kind": graph.node_kind(nodes, edges, canonical),
            "network": np.concatenate(network), "tree": np.concatenate(tree)}
