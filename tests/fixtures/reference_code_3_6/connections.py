"""
Cross-connections: rungs between neighbouring capillaries.

A grown tree joins its vessels only where they branch, and anastomosis closes
it into loops only at its tips, while a capillary bed is a mesh: neighbouring
capillaries are joined side to side by short connecting vessels, so that a
capillary meets a junction every few tens of diameters along its length. This
module adds such connections ("rungs") to a finished centreline. It places
sites at random along every capillary vessel and joins each site to a nearby
point of another capillary vessel by a short vessel traced across the gap.

Vessels. The vessels are the segments of the trees: the paths between the
vertices of degree other than two of the graph that the tree-labelled columns
form on their own, bridges excluded (graph.segments). An archive polyline is
not a vessel, since the interpreter chains a stem into its first daughter
down to a tip. A vessel is a capillary, and so carries sites and receives
rungs, when the median of its vertex diameters is below `below` times d_min.

Sites. Along each capillary vessel, in the order of graph.segments, sites are
drawn as a Poisson process: gaps rng.exponential(spacing * d), d the vessel's
median vertex diameter, from its first vertex until its arc length is passed.
Each site is snapped to the nearest interior vertex that has degree two and
that no other polyline shares (the earlier one on a tie); a vessel without
such a vertex keeps the site at column -1. A permutation then fixes the order
in which the sites are processed, so that no vessel is favoured by coming
first. The spacing is at least one diameter. Rung ends along a vessel stand
at least a junction zone apart, KIN_REACH diameters on a vessel of even
calibre, so sites any closer would mostly be refused, and the floor keeps
their records within the capillary length in diameters, however small a
spacing is asked for.

Junction zones. The centres of a vessel are its vertices of degree three or
more, its vertices shared by two or more archive polylines (where a daughter's
polyline starts, a dropped daughter left a joint, or a bridge ends), and every
rung end placed on it so far; tips are not centres. A vertex P lies in the
zone of a centre J when |P - J| < KIN_REACH (r_P + r_J + margin), the overlap
zone of collisions.py and anastomosis.py, radii being half the vertex
diameters and margin the collision margin (0 without one). Rung ends lie
outside every zone of their vessels, so a rung never starts inside the
overlap of vessels that already meet, and the rungs along a vessel stay at
least one zone apart.

Partners. A site's partners are the interior vertices of degree two, shared by
no other polyline, of the other capillary vessels within `radius` times the
site's vertex diameter, excluding:
- partners fewer than `min_separation` tree segments away, counting the two
  vessels the site and the partner lie on (two vessels that share an end are
  two apart), measured along the trees without the bridges; two different
  trees are any distance apart, so the default of 2 excludes nothing beyond
  the site's own vessel, and 3 also excludes the vessels that meet it;
- partners already touching the site, |chord| <= (d_s + d_p) / 2 + margin;
- chords within `lateral_deg` of the reference, which is `axis` when one is
  given and otherwise the site's vessel tangent, x[i + 1] - x[i - 1] along the
  vessel: |chord . ref| > cos(lateral_deg) |chord|;
- partners inside a zone of their own vessel.

The rest are ranked by chord length, then by column.

The rung. Up to `max_candidates` partners are tried in rank order. A rung has
diameter min(d_s, d_p), vertex diameters, so it is never wider than either
vessel it joins. It is traced by tortuosity.bridge_path along the chord,
sampled at the median spacing of the site's vessel, with the deviation of a
persistent walk of `persistence` diameters pinned to zero at both ends. With a
collision margin it is checked against the network, the rungs placed before it
and itself, and redrawn up to `attempts` times per partner. It is appended as
a new polyline labelled BRIDGE, after a NaN separator, whose first and last
columns are bitwise copies of the site and the partner, so that a graph
rebuilt from coordinates joins it at both ends: both ends become junctions of
degree three, and the rung joins two components or closes a cycle. No
existing column changes, so roots, earlier bridges and every column index the
caller holds stay valid. Rungs are obstacles to later rungs, never partners.

The collision rule is that of collisions.py, as anastomosis applies it to its
bridges: a rung point Q and a network point P collide when their surfaces come
closer than the margin, |P - Q| < r_P + r_Q + margin, unless P lies on the
archive polyline of the site (or of the partner) and arc(P, J) + arc(Q, J) <
KIN_REACH (r_P + r_Q + margin) for J that end, arc lengths measured along P's
polyline and along the rung; two points of the rung collide when they are
that close while further apart along it than 2 (r_Q + r_Q + margin). P's
radius is its vertex's (the widest of its columns) where it decides a
collision and its own column's where it excuses one, so a rung passes both
this rule and describe.clearance, which reads each column's own diameter: a
rung never adds a clearance violation. The capillaries and the larger vessels
are indexed apart, as describe's clearance scan indexes each octave of radius
apart, so that the search around a capillary rung reaches only about a
capillary diameter plus the margin into the dense capillary index instead of
the root's radius.

Outcomes. Every site gets exactly one, so that

    rung_sites = rung_sites_near_junction + rung_sites_consumed
                 + rung_no_partner + rung_collision_failed + rung_bridges:

near_junction when it has no vertex or its vertex lies in a zone of its vessel
at its turn; consumed when its vertex became a rung end or an earlier site
took it; no_partner when no candidate is left after the exclusions;
collision_failed when every partner tried collided within the attempts; and a
bridge. rung_bridges_cross_tree counts the rungs joining two trees,
rung_kin_skipped and rung_angle_skipped the candidates excluded by the tree
separation and by the angle, and rung_bridge_redraws the collisions.

Draws come from the numpy.random.Generator passed in, in a fixed order: the
gaps, vessel by vessel, then the permutation, then the rung walks in the
permuted order, partner by partner and attempt by attempt. Nothing is drawn
from the global generators. Every length scales with the network, so a
network scaled by a power of two, with d_min, the margin and the tolerance,
gives the same rungs scaled. Python 3.9 compatible; numpy only.
"""
import math
import numbers

import numpy as np

import graph
from anastomosis import BRIDGE, BRIDGE_PERSISTENCE
from collisions import KIN_REACH
from guidance import _is_number
from spatial import make_index
from tortuosity import bridge_path

# The counters the stage reports, appended to main.EVENT_KEYS. Append-only.
EVENT_KEYS = ("rung_sites", "rung_sites_near_junction", "rung_sites_consumed", "rung_no_partner",
              "rung_collision_failed", "rung_bridges", "rung_bridges_cross_tree",
              "rung_kin_skipped", "rung_angle_skipped", "rung_bridge_redraws")

# The outcome of a site, as its record names it. Append-only.
RUNG_OUTCOMES = ("bridge", "near_junction", "consumed", "no_partner", "collision_failed")

_OUTCOME_EVENTS = {"bridge": "rung_bridges", "near_junction": "rung_sites_near_junction",
                   "consumed": "rung_sites_consumed", "no_partner": "rung_no_partner",
                   "collision_failed": "rung_collision_failed"}

_TINY = 1e-12


def _positive(value, name):
    if not _is_number(value) or not value > 0.0:
        raise ValueError(f"the rung {name} must be a positive number, got {value!r}")
    return float(value)


def _whole(value, name, least):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < least:
        raise ValueError(f"the rung {name} must be an integer of at least {least}, got {value!r}")
    return int(value)


def _unit_axis(axis):
    """The axis as a unit list of three floats, or None; a unit axis is kept as it is."""
    if axis is None:
        return None
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    if not isinstance(axis, (list, tuple)) or len(axis) != 3 or not all(_is_number(v) for v in axis):
        raise ValueError(f"the rung axis must be None or three finite numbers, got {axis!r}")
    vector = np.asarray(axis, dtype=float)
    # the norm of the vector scaled by its largest component cannot overflow
    largest = float(np.max(np.abs(vector)))
    scaled = vector / largest if largest > 0.0 else vector
    norm = largest * float(np.linalg.norm(scaled))
    if norm <= _TINY:
        raise ValueError("the rung axis must not be a zero vector")
    if abs(norm - 1.0) > _TINY:
        vector = scaled / np.linalg.norm(scaled)
    return [float(v) for v in vector]


def check_settings(*, d_min=None, below=2.0, spacing=20.0, radius=8.0, lateral_deg=60.0, axis=None,
                   min_separation=2, max_candidates=5, attempts=10, collision_margin=None, persistence=None,
                   index_kind="auto"):
    """
    Validates the settings of cross_connect without drawing anything.

    Args:
        d_min (float or None): checked when given; cross_connect needs it.
        below, spacing, radius, lateral_deg, axis, min_separation,
        max_candidates, attempts, collision_margin, persistence, index_kind:
        as for cross_connect.

    Returns:
        list or None: the axis as a unit list of three floats, or None.

    Raises:
        ValueError: for a d_min, below, radius or persistence that is not a
        positive number; a spacing that is not a number of at least 1;
        lateral_deg outside [0, 90]; an axis that is not three finite
        numbers or is a zero vector; min_separation or max_candidates that
        is not an integer of at least 1, and attempts that is not an integer
        of at least 0; a negative or non-finite margin; an index_kind other
        than "grid", "kdtree" or "auto".
    """
    if d_min is not None:
        _positive(d_min, "d_min")
    _positive(below, "class bound (below)")
    if not _is_number(spacing) or not spacing >= 1.0:
        raise ValueError(f"the rung spacing must be a number of at least 1 (diameters), got {spacing!r}")
    _positive(radius, "search radius")
    if not _is_number(lateral_deg) or not 0.0 <= lateral_deg <= 90.0:
        raise ValueError(f"the rung lateral angle must lie in [0, 90] degrees, got {lateral_deg!r}")
    unit = _unit_axis(axis)
    _whole(min_separation, "min_separation", 1)
    _whole(max_candidates, "max_candidates", 1)
    _whole(attempts, "attempts", 0)
    if collision_margin is not None and (not _is_number(collision_margin) or collision_margin < 0.0):
        raise ValueError(f"the collision margin must be a non-negative number, got {collision_margin!r}")
    if persistence is not None:
        _positive(persistence, "persistence")
    if not isinstance(index_kind, str) or index_kind not in ("auto", "grid", "kdtree"):
        raise ValueError(f"the rung index kind must be 'grid', 'kdtree' or 'auto', got {index_kind!r}")
    return unit


class _Obstacles:
    """
    The network's vertices and the rungs placed so far, as obstacles.

    A query searches out to r_q + r_max + margin, r_max the widest radius
    stored, so the capillaries and the larger vessels are indexed apart, as
    describe's clearance scan indexes each octave of radius apart: the search
    around a capillary rung then reaches about a capillary diameter plus the
    margin into the dense capillary index, and only into the sparse index of
    the larger vessels as far as their own widest one, rather than as far as
    the root into everything. The rungs go into an index of their own, so
    that adding them never rebuilds the network's. Network points carry their
    column as tag, rung points -1.
    """

    def __init__(self, kind, points, radii, tags, margin, bound):
        self.kind = kind
        self.margin = margin
        self.indexes = []
        for members in (np.flatnonzero(radii <= bound), np.flatnonzero(radii > bound)):
            if members.size == 0:
                continue
            cell = 2.0 * float(radii[members].max()) + margin
            # stored once and queried many times: build the structure straight away
            index = make_index(kind, cell_size=max(cell, 1e-9), rebuild_every=0)
            index.add(points[members], radii[members], tags[members])
            self.indexes.append(index)
        self.rungs = None

    def add_rung(self, points, radius):
        if self.rungs is None:
            self.rungs = make_index(self.kind, cell_size=max(2.0 * radius + self.margin, 1e-9))
            self.indexes.append(self.rungs)
        self.rungs.add(points, radius, -1)

    def query(self, points, radius):
        """(qi, tags, radii) of every stored point closer to points[qi] than its radius + radius + margin."""
        found = []
        for index in self.indexes:
            qi, si = index.query(points, radius, self.margin)
            if qi.size:
                found.append((qi, index.tags[si], index.radii[si]))
        if not found:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0)
        return tuple(np.concatenate(parts) for parts in zip(*found))


class _Vessels:
    """
    What the stage needs of the network, as flat arrays over its columns,
    and the capillary vessels with their arc lengths.
    """

    def __init__(self, nodes, tree, tol, d_min, below):
        n = nodes.shape[1]
        self.nodes = nodes
        canonical = graph.canonical_columns(nodes, tol)
        finite = canonical >= 0
        self.degree = graph.degree(graph.edges_from_nodes(nodes, canonical=canonical), n)
        self.diameter = graph.vertex_diameter(nodes, canonical)
        self.vertices = np.flatnonzero(canonical == np.arange(n))
        self.polyline = graph.polyline_of_column(nodes)
        pairs = np.unique(self.polyline[finite] * n + canonical[finite])
        self.shared = np.bincount(pairs % n, minlength=n) >= 2

        # arc-length position along its polyline, and own radius, of every
        # column, computed as describe computes them for its clearance rules
        columns = np.flatnonzero(finite)
        points = nodes[:3, columns].T
        polyline = self.polyline[columns]
        first = np.ones(columns.size, dtype=bool)
        first[1:] = polyline[1:] != polyline[:-1]
        step = np.zeros(columns.size)
        inside = ~first[1:]
        step[1:][inside] = np.linalg.norm(points[1:][inside] - points[:-1][inside], axis=1)
        cumulative = np.cumsum(step)
        self.arc = np.zeros(n)
        if columns.size:
            self.arc[columns] = cumulative - cumulative[np.flatnonzero(first)][polyline]
        self.own_radius = np.maximum(np.nan_to_num(nodes[3] / 2.0, nan=0.0), 0.0)

        # the vessels: segments of the graph of the tree columns alone
        is_tree = finite & (tree >= 0) & (tree != BRIDGE)
        tree_edges = graph.edges_from_nodes(nodes, canonical=np.where(is_tree, canonical, -1))
        self.segments = graph.segments(nodes, tree_edges)
        count = len(self.segments)
        sizes = np.array([len(path) for path in self.segments], dtype=np.int64)
        flat = np.concatenate(self.segments) if count else np.zeros(0, dtype=np.int64)
        owner = np.repeat(np.arange(count), sizes)
        starts = np.cumsum(sizes) - sizes
        self.starts, self.sizes, self.flat = starts, sizes, flat

        # median vertex diameter and arc length per vessel, in one pass over all of them
        values = self.diameter[flat]
        ordered = values[np.lexsort((values, owner))]
        self.median = (ordered[starts + (sizes - 1) // 2] + ordered[starts + sizes // 2]) / 2.0
        xyz = nodes[:3, flat]
        edge = np.zeros(flat.size)
        if flat.size > 1:
            edge[1:] = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=0)
        edge[starts] = 0.0
        self.edge = edge
        travelled = np.cumsum(edge)
        self.flat_arc = travelled - travelled[starts][owner]
        self.length = self.flat_arc[starts + sizes - 1]
        self.capillary = (self.median < below * d_min) & (self.length > 0.0)

        # interior vertices of capillary vessels, which may carry a site or
        # receive a rung while they have degree two and are shared by nothing
        # (and have a diameter, which a rung's walk needs)
        place = np.arange(flat.size) - starts[owner]
        interior = (place > 0) & (place < sizes[owner] - 1) & self.capillary[owner]
        self.vessel_of = np.full(n, -1, dtype=np.int64)
        self.vessel_of[flat[interior]] = owner[interior]
        self.place = np.zeros(n, dtype=np.int64)
        self.place[flat[interior]] = place[interior]
        self.free = np.zeros(flat.size, dtype=bool)
        self.free[interior] = ((self.degree[flat[interior]] == 2) & ~self.shared[flat[interior]]
                               & (self.diameter[flat[interior]] > 0.0))
        self.candidates = flat[self.free]
        self._centres = {}
        self._spacing = {}
        self._neighbours = None

    def path(self, k):
        return self.flat[self.starts[k]:self.starts[k] + self.sizes[k]]

    def snap(self, k, arcs):
        """The free interior vertex of vessel k nearest each arc (the earlier on a tie), or -1 for none."""
        a, b = self.starts[k], self.starts[k] + self.sizes[k]
        free = np.flatnonzero(self.free[a:b])
        if free.size == 0:
            return np.full(len(arcs), -1, dtype=np.int64)
        along = self.flat_arc[a:b][free]
        after = np.clip(np.searchsorted(along, arcs), 1, max(free.size - 1, 1))
        if free.size == 1:
            nearest = np.zeros(len(arcs), dtype=np.int64)
        else:
            before = after - 1
            nearest = np.where(np.abs(along[after] - arcs) < np.abs(along[before] - arcs), after, before)
        return self.flat[a + free[nearest]]

    def centres(self, k):
        """The zone centres of vessel k so far, as a list of columns."""
        if k not in self._centres:
            path = self.path(k)
            self._centres[k] = path[(self.degree[path] >= 3) | self.shared[path]].tolist()
        return self._centres[k]

    def in_zone(self, v, k, margin):
        """Whether vertex v lies in the zone of a centre of vessel k other than itself."""
        centres = np.array([c for c in self.centres(k) if c != v], dtype=np.int64)
        if centres.size == 0:
            return False
        distance = np.linalg.norm(self.nodes[:3, centres] - self.nodes[:3, v, None], axis=0)
        reach = KIN_REACH * (self.diameter[v] / 2.0 + self.diameter[centres] / 2.0 + margin)
        return bool(np.any(distance < reach))

    def tangent(self, v):
        """Unit tangent of v's vessel at v, from its two neighbours along the vessel."""
        path = self.path(self.vessel_of[v])
        i = self.place[v]
        delta = self.nodes[:3, path[i + 1]] - self.nodes[:3, path[i - 1]]
        norm = np.linalg.norm(delta)
        if norm <= 0.0:
            delta = self.nodes[:3, path[i + 1]] - self.nodes[:3, v]
            norm = np.linalg.norm(delta)
        return delta / norm

    def spacing(self, k):
        """Median spacing of the vertices of vessel k, or None."""
        if k not in self._spacing:
            steps = self.edge[self.starts[k] + 1:self.starts[k] + self.sizes[k]]
            steps = steps[steps > 0.0]
            self._spacing[k] = float(np.median(steps)) if steps.size else None
        return self._spacing[k]

    def kin_ends(self, k, limit):
        """
        The segment ends at most `limit` - 3 segments beyond the two ends of
        vessel k along the trees: a vessel with an end among them lies fewer
        than `limit` segments from k, both counted.
        """
        if self._neighbours is None:
            self._neighbours = {}
            for path in self.segments:
                a, b = int(path[0]), int(path[-1])
                self._neighbours.setdefault(a, []).append(b)
                self._neighbours.setdefault(b, []).append(a)
        path = self.path(k)
        reached = {int(path[0]), int(path[-1])}
        frontier = list(reached)
        for _ in range(limit - 3):
            following = []
            for v in frontier:
                for w in self._neighbours.get(v, ()):
                    if w not in reached:
                        reached.add(w)
                        following.append(w)
            frontier = following
        return reached


def cross_connect(nodes, tree, rng, *, d_min, below=2.0, spacing=20.0, radius=8.0, lateral_deg=60.0,
                  axis=None, min_separation=2, max_candidates=5, persistence=None, collision_margin=None,
                  attempts=10, tol=graph.DEFAULT_TOL, events=None, index_kind="auto", sites=None):
    """
    Joins neighbouring capillaries side to side by rungs.

    Args:
        nodes (ndarray): (4, N) centreline with NaN separators, in grammar units.
        tree (ndarray or None): (N,) int8 tree label per column, -1 at
            separators and BRIDGE on bridges; None labels every column 0.
        rng (numpy.random.Generator): the stream of the stage.
        d_min (float): the unit of `below`.
        below (float): a vessel is a capillary when its median vertex
            diameter is below this many d_min.
        spacing (float): mean gap between sites along a capillary, in its
            median diameters; at least 1.
        radius (float): partner search radius, in site diameters.
        lateral_deg (float): smallest angle, in degrees, between a rung's
            chord and the reference: `axis`, else the site's vessel tangent.
        axis (sequence or None): a 3-vector the chords must stand off from;
            None uses the tangent.
        min_separation (int): smallest number of tree segments between the
            site's vessel and the partner's, both counted; 2 allows vessels
            that meet, 3 excludes them.
        max_candidates (int): partners tried per site before giving up.
        persistence (float or None): persistence of the rung's walk in
            diameters; None uses anastomosis.BRIDGE_PERSISTENCE.
        collision_margin (float or None): when given, every rung is checked
            against the network, the earlier rungs and itself at this margin,
            redrawn or abandoned on a collision, and the zones and the touch
            test include it; None places rungs unchecked.
        attempts (int): redraws of a colliding rung per partner.
        tol (float): coincidence tolerance for the graph.
        events (dict or None): counters incremented in place (see EVENT_KEYS).
        index_kind (str): spatial index for the partner search and the
            collision check (spatial.make_index).
        sites (list or None): when given, one record per site is appended to
            it, in drawing order: "unit" the index of its vessel among the
            tree segments (graph.segments of the tree columns' edges), "arc"
            the drawn position along it from its first vertex, "column" the
            vertex it snapped to (-1 for none) and "outcome" one of
            RUNG_OUTCOMES.

    Returns:
        tuple: (nodes, tree, rungs) with the rungs appended as polylines, the
        labels extended (BRIDGE on rung columns) and `rungs` a list of dicts
        describing each: "site" and "partner" the joined columns,
        "site_tree" and "partner_tree" their labels, "chord_um" and "arc_um"
        the straight and traced lengths, "diameter_um", "points", "redraws"
        (failed attempts on the partner joined) and "angle_deg", the chord's
        acute angle to the reference.

    Raises:
        ValueError: for a missing d_min, anything check_settings refuses and a
        `tree` that does not label every column.
    """
    if d_min is None:
        raise ValueError("cross-connection needs d_min, the unit of its class bound")
    axis = check_settings(d_min=d_min, below=below, spacing=spacing, radius=radius, lateral_deg=lateral_deg,
                          axis=axis, min_separation=min_separation, max_candidates=max_candidates,
                          attempts=attempts, collision_margin=collision_margin, persistence=persistence,
                          index_kind=index_kind)
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError("nodes must be a (4, N) array of x, y, z and diameter columns")
    if tree is None:
        tree = np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8)
    tree = np.asarray(tree).astype(np.int8)
    if tree.shape != (nodes.shape[1],):
        raise ValueError("tree must hold one label per column of nodes")
    if events is None:
        events = {}
    for key in EVENT_KEYS:
        events.setdefault(key, 0)
    if persistence is None:
        persistence = BRIDGE_PERSISTENCE
    d_min = float(d_min)
    margin = float(collision_margin) if collision_margin is not None else 0.0
    cos_lateral = math.cos(math.radians(lateral_deg))
    reference = None if axis is None else np.asarray(axis)
    min_separation = int(min_separation)

    net = _Vessels(nodes, tree, tol, d_min, float(below))
    records = []
    for k in np.flatnonzero(net.capillary).tolist():
        scale = spacing * float(net.median[k])
        length = float(net.length[k])
        arcs = []
        position = rng.exponential(scale)
        while position <= length:
            arcs.append(position)
            position += rng.exponential(scale)
        if arcs:
            for arc, column in zip(arcs, net.snap(k, np.array(arcs)).tolist()):
                records.append({"unit": k, "arc": float(arc), "column": int(column), "outcome": None})
    events["rung_sites"] += len(records)
    order = rng.permutation(len(records))

    partners = None
    obstacles = None
    degree = net.degree
    taken = set()
    pieces = []
    rungs = []
    for o in order.tolist():
        record = records[o]
        k, site = record["unit"], record["column"]
        if site < 0:
            outcome = "near_junction"
        elif site in taken or degree[site] != 2:
            outcome = "consumed"
        else:
            taken.add(site)
            outcome = "near_junction" if net.in_zone(site, k, margin) else None
        if outcome is not None:
            record["outcome"] = outcome
            events[_OUTCOME_EVENTS[outcome]] += 1
            continue

        if partners is None:
            partners = make_index(index_kind, cell_size=max(radius * d_min, 1e-9), rebuild_every=0)
            partners.add(nodes[:3, net.candidates].T, net.diameter[net.candidates] / 2.0, net.candidates)
        site_xyz = nodes[:3, site]
        d_site = float(net.diameter[site])
        ref = reference if reference is not None else net.tangent(site)
        found = partners.nearest_within(site_xyz, radius * d_site)
        v = partners.tags[found]
        keep = (v != site) & (degree[v] == 2) & (net.vessel_of[v] != k)
        if min_separation > 2 and keep.any():
            near = net.kin_ends(k, min_separation)
            ends = [(int(net.flat[net.starts[u]]), int(net.flat[net.starts[u] + net.sizes[u] - 1]))
                    for u in net.vessel_of[v[keep]].tolist()]
            kin = np.zeros(v.size, dtype=bool)
            kin[np.flatnonzero(keep)] = [a in near or b in near for a, b in ends]
            events["rung_kin_skipped"] += int(np.count_nonzero(kin))
            keep &= ~kin
        chord_vector = partners.points[found] - site_xyz
        chord = np.linalg.norm(chord_vector, axis=1)
        d_partner = net.diameter[v]
        keep &= chord > (d_site + d_partner) / 2.0 + margin
        steep = np.abs(chord_vector @ ref) > cos_lateral * chord
        events["rung_angle_skipped"] += int(np.count_nonzero(keep & steep))
        keep &= ~steep
        ranked = np.flatnonzero(keep)
        ranked = ranked[np.lexsort((v[ranked], chord[ranked]))]
        chosen = []
        for j in ranked.tolist():
            if not net.in_zone(int(v[j]), int(net.vessel_of[v[j]]), margin):
                chosen.append(j)
                if len(chosen) == max_candidates:
                    break
        if not chosen:
            record["outcome"] = "no_partner"
            events["rung_no_partner"] += 1
            continue

        if collision_margin is not None and obstacles is None:
            obstacles = _Obstacles(index_kind, nodes[:3, net.vertices].T,
                                   np.nan_to_num(net.diameter[net.vertices] / 2.0, nan=0.0), net.vertices, margin,
                                   below * d_min / 2.0)
        step = net.spacing(k) or 0.2 * d_site
        joined = None
        for j in chosen:
            partner = int(v[j])
            partner_xyz = nodes[:3, partner]
            d_rung = min(d_site, float(d_partner[j]))
            for redraw in range(attempts + 1):
                path = bridge_path(site_xyz, partner_xyz, step, rng=rng, persistence=persistence, diameter=d_rung)
                if collision_margin is None or len(path) <= 2:
                    break
                if not _collides(net, obstacles, path, d_rung / 2.0, site, partner, margin):
                    break
                events["rung_bridge_redraws"] += 1
                path = None
            if path is not None:
                joined = (j, partner, d_rung, path, redraw)
                break
        if joined is None:
            record["outcome"] = "collision_failed"
            events["rung_collision_failed"] += 1
            continue

        j, partner, d_rung, path, redraws = joined
        rows = np.empty((4, len(path)))
        rows[:3] = path.T
        rows[3] = d_rung
        rows[:3, 0] = site_xyz                 # bitwise copies, so the graph joins the rung
        rows[:3, -1] = partner_xyz
        pieces.append(rows)
        if obstacles is not None and len(path) > 2:
            obstacles.add_rung(path[1:-1], d_rung / 2.0)
        degree[site] += 1
        degree[partner] += 1
        net.centres(k).append(site)
        net.centres(int(net.vessel_of[partner])).append(partner)
        record["outcome"] = "bridge"
        events["rung_bridges"] += 1
        if tree[site] != tree[partner]:
            events["rung_bridges_cross_tree"] += 1
        cosine = min(1.0, abs(float(chord_vector[j] @ ref)) / float(chord[j]))
        rungs.append({
            "site": int(site), "partner": partner,
            "site_tree": int(tree[site]), "partner_tree": int(tree[partner]),
            "chord_um": float(chord[j]),
            "arc_um": float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
            "diameter_um": float(d_rung), "points": int(len(path)), "redraws": int(redraws),
            "angle_deg": math.degrees(math.acos(cosine)),
        })

    if sites is not None:
        sites.extend(records)
    if not pieces:
        return nodes, tree, rungs
    separator = np.full((4, 1), np.nan)
    blocks = [nodes]
    labels = [tree]
    for rows in pieces:
        blocks.extend([separator, rows])
        labels.extend([np.array([-1], dtype=np.int8), np.full(rows.shape[1], BRIDGE, dtype=np.int8)])
    return np.concatenate(blocks, axis=1), np.concatenate(labels), rungs


def _collides(net, obstacles, path, radius, site, partner, margin):
    """
    Whether a rung's interior points collide with the network, with the rungs
    before it or with the rung itself, under the rule of the module docstring.
    """
    interior = path[1:-1]
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    qi, tags, radii = obstacles.query(interior, radius)
    if qi.size:
        excused = np.zeros(qi.size, dtype=bool)
        on_network = tags >= 0
        for end, from_end in ((site, along[1:-1]), (partner, along[-1] - along[1:-1])):
            kin = np.flatnonzero(on_network)
            kin = kin[net.polyline[tags[kin]] == net.polyline[end]]
            if kin.size:
                # distances along the vessel and along the rung to the joined point
                together = np.abs(net.arc[tags[kin]] - net.arc[end]) + from_end[qi[kin]]
                reach = KIN_REACH * (net.own_radius[tags[kin]] + radius + margin)
                excused[kin[together < reach]] = True
        if not excused.all():
            return True
    # the rung against itself
    arc = along[1:-1]
    gap = np.abs(arc[:, None] - arc[None, :])
    close = np.linalg.norm(interior[:, None, :] - interior[None, :, :], axis=2) < 2.0 * radius + margin
    return bool(np.any(close & (gap > 2.0 * (2.0 * radius + margin))))
