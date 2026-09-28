"""
Anastomosis: closing a grown tree into a network by bridging tips.

A tree drawn by the grammar ends every terminal branch in a free tip. Real
capillary beds are closed networks: arterioles feed capillaries that drain
into venules, and almost no vessel ends blind. This module takes a finished
centreline, picks a seeded random fraction of its tips, and joins each to a
partner found within a search radius: another tip, or an interior point of a
segment that is not the tip's own. The bridge is traced by the pinned
persistent walk of tortuosity.bridge_path, checked against the network when a
collision margin is given (the rules of collisions.py, with the bridge's two
end vertices as its junctions and distances measured along the vessels), and appended to the archive as a new polyline
whose first and last columns are bitwise copies of the two joined points, so
that a graph rebuilt from coordinates joins it at both ends and gains a cycle
(or, when the two ends lie in different components, loses a component).

Partner search. Tips are graph vertices of degree one other than the roots
(the first point of each tree) and other than stubs: a branch terminated
within a step or two of its junction leaves a tip inside the junction's own
overlap zone, and a bridge to it would run through the parent vessel, so such
tips are neither sources nor partners and are counted. A tip's partners are
searched within `radius` times the tip's own diameter and ranked tips first,
interior points second, nearest first within a rank. Excluded as partners:
points of the tip's own polyline; junctions (degree three or more) and points
inside a junction's overlap zone; the roots; points inside the overlap zone of
the junction the tip's polyline leaves, which are its parent and sibling near
their common branch point; and points already overlapping the tip. The
overlap zone of a junction J for a point P of radius r_P is
|P - J| < KIN_REACH (r_P + r_J + margin), as in collisions.py. In `arteriovenous` mode, where two trees have been grown from opposite
faces of the volume, partners in the other tree outrank partners in the same
tree, so arterial tips join venous tips or venules first and only fall back to
arterial neighbours when no venous partner is in reach. A tip consumed as a
partner is no longer a tip and is skipped if its own turn comes later; an
interior point that has received a bridge is a junction and receives no
second one.

Bridge diameter is min(d_tip, d_partner): the bridge is the finest vessel of
the three that meet, as a capillary is. A Murray-consistent alternative would
resize the partner's downstream segment, which this module does not do. The
bridge is sampled at the spacing of the tip's own polyline.

Every skipped or failed join is counted in `events`: a selected tip that had
already been consumed as a partner, a tip with no partner in reach, and a tip
whose every candidate bridge collided within the attempt budget.

Draws come from the `numpy.random.Generator` passed in, in a fixed order:
the selection of tips first, then the bridge walks in tip order, so a run is
reproducible from its seed. Python 3.9 compatible.
"""
import numpy as np

import graph
from collisions import KIN_REACH
from spatial import make_index
from tortuosity import bridge_path

BRIDGE = 2                     # value of the `tree` label on bridge columns
EVENT_KEYS = ("anastomosis_tips", "anastomosis_tips_inside_junction", "anastomosis_selected",
              "anastomosis_bridges", "anastomosis_bridges_arteriovenous", "anastomosis_bridges_same_tree",
              "anastomosis_partner_tips", "anastomosis_partner_interior",
              "anastomosis_source_consumed", "anastomosis_no_partner",
              "anastomosis_collision_failed", "anastomosis_bridge_redraws")


def _polyline_spacing(nodes, columns):
    """Median spacing of consecutive points of one polyline, or None if it has fewer than two."""
    if len(columns) < 2:
        return None
    steps = np.linalg.norm(np.diff(nodes[:3, columns], axis=1), axis=0)
    steps = steps[steps > 0.0]
    return float(np.median(steps)) if steps.size else None


def _tip_tangent(nodes, columns, tip):
    """Unit direction a tip points in: away from the last distinct point of its polyline before it."""
    columns = [int(c) for c in columns]
    at = columns.index(tip)
    inward = columns[at - 1::-1] if at > 0 else columns[at + 1:]
    here = nodes[:3, tip]
    for other in inward:
        delta = here - nodes[:3, other]
        norm = np.linalg.norm(delta)
        if norm > 0.0:
            return delta / norm
    return None


class _Network:
    """Per-column bookkeeping that grows as bridges are appended."""

    def __init__(self, nodes, tree, tol):
        self.nodes = nodes
        self.n = nodes.shape[1]
        built = graph.build(nodes, tol)
        self.canonical = built["canonical"]
        deg = built["degree"]
        # a vertex is represented by the column that is its own canonical column
        representatives = np.flatnonzero(self.canonical == np.arange(self.n))
        self.degree = {int(c): int(deg[c]) for c in representatives}
        poly = graph.polyline_of_column(nodes)
        self.polyline = {int(c): int(poly[c]) for c in np.flatnonzero(poly >= 0)}
        self.tree = {int(c): int(tree[c]) for c in range(self.n) if tree[c] >= 0}
        diameter = graph.vertex_diameter(nodes, self.canonical)
        self.diameter = {int(c): float(diameter[c]) for c in representatives}
        self.columns_of = {pid: np.asarray(cols) for pid, cols in enumerate(graph.polylines(nodes))}
        # the junctions a polyline passes through: its start vertex and every
        # vertex of degree three or more along it
        self.junctions_of = {}
        for pid, cols in self.columns_of.items():
            vertices = self.canonical[cols]
            branching = vertices[deg[vertices] >= 3]
            self.junctions_of[pid] = np.unique(np.concatenate([[vertices[0]], branching]))
        self.extra = []                        # (4, m) arrays of bridge polylines
        self.extra_label = []
        self.next_column = self.n
        self._arcs = {}

    def inside_junction(self, column, margin):
        """Whether a point lies in the overlap zone of a junction of its own polyline."""
        junctions = self.junctions_of[self.polyline[column]]
        junctions = junctions[junctions != column]
        if junctions.size == 0:
            return False
        xyz = self.xyz(column)
        distance = np.array([np.linalg.norm(self.xyz(int(j)) - xyz) for j in junctions])
        reach = KIN_REACH * (self.diameter[column] / 2.0 + np.array([self.diameter[int(j)] for j in junctions]) / 2.0
                             + margin)
        return bool(np.any(distance < reach))

    def arc(self, column):
        """Arc-length position of a column along its polyline."""
        pid = self.polyline[column]
        if pid not in self._arcs:
            cols = self.columns_of[pid]
            xyz = np.array([self.xyz(int(c)) for c in cols]) if cols[-1] >= self.n else self.nodes[:3, cols].T
            steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            self._arcs[pid] = dict(zip((int(c) for c in cols), np.concatenate([[0.0], np.cumsum(steps)])))
        return self._arcs[pid][int(column)]

    def xyz(self, column):
        if column < self.n:
            return self.nodes[:3, column]
        offset = self.n
        for rows in self.extra:
            offset += 1                        # the separator column
            if column < offset + rows.shape[1]:
                return rows[:3, column - offset]
            offset += rows.shape[1]
        raise IndexError(f"column {column} does not exist")

    def append(self, rows, label):
        """Appends a polyline after a separator; returns its column indices."""
        columns = np.arange(self.next_column + 1, self.next_column + 1 + rows.shape[1])
        self.next_column += 1 + rows.shape[1]
        self.extra.append(rows)
        self.extra_label.append(np.full(rows.shape[1], label, dtype=np.int8))
        pid = len(self.columns_of)
        self.columns_of[pid] = columns
        self.junctions_of[pid] = np.array([columns[0], columns[-1]], dtype=np.int64)
        for c in columns:
            self.polyline[int(c)] = pid
            self.tree[int(c)] = label
            self.diameter[int(c)] = float(rows[3, 0])
        return columns

    def assemble(self, tree):
        if not self.extra:
            return self.nodes, tree
        separator = np.full((4, 1), np.nan)
        pieces = [self.nodes]
        labels = [tree]
        for rows, label in zip(self.extra, self.extra_label):
            pieces.extend([separator, rows])
            labels.extend([np.array([-1], dtype=np.int8), label])
        return np.concatenate(pieces, axis=1), np.concatenate(labels)


def _bridge_collides(net, index, path, radius, tip, partner, margin):
    """
    Whether a bridge's interior points collide with the network or with the
    bridge itself, under the rules of collisions.py: a stored point P and a
    bridge point Q collide when |P - Q| < r_P + r_Q + margin unless P lies on
    the polyline of the tip (or of the partner) and arc(P, J) + arc(Q, J) <
    KIN_REACH (r_P + r_Q + margin) for J that end vertex, arc lengths measured
    along P's polyline and along the bridge; and two points of the bridge
    collide when their arc-length separation exceeds 2 (r_Q + r_Q + margin)
    while they are within r_Q + r_Q + margin.
    """
    interior = path[1:-1]
    radii = np.full(len(interior), radius)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    bridge_arc = np.concatenate([[0.0], np.cumsum(steps)])
    from_tip = bridge_arc[1:-1]
    from_partner = bridge_arc[-1] - bridge_arc[1:-1]
    qi, si = index.query(interior, radii, margin)
    if len(qi):
        columns = index.tags[si]
        threshold = index.radii[si] + radius + margin
        excused = np.zeros(len(qi), dtype=bool)
        for end, along_bridge in ((tip, from_tip), (partner, from_partner)):
            pid = net.polyline[end]
            kin = np.array([net.polyline[int(c)] == pid for c in columns])
            if kin.any():
                # distances measured along the vessel to the joined point
                to_end = np.array([abs(net.arc(int(c)) - net.arc(end)) for c in columns[kin]])
                together = to_end + along_bridge[qi[kin]]
                excused[np.flatnonzero(kin)[together < KIN_REACH * threshold[kin]]] = True
        if np.any(~excused):
            return True
    # the bridge against itself
    arc = from_tip
    gap = np.abs(arc[:, None] - arc[None, :])
    close = np.linalg.norm(interior[:, None, :] - interior[None, :, :], axis=2) < 2.0 * radius + margin
    return bool(np.any(close & (gap > 2.0 * (2.0 * radius + margin))))


def anastomose(nodes, rng, fraction, radius, mode="any", tree=None, persistence=None,
               collision_margin=None, events=None, tol=graph.DEFAULT_TOL, attempts=10,
               max_candidates=5, index_kind="auto"):
    """
    Bridges a seeded fraction of tips to nearby partners.

    Args:
        nodes (ndarray): (4, N) centreline with NaN separators, in grammar units.
        rng (numpy.random.Generator): the anastomosis stream.
        fraction (float): share of tips selected as bridge sources, in [0, 1].
        radius (float): search radius in multiples of the source tip's diameter.
        mode (str): "any", or "arteriovenous" to prefer partners in the other tree.
        tree (ndarray or None): (N,) int8 tree label per column, -1 at separators;
            None labels every column 0.
        persistence (float or None): persistence of the bridge walk in diameters;
            None traces a smooth bridge without random deviation.
        collision_margin (float or None): when given, every bridge is checked
            against the network and the bridges before it with the rules of
            collisions.py at this margin, and redrawn or abandoned on a
            collision; None places bridges unchecked.
        events (dict or None): counters incremented in place (see EVENT_KEYS).
        tol (float): coincidence tolerance for the graph.
        attempts (int): redraws of a colliding bridge per candidate partner.
        max_candidates (int): partners tried per tip before giving up.
        index_kind (str): spatial index used for the partner search.

    Returns:
        tuple: (nodes, tree, bridges) with the bridges appended as polylines,
        the tree labels extended (BRIDGE on bridge columns) and `bridges` a
        list of dicts describing each join.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"the anastomosis fraction must lie in [0, 1], got {fraction!r}")
    if not radius > 0.0:
        raise ValueError("the anastomosis radius must be positive")
    if mode not in ("any", "arteriovenous"):
        raise ValueError(f"mode must be 'any' or 'arteriovenous', got {mode!r}")
    nodes = np.asarray(nodes, dtype=float)
    if tree is None:
        tree = np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8)
    tree = np.asarray(tree).astype(np.int8)
    if tree.shape != (nodes.shape[1],):
        raise ValueError("tree must hold one label per column of nodes")
    if events is None:
        events = {}
    for key in EVENT_KEYS:
        events.setdefault(key, 0)
    if collision_margin is not None and collision_margin < 0.0:
        raise ValueError("the collision margin cannot be negative")
    margin = float(collision_margin) if collision_margin is not None else 0.0

    net = _Network(nodes, tree, tol)
    vertices = np.array(sorted(net.degree), dtype=np.int64)
    roots = set()
    for label in np.unique(tree[tree >= 0]):
        first = np.flatnonzero((tree == label) & (net.canonical >= 0))
        if first.size:
            roots.add(int(net.canonical[first[0]]))
    tips = [int(v) for v in vertices if net.degree[int(v)] == 1 and int(v) not in roots]
    events["anastomosis_tips"] += len(tips)
    stubs = {t for t in tips if net.inside_junction(t, margin)}
    events["anastomosis_tips_inside_junction"] += len(stubs)
    tips = [t for t in tips if t not in stubs]
    picks = rng.random(len(tips)) < fraction
    selected = [t for t, pick in zip(tips, picks) if pick]
    events["anastomosis_selected"] += len(selected)

    reach = max(net.diameter.values()) if net.diameter else 1.0
    index = make_index(index_kind, cell_size=max(reach, 1e-9))
    if vertices.size:
        index.add(nodes[:3, vertices].T, np.array([net.diameter[int(v)] for v in vertices]) / 2.0,
                  vertices)

    bridges = []
    for tip in selected:
        if net.degree[tip] != 1:
            events["anastomosis_source_consumed"] += 1
            continue
        d_tip = net.diameter[tip]
        pid = net.polyline[tip]
        own = net.columns_of[pid]
        junction = int(net.canonical[own[0]])
        j_xyz = nodes[:3, junction]
        tip_xyz = nodes[:3, tip]
        tip_tree = net.tree[tip]
        candidates = []
        for stored in index.nearest_within(tip_xyz, radius * d_tip):
            v = int(index.tags[stored])
            if v == tip or v in roots or v in stubs or net.degree.get(v, 0) not in (1, 2):
                continue
            if net.polyline[v] == pid:
                continue
            v_xyz = index.points[stored]
            d_v = float(index.radii[stored]) * 2.0
            if np.linalg.norm(v_xyz - j_xyz) < KIN_REACH * ((d_v + net.diameter[junction]) / 2.0 + margin):
                continue                       # parent or sibling at the branch point
            chord = float(np.linalg.norm(v_xyz - tip_xyz))
            if chord <= (d_tip + d_v) / 2.0 + margin:
                continue                       # already touching
            if net.degree[v] == 2 and net.inside_junction(v, margin):
                continue                       # an interior point at a junction
            rank = 0 if net.degree[v] == 1 else 1
            if mode == "arteriovenous":
                other_tree = net.tree[v] != tip_tree and net.tree[v] != BRIDGE
                rank += 0 if other_tree else 2
            candidates.append((rank, chord, v, d_v))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        if not candidates:
            events["anastomosis_no_partner"] += 1
            continue

        step = _polyline_spacing(nodes, own) or 0.2 * d_tip
        tangent = _tip_tangent(nodes, own, tip)
        joined = None
        for rank, chord, partner, d_partner in candidates[:max_candidates]:
            d_bridge = min(d_tip, d_partner)
            partner_xyz = net.xyz(partner)
            for redraw in range(attempts + 1):
                path = bridge_path(tip_xyz, partner_xyz, step, start_tangent=tangent, rng=rng,
                                   persistence=persistence, diameter=d_bridge)
                if collision_margin is None or len(path) <= 2:
                    break
                if not _bridge_collides(net, index, path, d_bridge / 2.0, tip, partner, margin):
                    break
                events["anastomosis_bridge_redraws"] += 1
                path = None
            if path is not None:
                joined = (rank, chord, partner, d_partner, d_bridge, path, redraw)
                break
        if joined is None:
            events["anastomosis_collision_failed"] += 1
            continue

        rank, chord, partner, d_partner, d_bridge, path, redraws = joined
        rows = np.empty((4, len(path)))
        rows[:3] = path.T
        rows[3] = d_bridge
        rows[:3, 0] = tip_xyz                  # bitwise copies, so the graph joins the bridge
        rows[:3, -1] = partner_xyz
        columns = net.append(rows, BRIDGE)
        net.degree[tip] += 1
        net.degree[partner] += 1
        interior_columns = columns[1:-1]
        for c in interior_columns:
            net.degree[int(c)] = 2
        if len(interior_columns):
            index.add(path[1:-1], np.full(len(interior_columns), d_bridge / 2.0),
                      interior_columns.astype(np.int64))

        partner_is_tip = rank % 2 == 0
        events["anastomosis_bridges"] += 1
        events["anastomosis_partner_tips" if partner_is_tip else "anastomosis_partner_interior"] += 1
        if mode == "arteriovenous":
            events["anastomosis_bridges_arteriovenous" if rank < 2 else "anastomosis_bridges_same_tree"] += 1
        bridges.append({
            "tip": int(tip), "partner": int(partner),
            "partner_kind": "tip" if partner_is_tip else "interior",
            "tip_tree": int(tip_tree), "partner_tree": int(net.tree[partner]),
            "chord_um": chord,
            "arc_um": float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))),
            "diameter_um": float(d_bridge), "points": int(len(path)), "redraws": int(redraws),
        })

    nodes_out, tree_out = net.assemble(tree)
    return nodes_out, tree_out, bridges
