"""
Graph view of a centreline archive.

A centreline archive stores a network as polylines: runs of finite columns in a
(4, N) array of x, y, z and diameter, separated by all-NaN columns. Nothing in
that layout says which polylines touch. The connectivity is implicit in the
geometry: a branch starts on a point of its parent, so the junction exists only
as two columns holding the same coordinates. Every quantity that treats the
network as a graph -- degree, tips, junctions, components, cycles, the
segments between branch points -- therefore has to recover the vertices first
by merging the columns that coincide.

Coincidence is judged with a tolerance rather than bitwise, because a
polyline's own columns are not always bitwise stable: a stem's bare start vertex
is followed by the first sample of its B-spline, which reproduces the vertex
only up to floating-point rounding (differences of up to about 1e-13
micrometres). The default tolerance of 1e-6 micrometres sits some seven orders
of magnitude above that noise, so the two columns always collapse to one vertex
and the zero-length edge between them is dropped. The margin on the other side
is far thinner: a deep tree (nine iterations) holds genuine edges as short as
1.6e-5 micrometres, a single order of magnitude above the tolerance, and
hundreds shorter than 1e-3. Tolerances above about 1e-5 micrometres therefore
start merging real edges, quietly losing vertices and shortening the network,
so the default is not a value to round up for convenience.

Merging is done with a hash grid whose cells are one tolerance wide. Two
columns closer than the tolerance land in the same cell or in one of its 26
neighbours, so only those pairs are measured, and groups are closed
transitively by union-find; the work is linear in the number of columns
(plus a sort of the integer cell keys) instead of quadratic. Vertices are then
named by the smallest column index of their group, so every result is
deterministic and can be used directly to index back into `nodes`.

Vertex kinds are coded as small integers so they pack into an int8 array the
size of the archive: SEPARATOR (-1) for NaN columns, INTERIOR (0) for degree
two, JUNCTION (1) for degree three or more, TIP (2) for degree one, and
ISOLATED (3) for a vertex with no edge at all.
"""
import numpy as np

SEPARATOR, INTERIOR, JUNCTION, TIP, ISOLATED = -1, 0, 1, 2, 3

# Columns closer than this (in grammar units, micrometres) are one vertex. On
# the shipped archives the rounding noise between a stem's start vertex and its
# first spline sample is at most ~1e-13 um, and the shortest genuine edge (in a
# nine-iteration tree) is ~1.6e-5 um, so 1e-6 is about seven orders of magnitude
# above the noise but only one below the shortest edge. Above ~1e-5 um genuine
# edges start to merge: 2e-5 already removes one from that tree and 1e-4
# removes seventy-two.
DEFAULT_TOL = 1e-6

# The forward half of the 26-neighbourhood of a cell: visiting these from every
# cell reaches each unordered pair of neighbouring cells exactly once.
_FORWARD_OFFSETS = [(dx, dy, dz)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    if (dx, dy, dz) > (0, 0, 0)]

# Odd 64-bit multipliers (splitmix64) for mixing integer cell coordinates into
# one key. Collisions only add candidate pairs, which the distance check rejects.
_MIX = (np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9),
        np.uint64(0x94D049BB133111EB))


def _as_nodes(nodes):
    """Returns `nodes` as a float array of at least three rows; empty input is (4, 0)."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.size == 0:
        return np.empty((4, 0))
    if nodes.ndim != 2 or nodes.shape[0] < 3:
        raise ValueError("nodes must be a (4, N) array of x, y, z and diameter columns")
    return nodes


def _as_edges(edges, n_columns=None):
    """
    Returns `edges` as a (2, E) int64 array.

    Args:
        edges (ndarray): (2, E) array of column indices; empty input is (2, 0).
        n_columns (int): when given, every index must lie in 0..n_columns-1.
            The check belongs here rather than in each consumer because numpy
            indexing wraps a negative index round silently, so an unchecked
            -1 would quietly join the last column to whatever it is paired
            with instead of failing.
    """
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must be a (2, E) array of column indices")
    edges = edges.astype(np.int64, copy=False)
    if n_columns is not None and (edges.min() < 0 or edges.max() >= n_columns):
        raise ValueError(f"edges refer to columns outside 0..{n_columns - 1}")
    return edges


def _as_canonical(canonical, n_columns):
    """Returns `canonical` as an (N,) int64 array, checking it matches the archive."""
    canonical = np.asarray(canonical)
    if canonical.shape != (n_columns,):
        raise ValueError(f"canonical must have one entry per column, {n_columns}, "
                         f"got shape {canonical.shape}")
    return canonical.astype(np.int64, copy=False)


def _finite_columns(nodes):
    """Boolean mask of the columns that are points: finite x, y and z."""
    return np.isfinite(nodes[:3]).all(axis=0)


def polylines(nodes):
    """
    Splits an archive into its polylines.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.

    Returns:
        list: one int64 index array per polyline, each a run of consecutive
        finite columns in archive order. Runs are never empty, so consecutive
        or leading separators produce no entries.
    """
    nodes = _as_nodes(nodes)
    columns = np.flatnonzero(_finite_columns(nodes))
    if columns.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(columns) > 1) + 1
    return np.split(columns.astype(np.int64), breaks)


def polyline_of_column(nodes):
    """
    Labels each column with the index of its polyline.

    Returns:
        ndarray: (N,) int64, the polyline number in archive order for finite
        columns and -1 for separators.
    """
    nodes = _as_nodes(nodes)
    finite = _finite_columns(nodes)
    starts = finite.copy()
    starts[1:] &= ~finite[:-1]
    return np.where(finite, np.cumsum(starts) - 1, -1).astype(np.int64)


def _cell_keys(cells):
    """Mixes (M, 3) integer cell coordinates into one uint64 key per cell."""
    cells = cells.astype(np.uint64)
    key = cells[:, 0] * _MIX[0]
    key ^= cells[:, 1] * _MIX[1]
    key ^= cells[:, 2] * _MIX[2]
    key ^= key >> np.uint64(31)
    key *= _MIX[1]
    key ^= key >> np.uint64(29)
    return key


def _expand_ranges(source, low, high):
    """
    Pairs each `source[k]` with every integer in [low[k], high[k]).

    Returns:
        tuple: (a, b) int64 arrays of equal length, a the repeated sources and
        b the range members, in order of k then of b.
    """
    counts = high - low
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    a = np.repeat(source, counts)
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    b = np.repeat(low, counts) + (np.arange(total) - run_start)
    return a.astype(np.int64), b.astype(np.int64)


def _compress(parent):
    """Points every entry of a parent array straight at its root."""
    while True:
        grand = parent[parent]
        if np.array_equal(grand, parent):
            return parent
        parent = grand


def _min_root(count, a, b):
    """
    Connected components of `count` items joined by the pairs (a, b).

    Returns:
        ndarray: (count,) int64, for each item the smallest index in its
        component.

    Works in rounds: every pair whose ends have different roots hooks the
    larger root under the smaller, then all pointers are compressed. Each round
    removes at least one root from every component that is still split, so the
    loop ends, and in practice it takes one or two rounds because the largest
    root of a component always hooks. Pairs may repeat or join an item to itself.
    """
    parent = np.arange(count, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    while a.size:
        parent = _compress(parent)
        root_a, root_b = parent[a], parent[b]
        differ = root_a != root_b
        if not differ.any():
            break
        a, b = a[differ], b[differ]
        root_a, root_b = root_a[differ], root_b[differ]
        # a root may be hooked by several pairs at once; the minimum wins so
        # the result does not depend on the order the pairs were generated in
        np.minimum.at(parent, np.maximum(root_a, root_b), np.minimum(root_a, root_b))
    return _compress(parent)


def _hash_table(keys):
    """
    Open-addressing table over distinct uint64 keys, filled a round at a time.

    A binary search over a few hundred thousand keys costs some twenty
    dependent cache misses per probe; a table at most half full answers a
    probe with one gather and a compare, and a whole batch of probes at once.

    Returns:
        tuple: (table, shift) with table[slot] the index into `keys` of the
        key stored there or -1, and key >> shift a key's home slot. The keys
        are already mixed, so their top bits spread evenly over the slots.
    """
    bits = max(4, int(2 * keys.size - 1).bit_length())
    table = np.full(1 << bits, -1, dtype=np.int64)
    shift = np.uint64(64 - bits)
    mask = table.size - 1
    pending = np.arange(keys.size, dtype=np.int64)
    slots = (keys >> shift).astype(np.int64)
    while pending.size:
        # of the keys aimed at a free slot the first per slot takes it, the
        # rest move one slot along (linear probing); some slot is always free
        free = np.flatnonzero(table[slots] == -1)
        _, first = np.unique(slots[free], return_index=True)
        winners = free[first]
        table[slots[winners]] = pending[winners]
        keep = np.ones(pending.size, dtype=bool)
        keep[winners] = False
        pending = pending[keep]
        slots = (slots[keep] + 1) & mask
    return table, shift


def _lookup(table, shift, keys, wanted):
    """Index into `keys` of each `wanted` key, or -1 where it is absent."""
    mask = table.size - 1
    found = np.full(wanted.size, -1, dtype=np.int64)
    active = np.arange(wanted.size, dtype=np.int64)
    slots = (wanted >> shift).astype(np.int64)
    while active.size:
        entry = table[slots]
        occupied = entry >= 0
        hit = np.zeros(active.size, dtype=bool)
        hit[occupied] = keys[entry[occupied]] == wanted[active[occupied]]
        found[active[hit]] = entry[hit]
        # a probe ends at a hit or at an empty slot; otherwise it moves on
        probe = occupied & ~hit
        active = active[probe]
        slots = (slots[probe] + 1) & mask
    return found


def _neighbour_pairs(keys, cells):
    """
    Candidate pairs of representatives whose cells coincide or are neighbours.

    Args:
        keys (ndarray): (R,) uint64 cell keys, sorted ascending.
        cells (ndarray): (R, 3) int64 cell coordinates in the same order.

    Returns:
        tuple: (a, b) int64 arrays of representative indices, each unordered
        pair of neighbouring cells visited once.
    """
    count = keys.size
    new_cell = np.ones(count, dtype=bool)
    new_cell[1:] = keys[1:] != keys[:-1]
    cell_start = np.flatnonzero(new_cell)
    cell_end = np.append(cell_start[1:], count)
    cell_of = np.cumsum(new_cell) - 1
    unique_keys = keys[cell_start]
    table, shift = _hash_table(unique_keys)

    # within a cell, each representative against those after it
    sources = [np.arange(count)]
    lows = [np.arange(1, count + 1)]
    highs = [cell_end[cell_of]]
    for offset in _FORWARD_OFFSETS:
        wanted = _cell_keys(cells + np.array(offset, dtype=np.int64))
        cell = _lookup(table, shift, unique_keys, wanted)
        found = cell >= 0
        if not found.any():
            continue
        sources.append(np.flatnonzero(found))
        lows.append(cell_start[cell[found]])
        highs.append(cell_end[cell[found]])
    return _expand_ranges(np.concatenate(sources), np.concatenate(lows), np.concatenate(highs))


def canonical_columns(nodes, tol=DEFAULT_TOL):
    """
    Names the vertex each column belongs to by its smallest column index.

    Columns whose points lie within `tol` of each other (Euclidean distance,
    inclusive) are one vertex, and the relation is closed transitively, so a
    chain of columns each within `tol` of the next is one vertex however far
    its ends are apart. Points are binned into cells `tol` wide; only pairs in
    the same or a neighbouring cell are measured, so the cost is linear in the
    number of columns up to the sort of the integer cell keys. Bitwise-equal
    points are collapsed before binning, so any number of exact duplicates
    costs nothing extra.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        tol (float): merging distance in grammar units; 0 merges only columns
            whose coordinates compare equal as floats.

    Returns:
        ndarray: (N,) int64; for a finite column the smallest column index of
        its group (so a column c is a vertex exactly when the result at c is c),
        -1 for a separator.
    """
    nodes = _as_nodes(nodes)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be a non-negative number, got {tol!r}")
    n_columns = nodes.shape[1]
    canonical = np.full(n_columns, -1, dtype=np.int64)
    columns = np.flatnonzero(_finite_columns(nodes))
    if columns.size == 0:
        return canonical
    points = nodes[:3, columns].T

    # Sort by cell key, then by coordinates, so that a cell is one run and the
    # bitwise-equal points inside it are adjacent. lexsort is stable, so the
    # first of each run of equal points is also its lowest column.
    if tol == 0:
        cells = keys = None
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    else:
        cells = np.floor(points / tol).astype(np.int64)
        keys = _cell_keys(cells)
        order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], keys))
    sorted_points = points[order]
    sorted_columns = columns[order]

    first = np.ones(order.size, dtype=bool)
    first[1:] = np.any(sorted_points[1:] != sorted_points[:-1], axis=1)
    representative = np.cumsum(first) - 1          # per sorted position
    rep_position = np.flatnonzero(first)
    rep_column = sorted_columns[rep_position]

    if tol == 0 or rep_position.size == 1:
        root = np.arange(rep_position.size, dtype=np.int64)
    else:
        rep_points = sorted_points[rep_position]
        a, b = _neighbour_pairs(keys[order][rep_position], cells[order][rep_position])
        distance = np.sqrt(np.sum((rep_points[a] - rep_points[b]) ** 2, axis=1))
        close = distance <= tol
        root = _min_root(rep_position.size, a[close], b[close])

    # representatives are in cell order, not column order, so the group's name
    # is the least column over the representatives it gathers
    group_column = np.full(rep_position.size, n_columns, dtype=np.int64)
    np.minimum.at(group_column, root, rep_column)
    canonical[sorted_columns] = group_column[root[representative]]
    return canonical


def edges_from_nodes(nodes, tol=DEFAULT_TOL, canonical=None):
    """
    Undirected edges between the vertices of an archive.

    Each pair of consecutive columns of a polyline is an edge between their
    vertices. Edges between two columns of the same vertex have zero length and
    are dropped, as are repeats of an edge already present.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        tol (float): merging distance passed to canonical_columns when
            `canonical` is not given.
        canonical (ndarray): (N,) result of canonical_columns for `nodes`,
            computed here when omitted.

    Returns:
        ndarray: (2, E) int64 of canonical column indices, each edge stored
        with the smaller index first and the edges sorted lexicographically.
        (2, 0) when there are no edges.
    """
    nodes = _as_nodes(nodes)
    n_columns = nodes.shape[1]
    if canonical is None:
        canonical = canonical_columns(nodes, tol)
    else:
        canonical = _as_canonical(canonical, n_columns)
    finite = canonical >= 0
    linked = finite[:-1] & finite[1:]
    low = np.minimum(canonical[:-1][linked], canonical[1:][linked])
    high = np.maximum(canonical[:-1][linked], canonical[1:][linked])
    proper = low != high
    if not proper.any():
        return np.zeros((2, 0), dtype=np.int64)
    # one integer per edge orders them lexicographically and removes repeats
    code = np.unique(low[proper] * n_columns + high[proper])
    return np.stack([code // n_columns, code % n_columns]).astype(np.int64)


def degree(edges, n_columns):
    """
    Degree of every column: the number of edge ends at it.

    Args:
        edges (ndarray): (2, E) array of canonical column indices.
        n_columns (int): number of columns in the archive.

    Returns:
        ndarray: (N,) int64, nonzero only at canonical columns. A self-loop,
        should one be supplied, counts twice.
    """
    edges = _as_edges(edges, n_columns)
    return np.bincount(edges.ravel(), minlength=n_columns).astype(np.int64)


def node_kind(nodes, edges, canonical=None):
    """
    Classifies every column by the degree of its vertex.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        edges (ndarray): (2, E) edges of the archive.
        canonical (ndarray): (N,) result of canonical_columns for `nodes`,
            computed with DEFAULT_TOL when omitted.

    Returns:
        ndarray: (N,) int8 of TIP (degree 1), INTERIOR (2), JUNCTION (3 or
        more) or ISOLATED (0). A column that is not itself a vertex takes the
        kind of the vertex it belongs to; separators are SEPARATOR.
    """
    nodes = _as_nodes(nodes)
    n_columns = nodes.shape[1]
    if canonical is None:
        canonical = canonical_columns(nodes)
    else:
        canonical = _as_canonical(canonical, n_columns)
    deg = degree(edges, n_columns)
    by_degree = np.where(deg == 0, ISOLATED,
                         np.where(deg == 1, TIP,
                                  np.where(deg == 2, INTERIOR, JUNCTION))).astype(np.int8)
    kind = np.full(n_columns, SEPARATOR, dtype=np.int8)
    finite = canonical >= 0
    kind[finite] = by_degree[canonical[finite]]
    return kind


def components(edges, canonical):
    """
    Connected components of the graph, isolated vertices included.

    Args:
        edges (ndarray): (2, E) edges of the archive.
        canonical (ndarray): (N,) result of canonical_columns.

    Returns:
        tuple: (labels, count) with labels an (N,) int64 array holding a dense
        component number 0..count-1 at every canonical column and -1 elsewhere.
        Components are numbered in order of their lowest column, so the
        component holding the first point is 0.
    """
    canonical = np.asarray(canonical, dtype=np.int64)
    if canonical.ndim != 1:
        raise ValueError("canonical must be a one-dimensional array")
    n_columns = canonical.size
    edges = _as_edges(edges, n_columns)
    labels = np.full(n_columns, -1, dtype=np.int64)
    vertices = np.flatnonzero(canonical == np.arange(n_columns))
    if vertices.size == 0:
        return labels, 0
    root = _min_root(n_columns, edges[0], edges[1])
    _, dense = np.unique(root[vertices], return_inverse=True)
    dense = dense.reshape(-1)
    labels[vertices] = dense
    return labels, int(dense.max()) + 1


def betti(edges, canonical):
    """
    Counts of the graph and its first two Betti numbers.

    Returns:
        dict: "vertices" V, "edges" E, "components" b0 and "cycles"
        b1 = E - V + b0, the number of independent loops. A tree has none.
    """
    canonical = np.asarray(canonical, dtype=np.int64)
    edges = _as_edges(edges, canonical.size)
    vertices = int(np.count_nonzero(canonical == np.arange(canonical.size)))
    _, b0 = components(edges, canonical)
    n_edges = int(edges.shape[1])
    return {"vertices": vertices, "edges": n_edges, "components": b0,
            "cycles": n_edges - vertices + b0}


def segments(nodes, edges, canonical=None):
    """
    Splits the graph into the paths between vertices of degree other than two.

    A segment runs from a tip or junction through interior vertices to the next
    tip or junction (possibly the same one, when a loop hangs off a junction).
    Every edge lies on exactly one segment. A component whose vertices all have
    degree two has no place to break, and is returned whole as a closed path
    whose first and last entries are the same vertex.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN separators.
        edges (ndarray): (2, E) edges of the archive. Parallel edges are
            accepted and each is walked once.
        canonical (ndarray): unused when `edges` is given in canonical column
            indices, as edges_from_nodes returns them; kept for symmetry with
            the other functions.

    Returns:
        list: int64 arrays of canonical column indices, each of length two or
        more. Segments are ordered by their starting vertex, then by their first
        neighbour, and each open segment starts at its lower-index end; closed
        cycles come last, each starting at its lowest vertex. The order is a
        function of the edge list alone.
    """
    nodes = _as_nodes(nodes)
    n_columns = nodes.shape[1]
    edges = _as_edges(edges, n_columns)
    n_edges = edges.shape[1]
    if n_edges == 0:
        return []
    deg = degree(edges, n_columns)

    # Half-edges in compressed form: those leaving vertex v occupy
    # start[v]:start[v + 1], sorted by neighbour then by edge number so the
    # walk order is fixed. Plain lists, because the walk is a Python loop and
    # element access on lists is several times faster than on arrays.
    source = np.concatenate([edges[0], edges[1]])
    target = np.concatenate([edges[1], edges[0]])
    edge_id = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    order = np.lexsort((edge_id, target, source))
    start = np.searchsorted(source[order], np.arange(n_columns + 1)).tolist()
    target = target[order].tolist()
    edge_id = edge_id[order].tolist()
    deg_list = deg.tolist()
    used = [False] * n_edges
    paths = []

    def other_half(vertex, edge):
        """At a degree-two vertex, the half-edge that is not `edge`."""
        slot = start[vertex]
        if edge_id[slot] == edge:
            slot += 1
        return slot

    # Open segments first: leave every tip or junction along each unused
    # edge and follow interior vertices until the next tip or junction.
    for vertex in np.flatnonzero((deg != 2) & (deg > 0)).tolist():
        for slot in range(start[vertex], start[vertex + 1]):
            edge = edge_id[slot]
            if used[edge]:
                continue
            used[edge] = True
            path = [vertex]
            current = target[slot]
            while deg_list[current] == 2:
                path.append(current)
                slot = other_half(current, edge)
                edge = edge_id[slot]
                used[edge] = True
                current = target[slot]
            path.append(current)
            paths.append(path)

    # Whatever is left has degree two at both ends of every edge: pure cycles.
    for vertex in np.flatnonzero(deg == 2).tolist():
        slot = start[vertex]
        if used[edge_id[slot]]:
            slot += 1
            if used[edge_id[slot]]:
                continue
        edge = edge_id[slot]
        used[edge] = True
        path = [vertex]
        current = target[slot]
        while current != vertex:
            path.append(current)
            slot = other_half(current, edge)
            edge = edge_id[slot]
            used[edge] = True
            current = target[slot]
        path.append(vertex)
        paths.append(path)

    return [np.array(path, dtype=np.int64) for path in paths]


def vertex_diameter(nodes, canonical):
    """
    Diameter of each vertex: the largest over the columns it gathers.

    A junction holds the parent's diameter in one column and each child's
    starting diameter in others; the vessel there is as wide as the widest,
    which is what a rasteriser or a collision check needs.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter.
        canonical (ndarray): (N,) result of canonical_columns for `nodes`.

    Returns:
        ndarray: (N,) float, the maximum diameter at canonical columns and NaN
        elsewhere. A NaN diameter is ignored unless the whole group is NaN.
    """
    nodes = _as_nodes(nodes)
    n_columns = nodes.shape[1]
    if nodes.shape[0] < 4:
        raise ValueError("nodes has no diameter row")
    canonical = _as_canonical(canonical, n_columns)
    out = np.full(n_columns, np.nan)
    finite = canonical >= 0
    if not finite.any():
        return out
    largest = np.full(n_columns, -np.inf)
    np.fmax.at(largest, canonical[finite], nodes[3, finite])
    vertices = np.flatnonzero(canonical == np.arange(n_columns))
    out[vertices] = largest[vertices]
    out[np.isneginf(out)] = np.nan
    return out


def build(nodes, tol=DEFAULT_TOL):
    """
    Computes the graph of an archive in one call.

    Returns:
        dict: "canonical" from canonical_columns, "edges" from
        edges_from_nodes, "node_kind" from node_kind and "degree" from degree.
    """
    nodes = _as_nodes(nodes)
    canonical = canonical_columns(nodes, tol)
    edges = edges_from_nodes(nodes, tol, canonical)
    return {"canonical": canonical,
            "edges": edges,
            "node_kind": node_kind(nodes, edges, canonical),
            "degree": degree(edges, nodes.shape[1])}
