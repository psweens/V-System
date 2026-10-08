"""
Point index for collision checks between vessels.

A network is grown one stem at a time and every new stem has to be tested
against everything grown so far: two vessels collide when their centreline
points come closer than the sum of their radii. Comparing each new point with
every stored point is quadratic in the network size, which at the 1e5 .. 1e6
points of a large mesh is far too slow, so the points are kept in a spatial
index that compares a query only with the stored points near it.

Two indexes share one interface and return identical answers, so either can
stand in for the other:

- GridIndex hashes every point to the cube of side `cell_size` it falls in and
  keeps the cell keys of the stored points sorted, so the cells a query sphere
  covers are looked up by binary search. It needs only numpy. A reach larger
  than a cell merely covers more cells: the answer never depends on the cell
  size, only the speed does.
- KDTreeIndex wraps scipy.spatial.cKDTree, which searches a large static
  network faster but needs scipy, so scipy is imported only when this index
  is asked for.

Both amortise insertion the same way: a static search structure over the
points committed so far, plus a buffer of recently added points that is
searched directly and folded into the structure once it holds more than
`rebuild_every` points. The fold happens at the next query rather than at each
add, so a network added in one go is indexed once. The two folds differ in
cost, though. The grid merges the pending keys into its sorted keys in one
linear pass, whereas the tree is immutable and is rebuilt over every stored
point, so growing a network of n points builds n / rebuild_every trees at a
total cost of O(n^2 log n / rebuild_every). Under the generator's incremental
growth the grid is therefore the faster index once a network passes some 1e5
points (twice as fast at 3e5, over three times at 1e6), and the tree pays off
for a network indexed in one go and then queried many times; see `make_index`.

A query point q with radius r_q matches a stored point s with radius r_s when
|q - s| < r_q + r_s + margin, strictly, so that spheres which merely touch are
not a collision. A query therefore searches out to r_q + r_max + margin, r_max
being the largest stored radius, which is the farthest any stored point can be
and still satisfy the inequality; every point found is then tested against its
own radius. Each stored point carries an integer tag -- the branch it belongs
to, say -- so that a caller can tell a genuine collision from a vessel meeting
its own parent at a junction.

Coordinates and radii are in grammar units, which the pipeline takes to be
micrometres; see the Units section of the README.

    python spatial.py --benchmark 100000 1000000

times both indexes on the generator's insertion pattern: batches of a stem's
worth of points, each queried against the index before it is added.
"""
import argparse
import itertools
import sys
import time

import numpy as np

# Side of a GridIndex cell, in grammar units, when none is given: one default
# root diameter (main.py --d0 20). A collision query reaches about two radii
# plus a margin, so it covers a few dozen cells at most, while a cell holds only
# a handful of the points of a typical network.
DEFAULT_CELL_SIZE = 20.0

# Pending points tolerated before the search structure is rebuilt. Searching the
# buffer directly costs each query about this many comparisons per query point
# and rebuilding costs a pass over every stored point, so the two balance near
# the square root of the network size.
DEFAULT_REBUILD_EVERY = 2048

# Query points handled per pass, which bounds the memory a large query needs.
_CHUNK = 4096

# Grid cells enumerated per pass. A reach far beyond the cell size covers
# hundreds of cells per query point, so the covered cells are worked through a
# few query points at a time. Each covered cell costs a pass roughly 150 bytes
# of temporaries (its owner, coordinates, key and search bounds) plus the
# candidate pairs it yields, about 400 bytes in all on a network with eight
# points per cell, so a full pass peaks near 100 MB; a smaller budget would
# still vectorise over enough cells for the throughput not to change.
_CELL_BUDGET = 1 << 18

# Cell coordinates are packed into one int64 key with 21 bits per axis, so the
# packing is one-to-one across 2**21 cells (some forty metres at the default
# cell size). Cells farther apart than that share a key; a shared key only adds
# candidates, which the exact distance test then removes, so no pair is lost.
_KEY_BITS = 21
_KEY_MASK = (1 << _KEY_BITS) - 1


def _empty():
    return np.empty(0, dtype=np.int64)


def _as_points(points):
    """Returns `points` as an (n, 3) float array; a single (3,) point becomes (1, 3)."""
    points = np.asarray(points, dtype=float)
    if points.shape == (3,):
        points = points[None, :]
    if points.size == 0:
        return np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be an (n, 3) array, got shape {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    return points


def _as_radii(radii, count):
    """Returns `radii` as a (count,) array, accepting one radius for every point."""
    radii = np.asarray(radii, dtype=float)
    if radii.ndim > 1 or (radii.ndim == 1 and len(radii) != count):
        raise ValueError(
            f"radii must be a scalar or an array of {count}, got shape {radii.shape}")
    radii = np.array(np.broadcast_to(radii, (count,)))
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radii must be finite and non-negative")
    return radii


def _as_tags(tags, count):
    """Returns `tags` as a (count,) int64 array, accepting one tag for every point."""
    tags = np.asarray(tags)
    if not np.issubdtype(tags.dtype, np.integer):
        raise ValueError(f"tags must be integers, got dtype {tags.dtype}")
    if tags.ndim > 1 or (tags.ndim == 1 and len(tags) != count):
        raise ValueError(
            f"tags must be a scalar or an array of {count}, got shape {tags.shape}")
    return np.array(np.broadcast_to(tags, (count,)), dtype=np.int64)


def _widen(reach, points):
    """
    Widens a search reach by a few units in the last place of the coordinates
    involved, so that rounding in a distance cannot drop a pair whose distance
    sits on the very edge of the reach.
    """
    return reach + 8.0 * np.finfo(float).eps * (np.abs(points).max(axis=1) + reach)


def _expand_ranges(lo, hi):
    """
    Enumerates the half-open ranges lo[j] .. hi[j] as flat arrays (owner,
    position), owner being the range each position came from.
    """
    counts = hi - lo
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(lo), dtype=np.int64), counts)
    starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
    return owner, starts + np.arange(total, dtype=np.int64)


def _within_reach(q, reach, stored):
    """
    Returns (qi, si) for every stored point within reach[qi] of q[qi] by
    comparing every pair. Blocks of query points keep the distance matrix to
    about a million entries however large the inputs are.
    """
    if len(q) == 0 or len(stored) == 0:
        return _empty(), _empty()
    block = max(1, (1 << 20) // len(stored))
    found_q, found_s = [], []
    for start in range(0, len(q), block):
        stop = min(start + block, len(q))
        d2 = np.zeros((stop - start, len(stored)))
        for axis in range(3):     # one axis at a time: no (q, s, 3) difference array
            diff = q[start:stop, axis, None] - stored[None, :, axis]
            np.square(diff, out=diff)
            d2 += diff
        qi, si = np.nonzero(d2 <= (reach[start:stop] ** 2)[:, None])
        found_q.append(qi.astype(np.int64) + start)
        found_s.append(si.astype(np.int64))
    return np.concatenate(found_q), np.concatenate(found_s)


def _sorted_unique_pairs(qi, si):
    """Sorts pairs by qi then si and drops repeats, for a deterministic answer."""
    order = np.lexsort((si, qi))
    qi, si = qi[order], si[order]
    if len(qi) > 1:
        keep = np.empty(len(qi), dtype=bool)
        keep[0] = True
        keep[1:] = (qi[1:] != qi[:-1]) | (si[1:] != si[:-1])
        qi, si = qi[keep], si[keep]
    return qi, si


class _PointIndex:
    """
    Storage, validation and the exact tests shared by the indexes.

    A subclass supplies `_rebuild`, which absorbs every stored point into its
    search structure, and `_candidates`, which returns a superset of the stored
    points within a reach of each query point. The strict radius test, the
    ordering of the answer and the pending buffer live here, so the indexes
    cannot drift apart in what they return.
    """

    def __init__(self, rebuild_every=DEFAULT_REBUILD_EVERY):
        if rebuild_every < 0:
            raise ValueError("rebuild_every must be zero or positive")
        self.rebuild_every = int(rebuild_every)
        self.n = 0
        self._committed = 0            # points the search structure already holds
        self._r_max = 0.0              # largest stored radius, bounding every search
        self._points = np.empty((0, 3))
        self._radii = np.empty(0)
        self._tags = np.empty(0, dtype=np.int64)

    @property
    def points(self):
        """(n, 3) view of the stored points, valid until the next add."""
        return self._points[:self.n]

    @property
    def radii(self):
        """(n,) view of the stored radii, valid until the next add."""
        return self._radii[:self.n]

    @property
    def tags(self):
        """(n,) int64 view of the stored tags, valid until the next add."""
        return self._tags[:self.n]

    def __len__(self):
        return self.n

    def add(self, points, radii, tags):
        """
        Appends points to the index.

        Args:
            points (ndarray): (m, 3) coordinates, or one (3,) point.
            radii (float or ndarray): one radius per point, or one for all.
            tags (int or ndarray): one integer tag per point, or one for all.

        Returns:
            int: the index of the first new point; the new points occupy
            indices first .. first + m - 1 in the order given.
        """
        points = _as_points(points)
        count = len(points)
        radii = _as_radii(radii, count)
        tags = _as_tags(tags, count)
        first = self.n
        if count == 0:
            return first
        self._reserve(count)
        self._points[first:first + count] = points
        self._radii[first:first + count] = radii
        self._tags[first:first + count] = tags
        self.n += count
        self._r_max = max(self._r_max, float(radii.max()))
        self._added(first, count)
        return first

    def query(self, points, radii, margin=0.0):
        """
        Finds every stored point that collides with a query point.

        Args:
            points (ndarray): (q, 3) query coordinates, or one (3,) point.
            radii (float or ndarray): one radius per query point, or one for all.
            margin (float): extra clearance demanded between the two spheres;
                negative values tolerate that much overlap.

        Returns:
            tuple: (qi, si) int64 arrays of equal length listing every pair with
            |points[qi] - stored[si]| < radii[qi] + stored_radii[si] + margin,
            sorted by qi then si with no pair repeated. Both are empty when
            nothing collides or the index is empty.
        """
        q = _as_points(points)
        rq = _as_radii(radii, len(q))
        # a non-finite margin would make the reach non-finite, which the grid
        # cannot turn into cell coordinates while the tree happily searches, so
        # the two indexes would stop agreeing; reject it before either sees it
        margin = float(margin)
        if not np.isfinite(margin):
            raise ValueError("margin must be finite")
        if self.n == 0 or len(q) == 0:
            return _empty(), _empty()
        self._refresh()
        found_q, found_s = [], []
        for start in range(0, len(q), _CHUNK):
            stop = min(start + _CHUNK, len(q))
            qc, rc = q[start:stop], rq[start:stop]
            reach = np.maximum(rc + self._r_max + margin, 0.0)
            qi, si = self._candidates(qc, _widen(reach, qc))
            distance = np.linalg.norm(qc[qi] - self._points[si], axis=1)
            keep = distance < rc[qi] + self._radii[si] + margin
            qi, si = _sorted_unique_pairs(qi[keep], si[keep])
            found_q.append(qi + start)
            found_s.append(si)
        return np.concatenate(found_q), np.concatenate(found_s)

    def nearest_within(self, point, radius):
        """
        Lists the stored points within `radius` of `point`, nearest first.

        The test is inclusive, |stored - point| <= radius, since it asks which
        points lie within a distance rather than whether two spheres overlap.
        Points at equal distance come in index order.

        Returns:
            ndarray: int64 indices into the stored points, sorted by distance.
        """
        point = _as_points(point)
        if point.shape != (1, 3):
            raise ValueError(f"point must be a single (3,) point, got shape {point.shape}")
        radius = float(radius)
        if not np.isfinite(radius):
            raise ValueError("radius must be finite")
        if self.n == 0 or radius < 0.0:
            return _empty()
        self._refresh()
        _, si = self._candidates(point, _widen(np.array([radius]), point))
        si = np.unique(si)
        distance = np.linalg.norm(self._points[si] - point[0], axis=1)
        keep = distance <= radius
        si, distance = si[keep], distance[keep]
        return si[np.lexsort((si, distance))]

    def _reserve(self, count):
        """Grows the storage geometrically so that adds stay amortised O(1)."""
        needed = self.n + count
        capacity = len(self._radii)
        if needed <= capacity:
            return
        capacity = max(needed, 2 * capacity, 1024)
        points = np.empty((capacity, 3))
        radii = np.empty(capacity)
        tags = np.empty(capacity, dtype=np.int64)
        points[:self.n] = self._points[:self.n]
        radii[:self.n] = self._radii[:self.n]
        tags[:self.n] = self._tags[:self.n]
        self._points, self._radii, self._tags = points, radii, tags

    def _refresh(self):
        """Folds the pending buffer into the search structure once it has outgrown it."""
        if self.n - self._committed > self.rebuild_every:
            self._rebuild()

    def _added(self, first, count):
        """Hook for bookkeeping a subclass keeps per stored point."""

    def _rebuild(self):
        raise NotImplementedError

    def _candidates(self, q, reach):
        """
        Returns (qi, si) covering every stored point within reach[qi] of q[qi].
        Extra pairs and repeats are allowed; misses are not.
        """
        raise NotImplementedError

    def _pending_within_reach(self, q, reach):
        """Searches the pending buffer by direct comparison."""
        qi, si = _within_reach(q, reach, self._points[self._committed:self.n])
        return qi, si + self._committed


class GridIndex(_PointIndex):
    """
    Uniform grid index over the stored points, needing only numpy.

    Space is cut into cubes of side `cell_size` and each point is assigned the
    integer key of its cube. The keys of the committed points are kept sorted
    alongside the point they belong to, so the points of one cell form a run
    found by two binary searches, and a query enumerates the cells its sphere
    covers and looks each one up. A cell size near the usual search reach makes
    a query touch a few dozen cells holding a few points each; a reach far
    larger than a cell still returns the right answer, only through more cells.
    Should the spheres of a query cover more cells than there are stored points,
    the points are compared directly instead, which is then both faster and
    bounded in memory.

    Pending points are matched to a query through their keys as well, by looking
    each one up in the query's covered cells, so the buffer costs a query no
    distance evaluations; folding it in is a linear merge into the sorted keys.
    """

    def __init__(self, cell_size=DEFAULT_CELL_SIZE, rebuild_every=DEFAULT_REBUILD_EVERY):
        cell_size = float(cell_size)
        if not np.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError(f"cell_size must be a positive number, got {cell_size!r}")
        super().__init__(rebuild_every)
        self.cell_size = cell_size
        self._point_keys = np.empty(0, dtype=np.int64)    # cell key of every stored point
        self._sorted_keys = np.empty(0, dtype=np.int64)   # keys of the committed points, sorted
        self._sorted_index = np.empty(0, dtype=np.int64)  # the point held at each sorted slot

    def _cells(self, points):
        """Integer cell coordinates of points, (m, 3)."""
        return np.floor(points / self.cell_size).astype(np.int64)

    @staticmethod
    def _keys(cells):
        """Packs (…, 3) cell coordinates into one int64 key each."""
        return ((cells[..., 0] & _KEY_MASK)
                | ((cells[..., 1] & _KEY_MASK) << _KEY_BITS)
                | ((cells[..., 2] & _KEY_MASK) << (2 * _KEY_BITS)))

    def _reserve(self, count):
        super()._reserve(count)
        if len(self._point_keys) < len(self._radii):
            keys = np.empty(len(self._radii), dtype=np.int64)
            keys[:self.n] = self._point_keys[:self.n]
            self._point_keys = keys

    def _added(self, first, count):
        new = self._points[first:first + count]
        self._point_keys[first:first + count] = self._keys(self._cells(new))

    def _rebuild(self):
        """Merges the pending keys into the sorted committed keys in one linear pass."""
        pending = self._point_keys[self._committed:self.n]
        order = np.argsort(pending, kind="stable")
        new_keys = pending[order]
        new_index = order + self._committed
        # searching from the right puts a new key after equal committed ones,
        # so the points of a cell stay in insertion order across rebuilds
        slots = (np.searchsorted(self._sorted_keys, new_keys, side="right")
                 + np.arange(len(new_keys), dtype=np.int64))
        keys = np.empty(self.n, dtype=np.int64)
        index = np.empty(self.n, dtype=np.int64)
        old = np.ones(self.n, dtype=bool)
        old[slots] = False
        keys[old], keys[slots] = self._sorted_keys, new_keys
        index[old], index[slots] = self._sorted_index, new_index
        self._sorted_keys, self._sorted_index = keys, index
        self._committed = self.n

    def _candidates(self, q, reach):
        lo = self._cells(q - reach[:, None])
        hi = self._cells(q + reach[:, None])
        per_axis = np.maximum(hi - lo + 1, 0)
        per_query = per_axis.prod(axis=1, dtype=float)
        found_q, found_s = [], []

        # a sphere covering more cells than there are stored points is cheaper
        # to test against every point directly, which also caps the cells any
        # one query point can ask for
        direct = np.flatnonzero(per_query > self.n)
        if len(direct):
            qi, si = _within_reach(q[direct], reach[direct], self._points[:self.n])
            found_q.append(direct[qi])
            found_s.append(si)

        # the rest go through the cells, as many query points per pass as the
        # cell budget allows and never fewer than one
        gridded = np.flatnonzero(per_query <= self.n)
        cumulative = np.cumsum(per_query[gridded])
        start = 0
        while start < len(gridded):
            before = cumulative[start] - per_query[gridded[start]]
            stop = int(np.searchsorted(cumulative, before + _CELL_BUDGET, side="right"))
            part = gridded[start:max(stop, start + 1)]
            qi, si = self._search_cells(lo[part], per_axis[part])
            found_q.append(part[qi])
            found_s.append(si)
            start += len(part)
        if not found_q:
            return _empty(), _empty()
        return np.concatenate(found_q), np.concatenate(found_s)

    def _search_cells(self, lo, per_axis):
        """
        Returns (qi, si) for the stored points in the block of cells starting
        at lo[qi] and spanning per_axis[qi] cells along each axis.
        """
        # every cell each block covers, as flat arrays with their owner
        counts = per_axis.prod(axis=1)
        owner = np.repeat(np.arange(len(lo), dtype=np.int64), counts)
        local = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        ny, nz = per_axis[owner, 1], per_axis[owner, 2]
        cells = np.stack([lo[owner, 0] + local // (ny * nz),
                          lo[owner, 1] + (local // nz) % ny,
                          lo[owner, 2] + local % nz], axis=1)
        keys = self._keys(cells)

        # committed points: the run of each covered key in the sorted keys
        left = np.searchsorted(self._sorted_keys, keys, side="left")
        right = np.searchsorted(self._sorted_keys, keys, side="right")
        entry, slot = _expand_ranges(left, right)
        found_q, found_s = [owner[entry]], [self._sorted_index[slot]]

        # pending points: the same search with the roles swapped, each pending
        # key looked up among the covered keys
        if self._committed < self.n:
            pending = self._point_keys[self._committed:self.n]
            order = np.argsort(keys, kind="stable")
            covered = keys[order]
            left = np.searchsorted(covered, pending, side="left")
            right = np.searchsorted(covered, pending, side="right")
            point, entry = _expand_ranges(left, right)
            found_q.append(owner[order[entry]])
            found_s.append(point + self._committed)
        return np.concatenate(found_q), np.concatenate(found_s)


class KDTreeIndex(_PointIndex):
    """
    k-d tree index over the stored points, backed by scipy.spatial.cKDTree.

    The tree is immutable, so it is rebuilt over every stored point whenever
    the pending buffer has outgrown `rebuild_every`, and the pending points are
    compared with the query directly until then. Construction uses sliding
    midpoint splits without shrinking the bounding boxes, which builds several
    times faster than the balanced default and searches as fast at the small
    reaches of a collision query. Even so, each rebuild is a pass over the
    whole network, so a network grown by small batches is built and rebuilt
    n / rebuild_every times; for that pattern GridIndex, whose fold is a
    linear merge, is the faster choice beyond some 1e5 points.

    Raises:
        ImportError: when scipy is not installed; use GridIndex instead.
    """

    def __init__(self, rebuild_every=DEFAULT_REBUILD_EVERY, leafsize=32):
        from scipy.spatial import cKDTree
        super().__init__(rebuild_every)
        self._make_tree = cKDTree
        self.leafsize = int(leafsize)
        self._tree = None

    def _rebuild(self):
        self._tree = self._make_tree(self._points[:self.n], leafsize=self.leafsize,
                                     balanced_tree=False, compact_nodes=False)
        self._committed = self.n

    def _candidates(self, q, reach):
        found_q, found_s = [], []
        if self._tree is not None:
            hits = self._tree.query_ball_point(q, reach, return_sorted=False)
            counts = np.fromiter(map(len, hits), dtype=np.int64, count=len(hits))
            found_q.append(np.repeat(np.arange(len(q), dtype=np.int64), counts))
            found_s.append(np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64,
                                       count=int(counts.sum())))
        if self._committed < self.n:
            qi, si = self._pending_within_reach(q, reach)
            found_q.append(qi)
            found_s.append(si)
        return np.concatenate(found_q), np.concatenate(found_s)


def have_scipy():
    """Whether scipy imports, and so whether a KDTreeIndex can be built."""
    try:
        KDTreeIndex()
    except ImportError:
        return False
    return True


def make_index(kind="auto", cell_size=DEFAULT_CELL_SIZE, **options):
    """
    Builds an empty index.

    "auto" prefers the k-d tree, which searches fastest over a network that
    is indexed in one go, or in a few large additions, and then queried many
    times, as the descriptors do. A caller that grows a network point by
    point, querying each stem before adding it, should ask for "grid": its
    fold is a linear merge where the tree's is a full rebuild, and past some
    1e5 points that makes the grid twice as fast on the generator's pattern
    (`python spatial.py --benchmark 300000` shows the two side by side).

    Args:
        kind (str): "grid", "kdtree", or "auto" for the k-d tree when scipy
            imports and the grid otherwise.
        cell_size (float): cell side for the grid, in grammar units; the k-d
            tree ignores it.
        **options: passed to the index, such as rebuild_every.
    """
    if kind == "auto":
        kind = "kdtree" if have_scipy() else "grid"
    if kind == "grid":
        return GridIndex(cell_size, **options)
    if kind == "kdtree":
        return KDTreeIndex(**options)
    raise ValueError(f"kind must be 'grid', 'kdtree' or 'auto', got {kind!r}")


def benchmark(n_points, batch=40, seed=0, cell_size=DEFAULT_CELL_SIZE, margin=0.0, kinds=None):
    """
    Times the indexes on the generator's insertion pattern.

    Points are drawn uniformly in a cube sized for a mean spacing of 10 um, with
    radii of 0.5 .. 5 um, the small vessels a mesh mostly consists of, so a
    query finds a couple of neighbours on average. They are inserted in batches
    of `batch` points, a stem's worth, each batch queried against everything
    inserted before it and tagged with its batch number.

    Returns:
        dict: {kind: {"seconds": float, "pairs": int}} with the wall time of the
        queries and inserts together and the total number of pairs the queries
        returned, which must agree between the kinds.
    """
    if n_points < 0 or batch < 1:
        raise ValueError("n_points must be zero or more and batch at least 1")
    if kinds is None:
        kinds = ("grid", "kdtree") if have_scipy() else ("grid",)
    rng = np.random.default_rng(seed)
    side = 10.0 * max(n_points, 1) ** (1.0 / 3.0)
    points = rng.uniform(0.0, side, size=(n_points, 3))
    radii = rng.uniform(0.5, 5.0, size=n_points)
    tags = np.arange(n_points, dtype=np.int64) // batch

    results = {}
    for kind in kinds:
        index = make_index(kind, cell_size=cell_size)
        pairs = 0
        start = time.perf_counter()
        for first in range(0, n_points, batch):
            stop = min(first + batch, n_points)
            qi, _ = index.query(points[first:stop], radii[first:stop], margin)
            pairs += len(qi)
            index.add(points[first:stop], radii[first:stop], tags[first:stop])
        results[kind] = {"seconds": time.perf_counter() - start, "pairs": pairs}
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Time the spatial indexes on batches of points, each queried "
                    "for collisions against everything inserted before it.")
    parser.add_argument("--benchmark", type=int, nargs="+", metavar="N", required=True,
                        help="number of points to insert, once per value")
    parser.add_argument("--batch", type=int, default=40,
                        help="points per query-then-insert batch (default 40, a stem)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE,
                        help=f"grid cell side in grammar units (default {DEFAULT_CELL_SIZE})")
    parser.add_argument("--index", nargs="+", choices=("grid", "kdtree"),
                        help="indexes to time (default: both, or the grid without scipy)")
    args = parser.parse_args(argv)

    kinds = tuple(args.index) if args.index else None
    if kinds is None and not have_scipy():
        print("scipy is not installed: timing the grid only")
    status = 0
    for n_points in args.benchmark:
        results = benchmark(n_points, batch=args.batch, seed=args.seed,
                            cell_size=args.cell_size, kinds=kinds)
        queries = -(-n_points // args.batch)
        print(f"{n_points} points in batches of {args.batch} ({queries} queries), "
              f"cell size {args.cell_size:g}")
        for kind, result in results.items():
            print(f"    {kind:<8}{result['seconds']:9.3f} s  {result['pairs']:>12} pairs")
        if len({result["pairs"] for result in results.values()}) > 1:
            print("    MISMATCH: the indexes disagree on the pairs found")
            status = 1
        elif len(results) > 1:
            print("    the indexes agree")
    return status


if __name__ == "__main__":
    sys.exit(main())
