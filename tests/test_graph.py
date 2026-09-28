"""
Tests for the graph view of a centreline archive.

Run from the repository root with
    python -m unittest tests.test_graph -v
"""
import glob
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from graph import (SEPARATOR, INTERIOR, JUNCTION, TIP, ISOLATED, DEFAULT_TOL,  # noqa: E402
                   polylines, polyline_of_column, canonical_columns, edges_from_nodes,
                   degree, node_kind, components, betti, segments, vertex_diameter, build)

FIXTURES = sorted(glob.glob(os.path.join(ROOT, "tests", "fixtures", "*.npz")))


def archive(*runs, diameter=1.0):
    """
    Builds a (4, N) array from polylines given as lists of points.

    A point is (x, y, z) or (x, y, z, diameter); polylines are separated by one
    all-NaN column, as the generator writes them.
    """
    columns = []
    for k, run in enumerate(runs):
        if k:
            columns.append([np.nan] * 4)
        for point in run:
            point = list(point)
            if len(point) == 3:
                point.append(diameter)
            columns.append(point)
    if not columns:
        return np.empty((4, 0))
    return np.array(columns, dtype=float).T


def brute_force_canonical(nodes, tol):
    """Reference for canonical_columns: pairwise distances and a plain union-find."""
    finite = np.isfinite(nodes[:3]).all(axis=0)
    idx = np.flatnonzero(finite)
    parent = {int(i): int(i) for i in idx}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i in idx:
        for j in idx:
            if i < j and np.linalg.norm(nodes[:3, i] - nodes[:3, j]) <= tol:
                ri, rj = find(int(i)), find(int(j))
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    out = np.full(nodes.shape[1], -1, dtype=np.int64)
    for i in idx:
        out[i] = find(int(i))
    return out


def edge_multiset(paths):
    """Every edge walked by a list of segments, as sorted (low, high) pairs."""
    walked = []
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            walked.append((min(int(a), int(b)), max(int(a), int(b))))
    return sorted(walked)


def edge_pairs(edges):
    """A (2, E) edge array as the sorted (low, high) pairs edge_multiset produces."""
    return sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges.T)


# A Y: the parent runs along y, the child leaves its middle point.
#   columns 0 1 2 | 3 = NaN | 4 (= column 1) 5 6
Y_NODES = archive([(0, 0, 0, 4.0), (0, 1, 0, 3.0), (0, 2, 0, 2.0)],
                  [(0, 1, 0, 2.5), (1, 1, 0, 2.0), (2, 1, 0, 1.5)])

# A square drawn as one polyline that closes on its first point.
SQUARE_NODES = archive([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)])


class PolylineTests(unittest.TestCase):
    def test_runs_are_split_on_separators_and_empty_runs_skipped(self):
        nodes = archive([(0, 0, 0), (1, 0, 0)], [], [(5, 5, 5)], [(7, 7, 7), (8, 8, 8), (9, 9, 9)])
        runs = polylines(nodes)
        self.assertEqual([list(r) for r in runs], [[0, 1], [4], [6, 7, 8]])
        self.assertTrue(all(r.dtype == np.int64 for r in runs))
        np.testing.assert_array_equal(polyline_of_column(nodes),
                                      [0, 0, -1, -1, 1, -1, 2, 2, 2])

    def test_leading_and_trailing_separators(self):
        nodes = np.concatenate([np.full((4, 2), np.nan), archive([(0, 0, 0), (1, 1, 1)]),
                                np.full((4, 1), np.nan)], axis=1)
        self.assertEqual([list(r) for r in polylines(nodes)], [[2, 3]])
        np.testing.assert_array_equal(polyline_of_column(nodes), [-1, -1, 0, 0, -1])

    def test_empty_and_all_nan_archives(self):
        for nodes in (np.empty((4, 0)), np.array([]), np.full((4, 3), np.nan)):
            with self.subTest(shape=nodes.shape):
                self.assertEqual(polylines(nodes), [])
                self.assertTrue(np.all(polyline_of_column(nodes) == -1))

    def test_a_malformed_array_is_rejected(self):
        with self.assertRaises(ValueError):
            polylines(np.zeros((2, 5)))
        with self.assertRaises(ValueError):
            polylines(np.zeros(5))


class CanonicalColumnTests(unittest.TestCase):
    def test_exact_junction_maps_to_the_lowest_column(self):
        np.testing.assert_array_equal(canonical_columns(Y_NODES), [0, 1, 2, -1, 1, 5, 6])

    def test_columns_within_tol_merge_and_beyond_do_not(self):
        nodes = archive([(0, 0, 0), (0, 0, 5e-7), (1, 0, 0), (1, 0, 2e-6)])
        np.testing.assert_array_equal(canonical_columns(nodes), [0, 0, 2, 3])
        # the boundary is inclusive
        nodes = archive([(0, 0, 0), (1e-6, 0, 0)])
        np.testing.assert_array_equal(canonical_columns(nodes), [0, 0])

    def test_merging_is_transitive_across_cells(self):
        # each step is within tol, so the whole chain is one vertex although
        # its ends are several tol apart and it spans many grid cells
        step = 0.8 * DEFAULT_TOL
        nodes = archive([(k * step, 0.3 * DEFAULT_TOL, -0.1 * DEFAULT_TOL) for k in range(12)])
        np.testing.assert_array_equal(canonical_columns(nodes), np.zeros(12, dtype=np.int64))

    def test_points_straddling_a_cell_boundary_merge(self):
        for axis in range(3):
            a = np.zeros(3)
            b = np.zeros(3)
            a[axis] = DEFAULT_TOL - 1e-9
            b[axis] = DEFAULT_TOL + 1e-9
            with self.subTest(axis=axis):
                np.testing.assert_array_equal(canonical_columns(archive([b, a])), [0, 0])

    def test_zero_tolerance_is_exact_equality(self):
        nodes = archive([(0, 0, 0), (0, 0, 1e-300), (0, 0, 0), (-0.0, 0.0, 0.0)])
        np.testing.assert_array_equal(canonical_columns(nodes, tol=0), [0, 1, 0, 0])
        np.testing.assert_array_equal(canonical_columns(nodes), [0, 0, 0, 0])

    def test_exact_duplicates_collapse_whatever_their_number(self):
        # columns 0..59 repeat one point; 61 repeats it again after a
        # separator; 62 and 64 are a second point either side of another
        nodes = archive([(3, 3, 3)] * 60, [(3, 3, 3), (4, 4, 4)], [(4, 4, 4)])
        expected = np.zeros(65, dtype=np.int64)
        expected[[60, 63]] = -1
        expected[[62, 64]] = 62
        np.testing.assert_array_equal(canonical_columns(nodes), expected)

    def test_single_point_and_empty(self):
        np.testing.assert_array_equal(canonical_columns(archive([(1, 2, 3)])), [0])
        self.assertEqual(canonical_columns(np.empty((4, 0))).shape, (0,))
        np.testing.assert_array_equal(canonical_columns(np.full((4, 2), np.nan)), [-1, -1])

    def test_agrees_with_brute_force_on_dense_random_data(self):
        # a large tolerance relative to the spread puts several points in every
        # cell and exercises all 26 neighbour directions and negative cells
        rng = np.random.RandomState(7)
        for trial in range(6):
            tol = 0.35
            points = rng.uniform(-1.0, 1.0, size=(45, 3))
            points[::5] = points[1::5] + rng.uniform(-tol, tol, size=(9, 3)) / 2
            nodes = np.vstack([points.T, np.ones(45)])
            nodes[:, 20] = np.nan
            with self.subTest(trial=trial):
                np.testing.assert_array_equal(canonical_columns(nodes, tol),
                                              brute_force_canonical(nodes, tol))

    def test_result_is_idempotent_and_never_points_upwards(self):
        rng = np.random.RandomState(3)
        nodes = np.vstack([rng.uniform(-2, 2, size=(3, 300)), np.ones(300)])
        canonical = canonical_columns(nodes, tol=0.25)
        self.assertTrue(np.all(canonical <= np.arange(300)))
        np.testing.assert_array_equal(canonical[canonical], canonical)

    def test_bad_tolerances_are_rejected(self):
        for tol in (-1e-6, np.nan, np.inf):
            with self.subTest(tol=tol):
                with self.assertRaises(ValueError):
                    canonical_columns(Y_NODES, tol)


class EdgeTests(unittest.TestCase):
    def test_y_edges_are_canonical_sorted_and_unique(self):
        np.testing.assert_array_equal(edges_from_nodes(Y_NODES), [[0, 1, 1, 5], [1, 2, 5, 6]])

    def test_zero_length_and_repeated_edges_are_dropped(self):
        # column 1 repeats column 0 within tolerance; the second polyline
        # retraces the first in the opposite direction
        nodes = archive([(0, 0, 0), (0, 0, 1e-8), (1, 0, 0), (2, 0, 0)],
                        [(2, 0, 0), (1, 0, 0)])
        edges = edges_from_nodes(nodes)
        np.testing.assert_array_equal(edges, [[0, 2], [2, 3]])
        self.assertEqual(edges.dtype, np.int64)

    def test_supplied_canonical_is_used(self):
        # with a canonical map that keeps the Y's junction columns apart the
        # child polyline is disconnected from the parent
        canonical = np.array([0, 1, 2, -1, 4, 5, 6])
        np.testing.assert_array_equal(edges_from_nodes(Y_NODES, canonical=canonical),
                                      [[0, 1, 4, 5], [1, 2, 5, 6]])
        with self.assertRaises(ValueError):
            edges_from_nodes(Y_NODES, canonical=canonical[:-1])

    def test_no_edges_gives_a_two_by_zero_array(self):
        for nodes in (np.empty((4, 0)), archive([(0, 0, 0)]), archive([(0, 0, 0), (0, 0, 0)]),
                      archive([(0, 0, 0)], [(1, 1, 1)])):
            with self.subTest(n=nodes.shape[1]):
                edges = edges_from_nodes(nodes)
                self.assertEqual(edges.shape, (2, 0))
                self.assertEqual(edges.dtype, np.int64)

    def test_degree_counts_edge_ends_at_canonical_columns_only(self):
        edges = edges_from_nodes(Y_NODES)
        np.testing.assert_array_equal(degree(edges, 7), [1, 3, 1, 0, 0, 2, 1])
        np.testing.assert_array_equal(degree(np.zeros((2, 0)), 3), [0, 0, 0])
        with self.assertRaises(ValueError):
            degree(edges, 6)
        with self.assertRaises(ValueError):
            degree(np.array([[0, -1], [1, 2]]), 7)


class KindAndTopologyTests(unittest.TestCase):
    def test_y_kinds_copy_to_non_canonical_columns(self):
        edges = edges_from_nodes(Y_NODES)
        kind = node_kind(Y_NODES, edges)
        np.testing.assert_array_equal(kind, [TIP, JUNCTION, TIP, SEPARATOR, JUNCTION, INTERIOR, TIP])
        self.assertEqual(kind.dtype, np.int8)

    def test_y_topology(self):
        canonical = canonical_columns(Y_NODES)
        edges = edges_from_nodes(Y_NODES, canonical=canonical)
        labels, count = components(edges, canonical)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(labels, [0, 0, 0, -1, -1, 0, 0])
        self.assertEqual(betti(edges, canonical),
                         {"vertices": 5, "edges": 4, "components": 1, "cycles": 0})

    def test_square_is_one_cycle_of_interior_vertices(self):
        canonical = canonical_columns(SQUARE_NODES)
        np.testing.assert_array_equal(canonical, [0, 1, 2, 3, 0])
        edges = edges_from_nodes(SQUARE_NODES, canonical=canonical)
        np.testing.assert_array_equal(edges, [[0, 0, 1, 2], [1, 3, 2, 3]])
        self.assertEqual(betti(edges, canonical),
                         {"vertices": 4, "edges": 4, "components": 1, "cycles": 1})
        np.testing.assert_array_equal(node_kind(SQUARE_NODES, edges, canonical), [INTERIOR] * 5)
        self.assertEqual(int(np.count_nonzero(node_kind(SQUARE_NODES, edges, canonical) == TIP)), 0)

    def test_two_components_are_numbered_by_lowest_column(self):
        nodes = archive([(0, 0, 0), (1, 0, 0)], [(5, 5, 5), (6, 5, 5), (7, 5, 5)], [(9, 9, 9)])
        canonical = canonical_columns(nodes)
        edges = edges_from_nodes(nodes, canonical=canonical)
        labels, count = components(edges, canonical)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(labels, [0, 0, -1, 1, 1, 1, -1, 2])
        self.assertEqual(betti(edges, canonical),
                         {"vertices": 6, "edges": 3, "components": 3, "cycles": 0})
        np.testing.assert_array_equal(node_kind(nodes, edges, canonical),
                                      [TIP, TIP, SEPARATOR, TIP, INTERIOR, TIP, SEPARATOR, ISOLATED])

    def test_single_point_is_an_isolated_vertex(self):
        nodes = archive([(1, 2, 3, 7.0)])
        result = build(nodes)
        np.testing.assert_array_equal(result["canonical"], [0])
        self.assertEqual(result["edges"].shape, (2, 0))
        np.testing.assert_array_equal(result["node_kind"], [ISOLATED])
        np.testing.assert_array_equal(result["degree"], [0])
        labels, count = components(result["edges"], result["canonical"])
        np.testing.assert_array_equal(labels, [0])
        self.assertEqual(count, 1)
        self.assertEqual(betti(result["edges"], result["canonical"]),
                         {"vertices": 1, "edges": 0, "components": 1, "cycles": 0})
        self.assertEqual(segments(nodes, result["edges"], result["canonical"]), [])
        np.testing.assert_array_equal(vertex_diameter(nodes, result["canonical"]), [7.0])

    def test_edges_outside_the_archive_are_rejected_everywhere(self):
        # a negative index would otherwise wrap round to the last column and
        # an index past the end would surface as a bare IndexError
        canonical = np.arange(4)
        nodes = archive([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        for edges in (np.array([[-1], [0]]), np.array([[0], [4]])):
            with self.subTest(edges=edges.ravel().tolist()):
                with self.assertRaises(ValueError):
                    components(edges, canonical)
                with self.assertRaises(ValueError):
                    betti(edges, canonical)
                with self.assertRaises(ValueError):
                    segments(nodes, edges, canonical)
                with self.assertRaises(ValueError):
                    node_kind(nodes, edges, canonical)

    def test_empty_archive(self):
        nodes = np.empty((4, 0))
        result = build(nodes)
        self.assertEqual(set(result), {"canonical", "edges", "node_kind", "degree"})
        self.assertEqual(result["canonical"].shape, (0,))
        self.assertEqual(result["edges"].shape, (2, 0))
        self.assertEqual(result["node_kind"].shape, (0,))
        self.assertEqual(result["degree"].shape, (0,))
        labels, count = components(result["edges"], result["canonical"])
        self.assertEqual((labels.shape, count), ((0,), 0))
        self.assertEqual(betti(result["edges"], result["canonical"]),
                         {"vertices": 0, "edges": 0, "components": 0, "cycles": 0})
        self.assertEqual(segments(nodes, result["edges"]), [])
        self.assertEqual(vertex_diameter(nodes, result["canonical"]).shape, (0,))

    def test_vertex_diameter_is_the_group_maximum(self):
        canonical = canonical_columns(Y_NODES)
        diameter = vertex_diameter(Y_NODES, canonical)
        np.testing.assert_array_equal(diameter[[0, 1, 2, 5, 6]], [4.0, 3.0, 2.0, 2.0, 1.5])
        self.assertTrue(np.isnan(diameter[[3, 4]]).all())
        # the child's starting diameter can exceed the parent's at the junction
        wider = Y_NODES.copy()
        wider[3, 4] = 9.0
        self.assertEqual(vertex_diameter(wider, canonical)[1], 9.0)


class SegmentTests(unittest.TestCase):
    def test_y_splits_at_the_junction_from_the_lowest_endpoint(self):
        edges = edges_from_nodes(Y_NODES)
        paths = segments(Y_NODES, edges)
        self.assertEqual([list(p) for p in paths], [[0, 1], [1, 2], [1, 5, 6]])
        self.assertTrue(all(p.dtype == np.int64 for p in paths))

    def test_pure_cycle_is_returned_closed(self):
        edges = edges_from_nodes(SQUARE_NODES)
        paths = segments(SQUARE_NODES, edges)
        self.assertEqual([list(p) for p in paths], [[0, 1, 2, 3, 0]])

    def test_loop_hanging_off_a_junction_starts_and_ends_there(self):
        # a stem 0-1 then a triangle 1-2-3-1: the loop is one segment from the
        # junction back to itself, and the stem another
        nodes = archive([(0, 0, 0), (0, 1, 0), (1, 2, 0), (-1, 2, 0), (0, 1, 0)])
        canonical = canonical_columns(nodes)
        edges = edges_from_nodes(nodes, canonical=canonical)
        paths = segments(nodes, edges, canonical)
        self.assertEqual([list(p) for p in paths], [[0, 1], [1, 2, 3, 1]])
        self.assertEqual(betti(edges, canonical)["cycles"], 1)

    def test_every_edge_is_walked_once_on_a_mixed_graph(self):
        nodes = archive([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
                        [(1, 0, 0), (1, 1, 0), (1, 2, 0)],
                        [(2, 0, 0), (2, 1, 0), (1, 2, 0)],
                        [(9, 9, 9), (9, 10, 9), (10, 10, 9), (9, 9, 9)])
        edges = edges_from_nodes(nodes)
        paths = segments(nodes, edges)
        self.assertEqual(edge_multiset(paths), edge_pairs(edges))
        for path in paths:
            self.assertGreaterEqual(len(path), 2)
        closed = [list(p) for p in paths if p[0] == p[-1]]
        self.assertEqual(closed, [[13, 14, 15, 13]])       # the triangle, columns 13..16
        # the same input always yields the same order
        again = segments(nodes, edges)
        self.assertEqual([list(p) for p in again], [list(p) for p in paths])

    def test_parallel_edges_are_each_walked_once(self):
        nodes = archive([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        # a doubled edge 0-1 beside 1-2: vertex 0 has degree two, 1 degree three
        edges = np.array([[0, 0, 1], [1, 1, 2]])
        paths = segments(nodes, edges)
        self.assertEqual([list(p) for p in paths], [[1, 0, 1], [1, 2]])
        # two parallel edges alone form a two-cycle
        paths = segments(nodes[:, :2], np.array([[0, 0], [1, 1]]))
        self.assertEqual([list(p) for p in paths], [[0, 1, 0]])

    def test_no_edges_gives_no_segments(self):
        self.assertEqual(segments(archive([(0, 0, 0)], [(1, 1, 1)]), np.zeros((2, 0))), [])


class FixtureTests(unittest.TestCase):
    """
    The archives in tests/fixtures are binary trees. Every polyline of two or
    more columns is a stem; a single-column polyline is a point the interpreter
    re-emitted on returning from a branch, a vertex the tree already has.
    """

    def setUp(self):
        if not FIXTURES:
            self.skipTest("no fixture archives")

    def load(self, path):
        with np.load(path, allow_pickle=False) as handle:
            return np.asarray(handle["nodes"], dtype=float)

    def test_fixtures_are_connected_trees(self):
        for path in FIXTURES:
            nodes = self.load(path)
            with self.subTest(fixture=os.path.basename(path)):
                result = build(nodes)
                canonical, edges = result["canonical"], result["edges"]
                self.assertEqual(result["node_kind"].dtype, np.int8)
                topology = betti(edges, canonical)
                self.assertEqual(topology["components"], 1)
                self.assertEqual(topology["cycles"], 0)
                self.assertEqual(topology["vertices"], topology["edges"] + 1)
                self.assertEqual(topology["vertices"], int(np.count_nonzero(canonical == np.arange(canonical.size))))
                self.assertTrue(np.all(canonical[canonical >= 0] <= np.flatnonzero(canonical >= 0)))

    def test_fixture_tips_match_the_stems(self):
        for path in FIXTURES:
            nodes = self.load(path)
            with self.subTest(fixture=os.path.basename(path)):
                canonical = canonical_columns(nodes)
                edges = edges_from_nodes(nodes, canonical=canonical)
                deg = degree(edges, nodes.shape[1])
                stems = [run for run in polylines(nodes) if run.size > 1]
                leaves = sum(1 for run in stems if deg[canonical[run[-1]]] == 1)
                roots = [run for run in stems if deg[canonical[run[0]]] == 1]
                tips = int(np.count_nonzero(deg == 1))
                self.assertEqual(len(roots), 1)
                self.assertEqual(int(roots[0][0]), 0)
                self.assertEqual(tips, leaves + 1)
                self.assertTrue(all(deg[canonical[run[0]]] >= 3 for run in stems if run is not roots[0]))
                kind = node_kind(nodes, edges, canonical)
                self.assertEqual(int(np.count_nonzero(kind[canonical == np.arange(canonical.size)] == TIP)), tips)
                self.assertEqual(int(np.count_nonzero(kind == SEPARATOR)), int(np.count_nonzero(canonical < 0)))
                self.assertTrue(np.all(kind[canonical >= 0] == kind[canonical[canonical >= 0]]))

    def test_fixture_segments_cover_every_edge_once(self):
        for path in FIXTURES:
            nodes = self.load(path)
            with self.subTest(fixture=os.path.basename(path)):
                canonical = canonical_columns(nodes)
                edges = edges_from_nodes(nodes, canonical=canonical)
                deg = degree(edges, nodes.shape[1])
                paths = segments(nodes, edges, canonical)
                self.assertEqual(sum(len(p) - 1 for p in paths), edges.shape[1])
                self.assertEqual(edge_multiset(paths), edge_pairs(edges))
                for p in paths:
                    self.assertNotEqual(deg[p[0]], 2)
                    self.assertNotEqual(deg[p[-1]], 2)
                    self.assertTrue(np.all(deg[p[1:-1]] == 2))
                # a tree has as many segments as tips and junction arms
                self.assertEqual(len(paths), int(deg[deg != 2].sum()) // 2)

    def test_fixture_diameters_are_finite_at_vertices_only(self):
        nodes = self.load(FIXTURES[0])
        canonical = canonical_columns(nodes)
        diameter = vertex_diameter(nodes, canonical)
        vertices = canonical == np.arange(canonical.size)
        self.assertTrue(np.all(np.isfinite(diameter[vertices])))
        self.assertTrue(np.all(np.isnan(diameter[~vertices])))
        self.assertTrue(np.all(diameter[vertices] >= nodes[3, vertices]))

    def test_default_tolerance_has_the_documented_headroom(self):
        # the rounding noise inside a stem is at most ~1e-13 um and the
        # shortest genuine edge ~1.6e-5 um, so a tolerance a million times
        # smaller than the default must still merge every noisy pair and one
        # ten times larger must still keep every real edge; either failing
        # means the fixtures or the generator moved and DEFAULT_TOL's
        # documentation is out of date
        for path in FIXTURES:
            nodes = self.load(path)
            with self.subTest(fixture=os.path.basename(path)):
                default = canonical_columns(nodes)
                np.testing.assert_array_equal(canonical_columns(nodes, DEFAULT_TOL / 1e6), default)
                np.testing.assert_array_equal(canonical_columns(nodes, 10 * DEFAULT_TOL), default)

    def test_tiled_fixtures_scale_to_a_large_network(self):
        # ten shifted copies of the largest fixture make a ~3e5 point archive
        # of ten trees; the merge must not join copies nor split any of them
        nodes = self.load(FIXTURES[-1])
        for path in FIXTURES:
            nodes = max(nodes, self.load(path), key=lambda n: n.shape[1])
        copies = max(1, 300000 // nodes.shape[1])
        pieces = []
        for k in range(copies):
            shifted = nodes.copy()
            shifted[0] += 5000.0 * k
            pieces.extend([shifted, np.full((4, 1), np.nan)])
        big = np.concatenate(pieces, axis=1)
        started = time.perf_counter()
        canonical = canonical_columns(big)
        edges = edges_from_nodes(big, canonical=canonical)
        elapsed = time.perf_counter() - started
        # the merge is linear in the number of columns and takes well under a
        # second here; the bound is some forty times that, loose enough for a
        # slow machine, and exists so that a regression to a quadratic path
        # fails with a message instead of hanging the suite
        self.assertLess(elapsed, 30.0)
        topology = betti(edges, canonical)
        single = betti(edges_from_nodes(nodes), canonical_columns(nodes))
        self.assertEqual(topology["components"], copies)
        self.assertEqual(topology["cycles"], 0)
        self.assertEqual(topology["vertices"], copies * single["vertices"])


if __name__ == "__main__":
    unittest.main()
