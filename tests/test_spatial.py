"""
Tests for the spatial point indexes.

Both indexes are checked against one brute-force reference on the same data,
so their agreement with each other follows from their agreement with it. The
k-d tree tests are skipped where scipy is not installed.

Run from the repository root with
    python -m unittest tests.test_spatial -v
"""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import spatial  # noqa: E402
from spatial import GridIndex, KDTreeIndex, benchmark, have_scipy, make_index  # noqa: E402

HAVE_SCIPY = have_scipy()


def brute_pairs(qpoints, qradii, spoints, sradii, margin=0.0):
    """Every (qi, si) with |q - s| < r_q + r_s + margin, sorted by qi then si."""
    if len(qpoints) == 0 or len(spoints) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    distance = np.linalg.norm(qpoints[:, None, :] - spoints[None, :, :], axis=2)
    qi, si = np.nonzero(distance < qradii[:, None] + sradii[None, :] + margin)
    return qi.astype(np.int64), si.astype(np.int64)


def cloud(rng, count, side, radius_range):
    points = rng.uniform(0.0, side, size=(count, 3))
    radii = rng.uniform(radius_range[0], radius_range[1], size=count)
    return points, radii


def every_index(cell_size=2.0, **options):
    """One fresh instance of each index kind this machine can run."""
    kinds = [GridIndex(cell_size, **options)]
    if HAVE_SCIPY:
        kinds.append(KDTreeIndex(**options))
    return kinds


class AgreementTests(unittest.TestCase):
    """The indexes return exactly the brute-force pairs, whatever the history of adds."""

    def assert_pairs(self, got, expected):
        qi, si = got
        self.assertEqual(qi.dtype, np.int64)
        self.assertEqual(si.dtype, np.int64)
        np.testing.assert_array_equal(qi, expected[0])
        np.testing.assert_array_equal(si, expected[1])

    def grow_and_check(self, index, points, radii, batch, margin=0.0):
        """Queries each batch against everything before it, then adds it."""
        before = index.n
        for first in range(0, len(points), batch):
            stop = min(first + batch, len(points))
            stored_points, stored_radii = index.points.copy(), index.radii.copy()
            got = index.query(points[first:stop], radii[first:stop], margin)
            self.assert_pairs(got, brute_pairs(points[first:stop], radii[first:stop],
                                               stored_points, stored_radii, margin))
            self.assertEqual(index.add(points[first:stop], radii[first:stop], first), before + first)
        self.assertEqual(index.n, before + len(points))

    def test_random_batches_match_brute_force(self):
        # the default rebuild policy keeps all thousand points in the pending
        # buffer and the small one folds nearly all of them into the search
        # structure, so both the buffer and the structure meet every margin
        for margin in (0.0, 0.7, -0.3):
            for rebuild_every in (2048, 100):
                for index in every_index(cell_size=2.0, rebuild_every=rebuild_every):
                    rng = np.random.default_rng(1)
                    points, radii = cloud(rng, 1000, 30.0, (0.2, 1.5))
                    with self.subTest(index=type(index).__name__, margin=margin,
                                      rebuild_every=rebuild_every):
                        self.grow_and_check(index, points, radii, batch=40, margin=margin)
                        probes, probe_radii = cloud(rng, 200, 30.0, (0.0, 3.0))
                        self.assert_pairs(index.query(probes, probe_radii, margin),
                                          brute_pairs(probes, probe_radii, points, radii, margin))
                        if rebuild_every == 100:
                            self.assertGreater(index._committed, 0)
                        else:
                            self.assertEqual(index._committed, 0)

    def test_reach_beyond_the_cell_size_searches_the_covered_cells(self):
        # once the index is large enough for cells to pay, a reach of several
        # cells must be found through the grid itself, so the fallback to direct
        # comparison is made to fail for the rest of the test; a small cell
        # budget makes the search take several passes over the query points
        rng = np.random.default_rng(2)
        points, radii = cloud(rng, 3000, 40.0, (0.5, 1.5))
        index = GridIndex(cell_size=1.0, rebuild_every=500)
        self.grow_and_check(index, points[:1000], radii[:1000], batch=40)
        with mock.patch.object(spatial, "_within_reach",
                               side_effect=AssertionError("fell back to direct comparison")), \
                mock.patch.object(spatial, "_CELL_BUDGET", 4096):
            self.grow_and_check(index, points[1000:], radii[1000:], batch=40)
            probes, probe_radii = cloud(rng, 100, 40.0, (2.0, 4.0))
            self.assert_pairs(index.query(probes, probe_radii),
                              brute_pairs(probes, probe_radii, points, radii))
        self.assertGreater(probe_radii.max() + radii.max(), 5 * index.cell_size)

    def test_a_far_reaching_query_among_ordinary_ones(self):
        # one query sphere covering more cells than there are points is compared
        # directly while the others still go through the cells, whether the
        # points are still pending or committed to the sorted keys
        rng = np.random.default_rng(8)
        points, radii = cloud(rng, 2000, 30.0, (0.2, 0.6))
        probes, probe_radii = cloud(rng, 60, 30.0, (0.2, 0.6))
        probe_radii[17] = 40.0
        for rebuild_every in (2048, 100):
            with self.subTest(rebuild_every=rebuild_every):
                index = GridIndex(cell_size=0.5, rebuild_every=rebuild_every)
                index.add(points, radii, 0)
                self.assert_pairs(index.query(probes, probe_radii),
                                  brute_pairs(probes, probe_radii, points, radii))
                self.assertEqual(len(index.nearest_within(probes[17], 100.0)), 2000)
                self.assertEqual(index._committed, 2000 if rebuild_every == 100 else 0)

    def test_small_index_matches_brute_force(self):
        # too few points for cells to pay: the grid compares directly instead
        rng = np.random.default_rng(3)
        points, radii = cloud(rng, 12, 5.0, (0.5, 2.0))
        for index in every_index(cell_size=0.1):
            with self.subTest(index=type(index).__name__):
                self.grow_and_check(index, points, radii, batch=3)

    def test_rebuild_policies_do_not_change_the_answer(self):
        rng = np.random.default_rng(4)
        points, radii = cloud(rng, 600, 25.0, (0.3, 1.2))
        for rebuild_every in (0, 7, 10 ** 9):
            for index in every_index(cell_size=3.0, rebuild_every=rebuild_every):
                with self.subTest(index=type(index).__name__, rebuild_every=rebuild_every):
                    self.grow_and_check(index, points, radii, batch=25)
                    if rebuild_every == 10 ** 9:
                        self.assertEqual(index._committed, 0)   # served from the buffer alone
                    if rebuild_every == 0:
                        self.assertEqual(index._committed, 600 - 25)

    def test_a_query_larger_than_a_chunk(self):
        rng = np.random.default_rng(5)
        points, radii = cloud(rng, 500, 20.0, (0.2, 0.8))
        probes, probe_radii = cloud(rng, spatial._CHUNK + 100, 20.0, (0.2, 0.8))
        for rebuild_every in (2048, 100):    # served from the buffer, then from the structure
            for index in every_index(cell_size=1.0, rebuild_every=rebuild_every):
                with self.subTest(index=type(index).__name__, rebuild_every=rebuild_every):
                    index.add(points, radii, 0)
                    self.assert_pairs(index.query(probes, probe_radii),
                                      brute_pairs(probes, probe_radii, points, radii))
                    self.assertEqual(index._committed, 500 if rebuild_every == 100 else 0)


class SemanticsTests(unittest.TestCase):

    def test_empty_index(self):
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                self.assertEqual(index.n, 0)
                self.assertEqual(len(index), 0)
                qi, si = index.query(np.zeros((3, 3)), 1.0)
                self.assertEqual(qi.dtype, np.int64)
                self.assertEqual((len(qi), len(si)), (0, 0))
                self.assertEqual(len(index.nearest_within((0.0, 0.0, 0.0), 5.0)), 0)
                self.assertEqual(index.add(np.empty((0, 3)), np.empty(0), np.empty(0, int)), 0)
                self.assertEqual(index.points.shape, (0, 3))

    def test_the_inequality_is_strict(self):
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                index.add((0.0, 0.0, 0.0), 1.0, 0)
                touching = index.query((3.0, 0.0, 0.0), 2.0)      # distance 3 == 1 + 2
                self.assertEqual(len(touching[0]), 0)
                overlapping = index.query((3.0, 0.0, 0.0), 2.0 + 1e-9)
                self.assertEqual(overlapping[1].tolist(), [0])
                with_margin = index.query((3.0, 0.0, 0.0), 2.0, margin=1e-9)
                self.assertEqual(with_margin[1].tolist(), [0])
                tolerated = index.query((2.0, 0.0, 0.0), 2.0, margin=-1.0)  # 2 < 1 + 2 - 1 fails
                self.assertEqual(len(tolerated[0]), 0)

    def test_a_non_finite_margin_is_rejected(self):
        # a non-finite reach has no cell coordinates, so the grid would search
        # one garbage cell while the tree searched everything; both refuse the
        # margin, on an empty index as well as a filled one
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                for margin in (np.inf, -np.inf, np.nan):
                    with self.assertRaises(ValueError):
                        index.query((0.5, 0.0, 0.0), 1.0, margin=margin)
                index.add((0.0, 0.0, 0.0), 1.0, 0)
                for margin in (np.inf, -np.inf, np.nan):
                    with self.assertRaises(ValueError):
                        index.query((0.5, 0.0, 0.0), 1.0, margin=margin)
                self.assertEqual(index.query((0.5, 0.0, 0.0), 1.0, margin=0.0)[1].tolist(), [0])

    def test_tags_and_order_are_preserved(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                self.assertEqual(index.add(points[:2], [0.5, 0.5], [7, 9]), 0)
                self.assertEqual(index.add(points[2:], 0.5, 11), 2)
                np.testing.assert_array_equal(index.tags, [7, 9, 11])
                self.assertEqual(index.tags.dtype, np.int64)
                np.testing.assert_array_equal(index.points, points)
                np.testing.assert_array_equal(index.radii, [0.5, 0.5, 0.5])
                qi, si = index.query([[10.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 0.6)
                self.assertEqual(qi.tolist(), [0, 1, 1])
                self.assertEqual(si.tolist(), [2, 0, 1])
                self.assertEqual(index.tags[si].tolist(), [11, 7, 9])

    def test_storage_survives_growth(self):
        rng = np.random.default_rng(6)
        points = rng.uniform(0.0, 100.0, size=(5000, 3))
        tags = rng.integers(0, 50, size=5000)
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                for first in range(0, 5000, 37):
                    index.add(points[first:first + 37], 0.5, tags[first:first + 37])
                np.testing.assert_array_equal(index.points, points)
                np.testing.assert_array_equal(index.tags, tags)
                self.assertEqual(index.n, 5000)

    def test_nearest_within_is_inclusive_and_sorted_by_distance(self):
        points = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0],
                           [-1.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
        for index in every_index(cell_size=0.5):
            with self.subTest(index=type(index).__name__):
                index.add(points, 0.1, 0)
                self.assertEqual(index.nearest_within((0.0, 0.0, 0.0), 3.0).tolist(), [1, 3, 2, 0])
                self.assertEqual(index.nearest_within((0.0, 0.0, 0.0), 2.5).tolist(), [1, 3, 2])
                self.assertEqual(index.nearest_within((0.0, 0.0, 0.0), 1.0).tolist(), [1, 3])
                self.assertEqual(len(index.nearest_within((0.0, 0.0, 0.0), 0.5)), 0)
                self.assertEqual(len(index.nearest_within((0.0, 0.0, 0.0), -1.0)), 0)

    def test_pairs_are_unique_and_sorted(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(0.0, 4.0, size=(300, 3))
        probes = rng.uniform(0.0, 4.0, size=(50, 3))
        for index in every_index(cell_size=0.3):
            with self.subTest(index=type(index).__name__):
                index.add(points, 1.0, 0)
                qi, si = index.query(probes, 1.0)
                pairs = list(zip(qi.tolist(), si.tolist()))
                self.assertEqual(pairs, sorted(set(pairs)))
                self.assertGreater(len(pairs), 0)

    def test_cells_far_enough_apart_to_share_a_key_stay_distinct(self):
        # the two points sit 2**21 cells apart along x, so their cells pack to
        # the same key. Enough filler keeps each query sphere covering fewer
        # cells than there are points, so the lookup has to go through the
        # keys rather than fall back to direct comparison, which is made to
        # fail; the keys are committed and sorted first, then all pending
        far = float(2 ** 21)
        filler = np.stack(np.meshgrid(*[np.arange(6.0)] * 3, indexing="ij"),
                          axis=-1).reshape(-1, 3) + 100.0
        for rebuild_every in (0, 2048):
            with self.subTest(rebuild_every=rebuild_every):
                index = GridIndex(cell_size=1.0, rebuild_every=rebuild_every)
                index.add([[0.0, 0.0, 0.0], [far, 0.0, 0.0]], 1.0, [0, 1])
                index.add(filler, 1.0, 2)
                self.assertEqual(index._point_keys[0], index._point_keys[1])
                with mock.patch.object(spatial, "_within_reach",
                                       side_effect=AssertionError("fell back to direct comparison")):
                    self.assertEqual(index.query((0.5, 0.0, 0.0), 1.0)[1].tolist(), [0])
                    self.assertEqual(index.nearest_within((far + 0.5, 0.0, 0.0), 1.0).tolist(), [1])
                self.assertEqual(index._committed, index.n if rebuild_every == 0 else 0)

    def test_invalid_inputs_are_rejected(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                GridIndex(bad)
        for index in every_index():
            with self.subTest(index=type(index).__name__):
                with self.assertRaises(ValueError):
                    index.add([[0.0, np.nan, 0.0]], 1.0, 0)
                with self.assertRaises(ValueError):
                    index.add(np.zeros((2, 2)), 1.0, 0)
                with self.assertRaises(ValueError):
                    index.add(np.zeros((2, 3)), [1.0, 2.0, 3.0], 0)
                with self.assertRaises(ValueError):
                    index.add(np.zeros((2, 3)), -1.0, 0)
                with self.assertRaises(ValueError):
                    index.add(np.zeros((2, 3)), 1.0, 0.5)
                index.add(np.zeros((1, 3)), 1.0, 0)
                with self.assertRaises(ValueError):
                    index.query(np.zeros((2, 3)), [1.0])
                with self.assertRaises(ValueError):
                    index.nearest_within(np.zeros((2, 3)), 1.0)
                self.assertEqual(index.n, 1)


class FactoryAndBenchmarkTests(unittest.TestCase):

    def test_make_index(self):
        self.assertIsInstance(make_index("grid", cell_size=3.0), GridIndex)
        self.assertEqual(make_index("grid", cell_size=3.0).cell_size, 3.0)
        self.assertEqual(make_index("grid", rebuild_every=5).rebuild_every, 5)
        auto = make_index("auto")
        self.assertIsInstance(auto, KDTreeIndex if HAVE_SCIPY else GridIndex)
        with self.assertRaises(ValueError):
            make_index("octree")

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
    def test_make_index_kdtree(self):
        self.assertIsInstance(make_index("kdtree", rebuild_every=3), KDTreeIndex)

    def test_benchmark_indexes_agree(self):
        results = benchmark(500, batch=40, seed=1, cell_size=10.0)
        self.assertIn("grid", results)
        if HAVE_SCIPY:
            self.assertIn("kdtree", results)
        pairs = {result["pairs"] for result in results.values()}
        self.assertEqual(len(pairs), 1)
        self.assertGreater(pairs.pop(), 0)
        for result in results.values():
            self.assertGreaterEqual(result["seconds"], 0.0)

    def test_benchmark_matches_brute_force(self):
        # the benchmark's own data, replayed through the reference
        rng = np.random.default_rng(1)
        side = 10.0 * 300 ** (1.0 / 3.0)
        points = rng.uniform(0.0, side, size=(300, 3))
        radii = rng.uniform(0.5, 5.0, size=300)
        expected = sum(len(brute_pairs(points[f:f + 40], radii[f:f + 40], points[:f], radii[:f])[0])
                       for f in range(0, 300, 40))
        for result in benchmark(300, batch=40, seed=1).values():
            self.assertEqual(result["pairs"], expected)

    def test_command_line(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = spatial.main(["--benchmark", "200", "80", "--batch", "20", "--cell-size", "8"])
        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("200 points in batches of 20 (10 queries)", text)
        self.assertIn("80 points in batches of 20 (4 queries)", text)
        self.assertIn("grid", text)
        if HAVE_SCIPY:
            self.assertIn("kdtree", text)
            self.assertIn("the indexes agree", text)


if __name__ == "__main__":
    unittest.main()
