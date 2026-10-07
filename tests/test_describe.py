"""
Tests for the descriptors of a centreline network.

Run from the repository root with
    python -m unittest tests.test_describe -v
"""
import contextlib
import glob
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import graph  # noqa: E402
from collisions import KIN_REACH  # noqa: E402
from describe import (FLAG_KEYS, POLAR_BINS, AZIMUTH_BINS, describe, clearance,  # noqa: E402
                      load_archive, describe_archive, main)
import main as main_module  # noqa: E402

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


def straight(n_points=10, spacing=5.0, diameter=4.0, offset=(0.0, 0.0, 0.0), axis=0):
    """A straight polyline of equally spaced points along one axis."""
    points = []
    for k in range(n_points):
        point = list(offset)
        point[axis] += k * spacing
        points.append(tuple(point) + (diameter,))
    return points


def circle(n_edges=36, radius=50.0, diameter=4.0):
    """A ring drawn as one polyline whose last point repeats its first exactly."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_edges, endpoint=False)
    points = [(radius * math.cos(a), radius * math.sin(a), 0.0, diameter) for a in angles]
    return points + [points[0]]


# A Y: the parent runs along y, the child leaves its middle point.
Y_NODES = archive([(0, 0, 0, 4.0), (0, 1, 0, 3.0), (0, 2, 0, 2.0)],
                  [(0, 1, 0, 2.5), (1, 1, 0, 2.0), (2, 1, 0, 1.5)])

# Two vessels of diameter 4 running along x, 10 um apart: surfaces 6 um apart.
PARALLEL_NODES = archive(straight(), straight(offset=(0.0, 10.0, 0.0)))


def brute_force_clearance(nodes, margin, tol=graph.DEFAULT_TOL):
    """
    Reference for `clearance`: every pair of points, with the two exclusion
    rules applied literally.

    Returns:
        dict: "clearance" the clearance of every admitted pair (i < j) as a
        list of (value, column_i, column_j), and "violations" the number of
        them below the margin.
    """
    nodes = np.asarray(nodes, dtype=float)
    finite = np.isfinite(nodes[:3]).all(axis=0)
    canonical = graph.canonical_columns(nodes, tol)
    all_columns = np.flatnonzero(finite)
    all_polyline = graph.polyline_of_column(nodes)[all_columns]
    all_vertex = canonical[all_columns]
    columns = np.flatnonzero(finite & (canonical == np.arange(nodes.shape[1])))   # one column per point
    n = columns.size
    points = nodes[:3, columns].T
    radii = nodes[3, columns] / 2.0
    polyline = graph.polyline_of_column(nodes)[columns]
    vertex = canonical[columns]

    # arc positions along the full polyline, from its first column, whether or
    # not that column is kept: dropping a coincident copy drops a zero step
    all_arc = np.zeros(all_columns.size)
    for k in range(1, all_columns.size):
        if all_polyline[k] == all_polyline[k - 1]:
            all_arc[k] = all_arc[k - 1] + np.linalg.norm(nodes[:3, all_columns[k]] - nodes[:3, all_columns[k - 1]])
    arc = all_arc[np.isin(all_columns, columns)]
    n_polylines = int(polyline.max()) + 1 if n else 0
    total = np.zeros(n_polylines)
    closed = np.zeros(n_polylines, dtype=bool)
    for run in range(n_polylines):
        members = np.flatnonzero(polyline == run)
        if members.size == 0:
            continue
        total[run] = arc[members[-1]] - arc[members[0]]
        closed[run] = members.size > 1 and vertex[members[0]] == vertex[members[-1]] and total[run] > 0

    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    value = distance - radii[:, None] - radii[None, :]
    same = polyline[:, None] == polyline[None, :]
    separation = np.abs(arc[:, None] - arc[None, :])
    ring = same & closed[polyline][:, None]
    separation = np.where(ring, np.minimum(separation, total[polyline][:, None] - separation), separation)
    skip = same & (separation <= 2.0 * (radii[:, None] + radii[None, :] + margin))

    # rule (b): at a vertex J shared by the two polylines, the distances of
    # the two points to J sum to less than KIN_REACH (r_i + r_j + margin)
    # which polylines pass through a vertex is judged on every column, since a
    # polyline's own copy of a junction may have been dropped as a duplicate
    reach = KIN_REACH * (radii[:, None] + radii[None, :] + margin)
    for junction in np.unique(all_vertex):
        polylines_at = np.unique(all_polyline[all_vertex == junction])
        if polylines_at.size < 2:
            continue
        # arc-length distance of every kept point to J along its own polyline
        to_junction = np.full(n, np.inf)
        for p in polylines_at:
            arc_of_j = all_arc[np.flatnonzero((all_polyline == p) & (all_vertex == junction))[0]]
            on_p = polyline == p
            to_junction[on_p] = np.abs(arc[on_p] - arc_of_j)
        on = np.isin(polyline, polylines_at)
        together = to_junction[:, None] + to_junction[None, :]
        skip |= ~same & on[:, None] & on[None, :] & (together < reach)

    admitted = np.triu(~skip, k=1)
    i, j = np.nonzero(admitted)
    values = value[i, j]
    return {"clearance": [(float(v), int(columns[a]), int(columns[b])) for v, a, b in zip(values, i, j)],
            "violations": int(np.count_nonzero(values < margin))}


def numeric_leaves(value, path=""):
    """Every number in a nested result, with the key path leading to it."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for k, item in enumerate(value):
            yield from numeric_leaves(item, f"{path}[{k}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value


class StraightVesselTests(unittest.TestCase):
    def setUp(self):
        self.result = describe(archive(straight()), volume=(100.0, 100.0, 100.0))

    def test_shape_and_topology(self):
        r = self.result
        self.assertEqual((r["points"], r["polylines"], r["vertices"], r["edges"]), (10, 1, 10, 9))
        self.assertEqual((r["components"], r["cycles"]), (1, 0))
        self.assertEqual(r["tips"]["count"], 2)
        self.assertEqual(r["degree_histogram"], {"1": 2, "2": 8})
        self.assertAlmostEqual(r["total_length_mm"], 0.045)
        self.assertAlmostEqual(r["tips"]["per_mm"], 2 / 0.045)

    def test_arc_chord_is_one_and_curvature_zero(self):
        self.assertEqual(self.result["arc_chord"],
                         {"mean": 1.0, "median": 1.0, "p90": 1.0, "max": 1.0, "n_segments": 1, "n_closed": 0})
        self.assertEqual(self.result["curvature_per_um"], {"mean_abs": 0.0, "n_pairs": 8})

    def test_diameters(self):
        self.assertEqual(self.result["diameter_um"], {"p50": 4.0, "p90": 4.0, "p99": 4.0, "min": 4.0, "max": 4.0})

    def test_orientation_is_a_single_direction(self):
        orientation = self.result["orientation"]
        np.testing.assert_allclose(orientation["eigenvalues"], [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(orientation["principal_axis"], [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(orientation["fractional_anisotropy"], 1.0)
        # along x: 90 degrees from z (last polar bin), azimuth 0 (first bin)
        self.assertEqual(len(orientation["polar_histogram_deg"]), POLAR_BINS)
        self.assertEqual(len(orientation["azimuth_histogram_deg"]), AZIMUTH_BINS)
        self.assertEqual(orientation["polar_histogram_deg"][-1], 1.0)
        self.assertEqual(orientation["azimuth_histogram_deg"][0], 1.0)

    def test_length_density_against_the_given_volume(self):
        r = self.result
        self.assertEqual(r["volume_source"], "argument")
        self.assertAlmostEqual(r["volume_mm3"], 1e-3)
        self.assertAlmostEqual(r["length_density_mm_per_mm3"], 45.0)
        self.assertEqual(r["bounding_box_um"], {"min": [0.0, 0.0, 0.0], "max": [45.0, 0.0, 0.0]})

    def test_no_metadata_gives_empty_events_and_flags(self):
        self.assertEqual(self.result["events"], {})
        self.assertEqual(self.result["flags"], {})
        self.assertIsNone(self.result["per_tree"])
        self.assertEqual(self.result["units"], "um")


class ShapeTests(unittest.TestCase):
    def test_circle_is_one_cycle_without_tips(self):
        r = describe(archive(circle()))
        self.assertEqual((r["components"], r["cycles"], r["tips"]["count"]), (1, 1, 0))
        self.assertEqual(r["tips"]["per_mm"], 0.0)
        self.assertEqual(r["degree_histogram"], {"2": 36})
        self.assertEqual((r["vertices"], r["edges"]), (36, 36))
        # a closed segment has no chord, and its 36 turns of ten degrees, the
        # one across the seam included, give the curvature of a circle of
        # radius 50 to within the discretisation
        self.assertEqual(r["arc_chord"]["n_segments"], 0)
        self.assertEqual(r["arc_chord"]["n_closed"], 1)
        self.assertIsNone(r["arc_chord"]["mean"])
        self.assertEqual(r["curvature_per_um"]["n_pairs"], 36)
        self.assertAlmostEqual(r["curvature_per_um"]["mean_abs"], 1.0 / 50.0, delta=0.01 / 50.0)
        # the seam, where the last point repeats the first, is not a collision
        self.assertEqual(r["clearance_um"]["violations"], 0)

    def test_y_has_three_tips_and_one_junction(self):
        r = describe(Y_NODES)
        self.assertEqual((r["components"], r["cycles"], r["tips"]["count"]), (1, 0, 3))
        self.assertEqual(r["degree_histogram"], {"1": 3, "2": 1, "3": 1})
        self.assertEqual(r["arc_chord"]["n_segments"], 3)
        self.assertEqual(r["arc_chord"]["max"], 1.0)
        # the junction takes the largest diameter of its columns (3.0 over 2.5)
        self.assertEqual(r["diameter_um"]["max"], 4.0)
        self.assertEqual(r["diameter_um"]["min"], 1.5)

    def test_weighted_percentiles_follow_the_stated_definition(self):
        # edge lengths 1, 2, 3 with edge diameters 2, 3, 4: the edges of
        # diameter <= 3 hold half the length, so p50 is 3 and p90 is 4
        nodes = archive([(0, 0, 0, 2.0), (1, 0, 0, 2.0), (3, 0, 0, 4.0), (6, 0, 0, 4.0)])
        d = describe(nodes)["diameter_um"]
        self.assertEqual(d, {"p50": 3.0, "p90": 4.0, "p99": 4.0, "min": 2.0, "max": 4.0})

    def test_isotropic_orientation_has_no_anisotropy(self):
        nodes = archive(straight(axis=0), straight(axis=1, offset=(0, 100, 0)),
                        straight(axis=2, offset=(0, 0, 200)))
        orientation = describe(nodes)["orientation"]
        np.testing.assert_allclose(orientation["eigenvalues"], [1 / 3] * 3, atol=1e-12)
        self.assertAlmostEqual(orientation["fractional_anisotropy"], 0.0, places=6)
        self.assertAlmostEqual(sum(orientation["polar_histogram_deg"]), 1.0)
        self.assertAlmostEqual(sum(orientation["azimuth_histogram_deg"]), 1.0)
        # the z vessel is 0 degrees from z, the others 90; azimuths 0 and 90
        # (bin 3 of six 30-degree bins), the z vessel having no in-plane length
        polar = orientation["polar_histogram_deg"]
        self.assertAlmostEqual(polar[0], 1 / 3)
        self.assertAlmostEqual(polar[-1], 2 / 3)
        azimuth = orientation["azimuth_histogram_deg"]
        self.assertAlmostEqual(azimuth[0], 1 / 2)
        self.assertAlmostEqual(azimuth[3], 1 / 2)

    def test_curvature_of_a_right_angle(self):
        # two unit edges meeting at 90 degrees: kappa = (pi / 2) / 1
        r = describe(archive([(0, 0, 0), (1, 0, 0), (1, 1, 0)]))
        self.assertAlmostEqual(r["curvature_per_um"]["mean_abs"], math.pi / 2)
        self.assertAlmostEqual(r["arc_chord"]["mean"], 2.0 / math.sqrt(2.0))

    def test_supplied_edges_give_the_same_answer(self):
        nodes = archive(straight(), straight(offset=(0, 10, 0)))
        built = graph.build(nodes)
        self.assertEqual(describe(nodes, edges=built["edges"]), describe(nodes))

    def test_empty_and_single_point(self):
        r = describe(np.empty((4, 0)))
        self.assertEqual((r["points"], r["vertices"], r["edges"], r["components"], r["cycles"]), (0, 0, 0, 0, 0))
        self.assertEqual(r["total_length_mm"], 0.0)
        self.assertIsNone(r["tips"]["per_mm"])
        self.assertIsNone(r["orientation"]["eigenvalues"])
        self.assertIsNone(r["length_density_mm_per_mm3"])
        self.assertIsNone(r["clearance_um"]["min"])
        r = describe(archive([(1, 2, 3, 5.0)]))
        self.assertEqual((r["points"], r["vertices"], r["edges"], r["components"]), (1, 1, 0, 1))
        self.assertEqual(r["degree_histogram"], {"0": 1})
        self.assertEqual(r["diameter_um"]["max"], 5.0)
        self.assertIsNone(r["diameter_um"]["p50"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            describe(np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            describe(Y_NODES, edges=np.array([[0], [99]]))
        with self.assertRaises(ValueError):
            describe(Y_NODES, volume=(1.0, 2.0))
        with self.assertRaises(ValueError):
            describe(Y_NODES, tree=np.zeros(3, dtype=np.int8))
        with self.assertRaises(ValueError):
            clearance(Y_NODES, margin=-1.0)


class ClearanceTests(unittest.TestCase):
    def test_parallel_vessels_have_exact_clearance(self):
        # opposite points are 10 apart with radii 2 and 2: clearance 6, found
        # once the search has widened to reach it
        r = clearance(PARALLEL_NODES, margin=0.0)
        self.assertEqual(r["min"], 6.0)
        self.assertFalse(r["min_is_lower_bound"])
        self.assertEqual(r["violations"], 0)
        # at margin 5 the search reaches 7 straight away and the ten opposite
        # pairs are the only ones inside it, all clear
        r = clearance(PARALLEL_NODES, margin=5.0)
        self.assertEqual((r["min"], r["min_columns"], r["pairs_checked"], r["violations"]), (6.0, [0, 11], 10, 0))
        self.assertEqual(r["search_reach_um"], 7.0)
        # at margin 7 the same ten pairs are violations
        r = clearance(PARALLEL_NODES, margin=7.0)
        self.assertEqual((r["min"], r["violations"], r["pairs_checked"], r["margin"]), (6.0, 10, 10, 7.0))

    def test_no_admissible_pair_reports_a_lower_bound(self):
        # three points of one vessel, all within each other's arc-length window
        r = clearance(archive(straight(n_points=3, spacing=1.0, diameter=4.0)))
        self.assertIsNone(r["min"])
        self.assertIsNone(r["min_columns"])
        self.assertTrue(r["min_is_lower_bound"])
        self.assertEqual((r["violations"], r["pairs_checked"]), (0, 0))
        # the search gave up only once the reach exceeded the vessel's diagonal
        self.assertGreater(r["search_reach_um"], 2.0)

    def test_the_junction_rule_scales_with_the_radii(self):
        # the Y's far points are 1 and 2 from the shared vertex; excused only
        # while that sum is below KIN_REACH times the collision threshold
        r = clearance(Y_NODES)
        self.assertEqual(r["violations"], 0)
        self.assertAlmostEqual(r["min"], math.sqrt(5.0) - 1.75, places=9)   # (0,2,0) r 1 against (2,1,0) r 0.75
        fat = Y_NODES.copy()
        fat[3, np.isfinite(fat[3])] *= 2.0                                   # doubling every radius excuses that pair
        self.assertIsNone(clearance(fat)["min"])

    def test_junction_neighbourhood_is_excused_but_not_beyond(self):
        # a daughter leaving a junction at a shallow angle overlaps its parent
        # well past d_J + margin: the overlap inside the neighbourhood is
        # excused, the rest is a violation
        parent = [(x, 0.0, 0.0, 4.0) for x in np.arange(0.0, 41.0, 1.0)]
        child = [(20.0 + s, 0.15 * s, 0.0, 4.0) for s in np.arange(0.0, 21.0, 1.0)]
        nodes = archive(parent, child)
        r = clearance(nodes, margin=0.0)
        self.assertGreater(r["violations"], 0)
        self.assertLess(r["min"], 0.0)
        reference = brute_force_clearance(nodes, 0.0)
        self.assertEqual(r["violations"], reference["violations"])
        self.assertEqual(r["min"], min(reference["clearance"])[0])

    def test_ring_separation_is_measured_the_shorter_way_round(self):
        # a ring whose consecutive points are within the window of rule (a):
        # nothing is admitted below the margin although the seam repeats a point
        ring = archive(circle(n_edges=72, radius=20.0, diameter=4.0))
        r = clearance(ring, margin=1.0)
        self.assertEqual(r["violations"], 0)
        # the same points as an open arc that folds back onto its own start
        # are a collision: the ring rule applies only to closed polylines
        folded = archive(circle(n_edges=72, radius=20.0, diameter=4.0)[:-1]
                         + [(20.0 + 0.5, 0.0, 0.0, 4.0)])
        self.assertGreater(clearance(folded, margin=1.0)["violations"], 0)

    def test_agrees_with_brute_force(self):
        rng = np.random.default_rng(3)
        runs = []
        for k in range(40):
            start = rng.uniform(0.0, 60.0, 3)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            diameter = rng.uniform(0.5, 6.0)
            runs.append([tuple(start + direction * s) + (diameter,) for s in np.linspace(0.0, 20.0, 12)])
        # a few branches, so that rule (b) has junctions to excuse
        for k in range(10):
            base = runs[k][6]
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            runs.append([tuple(np.array(base[:3]) + direction * s) + (base[3] * 0.8,)
                         for s in np.linspace(0.0, 10.0, 8)])
        tangle = archive(*runs)
        cases = [(tangle, m) for m in (0.0, 0.5, 2.0)]
        cases.append((load_archive(FIXTURES[0])["nodes"], 0.0))
        cases.append((load_archive(FIXTURES[0])["nodes"], 1.0))
        for nodes, margin in cases:
            with self.subTest(points=int(np.isfinite(nodes[0]).sum()), margin=margin):
                for kind in ("grid", "auto"):
                    got = clearance(nodes, margin=margin, index=kind)
                    reference = brute_force_clearance(nodes, margin)
                    self.assertEqual(got["violations"], reference["violations"])
                    best = min(reference["clearance"])
                    self.assertAlmostEqual(got["min"], best[0], places=9)
                    self.assertEqual(got["min_columns"], [best[1], best[2]])
                    self.assertFalse(got["min_is_lower_bound"])
                    # exhaustive within the reach: the checked pairs are
                    # exactly the admitted pairs below it
                    within = sum(1 for value, _, _ in reference["clearance"] if value < got["search_reach_um"])
                    self.assertEqual(got["pairs_checked"], within)

    def test_margin_defaults_to_the_metadata(self):
        r = describe(PARALLEL_NODES, metadata={"collision_margin": 7.0, "avoid_collisions": True})
        self.assertEqual(r["clearance_um"]["margin"], 7.0)
        self.assertEqual(r["clearance_um"]["violations"], 10)
        r = describe(PARALLEL_NODES, metadata={"collision_margin": 7.0, "avoid_collisions": True}, margin=0.0)
        self.assertEqual(r["clearance_um"]["margin"], 0.0)
        self.assertEqual(r["clearance_um"]["violations"], 0)
        # a margin the generator did not enforce is not a promise to check
        r = describe(PARALLEL_NODES, metadata={"collision_margin": 7.0, "avoid_collisions": False})
        self.assertEqual(r["clearance_um"]["margin"], 0.0)


class MetadataTests(unittest.TestCase):
    def test_volume_sources_in_order_of_preference(self):
        nodes = archive(straight())
        record = {"growth_box_um": [100.0, 200.0, 50.0], "fit": "voxel_size",
                  "volume": [64, 64, 32], "voxel_size": 2.0}
        r = describe(nodes, metadata=record, volume=(10.0, 10.0, 10.0))
        self.assertEqual((r["volume_source"], r["volume_mm3"]), ("argument", 1e-6))
        r = describe(nodes, metadata=record)
        self.assertEqual(r["volume_source"], "growth_box_um")
        self.assertAlmostEqual(r["volume_mm3"], 1e6 / 1e9)
        del record["growth_box_um"]
        r = describe(nodes, metadata=record)
        self.assertEqual(r["volume_source"], "voxel_size")
        self.assertAlmostEqual(r["volume_mm3"], 64 * 64 * 32 * 8.0 / 1e9)
        record["fit"] = "isotropic"
        r = describe(nodes, metadata=record)
        # the box 0..45 along x padded by the radius 2 on every side
        self.assertEqual(r["volume_source"], "bounding_box")
        self.assertAlmostEqual(r["volume_mm3"], 49.0 * 4.0 * 4.0 / 1e9)
        self.assertAlmostEqual(r["length_density_mm_per_mm3"], 0.045 / (784.0 / 1e9))

    def test_events_and_flags_are_copied(self):
        record = {"events": {"collision_redraws": 3, "bound_terminations": 0},
                  "tortuosity": "walk", "persistence": 10.0, "seed": 7, "family": "mesh",
                  "properties": {"k": 3}, "units": "um"}
        r = describe(Y_NODES, metadata=record)
        self.assertEqual(r["events"], record["events"])
        self.assertEqual(r["flags"], {"tortuosity": "walk", "persistence": 10.0, "seed": 7, "family": "mesh"})
        self.assertTrue(set(r["flags"]) <= set(FLAG_KEYS))
        # the copy is independent of the record
        r["events"]["collision_redraws"] = 99
        self.assertEqual(record["events"]["collision_redraws"], 3)

    def test_metadata_as_json_string_or_array(self):
        record = {"collision_margin": 7.0, "avoid_collisions": True, "units": "um"}
        for form in (json.dumps(record), np.array(json.dumps(record))):
            r = describe(PARALLEL_NODES, metadata=form)
            self.assertEqual(r["clearance_um"]["margin"], 7.0)


class TreeAndArchiveTests(unittest.TestCase):
    def two_trees_with_a_bridge(self):
        """Two Y's labelled 0 and 1 whose nearest tips are joined by a bridge labelled 2."""
        tree_a = [[(0, 0, 0, 4.0), (0, 10, 0, 4.0), (0, 20, 0, 4.0)],
                  [(0, 10, 0, 2.0), (10, 10, 0, 2.0), (20, 10, 0, 2.0)]]
        tree_b = [[(100, 0, 0, 4.0), (100, 10, 0, 4.0), (100, 20, 0, 4.0)],
                  [(100, 10, 0, 2.0), (90, 10, 0, 2.0), (80, 10, 0, 2.0)]]
        bridge = [[(20, 10, 0, 2.0), (50, 10, 0, 2.0), (80, 10, 0, 2.0)]]
        nodes = archive(*(tree_a + tree_b + bridge))
        labels = [0] * 6 + [1] * 6 + [2] * 3
        tree = np.full(nodes.shape[1], -1, dtype=np.int8)
        tree[np.isfinite(nodes[0])] = labels
        return nodes, tree

    def test_per_tree_counts(self):
        nodes, tree = self.two_trees_with_a_bridge()
        r = describe(nodes, tree=tree)
        # the bridge joins the two trees into one component; a loop would need
        # both its ends in the same tree
        self.assertEqual((r["components"], r["cycles"], r["tips"]["count"]), (1, 0, 4))
        self.assertEqual(set(r["per_tree"]), {"0", "1", "2"})
        self.assertEqual(r["per_tree"]["0"], {"points": 6, "polylines": 2, "tips": 2, "length_mm": 0.04})
        self.assertEqual(r["per_tree"]["1"], {"points": 6, "polylines": 2, "tips": 2, "length_mm": 0.04})
        # the bridge's end points are the trees' tips, so it owns no tip
        self.assertEqual(r["per_tree"]["2"], {"points": 3, "polylines": 1, "tips": 0, "length_mm": 0.06})
        self.assertAlmostEqual(sum(t["length_mm"] for t in r["per_tree"].values()), r["total_length_mm"])

    def test_archive_round_trip_with_graph_arrays(self):
        nodes, tree = self.two_trees_with_a_bridge()
        built = graph.build(nodes)
        record = {"units": "um", "collision_margin": 0.5, "avoid_collisions": True, "events": {"collision_redraws": 1},
                  "growth_box_um": [200.0, 100.0, 100.0], "family": "mesh"}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "network.npz")
            np.savez(path, nodes=nodes, metadata=np.array(json.dumps(record)), edges=built["edges"],
                     node_kind=built["node_kind"], tree=tree)
            loaded = load_archive(path)
            np.testing.assert_array_equal(loaded["edges"], built["edges"])
            np.testing.assert_array_equal(loaded["tree"], tree)
            self.assertEqual(loaded["metadata"], record)
            r = describe_archive(path)
        self.assertEqual(r, describe(nodes, edges=built["edges"], metadata=record, tree=tree))
        self.assertEqual(r["clearance_um"]["margin"], 0.5)
        self.assertEqual(r["volume_source"], "growth_box_um")
        self.assertEqual(r["events"], {"collision_redraws": 1})
        self.assertEqual(r["per_tree"]["2"]["polylines"], 1)

    def test_archive_without_graph_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bare.npz")
            np.savez(path, nodes=Y_NODES)
            loaded = load_archive(path)
            self.assertIsNone(loaded["edges"])
            self.assertIsNone(loaded["metadata"])
            self.assertIsNone(loaded["tree"])
            self.assertEqual(describe_archive(path), describe(Y_NODES))
            np.savez(os.path.join(folder, "other.npz"), values=np.zeros(3))
            with self.assertRaises(ValueError):
                load_archive(os.path.join(folder, "other.npz"))


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for path in FIXTURES:
            cls.results[path] = describe_archive(path)

    def test_fixtures_exist(self):
        self.assertGreaterEqual(len(FIXTURES), 3)

    def test_fixtures_are_connected_trees_with_tips(self):
        for path, r in self.results.items():
            with self.subTest(fixture=os.path.basename(path)):
                self.assertEqual((r["components"], r["cycles"]), (1, 0))
                self.assertGreater(r["tips"]["count"], 0)
                self.assertEqual(r["cycles"], r["edges"] - r["vertices"] + r["components"])
                self.assertEqual(r["arc_chord"]["n_closed"], 0)
                self.assertEqual(r["units"], "um")
                self.assertIsInstance(r["flags"]["seed"], int)
                self.assertEqual(r["events"], {})

    def test_every_number_is_finite_and_defined(self):
        for path, r in self.results.items():
            with self.subTest(fixture=os.path.basename(path)):
                for key, value in numeric_leaves(r):
                    self.assertTrue(math.isfinite(value), f"{key} = {value}")
                for key in ("total_length_mm", "length_density_mm_per_mm3", "volume_mm3"):
                    self.assertIsNotNone(r[key], key)
                for key in ("p50", "p90", "p99", "min", "max"):
                    self.assertIsNotNone(r["diameter_um"][key], key)
                for key in ("mean", "median", "p90", "max"):
                    self.assertIsNotNone(r["arc_chord"][key], key)
                self.assertIsNotNone(r["curvature_per_um"]["mean_abs"])
                self.assertIsNotNone(r["orientation"]["fractional_anisotropy"])
                self.assertIsNotNone(r["clearance_um"]["min"])
                self.assertFalse(r["clearance_um"]["min_is_lower_bound"])

    def test_fixture_values_are_consistent_with_the_graph(self):
        for path, r in self.results.items():
            with self.subTest(fixture=os.path.basename(path)):
                archive_ = load_archive(path)
                nodes = archive_["nodes"]
                built = graph.build(nodes)
                self.assertEqual(r["tips"]["count"], int(np.sum(built["degree"] == 1)))
                topology = graph.betti(built["edges"], built["canonical"])
                self.assertEqual((r["vertices"], r["edges"]), (topology["vertices"], topology["edges"]))
                self.assertEqual(sum(r["degree_histogram"].values()), r["vertices"])
                positions = nodes[:3].T
                lengths = np.linalg.norm(positions[built["edges"][1]] - positions[built["edges"][0]], axis=1)
                self.assertAlmostEqual(r["total_length_mm"], lengths.sum() / 1000.0)
                self.assertEqual(r["points"], int(np.isfinite(nodes[0]).sum()))
                self.assertEqual(r["polylines"], len(graph.polylines(nodes)))
                self.assertEqual(r["arc_chord"]["n_segments"], len(graph.segments(nodes, built["edges"])))
                # a stem's five sub-segments bend it a little: ratios above 1
                self.assertGreater(r["arc_chord"]["mean"], 1.0)
                self.assertLess(r["arc_chord"]["max"], 1.5)
                self.assertGreater(r["curvature_per_um"]["mean_abs"], 0.0)
                orientation = r["orientation"]
                self.assertAlmostEqual(sum(orientation["eigenvalues"]), 1.0)
                self.assertEqual(orientation["eigenvalues"], sorted(orientation["eigenvalues"], reverse=True))
                self.assertAlmostEqual(sum(orientation["polar_histogram_deg"]), 1.0)
                self.assertAlmostEqual(sum(orientation["azimuth_histogram_deg"]), 1.0)
                self.assertEqual(r["volume_source"], "bounding_box")
                self.assertLessEqual(r["diameter_um"]["p50"], r["diameter_um"]["p90"])
                self.assertLessEqual(r["diameter_um"]["p90"], r["diameter_um"]["p99"])
                self.assertLessEqual(r["diameter_um"]["p99"], r["diameter_um"]["max"])

    def test_json_round_trip(self):
        for path, r in self.results.items():
            with self.subTest(fixture=os.path.basename(path)):
                text = json.dumps(r, allow_nan=False)
                self.assertEqual(json.loads(text), r)
        for nodes in (archive(straight()), archive(circle()), Y_NODES, np.empty((4, 0))):
            r = describe(nodes)
            self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)


class CommandLineTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(list(argv))
        return status, out.getvalue()

    def test_single_archive_prints_its_descriptors(self):
        status, text = self.run_main(FIXTURES[0], "--compact")
        self.assertEqual(status, 0)
        self.assertEqual(text.count("\n"), 1)
        self.assertEqual(json.loads(text), describe_archive(FIXTURES[0]))

    def test_several_archives_are_keyed_by_path(self):
        status, text = self.run_main(FIXTURES[0], FIXTURES[1])
        self.assertEqual(status, 0)
        result = json.loads(text)
        self.assertEqual(list(result), [FIXTURES[0], FIXTURES[1]])
        self.assertEqual(result[FIXTURES[1]]["points"], describe_archive(FIXTURES[1])["points"])

    def test_volume_and_margin_options(self):
        _, text = self.run_main(FIXTURES[0], "--volume", "500", "500", "250", "--margin", "1.5", "--compact")
        result = json.loads(text)
        self.assertEqual(result["volume_source"], "argument")
        self.assertAlmostEqual(result["volume_mm3"], 500 * 500 * 250 / 1e9)
        self.assertEqual(result["clearance_um"]["margin"], 1.5)
        with self.assertRaises(SystemExit):
            self.run_main(FIXTURES[0], "--margin", "-1")


# ---------------------------------------------------------------------------
# 3.5: the frame-relative and calibre-class descriptors
# ---------------------------------------------------------------------------

# The keys describe wrote before 3.5, in order, and the ones appended since.
OLD_KEYS = ("units", "points", "polylines", "vertices", "edges", "components", "cycles", "total_length_mm",
            "diameter_um", "arc_chord", "curvature_per_um", "tips", "degree_histogram", "orientation",
            "length_density_mm_per_mm3", "volume_mm3", "volume_source", "bounding_box_um", "clearance_um",
            "events", "flags", "per_tree")
NEW_KEYS = ("frame", "classes", "frame_orientation", "polar_order", "orientation_by_class",
            "calibre_shares", "segments_by_class", "transverse_spacing")
EMPTY_LENGTHS = {"count": 0, "median": None, "p10": None, "p90": None, "cv": None,
                 "median_d": None, "p10_d": None, "p90_d": None}
PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}


def frame_record(kind="axis", axis=(1.0, 0.0, 0.0), sense="nematic", tree_senses=None, normal=(0.0, 0.0, 1.0),
                 origin=(0.0, 0.0, 0.0), grow_direction=(0.0, 1.0, 0.0), grow_perpendicular=(0.0, 0.0, 1.0)):
    """A frame record as the generator writes it, without rules."""
    return {"frame_version": 1, "kind": kind,
            "axis": [float(v) for v in axis] if kind == "axis" else None,
            "sense": sense if kind == "axis" else None,
            "tree_senses": None if tree_senses is None else [int(v) for v in tree_senses],
            "normal": [float(v) for v in normal] if kind == "plane" else None,
            "origin": [float(v) for v in origin],
            "grow_direction": [float(v) for v in grow_direction],
            "grow_perpendicular": [float(v) for v in grow_perpendicular],
            "rules": []}


def unit_polylines(directions, rng, box=200.0, diameter=1.0):
    """One two-point polyline of unit length per row of `directions`, each from a random point of a box."""
    directions = np.asarray(directions, dtype=float)
    n = directions.shape[0]
    start = rng.uniform(0.0, box, (n, 3))
    nodes = np.full((4, 3 * n - 1), np.nan)
    nodes[:3, 0::3] = start.T
    nodes[:3, 1::3] = (start + directions).T
    nodes[3, 0::3] = diameter
    nodes[3, 1::3] = diameter
    return nodes


def directions_about_z(rng, cosines):
    """Unit vectors with the given cosines to the z axis and uniform azimuths."""
    cosines = np.asarray(cosines, dtype=float)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, cosines.size)
    sine = np.sqrt(np.maximum(1.0 - cosines ** 2, 0.0))
    return np.stack([sine * np.cos(azimuth), sine * np.sin(azimuth), cosines], axis=1)


def rejection_cosines(rng, n, log_density):
    """
    n cosines to the axis under a density on the sphere proportional to
    exp(log_density(u)), log_density at most 0, by rejection from the
    uniform sphere, whose cosine is uniform on [-1, 1].
    """
    kept = []
    drawn = 0
    while drawn < n:
        u = rng.uniform(-1.0, 1.0, 4 * n)
        accepted = u[rng.uniform(0.0, 1.0, 4 * n) < np.exp(log_density(u))]
        kept.append(accepted)
        drawn += accepted.size
    return np.concatenate(kept)[:n]


def watson_mean_square(k):
    """<u^2> of the Watson distribution exp(k u^2), by a fine trapezoidal rule, as a check on describe's quadrature."""
    # the rule is written out so that the test runs on every numpy the project supports
    u = np.linspace(0.0, 1.0, 200001)
    density = np.exp(k * u * u - max(k, 0.0))
    weight = np.full(u.size, 1.0 / (u.size - 1))
    weight[[0, -1]] /= 2.0
    return float(np.sum(weight * u * u * density) / np.sum(weight * density))


def langevin(k):
    """coth k - 1 / k."""
    return 1.0 / math.tanh(k) - 1.0 / k


def square_array(n, spacing, length, axis=2, diameter=1.0):
    """
    n x n straight vessels of the given length along one axis, each a
    two-point polyline from 0 to `length` along it, on a square grid of the
    given spacing centred on the axis.
    """
    across = [k for k in range(3) if k != axis]
    runs = []
    for i in range(n):
        for j in range(n):
            start = [0.0, 0.0, 0.0]
            start[across[0]] = (i - (n - 1) / 2.0) * spacing
            start[across[1]] = (j - (n - 1) / 2.0) * spacing
            end = list(start)
            end[axis] = length
            runs.append([tuple(start) + (diameter,), tuple(end) + (diameter,)])
    return archive(*runs)


def rotation(rng):
    """A random proper rotation matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return q


def rotated(nodes, frame, matrix):
    """The archive and the frame record turned by `matrix` about the origin."""
    turned = nodes.copy()
    finite = np.isfinite(nodes[0])
    turned[:3, finite] = matrix @ nodes[:3, finite]
    record = dict(frame)
    for key in ("axis", "normal", "origin", "grow_direction", "grow_perpendicular"):
        if record.get(key) is not None:
            record[key] = (matrix @ np.asarray(record[key], dtype=float)).tolist()
    return turned, record


_SMALL_TREE = {}


def small_tree():
    """A small tree grown once from fixed seeds, for the tests that need a generated network."""
    if not _SMALL_TREE:
        random.seed(3)
        np.random.seed(3)
        _SMALL_TREE.update(main_module.grow_network(5, 20.0, dict(PROPERTIES), (48, 48, 24), seed=3))
    return _SMALL_TREE


class FrameOrientationTests(unittest.TestCase):
    def test_order_is_exact_for_vessels_along_and_across_the_axis(self):
        along = describe(archive(straight()), frame=frame_record(axis=(1, 0, 0)))["frame_orientation"]
        self.assertEqual(along["kind"], "axis")
        # the classes need a reference diameter
        self.assertIsNone(along["capillary"])
        self.assertIsNone(along["larger"])
        entry = along["all"]
        self.assertEqual((entry["S"], entry["mean_abs_cos"], entry["crossing_ratio"]), (1.0, 1.0, 1.0))
        self.assertEqual((entry["mean_angle_deg"], entry["within_20_deg"], entry["within_45_deg"]), (0.0, 1.0, 1.0))
        self.assertEqual((entry["length"], entry["length_d"]), (45.0, None))
        # a single direction lies beyond the range of both concentrations
        self.assertIsNone(entry["watson_K"])
        self.assertIsNone(entry["fisher_axial_K"])

        across = describe(archive(straight(axis=1)), frame=frame_record(axis=(1, 0, 0)))["frame_orientation"]["all"]
        self.assertEqual((across["S"], across["mean_abs_cos"]), (-0.5, 0.0))
        self.assertAlmostEqual(across["mean_angle_deg"], 90.0)
        self.assertIsNone(across["crossing_ratio"])
        self.assertEqual((across["within_20_deg"], across["within_45_deg"]), (0.0, 0.0))
        self.assertIsNone(across["watson_K"])
        self.assertIsNone(across["fisher_axial_K"])

        # equal lengths along x and along y: every average is halfway
        both = archive(straight(), straight(axis=1, offset=(0.0, 10.0, 0.0)))
        mixed = describe(both, frame=frame_record(axis=(1, 0, 0)), d_ref=1.0)["frame_orientation"]
        entry = mixed["all"]
        self.assertAlmostEqual(entry["S"], 0.25)
        self.assertAlmostEqual(entry["mean_abs_cos"], 0.5)
        self.assertAlmostEqual(entry["crossing_ratio"], 2.0)
        self.assertAlmostEqual(entry["mean_angle_deg"], 45.0)
        self.assertEqual((entry["within_20_deg"], entry["within_45_deg"]), (0.5, 0.5))
        # each concentration inverts its defining moment
        self.assertAlmostEqual(watson_mean_square(entry["watson_K"]), 0.5, places=6)
        self.assertAlmostEqual(langevin(entry["fisher_axial_K"]), 0.5, places=9)
        # at d_ref 1 both vessels (diameter 4) are larger-class, so that entry is
        # the whole network and the capillary one has no length
        self.assertEqual(mixed["larger"], dict(entry, length_d=90.0))
        self.assertIsNone(mixed["capillary"])

        # a plane frame: both vessels lie in the xy plane, half of their
        # length runs along a normal of x
        plane = describe(both, frame=frame_record(kind="plane", normal=(0, 0, 1)))["frame_orientation"]
        self.assertEqual(plane["kind"], "plane")
        self.assertEqual(plane["all"], {"in_plane_fraction": 1.0, "S_n": -0.5, "length": 90.0, "length_d": None})
        tilted = describe(both, frame=frame_record(kind="plane", normal=(1, 0, 0)), d_ref=2.0)["frame_orientation"]
        self.assertAlmostEqual(tilted["all"]["in_plane_fraction"], 0.5)
        self.assertAlmostEqual(tilted["all"]["S_n"], 0.25)
        self.assertEqual(tilted["all"]["length_d"], 45.0)

    def test_isotropic_tangents_have_no_order(self):
        rng = np.random.default_rng(11)
        directions = rng.normal(size=(10000, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        nodes = unit_polylines(directions, rng)
        entry = describe(nodes, frame=frame_record(axis=(0, 0, 1)))["frame_orientation"]["all"]
        self.assertLess(abs(entry["S"]), 0.03)
        self.assertLess(abs(entry["watson_K"]), 0.3)
        self.assertAlmostEqual(entry["mean_abs_cos"], 0.5, delta=0.03)
        # the mean angle of a uniform direction is int_0^1 acos(u) du = 1 radian
        self.assertAlmostEqual(entry["mean_angle_deg"], math.degrees(1.0), delta=2.0)
        self.assertAlmostEqual(entry["length"], 10000.0, places=6)
        plane = describe(nodes, frame=frame_record(kind="plane"))["frame_orientation"]["all"]
        self.assertAlmostEqual(plane["in_plane_fraction"], 2.0 / 3.0, delta=0.03)
        self.assertLess(abs(plane["S_n"]), 0.03)

    def test_watson_concentration_is_recovered(self):
        rng = np.random.default_rng(5)
        cosines = rejection_cosines(rng, 10000, lambda u: 5.0 * (u * u - 1.0))
        nodes = unit_polylines(directions_about_z(rng, cosines), rng)
        entry = describe(nodes, frame=frame_record(axis=(0, 0, 1)))["frame_orientation"]["all"]
        self.assertLess(abs(entry["watson_K"] - 5.0) / 5.0, 0.1)
        # and the measured order agrees with the distribution's own
        self.assertAlmostEqual(entry["S"], (3.0 * watson_mean_square(5.0) - 1.0) / 2.0, delta=0.03)

    def test_fisher_axial_concentration_is_recovered(self):
        rng = np.random.default_rng(6)
        cosines = rejection_cosines(rng, 10000, lambda u: 5.0 * (np.abs(u) - 1.0))
        nodes = unit_polylines(directions_about_z(rng, cosines), rng)
        entry = describe(nodes, frame=frame_record(axis=(0, 0, 1)))["frame_orientation"]["all"]
        self.assertLess(abs(entry["fisher_axial_K"] - 5.0) / 5.0, 0.1)
        self.assertAlmostEqual(entry["mean_abs_cos"], langevin(5.0), delta=0.03)

    def test_polar_order_follows_the_tree_senses_and_the_column_order(self):
        nodes = archive(straight())                       # root at x = 0, tip at x = 45
        tree = np.zeros(nodes.shape[1], dtype=np.int8)
        polar = frame_record(axis=(1, 0, 0), sense="polar", tree_senses=[1])
        r = describe(nodes, tree=tree, frame=polar, d_ref=10.0)
        self.assertEqual(r["polar_order"], {"all": 1.0, "capillary": 1.0, "larger": None})
        against = frame_record(axis=(1, 0, 0), sense="polar", tree_senses=[-1])
        r = describe(nodes, tree=tree, frame=against)
        self.assertEqual(r["polar_order"], {"all": -1.0, "capillary": None, "larger": None})
        # the tangent runs from the lower column to the higher, so a polyline
        # written from its tip to its root runs against the axis
        self.assertEqual(describe(archive(straight()[::-1]), tree=tree, frame=polar)["polar_order"]["all"], -1.0)
        # an edge whose label indexes no sense, as a bridge's label 2, is left out of both sums
        with_bridge = archive(straight(), straight(offset=(0.0, 10.0, 0.0))[::-1])
        labels = np.full(with_bridge.shape[1], -1, dtype=np.int8)
        labels[np.isfinite(with_bridge[0])] = [0] * 10 + [2] * 10
        self.assertEqual(describe(with_bridge, tree=labels, frame=polar)["polar_order"]["all"], 1.0)
        labels[np.isfinite(with_bridge[0])] = [0] * 10 + [1] * 10
        self.assertEqual(describe(with_bridge, tree=labels, frame=polar)["polar_order"]["all"], 1.0)
        two = frame_record(axis=(1, 0, 0), sense="polar", tree_senses=[1, -1])
        self.assertEqual(describe(with_bridge, tree=labels, frame=two)["polar_order"]["all"], 1.0)
        two = frame_record(axis=(1, 0, 0), sense="polar", tree_senses=[1, 1])
        self.assertEqual(describe(with_bridge, tree=labels, frame=two)["polar_order"]["all"], 0.0)
        # None without tree labels, for a nematic sense and for a plane frame
        self.assertIsNone(describe(nodes, frame=polar)["polar_order"])
        self.assertIsNone(describe(nodes, tree=tree, frame=frame_record(axis=(1, 0, 0)))["polar_order"])
        self.assertIsNone(describe(nodes, tree=tree, frame=frame_record(kind="plane"))["polar_order"])

    def test_calibre_shares_are_exact(self):
        nodes = archive([(0, 0, 0, 1.0), (10, 0, 0, 1.0)], [(0, 10, 0, 4.0), (5, 10, 0, 4.0)])
        shares = describe(nodes, d_ref=1.0)["calibre_shares"]
        self.assertEqual(set(shares), {"length", "volume"})
        for key in ("below_1.5", "below_2", "below_3"):
            self.assertAlmostEqual(shares["length"][key], 2.0 / 3.0)
            self.assertAlmostEqual(shares["volume"][key], 10.0 / 90.0)
        self.assertEqual(list(shares["length"]), ["below_1.5", "below_2", "below_3"])
        # at d_ref 2 the thick vessel is below 3 d_ref but not below 2
        shares = describe(nodes, d_ref=2.0)["calibre_shares"]
        self.assertAlmostEqual(shares["length"]["below_2"], 2.0 / 3.0)
        self.assertAlmostEqual(shares["length"]["below_3"], 1.0)
        self.assertAlmostEqual(shares["volume"]["below_3"], 1.0)
        self.assertIsNone(describe(nodes)["calibre_shares"])
        # an edge of unknown diameter is outside the totals
        unknown = archive([(0, 0, 0, np.nan), (10, 0, 0, np.nan)], [(0, 10, 0, 1.0), (5, 10, 0, 1.0)])
        self.assertEqual(describe(unknown, d_ref=1.0)["calibre_shares"]["length"]["below_2"], 1.0)

    def test_transverse_spacing_of_a_square_array(self):
        n, spacing, length = 4, 3.0, 20.0
        nodes = square_array(n, spacing, length, axis=2)
        frame = frame_record(axis=(0, 0, 1), origin=(0.0, 0.0, length / 2.0))
        box = (n * spacing, n * spacing, length)
        r = describe(nodes, frame=frame, d_ref=1.0, volume=box)["transverse_spacing"]
        self.assertEqual((r["planes"], r["count"], r["area_source"]), (9, 9 * n * n, "argument"))
        for key in ("median", "mean", "p10", "p90", "median_d", "mean_d", "p10_d", "p90_d"):
            self.assertAlmostEqual(r[key], spacing, msg=key)
        self.assertAlmostEqual(r["crossing_density"], 1.0 / spacing ** 2)
        self.assertAlmostEqual(r["crossing_density_d"], 1.0 / spacing ** 2)
        # d_ref scales the _d forms only
        r = describe(nodes, frame=frame, d_ref=2.0, volume=box)["transverse_spacing"]
        self.assertAlmostEqual(r["median"], spacing)
        self.assertAlmostEqual(r["median_d"], spacing / 2.0)
        self.assertAlmostEqual(r["crossing_density_d"], 4.0 / spacing ** 2)
        # the growth box of the metadata serves when no volume is given
        r = describe(nodes, frame=frame, d_ref=1.0, metadata={"growth_box_um": list(box)})["transverse_spacing"]
        self.assertEqual(r["area_source"], "growth_box_um")
        self.assertAlmostEqual(r["crossing_density"], 1.0 / spacing ** 2)
        # without either, the rectangle bounding the capillary end points
        r = describe(nodes, frame=frame, d_ref=1.0)["transverse_spacing"]
        self.assertEqual(r["area_source"], "capillary_bounding_box")
        self.assertAlmostEqual(r["crossing_density"], n ** 2 / ((n - 1) * spacing) ** 2)
        self.assertAlmostEqual(r["median"], spacing)
        # the box follows the frame: with the axis along x the extents still
        # run along (g x p, g, p), which are x, y and z
        across = square_array(n, spacing, length, axis=0)
        frame_x = frame_record(axis=(1, 0, 0), origin=(length / 2.0, 0.0, 0.0))
        r = describe(across, frame=frame_x, d_ref=1.0, volume=(length, n * spacing, n * spacing))["transverse_spacing"]
        self.assertEqual(r["count"], 9 * n * n)
        self.assertAlmostEqual(r["median"], spacing)
        self.assertAlmostEqual(r["crossing_density"], 1.0 / spacing ** 2)
        # the heading along the axis: the in-plane axes come from the perpendicular instead
        frame_g = frame_record(axis=(0, 0, 1), origin=(0.0, 0.0, length / 2.0), grow_direction=(0, 0, 1),
                               grow_perpendicular=(1, 0, 0))
        r = describe(nodes, frame=frame_g, d_ref=1.0)["transverse_spacing"]
        self.assertAlmostEqual(r["crossing_density"], n ** 2 / ((n - 1) * spacing) ** 2)
        # None without an axis frame or a reference diameter, without a
        # capillary edge, and when the capillaries have no extent along the axis
        self.assertIsNone(describe(nodes, frame=frame)["transverse_spacing"])
        self.assertIsNone(describe(nodes, frame=frame_record(kind="plane"), d_ref=1.0)["transverse_spacing"])
        self.assertIsNone(describe(nodes, frame=frame, d_ref=0.1)["transverse_spacing"])
        flat = describe(archive(straight(axis=1)), frame=frame_record(axis=(1, 0, 0)), d_ref=10.0)
        self.assertIsNone(flat["transverse_spacing"])
        # end points spread along the axis but no edge crossing a plane:
        # nothing pooled, and a density of zero over the bounding rectangle
        r = describe(nodes, frame=frame_record(axis=(1, 0, 0)), d_ref=1.0)["transverse_spacing"]
        self.assertEqual((r["count"], r["median"], r["crossing_density"]), (0, None, 0.0))
        # a bounding rectangle without area gives no density
        r = describe(archive([(0, 0, 0, 1.0), (0, 0, 10, 1.0)]), frame=frame, d_ref=1.0)["transverse_spacing"]
        self.assertEqual((r["count"], r["area_source"]), (0, "capillary_bounding_box"))
        self.assertIsNone(r["crossing_density"])

    def test_transverse_spacing_does_not_depend_on_the_lab_axes(self):
        rng = np.random.default_rng(2)
        nodes = square_array(4, 3.0, 20.0)
        frame = frame_record(axis=(0, 0, 1), origin=(0.0, 0.0, 10.0))
        turned_nodes, turned_frame = rotated(nodes, frame, rotation(rng))
        for volume in ((12.0, 12.0, 20.0), None):
            with self.subTest(volume=volume):
                reference = describe(nodes, frame=frame, d_ref=1.0, volume=volume)["transverse_spacing"]
                turned = describe(turned_nodes, frame=turned_frame, d_ref=1.0, volume=volume)["transverse_spacing"]
                self.assertEqual(turned["count"], reference["count"])
                self.assertEqual(turned["area_source"], reference["area_source"])
                values = dict(numeric_leaves(turned))
                for key, value in numeric_leaves(reference):
                    self.assertAlmostEqual(values[key], value, places=9, msg=key)

    def test_section_area_of_an_oblique_plane(self):
        # one vessel along (1, 1, 0) through a cube of side 2 centred on the
        # origin: the plane at s = sqrt(2) t cuts the cube in a rectangle of
        # 2 by 2 sqrt(2) (1 - |t|), and the nine planes sit at t = -0.8 .. 0.8,
        # so the sections sum to 20 sqrt(2); the single crossing per plane is
        # not pooled but counts towards the density
        nodes = archive([(-1.0, -1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)])
        frame = frame_record(axis=(1.0, 1.0, 0.0), origin=(0.0, 0.0, 0.0))
        r = describe(nodes, frame=frame, d_ref=1.0, volume=(2.0, 2.0, 2.0))["transverse_spacing"]
        self.assertEqual((r["count"], r["median"], r["area_source"]), (0, None, "argument"))
        self.assertAlmostEqual(r["crossing_density"], 9.0 / (20.0 * math.sqrt(2.0)))
        self.assertAlmostEqual(r["crossing_density_d"], 9.0 / (20.0 * math.sqrt(2.0)))
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)

    def test_segments_by_class_on_a_y(self):
        # the Y's segments: the parent's two edges of diameter 3.5 and 2.5,
        # each of length 1, and the child of length 2 and diameter 2.125
        # (edges of 2.5 and 1.75); at d_ref 1.5 the bound falls at 3
        r = describe(Y_NODES, d_ref=1.5)["segments_by_class"]
        capillary = r["capillary"]
        self.assertEqual(capillary["count"], 2)
        self.assertAlmostEqual(capillary["median"], 1.5)
        self.assertAlmostEqual(capillary["p10"], 1.1)
        self.assertAlmostEqual(capillary["p90"], 1.9)
        self.assertAlmostEqual(capillary["cv"], 1.0 / 3.0)
        self.assertAlmostEqual(capillary["median_d"], 1.0)
        self.assertAlmostEqual(capillary["p10_d"], 1.1 / 1.5)
        self.assertAlmostEqual(capillary["p90_d"], 1.9 / 1.5)
        larger = r["larger"]
        self.assertEqual((larger["count"], larger["median"], larger["p10"], larger["p90"]), (1, 1.0, 1.0, 1.0))
        self.assertIsNone(larger["cv"])
        self.assertAlmostEqual(larger["median_d"], 1.0 / 1.5)
        # every segment capillary at a large d_ref: the larger entry is empty
        r = describe(Y_NODES, d_ref=10.0)["segments_by_class"]
        self.assertEqual(r["larger"], EMPTY_LENGTHS)
        self.assertEqual(r["capillary"]["count"], 3)
        self.assertAlmostEqual(r["capillary"]["median"], 1.0)
        # the bound moves with class_bound
        r = describe(Y_NODES, d_ref=1.0, class_bound=3.0)["segments_by_class"]
        self.assertEqual((r["capillary"]["count"], r["larger"]["count"]), (2, 1))
        r = describe(Y_NODES, d_ref=1.0)["segments_by_class"]
        self.assertEqual((r["capillary"]["count"], r["larger"]["count"]), (0, 3))
        self.assertIsNone(describe(Y_NODES)["segments_by_class"])
        # a segment of unknown diameter is left out
        unknown = archive([(0, 0, 0, np.nan), (1, 0, 0, np.nan)], [(0, 5, 0, 1.0), (1, 5, 0, 1.0)])
        r = describe(unknown, d_ref=1.0)["segments_by_class"]
        self.assertEqual((r["capillary"]["count"], r["larger"]["count"]), (1, 0))

    def test_orientation_by_class_measures_order_and_planarity(self):
        planar = archive(straight(axis=0), straight(axis=1, offset=(0.0, 100.0, 0.0)))
        r = describe(planar, d_ref=1.0)
        by_class = r["orientation_by_class"]
        self.assertEqual(list(by_class), ["all", "capillary", "larger"])
        self.assertAlmostEqual(by_class["all"]["planarity"], 1.0)
        self.assertAlmostEqual(by_class["all"]["S_max"], 0.25)
        # "all" is the "orientation" entry with the two keys appended
        self.assertEqual({k: v for k, v in by_class["all"].items() if k not in ("S_max", "planarity")},
                         r["orientation"])
        self.assertEqual(list(by_class["all"])[-2:], ["S_max", "planarity"])
        # vessels of diameter 4 at d_ref 1 are all larger-class
        self.assertEqual(by_class["larger"], by_class["all"])
        self.assertIsNone(by_class["capillary"]["eigenvalues"])
        self.assertIsNone(by_class["capillary"]["S_max"])
        self.assertIsNone(by_class["capillary"]["planarity"])
        line = describe(archive(straight()), d_ref=1.0)["orientation_by_class"]["larger"]
        self.assertAlmostEqual(line["S_max"], 1.0)
        self.assertAlmostEqual(line["planarity"], 1.0)
        isotropic = archive(straight(axis=0), straight(axis=1, offset=(0, 100, 0)),
                            straight(axis=2, offset=(0, 0, 200)))
        entry = describe(isotropic, d_ref=1.0)["orientation_by_class"]["all"]
        self.assertAlmostEqual(entry["S_max"], 0.0)
        self.assertAlmostEqual(entry["planarity"], 0.0)
        self.assertIsNone(describe(planar)["orientation_by_class"])


class ClassAndFrameSourceTests(unittest.TestCase):
    def test_reference_diameter_sources_in_order_of_preference(self):
        record = {"d_min": 2.5, "grow_kwargs": {"d_min": 1.5}}
        r = describe(Y_NODES, metadata=record, d_ref=1.0)
        self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": 1.0, "d_ref_source": "argument"})
        r = describe(Y_NODES, metadata=record)
        self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": 2.5, "d_ref_source": "d_min"})
        # a d_min that is not a positive finite number (a bool is not a number) is passed over
        for d_min in (None, True, False, float("nan"), float("inf"), 0.0, -1.0, "2.5"):
            with self.subTest(d_min=d_min):
                r = describe(Y_NODES, metadata={"d_min": d_min, "grow_kwargs": {"d_min": 1.5}})
                self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": 1.5, "d_ref_source": "grow_kwargs"})
        # the thinnest vertex is never the fallback: without a source every class key is None
        polar = frame_record(axis=(0, 1, 0), sense="polar", tree_senses=[1])
        tree = np.zeros(Y_NODES.shape[1], dtype=np.int8)
        for record in ({}, {"d_min": None}, {"d_min": True, "grow_kwargs": {"d_min": False}},
                       {"grow_kwargs": {"d_min": float("nan")}}, {"grow_kwargs": None}, {"grow_kwargs": 3.0}):
            with self.subTest(record=record):
                r = describe(Y_NODES, metadata=record, frame=polar, tree=tree)
                self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": None, "d_ref_source": None})
                for key in ("orientation_by_class", "calibre_shares", "segments_by_class", "transverse_spacing"):
                    self.assertIsNone(r[key], key)
                self.assertIsNotNone(r["frame_orientation"]["all"])
                self.assertIsNone(r["frame_orientation"]["capillary"])
                self.assertIsNone(r["frame_orientation"]["larger"])
                self.assertIsNotNone(r["polar_order"]["all"])
                self.assertIsNone(r["polar_order"]["capillary"])
                self.assertIsNone(r["polar_order"]["larger"])
        # with a source the classes follow the bound: the Y's edges have
        # diameters 3.5, 2.5, 2.5 and 1.75, each of length 1
        r = describe(Y_NODES, d_ref=1.0, class_bound=3.0, frame=polar)
        self.assertEqual(r["classes"], {"bound": 3.0, "d_ref": 1.0, "d_ref_source": "argument"})
        self.assertEqual(r["frame_orientation"]["capillary"]["length"], 3.0)
        self.assertEqual(r["frame_orientation"]["larger"]["length"], 1.0)
        self.assertEqual(r["frame_orientation"]["all"]["length_d"], 4.0)

    def test_invalid_arguments_are_rejected(self):
        for d_ref in (0.0, -1.0, float("nan"), float("inf"), True, "1"):
            with self.subTest(d_ref=d_ref), self.assertRaises(ValueError):
                describe(Y_NODES, d_ref=d_ref)
        for bound in (0.0, -2.0, float("nan"), True):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                describe(Y_NODES, d_ref=1.0, class_bound=bound)
        bad_frames = ("axis", dict(frame_record(), kind="spiral"), dict(frame_record(), axis=[0.0, 0.0, 0.0]),
                      dict(frame_record(), axis=None), dict(frame_record(kind="plane"), normal=[1.0, 2.0]))
        for frame in bad_frames:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                describe(Y_NODES, frame=frame)
        # a frame the metadata holds is checked the same way
        with self.assertRaises(ValueError):
            describe(Y_NODES, metadata={"frame": dict(frame_record(), axis=[1.0, float("nan"), 0.0])})
        # the box needs the frame's origin and heading
        box = square_array(2, 3.0, 10.0)
        with self.assertRaises(ValueError):
            describe(box, frame=dict(frame_record(axis=(0, 0, 1)), origin=None), d_ref=1.0, volume=(6.0, 6.0, 10.0))

    def test_archive_without_a_frame_has_no_frame_relative_keys(self):
        r = describe(Y_NODES)
        self.assertEqual(tuple(r), OLD_KEYS + NEW_KEYS)
        for key in NEW_KEYS:
            if key != "classes":
                self.assertIsNone(r[key], key)
        self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": None, "d_ref_source": None})
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        # the fixtures predate frames and record no d_min
        r = describe_archive(FIXTURES[0])
        self.assertIsNone(r["frame"])
        self.assertIsNone(r["frame_orientation"])
        self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": None, "d_ref_source": None})
        self.assertIsNone(r["calibre_shares"])

    def test_frame_of_kind_none_is_copied_but_sets_nothing(self):
        record = frame_record(kind="none")
        tree = np.zeros(Y_NODES.shape[1], dtype=np.int8)
        r = describe(Y_NODES, metadata={"frame": record, "d_min": 1.5}, tree=tree)
        self.assertEqual(r["frame"], record)
        # the copy is independent of the record
        r["frame"]["rules"].append("x")
        self.assertEqual(record["rules"], [])
        for key in ("frame_orientation", "polar_order", "transverse_spacing"):
            self.assertIsNone(r[key], key)
        # the classes do not need a frame
        self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": 1.5, "d_ref_source": "d_min"})
        self.assertIsNotNone(r["calibre_shares"])
        self.assertIsNotNone(r["orientation_by_class"])
        self.assertIsNotNone(r["segments_by_class"])

    def test_frame_argument_overrides_the_metadata(self):
        stored = frame_record(axis=(1, 0, 0))
        given = frame_record(axis=(0, 1, 0))
        nodes = archive(straight())
        r = describe(nodes, metadata={"frame": stored})
        self.assertEqual(r["frame"], stored)
        self.assertEqual(r["frame_orientation"]["all"]["S"], 1.0)
        r = describe(nodes, metadata={"frame": stored}, frame=given)
        self.assertEqual(r["frame"], given)
        self.assertEqual(r["frame_orientation"]["all"]["S"], -0.5)
        # a frame given as the argument is copied too, numpy vectors and all
        array_frame = dict(given, axis=np.array([0.0, 1.0, 0.0]))
        r = describe(nodes, frame=array_frame)
        self.assertEqual(r["frame"], given)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        # describe_archive passes the three arguments through
        record = {"frame": stored, "d_min": 2.0, "units": "um"}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "framed.npz")
            np.savez(path, nodes=nodes, metadata=np.array(json.dumps(record)))
            r = describe_archive(path)
            self.assertEqual(r["frame"], stored)
            self.assertEqual(r["classes"], {"bound": 2.0, "d_ref": 2.0, "d_ref_source": "d_min"})
            r = describe_archive(path, frame=given, d_ref=1.0, class_bound=5.0)
        self.assertEqual(r, describe(nodes, metadata=record, frame=given, d_ref=1.0, class_bound=5.0))
        self.assertEqual(r["classes"], {"bound": 5.0, "d_ref": 1.0, "d_ref_source": "argument"})
        self.assertEqual(r["frame"], given)

    def test_new_keys_are_unit_invariant(self):
        grown = small_tree()
        nodes, edges, tree = grown["nodes"], grown["edges"], grown["tree"]
        frame = frame_record(axis=(0, 1, 0), sense="polar", tree_senses=[1], origin=(0.0, 200.0, 0.0))
        volume = (500.0, 400.0, 300.0)
        base = describe(nodes, edges, tree=tree, frame=frame, d_ref=5.0, volume=volume)
        scaled = describe(nodes * 3.0, edges, tree=tree, frame=dict(frame, origin=[0.0, 600.0, 0.0]), d_ref=15.0,
                          volume=tuple(3.0 * v for v in volume))
        # every new quantity is defined on this network
        for key in NEW_KEYS:
            self.assertIsNotNone(base[key], key)
        self.assertGreater(base["transverse_spacing"]["count"], 0)
        self.assertIsNotNone(base["transverse_spacing"]["crossing_density"])
        self.assertGreater(base["frame_orientation"]["capillary"]["length"], 0.0)
        self.assertGreater(base["frame_orientation"]["larger"]["length"], 0.0)
        self.assertGreater(base["segments_by_class"]["capillary"]["count"], 1)
        self.assertGreater(base["segments_by_class"]["larger"]["count"], 1)
        # lengths (the class totals, the segment and spacing statistics, the
        # reference diameter and the origin) scale with the unit, the crossing
        # density with its inverse square, and everything dimensionless or in
        # d_ref stays the same
        base_leaves = dict(numeric_leaves({key: base[key] for key in NEW_KEYS}))
        scaled_leaves = dict(numeric_leaves({key: scaled[key] for key in NEW_KEYS}))
        self.assertEqual(set(base_leaves), set(scaled_leaves))
        lengths = ("length", "median", "mean", "p10", "p90")
        for key, value in base_leaves.items():
            if key.split(".")[-1] in lengths or key == ".classes.d_ref" or key.startswith(".frame.origin"):
                factor = 3.0
            elif key.endswith(".crossing_density"):
                factor = 1.0 / 9.0
            else:
                factor = 1.0
            self.assertAlmostEqual(scaled_leaves[key], factor * value, delta=1e-9 * max(1.0, abs(factor * value)),
                                   msg=key)
        self.assertEqual(json.loads(json.dumps(base, allow_nan=False)), base)

    def test_old_keys_are_unchanged_by_the_new_arguments(self):
        grown = small_tree()
        plain = describe(grown["nodes"], grown["edges"], tree=grown["tree"])
        frame = frame_record(axis=(0, 1, 0), sense="polar", tree_senses=[1], origin=(0.0, 200.0, 0.0))
        extended = describe(grown["nodes"], grown["edges"], tree=grown["tree"], frame=frame, d_ref=5.0,
                            class_bound=1.5)
        self.assertEqual(tuple(plain)[:len(OLD_KEYS)], OLD_KEYS)
        self.assertEqual({key: plain[key] for key in OLD_KEYS}, {key: extended[key] for key in OLD_KEYS})
        # the arguments only fill the keys after them
        self.assertIsNone(plain["frame"])
        self.assertEqual(plain["classes"], {"bound": 2.0, "d_ref": None, "d_ref_source": None})
        self.assertEqual(extended["frame"], frame)
        self.assertEqual(extended["classes"], {"bound": 1.5, "d_ref": 5.0, "d_ref_source": "argument"})
        self.assertEqual(extended["frame_orientation"]["kind"], "axis")
        self.assertIsNotNone(extended["polar_order"]["all"])
        self.assertEqual(extended["transverse_spacing"]["area_source"], "capillary_bounding_box")
        self.assertEqual(json.loads(json.dumps(extended, allow_nan=False)), extended)


if __name__ == "__main__":
    unittest.main()
