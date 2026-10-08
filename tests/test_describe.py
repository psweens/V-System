"""
Tests for the descriptors of a centreline network.

Run from the repository root with
    python -m unittest tests.test_describe -v
"""
import contextlib
import glob
import io
import itertools
import json
import math
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import computeVoxel  # noqa: E402
import describe as describe_module  # noqa: E402
import graph  # noqa: E402
from collisions import KIN_REACH  # noqa: E402
from describe import (FLAG_KEYS, POLAR_BINS, AZIMUTH_BINS, describe, clearance,  # noqa: E402
                      load_archive, describe_archive, main, LOOP_SAMPLES, LOOP_DEPTH, LOOP_VERTICES,
                      EVD_MAX_POINTS)
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
# and those appended in 3.6
NEW_KEYS_3_6 = ("junctions", "branch_angles_deg", "loops", "segment_diameter_variation", "tissue_distance")
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
        self.assertIsNone(entry["fisher_axial_K_exact"])

        across = describe(archive(straight(axis=1)), frame=frame_record(axis=(1, 0, 0)))["frame_orientation"]["all"]
        self.assertEqual((across["S"], across["mean_abs_cos"]), (-0.5, 0.0))
        self.assertAlmostEqual(across["mean_angle_deg"], 90.0)
        self.assertIsNone(across["crossing_ratio"])
        self.assertEqual((across["within_20_deg"], across["within_45_deg"]), (0.0, 0.0))
        self.assertIsNone(across["watson_K"])
        self.assertIsNone(across["fisher_axial_K"])
        self.assertIsNone(across["fisher_axial_K_exact"])

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
        # the Fisher-axial law's own <|u|> is 1/2 at K = 0, where the polar law's needs K of about 1.8
        self.assertAlmostEqual(entry["fisher_axial_K_exact"], 0.0, places=9)
        self.assertAlmostEqual(entry["fisher_axial_K"], 1.7968, places=3)
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
        # the exact inversion recovers the law's own concentration more closely
        self.assertLess(abs(entry["fisher_axial_K_exact"] - 5.0) / 5.0, 0.05)
        self.assertLess(abs(entry["fisher_axial_K_exact"] - 5.0), abs(entry["fisher_axial_K"] - 5.0))

    def test_the_exact_fisher_axial_concentration_inverts_its_moment_and_reads_zero_for_isotropic_tangents(self):
        from describe import _axial_mean, _fisher_axial_exact_concentration
        for k in (-150.0, -20.0, -1.0, -1e-5, 0.0, 3e-5, 1e-2, 0.5, 5.0, 40.0, 150.0):
            with self.subTest(k=k):
                # the moment by quadrature, against its closed form and series
                u = (np.arange(200000) + 0.5) / 200000
                weights = np.exp(k * (u - 1.0))
                self.assertAlmostEqual(_axial_mean(k), float(u @ weights / weights.sum()), places=8)
                self.assertAlmostEqual(_fisher_axial_exact_concentration(_axial_mean(k)), k, delta=1e-9 * max(1.0, abs(k)))
        # the series and the closed form meet at the switch
        for k in (-1e-2, 1e-2):
            self.assertAlmostEqual(_axial_mean(k * (1 - 1e-12)), _axial_mean(k * (1 + 1e-12)), places=13)
        rng = np.random.default_rng(7)
        nodes = unit_polylines(directions_about_z(rng, rng.uniform(-1.0, 1.0, 10000)), rng)
        entry = describe(nodes, frame=frame_record(axis=(0, 0, 1)))["frame_orientation"]["all"]
        self.assertLess(abs(entry["fisher_axial_K_exact"]), 0.1)
        self.assertGreater(entry["fisher_axial_K"], 1.6)
        # tangents gathered across the axis give a negative concentration
        cosines = rejection_cosines(rng, 10000, lambda u: -5.0 * np.abs(u))
        across = describe(unit_polylines(directions_about_z(rng, cosines), rng),
                          frame=frame_record(axis=(0, 0, 1)))["frame_orientation"]["all"]
        self.assertLess(abs(across["fisher_axial_K_exact"] + 5.0) / 5.0, 0.1)

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
        self.assertEqual(tuple(r), OLD_KEYS + NEW_KEYS + NEW_KEYS_3_6)
        for key in NEW_KEYS:
            if key != "classes":
                self.assertIsNone(r[key], key)
        # the topology keys need neither a frame nor d_ref; the tissue distance is asked for
        for key in NEW_KEYS_3_6[:-1]:
            self.assertIsNotNone(r[key], key)
        self.assertIsNone(r["tissue_distance"])
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
        # the tissue distance changes nothing else, the keys of 3.5 included
        with_tissue = describe(grown["nodes"], grown["edges"], tree=grown["tree"], frame=frame, d_ref=5.0,
                               class_bound=1.5, evd_spacing=8.0)
        self.assertEqual(tuple(with_tissue), OLD_KEYS + NEW_KEYS + NEW_KEYS_3_6)
        self.assertEqual({key: value for key, value in with_tissue.items() if key != "tissue_distance"},
                         {key: value for key, value in extended.items() if key != "tissue_distance"})
        self.assertIsNone(extended["tissue_distance"])
        self.assertGreater(with_tissue["tissue_distance"]["outside_points"], 0)
        # the arguments only fill the keys after them
        self.assertIsNone(plain["frame"])
        self.assertEqual(plain["classes"], {"bound": 2.0, "d_ref": None, "d_ref_source": None})
        self.assertEqual(extended["frame"], frame)
        self.assertEqual(extended["classes"], {"bound": 1.5, "d_ref": 5.0, "d_ref_source": "argument"})
        self.assertEqual(extended["frame_orientation"]["kind"], "axis")
        self.assertIsNotNone(extended["polar_order"]["all"])
        self.assertEqual(extended["transverse_spacing"]["area_source"], "capillary_bounding_box")
        self.assertEqual(json.loads(json.dumps(extended, allow_nan=False)), extended)


# ---------------------------------------------------------------------------
# 3.6: junctions, branch angles, loops, diameter variation and tissue distance
# ---------------------------------------------------------------------------

def graph_archive(points, edges, diameter=1.0):
    """
    An archive with one polyline per edge (i, j) of a graph on `points`,
    through the edge's midpoint, so that every edge of the graph is a
    segment of two archive edges; columns coincide where edges meet.
    """
    runs = []
    for i, j in edges:
        p, q = np.asarray(points[i], dtype=float), np.asarray(points[j], dtype=float)
        runs.append([tuple(p), tuple((p + q) / 2.0), tuple(q)])
    return archive(*runs, diameter=diameter)


def ladder_graph(rungs, spacing=1.0, width=1.5):
    """
    A ladder of rungs `width` long, `spacing` apart along x, its rails
    continued by a pendant edge past each end rung so that every rung end is
    a junction of degree three.
    """
    points = [(k * spacing, y, 0.0) for k in range(rungs) for y in (0.0, width)]
    edges = [(2 * k, 2 * k + 1) for k in range(rungs)]
    edges += [(2 * k + side, 2 * k + 2 + side) for k in range(rungs - 1) for side in (0, 1)]
    for k, x in ((0, -spacing), (rungs - 1, rungs * spacing)):
        for side, y in ((0, 0.0), (1, width)):
            edges.append((2 * k + side, len(points)))
            points.append((x, y, 0.0))
    return points, edges


def lattice_graph(shape, spacing=1.0):
    """
    A square (two extents) or cubic (three) lattice, each boundary vertex
    given a pendant tip half a spacing long towards every missing neighbour,
    so that every lattice vertex has the bulk degree. Returns the points,
    the edges and the number of lattice edges, which come first.
    """
    index, points = {}, []
    for cell in itertools.product(*(range(n) for n in shape)):
        index[cell] = len(points)
        points.append(tuple(spacing * float(c) for c in cell) + (0.0,) * (3 - len(shape)))
    edges = [(index[cell], index[cell[:axis] + (cell[axis] + 1,) + cell[axis + 1:]])
             for cell in index for axis in range(len(shape)) if cell[axis] + 1 < shape[axis]]
    n_lattice = len(edges)
    for cell in index:
        for axis in range(len(shape)):
            for sign in (-1, 1):
                if not 0 <= cell[axis] + sign < shape[axis]:
                    tip = list(points[index[cell]])
                    tip[axis] += sign * spacing / 2.0
                    edges.append((index[cell], len(points)))
                    points.append(tuple(tip))
    return points, edges, n_lattice


def honeycomb_graph(bricks, rows, side=1.0):
    """
    A sheet of regular hexagons of the given side, `rows` rows of `bricks`
    in the brick-wall layout, so that every edge lies on a hexagon; each
    vertex of degree two is given a pendant tip along z. Returns the points,
    the edges, and the numbers of hexagon edges, which come first, and of
    hexagon vertices.
    """
    index, points, edges = {}, [], []

    def vertex(i, j):
        if (i, j) not in index:
            index[(i, j)] = len(points)
            points.append((i * side * math.sqrt(3.0) / 2.0, 1.5 * j * side - (0.5 * side if (i + j) % 2 else 0.0),
                           0.0))
        return index[(i, j)]

    for j in range(rows):
        for i in range(j % 2, 2 * bricks + j % 2, 2):
            ring = [(i, j), (i + 1, j), (i + 2, j), (i + 2, j + 1), (i + 1, j + 1), (i, j + 1)]
            for p, q in zip(ring, ring[1:] + ring[:1]):
                edge = tuple(sorted((vertex(*p), vertex(*q))))
                if edge not in edges:
                    edges.append(edge)
    n_hexagon, n_vertices = len(edges), len(points)
    degree = np.bincount(np.array(edges).ravel(), minlength=n_vertices)
    for v in np.flatnonzero(degree == 2):
        edges.append((int(v), len(points)))
        points.append(points[v][:2] + (side / 2.0,))
    return points, edges, n_hexagon, n_vertices


def binary_tree_graph(depth, length=4.0, angle=0.6):
    """A planar binary tree of the given depth, each daughter 0.8 times its parent's length."""
    points, edges = [(0.0, 0.0, 0.0), (0.0, length, 0.0)], [(0, 1)]
    ends = [(1, np.array([0.0, length, 0.0]), np.array([0.0, 1.0, 0.0]), length)]
    for _ in range(depth):
        following = []
        for v, position, heading, size in ends:
            for turn in (angle, -angle):
                c, s = math.cos(turn), math.sin(turn)
                direction = np.array([c * heading[0] - s * heading[1], s * heading[0] + c * heading[1], 0.0])
                end = position + 0.8 * size * direction
                edges.append((v, len(points)))
                points.append(tuple(end))
                following.append((len(points) - 1, end, direction, 0.8 * size))
        ends = following
    return points, edges


def ring_graph(n, radius=10.0):
    """n vertices on a circle joined in a ring, each with a pendant tip outwards: a loop of n segments."""
    angle = 2.0 * np.pi * np.arange(n) / n
    ring = [(radius * math.cos(a), radius * math.sin(a), 0.0) for a in angle]
    tips = [(1.1 * x, 1.1 * y, 0.0) for x, y, _ in ring]
    return ring + tips, [(k, (k + 1) % n) for k in range(n)] + [(k, n + k) for k in range(n)]


# Theta: junctions U and V joined by one long segment of length 10, by U-X-V
# (two segments of sqrt(0.5)) and by U-Y-Z-V (sqrt(1.0625), 0.5 and
# sqrt(1.0625)); X, Y and Z carry pendant tips. The shortest loop through U
# counted in segments is the long segment and U-X-V, three segments 11.41
# long; the shortest by length is U-X-V-Z-Y-U, five segments 3.98 long.
_U, _V, _X, _Y, _Z = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, -0.5, 0.0), (0.25, -1.0, 0.0), (0.75, -1.0, 0.0)
THETA_NODES = archive([_U, (0.0, 4.5, 0.0), (1.0, 4.5, 0.0), _V], [_U, _X], [_X, _V], [_U, _Y], [_Y, _Z], [_Z, _V],
                      [_X, (0.5, -0.5, 1.0)], [_Y, (0.25, -1.0, 1.0)], [_Z, (0.75, -1.0, 1.0)], diameter=0.2)
THETA_LOOP = 2.0 * math.sqrt(0.5) + 2.0 * math.sqrt(1.0625) + 0.5


def star_archive(directions, length=10.0, n_points=6, diameter=1.0):
    """Straight arms from the origin along each direction, each a polyline of n_points equally spaced columns."""
    runs = []
    for direction in directions:
        unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        runs.append([tuple(t * unit) for t in np.linspace(0.0, length, n_points)])
    return archive(*runs, diameter=diameter)


def _sphere_points(n, centre, radius):
    """n points spread evenly over a sphere (a Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * k / n)
    azimuth = np.pi * (1.0 + math.sqrt(5.0)) * k
    return np.asarray(centre) + radius * np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth),
                                                   np.cos(polar)], axis=1)


def _spokes(hub, ends, lift):
    """
    Two polylines from the hub to each end, bowed apart by `lift` (a pair of
    parallel segments, a loop of two), and a small closed loop at each end,
    which makes the end a junction whose own shortest loop is found at once.
    """
    runs = []
    for p in ends:
        side = np.cross(p - hub, [0.0, 0.0, 1.0])
        side = side / np.linalg.norm(side) * lift if np.linalg.norm(side) > 1e-9 else np.array([lift, 0.0, 0.0])
        runs += [[hub, (hub + p) / 2.0 + side, p], [hub, (hub + p) / 2.0 - side, p]]
    for p in ends:
        out = (p - hub) / np.linalg.norm(p - hub) * lift
        runs.append([p, p + out, p + out + np.array([0.0, 0.0, lift]), p])
    return [[tuple(point) for point in run] for run in runs]


def segment_cap_archive(n_spokes):
    """
    A hub a with n_spokes neighbours on the unit sphere, each joined to it
    by a pair of parallel segments, and a segment e from a to b, which joins
    the nearest of them and carries a tip. The search through e reaches all
    n_spokes neighbours of a, its two ends besides, before it meets b's side.
    """
    a, b = np.zeros(3), np.array([4.0, 0.0, 0.0])
    ends = _sphere_points(n_spokes, a, 1.0)
    nearest = ends[np.argmax(ends[:, 0])]
    runs = [[tuple(a), tuple((a + b) / 2.0), tuple(b)], [tuple(b), tuple((b + nearest) / 2.0), tuple(nearest)],
            [tuple(b), (4.0, -1.0, 0.0)]] + _spokes(a, ends, 0.002)
    return archive(*runs, diameter=1e-3)


def node_cap_archive(n_spokes):
    """
    A junction s joined to u nearby and to z far away, u joined to z, and
    n_spokes neighbours of u on a small sphere about it, each joined to it by
    a pair of parallel segments. From s every one of them is nearer than z,
    which closes the only loop through s, s-u-z-s.
    """
    s, u, z = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 60.0, 0.0])
    runs = [[tuple(s), (5.0, 30.0, 0.0), tuple(z)], [tuple(s), (0.5, 0.0, 0.0), tuple(u)],
            [tuple(u), tuple((u + z) / 2.0), tuple(z)], [tuple(s), (0.0, -1.0, 0.0)], [tuple(z), (0.0, 61.0, 0.0)]]
    runs += _spokes(u, _sphere_points(n_spokes, u, 0.3), 0.001)
    return archive(*runs, diameter=1e-3)


def frontier_archive(n_spokes):
    """
    A segment e from a hub a to b, b joined to c and c to the last of
    n_spokes neighbours of a on the unit sphere, which closes a loop of four
    through e. Each neighbour p is joined to a, and to a vertex 2p of its
    own beyond it, by a pair of parallel segments. After one level from each
    end of e, a's side holds the n_spokes neighbours and b's side c alone:
    going on from the smaller side meets a's side at once, while going on
    from a's side would reach the vertices beyond its neighbours first.
    """
    a, b, c = np.zeros(3), np.array([4.0, 0.0, 0.0]), np.array([3.0, -1.0, 0.0])
    ends = _sphere_points(n_spokes, a, 1.0)
    runs = [[tuple(a), tuple(b)], [tuple(b), tuple(c)], [tuple(b), (5.0, 0.0, 0.0)], [tuple(c), (3.0, -2.0, 0.0)]]
    # the neighbours come in order, so the last one is the last a's side reaches
    runs += _spokes(a, ends, 0.002)
    for p in ends:
        runs += _spokes(p, [2.0 * p], 0.002)
    runs.append([tuple(c), tuple(ends[-1])])
    return archive(*runs, diameter=1e-3)


def depth_cap_archive():
    """
    A junction s whose shortest loop closes at a vertex its search reaches
    on the depth cap: a chain of 16 segments 0.01 long from s to x, a
    segment 0.1 long from x to y and one about 2.99 long from y back to s,
    a loop of 18 segments about 3.25 long. s is also joined to q by a pair
    of parallel segments about 2.57 long, a loop of two about 5.14 long,
    which the search from s closes before it reaches y. Every junction but
    s carries a tip.

    Returns:
        tuple: (nodes, pairs, lengths, n_junctions), as random_multigraph
        returns them, s numbered 0.
    """
    points = [(0.01 * k, 0.0, 0.0) for k in range(17)] + [(0.16, 0.1, 0.0), (0.0, -2.5, 0.0)]
    s, x, y, q = points[0], points[16], points[17], points[18]
    runs = [[points[k - 1], points[k]] for k in range(1, 17)]
    runs += [[x, y], [s, (0.0, 0.0, -1.4), (0.16, 0.1, -1.4), y], [s, (0.3, -1.25, 0.0), q],
             [s, (-0.3, -1.25, 0.0), q]]
    pairs = [(k - 1, k) for k in range(1, 17)] + [(16, 17), (0, 17), (0, 18), (0, 18)]
    for v in range(1, 19):
        runs.append([points[v], points[v][:2] + (0.005,)])
        pairs.append((v, 18 + v))
    lengths = [sum(math.dist(u, w) for u, w in zip(run, run[1:])) for run in runs]
    return archive(*runs), pairs, lengths, 19


def random_multigraph(rng, n_vertices=14, n_edges=22, n_parallel=3, n_loops=2):
    """
    A random multigraph with parallel edges and self-loops, every vertex
    brought to degree three or more with pendant tips, drawn as an archive
    whose segments are exactly its edges: an edge is a polyline through a
    point off its chord, a self-loop a closed polyline through two points.

    Returns:
        tuple: (nodes, pairs, lengths, n_vertices), the edges as vertex
        pairs (a tip's far end numbered from n_vertices up) and their arc
        lengths; the vertices 0..n_vertices-1 are the junctions.
    """
    points = list(rng.uniform(0.0, 10.0, (n_vertices, 3)))
    pairs = [tuple(rng.choice(n_vertices, 2, replace=False)) for _ in range(n_edges)]
    pairs += [pairs[k] for k in rng.choice(len(pairs), n_parallel, replace=False)]
    pairs += [(v, v) for v in rng.choice(n_vertices, n_loops, replace=False)]
    degree = np.zeros(n_vertices, dtype=np.int64)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    for v in range(n_vertices):
        for _ in range(max(0, 3 - int(degree[v]))):
            pairs.append((v, len(points)))
            points.append(points[v] + rng.normal(0.0, 1.0, 3))
    runs, lengths = [], []
    for i, j in pairs:
        p, q = points[i], points[j]
        if i == j:
            run = [p, p + rng.normal(0.0, 1.0, 3), p + rng.normal(0.0, 1.0, 3), p]
        elif j >= n_vertices:
            run = [p, q]
        else:
            run = [p, (p + q) / 2.0 + rng.normal(0.0, 1.0, 3), q]
        runs.append([tuple(point) for point in run])
        lengths.append(sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(run, run[1:])))
    return archive(*runs), pairs, lengths, n_vertices


def brute_force_loops(pairs, lengths, n_junctions):
    """
    Reference for "loops", without caps, on a multigraph given by its edges:
    the size of the shortest loop through each edge, by a plain one-way
    breadth-first search (0 for none), and the (length, segments) of the
    shortest loop through each junction by length, then by segments, as the
    least over the edges e = (s, u) at s of e followed by a shortest path
    from u back to s that avoids e, and over the self-loops at s, each path
    found by a plain Dijkstra search on (length, segments); (inf, 0) for none.
    """
    by_segment = []
    for e, (i, j) in enumerate(pairs):
        if i == j:
            by_segment.append(1)
            continue
        if any({k, m} == {i, j} for f, (k, m) in enumerate(pairs) if f != e):
            by_segment.append(2)
            continue
        level = {i: 0}
        queue = [i]
        while queue and j not in level:
            following = []
            for x in queue:
                for f, (k, m) in enumerate(pairs):
                    if f == e or k == m or x not in (k, m):
                        continue
                    y = m if x == k else k
                    if y not in level:
                        level[y] = level[x] + 1
                        following.append(y)
            queue = following
        by_segment.append(level[j] + 1 if j in level else 0)
    by_node = []
    for s in range(n_junctions):
        best = (math.inf, 0)
        for e, (i, j) in enumerate(pairs):
            if s not in (i, j):
                continue
            if i == j:
                best = min(best, (lengths[e], 1))
                continue
            start = j if i == s else i
            distance = {start: (lengths[e], 1)}
            done = set()
            while True:
                waiting = [v for v in distance if v not in done]
                if not waiting:
                    break
                x = min(waiting, key=lambda v: distance[v])
                done.add(x)
                if x == s:
                    best = min(best, distance[s])
                    break
                for f, (k, m) in enumerate(pairs):
                    if f == e or k == m or x not in (k, m):
                        continue
                    y = m if x == k else k
                    candidate = (distance[x][0] + lengths[f], distance[x][1] + 1)
                    if y not in done and (y not in distance or candidate < distance[y]):
                        distance[y] = candidate
        by_node.append(best if math.isfinite(best[0]) else (math.inf, 0))
    return by_segment, by_node


def random_capsule_archive(rng, n_polylines=8):
    """Random polylines of random diameters, some of them zero, for the tissue distance."""
    runs = []
    for _ in range(n_polylines):
        position = rng.uniform(0.0, 20.0, 3)
        run = []
        for _ in range(int(rng.integers(2, 10))):
            diameter = rng.choice([np.nan, 0.0, rng.uniform(0.2, 3.0)], p=[0.05, 0.05, 0.9])
            run.append(tuple(position) + (float(diameter),))
            position = position + rng.normal(0.0, 2.5, 3)
        runs.append(run)
    return archive(*runs)


def brute_force_tissue(nodes, axes):
    """
    Reference for "tissue_distance" on the grid with the given axes: the
    value of every capsule of the voxeliser at every grid point, minimised.
    """
    x, y, z = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    best = np.full(grid.shape[0], np.inf)
    for c in range(nodes.shape[1] - 1):
        if not (np.all(np.isfinite(nodes[:3, c])) and np.all(np.isfinite(nodes[:3, c + 1]))):
            continue
        p0, p1 = nodes[:3, c], nodes[:3, c + 1]
        if np.array_equal(p0, p1):
            continue
        r0, r1 = (max(0.0, float(np.nan_to_num(nodes[3, k] / 2.0))) for k in (c, c + 1))
        axis = p1 - p0
        t = np.clip((grid - p0) @ axis / (axis @ axis), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(grid - (p0 + t[:, None] * axis), axis=1) - (r0 + t * (r1 - r0)))
    return best


def grid_axes(low, high, spacing):
    """The grid "tissue_distance" documents, when the spacing is not raised."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    counts = [max(1, int(math.floor((h - l) / spacing + 1e-9))) for l, h in zip(low, high)]
    centre = (low + high) / 2.0
    return [centre[i] + (np.arange(counts[i]) - (counts[i] - 1) / 2.0) * spacing for i in range(3)]


def loop_histogram(sizes):
    """The "histogram" describe reports for a list of loop sizes, 0 meaning no loop."""
    found = [size for size in sizes if size]
    return {str(size): found.count(size) for size in sorted(set(found))}


def cylinder_mean_distance(a, r):
    """
    The mean distance to the wall of a cylinder of radius r on the axis of a
    square prism of side a, over the prism outside the cylinder: the mean
    distance from the centre of a square, a (sqrt 2 + ln(1 + sqrt 2)) / 6,
    taken over the square less the disc, less r.
    """
    square = a ** 3 * (math.sqrt(2.0) + math.log(1.0 + math.sqrt(2.0))) / 6.0
    outside = a * a - math.pi * r * r
    return (square - 2.0 * math.pi * r ** 3 / 3.0 - r * outside) / outside


class TopologyDescriptorTests(unittest.TestCase):
    def test_loops_of_a_ladder_are_four_segments(self):
        rungs, spacing, width = 20, 1.0, 1.5
        r = describe(graph_archive(*ladder_graph(rungs, spacing, width)))
        # the rungs, the rail edges between them and four pendant tips
        n_segments = rungs + 2 * (rungs - 1) + 4
        by_segment = r["loops"]["by_segment"]
        self.assertEqual({key: by_segment[key] for key in ("segments", "every", "sampled", "found", "cycles")},
                         {"segments": n_segments, "every": 1, "sampled": n_segments, "found": n_segments - 4,
                          "cycles": rungs - 1})
        self.assertEqual(by_segment["histogram"], {"4": n_segments - 4})
        self.assertEqual((by_segment["median"], by_segment["mean"]), (4.0, 4.0))
        self.assertAlmostEqual(by_segment["none_fraction"], 4.0 / n_segments, delta=1e-15)
        total = rungs * width + 2 * (rungs - 1) * spacing + 4 * spacing
        self.assertAlmostEqual(by_segment["cycles_per_length"], (rungs - 1) / total, delta=1e-15)
        self.assertIsNone(by_segment["cycles_per_length_d"])
        by_node = r["loops"]["by_node"]
        self.assertEqual({key: by_node[key] for key in ("junctions", "every", "sampled", "found", "none_fraction")},
                         {"junctions": 2 * rungs, "every": 1, "sampled": 2 * rungs, "found": 2 * rungs,
                          "none_fraction": 0.0})
        self.assertEqual(by_node["histogram"], {"4": 2 * rungs})
        for key in ("length_median", "length_mean"):
            self.assertAlmostEqual(by_node[key], 2.0 * spacing + 2.0 * width, delta=1e-12, msg=key)
        self.assertIsNone(by_node["length_median_d"])
        self.assertEqual(r["junctions"], {"count": 2 * rungs, "degree_3": 1.0, "degree_4": 0.0, "degree_5_plus": 0.0,
                                          "mean_degree": 3.0, "segments_per_junction": n_segments / (2 * rungs)})
        # the entries in d_ref
        loops = describe(graph_archive(*ladder_graph(rungs, spacing, width)), d_ref=0.5)["loops"]
        self.assertAlmostEqual(loops["by_segment"]["cycles_per_length_d"], 0.5 * (rungs - 1) / total, delta=1e-15)
        self.assertAlmostEqual(loops["by_node"]["length_median_d"], (2.0 * spacing + 2.0 * width) / 0.5, delta=1e-12)

    def test_loops_of_a_square_lattice_are_four_segments(self):
        side, spacing = 5, 2.0
        points, edges, n_lattice = lattice_graph((side, side), spacing)
        r = describe(graph_archive(points, edges))
        by_segment, by_node = r["loops"]["by_segment"], r["loops"]["by_node"]
        self.assertEqual(n_lattice, 2 * side * (side - 1))
        self.assertEqual((by_segment["segments"], by_segment["histogram"]), (len(edges), {"4": n_lattice}))
        self.assertEqual(by_segment["cycles"], (side - 1) ** 2)
        self.assertEqual((by_node["junctions"], by_node["histogram"]), (side * side, {"4": side * side}))
        self.assertAlmostEqual(by_node["length_median"], 4.0 * spacing, delta=1e-12)
        self.assertEqual((r["junctions"]["degree_4"], r["junctions"]["mean_degree"]), (1.0, 4.0))

    def test_loops_of_a_hexagonal_sheet_are_six_segments(self):
        bricks, rows, side = 3, 3, 1.5
        points, edges, n_hexagon, n_vertices = honeycomb_graph(bricks, rows, side)
        r = describe(graph_archive(points, edges))
        by_segment, by_node = r["loops"]["by_segment"], r["loops"]["by_node"]
        self.assertEqual((by_segment["histogram"], by_segment["cycles"]), ({"6": n_hexagon}, bricks * rows))
        self.assertEqual((by_node["junctions"], by_node["histogram"]), (n_vertices, {"6": n_vertices}))
        self.assertAlmostEqual(by_node["length_median"], 6.0 * side, delta=1e-12)
        self.assertEqual(r["junctions"]["degree_3"], 1.0)

    def test_loops_of_a_cubic_lattice_are_four_segments(self):
        side = 3
        points, edges, n_lattice = lattice_graph((side, side, side))
        r = describe(graph_archive(points, edges))
        by_segment, by_node = r["loops"]["by_segment"], r["loops"]["by_node"]
        self.assertEqual(n_lattice, 3 * side * side * (side - 1))
        self.assertEqual((by_segment["histogram"], by_segment["cycles"]), ({"4": n_lattice}, n_lattice - side ** 3 + 1))
        self.assertEqual((by_node["junctions"], by_node["histogram"]), (side ** 3, {"4": side ** 3}))
        self.assertAlmostEqual(by_node["length_median"], 4.0, delta=1e-12)
        self.assertEqual((r["junctions"]["degree_5_plus"], r["junctions"]["mean_degree"]), (1.0, 6.0))

    def test_a_tree_has_no_loops(self):
        depth = 6
        r = describe(graph_archive(*binary_tree_graph(depth)))
        n_segments, n_junctions = 2 ** (depth + 1) - 1, 2 ** depth - 1
        self.assertEqual(r["loops"]["by_segment"],
                         {"segments": n_segments, "every": 1, "sampled": n_segments, "found": 0, "none_fraction": 1.0,
                          "median": None, "mean": None, "histogram": {}, "cycles": 0, "cycles_per_length": 0.0,
                          "cycles_per_length_d": None})
        self.assertEqual(r["loops"]["by_node"],
                         {"junctions": n_junctions, "every": 1, "sampled": n_junctions, "found": 0,
                          "none_fraction": 1.0, "median": None, "mean": None, "histogram": {}, "length_median": None,
                          "length_mean": None, "length_median_d": None, "length_mean_d": None})
        self.assertEqual(r["junctions"]["segments_per_junction"], n_segments / n_junctions)

    def test_by_node_picks_the_shortest_loop_by_length_not_by_count(self):
        r = describe(THETA_NODES)["loops"]
        # counted in segments, the long segment closes a loop of three with U-X-V
        # and the segments through Y and Z lie on loops of four
        self.assertEqual(r["by_segment"]["histogram"], {"3": 3, "4": 3})
        # by length, every junction's shortest loop is U-X-V-Z-Y, five segments
        self.assertEqual(r["by_node"]["histogram"], {"5": 5})
        for key in ("length_median", "length_mean"):
            self.assertAlmostEqual(r["by_node"][key], THETA_LOOP, delta=1e-12, msg=key)

    def test_self_loops_and_parallel_segments_close_one_and_two_segment_loops(self):
        # junction j: a closed loop through two points, two parallel segments
        # to k (lengths 2 sqrt(0.26) and 2 sqrt(0.29)) and a tip; k has a tip
        j, k = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        pair = 2.0 * math.sqrt(0.26) + 2.0 * math.sqrt(0.29)
        for size, through_j in ((1.0, 2), (0.5, 1)):
            loop = (1.0 + math.sqrt(5.0)) * size
            with self.subTest(loop=loop):
                nodes = archive([j, (-size, size / 2.0, 0.0), (-size, -size / 2.0, 0.0), j], [j, (0.5, 0.1, 0.0), k],
                                [j, (0.5, -0.2, 0.0), k], [j, (0.0, 0.0, 1.0)], [k, (1.0, 0.0, 1.0)])
                r = describe(nodes)
                self.assertEqual(r["loops"]["by_segment"]["histogram"], {"1": 1, "2": 2})
                self.assertEqual(r["junctions"]["degree_5_plus"], 0.5)
                # j takes the shorter of its own loop and the pair, k the pair
                self.assertEqual(r["loops"]["by_node"]["histogram"], {"1": 1, "2": 1} if through_j == 1 else {"2": 2})
                self.assertAlmostEqual(r["loops"]["by_node"]["length_mean"], (min(loop, pair) + pair) / 2.0,
                                       delta=1e-12)

    def test_loops_agree_with_an_independent_search(self):
        # development measurement, seeds 0-9: every histogram and count equal,
        # loop lengths within 2.2e-16 relatively, loops of up to 8 segments
        for seed in range(10):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                nodes, pairs, lengths, n_junctions = random_multigraph(rng)
                r = describe(nodes)["loops"]
                by_segment, by_node = brute_force_loops(pairs, lengths, n_junctions)
                # small enough that no search reaches a cap, so the capped
                # searches must agree with the uncapped reference
                self.assertLessEqual(max(by_segment), LOOP_DEPTH + 1)
                self.assertLessEqual(max(size for _, size in by_node), LOOP_DEPTH + 1)
                self.assertEqual(r["by_segment"]["segments"], len(pairs))
                self.assertEqual(r["by_segment"]["histogram"], loop_histogram(by_segment))
                self.assertEqual(r["by_segment"]["found"], sum(1 for size in by_segment if size))
                self.assertEqual(r["by_node"]["junctions"], n_junctions)
                self.assertEqual(r["by_node"]["histogram"], loop_histogram([size for _, size in by_node]))
                found = [length for length, size in by_node if size]
                self.assertEqual(r["by_node"]["found"], len(found))
                for key, value in (("length_median", float(np.median(found))), ("length_mean", float(np.mean(found)))):
                    self.assertAlmostEqual(r["by_node"][key], value, delta=1e-12 * value, msg=key)
                # the median and mean of the loop sizes, which differ on every
                # one of seeds 0-9 in both entries
                for key, sizes in (("by_segment", by_segment), ("by_node", [size for _, size in by_node])):
                    sizes = [size for size in sizes if size]
                    self.assertEqual((r[key]["median"], r[key]["mean"]),
                                     (float(np.median(sizes)), float(np.mean(sizes))), msg=key)

    def test_loop_caps(self):
        self.assertEqual((LOOP_SAMPLES, LOOP_DEPTH, LOOP_VERTICES), (2000, 16, 4096))
        # depth: a ring of n segments; from a segment the two sides meet
        # after at most 16 segments between them, from a junction each side
        # goes at most 16 segments out
        for n, through_segment, through_junction in ((17, True, True), (18, False, True), (33, False, True),
                                                     (34, False, False)):
            with self.subTest(ring=n):
                r = describe(graph_archive(*ring_graph(n)))["loops"]
                self.assertEqual(r["by_segment"]["histogram"], {str(n): n} if through_segment else {})
                self.assertEqual(r["by_node"]["histogram"], {str(n): n} if through_junction else {})

        # vertices: from the segment e of segment_cap_archive, its hub's side
        # reaches the hub's neighbours before the far side can meet it, so
        # e's loop of three is found with 4094 of them (4096 vertices with
        # e's ends) and not with 4095
        for n_spokes, found in ((4094, True), (4095, False)):
            with self.subTest(spokes=n_spokes):
                nodes = segment_cap_archive(n_spokes)
                canonical = graph.canonical_columns(nodes)
                edges = graph.edges_from_nodes(nodes, canonical=canonical)
                degree = graph.degree(edges, nodes.shape[1])
                paths = graph.segments(nodes, edges, canonical)
                # e runs from the hub, column 0, to b, column 2, and comes first
                self.assertEqual((paths[0][0], paths[0][-1]), (0, 2))
                ends = [frozenset((path[0], path[-1])) for path in paths]
                r = describe(nodes)["loops"]["by_segment"]
                expected = []
                for k in range(0, len(paths), r["every"]):
                    path = paths[k]
                    if path[0] == path[-1]:
                        expected.append(1)
                    elif ends.count(ends[k]) > 1:
                        expected.append(2)
                    elif degree[path[0]] == 1 or degree[path[-1]] == 1:
                        expected.append(0)
                    else:
                        # e, or the segment from b to the hub's nearest neighbour
                        expected.append(3 if k or found else 0)
                self.assertEqual(r["histogram"], loop_histogram(expected))
                self.assertEqual(r["histogram"].get("3", 0) >= 1, found)
        # and from a junction: every one of node_cap_archive's 4093 or 4094
        # spokes is settled before z, which closes the only loop through s
        for n_spokes, found in ((4093, True), (4094, False)):
            with self.subTest(spokes=n_spokes):
                r = describe(node_cap_archive(n_spokes))["loops"]["by_node"]
                # s is the first junction and so sampled; every spoke end
                # sampled finds the small loop on itself, of one segment
                self.assertEqual(r["every"], 3)
                self.assertEqual(r["histogram"], {"1": r["sampled"] - 1, "3": 1} if found else {"1": r["sampled"] - 1})

    def test_by_segment_goes_on_from_the_smaller_frontier(self):
        # from e, a's side holds 2100 neighbours after one level and b's side
        # one vertex; going on from b's side meets a's side at c's neighbour
        # with 2103 vertices reached, where going on from a's side, as
        # alternating by depth would, reaches 4096 among the vertices beyond
        nodes = frontier_archive(2100)
        canonical = graph.canonical_columns(nodes)
        edges = graph.edges_from_nodes(nodes, canonical=canonical)
        degree = graph.degree(edges, nodes.shape[1])
        paths = graph.segments(nodes, edges, canonical)
        # e runs from a, column 0, to b, column 1, and comes first
        self.assertEqual((paths[0][0], paths[0][-1]), (0, 1))
        ends = [frozenset((path[0], path[-1])) for path in paths]
        r = describe(nodes)["loops"]["by_segment"]
        expected = []
        for k in range(0, len(paths), r["every"]):
            path = paths[k]
            if path[0] == path[-1]:
                expected.append(1)
            elif ends.count(ends[k]) > 1:
                expected.append(2)
            elif degree[path[0]] == 1 or degree[path[-1]] == 1:
                expected.append(0)
            else:
                # e, b-c or c's segment to a's last neighbour, all on the loop of four
                expected.append(4)
        self.assertEqual(r["histogram"], loop_histogram(expected))

    def test_by_node_goes_on_past_a_vertex_on_the_depth_cap(self):
        # x is settled 16 segments from s and not searched beyond, so the
        # loop through it, 18 segments 3.25 long, closes only when y, 2.99
        # from s, is settled: after the pair has closed a loop of 5.14, less
        # than twice 2.99. Every junction's loop is that of the uncapped
        # search; development measurement: lengths within 6.7e-16 relatively
        nodes, pairs, lengths, n_junctions = depth_cap_archive()
        _, by_node = brute_force_loops(pairs, lengths, n_junctions)
        self.assertEqual(by_node[0][1], 18)
        self.assertAlmostEqual(by_node[0][0], 0.26 + 2.8 + math.hypot(0.16, 0.1), delta=1e-12)
        r = describe(nodes)["loops"]["by_node"]
        self.assertEqual(r["histogram"], {"2": 1, "18": 18})
        self.assertEqual(r["histogram"], loop_histogram([size for _, size in by_node]))
        found = [length for length, size in by_node if size]
        for key, value in (("length_median", float(np.median(found))), ("length_mean", float(np.mean(found)))):
            self.assertAlmostEqual(r[key], value, delta=1e-12 * value, msg=key)

    def test_loop_sampling_takes_every_kth_up_to_2000(self):
        rungs = 1100
        nodes = graph_archive(*ladder_graph(rungs))
        r = describe(nodes)["loops"]
        n_segments, n_junctions = 3 * rungs + 2, 2 * rungs
        self.assertEqual((r["by_segment"]["every"], r["by_segment"]["sampled"]), (2, (n_segments + 1) // 2))
        self.assertEqual((r["by_node"]["every"], r["by_node"]["sampled"]), (2, n_junctions // 2))
        self.assertEqual(r["by_node"]["histogram"], {"4": n_junctions // 2})
        # every other segment in the order of graph.segments, the tips among them without a loop
        canonical = graph.canonical_columns(nodes)
        edges = graph.edges_from_nodes(nodes, canonical=canonical)
        degree = graph.degree(edges, nodes.shape[1])
        sample = graph.segments(nodes, edges, canonical)[::2]
        tips = sum(1 for path in sample if degree[path[0]] == 1 or degree[path[-1]] == 1)
        self.assertEqual(r["by_segment"]["histogram"], {"4": len(sample) - tips})

    def test_junction_fractions_and_angles_of_a_symmetric_y(self):
        third = 2.0 * math.pi / 3.0
        r = describe(star_archive([(math.cos(k * third), math.sin(k * third), 0.0) for k in range(3)]))
        self.assertEqual(r["junctions"], {"count": 1, "degree_3": 1.0, "degree_4": 0.0, "degree_5_plus": 0.0,
                                          "mean_degree": 3.0, "segments_per_junction": 3.0})
        angles = r["branch_angles_deg"]
        self.assertEqual((angles["count"], angles["skipped"]), (1, 0))
        for key in ("min", "median", "max"):
            self.assertAlmostEqual(angles[key], 120.0, delta=1e-9, msg=key)
        # a parent and two daughters each 40 degrees off its continuation
        off = math.radians(40.0)
        angles = describe(star_archive([(0.0, -1.0, 0.0), (math.sin(off), math.cos(off), 0.0),
                                        (-math.sin(off), math.cos(off), 0.0)]))["branch_angles_deg"]
        for key, value in (("min", 80.0), ("median", 140.0), ("max", 140.0)):
            self.assertAlmostEqual(angles[key], value, delta=1e-9, msg=key)
        # junctions of every degree class, the segments counted once
        r = describe(star_archive([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]))["junctions"]
        self.assertEqual((r["degree_3"], r["degree_4"], r["mean_degree"], r["segments_per_junction"]),
                         (0.0, 1.0, 4.0, 4.0))
        self.assertIsNone(describe(star_archive([(1, 0, 0), (-1, 0, 0)]))["junctions"]["degree_3"])

    def test_branch_angles_of_a_t_and_the_arc_rule(self):
        # arms of diameter 1 along -x and +x and a stem along +y; at 2 diameters
        # out the arms point along their first edges
        def angles(stem, diameter=1.0):
            nodes = archive([(0, 0, 0), (-5, 0, 0)], [(0, 0, 0), (5, 0, 0)], stem, diameter=diameter)
            r = describe(nodes)["branch_angles_deg"]
            return r["count"], r["skipped"], r["min"], r["median"], r["max"]

        t = angles([(0, 0, 0), (0, 5, 0)])
        self.assertEqual(t[:2], (1, 0))
        for value, expected in zip(t[2:], (90.0, 90.0, 180.0)):
            self.assertAlmostEqual(value, expected, delta=1e-12)
        # a bend beyond 2 diameters does not change the angles, one before it does
        self.assertEqual(angles([(0, 0, 0), (0, 2.5, 0), (4, 2.5, 0)]), t)
        bent = angles([(0, 0, 0), (0, 1, 0), (4, 1, 0)])
        for value, expected in zip(bent[2:], (45.0, 135.0, 180.0)):
            self.assertAlmostEqual(value, expected, delta=1e-12)
        # a stem shorter than 4 diameters is measured halfway along it
        self.assertEqual(angles([(0, 0, 0), (0, 1, 0), (1, 1, 0)]), t)
        # a junction of unknown diameter is skipped
        self.assertEqual(angles([(0, 0, 0), (0, 5, 0)], diameter=float("nan")), (0, 1, None, None, None))

    def test_branch_angles_are_medians_over_the_junctions(self):
        # four junctions 30 apart, each a parent and two daughters the given
        # angle a off its continuation, whose sorted angles are 2 a and twice
        # 180 - a below 60 degrees, twice 180 - a and 2 a above: the medians
        # over the junctions are 85, 135 and 152.5, the means 80, 132.5 and
        # 147.5; development measurement: within 5.7e-14 degrees
        offs = (20.0, 35.0, 55.0, 80.0)
        runs = []
        for k, off in enumerate(offs):
            a = math.radians(off)
            for dx, dy in ((0.0, -1.0), (math.sin(a), math.cos(a)), (-math.sin(a), math.cos(a))):
                runs.append([(30.0 * k + t * dx, t * dy, 0.0) for t in np.linspace(0.0, 10.0, 6)])
        r = describe(archive(*runs))["branch_angles_deg"]
        self.assertEqual((r["count"], r["skipped"]), (4, 0))
        angles = np.sort([[2.0 * off, 180.0 - off, 180.0 - off] for off in offs], axis=1)
        for k, key in enumerate(("min", "median", "max")):
            self.assertAlmostEqual(r[key], float(np.median(angles[:, k])), delta=1e-12, msg=key)

    def test_segment_diameter_variation_of_one_stenosis(self):
        diameters = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]
        nodes = archive([(float(k), 0.0, 0.0, d) for k, d in enumerate(diameters)])
        r = describe(nodes, d_ref=1.0)["segment_diameter_variation"]
        # (1 - 0.5) / (6.5 / 7) = 7 / 13, a capillary at d_ref 1
        stenosis = {"count": 1, "excluded": 0, "median": 7.0 / 13.0, "p90": 7.0 / 13.0}
        self.assertEqual(r, {"all": stenosis, "capillary": stenosis,
                             "larger": {"count": 0, "excluded": 0, "median": None, "p90": None}})
        self.assertEqual(describe(nodes)["segment_diameter_variation"], {"all": stenosis, "capillary": None,
                                                                         "larger": None})

    def test_junction_ends_are_left_out_of_diameter_variation(self):
        # a Murray Y: a parent of diameter 2 and daughters of 2 / 2^(1/3); the
        # junction takes the parent's diameter, which is left out of the daughters
        daughter, off = 2.0 / 2.0 ** (1.0 / 3.0), math.radians(37.5)
        parent = [(0.0, -5.0 + k, 0.0, 2.0) for k in range(6)]
        nodes = archive(parent, [(t * math.sin(off), t * math.cos(off), 0.0, daughter) for t in range(6)],
                        [(-t * math.sin(off), t * math.cos(off), 0.0, daughter) for t in range(6)])
        r = describe(nodes)["segment_diameter_variation"]["all"]
        self.assertEqual(r, {"count": 3, "excluded": 0, "median": 0.0, "p90": 0.0})
        # the same step inside one vessel counts
        nodes = archive(parent + [(0.0, float(t), 0.0, daughter) for t in range(1, 6)])
        r = describe(nodes)["segment_diameter_variation"]["all"]
        self.assertAlmostEqual(r["median"], (2.0 - daughter) / ((12.0 + 5.0 * daughter) / 11.0), delta=1e-15)
        # a segment between two junctions with nothing between them has no value
        r = describe(graph_archive(*ladder_graph(3)))["segment_diameter_variation"]["all"]
        self.assertEqual((r["count"], r["excluded"]), (4, 7))
        # a ring's first vertex, repeated at its end, counts once: (2 - 1) / 1.75
        ring = archive([(10.0, 0.0, 0.0, 1.0), (0.0, 10.0, 0.0, 2.0), (-10.0, 0.0, 0.0, 2.0), (0.0, -10.0, 0.0, 2.0),
                        (10.0, 0.0, 0.0, 1.0)])
        r = describe(ring)["segment_diameter_variation"]["all"]
        self.assertAlmostEqual(r["median"], 4.0 / 7.0, delta=1e-15)

    def test_segment_diameter_variation_takes_the_median_and_p90_over_the_segments(self):
        # five vessels of diameter 1 each narrowed to `dip` at one of seven
        # vertices, (1 - dip) / ((6 + dip) / 7), and a vessel of diameter 2,
        # which at d_ref 1 lies on the class bound and so is larger-class.
        # Over the five, the median is 0.424 and the p90 0.804, where the
        # mean is 0.469 and the p80 0.706; development measurement: equal
        # to the expressions here
        dips = (0.9, 0.75, 0.6, 0.4, 0.2)
        runs = [[(float(i), 10.0 * k, 0.0, dip if i == 3 else 1.0) for i in range(7)] for k, dip in enumerate(dips)]
        runs.append([(float(i), -10.0, 0.0, 2.0) for i in range(7)])
        r = describe(archive(*runs), d_ref=1.0)["segment_diameter_variation"]
        values = [(1.0 - dip) / ((6.0 + dip) / 7.0) for dip in dips]
        self.assertEqual(r["larger"], {"count": 1, "excluded": 0, "median": 0.0, "p90": 0.0})
        for key, expected in (("capillary", values), ("all", values + [0.0])):
            self.assertEqual((r[key]["count"], r[key]["excluded"]), (len(expected), 0), msg=key)
            for statistic, value in zip(("median", "p90"), np.percentile(expected, [50.0, 90.0])):
                self.assertAlmostEqual(r[key][statistic], float(value), delta=1e-15, msg=key + " " + statistic)

    def test_a_straight_cylinder_has_the_analytic_tissue_distance(self):
        radius = 1.0
        side = 4.0 * radius
        nodes = archive([(0.0, 0.0, 0.0), (side, 0.0, 0.0)], diameter=2.0 * radius)
        r = describe(nodes, volume=(side, side, side), evd_spacing=side / 64.0)["tissue_distance"]
        self.assertEqual((r["shape"], r["points"], r["raised"], r["capsules"]), ([64, 64, 64], 64 ** 3, False, 1))
        self.assertEqual(r["domain"], {"min": [0.0, -side / 2.0, -side / 2.0], "max": [side, side / 2.0, side / 2.0],
                                       "source": "argument"})
        # measured +0.214 % on the mean and +0.964 % on the inside fraction,
        # the error of counting a disc 16 grid spacings across on a lattice
        self.assertLess(abs(r["mean"] / cylinder_mean_distance(side, radius) - 1.0), 0.01)
        self.assertLess(abs(r["inside_fraction"] / (math.pi * radius ** 2 / side ** 2) - 1.0), 0.02)
        self.assertEqual(r["outside_points"], round(64 ** 3 * (1.0 - r["inside_fraction"])))
        # the farthest point is a corner cell of the grid
        corner = math.hypot(side / 2.0 - side / 128.0, side / 2.0 - side / 128.0) - radius
        self.assertAlmostEqual(r["max"], corner, delta=1e-12)

    def test_tissue_distance_equals_brute_force(self):
        # development measurement, seeds 0-9: the five statistics within
        # 2.5e-16 relatively and the counts of points outside equal
        spacing = 1.0
        for seed in range(10):
            with self.subTest(seed=seed):
                nodes = random_capsule_archive(np.random.default_rng(seed))
                r = describe(nodes, evd_spacing=spacing)["tissue_distance"]
                points = nodes[:3, np.isfinite(nodes[0])].T
                r_max = float(np.nanmax(nodes[3])) / 2.0
                axes = grid_axes(points.min(axis=0) - r_max, points.max(axis=0) + r_max, spacing)
                self.assertEqual((r["shape"], r["raised"]), ([axis.size for axis in axes], False))
                values = brute_force_tissue(nodes, axes)
                outside = values[values > 0.0]
                self.assertEqual(r["outside_points"], outside.size)
                self.assertEqual(r["inside_fraction"], float(values.size - outside.size) / values.size)
                for key, value in (("mean", outside.mean()), ("median", np.percentile(outside, 50.0)),
                                   ("p90", np.percentile(outside, 90.0)), ("p99", np.percentile(outside, 99.0)),
                                   ("max", outside.max())):
                    self.assertAlmostEqual(r[key], float(value), delta=1e-12 * float(value), msg=key)

    def test_a_pass_of_the_tissue_search_holds_at_most_its_budget_of_pairs(self):
        # development measurement, seed 5 at spacing 0.5: a pass of 4096
        # points held up to 72126 pairs of a point and a cell or piece; with a
        # budget of 4096 the passes shrank to as few as 128 points, and the
        # result was the same
        nodes = random_capsule_archive(np.random.default_rng(5))
        expected = describe(nodes, evd_spacing=0.5)["tissue_distance"]
        held = []
        run_minimum = describe_module._run_minimum

        def recording(index, values, n):
            held.append((index.size, n))
            return run_minimum(index, values, n)

        with mock.patch.object(describe_module, "_EVD_PAIRS", 4096):
            with mock.patch.object(describe_module, "_run_minimum", recording):
                r = describe(nodes, evd_spacing=0.5)["tissue_distance"]
        self.assertEqual(r, expected)
        self.assertLessEqual(max(size for size, _ in held), 4096)
        self.assertLess(min(n for _, n in held), describe_module._EVD_CHUNK)
        # and at the finest level: 3000 copies of one short capsule share a
        # cell, so that two grid points beside them would hold 6000 pairs of
        # a point and a piece, and are searched one at a time instead
        start, end = np.zeros((3000, 3)), np.tile([0.001, 0.0, 0.0], (3000, 1))
        radii = np.full(3000, 0.1)
        axes = [np.array([1.0, 2.0]), np.zeros(1), np.zeros(1)]
        expected = describe_module._tissue_values(axes, start, end, radii, radii, 1.0)
        held.clear()
        with mock.patch.object(describe_module, "_EVD_PAIRS", 4096):
            with mock.patch.object(describe_module, "_run_minimum", recording):
                values = describe_module._tissue_values(axes, start, end, radii, radii, 1.0)
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(max(size for size, _ in held), 3000)

    def test_inside_is_the_voxelisers_rule(self):
        # grid points on a vessel's wall are inside: a vessel of radius 0.5
        # along x, sampled every 0.5 across it from -1 to 1, has 3 of every 5
        # points inside and the other 2 at 0.5 from its wall
        nodes = archive([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)], diameter=1.0)
        r = describe(nodes, volume=(4.0, 2.5, 0.5), evd_spacing=0.5)["tissue_distance"]
        self.assertEqual((r["shape"], r["outside_points"], r["mean"], r["max"]), ([8, 5, 1], 16, 0.5, 0.5))
        self.assertEqual(r["inside_fraction"], 0.6)
        # the points inside are the voxels computeVoxel sets for the same
        # capsules at the same centres; development measurement, seeds 0-9:
        # equal counts on every seed, 700 to 1384 points inside in the growth
        # box and 710 to 1551 in the voxel_size field
        shape = (40, 36, 30)
        for seed in range(10):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                runs = []
                for _ in range(6):
                    position = rng.uniform(5.0, 25.0, 3)
                    run = []
                    for _ in range(int(rng.integers(2, 8))):
                        run.append(tuple(position) + (float(rng.uniform(0.5, 6.0)),))
                        position = position + rng.normal(0.0, 3.0, 3)
                    runs.append(run)
                nodes = archive(*runs)
                # the growth box's grid points sit at the voxel centres shifted by half a voxel
                r = describe(nodes, metadata={"growth_box_um": list(shape)}, evd_spacing=1.0)["tissue_distance"]
                self.assertEqual(r["shape"], list(shape))
                voxels = computeVoxel.rasterise_segments(nodes[:3] - 0.5, nodes[3] / 2.0, shape, connect=False)
                self.assertEqual(r["points"] - r["outside_points"], int(voxels.sum()))
                # the voxel_size field's grid points, at its voxel size, are its voxel centres
                metadata = {"fit": "voxel_size", "volume": list(shape), "voxel_size": 1.0}
                r = describe(nodes, metadata=metadata, evd_spacing=1.0)["tissue_distance"]
                self.assertEqual(r["shape"], list(shape))
                voxels = computeVoxel.process_network(nodes, shape, fit="voxel_size", voxel_size=1.0, connect=False)
                self.assertEqual(r["points"] - r["outside_points"], int(voxels.sum()))
        # a missing or negative diameter counts as zero, where the voxeliser
        # would draw nothing of a capsule with a missing diameter and less of
        # one with a negative diameter
        for diameters in ((6.0, float("nan")), (float("nan"), 6.0), (6.0, -2.0)):
            with self.subTest(diameters=diameters):
                nodes = archive([(5.0, 10.0, 10.0, diameters[0]), (35.0, 10.0, 10.0, diameters[1])])
                r = describe(nodes, metadata={"growth_box_um": [40.0, 20.0, 20.0]}, evd_spacing=1.0)["tissue_distance"]
                radii = np.where(nodes[3] > 0.0, nodes[3] / 2.0, 0.0)
                voxels = computeVoxel.rasterise_segments(nodes[:3] - 0.5, radii, (40, 20, 20), connect=False)
                self.assertEqual(r["points"] - r["outside_points"], int(voxels.sum()))

    def test_tissue_distance_raises_the_spacing_to_the_point_cap(self):
        self.assertEqual(EVD_MAX_POINTS, 64 ** 3)
        nodes = archive([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], diameter=0.5)
        # stepping on through the spacings E_i / (n_i - 1) at which an axis
        # loses a point can pass the least E_i / m that fits: from the
        # requested spacing it reaches 1474 / 175 in the second box, and from
        # where a bisection on the spacing stops, 1903.404 / 140 in the third
        for extents, spacing, expected in (((1000.0, 100.0, 10.0), 1.0, None),
                                           ((1389.0, 1474.0, 84.0), 0.75, (1389.0 / 165.0, [165, 175, 9])),
                                           ((1048.052, 1903.404, 331.111), 1.346, (1903.404 / 141.0, [77, 141, 24]))):
            with self.subTest(extents=extents):
                r = describe(nodes, volume=extents, evd_spacing=spacing)["tissue_distance"]
                box = np.subtract(r["domain"]["max"], r["domain"]["min"]).tolist()

                def size(h):
                    return int(np.prod([max(1, math.floor(e / h + 1e-9)) for e in box]))

                # the least E_i / m above the requested spacing at which the grid fits
                candidates = sorted({e / m for e in box for m in range(1, int(e / spacing) + 2) if e / m > spacing})
                least = next(h for h in candidates if size(h) <= EVD_MAX_POINTS)
                self.assertEqual((r["spacing_requested"], r["raised"]), (spacing, True))
                self.assertEqual(r["spacing"], least)
                self.assertEqual(r["shape"], [max(1, math.floor(e / least + 1e-9)) for e in box])
                self.assertEqual(r["points"], size(least))
                self.assertLessEqual(r["points"], EVD_MAX_POINTS)
                if expected is not None:
                    self.assertEqual((r["spacing"], r["shape"]), expected)
        # a spacing the cap allows is kept
        r = describe(nodes, volume=(1000.0, 100.0, 10.0), evd_spacing=50.0)["tissue_distance"]
        self.assertEqual((r["spacing"], r["raised"], r["shape"]), (50.0, False, [20, 2, 1]))

    def test_tissue_distance_domain_follows_the_volume_source(self):
        nodes = archive([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 4.0, 0.0)], diameter=1.0)
        cases = [({}, {}, [-0.5, -0.5, -0.5], [10.5, 4.5, 0.5], "bounding_box", [11, 5, 1]),
                 ({"volume": (20.0, 10.0, 6.0)}, {}, [-5.0, -3.0, -3.0], [15.0, 7.0, 3.0], "argument", [20, 10, 6]),
                 ({}, {"growth_box_um": [12.0, 6.0, 2.0]}, [0.0, 0.0, 0.0], [12.0, 6.0, 2.0], "growth_box_um",
                  [12, 6, 2]),
                 ({}, {"fit": "voxel_size", "volume": [24, 12, 4], "voxel_size": 0.5}, [-1.25, -1.25, -1.25],
                  [10.75, 4.75, 0.75], "voxel_size", [12, 6, 2])]
        for arguments, metadata, low, high, source, shape in cases:
            with self.subTest(source=source):
                r = describe(nodes, metadata=metadata, evd_spacing=1.0, **arguments)
                self.assertEqual(r["tissue_distance"]["domain"], {"min": low, "max": high, "source": source})
                self.assertEqual(r["tissue_distance"]["shape"], shape)
                self.assertEqual(r["volume_source"], source)

    def test_a_library_archive_samples_its_tissue_over_its_growth_box(self):
        # a library archive records grow_network's box as "growth_box", which
        # only the tissue distance reads: every other key stays as without it
        nodes = archive([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 4.0, 0.0)], diameter=1.0)
        plain = describe(nodes, evd_spacing=1.0)
        r = describe(nodes, metadata={"growth_box": [12.0, 6.0, 2.0]}, evd_spacing=1.0)
        self.assertEqual(r["tissue_distance"]["domain"],
                         {"min": [0.0, 0.0, 0.0], "max": [12.0, 6.0, 2.0], "source": "growth_box"})
        self.assertEqual(r["tissue_distance"]["shape"], [12, 6, 2])
        self.assertEqual(r["volume_source"], "bounding_box")
        self.assertEqual({k: v for k, v in r.items() if k != "tissue_distance"},
                         {k: v for k, v in plain.items() if k != "tissue_distance"})
        self.assertEqual(describe(nodes, metadata={"growth_box": [12.0, 6.0, 2.0]}), describe(nodes))
        # the argument and growth_box_um come first; None is no box
        cases = [({"volume": (20.0, 10.0, 6.0)}, {"growth_box": [12.0, 6.0, 2.0]}, "argument"),
                 ({}, {"growth_box": [12.0, 6.0, 2.0], "growth_box_um": [14.0, 8.0, 4.0]}, "growth_box_um"),
                 ({}, {"growth_box": [12.0, 6.0, 2.0], "fit": "voxel_size", "volume": [24, 12, 4], "voxel_size": 0.5},
                  "growth_box"),
                 ({}, {"growth_box": None}, "bounding_box")]
        for arguments, metadata, source in cases:
            with self.subTest(source=source, metadata=metadata):
                r = describe(nodes, metadata=metadata, evd_spacing=1.0, **arguments)
                self.assertEqual(r["tissue_distance"]["domain"]["source"], source)
        for box in ([12.0, 6.0], [12.0, float("nan"), 2.0]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                describe(nodes, metadata={"growth_box": box}, evd_spacing=1.0)

    def test_tissue_distance_is_opt_in_and_draws_nothing(self):
        self.assertIsNone(describe(Y_NODES)["tissue_distance"])
        for spacing in (0.0, -1.0, float("nan"), float("inf"), True, "1"):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                describe(Y_NODES, evd_spacing=spacing)
        random.seed(11)
        np.random.seed(11)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        nodes = THETA_NODES.copy()
        r = describe(nodes, evd_spacing=0.05, d_ref=0.2)
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual((after[0],) + after[2:], (numpy_state[0],) + numpy_state[2:])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        np.testing.assert_array_equal(nodes, THETA_NODES)
        self.assertEqual(describe(nodes, evd_spacing=0.05, d_ref=0.2), r)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)
        tissue = r["tissue_distance"]
        for key in ("mean", "median", "p90", "p99", "max", "spacing"):
            self.assertAlmostEqual(tissue[key + "_d"], tissue[key] / 0.2, delta=1e-12 * tissue[key], msg=key)
        # describe_archive passes the spacing through
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "theta.npz")
            np.savez(path, nodes=nodes)
            self.assertEqual(describe_archive(path, evd_spacing=0.05, d_ref=0.2), r)
        # no points, no grid; points but no capsule, every grid point outside
        self.assertIsNone(describe(np.empty((4, 0)), evd_spacing=1.0)["tissue_distance"])
        r = describe(archive([(0.0, 0.0, 0.0)], [(4.0, 0.0, 0.0)], diameter=1.0), evd_spacing=1.0)["tissue_distance"]
        self.assertEqual((r["capsules"], r["inside_fraction"], r["outside_points"], r["mean"]),
                         (0, 0.0, r["points"], None))

    def test_topology_and_tissue_keys_are_unit_invariant(self):
        grown = small_tree()
        theta = THETA_NODES.copy()
        theta[:3] += 1000.0
        theta[3] *= 50.0
        nodes = np.concatenate([grown["nodes"], np.full((4, 1), np.nan), theta], axis=1)
        # d_ref 5.3 keeps every segment's diameter clear of the class bound,
        # which the theta's 10 would sit on at d_ref 5; the merging distance
        # scales with the unit too
        base = describe(nodes, d_ref=5.3, evd_spacing=12.0)
        scaled = describe(nodes * 3.0, d_ref=15.9, evd_spacing=36.0, tol=3.0 * graph.DEFAULT_TOL)
        self.assertGreater(base["branch_angles_deg"]["count"], 1)
        self.assertGreater(base["loops"]["by_node"]["found"], 0)
        self.assertGreater(base["segment_diameter_variation"]["capillary"]["count"], 1)
        self.assertGreater(base["tissue_distance"]["outside_points"], 0)
        # lengths scale with the unit, cycles per length with its inverse, and
        # everything else (counts, fractions, angles, loop sizes, quantities
        # in d_ref) stays the same; development measurement over ten scale
        # factors from 0.32 to 7.4: at most 1.3e-13 relatively. The count of
        # capsules is left out: a few consecutive columns of a grown network
        # differ only in their last bit, and scaling can round that away
        lengths = {".loops.by_node.length_median", ".loops.by_node.length_mean"}
        lengths |= {".tissue_distance." + key
                    for key in ("spacing_requested", "spacing", "mean", "median", "p90", "p99", "max")}
        base_leaves = dict(numeric_leaves({key: base[key] for key in NEW_KEYS_3_6}))
        scaled_leaves = dict(numeric_leaves({key: scaled[key] for key in NEW_KEYS_3_6}))
        self.assertEqual(set(base_leaves), set(scaled_leaves))
        for key, value in base_leaves.items():
            if key == ".tissue_distance.capsules":
                continue
            if key in lengths or key.startswith(".tissue_distance.domain."):
                factor = 3.0
            elif key == ".loops.by_segment.cycles_per_length":
                factor = 1.0 / 3.0
            else:
                factor = 1.0
            self.assertAlmostEqual(scaled_leaves[key], factor * value, delta=1e-9 * max(1.0, abs(factor * value)),
                                   msg=key)

    def test_empty_and_unbranched_networks(self):
        r = describe(np.empty((4, 0)))
        self.assertEqual(r["junctions"]["count"], 0)
        self.assertEqual(r["branch_angles_deg"], {"count": 0, "skipped": 0, "min": None, "median": None, "max": None})
        self.assertEqual((r["loops"]["by_segment"]["sampled"], r["loops"]["by_segment"]["none_fraction"],
                          r["loops"]["by_segment"]["cycles_per_length"]), (0, None, None))
        self.assertEqual(r["segment_diameter_variation"]["all"], {"count": 0, "excluded": 0, "median": None,
                                                                  "p90": None})
        r = describe(archive(straight()))
        self.assertEqual((r["junctions"]["count"], r["junctions"]["mean_degree"]), (0, None))
        self.assertEqual((r["loops"]["by_segment"]["segments"], r["loops"]["by_segment"]["none_fraction"]), (1, 1.0))
        self.assertEqual(r["loops"]["by_node"]["sampled"], 0)
        self.assertEqual(r["segment_diameter_variation"]["all"]["median"], 0.0)


if __name__ == "__main__":
    unittest.main()
