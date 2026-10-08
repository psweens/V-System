"""
Tests of join.py: cropping networks to a box and joining separately grown
networks inside it.

Run from the repository root with
    python -m unittest tests.test_join
The benchmarks at the two performance targets are slow and run only with
VSYSTEM_SLOW_TESTS=1; they print per-stage timings and the outcome mix.
"""
import math
import os
import random
import resource
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import graph  # noqa: E402
import join  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from describe import clearance  # noqa: E402
from join import EVENT_KEYS, TIP_OUTCOMES, crop_network, join_networks  # noqa: E402
from spatial import GridIndex  # noqa: E402

PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True,
              "aneurysm_prob": 0.0, "stenosis_prob": 0.0}
OUTCOME = {name: code for code, name in enumerate(TIP_OUTCOMES)}


def grow(seed, niter, d0=8.0, d_min=1.0, family="tree", subdivisions=3, **options):
    """One network of a family in the test's own frame, with the preset's options."""
    random.seed(seed)
    np.random.seed(seed)
    preset = dict(main.FAMILIES[family])
    properties = dict(PROPERTIES)
    for key in ("aneurysm_prob", "stenosis_prob"):
        if key in preset:
            properties[key] = preset.pop(key)
    preset.pop("d0", None)
    preset.update(options)
    grown = main.grow_network(niter, d0, properties, (48, 48, 24), d_min=d_min, seed=seed,
                              subdivisions=subdivisions, **preset)
    return {"nodes": grown["nodes"], "node_kind": grown["node_kind"], "tree": grown["tree"]}


def rotation(rng):
    """A uniformly random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def place(network, matrix, offset):
    """The network rotated about its centroid by `matrix` and moved to `offset`."""
    nodes = network["nodes"].copy()
    finite = np.isfinite(nodes[0])
    centre = np.nanmean(nodes[:3], axis=1)
    nodes[:3, finite] = matrix @ (nodes[:3, finite] - centre[:, None]) + np.asarray(offset, dtype=float)[:, None]
    out = dict(network)
    out["nodes"] = nodes
    return out


def build_forest(library, count, lo, hi, rng, clearance_gap=2.0, tries=40):
    """
    Places `count` networks from `library` at random rotations and offsets in
    the box, rejecting a placement whose points come within `clearance_gap`
    of the surface of any point already placed.
    """
    index = GridIndex(cell_size=20.0)
    placed = []
    for i in range(count):
        network = library[i % len(library)]
        for _ in range(tries):
            candidate = place(network, rotation(rng), rng.uniform(lo, hi))
            finite = np.isfinite(candidate["nodes"][0])
            points = candidate["nodes"][:3, finite].T
            radii = candidate["nodes"][3, finite] / 2.0
            if index.n:
                qi, _ = index.query(points, radii, clearance_gap)
                if len(qi):
                    continue
            index.add(points, radii, np.full(len(points), i))
            placed.append(candidate)
            break
    return placed


def facing_pair(seed=1, niter=4, gap=15.0):
    """Two copies of one tree, the second turned round so that their tips face each other."""
    tree = grow(seed, niter)
    nodes = tree["nodes"]
    height = float(np.nanmax(nodes[1]) - np.nanmin(nodes[1]))
    first = place(tree, np.eye(3), np.zeros(3))
    second = place(tree, np.diag([-1.0, -1.0, 1.0]), np.array([0.0, height + gap, 0.0]))
    return [first, second]


def bounding_box(networks, pad):
    lo = np.min([np.nanmin(n["nodes"][:3], axis=1) for n in networks], axis=0) - pad
    hi = np.max([np.nanmax(n["nodes"][:3], axis=1) for n in networks], axis=0) + pad
    return lo, hi


def concatenate(networks):
    pieces = []
    for network in networks:
        if pieces:
            pieces.append(np.full((4, 1), np.nan))
        pieces.append(network["nodes"])
    return np.concatenate(pieces, axis=1)


def outcome_counts(result):
    return {name: int(np.count_nonzero(result["tips"]["outcome"] == code)) for code, name in enumerate(TIP_OUTCOMES)}


class RegistryTests(unittest.TestCase):
    def test_rng_tags_do_not_collide_with_main(self):
        self.assertFalse(set(join.RNG_STREAMS.values()) & set(main.RNG_STREAMS.values()))
        self.assertFalse(set(join.RNG_STREAMS) & set(main.RNG_STREAMS))

    def test_event_keys_are_listed_with_zero_counts(self):
        result = join_networks([], np.random.default_rng(0), fraction=1.0, collision_margin=1.0,
                               box=(np.zeros(3), np.ones(3)), boundary_margin=0.0)
        self.assertEqual(set(result["events"]), set(EVENT_KEYS))
        self.assertTrue(all(v == 0 for v in result["events"].values()))
        self.assertEqual(result["bridges"], [])
        self.assertEqual(result["tips"].shape, (0,))

    def test_main_does_not_import_frames_join_or_library(self):
        for name in ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance", "connections"):
            with open(os.path.join(ROOT, name + ".py")) as handle:
                source = handle.read()
            for other in ("frames", "join", "library"):
                self.assertNotIn(f"import {other}", source, name)
                self.assertNotIn(f"from {other} ", source, name)

    def test_the_stream_tags_of_the_generator_are_what_they_were(self):
        self.assertEqual(main.RNG_STREAMS, {"walk": 1, "anastomosis": 2, "rungs": 5})


class CropTests(unittest.TestCase):
    def test_columns_kept_by_segment_reach_in_archive_order_with_one_separator_per_dropped_run(self):
        # a polyline that leaves the box and comes back is two polylines, in order
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        nodes = np.stack([x, np.zeros(9), np.zeros(9), np.ones(9)])    # radius 0.5
        nodes[:, 4] = [4.0, 20.0, 0.0, 1.0]                      # one far excursion
        network = {"nodes": nodes, "node_kind": np.full(9, graph.INTERIOR, dtype=np.int8),
                   "tree": np.zeros(9, dtype=np.int8), "roots": np.array([0, 8])}
        cropped = crop_network(network, np.array([-1.0, -1.0, -1.0]), np.array([9.0, 1.0, 1.0]), margin=0.5)
        # column 4 lies 20 away but the segments 3-4 and 4-5 reach the box, so every column stays
        np.testing.assert_array_equal(cropped["columns"], np.arange(9))
        # segment 2-3 padded by the radius reaches x = 1.5, segment 3-4 only 2.5
        tight = crop_network(network, np.array([-1.0, -1.0, -1.0]), np.array([2.4, 1.0, 1.0]), margin=0.0)
        np.testing.assert_array_equal(tight["columns"], [0, 1, 2, 3, -1])
        self.assertTrue(np.isnan(tight["nodes"][:, -1]).all())
        np.testing.assert_array_equal(tight["node_kind"], [0, 0, 0, 0, -1])
        np.testing.assert_array_equal(tight["tree"], [0, 0, 0, 0, -1])
        np.testing.assert_array_equal(tight["roots"], [0])                 # root 8 was dropped
        # segment 4-5 reaches x = 5.5 through its far column, segment 7-8 starts at 6.5
        inner = crop_network(network, np.array([5.5, -1.0, -1.0]), np.array([6.4, 1.0, 1.0]), margin=0.0)
        np.testing.assert_array_equal(inner["columns"], [-1, 4, 5, 6, 7, -1])
        np.testing.assert_array_equal(inner["roots"], [])

    def test_an_empty_crop_is_accepted_by_the_join(self):
        network = grow(1, 3)
        empty = crop_network(network, np.full(3, 1e6), np.full(3, 1e6 + 1.0), margin=0.0)
        self.assertEqual(empty["nodes"].shape, (4, 0))
        self.assertEqual(empty["columns"].shape, (0,))
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        result = join_networks([empty] + pair + [empty], np.random.default_rng(0), fraction=1.0,
                               collision_margin=0.5, box=(lo, hi), boundary_margin=0.0)
        self.assertEqual(result["events"]["join_networks"], 4)
        self.assertEqual(set(result["tips"]["network"].tolist()), {1, 2})

    def test_a_cropped_network_keeps_its_polylines_whole(self):
        network = grow(2, 5)
        lo = np.nanmin(network["nodes"][:3], axis=1)
        hi = np.nanmax(network["nodes"][:3], axis=1)
        mid = (lo + hi) / 2.0
        cropped = crop_network(network, lo, mid, margin=0.0)
        kept = cropped["columns"][cropped["columns"] >= 0]
        self.assertGreater(kept.size, 0)
        self.assertLess(kept.size, network["nodes"].shape[1])
        self.assertTrue(np.all(np.diff(kept) > 0))                        # archive order
        for run in graph.polylines(cropped["nodes"]):
            original = cropped["columns"][run]
            self.assertTrue(np.all(np.diff(original) == 1))             # a run of consecutive columns
        np.testing.assert_array_equal(cropped["nodes"][:, cropped["columns"] >= 0],
                                      network["nodes"][:, kept])


class JoinTests(unittest.TestCase):
    def test_two_facing_networks_join_into_one_component_without_intra_network_bridges(self):
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        result = join_networks(pair, np.random.default_rng(1), fraction=1.0, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0, merged=True)
        self.assertGreater(len(result["bridges"]), 0)
        self.assertEqual(result["summary"]["components"]["before"], 2)
        self.assertEqual(result["summary"]["components"]["after"], 1)
        for bridge in result["bridges"]:
            self.assertNotEqual(bridge["tip"][0], bridge["partner"][0])
            geometry = bridge["geometry"]
            self.assertEqual(geometry.shape[0], 4)
            self.assertTrue(np.all(geometry[3] == bridge["diameter"]))
            tip_net, tip_col = bridge["tip"]
            partner_net, partner_col = bridge["partner"]
            np.testing.assert_array_equal(geometry[:3, 0], pair[tip_net]["nodes"][:3, tip_col])
            np.testing.assert_array_equal(geometry[:3, -1], pair[partner_net]["nodes"][:3, partner_col])
            self.assertAlmostEqual(bridge["volume"], math.pi * (bridge["diameter"] / 2.0) ** 2 * bridge["arc"])
            self.assertGreater(bridge["arc"], bridge["chord"])
        merged = result["merged"]
        rebuilt = graph.build(merged["nodes"])
        np.testing.assert_array_equal(rebuilt["edges"], merged["edges"])
        np.testing.assert_array_equal(rebuilt["node_kind"], merged["node_kind"])
        self.assertEqual(graph.betti(merged["edges"], rebuilt["canonical"])["components"], 1)
        self.assertEqual(set(merged["network"].tolist()), {-1, 0, 1, join.BRIDGE_NETWORK})
        self.assertEqual(int(np.count_nonzero(merged["tree"] == BRIDGE)),
                         sum(b["geometry"].shape[1] for b in result["bridges"]))
        # bridge_nodes is the bridges with separators, in order
        runs = graph.polylines(result["bridge_nodes"])
        self.assertEqual(len(runs), len(result["bridges"]))
        for run, bridge in zip(runs, result["bridges"]):
            np.testing.assert_array_equal(result["bridge_nodes"][:, run], bridge["geometry"])

    def test_bridges_keep_the_margin_from_every_vessel_and_from_earlier_bridges(self):
        rng = np.random.default_rng(3)
        library = [grow(seed, 5) for seed in (1, 2, 3)]
        lo, hi = np.zeros(3), np.full(3, 220.0)
        forest = build_forest(library, 24, lo, hi, rng, clearance_gap=2.0)
        self.assertGreaterEqual(len(forest), 12)
        margin = 1.0
        before = clearance(concatenate(forest), margin=margin)
        result = join_networks(forest, np.random.default_rng(4), fraction=1.0, collision_margin=margin,
                               box=(lo - 50.0, hi + 50.0), boundary_margin=0.0, merged=True)
        self.assertGreater(len(result["bridges"]), 5)
        after = clearance(result["merged"]["nodes"], margin=margin)
        self.assertEqual(after["violations"], before["violations"])
        # and the bridges against each other, excused nowhere
        bridges = result["bridges"]
        for i, b in enumerate(bridges):
            for c in bridges[:i]:
                p, q = b["geometry"][:3, 1:-1].T, c["geometry"][:3, 1:-1].T
                if p.shape[0] and q.shape[0]:
                    gap = np.min(np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2))
                    self.assertGreaterEqual(gap, b["diameter"] / 2.0 + c["diameter"] / 2.0 + margin - 1e-9)

    def test_cut_ends_are_never_used_and_no_bridge_leaves_the_box(self):
        rng = np.random.default_rng(5)
        library = [grow(seed, 5) for seed in (1, 2)]
        outer_lo, outer_hi = np.zeros(3), np.full(3, 200.0)
        forest = build_forest(library, 20, outer_lo, outer_hi, rng, clearance_gap=2.0)
        lo, hi = np.full(3, 40.0), np.full(3, 160.0)
        margin = 1.0
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        cropped = [crop_network(n, lo, hi, r_max + margin) for n in forest]
        boundary = 3.0
        result = join_networks(cropped, np.random.default_rng(6), fraction=1.0, collision_margin=margin,
                               box=(lo, hi), boundary_margin=boundary)
        self.assertGreater(len(result["bridges"]), 0)
        counts = outcome_counts(result)
        self.assertGreater(counts["cut_end"], 0)
        cut = set()
        for row in result["tips"]:
            network, column = int(row["network"]), int(row["column"])
            point = cropped[network]["nodes"][:3, column]
            kind = cropped[network]["node_kind"][column]
            outside = np.any(point < lo + boundary) or np.any(point > hi - boundary)
            if row["outcome"] == OUTCOME["cut_end"]:
                cut.add((network, column))
                self.assertTrue(outside or kind != graph.TIP)
            elif row["outcome"] != OUTCOME["root"]:
                self.assertFalse(outside)
                self.assertEqual(kind, graph.TIP)
        for bridge in result["bridges"]:
            self.assertNotIn(tuple(bridge["tip"]), cut)
            self.assertNotIn(tuple(bridge["partner"]), cut)
            self.assertTrue(np.all(bridge["geometry"][:3] >= lo[:, None]))
            self.assertTrue(np.all(bridge["geometry"][:3] <= hi[:, None]))

    def test_tip_outcomes_partition_the_degree_one_vertices_and_agree_with_events(self):
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        result = join_networks(pair, np.random.default_rng(2), fraction=0.6, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0)
        tips = result["tips"]
        degree_one = []
        for k, network in enumerate(pair):
            built = graph.build(network["nodes"])
            vertices = np.flatnonzero(built["canonical"] == np.arange(network["nodes"].shape[1]))
            degree_one.extend((k, int(v)) for v in vertices if built["degree"][v] == 1)
        self.assertEqual(sorted(degree_one), [(int(r["network"]), int(r["column"])) for r in tips])
        counts = outcome_counts(result)
        events = result["events"]
        self.assertEqual(sum(counts.values()), events["join_tips"])
        self.assertEqual(counts["root"], events["join_roots"])
        self.assertEqual(counts["cut_end"], events["join_cut_ends"])
        self.assertEqual(counts["stub"], events["join_stubs"])
        self.assertEqual(counts["not_selected"], events["join_not_selected"])
        self.assertEqual(counts["bridged_source"], events["join_bridges"])
        self.assertEqual(counts["bridged_source"], len(result["bridges"]))
        self.assertEqual(counts["bridged_partner"], events["join_bridged_partners"])
        self.assertEqual(counts["bridged_partner"], events["join_partner_tips"])
        self.assertEqual(counts["no_partner"], events["join_no_partner"])
        self.assertEqual(counts["collision_failed"], events["join_collision_failed"])
        self.assertEqual(counts["over_budget"], events["join_over_budget"])
        self.assertEqual(events["join_eligible"], events["join_tips"] - counts["root"] - counts["cut_end"] - counts["stub"])
        self.assertGreater(counts["not_selected"], 0)
        self.assertGreater(counts["bridged_source"], 0)
        self.assertEqual(result["summary"]["free_tips"]["before"], events["join_tips"] - counts["root"] - counts["cut_end"])
        self.assertEqual(result["summary"]["free_tips"]["after"],
                         result["summary"]["free_tips"]["before"] - counts["bridged_source"] - counts["bridged_partner"])

    def test_identical_inputs_and_generator_state_give_identical_output(self):
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        runs = [join_networks(pair, np.random.default_rng(7), fraction=0.8, collision_margin=0.5,
                              box=(lo, hi), boundary_margin=0.0) for _ in range(2)]
        self.assertEqual(len(runs[0]["bridges"]), len(runs[1]["bridges"]))
        for a, b in zip(runs[0]["bridges"], runs[1]["bridges"]):
            np.testing.assert_array_equal(a["geometry"], b["geometry"])
            self.assertEqual((a["tip"], a["partner"], a["redraws"]), (b["tip"], b["partner"], b["redraws"]))
        np.testing.assert_array_equal(runs[0]["tips"], runs[1]["tips"])
        self.assertEqual(runs[0]["events"], runs[1]["events"])
        other = join_networks(pair, np.random.default_rng(8), fraction=0.8, collision_margin=0.5,
                              box=(lo, hi), boundary_margin=0.0)
        self.assertNotEqual([b["tip"] for b in other["bridges"]], [b["tip"] for b in runs[0]["bridges"]])

    def test_the_result_is_invariant_under_a_change_of_unit(self):
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        settings = dict(fraction=0.9, collision_margin=0.5, boundary_margin=1.0, tol=graph.DEFAULT_TOL,
                        max_bridge_volume=None)
        reference = join_networks(pair, np.random.default_rng(9), box=(lo, hi), **settings)
        volume = sum(b["volume"] for b in reference["bridges"][:3])
        budgeted = join_networks(pair, np.random.default_rng(9), box=(lo, hi),
                                 **dict(settings, max_bridge_volume=volume * (1 + 1e-12)))
        self.assertGreater(len(reference["bridges"]), 3)
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                scaled = []
                for network in pair:
                    nodes = network["nodes"] * scale
                    scaled.append(dict(network, nodes=nodes))
                for original, budget in ((reference, None), (budgeted, volume * (1 + 1e-12) * scale ** 3)):
                    result = join_networks(scaled, np.random.default_rng(9), box=(lo * scale, hi * scale),
                                           fraction=0.9, collision_margin=0.5 * scale,
                                           boundary_margin=1.0 * scale, tol=graph.DEFAULT_TOL * scale,
                                           max_bridge_volume=budget)
                    np.testing.assert_array_equal(result["tips"], original["tips"])
                    self.assertEqual(result["events"], original["events"])
                    self.assertEqual(len(result["bridges"]), len(original["bridges"]))
                    for a, b in zip(result["bridges"], original["bridges"]):
                        self.assertEqual((a["tip"], a["partner"], a["partner_kind"]), (b["tip"], b["partner"], b["partner_kind"]))
                        np.testing.assert_allclose(a["geometry"], b["geometry"] * scale, rtol=1e-12, atol=0.0)
                        np.testing.assert_allclose(a["arc"], b["arc"] * scale, rtol=1e-12)
                        np.testing.assert_allclose(a["volume"], b["volume"] * scale ** 3, rtol=1e-12)

    def test_more_than_127_networks_are_told_apart(self):
        tree = grow(1, 3)
        count = 130
        networks = []
        for i in range(count):
            # a row of copies, each turned round against the next, 25 units apart
            matrix = np.eye(3) if i % 2 == 0 else np.diag([-1.0, -1.0, 1.0])
            networks.append(place(tree, matrix, np.array([0.0, 60.0 * i, 0.0])))
        lo, hi = bounding_box(networks, 20.0)
        result = join_networks(networks, np.random.default_rng(10), fraction=1.0, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0, merged=True)
        self.assertEqual(result["events"]["join_networks"], count)
        self.assertEqual(result["tips"]["network"].dtype, np.int32)
        self.assertEqual(int(result["tips"]["network"].max()), count - 1)
        self.assertEqual(result["events"]["join_roots"], count)
        networks_bridged = {b["tip"][0] for b in result["bridges"]} | {b["partner"][0] for b in result["bridges"]}
        self.assertGreater(max(networks_bridged), 127)
        for bridge in result["bridges"]:
            self.assertNotEqual(bridge["tip"][0], bridge["partner"][0])
        merged = result["merged"]
        self.assertEqual(int(merged["network"].max()), count - 1)
        self.assertEqual(result["summary"]["components"]["before"], count)
        self.assertLess(result["summary"]["components"]["after"], count)

    def test_a_mesh_has_two_roots_that_are_never_bridged(self):
        mesh = grow(3, 6, family="mesh")
        self.assertEqual(set(np.unique(mesh["tree"]).tolist()) - {-1, BRIDGE}, {0, 1})
        nodes = mesh["nodes"]
        height = float(np.nanmax(nodes[1]) - np.nanmin(nodes[1]))
        networks = [place(mesh, np.eye(3), np.zeros(3)),
                    place(mesh, np.diag([-1.0, -1.0, 1.0]), np.array([0.0, height + 10.0, 0.0]))]
        lo, hi = bounding_box(networks, 20.0)
        result = join_networks(networks, np.random.default_rng(11), fraction=1.0, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0)
        tips = result["tips"]
        roots = set()
        for k, network in enumerate(networks):
            tree = network["tree"]
            for label in (0, 1):
                column = int(np.flatnonzero(tree == label)[0])
                roots.add((k, column))
                row = tips[(tips["network"] == k) & (tips["column"] == column)]
                self.assertEqual(row.shape, (1,), (k, column))
                self.assertEqual(int(row["outcome"][0]), OUTCOME["root"])
        self.assertEqual(result["events"]["join_roots"], 4)
        for bridge in result["bridges"]:
            self.assertNotIn(tuple(bridge["tip"]), roots)
            self.assertNotIn(tuple(bridge["partner"]), roots)
        # a mesh's own anastomosis bridges are ordinary vessels: their points are obstacles,
        # never partners, since every bridge end is an input column of a tree label
        for bridge in result["bridges"]:
            for network, column in (bridge["tip"], bridge["partner"]):
                self.assertNotEqual(networks[network]["tree"][column], BRIDGE)
        self.assertGreater(len(result["bridges"]), 0)

    def test_a_forest_loses_components_and_free_tips(self):
        rng = np.random.default_rng(12)
        library = [grow(seed, 4) for seed in (1, 2, 3, 4)]
        lo, hi = np.zeros(3), np.full(3, 480.0)
        forest = build_forest(library, 300, lo, hi, rng, clearance_gap=2.0)
        self.assertGreaterEqual(len(forest), 200)
        margin = 1.0
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        cropped = [crop_network(n, lo, hi, r_max + margin) for n in forest]
        result = join_networks(cropped, np.random.default_rng(13), fraction=0.9, collision_margin=margin,
                               box=(lo, hi), boundary_margin=2.0)
        summary = result["summary"]
        self.assertLess(summary["components"]["after"], summary["components"]["before"])
        self.assertLess(summary["free_tips_per_length"]["after"], summary["free_tips_per_length"]["before"])
        self.assertGreater(summary["length"]["after"], summary["length"]["before"])
        self.assertGreater(result["events"]["join_bridges"], len(forest) // 2)
        print(f"\nforest of {len(cropped)} networks, {result['events']['join_points']} points: "
              f"components {summary['components']['before']} -> {summary['components']['after']}, "
              f"free in-box tips per unit length {summary['free_tips_per_length']['before']:.4f} -> "
              f"{summary['free_tips_per_length']['after']:.4f}, "
              f"bridges {result['events']['join_bridges']}", flush=True)

    def test_any_prefix_of_the_bridges_is_a_valid_result(self):
        pair = facing_pair()
        lo, hi = bounding_box(pair, 20.0)
        full = join_networks(pair, np.random.default_rng(14), fraction=1.0, collision_margin=0.5,
                             box=(lo, hi), boundary_margin=0.0)
        self.assertGreaterEqual(len(full["bridges"]), 3)
        for k in range(len(full["bridges"]) + 1):
            with self.subTest(k=k):
                budget = sum(b["volume"] for b in full["bridges"][:k])
                result = join_networks(pair, np.random.default_rng(14), fraction=1.0, collision_margin=0.5,
                                       box=(lo, hi), boundary_margin=0.0, max_bridge_volume=budget * (1 + 1e-12))
                self.assertEqual(len(result["bridges"]), k)
                tips = result["tips"]
                for a, b in zip(result["bridges"], full["bridges"][:k]):
                    self.assertEqual((a["tip"], a["partner"]), (b["tip"], b["partner"]))
                    np.testing.assert_array_equal(a["geometry"], b["geometry"])
                    # both ends keep their outcome whatever happened to the budget afterwards
                    source = tips[(tips["network"] == a["tip"][0]) & (tips["column"] == a["tip"][1])]
                    self.assertEqual(int(source["outcome"][0]), OUTCOME["bridged_source"])
                    if a["partner_kind"] == "tip":
                        partner = tips[(tips["network"] == a["partner"][0]) & (tips["column"] == a["partner"][1])]
                        self.assertEqual(int(partner["outcome"][0]), OUTCOME["bridged_partner"])
                counts = outcome_counts(result)
                if k < len(full["bridges"]):
                    self.assertGreater(counts["over_budget"], 0)
                self.assertEqual(result["events"]["join_over_budget"], counts["over_budget"])
                self.assertEqual(counts["bridged_source"], k)

    def test_policy_any_allows_the_same_network_beyond_the_kin_separation(self):
        tree = grow(4, 6)
        lo, hi = bounding_box([tree], 20.0)
        cross = join_networks([tree], np.random.default_rng(15), fraction=1.0, collision_margin=0.5,
                              box=(lo, hi), boundary_margin=0.0, policy="cross")
        self.assertEqual(cross["bridges"], [])
        self.assertEqual(outcome_counts(cross)["no_partner"], cross["events"]["join_eligible"])
        same = join_networks([tree], np.random.default_rng(15), fraction=1.0, collision_margin=0.5,
                             box=(lo, hi), boundary_margin=0.0, policy="any")
        self.assertGreater(len(same["bridges"]), 0)
        self.assertGreater(same["events"]["join_kin_skipped"], 0)
        # the kin rule of anastomosis: never a sister tip at the default separation
        built = graph.build(tree["nodes"])
        canonical = built["canonical"]
        junction_of_tip = {}
        for path in graph.segments(tree["nodes"], built["edges"], canonical):
            junction_of_tip[int(path[0])] = int(path[-1])
            junction_of_tip[int(path[-1])] = int(path[0])
        for bridge in same["bridges"]:
            if bridge["partner_kind"] == "tip":
                a = junction_of_tip.get(int(canonical[bridge["tip"][1]]))
                b = junction_of_tip.get(int(canonical[bridge["partner"][1]]))
                self.assertNotEqual(a, b)


@unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmarks")
class BenchmarkTests(unittest.TestCase):
    """
    The two performance targets, on forests of grown tree-family networks
    placed at random rotations and offsets with a point-clearance rejection
    and cropped to the box. Per-stage times, peak memory and the outcome mix
    are printed.
    """

    def _run(self, label, library, count, side, seed, time_limit, memory_limit):
        rng = np.random.default_rng(seed)
        lo, hi = np.zeros(3), np.full(3, side)
        started = time.perf_counter()
        forest = build_forest(library, count, lo, hi, rng, clearance_gap=2.0, tries=60)
        placed = time.perf_counter() - started
        margin = 1.0
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        started = time.perf_counter()
        cropped = [crop_network(n, lo, hi, r_max + margin) for n in forest]
        cropping = time.perf_counter() - started
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        started = time.perf_counter()
        result = join_networks(cropped, np.random.default_rng(seed + 1), fraction=0.9, collision_margin=margin,
                               box=(lo, hi), boundary_margin=2.0)
        joining = time.perf_counter() - started
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        events = result["events"]
        print(f"\n{label}: {len(cropped)} networks, {events['join_points']} points, "
              f"{events['join_eligible']} eligible tips, {events['join_selected']} selected; "
              f"placing {placed:.1f} s, cropping {cropping:.1f} s, joining {joining:.1f} s "
              f"({1000.0 * joining / max(events['join_selected'], 1):.2f} ms per selected tip); "
              f"peak RSS {rss_after:.0f} MB (before joining {rss_before:.0f} MB)")
        print("  outcomes: " + ", ".join(f"{k}={v}" for k, v in outcome_counts(result).items() if v))
        print("  events: " + ", ".join(f"{k}={v}" for k, v in events.items()
                                       if v and k in ("join_bridges", "join_redraws", "join_box_redraws",
                                                      "join_source_consumed", "join_partner_interior",
                                                      "join_components_joined")))
        print(f"  summary: {result['summary']}")
        self.assertLess(joining, time_limit)
        self.assertLess(rss_after, memory_limit)

    def test_typical_forest_joins_in_under_30_seconds_and_2_GB(self):
        library = [grow(seed, 5, subdivisions=2) for seed in range(1, 9)]
        self._run("typical", library, 330, 700.0, 20, 30.0, 2048.0)

    def test_large_forest_joins_in_under_3_minutes_and_4_GB(self):
        library = [grow(seed, 6) for seed in range(1, 9)]
        self._run("large", library, 650, 1300.0, 30, 180.0, 4096.0)


if __name__ == "__main__":
    unittest.main()
