"""
Tests of the fibre bed, bed.py's kind "fibre": capillaries in lanes along
an axis, placed by a two-dimensional hard core, swayed across the axis at
their knots and cross-connected between neighbours. In order: the
kind-specific settings and every refusal before a generator exists
(SettingsTests), the planned counts (PlannedCountTests), the frame record
and describe's sections about it (FrameTests), the registries
(RegistryTests), the lanes against a one-at-a-time loop and the boundary
knots (LaneTests), pruning against an independent rewrite of F7
(PruneTests), every sub-stream recreated (DrawTests), determinism and scale
(FibreScaleTests), the density law on the chord graph and the walked bed
(DensityLawTests), the chord law and the sway trends (OrientationTests),
the link sites, partners and describe's cross-connections (LinkTests), the
greedy pass's rules and F9 (RuleTests), the written columns and clearance
(ColumnTests), the counters' laws (LawTests), the keys build returns
(ReturnTests), and the settings' dataclass behaviour (SettingsObjectTests).
The development measurements each tolerance was set from are recorded
beside the test that uses it.

Run from the repository root with
    python -m unittest tests.test_fibre
The fast tests use seed 1; seeds 1-10 run only with VSYSTEM_SLOW_TESTS=1.

The builds the tests read are made once each and shared (`alone`, `fed`): a
bed alone (feeders None) in a cube of side 48 d_min along x, 44 lanes at the
defaults, and a bed between the feeder pair of the foam member of ratio 4
(its cube of side 60, its feeders and its unguided frame, captured where
main calls bed.build), along x.
"""
import ast
import collections
import contextlib
import copy
import dataclasses
import functools
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bed  # noqa: E402
import describe  # noqa: E402
import frames  # noqa: E402
import graph  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
import tortuosity  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from guidance import Guidance, unguided_frame  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")

D_MIN = 1.0
MARGIN = 1.0
SIDE = 48.0
SEEDS = tuple(range(1, 11))
X_AXIS = [1.0, 0.0, 0.0]

# The defaults as approved, every key written out, and the same along x.
FIBRE_DEFAULTS = {"kind": "fibre", "spacing": 5.0, "girth": 8, "min_angle": 60.0, "reach": 2.0, "diameter": 1.0,
                  "persistence": 16.0, "feeder_stop": 2.0, "tip_edges": 2, "density": 0.5, "jitter": 0.25,
                  "step": 0.5, "axis": None, "knot": 4.0, "sway": 1.0, "link_spacing": 20.0, "link_reach": 1.75,
                  "run": None, "layout": "pair", "unit": 60.0, "sheet_length": 2.0}
FIBRE = dict(FIBRE_DEFAULTS, axis=X_AXIS)


@contextlib.contextmanager
def recorded_generators():
    """Every numpy Generator created inside the block, with the seed it was given."""
    created = []
    original = np.random.default_rng

    def recording(*args, **kwargs):
        generator = original(*args, **kwargs)
        created.append((args[0] if args else None, generator))
        return generator

    with mock.patch.object(np.random, "default_rng", recording):
        yield created


def seeds_of(created):
    return [list(seed) if isinstance(seed, (list, tuple)) else seed for seed, _ in created]


def global_states():
    return random.getstate(), np.random.get_state()


def assert_same_states(test, a, b):
    test.assertEqual(a[0], b[0])
    for x, y in zip(a[1], b[1]):
        np.testing.assert_array_equal(x, y)


# --------------------------------------------------------------------------
# The builds, made once and shared


_BUILDS = {}


def _cached(key, make):
    if key not in _BUILDS:
        _BUILDS[key] = make()
    return _BUILDS[key]


def _key(changes):
    return tuple(sorted((name, str(value)) for name, value in changes.items()))


def build_case(label, domain, settings, *, seed, feeders=None, frame=None, d_min=D_MIN, margin=MARGIN, attempts=10):
    """bed.build's result with its counters and timings, as the checks read it."""
    events, timings = {}, {}
    settings = bed.parse_settings(settings)
    result = bed.build(domain, settings, d_min=d_min, seed=seed, collision_margin=margin, feeders=feeders,
                       frame=frame, attempts=attempts, events=events, timings=timings)
    low, high = (np.asarray(corner, dtype=float) for corner in domain)
    return {"label": label, "result": result, "events": events, "timings": timings, "settings": settings,
            "d_min": d_min, "margin": margin, "attempts": attempts, "feeders": feeders, "frame": frame,
            "domain": (low, high), "extents": high - low, "seed": seed}


def alone(seed=1, side=SIDE, **changes):
    """A bed alone (feeders None) in a cube of the given side, at the defaults along x changed as given."""
    settings = dict(FIBRE, **changes)
    label = f"alone, side {side:g}, seed {seed}" + "".join(f", {k} {v}" for k, v in sorted(changes.items()))
    return _cached(("alone", seed, side, _key(changes)),
                   lambda: build_case(label, (np.zeros(3), np.full(3, float(side))), settings, seed=seed))


def foam_call(seed=1, ratio=4.0):
    """The domain, feeders and frame main hands bed.build for the foam member of the given ratio and seed."""
    def make():
        calls = []

        def capture(*args, **kwargs):
            calls.append((args, kwargs))
            return bed.build(*args, **kwargs)

        with mock.patch.object(main, "build_bed", side_effect=capture):
            library.grow_member("foam", ratio, seed)
        (args, kwargs), = calls
        return {"domain": args[0], "feeders": kwargs["feeders"], "frame": kwargs["frame"]}
    return _cached(("foam call", seed, ratio), make)


def fed(seed=1, ratio=4.0, **changes):
    """A bed between the foam member's feeders, in its domain and with its frame, at the defaults along x."""
    call = foam_call(seed, ratio)
    settings = dict(FIBRE, **changes)
    label = f"fed, foam R {ratio:g}, seed {seed}" + "".join(f", {k} {v}" for k, v in sorted(changes.items()))
    return _cached(("fed", seed, ratio, _key(changes)),
                   lambda: build_case(label, call["domain"], settings, seed=seed, feeders=call["feeders"],
                                      frame=call["frame"]))


def vertex_points(case):
    """The bed vertices' coordinates: the knots, then the eligible tips' feeder columns."""
    result = case["result"]
    if case["feeders"] is None:
        return result["knots"]
    return np.concatenate([result["knots"], np.asarray(case["feeders"][0])[:3, result["tips"]].T])


def resolved_axis(case):
    return np.array(case["result"]["axis"])


def interior_count(case):
    return int(np.count_nonzero(~case["result"]["knot_boundary"]))


def lane_edge_mask(case, edges):
    """Whether each edge (vertex ids) joins two knots of one lane."""
    result = case["result"]
    n_knots = len(result["knots"])
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    both = edges[:, 1] < n_knots
    lane = np.full(edges.shape, -1, dtype=np.int64)
    lane[both] = result["knot_lane"][edges[both]]
    return both & (lane[:, 0] == lane[:, 1])


def link_edge_mask(case, edges):
    """Whether each edge joins two knots of different lanes."""
    result = case["result"]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    both = edges[:, 1] < len(result["knots"])
    return both & ~lane_edge_mask(case, edges)


# --------------------------------------------------------------------------
# The checks, each re-evaluated from the record


def f9_holds_by_condition(settings, d_min, margin):
    """F9's condition, (h - T) <= l tan(90 - max(min_angle, KIN_ANGLE_DEG) / 2)."""
    h = settings["spacing"] * d_min
    T = settings["diameter"] * d_min + margin
    knot = settings["knot"] * d_min
    return h - T <= knot * math.tan(math.radians(90.0 - max(settings["min_angle"], bed.KIN_ANGLE_DEG) / 2.0))


def check_laws(test, case):
    """F1-F9, E1-E5 and E7-E9 of bed.py's docstring, on the counters and the record."""
    ev, result, settings = case["events"], case["result"], case["settings"]
    n_lanes = len(result["seeds"])
    n_knots = len(result["knots"])
    m = interior_count(case)
    low, high = case["domain"]
    # E1 and F1
    test.assertEqual(ev["bed_proposals"], ev["bed_proposals_on_feeders"] + ev["bed_proposals_overlapping"]
                     + ev["bed_seeds"] + ev["bed_proposals_outside"])
    test.assertEqual(ev["bed_proposals_on_feeders"], 0)
    # E2 and F2
    test.assertEqual(ev["bed_seeds"], n_lanes)
    test.assertEqual(ev["bed_seeds"], bed.planned_seeds(high - low, settings, case["d_min"], axis=resolved_axis(case)))
    # F3
    test.assertEqual(ev["bed_knots"], n_knots)
    test.assertEqual(ev["bed_knots"], 2 * ev["bed_seeds"] + m)
    test.assertEqual(ev["bed_knots_boundary"], 2 * ev["bed_seeds"])
    test.assertEqual(ev["bed_knots_boundary"], int(np.count_nonzero(result["knot_boundary"])))
    # F4
    test.assertEqual(ev["bed_knots"] - ev["bed_seeds"],
                     ev["bed_lane_candidates"] + ev["bed_lane_edges_on_feeders"] + ev["bed_breaks"])
    if settings["run"] is None:
        test.assertEqual(ev["bed_breaks"], 0)
    # F5
    test.assertEqual(ev["bed_link_sites"],
                     ev["bed_link_sites_no_partner"] + ev["bed_link_sites_angle"] + ev["bed_link_sites_on_feeders"]
                     + ev["bed_candidates_link"])
    if settings["link_spacing"] is None:
        test.assertEqual(ev["bed_link_sites"], 0)
    # F6, E3 and E4
    test.assertEqual(ev["bed_candidates"],
                     ev["bed_lane_candidates"] + ev["bed_candidates_tip"] + ev["bed_candidates_link"])
    test.assertEqual(ev["bed_candidates"],
                     ev["bed_refused_degree"] + ev["bed_refused_angle"] + ev["bed_refused_girth"]
                     + ev["bed_refused_clearance"] + ev["bed_edges"])
    test.assertLessEqual(ev["bed_candidates_tip"], ev["bed_candidates"])
    test.assertLessEqual(ev["bed_refused_clearance_at_end"], ev["bed_refused_clearance"])
    test.assertEqual(ev["bed_edges"], len(result["greedy_edges"]))
    test.assertLessEqual(ev["bed_lane_edges"], ev["bed_lane_candidates"])
    test.assertLessEqual(ev["bed_links"], ev["bed_candidates_link"])
    greedy = result["greedy_edges"]
    test.assertEqual(ev["bed_lane_edges"], int(np.count_nonzero(lane_edge_mask(case, greedy))))
    test.assertEqual(ev["bed_links"], int(np.count_nonzero(link_edge_mask(case, greedy))))
    tip_accepted = int(np.count_nonzero(greedy[:, 1] >= n_knots)) if len(greedy) else 0
    test.assertEqual(tip_accepted, ev["bed_edges"] - ev["bed_lane_edges"] - ev["bed_links"])
    # E5
    test.assertEqual(ev["bed_edges"], len(result["edges"]) + ev["bed_edges_dead_end"] + ev["bed_edges_island"]
                     + ev["bed_edges_dropped"])
    # F7, in place of E6
    test.assertEqual(ev["bed_knots"], int(result["kept"].sum()) + ev["bed_knots_on_feeders"]
                     + ev["bed_knots_dead_end"] + ev["bed_knots_island"])
    test.assertEqual((ev["bed_seeds_dead_end"], ev["bed_seeds_island"]), (0, 0))
    if case["feeders"] is None:
        test.assertEqual((ev["bed_knots_on_feeders"], ev["bed_knots_island"], ev["bed_edges_island"]), (0, 0, 0))
    # E7
    test.assertEqual(ev["bed_segments"], ev["bed_segments_walked"] + ev["bed_segments_chord"]
                     + ev["bed_segments_dropped"])
    test.assertEqual(len(result["chains"]),
                     ev["bed_segments_walked"] + ev["bed_segments_chord"] - ev["bed_segments_removed"])
    # F8
    test.assertLessEqual(ev["bed_links_kept"], ev["bed_links"])
    test.assertEqual(ev["bed_links_kept"], len(result["links"]))
    test.assertEqual(ev["bed_links_kept"], int(np.count_nonzero(link_edge_mask(case, result["edges"]))))
    # E8 and E9
    test.assertEqual(ev["bed_tip_edges"], int(result["tip_edges"].sum()))
    test.assertEqual(ev["bed_tips_attached"], int(np.count_nonzero(result["tip_edges"])))
    test.assertLessEqual(ev["bed_tips_attached"], ev["bed_tips"] - ev["bed_tips_inside_junction"])
    test.assertLessEqual(ev["bed_components_one_tree"], ev["bed_components"])
    test.assertLessEqual(ev["bed_walk_redraws"], case["attempts"] * ev["bed_segments"])
    # F9
    if case["feeders"] is None and f9_holds_by_condition(settings, case["d_min"], case["margin"]):
        test.assertEqual(ev["bed_lane_edges"], ev["bed_lane_candidates"])
        test.assertEqual(ev["bed_segments_dropped"], 0)


def shortest_cycle(adjacency):
    """The girth of a simple graph by breadth-first search from every vertex; inf for a forest."""
    best = math.inf
    for source in adjacency:
        depth = {source: 0}
        parent = {source: None}
        queue = collections.deque([source])
        while queue:
            u = queue.popleft()
            if 2 * depth[u] + 1 >= best:
                break                           # every cycle found from here on is no shorter
            for w in adjacency[u]:
                if w not in depth:
                    depth[w] = depth[u] + 1
                    parent[w] = u
                    queue.append(w)
                elif parent[u] != w:
                    best = min(best, depth[u] + depth[w] + 1)
    return best


def check_rules(test, case):
    """
    The greedy pass's rules on the kept edges, re-evaluated: the degree caps,
    min_angle (the feeder's inward direction at a tip), the kin angle
    between two bed edges at a shared end, chord to chord at least T for
    every pair sharing no end, and the girth by an exact search over the bed
    and the feeder segment graph. Returns the counts of what was checked.
    """
    result, settings = case["result"], case["settings"]
    edges, kept = result["edges"], result["kept"]
    n_knots, n_tips = len(result["knots"]), len(result["tips"])
    X = vertex_points(case)
    T = settings["diameter"] * case["d_min"] + case["margin"]
    degree = np.bincount(edges.ravel(), minlength=n_knots + n_tips)
    test.assertTrue(np.all(degree[:n_knots] <= 3))
    test.assertTrue(np.all(degree[:n_knots][~kept] == 0))
    np.testing.assert_array_equal(degree[n_knots:], result["tip_edges"])
    test.assertTrue(np.all(degree[n_knots:] <= settings["tip_edges"]))
    # a kept interior knot has two or three edges, a kept boundary knot one to three
    interior = ~result["knot_boundary"]
    test.assertTrue(np.all(degree[:n_knots][kept & interior] >= 2))
    test.assertTrue(np.all(degree[:n_knots][kept] >= 1))

    cos_min = math.cos(math.radians(settings["min_angle"]))
    cos_kin = math.cos(math.radians(bed.KIN_ANGLE_DEG))
    neighbours = collections.defaultdict(list)
    for a, b in edges.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    counts = {"pairs": 0, "tip pairs": 0, "chord pairs": 0}
    fnodes = None
    if case["feeders"] is not None:
        fnodes = np.asarray(case["feeders"][0])
        polylines = graph.polylines(fnodes)
        polyline_of = graph.polyline_of_column(fnodes)
    for v, others in neighbours.items():
        directions = [(X[w] - X[v]) / bed._dist(X[w][None, :], X[v][None, :])[0] for w in others]
        inward = None
        if v >= n_knots:
            column = int(result["tips"][v - n_knots])
            inward = bed._inward(fnodes, polylines[polyline_of[column]], column)
        for p in range(len(directions)):
            for q in range(p + 1, len(directions)):
                dot = float(bed._dot3(directions[p], directions[q]))
                test.assertLessEqual(dot, cos_min)
                test.assertLessEqual(dot, cos_kin)
                counts["pairs"] += 1
            if inward is not None:
                test.assertLessEqual(float(bed._dot3(inward, directions[p])), cos_min)
                counts["tip pairs"] += 1

    # chords that share no end stand at least T apart (the clamped closed form)
    if len(edges) > 1:
        a, b = np.triu_indices(len(edges), 1)
        ea, eb = edges[a], edges[b]
        share = ((ea[:, 0] == eb[:, 0]) | (ea[:, 0] == eb[:, 1]) | (ea[:, 1] == eb[:, 0])
                 | (ea[:, 1] == eb[:, 1]))
        p0, p1 = X[edges[:, 0]], X[edges[:, 1]]
        mid, half = (p0 + p1) / 2.0, bed._dist(p0, p1) / 2.0
        near = bed._dist(mid[a], mid[b]) < half[a] + half[b] + T
        a, b = a[~share & near], b[~share & near]
        if a.size:
            distance = bed._segment_distance(p0[a], p1[a], p0[b], p1[b])
            test.assertTrue(np.all(distance >= T), float(distance.min()))
        counts["chord pairs"] = int(a.size)

    # girth, over the kept edges and the feeder segment graph, a tip being the feeder vertex it stands on
    def vertex(v):
        return v if v < n_knots else n_knots + int(result["tips"][v - n_knots])

    links = [(vertex(a), vertex(b)) for a, b in edges.tolist()]
    if fnodes is not None:
        for path in graph.segments(fnodes, graph.build(fnodes)["edges"]):
            links.append((n_knots + int(path[0]), n_knots + int(path[-1])))
    test.assertEqual(len(set(tuple(sorted(link)) for link in links)), len(links))
    adjacency = collections.defaultdict(set)
    for a, b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)
    counts["girth"] = shortest_cycle(adjacency)
    test.assertGreaterEqual(counts["girth"], settings["girth"])
    return counts


def check_columns(test, case):
    """
    The written bed: one separator before each chain, BRIDGE labels and the
    bed diameter on its columns, its ends and every knot it passes written
    bitwise from the stored coordinates, the feeders' columns unchanged and
    a tip with k bed edges of degree 1 + k. Returns what was checked.
    """
    result, feeders = case["result"], case["feeders"]
    nodes, tree, chains = result["nodes"], result["tree"], result["chains"]
    X = vertex_points(case)
    test.assertGreater(len(chains), 0)
    n_feeders = 0 if feeders is None else np.asarray(feeders[0]).shape[1]
    d = case["settings"]["diameter"] * case["d_min"]
    column = n_feeders - 1
    for chain in chains:
        first, last, vertices = chain["first"], chain["last"], chain["vertices"]
        test.assertEqual(first, column + 2)
        test.assertTrue(np.all(np.isnan(nodes[:, first - 1])))
        test.assertEqual(int(tree[first - 1]), -1)
        test.assertTrue(np.all(np.isfinite(nodes[:, first:last + 1])))
        test.assertTrue(np.all(tree[first:last + 1] == BRIDGE))
        test.assertTrue(np.all(nodes[3, first:last + 1] == d))
        column = last
        test.assertEqual(nodes[:3, first].tobytes(), X[vertices[0]].tobytes())
        test.assertEqual(nodes[:3, last].tobytes(), X[vertices[-1]].tobytes())
        written = {nodes[:3, k].tobytes() for k in range(first, last + 1)}
        for v in vertices[1:-1]:
            test.assertIn(X[v].tobytes(), written)
        test.assertIn(chain["outcome"], ("walked", "chord"))
    test.assertEqual(column, nodes.shape[1] - 1)
    attached = 0
    if feeders is None:
        test.assertTrue(np.all(tree[np.isfinite(nodes[0])] == BRIDGE))
    else:
        test.assertEqual(nodes[:, :n_feeders].tobytes(), np.asarray(feeders[0], dtype=float).tobytes())
        np.testing.assert_array_equal(tree[:n_feeders], feeders[1])
        degree = graph.build(nodes)["degree"]
        for t, tip in enumerate(result["tips"].tolist()):
            test.assertEqual(int(degree[tip]), 1 + int(result["tip_edges"][t]))
            attached += int(result["tip_edges"][t]) > 0
    return {"chains": len(chains), "tips attached": attached}


# --------------------------------------------------------------------------
# 3.7's registries and the order's interface, which tests 2 and 19 read


# 3.7's registries, written out as released.
FOAM_3_7 = {"kind": "foam", "spacing": 7.5, "girth": 8, "min_angle": 60.0, "reach": 2.5, "diameter": 1.0,
            "persistence": 8.0, "feeder_stop": 2.0, "tip_edges": 2, "stretch": 1.0, "density": 0.5,
            "jitter": 0.25, "step": 0.5}
SETTING_KEYS_3_7 = ("kind", "spacing", "girth", "min_angle", "reach", "diameter", "persistence", "feeder_stop",
                    "tip_edges", "stretch", "density", "jitter", "step")
BED_EVENT_KEYS_3_7 = ("bed_proposals", "bed_proposals_on_feeders", "bed_proposals_overlapping", "bed_seeds",
                      "bed_tips", "bed_tips_inside_junction", "bed_candidates", "bed_candidates_tip",
                      "bed_refused_degree", "bed_refused_angle", "bed_refused_girth", "bed_refused_clearance",
                      "bed_refused_clearance_at_end", "bed_edges", "bed_edges_dead_end", "bed_edges_island",
                      "bed_edges_dropped", "bed_seeds_dead_end", "bed_seeds_island", "bed_segments",
                      "bed_segments_walked", "bed_segments_chord", "bed_segments_dropped", "bed_segments_removed",
                      "bed_walk_redraws", "bed_tips_attached", "bed_tip_edges", "bed_components",
                      "bed_components_one_tree")
MAIN_EVENT_KEYS_3_7 = (("bound_terminations", "walk_bound_redraws", "walk_bound_terminations", "collision_redraws",
                        "collision_terminations", "collision_truncated_stems", "root_relocations", "root_collisions",
                        "anastomosis_tips", "anastomosis_tips_inside_junction", "anastomosis_kin_skipped",
                        "anastomosis_behind_skipped", "anastomosis_selected", "anastomosis_bridges",
                        "anastomosis_bridges_arteriovenous", "anastomosis_bridges_same_tree",
                        "anastomosis_partner_tips", "anastomosis_partner_interior", "anastomosis_source_consumed",
                        "anastomosis_no_partner", "anastomosis_collision_failed", "anastomosis_bridge_redraws",
                        "guided_steps", "guidance_onset_steps", "unguided_steps", "guidance_undefined_steps",
                        "guidance_sense_flips", "guidance_bank_steps", "rung_sites", "rung_sites_near_junction",
                        "rung_sites_consumed", "rung_no_partner", "rung_collision_failed", "rung_bridges",
                        "rung_bridges_cross_tree", "rung_kin_skipped", "rung_angle_skipped", "rung_bridge_redraws")
                       + BED_EVENT_KEYS_3_7)
KINDS_3_7 = ("foam",)
SEED_OUTCOMES_3_7 = ("seed", "on_feeders", "overlapping")
SUBSTREAMS_3_7 = {"seeds": 1, "order": 2, "walks": 3}
INDEX_COLUMNS_3_7 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points", "polylines",
                     "tips", "junctions", "cycles", "components", "total_length", "diameter_p50", "diameter_p90",
                     "diameter_p99", "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min",
                     "diameter_max", "min_point_diameter", "clearance_min", "clearance_violations", "bridges",
                     "events", "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                     "nodes_sha256", "program_sha256", "frame_kind", "capillary_order", "larger_order",
                     "capillary_polar_order", "capillary_length_share", "capillary_volume_share",
                     "capillary_segment_median", "transverse_spacing_median", "rungs")
MAIN_RNG_STREAMS_3_7 = {"walk": 1, "anastomosis": 2, "rungs": 5}
LIBRARY_RNG_STREAMS_3_7 = {"ratio": 4}
MAIN_CLOSURE_3_7 = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel", "tortuosity",
                    "collisions", "anastomosis", "graph", "spatial", "guidance", "connections", "bed")

# The order's interface, written out: the fibre keys and what 3.8 appends.
FIBRE_KEYS = ("kind", "spacing", "girth", "min_angle", "reach", "diameter", "persistence", "feeder_stop",
              "tip_edges", "density", "jitter", "step", "axis", "knot", "sway", "link_spacing", "link_reach", "run",
              "layout", "unit", "sheet_length")
FIBRE_ONLY_KEYS = ("axis", "knot", "sway", "link_spacing", "link_reach", "run", "layout", "unit", "sheet_length")
EVENT_KEYS_APPENDED = ("bed_proposals_outside", "bed_knots", "bed_knots_boundary", "bed_knots_on_feeders",
                       "bed_lane_candidates", "bed_lane_edges_on_feeders", "bed_breaks", "bed_lane_edges",
                       "bed_link_sites", "bed_link_sites_no_partner", "bed_link_sites_angle",
                       "bed_link_sites_on_feeders", "bed_candidates_link", "bed_links", "bed_links_kept",
                       "bed_knots_dead_end", "bed_knots_island")
SUBSTREAMS_3_8 = {"seeds": 1, "order": 2, "walks": 3, "lanes": 4, "knots": 5, "links": 6, "breaks": 7,
                  "tip_order": 8, "fibre_walks": 9}


# --------------------------------------------------------------------------
# Test 2: the settings


# Every bound of the order's fibre table refused by parse_settings, as
# (name, the change to FIBRE, a word of the message): each is refused for
# its own reason. The other kind's key, a kind outside KINDS and an unknown
# key join them.
PARSE_REFUSED = (
    ("spacing 0", {"spacing": 0.0}, "spacing"),
    ("spacing -1", {"spacing": -1.0}, "spacing"),
    ("density 0", {"density": 0.0}, "density"),
    ("density 0.61", {"density": 0.61}, "density"),
    ("girth 2", {"girth": 2}, "girth"),
    ("girth 17", {"girth": 17}, "girth"),
    ("girth 8.0", {"girth": 8.0}, "girth"),
    ("girth true", {"girth": True}, "girth"),
    ("min_angle -1", {"min_angle": -1.0}, "min_angle"),
    ("min_angle 121", {"min_angle": 121.0}, "min_angle"),
    ("reach 1", {"reach": 1.0}, "reach"),
    ("diameter 0", {"diameter": 0.0}, "diameter"),
    ("persistence 0", {"persistence": 0.0}, "persistence"),
    ("feeder_stop 0.5", {"feeder_stop": 0.5}, "feeder_stop"),
    ("tip_edges 0", {"tip_edges": 0}, "tip_edges"),
    ("tip_edges 3", {"tip_edges": 3}, "tip_edges"),
    ("jitter -0.1", {"jitter": -0.1}, "jitter"),
    ("step 0.05", {"step": 0.05}, "step"),
    ("axis zero", {"axis": [0.0, 0.0, 0.0]}, "axis"),
    ("axis of two numbers", {"axis": [1.0, 0.0]}, "axis"),
    ("axis with a nan", {"axis": [1.0, math.nan, 0.0]}, "axis"),
    ("axis with an inf", {"axis": [1.0, math.inf, 0.0]}, "axis"),
    ("axis with a bool", {"axis": [True, 0.0, 0.0]}, "axis"),
    ("axis a word", {"axis": "x"}, "axis"),
    ("knot below step x diameter", {"knot": 0.49}, "knot"),
    ("sway -0.1", {"sway": -0.1}, "sway"),
    ("link_spacing below knot / 2", {"link_spacing": 1.99}, "link_spacing"),
    ("link_reach 1", {"link_reach": 1.0}, "link_reach"),
    ("run 0", {"run": 0.0}, "run"),
    ("run -1", {"run": -1.0}, "run"),
    ("run with link_spacing null", {"run": 50.0, "link_spacing": None}, "run needs link_spacing"),
    ("layout grid", {"layout": "grid"}, "layout"),
    ("layout null", {"layout": None}, "layout"),
    ("unit 0", {"unit": 0.0}, "bed unit"),
    ("sheet_length 0", {"sheet_length": 0.0}, "sheet_length"),
    ("sheet_length -1", {"sheet_length": -1.0}, "sheet_length"),
    ("foam's stretch", {"stretch": 1.0}, "stretch"),
    ("kind lattice", {"kind": "lattice"}, "kind"),
    ("an unknown key", {"colour": 1}, "colour"),
)
# The refusals check_settings adds at d_min 1 and margin 1, T = 2: the
# lanes' hard core and the knot spacing must exceed T. parse_settings takes
# each of them.
CHECK_REFUSED = (
    ("spacing at T", {"spacing": 2.0}, "spacing"),
    ("spacing below T", {"spacing": 1.5}, "spacing"),
    ("knot at T", {"knot": 2.0}, "knot"),
    ("knot below T", {"knot": 1.5}, "knot"),
)
# Foam refusing each fibre key, with its kind left out and given.
CROSS_KIND = tuple((f"foam with {key}{' (kind given)' if given else ''}", dict(given, **{key: FIBRE_DEFAULTS[key]}),
                    key) for key in FIBRE_ONLY_KEYS for given in ({}, {"kind": "foam"}))
# Every refusal above as (name, the whole spec, the word, whether
# parse_settings takes the spec and only check_settings refuses it).
REFUSED = (tuple((name, dict(FIBRE, **change), word, False) for name, change, word in PARSE_REFUSED)
           + tuple((name, dict(FIBRE, **change), word, True) for name, change, word in CHECK_REFUSED)
           + tuple((name, spec, word, False) for name, spec, word in CROSS_KIND))
# The value at each bound that is still accepted, so that every bound stands
# where the table puts it.
ACCEPTED_AT_BOUND = ({"density": 0.6}, {"girth": 3}, {"girth": 16}, {"min_angle": 0.0}, {"min_angle": 120.0},
                     {"tip_edges": 1}, {"jitter": 0.0}, {"feeder_stop": 1.0}, {"step": 0.1, "knot": 0.1},
                     {"knot": 0.5}, {"sway": 0.0}, {"link_spacing": 2.0}, {"link_spacing": None},
                     {"run": 50.0}, {"run": 50.0, "link_spacing": 2.0}, {"axis": None}, {"layout": "sheets"},
                     {"spacing": 2.01}, {"knot": 2.01}, {"unit": 1e-3}, {"sheet_length": 1e-3},
                     {"reach": 1.001}, {"link_reach": 1.001})


def bed_run_argv(spec, out):
    """
    A bed run's command line, as tests/test_bed.py's cli_argv writes it (a
    48 cube at voxel size 1, d_min 1, the walk, d0 8, seed 3), with --bed
    the spec as JSON and --out the directory given.
    """
    return ["--grow-in-volume", "--d-min", "1", "--volume", "48", "48", "48", "--fit", "voxel_size",
            "--voxel-size", "1", "--tortuosity", "walk", "--persistence", "10", "--d0", "8", "0",
            "--iterations", "6", "6", "--count", "1", "--seed", "3", "--bed", json.dumps(spec), "--out", out]


def run_main(argv):
    """main.main(argv) with its output captured; returns (status, SystemExit or None)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return main.main(argv), None
        except SystemExit as caught:
            return None, caught


def bits(values):
    """The exact bits of a sequence of floats."""
    return [float(v).hex() for v in values]


class SettingsTests(unittest.TestCase):
    SPECS = ({"kind": "fibre"}, FIBRE, dict(FIBRE_DEFAULTS, axis=[2, 0, 0]), {"kind": "fibre", "axis": [1, 2, 2]},
             {"kind": "fibre", "axis": np.array([0.0, 0.0, -3.0])},
             {"kind": "fibre", "spacing": np.float64(3.2), "girth": np.int64(5), "density": 0.6, "knot": 2.5,
              "sway": 0, "link_spacing": None},
             {"kind": "fibre", "run": 50, "link_spacing": 10, "layout": "sheets", "unit": 40, "sheet_length": 1},
             {"kind": "fibre", "min_angle": 0, "tip_edges": 1, "jitter": 0, "step": 0.1, "knot": 0.1},
             bed.BedSettings(kind="fibre", sway=0.5))

    def test_foam_s_defaults_and_preset_are_3_7_s_thirteen_keys(self):
        # byte for byte: the dumps are equal, so the keys, their order and every value's type are 3.7's
        self.assertEqual(bed.SETTING_KEYS, SETTING_KEYS_3_7)
        self.assertEqual(bed.KIND_KEYS["foam"], SETTING_KEYS_3_7)
        for name, settings in (("{}", bed.parse_settings({})), ("kind foam", bed.parse_settings({"kind": "foam"})),
                               ("BedSettings()", bed.BedSettings().as_dict()),
                               ("the foam preset", bed.parse_settings(main.FAMILIES["foam"]["bed"]))):
            with self.subTest(settings=name):
                self.assertEqual(settings, FOAM_3_7)
                self.assertEqual(json.dumps(settings), json.dumps(FOAM_3_7))

    def test_fibre_s_defaults_are_the_order_s_table_in_its_key_order(self):
        self.assertEqual(bed.FIBRE_SETTING_KEYS, FIBRE_KEYS)
        self.assertEqual(bed.KIND_KEYS, {"foam": SETTING_KEYS_3_7, "fibre": FIBRE_KEYS})
        self.assertEqual(list(FIBRE_DEFAULTS), list(FIBRE_KEYS))
        settings = bed.parse_settings({"kind": "fibre"})
        self.assertEqual(json.dumps(settings), json.dumps(FIBRE_DEFAULTS))
        self.assertIs(type(settings["girth"]), int)
        self.assertIs(type(settings["tip_edges"]), int)
        # the kind's defaults fill a field left out, so that the two ways of making it are one
        self.assertEqual(bed.BedSettings(kind="fibre"), bed.BedSettings.parse({"kind": "fibre"}))
        self.assertEqual(json.dumps(bed.BedSettings(kind="fibre").as_dict()), json.dumps(FIBRE_DEFAULTS))
        self.assertEqual(bed.BedSettings(kind="fibre", axis=[2, 0, 0]),
                         bed.BedSettings.parse({"kind": "fibre", "axis": [2, 0, 0]}))

    def test_normalising_twice_changes_nothing_and_survives_json(self):
        for spec in self.SPECS:
            with self.subTest(spec=spec):
                settings = bed.parse_settings(spec)
                self.assertEqual(list(settings), list(FIBRE_KEYS))
                again = bed.parse_settings(settings)
                self.assertEqual(json.dumps(again), json.dumps(settings))
                self.assertEqual(json.loads(json.dumps(settings)), settings)
                self.assertEqual(bed.parse_settings(json.loads(json.dumps(settings))), settings)
                self.assertEqual(bed.BedSettings.parse(spec).as_dict(), settings)
                self.assertEqual(bed.BedSettings(**settings), bed.BedSettings.parse(spec))
                self.assertIs(type(settings["girth"]), int)
                self.assertIs(type(settings["tip_edges"]), int)
                for key in FIBRE_KEYS:
                    if key not in ("kind", "girth", "tip_edges", "axis", "layout") and settings[key] is not None:
                        self.assertIs(type(settings[key]), float, key)
                if settings["axis"] is not None:
                    self.assertEqual([type(v) for v in settings["axis"]], [float] * 3)
                    self.assertEqual(bits(again["axis"]), bits(settings["axis"]))

    def test_each_kind_refuses_the_other_s_keys_and_names_its_own(self):
        cases = [(f"foam with {key}", dict(given, **{key: FIBRE_DEFAULTS[key]}), key, bed.SETTING_KEYS)
                 for key in FIBRE_ONLY_KEYS for given in ({}, {"kind": "foam"})]
        cases += [("fibre with stretch", {"kind": "fibre", "stretch": 1.0}, "stretch", FIBRE_KEYS),
                  ("fibre with stretch 2", dict(FIBRE, stretch=2.0), "stretch", FIBRE_KEYS),
                  ("foam with an unknown key", {"colour": 1}, "colour", bed.SETTING_KEYS),
                  ("fibre with an unknown key", dict(FIBRE, colour=1), "colour", FIBRE_KEYS)]
        for name, spec, key, keys in cases:
            with self.subTest(spec=name):
                for refuse in (bed.parse_settings, bed.BedSettings.parse,
                               lambda s: bed.check_settings(s, d_min=D_MIN, collision_margin=MARGIN)):
                    with self.assertRaises(ValueError) as caught:
                        refuse(spec)
                    message = str(caught.exception)
                    self.assertIn(f"unknown bed setting {key!r}", message)
                    self.assertIn(", ".join(keys), message)                 # the kind's keys, in order
        # the dataclass refuses them too, whether the kind is given or defaulted, given any value but None,
        # which is what an instance holds for a field its kind does not take (so that dataclasses.replace works)
        given_values = dict(FIBRE_DEFAULTS, axis=X_AXIS, run=50.0)
        for key in FIBRE_ONLY_KEYS:
            with self.subTest(field=key):
                for kind in ({}, {"kind": "foam"}):
                    with self.assertRaises(ValueError):
                        bed.BedSettings(**dict(kind, **{key: given_values[key]}))
                    self.assertEqual(bed.BedSettings(**dict(kind, **{key: None})), bed.BedSettings())
        with self.assertRaises(ValueError):
            bed.BedSettings(kind="fibre", stretch=1.0)
        self.assertEqual(bed.BedSettings(kind="fibre", stretch=None), bed.BedSettings(kind="fibre"))

    def test_each_bound_stands_where_the_table_puts_it(self):
        # the last value accepted at each bound: parse_settings keeps it, and
        # check_settings at d_min 1 and margin 1 takes every one whose spacing
        # and knot exceed T = 2
        for change in ACCEPTED_AT_BOUND:
            with self.subTest(change=change):
                settings = bed.parse_settings(dict(FIBRE, **change))
                for key, value in change.items():
                    self.assertEqual(settings[key], value)
                if settings["spacing"] > 2.0 and settings["knot"] > 2.0:
                    self.assertEqual(bed.check_settings(settings, d_min=D_MIN, collision_margin=MARGIN).as_dict(),
                                     settings)

    def test_check_settings_holds_the_spacing_and_the_knot_above_the_diameter_plus_the_margin_in_d_min(self):
        # (change, d_min, margin, refused): h d_min > d d_min + margin and l d_min > d d_min + margin
        cases = (({"spacing": 1.5}, 2.0, 1.0, True), ({"spacing": 1.6}, 2.0, 1.0, False),
                 ({"knot": 1.5}, 2.0, 1.0, True), ({"knot": 1.6}, 2.0, 1.0, False),
                 ({"spacing": 1.0}, 1.0, 0.0, True), ({"spacing": 1.01, "knot": 1.01}, 1.0, 0.0, False),
                 ({"knot": 1.0}, 1.0, 0.0, True), ({"diameter": 2.0, "spacing": 3.0}, 1.0, 1.0, True),
                 ({"diameter": 2.0, "knot": 3.0}, 1.0, 1.0, True), ({"diameter": 2.0, "knot": 3.5}, 1.0, 1.0, False))
        for change, d_min, margin, refused in cases:
            with self.subTest(change=change, d_min=d_min, margin=margin):
                spec = dict(FIBRE, **change)
                self.assertEqual(bed.parse_settings(spec)["kind"], "fibre")
                if refused:
                    with self.assertRaises(ValueError):
                        bed.check_settings(spec, d_min=d_min, collision_margin=margin)
                else:
                    self.assertEqual(bed.check_settings(spec, d_min=d_min, collision_margin=margin).as_dict(),
                                     bed.parse_settings(spec))

    def test_parse_settings_and_check_settings_refuse_each_bound_before_any_generator_exists(self):
        for name, spec, word, parsed in REFUSED:
            with self.subTest(refusal=name):
                random.seed(11)
                np.random.seed(11)
                before = global_states()
                with recorded_generators() as created:
                    if parsed:
                        self.assertEqual(bed.parse_settings(spec)["kind"], "fibre")
                    else:
                        with self.assertRaises(ValueError) as caught:
                            bed.parse_settings(spec)
                        self.assertIn(word, str(caught.exception))
                    with self.assertRaises(ValueError) as caught:
                        bed.check_settings(spec, d_min=D_MIN, collision_margin=MARGIN)
                    self.assertIn(word, str(caught.exception))
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)

    def test_build_refuses_each_bound_before_any_generator_exists(self):
        # a bed alone of 16 lanes in a cube of side 30; the base call is a bed
        domain = (np.zeros(3), np.full(3, 30.0))
        result = bed.build(domain, FIBRE, d_min=D_MIN, seed=1, collision_margin=MARGIN)
        self.assertEqual(len(result["seeds"]), 16)
        for name, spec, word, _ in REFUSED:
            with self.subTest(refusal=name):
                events = {"bed_seeds": 5}
                random.seed(13)
                np.random.seed(13)
                before = global_states()
                with recorded_generators() as created:
                    with self.assertRaises(ValueError) as caught:
                        bed.build(domain, spec, d_min=D_MIN, seed=1, collision_margin=MARGIN, events=events)
                self.assertNotIsInstance(caught.exception, bed.SeedsNotPlaced)
                self.assertIn(word, str(caught.exception))
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)
                self.assertEqual(events, {"bed_seeds": 5})

    def test_the_command_line_refuses_each_bound_before_anything_is_written(self):
        # --bed through main.main: refused as a bed run, for its own reason,
        # with no generator created and nothing in --out
        for name, spec, word, _ in REFUSED:
            with self.subTest(refusal=name), tempfile.TemporaryDirectory() as directory:
                out = os.path.join(directory, "out")
                random.seed(17)
                np.random.seed(17)
                before = global_states()
                with recorded_generators() as created:
                    status, caught = run_main(bed_run_argv(spec, out))
                self.assertIsNone(status)
                self.assertIsInstance(caught.code, str)
                self.assertTrue(caught.code.startswith("--bed"), caught.code)
                self.assertIn(word, caught.code)
                self.assertTrue(not os.path.exists(out) or os.listdir(out) == [])
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)

    def test_an_axis_is_stored_as_a_unit_list_kept_bitwise_when_already_unit(self):
        third = 1.0 / math.sqrt(3.0)
        for given, stored in (([2, 0, 0], [1.0, 0.0, 0.0]), ([0, 3, 4], [0.0, 0.6, 0.8]),
                              ([0.0, 0.0, -2.0], [0.0, 0.0, -1.0]), (np.array([0.0, 2.0, 0.0]), [0.0, 1.0, 0.0]),
                              ((np.float64(5.0), 0, 0), [1.0, 0.0, 0.0]),
                              ([1.0 + 2e-12, 0.0, 0.0], [1.0, 0.0, 0.0]),
                              # unit within 1e-12: kept as given, bit for bit
                              ([0.6, 0.8, 0.0], [0.6, 0.8, 0.0]), ([third] * 3, [third] * 3),
                              ([1.0 + 5e-13, 0.0, 0.0], [1.0 + 5e-13, 0.0, 0.0]),
                              ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0])):
            with self.subTest(axis=given):
                settings = bed.parse_settings({"kind": "fibre", "axis": given})
                self.assertIs(type(settings["axis"]), list)
                self.assertEqual(bits(settings["axis"]), bits(stored))
                self.assertEqual(bits(bed.BedSettings(kind="fibre", axis=given).axis), bits(stored))
                self.assertEqual(bits(bed.parse_settings(settings)["axis"]), bits(stored))


# --------------------------------------------------------------------------
# Test 3: the planned counts


# (extents, the change to FIBRE, d_min, the lanes N, planned_candidates),
# for coordinate and oblique axes; the counts are the order's closed forms,
# written out (N's and each ceiling's fraction is far from an integer, or
# the quotient is exact).
PLANNED = (
    ((40.0, 60.0, 80.0), {"axis": [1.0, 0.0, 0.0]}, 1.0, 93, 1116),
    ((40.0, 60.0, 80.0), {"axis": [0.0, 1.0, 0.0]}, 1.0, 61, 1068),
    ((40.0, 60.0, 80.0), {"axis": [0.0, 0.0, -1.0]}, 1.0, 46, 1058),
    ((40.0, 60.0, 80.0), {"axis": [1.0, 0.0, 0.0], "link_spacing": None}, 1.0, 93, 1023),
    ((160.0, 120.0, 90.0), {"axis": [0.0, 0.0, 1.0]}, 2.0, 93, 1219),
    ((40.0, 60.0, 80.0), {"axis": [1.0, 2.0, 2.0]}, 1.0, 102, 3132),
    ((48.0, 48.0, 48.0), {"axis": [1.0, 1.0, 0.0]}, 1.0, 62, 1222),
    ((90.0, 60.0, 45.0), {"axis": [3.0, -4.0, 12.0]}, 1.0, 133, 3059),
    ((160.0, 120.0, 90.0), {"axis": [1.0, 1.0, 1.0], "spacing": 3.2, "density": 0.6, "knot": 2.5,
                            "link_spacing": 10.0}, 2.0, 363, 17924),
)
SECTION = (40.0, 60.0, 80.0)          # the non-cubic domain of tests 3 and 14
COORDINATE_AXES = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
OBLIQUE = [1.0, 2.0, 2.0]

_OWN_BUILDS = {}


def framed(seed=1):
    """
    A bed alone in the 40 x 60 x 80 box at the defaults, its settings axis
    null: it runs along the grow_direction (y) of main's unguided frame.
    """
    key = ("framed", seed)
    if key not in _OWN_BUILDS:
        frame = unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR, [v / 2.0 for v in SECTION])
        _OWN_BUILDS[key] = build_case(f"alone, 40 x 60 x 80, seed {seed}, axis null, main's frame",
                                      (np.zeros(3), np.array(SECTION)), FIBRE_DEFAULTS, seed=seed, frame=frame)
    return _OWN_BUILDS[key]


def sectioned(axis, seed=1):
    """A bed alone in the 40 x 60 x 80 box along a coordinate axis, link_spacing null and no frame."""
    key = ("sectioned", tuple(axis), seed)
    if key not in _OWN_BUILDS:
        _OWN_BUILDS[key] = build_case(f"alone, 40 x 60 x 80, seed {seed}, axis {axis}, no links",
                                      (np.zeros(3), np.array(SECTION)), dict(FIBRE, axis=axis, link_spacing=None),
                                      seed=seed)
    return _OWN_BUILDS[key]


def closed_form(extents, settings, d_min, a):
    """
    The order's (N, planned candidates) for the unit axis a: N =
    floor(density A / h^2) and N (ceil(E / l) + 1) + ceil(N ceil(E / l) l /
    (2 lambda)), with A = |a_x| Y'Z' + |a_y| X'Z' + |a_z| X'Y' and E =
    |a_x| X' + |a_y| Y' + |a_z| Z', X', Y', Z' the extents less d.
    """
    d, h, l = (settings[key] * d_min for key in ("diameter", "spacing", "knot"))
    x, y, z = (extent - d for extent in extents)
    area = abs(a[0]) * y * z + abs(a[1]) * x * z + abs(a[2]) * x * y
    extent = abs(a[0]) * x + abs(a[1]) * y + abs(a[2]) * z
    n = math.floor(settings["density"] * area / h ** 2)
    intervals = math.ceil(extent / l)
    links = 0
    if settings["link_spacing"] is not None:
        links = math.ceil(n * intervals * l / (2.0 * settings["link_spacing"] * d_min))
    return n, n * (intervals + 1) + links


def lane_term(case):
    """(ceil(E / l) + 1, N (ceil(E / l) + 1)): the most lane edges a lane, and all N lanes, can carry."""
    settings, d_min, a = case["settings"], case["d_min"], resolved_axis(case)
    x, y, z = (extent - settings["diameter"] * d_min for extent in case["extents"])
    extent = abs(a[0]) * x + abs(a[1]) * y + abs(a[2]) * z
    per_lane = math.ceil(extent / (settings["knot"] * d_min)) + 1
    return per_lane, len(case["result"]["seeds"]) * per_lane


class PlannedCountTests(unittest.TestCase):
    """
    The planned counts of a fibre bed. Built over planned, as GuardTests
    records foam's (d_min 1, margin 1; the lane term N (ceil(E / l) + 1),
    the link term ceil(N ceil(E / l) l / (2 lambda)), the total
    planned_candidates; built: bed_lane_candidates, bed_candidates_link and
    their sum), measured on seeds 11-20 and then on seed 1, which the fast
    tests read:
      bed alone, side 48, axis x (44 lanes; 572 + 53 = 625 planned): lanes
        0.897-0.907, links 0.528-0.849 (sites 0.736-1.038), total
        0.869-0.893; seed 1: 0.907, 0.623, 0.883;
      the foam member's feeders, R 4, axis x (69 lanes; 1104 + 104 =
        1208): lanes 0.837-0.887, links 0.519-0.827 (sites 0.788-1.029),
        total 0.821-0.872; seed 1: 0.892, 0.769, 0.882;
      bed alone, side 48, axis (1, 2, 2) / 3 (73 lanes; 1533 + 146 =
        1679): lanes 0.303-0.337, links 0.185-0.267 (sites 0.247-0.308),
        total 0.295-0.326; seed 1: 0.288, 0.178, 0.279 (an oblique lane is
        shorter than E);
      40 x 60 x 80, settings axis null along main's frame (y; 61 lanes;
        976 + 92 = 1068): lanes 0.914-0.926, links 0.598-0.880 (sites
        0.793-1.087), total 0.895-0.913; seed 1: 0.926, 0.772, 0.913;
      40 x 60 x 80, link_spacing null, along x / y / z (1023, 976 and 966
        planned): 0.877-0.889 / 0.914-0.926 / 0.936-0.943; seed 1: 0.894 /
        0.926 / 0.943.
    The site ratio passes 1 on some seeds (the planned link term is the
    expected number of sites), the link ratio standing below it by the
    sites with no feasible partner (F5), so only the total is asserted, at
    most 1; the most
    lane edges one lane carried was ceil(E / l) + 1 less 1 along a
    coordinate axis and less 3 to 5 along the oblique one.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = ([alone(1), fed(1), alone(1, axis=OBLIQUE), framed(1)]
                     + [sectioned(axis) for axis in COORDINATE_AXES])

    def test_the_planned_counts_are_the_closed_forms(self):
        for extents, change, d_min, lanes, candidates in PLANNED:
            settings = bed.parse_settings(dict(FIBRE, **change))
            with self.subTest(extents=extents, axis=change["axis"], d_min=d_min):
                self.assertEqual(closed_form(extents, settings, d_min, settings["axis"]), (lanes, candidates))
                self.assertEqual(bed.planned_seeds(extents, settings, d_min), lanes)
                self.assertEqual(bed.planned_candidates(extents, settings, d_min), candidates)
                # without links the second term is 0
                if settings["link_spacing"] is not None:
                    unlinked = dict(settings, link_spacing=None)
                    self.assertEqual(bed.planned_candidates(extents, unlinked, d_min),
                                     closed_form(extents, unlinked, d_min, settings["axis"])[1])
                    self.assertLess(bed.planned_candidates(extents, unlinked, d_min), candidates)

    def test_a_null_settings_axis_plans_along_the_axis_argument_and_is_refused_without_it(self):
        for extents, change, d_min, lanes, candidates in PLANNED:
            settings = bed.parse_settings(dict(FIBRE, **dict(change, axis=None)))
            raw = change["axis"]
            with self.subTest(extents=extents, axis=raw, d_min=d_min):
                # the argument is normalised as the settings' axis is
                self.assertEqual(bed.planned_seeds(extents, settings, d_min, axis=raw), lanes)
                self.assertEqual(bed.planned_candidates(extents, settings, d_min, axis=raw), candidates)
                self.assertEqual(bed.planned_seeds(extents, settings, d_min, axis=[3.0 * v for v in raw]), lanes)
                self.assertEqual(bed.planned_candidates(extents, settings, d_min, axis=np.array(raw)), candidates)
                for plan in (bed.planned_seeds, bed.planned_candidates):
                    with self.assertRaises(ValueError) as caught:
                        plan(extents, settings, d_min)
                    self.assertIn("axis", str(caught.exception))
                    with self.assertRaises(ValueError):
                        plan(extents, settings, d_min, axis=[0.0, 0.0, 0.0])
        # the settings' axis wins over the argument, and foam ignores it
        along_x = dict(FIBRE, axis=[1.0, 0.0, 0.0])
        self.assertEqual(bed.planned_seeds(SECTION, along_x, D_MIN, axis=[0.0, 0.0, 1.0]), 93)
        self.assertEqual(bed.planned_candidates(SECTION, along_x, D_MIN, axis=[0.0, 0.0, 1.0]), 1116)
        for foam in (FOAM_3_7, dict(FOAM_3_7, stretch=2.0)):
            with self.subTest(foam_stretch=foam["stretch"]):
                for axis in ([0.0, 0.0, 1.0], [1.0, 2.0, 2.0]):
                    self.assertEqual(bed.planned_seeds(SECTION, foam, D_MIN, axis=axis),
                                     bed.planned_seeds(SECTION, foam, D_MIN))
                    self.assertEqual(bed.planned_candidates(SECTION, foam, D_MIN, axis=axis),
                                     bed.planned_candidates(SECTION, foam, D_MIN))

    def test_an_inset_box_without_volume_plans_no_lane(self):
        # extent 1 = d along the axis: the inset box is flat, though its section is not
        for extents in ((1.0, 60.0, 80.0), (40.0, 1.0, 80.0), (0.5, 60.0, 80.0)):
            with self.subTest(extents=extents):
                self.assertEqual(bed.planned_seeds(extents, FIBRE, D_MIN), 0)
                self.assertEqual(bed.planned_candidates(extents, FIBRE, D_MIN), 0)

    def test_the_built_beds_have_the_planned_lanes(self):
        # E2 / F2, the counts each build planned before drawing: N along the resolved axis
        for case in self.cases:
            with self.subTest(build=case["label"]):
                planned = bed.planned_seeds(case["extents"], case["settings"], case["d_min"], axis=resolved_axis(case))
                self.assertEqual(case["events"]["bed_seeds"], planned)
                self.assertEqual(len(case["result"]["seeds"]), planned)
        self.assertEqual([len(sectioned(axis)["result"]["seeds"]) for axis in COORDINATE_AXES], [93, 61, 46])
        self.assertEqual(resolved_axis(framed(1)).tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(len(framed(1)["result"]["seeds"]), 61)

    def test_no_lane_carries_more_than_ceil_e_over_l_plus_one_lane_edges(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                result = case["result"]
                per_lane, total = lane_term(case)
                edges = np.bincount(result["knot_lane"], minlength=len(result["seeds"])) - 1
                self.assertTrue(np.all(edges >= 1))
                self.assertLessEqual(int(edges.max()), per_lane)
                self.assertLessEqual(int(edges.sum()), total)
                self.assertLessEqual(case["events"]["bed_lane_candidates"], total)
                self.assertLessEqual(case["events"]["bed_lane_candidates"], int(edges.sum()))

    def test_the_planned_candidates_bound_the_candidates_built(self):
        # the guard's count leaves the tips out, as foam's does; the ratios
        # of seed 1 are in the class docstring's ranges
        for case in self.cases:
            with self.subTest(build=case["label"]):
                ev = case["events"]
                planned = bed.planned_candidates(case["extents"], case["settings"], case["d_min"],
                                                 axis=resolved_axis(case))
                built = ev["bed_candidates"] - ev["bed_candidates_tip"]
                self.assertEqual(built, ev["bed_lane_candidates"] + ev["bed_candidates_link"])
                self.assertGreater(built, 0)
                self.assertLessEqual(built, planned)

    @SLOW
    def test_the_lane_and_total_bounds_hold_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for case in (alone(seed), fed(seed), alone(seed, axis=OBLIQUE), framed(seed)):
                with self.subTest(build=case["label"]):
                    ev = case["events"]
                    per_lane, total = lane_term(case)
                    edges = np.bincount(case["result"]["knot_lane"]) - 1
                    self.assertLessEqual(int(edges.max()), per_lane)
                    self.assertLessEqual(ev["bed_lane_candidates"], total)
                    planned = bed.planned_candidates(case["extents"], case["settings"], case["d_min"],
                                                     axis=resolved_axis(case))
                    self.assertLessEqual(ev["bed_candidates"] - ev["bed_candidates_tip"], planned)


# --------------------------------------------------------------------------
# Test 14: the frame


# The fields a fibre record sets, whatever its source.
AXIS_FIELDS = {"kind": "axis", "sense": "nematic", "tree_senses": None, "normal": None, "rules": []}
SMALL = (np.zeros(3), np.full(3, 24.0))          # a cube of side 24: 10 lanes along a coordinate axis


def polar_axis_frame():
    """The record of a pair grown under a polar axis rule along (0, 3, 4) / 5: kind "axis", its tree senses and rule."""
    guidance = Guidance([{"below": None, "field": "axis", "axis": [0, 3, 4], "sense": "polar", "length": 5.0}], D_MIN)
    bound = [guidance.bind(np.array([0.0, 1.0, 0.0])), guidance.bind(np.array([0.0, -1.0, 0.0]))]
    return guidance.frame_record([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [12.0, 12.0, 12.0], bound)


def plane_frame():
    """The record of a tree grown under a plane rule of normal z: kind "plane", no axis."""
    guidance = Guidance([{"below": None, "field": "plane", "normal": [0, 0, 1], "length": 2.0}], D_MIN)
    return guidance.frame_record([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [12.0, 12.0, 12.0],
                                 [guidance.bind(np.array([0.0, 1.0, 0.0]))])


def axis_record(frame, a):
    """The order's fibre record of a frame: the frame's fields kept, kind "axis" along a, nematic, no rules."""
    return dict(frame, **AXIS_FIELDS, axis=[float(v) for v in a])


def constructed_record(a, domain):
    """The record the order builds with no frame: about the domain's centre, the domain's own axes."""
    low, high = (np.asarray(corner, dtype=float) for corner in domain)
    return {"frame_version": 1, "kind": "axis", "axis": [float(v) for v in a], "sense": "nematic",
            "tree_senses": None, "normal": None, "origin": [float(v) for v in (low + high) / 2.0],
            "grow_direction": [0.0, 1.0, 0.0], "grow_perpendicular": [0.0, 0.0, 1.0], "rules": []}


def chords_as_polylines(case):
    """
    The chord graph of a bed alone: every kept edge as a straight polyline
    of its two knots, at the bed diameter, NaN between polylines.
    """
    result = case["result"]
    d = case["settings"]["diameter"] * case["d_min"]
    columns = []
    for a, b in result["edges"].tolist():
        columns += [np.full(4, np.nan), np.append(result["knots"][a], d), np.append(result["knots"][b], d)]
    return np.array(columns[1:]).T


def describe_bed(case, nodes):
    """
    describe as the order calls it on a bed alone, its edges from
    graph.build, the domain as the volume, the bed's frame and d_ref d_min,
    with the planes of transverse_spacing (their crossings and section areas)
    caught on their way out.
    """
    real = describe._transverse_spacing
    planes = []

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        planes.append(args[8] if len(args) > 8 else kwargs.get("planes"))
        return out

    low, high = case["domain"]
    with mock.patch.object(describe, "_transverse_spacing", side_effect=spy):
        report = describe.describe(nodes, graph.build(nodes)["edges"], volume=high - low,
                                   frame=case["result"]["frame"], d_ref=case["d_min"])
    (caught,) = planes
    return report, caught


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {"polar axis": polar_axis_frame(), "plane": plane_frame(),
                      "unguided": unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR, (12.0,) * 3),
                      "turned": frames.transform_frame(
                          unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR, (12.0,) * 3),
                          frames.rotation_about([1.0, 1.0, 1.0], 0.7))}

    def check_record(self, record, expected):
        """The record is the one expected, key for key in order and byte for byte as JSON."""
        self.assertEqual(record, expected)
        self.assertEqual(list(record), list(expected))
        self.assertEqual(json.dumps(record), json.dumps(expected))
        self.assertEqual(json.loads(json.dumps(record)), record)            # JSON-ready

    def test_the_record_from_a_settings_axis_keeps_the_frame_s_other_fields(self):
        # the settings' axis wins over an axis frame's and over a grow_direction
        for name, frame in self.frames.items():
            with self.subTest(frame=name):
                given = copy.deepcopy(frame)
                for axis in ([1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [1.0, 2.0, 2.0]):
                    settings = bed.parse_settings(dict(FIBRE, axis=axis))
                    record = bed.frame_record(frame, settings, SMALL)
                    self.check_record(record, axis_record(frame, settings["axis"]))
                    self.assertEqual(frame, given)                        # the input is left alone
        # the bed between the foam member's feeders, along x, in main's frame
        case = fed(1)
        self.check_record(case["result"]["frame"], axis_record(case["frame"], [1.0, 0.0, 0.0]))

    def test_the_record_from_an_axis_frame_takes_its_axis(self):
        frame = self.frames["polar axis"]
        self.assertEqual((frame["kind"], frame["sense"], frame["tree_senses"]), ("axis", "polar", [1, -1]))
        record = bed.frame_record(frame, FIBRE_DEFAULTS, SMALL)
        self.check_record(record, axis_record(frame, frame["axis"]))
        self.assertEqual(bits(record["axis"]), bits([0.0, 0.6, 0.8]))
        # an axis frame's axis is normalised as the settings' is
        long = dict(frame, axis=[0.0, 3.0, 4.0])
        self.check_record(bed.frame_record(long, FIBRE_DEFAULTS, SMALL), axis_record(long, [0.0, 0.6, 0.8]))

    def test_the_record_from_a_frame_without_an_axis_takes_its_grow_direction(self):
        for name in ("plane", "unguided", "turned"):
            frame = self.frames[name]
            with self.subTest(frame=name):
                record = bed.frame_record(frame, FIBRE_DEFAULTS, SMALL)
                self.check_record(record, axis_record(frame, frame["grow_direction"]))
                self.assertEqual(bits(record["axis"]), bits(frame["grow_direction"]))   # unit, kept bitwise
        # a grow_direction that is not unit is normalised
        frame = dict(self.frames["unguided"], grow_direction=[0.0, 2.0, 0.0])
        self.check_record(bed.frame_record(frame, FIBRE_DEFAULTS, SMALL), axis_record(frame, [0.0, 1.0, 0.0]))

    def test_without_a_frame_the_record_is_built_about_the_domain_whatever_the_axis(self):
        domain = (np.array([-5.0, 0.0, 10.0]), np.array([35.0, 60.0, 90.0]))
        for axis in COORDINATE_AXES + ([0.0, 0.0, -1.0], [0.6, 0.8, 0.0], [1.0, 2.0, 2.0]):
            with self.subTest(axis=axis):
                settings = bed.parse_settings(dict(FIBRE, axis=axis))
                record = bed.frame_record(None, settings, domain)
                self.check_record(record, constructed_record(settings["axis"], domain))
                self.assertEqual(record["origin"], [15.0, 30.0, 50.0])
                # the domain as lists is the same domain
                self.assertEqual(bed.frame_record(None, settings, (domain[0].tolist(), domain[1].tolist())), record)

    def test_foam_s_record_takes_no_domain(self):
        # the keyword appended leaves foam's record as 3.7 made it
        frame = self.frames["unguided"]
        for domain in (None, SMALL):
            with self.subTest(domain=domain is not None):
                self.assertIs(bed.frame_record(frame, FOAM_3_7, domain), frame)
                self.assertIsNone(bed.frame_record(None, FOAM_3_7, domain))
                self.assertEqual(bed.frame_record(frame, dict(FOAM_3_7, stretch=2.0), domain),
                                 axis_record(frame, frame["grow_direction"]))
                with self.assertRaises(ValueError):
                    bed.frame_record(None, dict(FOAM_3_7, stretch=2.0), domain)

    def test_build_returns_the_record_of_each_source(self):
        # one small bed alone per source: settings axis with an axis frame,
        # the axis frame's axis, the grow_direction, and no frame
        sources = (("settings axis", dict(FIBRE, axis=[0.0, 0.0, 1.0]), self.frames["polar axis"], [0.0, 0.0, 1.0]),
                   ("axis frame", FIBRE_DEFAULTS, self.frames["polar axis"], [0.0, 0.6, 0.8]),
                   ("grow_direction", FIBRE_DEFAULTS, self.frames["unguided"], [0.0, 1.0, 0.0]),
                   ("no frame", FIBRE, None, [1.0, 0.0, 0.0]))
        for name, settings, frame, a in sources:
            with self.subTest(source=name):
                given = copy.deepcopy(frame)
                case = build_case(f"small, {name}", SMALL, settings, seed=1, frame=frame)
                record = case["result"]["frame"]
                self.assertEqual(record, bed.frame_record(frame, case["settings"], SMALL))
                self.check_record(record, axis_record(frame, a) if frame is not None else constructed_record(a, SMALL))
                self.assertEqual(bits(case["result"]["axis"]), bits(a))
                self.assertEqual(frame, given)
                self.assertGreater(len(case["result"]["chains"]), 0)

    def test_a_missing_axis_is_refused_before_any_draw(self):
        # no settings axis, and no frame or one that holds neither an axis nor a grow_direction
        unguided = self.frames["unguided"]
        missing = (("no frame", None), ("a frame without a grow_direction", {"kind": "none"}),
                   ("an axis frame without its axis", dict(unguided, kind="axis", axis=None)),
                   ("a frame of no dict", "axis"))
        for name, frame in missing:
            with self.subTest(frame=name):
                random.seed(19)
                np.random.seed(19)
                before = global_states()
                events = {"bed_seeds": 5}
                with recorded_generators() as created:
                    with self.assertRaises(ValueError):
                        bed.frame_record(frame, FIBRE_DEFAULTS, SMALL)
                    with self.assertRaises(ValueError) as caught:
                        bed.build(SMALL, FIBRE_DEFAULTS, d_min=D_MIN, seed=1, collision_margin=MARGIN, frame=frame,
                                  events=events)
                self.assertNotIsInstance(caught.exception, bed.SeedsNotPlaced)
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)
                self.assertEqual(events, {"bed_seeds": 5})

    def test_frames_align_and_transform_frame_accept_the_record(self):
        settings = bed.parse_settings(dict(FIBRE, axis=[1.0, 2.0, 2.0]))
        records = {"settings axis": bed.frame_record(self.frames["plane"], settings, SMALL),
                   "axis frame": bed.frame_record(self.frames["polar axis"], FIBRE_DEFAULTS, SMALL),
                   "grow_direction": bed.frame_record(self.frames["turned"], FIBRE_DEFAULTS, SMALL),
                   "no frame": bed.frame_record(None, settings, SMALL)}
        target = np.array([0.0, 0.0, 1.0])
        for name, record in records.items():
            with self.subTest(record=name):
                Q = frames.align(record, axis=target, spin=0.4)
                a = np.asarray(record["axis"])
                mapped = Q @ a
                # nematic: the axis goes onto the line of the target by its nearer end
                self.assertLessEqual(abs(abs(float(mapped[2])) - 1.0), 1e-12)
                self.assertLessEqual(float(np.max(np.abs(mapped[:2]))), 1e-12)
                moved = frames.transform_frame(record, Q, t=(1.0, -2.0, 3.0), scale=2.0)
                self.assertEqual(moved["axis"], [float(v) for v in Q @ a])
                for key in ("kind", "sense", "tree_senses", "normal", "rules", "frame_version"):
                    self.assertEqual(moved[key], record[key], key)
                again = frames.align(moved, axis=target)
                self.assertLessEqual(float(np.max(np.abs(again @ np.asarray(moved["axis"]) - mapped))), 1e-12)

    def test_describe_s_section_is_the_domain_s_cross_section_perpendicular_to_the_axis(self):
        # 40 x 60 x 80 with no frame: describe's box basis is the lab axes
        # for every a, so each plane's section is the domain's two sides
        # across a; on the walked bed and on its chord graph
        for axis in COORDINATE_AXES:
            case = sectioned(axis)
            j = axis.index(1.0)
            across = [side for k, side in enumerate(SECTION) if k != j]
            with self.subTest(axis=axis):
                self.check_record(case["result"]["frame"], constructed_record(axis, case["domain"]))
                for nodes in (case["result"]["nodes"], chords_as_polylines(case)):
                    report, planes = describe_bed(case, nodes)
                    self.assertEqual(report["transverse_spacing"]["area_source"], "argument")
                    self.assertEqual(planes["areas"], [across[0] * across[1]] * describe.TRANSVERSE_PLANES)

    def test_on_the_chord_graph_the_crossing_density_is_the_lanes_over_the_cross_section(self):
        # the density law along a coordinate axis, bed alone, run null and
        # no links: every plane meets each of the N lanes once
        for axis, lanes in zip(COORDINATE_AXES, (93, 61, 46)):
            case = sectioned(axis)
            j = axis.index(1.0)
            across = [side for k, side in enumerate(SECTION) if k != j]
            with self.subTest(axis=axis):
                ev = case["events"]
                self.assertEqual(ev["bed_seeds"], lanes)
                self.assertEqual((ev["bed_link_sites"], ev["bed_links"]), (0, 0))
                self.assertEqual((ev["bed_segments_dropped"], ev["bed_edges_dead_end"]), (0, 0))
                report, planes = describe_bed(case, chords_as_polylines(case))
                spacing = report["transverse_spacing"]
                self.assertEqual(planes["crossings"], [lanes] * describe.TRANSVERSE_PLANES)
                self.assertEqual(spacing["crossing_density"], lanes / (across[0] * across[1]))
                self.assertEqual(spacing["crossing_density_d"], lanes / (across[0] * across[1]) * D_MIN ** 2)
                self.assertEqual(report["cross_section"]["per_plane_density_d"],
                                 [lanes / (across[0] * across[1])] * describe.TRANSVERSE_PLANES)


# --------------------------------------------------------------------------
# Test 19: the registries


def static_closure(name, root):
    """The repository modules `name` imports, transitively, by every import statement in each module's source."""
    modules = {entry[:-3] for entry in os.listdir(root) if entry.endswith(".py")}
    seen, todo = set(), [name]
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        with open(os.path.join(root, module + ".py"), "rb") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found = {node.module.split(".")[0]}
            else:
                continue
            todo += sorted(found & modules)
    return seen


class RegistryTests(unittest.TestCase):
    def test_the_random_streams_are_3_7_s(self):
        self.assertEqual(main.RNG_STREAMS, MAIN_RNG_STREAMS_3_7)
        self.assertEqual(library.RNG_STREAMS, LIBRARY_RNG_STREAMS_3_7)
        self.assertEqual(bed.RNG_TAG, 7)

    def test_the_3_7_tuples_are_prefixes_and_what_follows_is_the_order_s(self):
        for name, now, released in (("bed.EVENT_KEYS", bed.EVENT_KEYS, BED_EVENT_KEYS_3_7),
                                    ("main.EVENT_KEYS", main.EVENT_KEYS, MAIN_EVENT_KEYS_3_7),
                                    ("bed.KINDS", bed.KINDS, KINDS_3_7),
                                    ("bed.SEED_OUTCOMES", bed.SEED_OUTCOMES, SEED_OUTCOMES_3_7),
                                    ("bed.SUBSTREAMS", tuple(bed.SUBSTREAMS.items()), tuple(SUBSTREAMS_3_7.items())),
                                    ("library.INDEX_COLUMNS", library.INDEX_COLUMNS, INDEX_COLUMNS_3_7)):
            with self.subTest(registry=name):
                self.assertIsInstance(now, tuple)
                self.assertEqual(now[:len(released)], released)
        # what 3.8 appends, as the order lists it
        self.assertEqual(bed.EVENT_KEYS[len(BED_EVENT_KEYS_3_7):], EVENT_KEYS_APPENDED)
        self.assertEqual(main.EVENT_KEYS[len(MAIN_EVENT_KEYS_3_7):], EVENT_KEYS_APPENDED)
        self.assertEqual(main.EVENT_KEYS[-len(bed.EVENT_KEYS):], bed.EVENT_KEYS)
        self.assertEqual(bed.KINDS, ("foam", "fibre"))
        self.assertEqual(bed.LAYOUTS, ("pair", "sheets"))
        self.assertEqual(bed.SEED_OUTCOMES, ("seed", "on_feeders", "overlapping", "outside"))
        self.assertEqual(list(bed.SUBSTREAMS.items()), list(SUBSTREAMS_3_8.items()))
        self.assertEqual(len(set(bed.EVENT_KEYS)), len(bed.EVENT_KEYS))
        self.assertEqual(len(set(main.EVENT_KEYS)), len(main.EVENT_KEYS))

    def test_the_setting_keys_are_3_7_s(self):
        self.assertEqual(bed.SETTING_KEYS, SETTING_KEYS_3_7)
        self.assertEqual(len(bed.SETTING_KEYS), 13)

    def test_main_s_import_closure_is_3_7_s(self):
        root = os.path.dirname(os.path.abspath(bed.__file__))
        # by every import statement, at any depth of each module
        self.assertEqual(static_closure("main", root), set(MAIN_CLOSURE_3_7))
        # by what a fresh interpreter has loaded from the repository once it imports main
        script = ("import json, os, sys; root = sys.argv[1]; sys.path.insert(0, root); import main; "
                  "files = {name: getattr(module, '__file__', None) for name, module in list(sys.modules.items())}; "
                  "print(json.dumps(sorted(name for name, path in files.items() "
                  "if path and os.path.dirname(os.path.abspath(path)) == root)))")
        loaded = subprocess.run([sys.executable, "-I", "-c", script, root], capture_output=True, text=True,
                                check=True, cwd=root)
        self.assertEqual(set(json.loads(loaded.stdout)), set(MAIN_CLOSURE_3_7))


# --------------------------------------------------------------------------
# Test 4: the lanes


class _LaneCapReached(Exception):
    """The brute-force loop's refusal: (lanes accepted, proposals considered)."""


def f0_basis(axis):
    """
    F0's plane basis written out for the tests: (j, e1, e2) with e_j the
    coordinate axis of the smallest |a_j| (the lowest j on ties),
    e1 = unit(e_j - (e_j . a) a) through bed._dist and e2 = a x e1, each
    component written out.
    """
    a = [float(v) for v in axis]
    j = min(range(3), key=lambda k: (abs(a[k]), k))
    e1 = np.array([(1.0 if k == j else 0.0) - a[j] * a[k] for k in range(3)])
    e1 = e1 / bed._dist(e1[None, :], np.zeros((1, 3)))[0]
    e2 = np.array([a[1] * e1[2] - a[2] * e1[1], a[2] * e1[0] - a[0] * e1[2], a[0] * e1[1] - a[1] * e1[0]])
    return j, e1, e2


def lane_rectangle(low, high, axis):
    """The bounding rectangle (u_lo, u_hi, v_lo, v_hi) of the box's eight corners projected onto (e1, e2) about o."""
    centre = [(lo + hi) / 2.0 for lo, hi in zip(low, high)]
    _, e1, e2 = f0_basis(axis)
    us, vs = [], []
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                c = (x - centre[0], y - centre[1], z - centre[2])
                us.append(c[0] * e1[0] + c[1] * e1[1] + c[2] * e1[2])
                vs.append(c[0] * e2[0] + c[1] * e2[1] + c[2] * e2[2])
    return min(us), max(us), min(vs), max(vs)


def slab_clip(p, axis, low, high):
    """
    The slab rule for one line p + t a in the box [low, high], in floats:
    (t0, t1, inside), a coordinate with a_j = 0 needing p_j strictly inside
    and the interval needing t1 > t0.
    """
    t0, t1, inside = -math.inf, math.inf, True
    for j in range(3):
        a_j, p_j = float(axis[j]), float(p[j])
        if a_j == 0.0:
            inside = inside and float(low[j]) < p_j < float(high[j])
            continue
        lo, hi = (float(low[j]) - p_j) / a_j, (float(high[j]) - p_j) / a_j
        if a_j < 0.0:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
    return t0, t1, inside and t1 > t0


def brute_force_lanes(rng, low, high, axis, *, spacing, count, block, cap):
    """
    F1 one proposal at a time, from the same blocks rng.random((block, 2))
    place_lanes draws: proposal p = o + x e1 + y e2 in floats, its line
    clipped by slab_clip ("outside", 3, tested first), then tested against
    the accepted points as a (1, 3) - (m, 3) array through bed._dist
    ("overlapping", 2), and accepted otherwise (0). Returns (points,
    extents, outcomes); raises _LaneCapReached at `cap` proposals.
    """
    low, high = [float(v) for v in low], [float(v) for v in high]
    centre = [(lo + hi) / 2.0 for lo, hi in zip(low, high)]
    _, e1, e2 = f0_basis(axis)
    u_lo, u_hi, v_lo, v_hi = lane_rectangle(low, high, axis)
    points, extents, outcomes = np.empty((count, 3)), np.empty((count, 2)), []
    accepted = considered = 0
    while accepted < count:
        if considered >= cap:
            raise _LaneCapReached(accepted, considered)
        R = rng.random((block, 2))
        for r0, r1 in R.tolist():
            if considered >= cap or accepted == count:
                break
            considered += 1
            x = u_lo + (u_hi - u_lo) * r0
            y = v_lo + (v_hi - v_lo) * r1
            p = [centre[k] + x * e1[k] + y * e2[k] for k in range(3)]
            t0, t1, inside = slab_clip(p, axis, low, high)
            if not inside:
                outcomes.append(3)
                continue
            if accepted and np.any(bed._dist(np.array([p]), points[:accepted]) < spacing):
                outcomes.append(2)
                continue
            outcomes.append(0)
            points[accepted] = p
            extents[accepted] = (t0, t1)
            accepted += 1
    return points, extents, np.array(outcomes, dtype=np.int8)


class LatticeStream:
    """
    A stand-in for the lane stream: random((n, 2)) hands out the next n rows
    of `rows`, round and round, so that the proposals land where the test
    puts them.
    """

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float).reshape(-1, 2)
        self.drawn = 0

    def random(self, shape):
        n, columns = shape
        if columns != 2:
            raise AssertionError(f"the lanes draw pairs, not {columns}")
        rows = self.rows[(self.drawn + np.arange(n)) % len(self.rows)]
        self.drawn += n
        return rows.copy()


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / bed._dist(vector[None, :], np.zeros((1, 3)))[0]


# The axes the lane tests place along: the coordinate axes (one reversed),
# one oblique axis in a coordinate plane and two oblique in no such plane.
LANE_AXES = (("x", np.array([1.0, 0.0, 0.0])), ("y", np.array([0.0, 1.0, 0.0])), ("z", np.array([0.0, 0.0, 1.0])),
             ("-y", np.array([0.0, -1.0, 0.0])), ("(1, 2, 0)", unit([1.0, 2.0, 0.0])),
             ("(1, 1, 1)", unit([1.0, 1.0, 1.0])), ("(3, -1, 2)", unit([3.0, -1.0, 2.0])))
LANE_COORDINATE_AXES = ("x", "y", "z", "-y")
FULLY_OBLIQUE_AXES = ("(1, 1, 1)", "(3, -1, 2)")


def lane_grid():
    """
    The lanes' grid: a cube of side 48 and a box of 40 x 60 x 80 (d_min 1),
    inset by d/2 as build does; every axis of LANE_AXES; spacing and density
    (5, 0.5) and (3.2, 0.6). 28 cases of 44-339 lanes.
    """
    grid = []
    for sides in ((48.0, 48.0, 48.0), (40.0, 60.0, 80.0)):
        low, high = np.full(3, 0.5), np.array(sides) - 0.5
        for name, axis in LANE_AXES:
            for spacing, density in ((5.0, 0.5), (3.2, 0.6)):
                settings = dict(FIBRE, spacing=spacing, density=density, axis=[float(v) for v in axis])
                count = bed.planned_seeds(sides, settings, D_MIN)
                grid.append({"sides": sides, "axis_name": name, "axis": axis, "spacing": spacing,
                             "density": density, "low": low, "high": high, "count": count})
    return grid


def lanes_of_build(case):
    """Each lane's knot ids in increasing t (the knots are numbered lane by lane)."""
    lane = case["result"]["knot_lane"]
    return [np.flatnonzero(lane == i) for i in range(len(case["result"]["seeds"]))]


def value_bits(value):
    return np.float64(value).tobytes()


# A lane point p = o + x e1 + y e2 lies in the plane through o perpendicular
# to a: (p - o) . a is 0 but for rounding, a few ulp of the box's side. The
# development measurement (the lanes' grid and beds alone of side 32 along
# x, (1, 2, 0), (1, 1, 1) and (3, -1, 2), seeds 11-20) found at most
# 3.0e-16 of the box's largest side; the bound is 1e-14 of it.
PLANE_ROUNDING = 1e-14


def check_lanes_of_a_build(test, case):
    """
    F1 and F2 on a build's lanes: the planned number of lane points, a hard
    core apart, in the plane through o perpendicular to a; each lane's
    extent the slab rule's clip of its line through the inset box, of
    positive length; each lane's first and last knot its boundary knots, at
    t0 and t1, every coordinate of a face the lane meets written exactly
    (at least one) and every other p_j + t a_j. Returns what was checked.
    """
    result, settings = case["result"], case["settings"]
    low, high = case["domain"]
    d = settings["diameter"] * case["d_min"]
    inset_low, inset_high = low + d / 2.0, high - d / 2.0
    h = settings["spacing"] * case["d_min"]
    a = result["axis"]
    P, E = result["seeds"], result["lane_extent"]
    n = len(P)
    test.assertEqual(n, case["events"]["bed_seeds"])
    test.assertEqual(n, bed.planned_seeds(high - low, settings, case["d_min"], axis=a))
    test.assertGreater(n, 1)
    i, j = np.triu_indices(n, 1)
    test.assertTrue(np.all(bed._dist(P[i], P[j]) >= h))
    test.assertTrue(np.all(E[:, 1] > E[:, 0]))
    centre = (inset_low + inset_high) / 2.0
    test.assertLessEqual(float(np.max(np.abs(bed._dot3(P - centre, np.array(a))))),
                         PLANE_ROUNDING * float(np.max(high - low)))
    for p, extent in zip(P, E):
        t0, t1, inside = slab_clip(p, a, inset_low, inset_high)
        test.assertTrue(inside)
        test.assertEqual((value_bits(t0), value_bits(t1)), (value_bits(extent[0]), value_bits(extent[1])))

    knots, t, boundary = result["knots"], result["knot_t"], result["knot_boundary"]
    test.assertTrue(np.all(np.diff(result["knot_lane"]) >= 0))
    on_faces = 0
    for lane, ids in enumerate(lanes_of_build(case)):
        test.assertGreaterEqual(ids.size, 2)
        test.assertEqual(boundary[ids].tolist(), [True] + [False] * (ids.size - 2) + [True])
        test.assertTrue(np.all(np.diff(t[ids]) > 0.0))
        p = P[lane]
        for end, at, faces in ((ids[0], E[lane, 0], (inset_low, inset_high)),
                               (ids[-1], E[lane, 1], (inset_high, inset_low))):
            test.assertEqual(value_bits(t[end]), value_bits(at))
            met = 0
            for c in range(3):
                if a[c] != 0.0:
                    face = faces[0][c] if a[c] > 0.0 else faces[1][c]
                    if (face - p[c]) / a[c] == at:
                        test.assertEqual(value_bits(knots[end, c]), value_bits(face))
                        met += 1
                        continue
                test.assertEqual(value_bits(knots[end, c]), value_bits(p[c] + at * a[c]))
            test.assertGreaterEqual(met, 1)
            on_faces += met
    return {"lanes": n, "face coordinates": on_faces}


class LaneTests(unittest.TestCase):
    """
    Test 4: bed.place_lanes, the vectorised two-dimensional RSA of F1,
    against brute_force_lanes, a loop over the proposals one at a time, at
    block sizes RSA_BLOCK, 7 and 64; on the lanes' grid (one random draw per
    case in the fast test, every case on seeds 1-5 in the slow one) and on
    proposals laid on a lattice; the proposal cap; and the lanes and
    boundary knots of built beds along coordinate and oblique axes.
    """

    BLOCKS = (bed.RSA_BLOCK, 7, 64)

    def check_case(self, case, seed):
        # the loop's proposals are the stream's pairs in order whatever its
        # block, so it runs once; each block's generator is then the loop's
        # advanced to whole blocks
        loop = np.random.default_rng([seed, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
        expected, expected_extents, expected_outcomes = brute_force_lanes(
            loop, case["low"], case["high"], case["axis"], spacing=case["spacing"], count=case["count"],
            block=bed.RSA_BLOCK, cap=math.inf)
        for block in self.BLOCKS:
            rng = np.random.default_rng([seed, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
            points, extents, outcomes = bed.place_lanes(rng, case["low"], case["high"], case["axis"],
                                                        spacing=case["spacing"], count=case["count"], block=block)
            self.assertEqual(points.tobytes(), expected.tobytes())
            self.assertEqual(extents.tobytes(), expected_extents.tobytes())
            np.testing.assert_array_equal(outcomes, expected_outcomes)
            self.assertEqual(outcomes.dtype, np.int8)
            fresh = np.random.default_rng([seed, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
            for _ in range(math.ceil(len(expected_outcomes) / block)):
                fresh.random((block, 2))
            self.assertEqual(rng.bit_generator.state, fresh.bit_generator.state)
            if block == bed.RSA_BLOCK:
                self.assertEqual(rng.bit_generator.state, loop.bit_generator.state)
        # the planned count, a hard core apart, every lane of positive length
        self.assertEqual(len(points), case["count"])
        self.assertGreater(len(points), 0)
        i, j = np.triu_indices(len(points), 1)
        self.assertTrue(np.all(bed._dist(points[i], points[j]) >= case["spacing"]))
        self.assertTrue(np.all(extents[:, 1] > extents[:, 0]))
        centre = (case["low"] + case["high"]) / 2.0
        self.assertLessEqual(float(np.max(np.abs(bed._dot3(points - centre, case["axis"])))),
                             PLANE_ROUNDING * max(case["sides"]))
        self.assertGreater(int(np.count_nonzero(outcomes == 2)), 0)
        # a coordinate axis's rectangle is the inset cross-section: nothing falls outside
        outside = int(np.count_nonzero(outcomes == 3))
        if case["axis_name"] in LANE_COORDINATE_AXES:
            self.assertEqual(outside, 0)
        if case["axis_name"] in FULLY_OBLIQUE_AXES:
            self.assertGreater(outside, 0)
        return outcomes

    def test_the_vectorised_lanes_equal_a_loop_over_the_proposals(self):
        grid = lane_grid()
        self.assertEqual(len(grid), 28)
        for k, case in enumerate(grid):
            seed = 1 + k % 5
            with self.subTest(sides=case["sides"], axis=case["axis_name"], spacing=case["spacing"], seed=seed):
                self.check_case(case, seed)

    @SLOW
    def test_the_vectorised_lanes_equal_a_loop_over_the_whole_grid(self):
        for case in lane_grid():
            for seed in range(1, 6):
                with self.subTest(sides=case["sides"], axis=case["axis_name"], spacing=case["spacing"], seed=seed):
                    self.check_case(case, seed)

    def test_proposals_on_a_lattice_are_resolved_one_at_a_time(self):
        # Along x in the inset box [0, 30] x [0, 40] x [0, 40] the basis is
        # (e_y, e_z) and the rectangle [-20, 20]^2 about o = (15, 20, 20), so
        # that the draws (i/8, k/8) put proposals exactly at (15, 5i, 5k).
        # Those with i = 0 or k = 0 lie on a face (outside); every lattice
        # point is proposed twice, in a fixed shuffled order.
        low, high = np.zeros(3), np.array([30.0, 40.0, 40.0])
        axis = np.array([1.0, 0.0, 0.0])
        self.assertEqual(lane_rectangle(low, high, axis), (-20.0, 20.0, -20.0, 20.0))
        lattice = [(i, k) for i in range(8) for k in range(8)]
        order = np.random.default_rng(2024).permutation(2 * len(lattice)) % len(lattice)
        sequence = [lattice[m] for m in order.tolist()]
        rows = [(i / 8.0, k / 8.0) for i, k in sequence]

        def expected(conflict):
            """The sequential answer read off the lattice indices: (points, outcomes) up to the last acceptance."""
            accepted, outcomes = [], []
            for i, k in sequence:
                if i == 0 or k == 0:
                    outcomes.append(3)
                elif any(conflict(i - a, k - b) for a, b in accepted):
                    outcomes.append(2)
                else:
                    outcomes.append(0)
                    accepted.append((i, k))
            last = max(m for m, outcome in enumerate(outcomes) if outcome == 0)
            return [(15.0, 5.0 * i, 5.0 * k) for i, k in accepted], outcomes[:last + 1]

        # h = 5 exactly: lattice neighbours stand exactly h apart and are not
        # strictly within it, so every interior point stands (the tie); h a
        # hair above 5: neighbours conflict, diagonals do not
        cases = (("h = 5", 5.0, lambda di, dk: di == 0 and dk == 0),
                 ("h = 5 (1 + 2^-40)", 5.0 * (1.0 + 2.0 ** -40), lambda di, dk: abs(di) + abs(dk) <= 1))
        for name, h, conflict in cases:
            points_expected, outcomes_expected = expected(conflict)
            count = len(points_expected)
            if name == "h = 5":
                self.assertEqual(count, 49)
            for block in self.BLOCKS:
                with self.subTest(case=name, block=block):
                    stream = LatticeStream(rows)
                    points, extents, outcomes = bed.place_lanes(stream, low, high, axis, spacing=h, count=count,
                                                                block=block)
                    loop = LatticeStream(rows)
                    looped, looped_extents, looped_outcomes = brute_force_lanes(
                        loop, low, high, axis, spacing=h, count=count, block=block, cap=math.inf)
                    self.assertEqual(points.tobytes(), looped.tobytes())
                    self.assertEqual(extents.tobytes(), looped_extents.tobytes())
                    np.testing.assert_array_equal(outcomes, looped_outcomes)
                    self.assertEqual(stream.drawn, loop.drawn)
                    self.assertEqual(points.tobytes(), np.array(points_expected).tobytes())
                    self.assertEqual(outcomes.tolist(), outcomes_expected)
                    self.assertEqual(extents.tobytes(), np.tile([-15.0, 15.0], (count, 1)).tobytes())
                    self.assertEqual(stream.drawn, block * math.ceil(len(outcomes_expected) / block))

    def test_a_target_above_jamming_is_refused_after_exactly_the_proposal_cap(self):
        # 200 lanes 5 apart in a cross-section of 33 x 33, far above jamming
        # (rho h^2 about 0.7); the cap ceil(64 x 33^2 / 25) = 2788 is no whole
        # block. Along (1, 1, 1) in a cube of side 33 the cap is the test's
        # own rectangle's; then an explicit cap.
        oblique = unit([1.0, 1.0, 1.0])
        u_lo, u_hi, v_lo, v_hi = lane_rectangle(np.zeros(3), np.full(3, 33.0), oblique)
        cases = (("x", np.array([20.0, 33.0, 33.0]), np.array([1.0, 0.0, 0.0]), None, 2788),
                 ("(1, 1, 1)", np.full(3, 33.0), oblique, None,
                  math.ceil(64.0 * (u_hi - u_lo) * (v_hi - v_lo) / 25.0)),
                 ("x, cap 1000", np.array([20.0, 33.0, 33.0]), np.array([1.0, 0.0, 0.0]), 1000, 1000))
        self.assertEqual(math.ceil(64.0 * 33.0 * 33.0 / 25.0), 2788)
        for name, high, axis, given, cap in cases:
            for block in self.BLOCKS:
                with self.subTest(case=name, block=block):
                    rng = np.random.default_rng([2, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
                    with self.assertRaises(bed.SeedsNotPlaced) as caught:
                        bed.place_lanes(rng, np.zeros(3), high, axis, spacing=5.0, count=200, block=block, cap=given)
                    found = re.search(r"only (\d+) of 200 lanes within (\d+) proposals", str(caught.exception))
                    self.assertIsNotNone(found)
                    self.assertEqual(int(found.group(2)), cap)
                    with self.assertRaises(_LaneCapReached) as reached:
                        brute_force_lanes(np.random.default_rng([2, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]]),
                                          np.zeros(3), high, axis, spacing=5.0, count=200, block=block, cap=cap)
                    self.assertEqual(reached.exception.args, (int(found.group(1)), cap))
                    fresh = np.random.default_rng([2, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
                    for _ in range(math.ceil(cap / block)):
                        fresh.random((block, 2))
                    self.assertEqual(rng.bit_generator.state, fresh.bit_generator.state)

    def test_a_count_of_zero_places_and_draws_nothing(self):
        for block in (bed.RSA_BLOCK, 7):
            rng = np.random.default_rng([1, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]])
            points, extents, outcomes = bed.place_lanes(rng, np.zeros(3), np.full(3, 30.0), [1.0, 0.0, 0.0],
                                                        spacing=5.0, count=0, block=block)
            self.assertEqual((points.shape, extents.shape, outcomes.shape), ((0, 3), (0, 2), (0,)))
            self.assertEqual(rng.bit_generator.state,
                             np.random.default_rng([1, bed.RNG_TAG, bed.SUBSTREAMS["lanes"]]).bit_generator.state)

    def test_the_plane_basis_is_f0_s(self):
        # e_j of the smallest |a_j|, the lowest j on ties; e1 and e2
        # orthonormal and perpendicular to a up to rounding; the coordinate
        # axes exactly
        axes = [axis for _, axis in LANE_AXES] + [unit([1.0, 1.0, 0.0]), unit([0.0, 1.0, 1.0]),
                                                   unit([-2.0, 0.5, 0.5]), unit([0.3, -0.3, 0.9])]
        expected_j = [1, 0, 0, 0, 2, 0, 1, 2, 0, 1, 0]
        for axis, j in zip(axes, expected_j):
            with self.subTest(axis=axis.tolist()):
                e1, e2 = bed._plane_basis(axis)
                found_j, f1, f2 = f0_basis(axis)
                self.assertEqual(found_j, j)
                self.assertEqual((e1.tobytes(), e2.tobytes()), (f1.tobytes(), f2.tobytes()))
                for u, v in ((e1, axis), (e2, axis), (e1, e2)):
                    self.assertLessEqual(abs(float(bed._dot3(u, v))), 4e-16)
                for u in (e1, e2):
                    self.assertLessEqual(abs(float(bed._dist(u[None, :], np.zeros((1, 3)))[0]) - 1.0), 4e-16)
        exact = {(1.0, 0.0, 0.0): ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), (0.0, 1.0, 0.0): ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
                 (0.0, 0.0, 1.0): ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), (0.0, -1.0, 0.0): ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])}
        for axis, (e1, e2) in exact.items():
            with self.subTest(axis=axis):
                found = bed._plane_basis(np.array(axis))
                self.assertEqual([found[0].tolist(), found[1].tolist()], [e1, e2])
                # the rectangle of a coordinate axis is the inset cross-section
                low, high = np.full(3, 0.5), np.array([39.5, 59.5, 79.5])
                centre = (low + high) / 2.0
                u = [k for k in range(3) if e1[k] != 0.0][0]
                v = [k for k in range(3) if e2[k] != 0.0][0]
                rectangle = bed._rectangle(low, high, centre, *found)
                half_u, half_v = sorted([low[u] - centre[u], high[u] - centre[u]]), sorted([low[v] - centre[v],
                                                                                            high[v] - centre[v]])
                self.assertEqual(rectangle, (half_u[0], half_u[1], half_v[0], half_v[1]))

    def test_built_lanes_stand_a_hard_core_apart_and_their_boundary_knots_lie_on_the_inset_faces(self):
        cases = [alone(1), fed(1)] + [alone(1, side=32.0, axis=axis) for axis in
                                       ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 2.0, 0.0],
                                        [1.0, 1.0, 1.0], [3.0, -1.0, 2.0])]
        for case in cases:
            with self.subTest(build=case["label"]):
                checked = check_lanes_of_a_build(self, case)
                self.assertGreaterEqual(checked["face coordinates"], 2 * checked["lanes"])


# --------------------------------------------------------------------------
# Test 10: pruning


def union_find_labels(n, edges):
    """Component labels of the graph of `edges` over n vertices (a root per component), by union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2).tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(v) for v in range(n)], dtype=np.int64)


def prune_reference(edges, alive, kept, *, anchored, is_tip, boundary, islands, counts, reason):
    """
    F7 written out, for the tests: peel every kept vertex that is not
    anchored and has at most one alive edge, round after round until none
    is left; with `islands`, remove every component of the alive edges that
    holds no tip, its kept knots with it; then drop every kept boundary knot
    left with no alive edge. Updates alive, kept, counts ("knots_dead_end",
    "edges_dead_end", "knots_island", "edges_island") and reason (why each
    vertex went: "peeled", "island" or "lone") in place.
    """
    n = len(kept)
    while True:
        degree = np.bincount(edges[alive].ravel(), minlength=n)
        peel = kept & ~anchored & (degree <= 1)
        if not peel.any():
            break
        kept[peel] = False
        reason[peel] = "peeled"
        counts["knots_dead_end"] += int(peel.sum())
        gone = alive & (peel[edges[:, 0]] | peel[edges[:, 1]])
        alive[gone] = False
        counts["edges_dead_end"] += int(gone.sum())
    if islands and alive.any():
        label = union_find_labels(n, edges[alive])
        with_tip = np.isin(label, label[is_tip])
        gone = alive & ~with_tip[edges[:, 0]]
        alive[gone] = False
        counts["edges_island"] += int(gone.sum())
        lost = np.zeros(n, dtype=bool)
        lost[edges[gone].ravel()] = True
        lost &= kept & ~is_tip
        kept[lost] = False
        reason[lost] = "island"
        counts["knots_island"] += int(lost.sum())
    degree = np.bincount(edges[alive].ravel(), minlength=n)
    lone = boundary & kept & (degree == 0)
    kept[lone] = False
    reason[lone] = "lone"
    counts["knots_dead_end"] += int(lone.sum())


def knots_on_feeders(case):
    """F3 for the tests: whether each knot lies strictly within r_f + d/2 + margin of a feeder column (bed._dist)."""
    knots = case["result"]["knots"]
    on = np.zeros(len(knots), dtype=bool)
    if case["feeders"] is None:
        return on
    fnodes, ftree = bed._feeders(case["feeders"])
    radius = case["settings"]["diameter"] * case["d_min"] / 2.0
    points, radii = bed._feeder_parts(fnodes, ftree, True, case["margin"], radius)[4]
    for k, knot in enumerate(knots):
        on[k] = bool(np.any(bed._dist(knot[None, :], points) < radii))
    return on


def expected_pruning(case, dropped=()):
    """
    F7 on a case's greedy edges by prune_reference: the first prune, then,
    for each dropped chain (the ids of its greedy edges), its edges removed
    and a re-prune. Returns (alive, kept, counts, reason) over the bed
    vertices, knots then tips.
    """
    result = case["result"]
    n_knots, n_tips = len(result["knots"]), len(result["tips"])
    n = n_knots + n_tips
    edges = result["greedy_edges"]
    is_tip = np.zeros(n, dtype=bool)
    is_tip[n_knots:] = True
    boundary = np.zeros(n, dtype=bool)
    boundary[:n_knots] = result["knot_boundary"]
    on = knots_on_feeders(case)
    kept = np.ones(n, dtype=bool)
    kept[:n_knots] = ~on
    reason = np.full(n, "", dtype=object)
    reason[:n_knots][on] = "on feeders"
    alive = np.ones(len(edges), dtype=bool)
    counts = {"knots_dead_end": 0, "edges_dead_end": 0, "knots_island": 0, "edges_island": 0, "edges_dropped": 0}
    options = dict(anchored=is_tip | boundary, is_tip=is_tip, boundary=boundary, islands=case["feeders"] is not None,
                   counts=counts, reason=reason)
    prune_reference(edges, alive, kept, **options)
    for gone in dropped:
        for e in gone:
            if alive[e]:
                alive[e] = False
                counts["edges_dropped"] += 1
        prune_reference(edges, alive, kept, **options)
    return alive, kept, counts, reason


def chain_edge_ids(case, vertices):
    """The ids in greedy_edges of a chain's edges."""
    index = {(int(a), int(b)): e for e, (a, b) in enumerate(case["result"]["greedy_edges"].tolist())}
    return [index[(min(a, b), max(a, b))] for a, b in zip(vertices[:-1], vertices[1:])]


@contextlib.contextmanager
def open_chain_refused(ends):
    """
    bed._collides answers True to every walk and to the chord of the open
    chain whose shared vertices are `ends`, so that it is dropped; every
    other test is answered as before.
    """
    real = bed._collides
    ends = set(ends)

    def refusing(obstacles, points, chain, shared, closed, self_check=True):
        if chain >= 0 and not closed and set(shared) == ends:
            return True
        return real(obstacles, points, chain, shared, closed, self_check)

    with mock.patch.object(bed, "_collides", side_effect=refusing):
        yield


_DROPPED_BUILDS = {}


def chain_dropped(base, chain):
    """The base case built again with the open chain of record `chain` refused, so that it is dropped."""
    key = (base["label"], chain)
    if key not in _DROPPED_BUILDS:
        vertices = base["result"]["chains"][chain]["vertices"]
        with open_chain_refused((vertices[0], vertices[-1])):
            case = build_case(base["label"] + f", chain {chain} dropped", base["domain"], base["settings"],
                              seed=base["seed"], feeders=base["feeders"], frame=base["frame"])
        case["dropped"] = [chain_edge_ids(case, vertices)]
        _DROPPED_BUILDS[key] = case
    return _DROPPED_BUILDS[key]


def first_chain(case, wanted):
    """The first open chain of a case's record whose two ends, as (kind, degree) pairs, are `wanted` in either order."""
    result = case["result"]
    n_knots = len(result["knots"])
    degree = np.bincount(result["edges"].ravel(), minlength=n_knots + len(result["tips"]))

    def kind(v):
        if v >= n_knots:
            return ("tip", int(degree[v]))
        return ("boundary" if result["knot_boundary"][v] else "knot", int(degree[v]))

    for c, chain in enumerate(result["chains"]):
        ends = {kind(chain["vertices"][0]), kind(chain["vertices"][-1])}
        if not chain["closed"] and ends == set(wanted):
            return c
    raise AssertionError(f"{case['label']} has no open chain between {wanted}")


def lane_runs(case):
    """
    For each lane, its maximal runs of consecutive lane edges the greedy
    pass accepted and no dropped chain took: lists of knot ids in
    increasing t.
    """
    gone = {e for edges in case.get("dropped", ()) for e in edges}
    accepted = {tuple(edge) for e, edge in enumerate(case["result"]["greedy_edges"].tolist()) if e not in gone}
    runs = []
    for ids in lanes_of_build(case):
        run = [int(ids[0])]
        for a, b in zip(ids[:-1].tolist(), ids[1:].tolist()):
            if (a, b) in accepted:
                run.append(b)
                continue
            if len(run) > 1:
                runs.append(run)
            run = [b]
        if len(run) > 1:
            runs.append(run)
    return runs


class _PruneCases:
    """The builds the pruning tests read, made once."""

    @staticmethod
    def without_drops():
        return [alone(1), alone(1, run=50.0), fed(1), fed(1, run=50.0), fed(1, link_spacing=None)]

    @staticmethod
    def with_drops():
        base_alone, base_fed = alone(1), fed(1)
        return [chain_dropped(base_alone, first_chain(base_alone, [("boundary", 1), ("knot", 3)])),
                chain_dropped(base_fed, first_chain(base_fed, [("knot", 3), ("tip", 1)])),
                chain_dropped(base_fed, first_chain(base_fed, [("knot", 3), ("tip", 2)]))]


_REFERENCE_3_7 = {}


def bed_3_7():
    """
    bed.py as released in 3.7 (3b100ec), loaded as a module of its own: the
    pin's unchanged copy when it exists, else git's; None when neither can
    be read.
    """
    if "module" not in _REFERENCE_3_7:
        path = os.path.join(ROOT, "tests", "fixtures", "reference_code_3_7", "bed.py")
        source = None
        if os.path.exists(path):
            with open(path, "rb") as handle:
                source = handle.read()
        else:
            try:
                source = subprocess.run(["git", "-C", ROOT, "show", "3b100ec:bed.py"], capture_output=True,
                                        check=True).stdout
            except (OSError, subprocess.CalledProcessError):
                source = None
        module = None
        if source:
            module = types.ModuleType("bed_3_7_reference")
            module.__file__ = path
            exec(compile(source, path, "exec"), module.__dict__)
        _REFERENCE_3_7["module"] = module
    return _REFERENCE_3_7["module"]


def check_retraction(test, case):
    """
    A cut lane retracts to its last junction. Within each run of lane edges
    the greedy pass accepted (and no dropped chain took), the edges kept are
    exactly those between the run's first and last anchor, an anchor being a
    kept knot that is a boundary knot or a junction (it holds a kept link or
    tip edge); a run with fewer than two anchors goes entirely. So a cut
    retracts the lane to the junction before it, and the part between a
    junction and the box stays. Returns the runs trimmed back to a junction
    and those gone entirely.
    """
    result = case["result"]
    n_knots = len(result["knots"])
    edges = result["edges"]
    lane = lane_edge_mask(case, edges)
    kept_lane = {tuple(edge) for edge in edges[lane].tolist()}
    junction = np.zeros(n_knots + len(result["tips"]), dtype=bool)
    junction[edges[~lane].ravel()] = True
    kept = result["kept"]
    anchor = kept & (result["knot_boundary"] | junction[:n_knots])
    degree = np.bincount(edges.ravel(), minlength=n_knots + len(result["tips"]))
    test.assertTrue(np.all(degree[:n_knots][kept & ~result["knot_boundary"]] >= 2))
    found = set()
    trimmed = collections.Counter()
    for run in lane_runs(case):
        at = [m for m, v in enumerate(run) if anchor[v]]
        pairs = list(zip(run[:-1], run[1:]))
        expected = set(pairs[at[0]:at[-1]]) if len(at) >= 2 else set()
        test.assertEqual({pair for pair in pairs if pair in kept_lane}, expected)
        found |= expected
        if len(at) < 2:
            trimmed["away"] += 1
            continue
        for end, outer in ((at[0], 0), (at[-1], len(run) - 1)):
            if end != outer and not result["knot_boundary"][run[end]]:
                trimmed["to a junction"] += 1
    test.assertEqual(found, kept_lane)
    return trimmed


def check_pruning(test, case):
    """
    The bed's pruning is F7's: its kept edges, kept knots and pruning
    counters are prune_reference's from its greedy edges, the knots on
    feeders and its dropped chains; no boundary knot was peeled. Returns the
    boundary knots removed, by reason.
    """
    result, events = case["result"], case["events"]
    n_knots = len(result["knots"])
    dropped = case.get("dropped", ())
    test.assertEqual(events["bed_segments_dropped"], len(dropped))
    alive, kept, counts, reason = expected_pruning(case, dropped)
    test.assertEqual(result["edges"].tobytes(), result["greedy_edges"][alive].tobytes())
    np.testing.assert_array_equal(result["kept"], kept[:n_knots])
    test.assertEqual({key: events["bed_" + key] for key in counts}, counts)
    test.assertEqual(events["bed_knots_on_feeders"], int(np.count_nonzero(knots_on_feeders(case))))
    removed = collections.Counter(why for why in reason[:n_knots][result["knot_boundary"]] if why)
    test.assertNotIn("peeled", removed)
    test.assertEqual(sum(removed.values()), int(np.count_nonzero(~result["kept"] & result["knot_boundary"])))
    return removed


class PruneTests(unittest.TestCase):
    """
    Test 10: F7's pruning. Every bed's kept knots, kept edges and pruning
    counters are compared with prune_reference, F7 written out from the
    greedy edges and the knots on feeders (re-evaluated by the test), the
    re-prunes after forced drops included; a lane cut by a feeder, a break
    or a refusal retracts exactly to its last junction; islands go only with
    feeders and boundary knots anchor none; and _prune without `anchored`
    is 3.7's on foam beds.
    """

    def test_boundary_knots_are_never_peeled_and_one_left_with_no_edge_is_dropped_and_counted(self):
        # a bed alone at F9: every lane is a path from face to face, so every
        # boundary knot has exactly one edge and stands; foam's peel (no
        # anchor) would take them all
        case = alone(1)
        result = case["result"]
        n_knots = len(result["knots"])
        degree = np.bincount(result["edges"].ravel(), minlength=n_knots)
        boundary = result["knot_boundary"]
        self.assertTrue(np.all(result["kept"][boundary]))
        self.assertTrue(np.all(degree[boundary] == 1))
        alive = np.ones(len(result["greedy_edges"]), dtype=bool)
        kept = np.ones(n_knots, dtype=bool)
        removed = bed._prune(n_knots, result["greedy_edges"], alive, np.zeros(n_knots, dtype=bool), kept, False)
        self.assertFalse(np.any(kept[boundary]))
        self.assertGreaterEqual(removed["seeds_dead_end"], int(boundary.sum()))
        # elsewhere, a boundary knot goes only on a feeder, with an island
        # or once it is left without an edge, and the last is counted a dead end
        removed = collections.Counter()
        for case in _PruneCases.without_drops() + _PruneCases.with_drops():
            with self.subTest(build=case["label"]):
                removed.update(check_pruning(self, case))
        self.assertGreater(removed["lone"], 0)
        self.assertGreater(removed["island"], 0)
        # the dropped chain of the bed alone ran from a boundary knot to a
        # junction: its interior knots are peeled and its boundary knot, left
        # with no edge, is dropped as a dead end
        case = _PruneCases.with_drops()[0]
        (chain,) = case["dropped"]
        self.assertEqual(case["events"]["bed_knots_dead_end"] - alone(1)["events"]["bed_knots_dead_end"], len(chain))
        self.assertEqual(case["events"]["bed_edges_dropped"], len(chain))

    def test_a_cut_lane_retracts_to_its_last_junction(self):
        # check_retraction on every build, the forced drops included
        trimmed = collections.Counter()
        for case in _PruneCases.without_drops() + _PruneCases.with_drops():
            with self.subTest(build=case["label"]):
                trimmed.update(check_retraction(self, case))
        self.assertGreater(trimmed["to a junction"], 0)
        self.assertGreater(trimmed["away"], 0)

    def test_islands_go_only_with_feeders_and_a_boundary_knot_anchors_none(self):
        # _prune on a constructed graph: lane A (knots 0-3, boundary 0 and 3)
        # holds no tip; lane B (4-7, boundary 4 and 7) reaches tip 8 by two edges
        edges = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [5, 8], [6, 8]])
        is_tip = np.zeros(9, dtype=bool)
        is_tip[8] = True
        boundary = np.zeros(9, dtype=bool)
        boundary[[0, 3, 4, 7]] = True
        for islands, anchored, alive_expected, kept_expected, counts_expected in (
                (True, is_tip | boundary, [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1], (0, 0, 3, 4)),
                (False, is_tip | boundary, [1] * 8, [1] * 9, (0, 0, 0, 0)),
                (True, None, [0, 0, 0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 1, 0, 1], (5, 6, 0, 0))):
            with self.subTest(islands=islands, anchored=anchored is not None):
                alive = np.ones(len(edges), dtype=bool)
                kept = np.ones(9, dtype=bool)
                counts = bed._prune(9, edges, alive, is_tip, kept, islands, anchored=anchored)
                self.assertEqual(alive.astype(int).tolist(), alive_expected)
                self.assertEqual(kept.astype(int).tolist(), kept_expected)
                self.assertEqual((counts["edges_dead_end"], counts["seeds_dead_end"], counts["edges_island"],
                                  counts["seeds_island"]), counts_expected)
        # beds alone keep components without tips; with feeders every kept
        # component holds a tip, and boundary knots went with islands
        for case in _PruneCases.without_drops():
            with self.subTest(build=case["label"]):
                result, events = case["result"], case["events"]
                n_knots, n_tips = len(result["knots"]), len(result["tips"])
                if case["feeders"] is None:
                    self.assertEqual((events["bed_knots_island"], events["bed_edges_island"]), (0, 0))
                    self.assertGreater(events["bed_components"], 1)
                    self.assertEqual(n_tips, 0)
                    continue
                label = union_find_labels(n_knots + n_tips, result["edges"])
                self.assertTrue(np.all(np.isin(label[result["edges"][:, 0]], label[n_knots:])))
                _, _, counts, reason = expected_pruning(case)
                self.assertGreater(int(np.count_nonzero(reason[:n_knots][result["knot_boundary"]] == "island")), 0)
                self.assertEqual(events["bed_knots_island"], counts["knots_island"])
                self.assertEqual(events["bed_edges_island"], counts["edges_island"])

    def test_prune_without_anchored_is_3_7_s_on_foam_beds(self):
        reference = bed_3_7()
        if reference is None:
            self.skipTest("bed.py of 3.7 can be read neither from the pin nor from git")
        self.assertNotIn("anchored", reference._prune.__code__.co_varnames)
        compared = collections.Counter()
        for case in (foam_alone(), foam_fed()):
            result = case["result"]
            n_seeds, n_tips = len(result["seeds"]), len(result["tips"])
            n = n_seeds + n_tips
            edges = result["greedy_edges"]
            is_tip = np.zeros(n, dtype=bool)
            is_tip[n_seeds:] = True
            # the greedy edges whole, and with a fifth of them cut at random
            masks = [np.ones(len(edges), dtype=bool)]
            masks += [np.random.default_rng([2024, k]).random(len(edges)) >= 0.2 for k in range(3)]
            for m, mask in enumerate(masks):
                for islands in (False, True):
                    with self.subTest(build=case["label"], mask=m, islands=islands):
                        outputs = []
                        for prune, keywords in ((bed._prune, {}), (bed._prune, {"anchored": is_tip}),
                                                (reference._prune, {})):
                            alive, kept = mask.copy(), np.ones(n, dtype=bool)
                            counts = prune(n, edges, alive, is_tip, kept, islands, **keywords)
                            outputs.append((alive.tolist(), kept.tolist(), counts))
                        self.assertEqual(outputs[0], outputs[2])
                        self.assertEqual(outputs[1], outputs[2])
                        compared["dead ends"] += outputs[2][2]["seeds_dead_end"]
                        compared["islands"] += outputs[2][2]["seeds_island"]
            # and the foam build's own pruning is 3.7's on its greedy edges
            alive, kept = np.ones(len(edges), dtype=bool), np.ones(n, dtype=bool)
            reference._prune(n, edges, alive, is_tip, kept, case["feeders"] is not None)
            self.assertEqual(case["events"]["bed_segments_dropped"], 0)
            self.assertEqual(result["edges"].tobytes(), edges[alive].tobytes())
            self.assertEqual(result["kept"].tolist(), kept[:n_seeds].tolist())
        self.assertGreater(compared["dead ends"], 0)
        self.assertGreater(compared["islands"], 0)

    def test_f7_holds_and_the_pruning_is_f7_s_including_re_prunes_after_a_dropped_chain(self):
        for case in _PruneCases.without_drops() + _PruneCases.with_drops():
            with self.subTest(build=case["label"]):
                result, events = case["result"], case["events"]
                dropped = case.get("dropped", ())
                check_pruning(self, case)
                # F7, E5 and E7 (check_laws holds F9 to a bed alone, which a forced drop breaks)
                self.assertEqual(events["bed_knots"], int(result["kept"].sum()) + events["bed_knots_on_feeders"]
                                 + events["bed_knots_dead_end"] + events["bed_knots_island"])
                self.assertEqual((events["bed_seeds_dead_end"], events["bed_seeds_island"]), (0, 0))
                self.assertEqual(events["bed_edges"], len(result["edges"]) + events["bed_edges_dead_end"]
                                 + events["bed_edges_island"] + events["bed_edges_dropped"])
                self.assertEqual(len(result["chains"]), events["bed_segments_walked"] + events["bed_segments_chord"]
                                 - events["bed_segments_removed"])
                if not dropped or case["feeders"] is not None:
                    check_laws(self, case)
        # the drops were re-pruned: the bed alone peeled the dropped chain's
        # knots, the first drop with feeders cut its tip's only edge and made
        # an island of the rest
        alone_drop, tip_drop, _ = _PruneCases.with_drops()
        self.assertGreater(alone_drop["events"]["bed_knots_dead_end"], alone(1)["events"]["bed_knots_dead_end"])
        self.assertGreater(tip_drop["events"]["bed_knots_island"], fed(1)["events"]["bed_knots_island"])
        self.assertGreater(tip_drop["events"]["bed_segments_removed"], 0)
        # a traced build hands over the dropped chain's edges as the slow test reads them
        traced = traced_dropped()
        self.assertEqual(traced["dropped"], [sorted(chain_edge_ids(traced, traced["refused"]))])
        self.assertEqual(traced["events"], tip_drop["events"])
        check_pruning(self, traced)

    @SLOW
    def test_the_pruning_is_f7_s_on_seeds_one_to_ten(self):
        # beds alone with breaks, and with feeders with and without breaks;
        # the traced builds hand over the edges of any chain dropped
        removed = collections.Counter()
        for seed in SEEDS:
            for case in (traced_alone(seed, side=SIDE, run=50.0), traced_fed(seed), traced_fed(seed, run=50.0)):
                with self.subTest(build=case["label"]):
                    removed.update(check_pruning(self, case))
                    check_retraction(self, case)
                    check_laws(self, case)
        self.assertGreater(removed["lone"], 0)
        self.assertGreater(removed["island"], 0)


# --------------------------------------------------------------------------
# Test 11: the draws


_TRACED_BUILDS = {}


def traced_build(label, domain, settings, *, seed, feeders=None, frame=None):
    """
    build_case with every Generator created recorded, every bridge_path
    call (its arguments, generator and result), _link_sites's arguments
    and return, _breaks's return, the alive edges before and after each _prune (and so the edges
    of every dropped chain) and the global states before and after
    captured. Made once per label.
    """
    if label in _TRACED_BUILDS:
        return _TRACED_BUILDS[label]
    calls, sites, gaps, prunes = [], [], [], []
    real_path, real_sites, real_breaks, real_prune = bed.bridge_path, bed._link_sites, bed._breaks, bed._prune

    def path(start, end, step, **kwargs):
        out = real_path(start, end, step, **kwargs)
        calls.append({"start": np.array(start, dtype=float), "end": np.array(end, dtype=float), "step": step,
                      "kwargs": dict(kwargs), "path": out.copy()})
        return out

    def link_sites(*args, **kwargs):
        out = real_sites(*args, **kwargs)
        sites.append({"on_feeders": np.array(args[6]), "found": out, "args": args, "kwargs": kwargs})
        return out

    def breaks(*args, **kwargs):
        out = real_breaks(*args, **kwargs)
        gaps.append(out)
        return out

    def prune(n_vertices, edges, alive, *args, **kwargs):
        before = alive.copy()
        out = real_prune(n_vertices, edges, alive, *args, **kwargs)
        prunes.append({"before": before, "after": alive.copy()})
        return out

    before = global_states()
    with recorded_generators() as created, mock.patch.object(bed, "bridge_path", path), \
            mock.patch.object(bed, "_link_sites", link_sites), mock.patch.object(bed, "_breaks", breaks), \
            mock.patch.object(bed, "_prune", prune):
        case = build_case(label, domain, settings, seed=seed, feeders=feeders, frame=frame)
    # the edges a dropped chain took: alive after one prune and gone before the next
    dropped = [np.flatnonzero(earlier["after"] & ~later["before"]).tolist() for earlier, later in zip(prunes, prunes[1:])]
    case.update(calls=calls, sites=sites, gaps=gaps, prunes=prunes, dropped=dropped, created=created, before=before,
                after=global_states())
    _TRACED_BUILDS[label] = case
    return case


def traced_alone(seed=1, side=32.0, **changes):
    """A traced bed alone in a cube of the given side (19 lanes at the defaults in side 32), along x."""
    label = f"traced alone, side {side:g}, seed {seed}" + "".join(f", {k} {v}" for k, v in sorted(changes.items()))
    return traced_build(label, (np.zeros(3), np.full(3, float(side))), dict(FIBRE, **changes), seed=seed)


def traced_fed(seed=1, **changes):
    """A traced bed between the feeders of the foam member of ratio 4 (fed's), along x."""
    call = foam_call(seed)
    label = f"traced fed, foam R 4, seed {seed}" + "".join(f", {k} {v}" for k, v in sorted(changes.items()))
    return traced_build(label, call["domain"], dict(FIBRE, **changes), seed=seed, feeders=call["feeders"],
                        frame=call["frame"])


FOAM_BED = {"kind": "foam"}


def foam_alone():
    """A traced foam bed alone at its defaults in a cube of side 40 (70 seeds)."""
    return traced_build("foam alone, side 40, seed 1", (np.zeros(3), np.full(3, 40.0)), FOAM_BED, seed=1)


def foam_fed():
    """A traced foam bed at its defaults between the foam member's feeders, in its domain and with its frame."""
    call = foam_call(1)
    return traced_build("foam fed, foam R 4, seed 1", call["domain"], FOAM_BED, seed=1, feeders=call["feeders"],
                        frame=call["frame"])


def traced_dropped():
    """
    traced_fed with the first open chain of fed(1) from a junction to a tip
    of one bed edge refused, so that it is dropped and its component goes
    as an island.
    """
    label = "traced fed, foam R 4, seed 1, a chain dropped"
    if label not in _TRACED_BUILDS:
        base = fed(1)
        vertices = base["result"]["chains"][first_chain(base, [("knot", 3), ("tip", 1)])]["vertices"]
        call = foam_call(1)
        with open_chain_refused((vertices[0], vertices[-1])):
            case = traced_build(label, call["domain"], FIBRE, seed=1, feeders=call["feeders"], frame=call["frame"])
        case["refused"] = vertices
    return _TRACED_BUILDS[label]


def generator_of(case, k):
    """The generator build created for sub-stream k, or None."""
    for given, generator in case["created"]:
        if list(given) == [case["seed"], bed.RNG_TAG, k]:
            return generator
    return None


def fresh_stream(case, name):
    return np.random.default_rng([case["seed"], bed.RNG_TAG, bed.SUBSTREAMS[name]])


def expected_streams(case):
    """The sub-streams the rules say a build creates: those that draw."""
    settings, events = case["settings"], case["events"]
    if settings["kind"] == "foam":
        streams = {1}
        if settings["jitter"] > 0.0 and events["bed_candidates"] > 0:
            streams.add(2)
    else:
        streams = {4, 5}
        if settings["run"] is not None:
            streams.add(7)
        if settings["link_spacing"] is not None and interior_count(case) > 0:
            streams.add(6)
        if settings["jitter"] > 0.0 and events["bed_candidates_tip"] > 0:
            streams.add(8)
    if events["bed_segments"] > 0:
        streams.add(3 if settings["kind"] == "foam" else 9)
    return streams


def recreated_knots(case, fresh):
    """
    F2 from a fresh knot stream: phi = random(N), each lane's interior t,
    xi = standard_normal((M, 2)) and the interior knots, offsets cut at w.
    Returns (interior t, interior knots).
    """
    result, settings, d_min = case["result"], case["settings"], case["d_min"]
    P, E, axis = result["seeds"], result["lane_extent"], np.array(result["axis"])
    knot = settings["knot"] * d_min
    phi = fresh.random(len(P))
    lanes, ts = [], []
    for i, ((t0, t1), offset) in enumerate(zip(E.tolist(), phi.tolist())):
        for k in range(int(math.ceil((t1 - t0) / knot)) + 2):
            t = t0 + (offset + k) * knot
            if t0 + knot / 2.0 <= t <= t1 - knot / 2.0:
                lanes.append(i)
                ts.append(t)
    lanes, ts = np.array(lanes, dtype=np.int64), np.array(ts)
    xi = fresh.standard_normal((len(ts), 2))
    _, e1, e2 = f0_basis(axis)
    h = settings["spacing"] * d_min
    T = settings["diameter"] * d_min + case["margin"]
    w = (h - T) / 2.0
    offset = settings["sway"] * d_min * (xi[:, 0:1] * e1 + xi[:, 1:2] * e2)
    length = bed._dist(offset, np.zeros((1, 3)))
    wide = length > w
    offset[wide] = offset[wide] * (w / length[wide])[:, None]
    return ts, P[lanes] + ts[:, None] * axis + offset


def recreated_sites(case, fresh):
    """F5's draws from a fresh link stream: u = random(M), the sites u < q, then v = random(sites)."""
    settings, d_min = case["settings"], case["d_min"]
    interior = np.flatnonzero(~case["result"]["knot_boundary"])
    u = fresh.random(interior.size)
    q = settings["knot"] * d_min / (2.0 * settings["link_spacing"] * d_min)
    sites = interior[u < q]
    fresh.random(sites.size)
    return sites


def recreated_breaks(case, fresh):
    """F4's draws from a fresh break stream: per lane, gaps exponential(run d_min) one call at a time from t0 past t1."""
    scale = case["settings"]["run"] * case["d_min"]
    gaps = []
    for t0, t1 in case["result"]["lane_extent"].tolist():
        at, here = t0, []
        while True:
            at += float(fresh.exponential(scale))
            if at > t1:
                break
            here.append(at)
        gaps.append(here)
    return gaps


def broken_edges(case, gaps):
    """The lane edges (lower, upper knot id) whose interval [t_k, t_k+1) holds a break, each once."""
    t = case["result"]["knot_t"]
    edges = set()
    for ids, here in zip(lanes_of_build(case), gaps):
        for at in here:
            k = int(np.searchsorted(t[ids], at, side="right")) - 1
            if 0 <= k < ids.size - 1:
                edges.add((int(ids[k]), int(ids[k + 1])))
    return edges


def walk_order(case, record, degree):
    """A chain's vertices in the order it was walked: as written, but a lasso from its junction J."""
    vertices = list(record["vertices"])
    if not record["closed"]:
        return vertices
    n_knots = len(case["result"]["knots"])
    joints = [v for v in vertices[:-1] if v >= n_knots or degree[v] != 2]
    if not joints:
        return vertices                                     # a ring, written as walked
    at = len(vertices) - 1 - len(vertices) // 2              # where J sits in a lasso written from its middle
    if vertices[at] != joints[0]:
        raise AssertionError(f"a lasso {vertices} does not hold its junction {joints[0]} where it is written")
    return vertices[at:] + vertices[1:at + 1]


class DrawTests(unittest.TestCase):
    """
    Test 11: each sub-stream draws exactly what F1-F8 state, shown by
    recreating it from a fresh default_rng([seed, 7, k]) and comparing the
    generators' states (and what the draws make); the parts of a bed each
    setting leaves alone; nothing global drawn; foam and fibre on their own
    sub-streams.
    """

    @staticmethod
    def fibre_cases():
        return [traced_alone(), traced_alone(run=50.0), traced_alone(link_spacing=None, jitter=0.0),
                traced_fed(), traced_fed(run=50.0), traced_fed(jitter=0.0), traced_dropped()]

    def test_each_sub_stream_draws_exactly_what_its_rule_states(self):
        seen = set()
        for case in self.fibre_cases():
            with self.subTest(build=case["label"]):
                seen |= self.check_draws(case)
        self.assertEqual(seen, {4, 5, 6, 7, 8, 9})

    @SLOW
    def test_each_sub_stream_draws_exactly_what_its_rule_states_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for case in (traced_alone(seed, side=SIDE), traced_fed(seed), traced_fed(seed, run=50.0)):
                with self.subTest(build=case["label"]):
                    self.check_draws(case)

    def check_draws(self, case):
        """Every sub-stream of one build recreated from a fresh generator; returns the sub-streams it created."""
        result, events, settings = case["result"], case["events"], case["settings"]
        created = seeds_of(case["created"])
        self.assertEqual(len(created), len({tuple(seed) for seed in created}))
        self.assertEqual({seed[2] for seed in created}, expected_streams(case))
        self.assertTrue(all(seed[:2] == [case["seed"], bed.RNG_TAG] for seed in created))
        low, high = case["domain"]
        d = settings["diameter"] * case["d_min"]
        # k = 4: whole blocks rng.random((RSA_BLOCK, 2)), the lanes of the loop
        fresh = fresh_stream(case, "lanes")
        for _ in range(math.ceil(events["bed_proposals"] / bed.RSA_BLOCK)):
            fresh.random((bed.RSA_BLOCK, 2))
        self.assertEqual(generator_of(case, 4).bit_generator.state, fresh.bit_generator.state)
        points, extents, outcomes = brute_force_lanes(
            fresh_stream(case, "lanes"), low + d / 2.0, high - d / 2.0, result["axis"],
            spacing=settings["spacing"] * case["d_min"], count=events["bed_seeds"], block=bed.RSA_BLOCK,
            cap=math.inf)
        self.assertEqual(points.tobytes(), result["seeds"].tobytes())
        self.assertEqual(extents.tobytes(), result["lane_extent"].tobytes())
        self.assertEqual((len(outcomes), int(np.count_nonzero(outcomes == 2)),
                          int(np.count_nonzero(outcomes == 3))),
                         (events["bed_proposals"], events["bed_proposals_overlapping"],
                          events["bed_proposals_outside"]))
        # k = 5: random(N), then standard_normal((M, 2)); they make the knots
        fresh = fresh_stream(case, "knots")
        t, knots = recreated_knots(case, fresh)
        self.assertEqual(generator_of(case, 5).bit_generator.state, fresh.bit_generator.state)
        interior = ~result["knot_boundary"]
        self.assertEqual(t.tobytes(), result["knot_t"][interior].tobytes())
        self.assertEqual(knots.tobytes(), result["knots"][interior].tobytes())
        # k = 6: random(M), then random(sites)
        if 6 in expected_streams(case):
            fresh = fresh_stream(case, "links")
            sites = recreated_sites(case, fresh)
            self.assertEqual(generator_of(case, 6).bit_generator.state, fresh.bit_generator.state)
            (captured,) = case["sites"]
            self.assertEqual(sites.tobytes(), captured["found"]["sites"].tobytes())
            self.assertEqual(sites.size, events["bed_link_sites"])
        else:
            self.assertEqual((case["sites"], events["bed_link_sites"]), ([], 0))
        # k = 7: exponential(run d_min) one call at a time, lane by lane
        if settings["run"] is not None:
            fresh = fresh_stream(case, "breaks")
            gaps = recreated_breaks(case, fresh)
            self.assertEqual(generator_of(case, 7).bit_generator.state, fresh.bit_generator.state)
            self.assertEqual(case["gaps"], [gaps])
            on = knots_on_feeders(case)
            broken = broken_edges(case, gaps)
            standing = [edge for edge in broken if not on[edge[0]] and not on[edge[1]]]
            self.assertEqual(events["bed_breaks"], len(standing))
            self.assertGreater(events["bed_breaks"], 0)
            # the broken lane edges are not candidates; alone (F9) every other lane edge is accepted
            greedy = result["greedy_edges"]
            accepted = {tuple(edge) for edge in greedy[lane_edge_mask(case, greedy)].tolist()}
            self.assertFalse(accepted & broken)
            if case["feeders"] is None:
                lane_pairs = {(int(a), int(b)) for ids in lanes_of_build(case) for a, b in zip(ids[:-1], ids[1:])}
                self.assertEqual(accepted, lane_pairs - broken)
        else:
            self.assertEqual((case["gaps"], events["bed_breaks"]), ([], 0))
        # k = 8: random(the tip pairs)
        if 8 in expected_streams(case):
            fresh = fresh_stream(case, "tip_order")
            fresh.random(events["bed_candidates_tip"])
            self.assertEqual(generator_of(case, 8).bit_generator.state, fresh.bit_generator.state)
        # k = 9: bridge_path's draws, by chain, then attempt, then edge
        self.check_walks(case)
        return {seed[2] for seed in created}

    def check_walks(self, case):
        """
        The bridge_path calls that draw run on the k = 9 generator; replayed
        in order on a fresh stream they give the same paths and leave the
        same state, and the stored chords draw nothing. Without a dropped
        chain they are, in order, each chain's edges once per attempt, 1 to
        1 + attempts times, the extra attempts bed_walk_redraws.
        """
        result, events = case["result"], case["events"]
        walker = generator_of(case, 9)
        drawing = [call for call in case["calls"] if call["kwargs"].get("rng") is not None]
        if walker is None:
            self.assertEqual((drawing, events["bed_segments"]), ([], 0))
            return
        for call in case["calls"]:
            if call["kwargs"].get("rng") is None:
                self.assertEqual(call["kwargs"], {})                # a stored chord
        self.assertTrue(all(call["kwargs"]["rng"] is walker for call in drawing))
        fresh = fresh_stream(case, "fibre_walks")
        for call in drawing:
            kwargs = dict(call["kwargs"], rng=fresh)
            replayed = tortuosity.bridge_path(call["start"], call["end"], call["step"], **kwargs)
            self.assertEqual(replayed.tobytes(), call["path"].tobytes())
        self.assertEqual(walker.bit_generator.state, fresh.bit_generator.state)
        if events["bed_segments_dropped"]:
            return
        points = vertex_points(case)
        degree = np.bincount(result["edges"].ravel(), minlength=len(points))
        at, redraws = 0, 0
        for record in result["chains"]:
            chain = walk_order(case, record, degree)
            sequence = [(points[a].tobytes(), points[b].tobytes()) for a, b in zip(chain[:-1], chain[1:])]
            attempts = 0
            while sequence == [(call["start"].tobytes(), call["end"].tobytes())
                               for call in drawing[at:at + len(sequence)]]:
                at += len(sequence)
                attempts += 1
            self.assertGreaterEqual(attempts, 1)
            self.assertLessEqual(attempts, 1 + case["attempts"])
            redraws += attempts - 1
        self.assertEqual(at, len(drawing))
        self.assertEqual(redraws, events["bed_walk_redraws"])

    def assert_same_arrays(self, a, b, keys):
        for key in keys:
            self.assertEqual(np.asarray(a[key]).shape, np.asarray(b[key]).shape, key)
            self.assertEqual(np.asarray(a[key]).tobytes(), np.asarray(b[key]).tobytes(), key)

    LANE_KEYS = ("seeds", "lane_extent")
    KNOT_KEYS = ("knots", "knot_lane", "knot_t", "knot_boundary")
    # the counters of F1-F3, which every draw before the breaks makes
    BEFORE_BREAKS = ("bed_proposals", "bed_proposals_on_feeders", "bed_proposals_overlapping",
                     "bed_proposals_outside", "bed_seeds", "bed_tips", "bed_tips_inside_junction", "bed_knots",
                     "bed_knots_boundary", "bed_knots_on_feeders", "bed_lane_edges_on_feeders")

    def test_changing_link_spacing_leaves_the_lanes_and_knots_bitwise(self):
        for reference, others in ((alone(1, side=32.0), [alone(1, side=32.0, link_spacing=value)
                                                         for value in (None, 10.0, 40.0)]),
                                  (fed(1), [fed(1, link_spacing=None)])):
            for case in others:
                with self.subTest(build=case["label"]):
                    self.assertNotEqual(case["events"]["bed_link_sites"], reference["events"]["bed_link_sites"])
                    self.assert_same_arrays(case["result"], reference["result"], self.LANE_KEYS + self.KNOT_KEYS)
                    for key in self.BEFORE_BREAKS:
                        self.assertEqual(case["events"][key], reference["events"][key], key)

    def test_changing_sway_leaves_the_lanes_and_the_sites(self):
        # the knots of every sway are those of the same phi and xi (F2), and
        # the sites and the sites without a neighbour do not move; which
        # neighbours are feasible moves with the knots
        reference = traced_alone()
        for sway in (0.0, 0.5, 2.0):
            case = traced_alone(sway=sway)
            with self.subTest(sway=sway):
                self.assert_same_arrays(case["result"], reference["result"],
                                        self.LANE_KEYS + ("knot_lane", "knot_t", "knot_boundary"))
                boundary = reference["result"]["knot_boundary"]
                self.assertEqual(case["result"]["knots"][boundary].tobytes(),
                                 reference["result"]["knots"][boundary].tobytes())
                _, knots = recreated_knots(case, fresh_stream(case, "knots"))
                self.assertEqual(knots.tobytes(), case["result"]["knots"][~boundary].tobytes())
                if sway == 0.0:
                    lane = case["result"]["knot_lane"][~boundary]
                    axis = np.array(case["result"]["axis"])
                    on_line = case["result"]["seeds"][lane] + case["result"]["knot_t"][~boundary, None] * axis
                    self.assertEqual(knots.tobytes(), on_line.tobytes())
                (found,), (expected,) = [c["found"] for c in case["sites"]], [c["found"] for c in reference["sites"]]
                self.assertEqual(found["sites"].tobytes(), expected["sites"].tobytes())
                self.assertEqual(found["no_partner"], expected["no_partner"])
                for key in self.BEFORE_BREAKS + ("bed_link_sites", "bed_link_sites_no_partner"):
                    self.assertEqual(case["events"][key], reference["events"][key], key)

    def test_changing_run_leaves_everything_before_the_breaks(self):
        # F0-F3 (axis, lanes, knots, knots on feeders, tips) and the link
        # sites, drawn on their own stream, are those of unbroken lanes; the
        # partners see the breaks only through F4's lane-edge candidates
        for reference, others in ((traced_alone(), [traced_alone(run=50.0)]),
                                  (traced_fed(), [traced_fed(run=50.0), traced_fed(run=100.0)])):
            for case in others:
                with self.subTest(build=case["label"]):
                    self.assertGreater(case["events"]["bed_breaks"], 0)
                    self.assertEqual(case["result"]["axis"], reference["result"]["axis"])
                    self.assert_same_arrays(case["result"], reference["result"],
                                            self.LANE_KEYS + self.KNOT_KEYS + ("tips",))
                    for key in self.BEFORE_BREAKS:
                        self.assertEqual(case["events"][key], reference["events"][key], key)
                    for k in (4, 5):
                        self.assertEqual(generator_of(case, k).bit_generator.state,
                                         generator_of(reference, k).bit_generator.state)
                    (found,), (expected,) = case["sites"], reference["sites"]
                    np.testing.assert_array_equal(found["on_feeders"], expected["on_feeders"])
                    np.testing.assert_array_equal(found["on_feeders"], knots_on_feeders(case))
                    self.assertEqual(found["found"]["sites"].tobytes(), expected["found"]["sites"].tobytes())
                    self.assertEqual(found["found"]["no_partner"], expected["found"]["no_partner"])
                    self.assertLess(len(found["args"][8]), len(expected["args"][8]))
                    # with the unbroken lanes' candidates, the same draws pick the unbroken bed's links
                    args = list(found["args"])
                    args[0] = bed._stream(case["seed"], bed.SUBSTREAMS["links"])
                    args[8] = expected["args"][8]
                    again = bed._link_sites(*args, **found["kwargs"])
                    self.assertEqual(again["links"].tobytes(), expected["found"]["links"].tobytes())
                    self.assertEqual((again["angle"], again["on_feeders"]),
                                     (expected["found"]["angle"], expected["found"]["on_feeders"]))

    def test_nothing_global_is_drawn(self):
        for case in self.fibre_cases():
            with self.subTest(build=case["label"]):
                assert_same_states(self, case["after"], case["before"])
                self.assertTrue(all(len(seed) == 3 for seed in seeds_of(case["created"])))
        # and from seeded global generators
        random.seed(17)
        np.random.seed(17)
        before = global_states()
        build_case("seeded", (np.zeros(3), np.full(3, 32.0)), FIBRE, seed=3)
        assert_same_states(self, global_states(), before)

    def test_foam_creates_no_sub_stream_from_4_on_and_fibre_none_of_1_to_3(self):
        for case in (foam_alone(), foam_fed()):
            with self.subTest(build=case["label"]):
                self.assertEqual(seeds_of(case["created"]), [[1, bed.RNG_TAG, k] for k in (1, 2, 3)])
                assert_same_states(self, case["after"], case["before"])
        for case in self.fibre_cases():
            with self.subTest(build=case["label"]):
                created = sorted(seed[2] for seed in seeds_of(case["created"]))
                self.assertTrue(set(created) <= {4, 5, 6, 7, 8, 9})
                self.assertIn(4, created)


# --------------------------------------------------------------------------
# Test 13: determinism and scale


def scaled_case(base, s):
    """The base case built again with d_min, the margin, the domain, the feeders and the frame's origin times s."""
    low, high = base["domain"]
    feeders = None
    if base["feeders"] is not None:
        feeders = (np.asarray(base["feeders"][0], dtype=float) * s, base["feeders"][1])
    frame = None
    if base["frame"] is not None:
        frame = copy.deepcopy(base["frame"])
        frame["origin"] = [s * v for v in frame["origin"]]
    return build_case(base["label"] + f", scaled {s:g}", (s * low, s * high), base["settings"], seed=base["seed"],
                      feeders=feeders, frame=frame, d_min=s * base["d_min"], margin=s * base["margin"])


class FibreScaleTests(unittest.TestCase):
    """
    Test 13. A fibre bed is a function of its inputs, and scaling d_min,
    the margin and the domain (with the feeders and the frame's origin) by
    s scales the bed by s with the same counters. Powers of two keep every
    operation exact, the walk's included (bridge_path's step / d and the
    feeders scaled with the rest), so the order's 1e-12 relative bound is
    met with no absolute floor. The development measurement, beds alone in
    cubes of side 48 and beds between the foam member's feeders, seeds
    11-20, s = 0.5 and 2: the largest difference from s x nodes was 0.0
    (every node bitwise) and the counters were equal on every seed.
    """

    KEYS = ("nodes", "tree", "seeds", "greedy_edges", "kept", "tips", "tip_edges", "edges", "lane_extent", "knots",
            "knot_lane", "knot_t", "knot_boundary", "links")

    def check_same(self, a, b):
        self.assertEqual(sorted(a["result"]), sorted(b["result"]))
        for key in self.KEYS:
            self.assertEqual(np.asarray(a["result"][key]).shape, np.asarray(b["result"][key]).shape, key)
            self.assertEqual(np.asarray(a["result"][key]).tobytes(), np.asarray(b["result"][key]).tobytes(), key)
            self.assertEqual(np.asarray(a["result"][key]).dtype, np.asarray(b["result"][key]).dtype, key)
        for key in ("chains", "frame", "axis"):
            self.assertEqual(a["result"][key], b["result"][key], key)
        self.assertEqual(a["events"], b["events"])

    def test_the_same_inputs_give_identical_nodes_and_events(self):
        for base in (alone(1), fed(1)):
            with self.subTest(build=base["label"]):
                again = build_case(base["label"], base["domain"], base["settings"], seed=base["seed"],
                                   feeders=base["feeders"], frame=base["frame"])
                self.check_same(again, base)
                self.assertGreater(base["events"]["bed_segments_walked"], 0)

    def check_scaled(self, base, s):
        scaled = scaled_case(base, s)
        reference, result = base["result"], scaled["result"]
        self.assertGreater(base["events"]["bed_segments_walked"], 0)
        self.assertEqual(result["nodes"].shape, reference["nodes"].shape)
        np.testing.assert_allclose(result["nodes"], s * reference["nodes"], rtol=1e-12, atol=0.0)
        for key in ("seeds", "lane_extent", "knots", "knot_t"):
            np.testing.assert_allclose(result[key], s * reference[key], rtol=1e-12, atol=0.0, err_msg=key)
        for key in ("tree", "greedy_edges", "kept", "tips", "tip_edges", "edges", "knot_lane", "knot_boundary",
                    "links"):
            self.assertEqual(result[key].tobytes(), reference[key].tobytes(), key)
        self.assertEqual(result["axis"], reference["axis"])
        self.assertEqual([dict(chain) for chain in result["chains"]], reference["chains"])
        self.assertEqual(result["frame"]["origin"], [s * v for v in reference["frame"]["origin"]])
        self.assertEqual(scaled["events"], base["events"])

    def test_a_fibre_bed_scales_with_d_min_the_margin_and_the_domain(self):
        for base in (alone(1, side=32.0), fed(1)):
            for s in (0.5, 2.0):
                with self.subTest(build=base["label"], scale=s):
                    self.check_scaled(base, s)

    @SLOW
    def test_a_fibre_bed_scales_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for base in (alone(seed), fed(seed)):
                for s in (0.5, 2.0):
                    with self.subTest(build=base["label"], scale=s):
                        self.check_scaled(base, s)


# --------------------------------------------------------------------------
# Tests 5-7: the density law, the orientation and the links


# The cube of the beds the density law and the links are read on, side 32
# d_min: 15 lanes at spacing 5 and density 0.4, 56 at spacing 3.2 and
# density 0.6, 19 at the defaults.
LAW_SIDE = 32.0
LAW_GRID = ((5.0, 0.4), (5.0, 0.6), (3.2, 0.4), (3.2, 0.6))

# The walked bed's excess crossings over the chord graph's, from the
# development measurement in DensityLawTests' docstring (seeds 11-20): with
# links the largest relative excess measured, 0.0294, doubled and rounded up
# to one significant figure; without links the excess measured, 0 crossings
# on every bed, never widened.
WALKED_EXCESS_LINKS = 0.06
WALKED_EXCESS_NO_LINKS = 0

# The chord law's settings, (name, changes, extents): sigma = w / 4 at the
# default spacing (beta 0.13), sigma = w / 4 at a wide spacing (w 6, beta
# 0.85) and sigma = w at the default spacing (beta 0.53).
CHORD_LAW_CASES = (
    ("sigma w/4, beta 0.13", {"sway": 0.375}, (LAW_SIDE,) * 3),
    ("sigma w/4, beta 0.85", {"spacing": 14.0, "knot": 2.5, "sway": 1.5}, (40.0, 100.0, 100.0)),
    ("sigma w, beta 0.53", {"sway": 1.5}, (LAW_SIDE,) * 3),
)
CHORD_LAW_REPLICATES = 20
CHORD_LAW_PAIRS = 50000

# The geometric law's beds: lanes 200 d_min long (about 49 interior knots),
# and the window of the gap test fixed for each q so that every expected
# count is at least 5.
GAP_EXTENTS = (200.0, 40.0, 40.0)
GAP_WINDOWS = {8.0: 10, 20.0: 15}          # link_spacing: window (q 0.25 and 0.1 at knot 4)

# The settings of the cross-connection checks: every link within 60 degrees
# of transverse even beside a lane end, and walks that keep to their chords.
CROSS_SETTINGS = {"sway": 0.0, "knot": 2.5, "persistence": 1e6}


_LAW_BUILDS = {}


def _law_key(changes):
    return tuple(sorted((name, str(value)) for name, value in changes.items()))


def law_build(seed=1, extents=(LAW_SIDE,) * 3, fed_ratio=None, **changes):
    """
    A bed at the defaults along x changed as given, alone in the box (0,
    extents), or between the feeders of the foam member of ratio
    `fed_ratio` in its domain and with its frame, built with bed._link_sites
    recorded: case["sites"] is its call (args, kwargs, return), None when it
    did not run. Made once and shared.
    """
    key = (seed, tuple(float(v) for v in extents), fed_ratio, _law_key(changes))
    if key not in _LAW_BUILDS:
        calls = []
        real = bed._link_sites

        def recording(*args, **kwargs):
            found = real(*args, **kwargs)
            calls.append((args, kwargs, found))
            return found

        settings = dict(FIBRE, **changes)
        where = (f"fed, foam R {fed_ratio:g}" if fed_ratio is not None
                 else "alone, " + " x ".join(f"{v:g}" for v in extents))
        label = f"{where}, seed {seed}" + "".join(f", {k} {v}" for k, v in sorted(changes.items()))
        with mock.patch.object(bed, "_link_sites", side_effect=recording):
            if fed_ratio is None:
                case = build_case(label, (np.zeros(3), np.array(extents, dtype=float)), settings, seed=seed)
            else:
                call = foam_call(seed, fed_ratio)
                case = build_case(label, call["domain"], settings, seed=seed, feeders=call["feeders"],
                                  frame=call["frame"])
        case["sites"] = calls[0] if calls else None
        _LAW_BUILDS[key] = case
    return _LAW_BUILDS[key]


def law_grid_case(seed, spacing, density, sway, links):
    """A bed of the density law's grid, alone in the cube of side LAW_SIDE, run null."""
    return law_build(seed, spacing=spacing, density=density, sway=sway, link_spacing=20.0 if links else None)


def law_describe(case, nodes=None):
    """describe on a bed (its own nodes unless given), as the order calls it: the domain's extents as the volume."""
    low, high = case["domain"]
    nodes = case["result"]["nodes"] if nodes is None else nodes
    return describe.describe(nodes, graph.build(nodes)["edges"], volume=high - low, frame=case["result"]["frame"],
                             d_ref=case["d_min"])


def law_levels(s):
    """describe's nine transverse_spacing planes over positions s along a: s_min + (0.1 + 0.1 k)(s_max - s_min)."""
    s_min, s_max = float(np.min(s)), float(np.max(s))
    steps = describe._TRANSVERSE_FIRST + describe._TRANSVERSE_STEP * np.arange(describe.TRANSVERSE_PLANES)
    return s_min + steps * (s_max - s_min)


def law_along(points, a):
    """Each point's coordinate along the unit axis a, s = x . a, the three products summed in order."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points[:, 0] * a[0] + points[:, 1] * a[1] + points[:, 2] * a[2]


def law_chord_archive(case):
    """The chord graph as an archive: every kept edge a polyline of its two vertices, after one NaN separator."""
    edges = case["result"]["edges"]
    X = vertex_points(case)
    nodes = np.full((4, 3 * len(edges)), np.nan)
    nodes[:3, 1::3] = X[edges[:, 0]].T
    nodes[:3, 2::3] = X[edges[:, 1]].T
    nodes[3, 1::3] = nodes[3, 2::3] = case["settings"]["diameter"] * case["d_min"]
    return nodes


def law_chord_crossings(case, levels):
    """
    On the chord graph (the kept edges as straight segments), per plane
    s = c: the lane edges and the links with min(s) < c < max(s) of their
    two knots, counted on the knots' coordinates.
    """
    edges = case["result"]["edges"]
    s = law_along(vertex_points(case), resolved_axis(case))[edges]
    lo, hi = s.min(axis=1), s.max(axis=1)
    lane, link = lane_edge_mask(case, edges), link_edge_mask(case, edges)
    lanes = [int(np.count_nonzero(lane & (lo < c) & (c < hi))) for c in levels]
    links = [int(np.count_nonzero(link & (lo < c) & (c < hi))) for c in levels]
    return lanes, links


def law_spanning_links(case, levels):
    """Per plane s = c, the kept links with min(t_s, t_p) < c - o . a < max(t_s, t_p), from the knots' t values."""
    result = case["result"]
    low, high = case["domain"]
    origin = float(law_along((low + high) / 2.0, resolved_axis(case))[0])
    t = result["knot_t"][result["links"]].reshape(-1, 2)
    lo, hi = t.min(axis=1), t.max(axis=1)
    return [int(np.count_nonzero((lo < c - origin) & (c - origin < hi))) for c in levels]


def law_section(case):
    """B_perp1 B_perp2, the domain's section perpendicular to the coordinate axis a."""
    extents = case["extents"]
    j = int(np.argmax(np.abs(resolved_axis(case))))
    return float(np.prod([extents[k] for k in range(3) if k != j]))


def law_interior_chords(case):
    """The chords between consecutive interior knots of one lane: (lengths, |c . a|)."""
    result = case["result"]
    knots, lane, interior = result["knots"], result["knot_lane"], ~result["knot_boundary"]
    pair = interior[:-1] & interior[1:] & (lane[:-1] == lane[1:])
    chord = knots[1:][pair] - knots[:-1][pair]
    return np.sqrt(np.sum(chord * chord, axis=1)), np.abs(law_along(chord, resolved_axis(case)))


def law_cut_ratio(rng, sway, knot, w, pairs):
    """
    The test's own Monte Carlo of the cut law: `pairs` chords of axial
    length `knot` between two knots whose offsets are independent Gaussians
    of standard deviation `sway` per transverse component, each scaled
    radially to w when longer; the length-weighted crossing ratio
    sum(length) / sum(axial) of the chords.
    """
    offsets = sway * rng.standard_normal((pairs, 2, 2))
    radius = np.sqrt(np.sum(offsets * offsets, axis=2, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        offsets = np.where(radius > w, offsets * (w / radius), offsets)
    delta = offsets[:, 1] - offsets[:, 0]
    return float(np.sqrt(knot * knot + np.sum(delta * delta, axis=1)).sum() / (pairs * knot))


def law_j(beta):
    """J(beta) written out: 1 + beta sqrt(pi / 2) exp(1 / (2 beta^2)) erfc(1 / (sqrt 2 beta))."""
    return 1.0 + beta * math.sqrt(math.pi / 2.0) * math.exp(1.0 / (2.0 * beta * beta)) \
        * math.erfc(1.0 / (math.sqrt(2.0) * beta))


def law_series(beta):
    """J's series below beta 0.05: 1 + beta^2 - beta^4 + 3 beta^6 - 15 beta^8."""
    b2 = beta * beta
    return 1.0 + b2 - b2 ** 2 + 3.0 * b2 ** 3 - 15.0 * b2 ** 4


def law_beta(settings):
    """beta = sqrt 2 sway / knot, in bed.chord_crossing_ratio's own arithmetic."""
    return math.sqrt(2.0) * settings["sway"] / settings["knot"]


def law_unit(delta):
    """delta over its length, as Python floats: the row bed._unit_rows makes of it."""
    delta = np.asarray(delta, dtype=float)
    return (delta / np.linalg.norm(delta[None, :], axis=1)[0]).tolist()


def law_lane_ends(lane_pairs):
    """{knot: its lane-edge candidates (lower, upper)} of the (E, 2) lane_pairs."""
    ends = {}
    for a, b in np.asarray(lane_pairs).tolist():
        ends.setdefault(a, []).append((a, b))
        ends.setdefault(b, []).append((a, b))
    return ends


def law_feasible(knots, ends, site, partner, limit):
    """
    Whether the link from site to partner clears `limit` at both ends
    against the lane-edge candidates there, as _greedy decides: e from the
    lower knot id to the higher, a lane edge's direction +u at its lower
    knot and -u at its higher, refused where u . e (u . (-e) at the higher
    end), summed in order as Python floats, is above `limit`.
    """
    lo, hi = min(site, partner), max(site, partner)
    e = law_unit(knots[hi] - knots[lo])
    for at, sign in ((lo, 1.0), (hi, -1.0)):
        f = [sign * c for c in e]
        for a, b in ends.get(at, ()):
            u = law_unit(knots[b] - knots[a])
            if at == b:
                u = [-c for c in u]
            if u[0] * f[0] + u[1] * f[1] + u[2] * f[2] > limit:
                return False
    return True


def law_partners(case):
    """
    F5 rebuilt from the record, its draws recreated on sub-stream 6: the
    sites (u < q, one u per interior knot), then per site in site order its
    neighbouring lanes (2-D distance below link_reach h, extent holding the
    site's t, holding an interior knot, in increasing j), each with its
    interior knot nearest in t (the lower t on ties), those whose link
    clears max(min_angle, kin angle) against the lane-edge candidates at
    both ends kept (law_feasible), the partner lane kept[floor(v count)].
    The lane-edge candidates are rebuilt (consecutive knots of a lane, both
    off the feeders) on a bed without breaks, and read from the call with
    them. Returns the sites, the generator after its draws, the lane-edge
    candidates and, per site, ("no partner", None), ("angle", None), ("on
    feeders", partner) or ("link", partner), and the neighbours as (knot,
    feasible) pairs.
    """
    args, kwargs, _ = case["sites"]
    result, settings = case["result"], case["settings"]
    lane, t, boundary, knots = result["knot_lane"], result["knot_t"], result["knot_boundary"], result["knots"]
    points, extents = result["seeds"], result["lane_extent"]
    on_feeders = args[6]
    d_min = case["d_min"]
    h = settings["spacing"] * d_min
    q = settings["knot"] * d_min / (2.0 * settings["link_spacing"] * d_min)
    reach = settings["link_reach"] * h
    limit = min(math.cos(math.radians(settings["min_angle"])), math.cos(math.radians(bed.KIN_ANGLE_DEG)))
    if settings["run"] is None:
        same = np.flatnonzero((lane[1:] == lane[:-1]) & ~on_feeders[1:] & ~on_feeders[:-1])
        lane_pairs = np.stack([same, same + 1], axis=1)
    else:
        lane_pairs = np.asarray(args[8])
    ends = law_lane_ends(lane_pairs)
    interior = np.flatnonzero(~boundary)
    rng = np.random.default_rng([case["seed"], bed.RNG_TAG, bed.SUBSTREAMS["links"]])
    u = rng.random(interior.size)
    sites = interior[u < q]
    v = rng.random(sites.size)
    members = [interior[lane[interior] == i] for i in range(len(points))]
    # the lane points lie in one plane perpendicular to a, so the 2-D distance is that of the points
    e1, e2 = bed._plane_basis(resolved_axis(case))
    plane = np.stack([law_along(points, e1), law_along(points, e2)], axis=1)
    outcomes, neighbours = [], []
    for site, draw in zip(sites.tolist(), v.tolist()):
        i = int(lane[site])
        gap = plane - plane[i]
        near = np.flatnonzero(np.sqrt(gap[:, 0] ** 2 + gap[:, 1] ** 2) < reach).tolist()
        lanes = [j for j in near if j != i and members[j].size and extents[j, 0] <= t[site] <= extents[j, 1]]
        nearest = [int(members[j][int(np.argmin(np.abs(t[members[j]] - t[site])))]) for j in lanes]
        here = [(p, law_feasible(knots, ends, site, p, limit)) for p in nearest]
        neighbours.append(here)
        kept = [p for p, feasible in here if feasible]
        if not here:
            outcomes.append(("no partner", None))
        elif not kept:
            outcomes.append(("angle", None))
        else:
            partner = kept[min(int(draw * len(kept)), len(kept) - 1)]
            outcomes.append(("on feeders" if on_feeders[site] or on_feeders[partner] else "link", partner))
    return {"sites": sites, "rng": rng, "outcomes": outcomes, "neighbours": neighbours, "lane_pairs": lane_pairs,
            "ends": ends, "limit": limit, "q": q, "reach": reach, "plane": plane, "members": members}


def law_gaps(case, window):
    """
    The site gaps along each lane, in interior knots: from every site with
    window - 1 interior knots after it on its lane, the number of knots to
    the next site among them, or `window` when none is one. Each such gap
    reads only knots after its site, so the gaps are independent draws of
    the geometric law truncated at `window`, whatever the lanes' lengths.
    """
    result = case["result"]
    is_site = np.zeros(len(result["knots"]), dtype=bool)
    is_site[case["sites"][2]["sites"]] = True
    interior = ~result["knot_boundary"]
    gaps = []
    for i in range(len(result["seeds"])):
        flags = is_site[interior & (result["knot_lane"] == i)]           # in t order
        for k in np.flatnonzero(flags).tolist():
            if k + window - 1 > flags.size - 1:
                continue
            after = np.flatnonzero(flags[k + 1:k + window])
            gaps.append(int(after[0]) + 1 if after.size else window)
    return np.array(gaps, dtype=np.int64)


def law_cross_connections(case, feeder_ends=True):
    """
    The segments describe's "cross_connections" counts on the bed's archive,
    re-derived from describe's own segments (graph.segments, its layout and
    calibres) by the definition in its docstring: open capillary segments
    whose chord lies at least CROSS_ANGLE_DEG from a, both of whose end
    vertices have degree three or more, and at neither of whose ends a
    segment is not capillary-class (left out with feeder_ends False). Each
    is given as the bed vertices (knots, then tips) written on it, in
    order.
    """
    nodes = case["result"]["nodes"]
    built = graph.build(nodes)
    canonical, degree = built["canonical"], built["degree"]
    paths = graph.segments(nodes, built["edges"], canonical)
    layout = describe._segment_layout(nodes, paths)
    arc, mean = describe._segment_calibres(nodes, paths, graph.vertex_diameter(nodes, canonical))
    first, last = layout["first"], layout["last"]
    capillary = (arc > 0.0) & np.isfinite(mean) & (mean < describe.DEFAULT_CLASS_BOUND * case["d_min"])
    positions = nodes[:3].T
    chord = positions[last] - positions[first]
    length = np.sqrt(np.sum(chord * chord, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.abs(law_along(chord, resolved_axis(case))) / length
    transverse = capillary & (first != last) & (length > 0.0) & (u <= math.cos(math.radians(describe.CROSS_ANGLE_DEG)))
    joined = (degree[first] >= 3) & (degree[last] >= 3)
    if feeder_ends:
        larger = np.zeros(positions.shape[0], dtype=bool)
        larger[first[~capillary]] = larger[last[~capillary]] = True
        joined &= ~larger[first] & ~larger[last]
    vertex = {X.tobytes(): v for v, X in enumerate(vertex_points(case))}
    return [[vertex[positions[c].tobytes()] for c in paths[k] if positions[c].tobytes() in vertex]
            for k in np.flatnonzero(transverse & joined)]


def law_bed_cross_connections(case):
    """
    The cross-connections of a bed read off its return alone: the chains of
    the kept bed graph between two knots of degree three or more, through
    knots of degree two, that are open and whose chord between their end
    knots stands at least CROSS_ANGLE_DEG from a. A chain ending at a tip
    ends at a feeder, so it is never one. Each is the bed vertices in order.
    """
    result = case["result"]
    n_knots = len(result["knots"])
    n_vertices = n_knots + len(result["tips"])
    neighbours = [[] for _ in range(n_vertices)]
    for a, b in result["edges"].tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    degree = [len(others) for others in neighbours]
    X = vertex_points(case)
    a_hat = resolved_axis(case)
    across = math.cos(math.radians(describe.CROSS_ANGLE_DEG))
    chains, seen = [], set()
    for start in range(n_knots):
        if degree[start] < 3:
            continue
        for step in neighbours[start]:
            chain, previous, here = [start], start, step
            while here < n_knots and degree[here] == 2:
                chain.append(here)
                first, second = neighbours[here]
                previous, here = here, (second if first == previous else first)
            chain.append(here)
            key = tuple(chain) if chain[0] <= chain[-1] else tuple(reversed(chain))
            if key in seen:
                continue
            seen.add(key)
            end = chain[-1]
            if end >= n_knots or degree[end] < 3 or end == start:
                continue
            chord = X[end] - X[start]
            if abs(float(bed._dot3(chord, a_hat))) / math.sqrt(float(bed._dot3(chord, chord))) <= across:
                chains.append(chain)
    return chains


def law_upper_gamma(s, x):
    """
    The regularised upper incomplete gamma function Q(s, x), by its series
    below x = s + 1 and by its continued fraction (modified Lentz) above,
    with math.lgamma; the chi-square p-value of X^2 on k degrees of freedom
    is Q(k / 2, X^2 / 2).
    """
    if x <= 0.0:
        return 1.0
    scale = math.exp(-x + s * math.log(x) - math.lgamma(s))
    if x < s + 1.0:
        term = total = 1.0 / s
        a = s
        for _ in range(100000):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * scale)
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    fraction = d
    for i in range(1, 100000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        fraction *= d * c
        if abs(d * c - 1.0) < 1e-16:
            break
    return scale * fraction


def law_geometric_fit(gaps, q, window):
    """Pearson's X^2 of the gaps against the geometric law truncated at `window`: (X^2, df, p, expected counts)."""
    k = np.arange(1, window)
    p = np.concatenate([q * (1.0 - q) ** (k - 1), [(1.0 - q) ** (window - 1)]])
    observed = np.bincount(gaps, minlength=window + 1)[1:]
    expected = gaps.size * p
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    df = window - 1
    return statistic, df, law_upper_gamma(df / 2.0, statistic / 2.0), expected


class DensityLawTests(unittest.TestCase):
    """
    Test 5, the density law: on a bed alone along a coordinate axis without
    breaks every plane perpendicular to a inside the lanes' common extent
    meets each lane's chord path exactly once, whatever the sway or the link
    rate, and the capillary crossings of the chord graph there are N plus
    the kept links that span the plane. Beds alone in a cube of side 32
    d_min along x at spacing {5, 3.2} x density {0.4, 0.6}, sway 0 and 1,
    links at the default spacing 20 and none.

    The walked bed crosses each plane at least as often as its chord graph,
    by an even excess: a walk keeps its two ends, so it crosses a plane its
    chord crosses an odd number of times and one its chord misses an even
    number. The excess has no closed form; its tolerance comes from the
    development measurement on seeds 11-20 of the same sixteen beds (160
    builds):
      without links, the walked crossings equalled N on all nine planes of
        every bed: excess 0 on all 80;
      with links, the excess was 0, 2 or 4 crossings over the nine planes,
        at most 0.0294 / 0.0192 / 0.0118 / 0.0079 of the expected crossings
        at (5, 0.4) / (5, 0.6) / (3.2, 0.4) / (3.2, 0.6);
      no lane edge refused and no chain dropped on any of the 160;
      the link share of the chord graph's crossings, mean over the ten
        seeds at sway 0 / 1: 0.0167 / 0.0088 at (5, 0.4), 0.0175 / 0.0151 at
        (5, 0.6), 0.0121 / 0.0095 at (3.2, 0.4), 0.0154 / 0.0102 at (3.2, 0.6).
    The tolerances, WALKED_EXCESS_LINKS and WALKED_EXCESS_NO_LINKS, are set
    from these above. The fast tests read seed 1, the slow one seeds 1-10.
    """

    def check_chord_graph(self, case):
        """The chord-graph identity on describe's nine planes, counted here and by describe; returns the counts."""
        result, ev = case["result"], case["events"]
        n_lanes = len(result["seeds"])
        a = resolved_axis(case)
        # a coordinate axis: every lane spans the inset box along it, so the lanes' common extent is I's
        extent = result["lane_extent"]
        self.assertTrue(np.all(extent == extent[0]))
        self.assertEqual(ev["bed_lane_edges"], ev["bed_lane_candidates"])
        self.assertEqual(ev["bed_segments_dropped"], 0)
        levels = law_levels(law_along(vertex_points(case), a)[result["edges"].ravel()])
        lanes, links = law_chord_crossings(case, levels)
        spanning = law_spanning_links(case, levels)
        self.assertEqual(lanes, [n_lanes] * describe.TRANSVERSE_PLANES)
        self.assertEqual(links, spanning)
        crossings = [n_lanes + k for k in spanning]
        if case["settings"]["link_spacing"] is None:
            self.assertEqual(crossings, [n_lanes] * describe.TRANSVERSE_PLANES)
        # describe's own count on the chord graph written as an archive
        report = law_describe(case, law_chord_archive(case))
        spacing, section = report["transverse_spacing"], report["cross_section"]
        self.assertEqual(spacing["area_source"], "argument")
        self.assertEqual(spacing["count"], sum(crossings))          # every plane holds two crossings or more
        area = law_section(case)
        d2 = case["d_min"] ** 2
        for found, density in zip(crossings, section["per_plane_density_d"]):
            self.assertAlmostEqual(density * area / d2, found, delta=1e-9 * found)
        self.assertAlmostEqual(spacing["crossing_density_d"] * area * describe.TRANSVERSE_PLANES / d2,
                               sum(crossings), delta=1e-9 * sum(crossings))
        return crossings

    def check_walked(self, case):
        """
        describe's crossing_density_d of the walked bed against
        (N + spanning links) / area on its own nine planes (N / area without
        links), within the development tolerance; returns the link share of
        the chord graph's crossings there.
        """
        result = case["result"]
        n_lanes = len(result["seeds"])
        a = resolved_axis(case)
        nodes = result["nodes"]
        finite = np.isfinite(nodes[0])
        levels = law_levels(law_along(nodes[:3, finite].T, a))
        expected = describe.TRANSVERSE_PLANES * n_lanes + sum(law_spanning_links(case, levels))
        report = law_describe(case)
        spacing = report["transverse_spacing"]
        self.assertEqual(spacing["area_source"], "argument")
        area = law_section(case) * describe.TRANSVERSE_PLANES
        measured = spacing["count"]                                  # every plane holds two crossings or more
        self.assertAlmostEqual(spacing["crossing_density_d"] * area / case["d_min"] ** 2, measured,
                               delta=1e-9 * measured)
        excess = measured - expected
        self.assertGreaterEqual(excess, 0)
        self.assertEqual(excess % 2, 0)
        # the density against (N + spanning links) / area, N / area without links
        target = expected / area * case["d_min"] ** 2
        if case["settings"]["link_spacing"] is None:
            self.assertEqual(expected, describe.TRANSVERSE_PLANES * n_lanes)
            self.assertAlmostEqual(target, n_lanes / law_section(case) * case["d_min"] ** 2, delta=1e-15 * target)
            self.assertLessEqual(excess, WALKED_EXCESS_NO_LINKS)
            self.assertLessEqual(spacing["crossing_density_d"] - target,
                                 WALKED_EXCESS_NO_LINKS / area * case["d_min"] ** 2 + 1e-12 * target)
        else:
            self.assertLessEqual(spacing["crossing_density_d"] / target - 1.0, WALKED_EXCESS_LINKS)
        return (expected - describe.TRANSVERSE_PLANES * n_lanes) / expected

    def grid(self, seed):
        for spacing, density in LAW_GRID:
            for sway in (0.0, 1.0):
                for links in (False, True):
                    yield (spacing, density, sway, links), law_grid_case(seed, spacing, density, sway, links)

    def test_on_the_chord_graph_each_plane_meets_every_lane_once_and_the_spanning_links(self):
        for (spacing, density, sway, links), case in self.grid(1):
            with self.subTest(spacing=spacing, density=density, sway=sway, links=links):
                crossings = self.check_chord_graph(case)
                if links:
                    self.assertGreater(case["events"]["bed_links_kept"], 0)
                    self.assertGreater(sum(crossings), describe.TRANSVERSE_PLANES * len(case["result"]["seeds"]))

    def test_the_walked_bed_reads_n_over_the_section_with_the_spanning_links_on_top(self):
        for (spacing, density, sway, links), case in self.grid(1):
            with self.subTest(spacing=spacing, density=density, sway=sway, links=links):
                share = self.check_walked(case)
                if links:
                    self.assertGreater(share, 0.0)
                    self.assertLess(share, 0.05)
                else:
                    self.assertEqual(share, 0.0)

    @SLOW
    def test_the_density_law_holds_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for (spacing, density, sway, links), case in self.grid(seed):
                with self.subTest(seed=seed, spacing=spacing, density=density, sway=sway, links=links):
                    self.check_chord_graph(case)
                    self.check_walked(case)

    def test_the_lanes_knots_and_link_sites_do_not_depend_on_the_sway(self):
        cases = {sway: law_build(1, sway=sway) for sway in (0.0, 0.5, 1.0, 2.0)}
        reference = cases[0.0]
        # the partners are chosen among the feasible neighbours (F5), which move with the knots
        counts = ("bed_proposals", "bed_proposals_overlapping", "bed_proposals_outside", "bed_seeds", "bed_knots",
                  "bed_knots_boundary", "bed_link_sites", "bed_link_sites_no_partner")
        self.assertGreater(reference["events"]["bed_link_sites"], 0)
        for sway, case in cases.items():
            with self.subTest(sway=sway):
                result = case["result"]
                for key in ("seeds", "lane_extent", "knot_t", "knot_lane", "knot_boundary"):
                    self.assertEqual(result[key].tobytes(), reference["result"][key].tobytes(), key)
                self.assertEqual(case["sites"][2]["sites"].tobytes(), reference["sites"][2]["sites"].tobytes())
                self.assertEqual([case["events"][key] for key in counts],
                                 [reference["events"][key] for key in counts])
                # what the sway moves: the interior knots, and nothing on a lane's faces
                boundary = result["knot_boundary"]
                self.assertEqual(result["knots"][boundary].tobytes(), reference["result"]["knots"][boundary].tobytes())
                if sway > 0.0:
                    self.assertFalse(np.array_equal(result["knots"][~boundary], reference["result"]["knots"][~boundary]))


class OrientationTests(unittest.TestCase):
    """
    Test 6, the orientation of the lanes: parallel to a at sway 0; the chord
    law J(beta) of bed.chord_crossing_ratio, its series and its closed form;
    the interior chords' crossing ratio against the test's own Monte Carlo
    of the radially cut law within 4 combined standard errors (the bed's
    from its spread over seeds, the Monte Carlo's from its spread over
    CHORD_LAW_REPLICATES replicates), and that Monte Carlo against J(beta)
    within 4 of its standard errors at sigma <= w / 4; and the trends of
    describe's orientation over the sway.

    The trends have no closed form. Their development measurement, beds
    alone at the defaults along x in the cube of side 48, means over seeds
    11-20 at sway 0 / 0.5 / 1 / 1.5:
      capillary crossing_ratio 1.11884 / 1.14450 / 1.17453 / 1.19371
        (rising by 0.0257 / 0.0300 / 0.0192, and on every seed at every
        step; the kept links 36.8 / 35.4 / 30.4 / 29.5 on average);
      watson_K 8.152 / 6.474 / 5.161 / 4.658;
      fisher_axial_K_exact 9.491 / 7.951 / 6.704 / 6.090;
      both falling on every seed at every step.
    The slow test asserts the order's strict trends of the means over seeds
    1-10, with no margin: every step measured on a seed went the same way.
    """

    def test_at_sway_zero_every_lane_chord_is_parallel_to_the_axis(self):
        cases = [law_grid_case(1, spacing, density, 0.0, links) for spacing, density in LAW_GRID
                 for links in (False, True)]
        cases += [law_build(1, sway=0.0, axis=axis) for axis in ([1.0, 2.0, 2.0], [0.0, 0.6, 0.8])]
        for case in cases:
            with self.subTest(build=case["label"]):
                result = case["result"]
                a = resolved_axis(case)
                same = result["knot_lane"][1:] == result["knot_lane"][:-1]
                chord = (result["knots"][1:] - result["knots"][:-1])[same]
                self.assertEqual(len(chord), len(result["knots"]) - len(result["seeds"]))
                across = np.stack([chord[:, 1] * a[2] - chord[:, 2] * a[1], chord[:, 2] * a[0] - chord[:, 0] * a[2],
                                   chord[:, 0] * a[1] - chord[:, 1] * a[0]], axis=1)
                length = np.sqrt(np.sum(chord * chord, axis=1))
                self.assertTrue(np.all(np.sqrt(np.sum(across * across, axis=1)) <= 1e-12 * length))
                self.assertTrue(np.all(law_along(chord, a) > 0.0))          # in increasing t

    def test_the_chord_law_is_one_at_sway_zero_and_its_series_meets_its_closed_form(self):
        self.assertEqual(bed.chord_crossing_ratio(dict(FIBRE, sway=0.0)), 1.0)
        # either side of beta = 0.05 at knot 4: the series below, the closed form from 0.05 on
        below = above = 0.05 * 4.0 / math.sqrt(2.0)
        while law_beta(dict(FIBRE, sway=below)) >= 0.05:
            below = float(np.nextafter(below, 0.0))
        while law_beta(dict(FIBRE, sway=above)) < 0.05:
            above = float(np.nextafter(above, 1.0))
        low, high = bed.chord_crossing_ratio(dict(FIBRE, sway=below)), bed.chord_crossing_ratio(dict(FIBRE, sway=above))
        self.assertLessEqual(abs(low - high), 1e-10)
        self.assertAlmostEqual(low, law_series(law_beta(dict(FIBRE, sway=below))), delta=1e-15)
        self.assertAlmostEqual(high, law_j(law_beta(dict(FIBRE, sway=above))), delta=1e-15)
        self.assertLessEqual(abs(law_series(0.05) - law_j(0.05)), 1e-10)
        # the series' bound below 0.05, where J - 1 - beta^2 stands well above the rounding of J
        for beta in np.geomspace(1e-3, 0.0499, 60).tolist():
            settings = dict(FIBRE, sway=beta * 4.0 / math.sqrt(2.0))
            b = law_beta(settings)
            with self.subTest(beta=b):
                self.assertLessEqual(abs(bed.chord_crossing_ratio(settings) - 1.0 - b * b), 2.0 * b ** 4)
        # the closed form is the mean chord ratio of untruncated offsets, E sqrt(1 + beta^2 R^2), R Rayleigh:
        # Simpson's rule over r in [0, 40]
        r = np.linspace(0.0, 40.0, 200001)
        for sway, knot in ((0.2, 4.0), (1.0, 4.0), (1.5, 2.5), (4.0, 2.5)):
            settings = dict(FIBRE, sway=sway, knot=knot)
            b = law_beta(settings)
            f = np.sqrt(1.0 + b * b * r * r) * r * np.exp(-r * r / 2.0)
            quadrature = (r[1] - r[0]) / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
            with self.subTest(beta=b):
                self.assertAlmostEqual(bed.chord_crossing_ratio(settings), quadrature, delta=1e-10)

    def check_chord_law(self, seeds):
        rng = np.random.default_rng(314159)                 # the test's own Monte Carlo
        for name, changes, extents in CHORD_LAW_CASES:
            with self.subTest(case=name):
                settings = bed.parse_settings(dict(FIBRE, **changes))
                d_min = D_MIN
                h, T = settings["spacing"] * d_min, settings["diameter"] * d_min + MARGIN
                w, sway, knot = (h - T) / 2.0, settings["sway"] * d_min, settings["knot"] * d_min
                ratios = []
                for seed in seeds:
                    length, axial = law_interior_chords(law_build(seed, extents, **changes))
                    self.assertGreater(length.size, 50)
                    ratios.append(float(length.sum() / axial.sum()))
                ratios = np.array(ratios)
                carlo = np.array([law_cut_ratio(rng, sway, knot, w, CHORD_LAW_PAIRS)
                                  for _ in range(CHORD_LAW_REPLICATES)])
                se_bed = float(ratios.std(ddof=1)) / math.sqrt(ratios.size)
                se_carlo = float(carlo.std(ddof=1)) / math.sqrt(carlo.size)
                self.assertLessEqual(abs(float(ratios.mean()) - float(carlo.mean())),
                                     4.0 * math.sqrt(se_bed ** 2 + se_carlo ** 2))
                j = bed.chord_crossing_ratio(settings)
                if sway <= w / 4.0:
                    self.assertLessEqual(abs(float(carlo.mean()) - j), 4.0 * se_carlo)
                else:
                    # at sigma = w the cut lowers the ratio below J well beyond the Monte Carlo's error
                    self.assertLess(float(carlo.mean()), j - 4.0 * se_carlo)

    def test_the_interior_chords_follow_the_cut_law_and_it_follows_j_below_a_quarter_of_w(self):
        self.check_chord_law(range(1, 6))

    @SLOW
    def test_the_interior_chords_follow_the_cut_law_on_seeds_one_to_ten(self):
        self.check_chord_law(SEEDS)

    @SLOW
    def test_the_crossing_ratio_rises_and_the_concentrations_fall_with_the_sway(self):
        means = {}
        for sway in (0.0, 0.5, 1.0, 1.5):
            values = []
            for seed in SEEDS:
                capillary = law_describe(alone(seed, sway=sway))["frame_orientation"]["capillary"]
                values.append((capillary["crossing_ratio"], capillary["watson_K"], capillary["fisher_axial_K_exact"]))
            means[sway] = np.mean(np.array(values, dtype=float), axis=0)
        steps = list(zip((0.0, 0.5, 1.0), (0.5, 1.0, 1.5)))
        for k, name in enumerate(("crossing_ratio", "watson_K", "fisher_axial_K_exact")):
            for lower, upper in steps:
                with self.subTest(quantity=name, sway=(lower, upper)):
                    if k == 0:
                        self.assertGreater(means[upper][k], means[lower][k])
                    else:
                        self.assertLess(means[upper][k], means[lower][k])


class LinkTests(unittest.TestCase):
    """
    Test 7, the links: the site rate against q within 4 binomial standard
    deviations, the site gaps along a lane against the geometric law
    (Pearson's chi-square, p > 0.01), every partner within link_reach h at
    the nearest t and clear of the lane edges at both ends (against the
    record and against _greedy itself), F5's partition of the sites, and
    describe's cross-connections, alone and between feeders.
    """

    def binomial(self, case):
        q = case["settings"]["knot"] / (2.0 * case["settings"]["link_spacing"])
        m = interior_count(case)
        sites = case["events"]["bed_link_sites"]
        self.assertEqual(sites, case["sites"][2]["sites"].size)
        self.assertLessEqual(abs(sites - q * m), 4.0 * math.sqrt(m * q * (1.0 - q)))

    def test_the_sites_per_interior_knot_agree_with_q(self):
        cases = [law_grid_case(1, spacing, density, 1.0, True) for spacing, density in LAW_GRID]
        cases += [law_build(1, GAP_EXTENTS, link_spacing=8.0), law_build(1, (48.0,) * 3, **CROSS_SETTINGS),
                  law_build(1, fed_ratio=4.0)]
        for case in cases:
            with self.subTest(build=case["label"]):
                self.binomial(case)

    def check_gaps(self, cases, link_spacing):
        window = GAP_WINDOWS[link_spacing]
        q = FIBRE["knot"] / (2.0 * link_spacing)
        gaps = np.concatenate([law_gaps(case, window) for case in cases])
        statistic, df, p, expected = law_geometric_fit(gaps, q, window)
        self.assertGreaterEqual(float(expected.min()), 5.0)          # the chi-square's approximation holds
        self.assertGreater(p, 0.01, f"X^2 {statistic:.2f} on {df} df over {gaps.size} gaps")

    def test_the_site_gaps_along_a_lane_are_geometric(self):
        # the p-value's own function against two closed forms, on both sides of x = s + 1
        for y in (0.1, 0.7, 2.0, 5.0, 30.0):
            self.assertAlmostEqual(law_upper_gamma(1.0, y), math.exp(-y), delta=1e-13)
            self.assertAlmostEqual(law_upper_gamma(0.5, y), math.erfc(math.sqrt(y)), delta=1e-13)
        self.check_gaps([law_build(1, GAP_EXTENTS, link_spacing=8.0)], 8.0)

    @SLOW
    def test_the_site_gaps_along_a_lane_are_geometric_over_seeds_one_to_ten(self):
        for link_spacing in (8.0, 20.0):
            with self.subTest(link_spacing=link_spacing):
                self.check_gaps([law_build(seed, GAP_EXTENTS, link_spacing=link_spacing) for seed in SEEDS],
                                link_spacing)

    def check_partners(self, case):
        """Every candidate link joins its site to the nearest interior knot in t of a lane within link_reach h."""
        args, kwargs, found = case["sites"]
        rebuilt = law_partners(case)
        result, settings = case["result"], case["settings"]
        self.assertEqual((kwargs["q"], kwargs["reach"], kwargs["min_angle"]),
                         (rebuilt["q"], rebuilt["reach"], settings["min_angle"]))
        self.assertEqual(args[7].tobytes(), result["knots"].tobytes())
        # the lane-edge candidates passed are F4's: rebuilt without breaks, as many as counted
        self.assertEqual(len(args[8]), case["events"]["bed_lane_candidates"])
        if settings["run"] is None:
            self.assertEqual(np.asarray(args[8]).tolist(), rebuilt["lane_pairs"].tolist())
        self.assertEqual(found["sites"].tobytes(), rebuilt["sites"].tobytes())
        self.assertEqual(args[0].bit_generator.state, rebuilt["rng"].bit_generator.state)   # u, then v
        candidates = [(min(site, partner), max(site, partner))
                      for site, (outcome, partner) in zip(rebuilt["sites"].tolist(), rebuilt["outcomes"])
                      if outcome == "link"]
        self.assertEqual(found["links"].tolist(), [list(pair) for pair in candidates])
        self.assertGreater(len(candidates), 0)
        lane, t, knots = result["knot_lane"], result["knot_t"], result["knots"]
        a = resolved_axis(case)
        knot = settings["knot"] * case["d_min"]
        T = settings["diameter"] * case["d_min"] + case["margin"]
        for site, (outcome, partner), here in zip(rebuilt["sites"].tolist(), rebuilt["outcomes"],
                                                  rebuilt["neighbours"]):
            # an "angle" site has neighbours, none of them feasible; a partner is a feasible one
            if outcome == "angle":
                self.assertTrue(here)
                self.assertFalse(any(feasible for _, feasible in here))
            if partner is None:
                continue
            self.assertIn((partner, True), here)
            gap = rebuilt["plane"][lane[partner]] - rebuilt["plane"][lane[site]]
            self.assertLess(math.sqrt(gap[0] ** 2 + gap[1] ** 2), rebuilt["reach"])
            self.assertNotEqual(lane[partner], lane[site])
            along = t[rebuilt["members"][lane[partner]]]
            offset = abs(t[partner] - t[site])
            self.assertEqual(offset, float(np.min(np.abs(along - t[site]))))
            # beside a lane end the offset can reach l, elsewhere at most l / 2; the link crosses at least T
            self.assertLessEqual(offset, knot)
            if along.min() <= t[site] <= along.max():
                self.assertLessEqual(offset, knot / 2.0 * (1.0 + 1e-12))
            delta = knots[partner] - knots[site]
            across = delta - law_along(delta, a)[0] * a
            self.assertGreaterEqual(float(np.sqrt(np.sum(across * across))), T * (1.0 - 1e-12))
        kept = set(map(tuple, result["links"].tolist()))
        self.assertLessEqual(kept, set(candidates))
        self.assertEqual(len(kept), case["events"]["bed_links_kept"])

    def check_f5(self, case):
        """F5's partition of the sites, on the counters and on the rebuilt outcomes."""
        ev = case["events"]
        outcomes = [outcome for outcome, _ in law_partners(case)["outcomes"]]
        self.assertEqual(ev["bed_link_sites"],
                         ev["bed_link_sites_no_partner"] + ev["bed_link_sites_angle"] + ev["bed_link_sites_on_feeders"]
                         + ev["bed_candidates_link"])
        self.assertEqual((ev["bed_link_sites"], ev["bed_link_sites_no_partner"], ev["bed_link_sites_angle"],
                          ev["bed_link_sites_on_feeders"], ev["bed_candidates_link"]),
                         (len(outcomes), outcomes.count("no partner"), outcomes.count("angle"),
                          outcomes.count("on feeders"), outcomes.count("link")))
        if case["feeders"] is None:
            self.assertEqual(ev["bed_link_sites_on_feeders"], 0)
        return outcomes

    def link_cases(self, seed):
        return ([law_grid_case(seed, spacing, density, sway, True) for spacing, density in LAW_GRID
                 for sway in (0.0, 1.0)]
                + [law_build(seed, fed_ratio=4.0)])

    def test_each_partner_lies_within_link_reach_on_a_neighbouring_lane_at_the_nearest_t(self):
        for case in self.link_cases(1):
            with self.subTest(build=case["label"]):
                self.check_partners(case)

    def test_f5_partitions_the_sites_alone_and_with_feeders(self):
        for case in self.link_cases(1):
            with self.subTest(build=case["label"]):
                self.check_f5(case)
                check_laws(self, case)
        # with feeders, sites whose own or partner knot is on the feeders are refused
        outcomes = self.check_f5(law_build(1, fed_ratio=4.0))
        self.assertGreater(outcomes.count("on feeders"), 0)

    def lone_greedy(self, case, ends, site, partner):
        """_greedy over the lane-edge candidates at a link's two knots, in increasing order, then the link."""
        settings, knots = case["settings"], case["result"]["knots"]
        pairs = sorted(set(ends.get(site, []) + ends.get(partner, [])))
        pairs.append((min(site, partner), max(site, partner)))
        i, j = (np.array(column, dtype=np.int64) for column in zip(*pairs))
        n = len(knots)
        greedy = bed._greedy(knots, i, j, np.arange(len(pairs)), caps=[3] * n, inward={},
                             adjacency=[[] for _ in range(n)], girth=settings["girth"],
                             min_angle=settings["min_angle"], clearance=settings["diameter"] * case["d_min"]
                             + case["margin"], step=settings["step"] * settings["diameter"] * case["d_min"])
        return greedy, len(pairs) - 1

    def test_no_candidate_link_is_refused_by_the_angle_or_at_a_shared_end_against_the_lane_edges(self):
        # _greedy itself, over the lane-edge candidates at a link's two knots
        # and then the link: the link shares an end with each of them and
        # closes no cycle, so only the angle or the clearance at a shared end
        # can refuse it. Every candidate link is accepted; every neighbour of
        # an "angle" site is refused there, wherever the lane edges all stood
        cases = self.link_cases(1) + [law_build(1, sway=2.0), law_build(1, **BEYOND_F9)]
        angle = 0
        for case in cases:
            with self.subTest(build=case["label"]):
                rebuilt = law_partners(case)
                ends = rebuilt["ends"]
                links = case["sites"][2]["links"].tolist()
                self.assertGreater(len(links), 0)
                for site, partner in links:
                    greedy, _ = self.lone_greedy(case, ends, site, partner)
                    self.assertIn([site, partner], greedy["edges"].tolist())
                for site, (outcome, _), here in zip(rebuilt["sites"].tolist(), rebuilt["outcomes"],
                                                    rebuilt["neighbours"]):
                    if outcome != "angle":
                        continue
                    for partner, _ in here:
                        greedy, lanes = self.lone_greedy(case, ends, site, partner)
                        if len(greedy["edges"]) < lanes:
                            continue                           # a lane edge refused beside it, beyond F9
                        angle += 1
                        self.assertEqual(len(greedy["edges"]), lanes)
                        self.assertEqual(greedy["counts"]["angle"] + greedy["counts"]["at_end"], 1)
        self.assertGreater(angle, 0)

    @SLOW
    def test_the_site_rate_the_partners_and_f5_hold_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for case in self.link_cases(seed):
                with self.subTest(build=case["label"]):
                    self.binomial(case)
                    self.check_partners(case)
                    self.check_f5(case)

    def test_alone_every_kept_link_is_one_cross_connection_and_every_capillary_junction_is_t(self):
        case = law_build(1, (48.0,) * 3, **CROSS_SETTINGS)
        result, ev = case["result"], case["events"]
        self.assertEqual(ev["bed_segments_dropped"], 0)
        links = result["links"]
        self.assertGreater(len(links), 0)
        # the premise: every link's chord is within 60 degrees of transverse, beside a lane end too
        a = resolved_axis(case)
        chord = result["knots"][links[:, 1]] - result["knots"][links[:, 0]]
        u = np.abs(law_along(chord, a)) / np.sqrt(np.sum(chord * chord, axis=1))
        self.assertTrue(np.all(u <= math.cos(math.radians(describe.CROSS_ANGLE_DEG))))
        report = law_describe(case)
        cross = report["cross_connections"]
        self.assertEqual(cross["count"], ev["bed_links_kept"])
        # and the segments counted are the kept links themselves
        counted = law_cross_connections(case)
        self.assertEqual(len(counted), cross["count"])
        self.assertEqual(sorted(tuple(sorted(vertices)) for vertices in counted), sorted(map(tuple, links.tolist())))
        # each link ends at two interior knots of degree three, one link at each
        junctions = cross["junction_types"]
        self.assertEqual(junctions["count"], 2 * ev["bed_links_kept"])
        self.assertEqual((junctions["T"], junctions["Y"], junctions["other"]), (1.0, 0.0, 0.0))

    def test_with_feeders_the_cross_connections_are_the_links_between_two_knots_of_degree_three_and_merged_runs(self):
        # With feeders a cut can leave a link's end knot of degree 2, so that
        # the link and the lane runs beside it make one segment between two
        # knots of degree 3, counted when its chord is transverse (an L or a U
        # of lane runs whose axial parts cancel). describe's count is
        # therefore the kept links whose two end knots keep degree 3, which
        # are the chains of one edge, plus those merged runs; both are read
        # off the bed's return.
        for seed in (1, 3):
            with self.subTest(seed=seed):
                case = fed(seed, **CROSS_SETTINGS)
                result = case["result"]
                n_knots = len(result["knots"])
                degree = np.bincount(result["edges"].ravel(), minlength=n_knots + len(result["tips"]))
                links = result["links"]
                between = {tuple(link) for link in links.tolist() if degree[link[0]] == 3 and degree[link[1]] == 3}
                self.assertGreater(len(between), 0)
                self.assertGreater(case["events"]["bed_tips_attached"], 0)
                counted = law_bed_cross_connections(case)
                single = {tuple(sorted(chain)) for chain in counted if len(chain) == 2}
                merged = [chain for chain in counted if len(chain) > 2]
                self.assertEqual(single, between)
                kept_links = {tuple(link) for link in links.tolist()}
                for chain in merged:
                    pairs = {tuple(sorted(pair)) for pair in zip(chain[:-1], chain[1:])}
                    self.assertEqual(len(pairs & kept_links), 1, chain)
                self.assertEqual(law_describe(case)["cross_connections"]["count"], len(between) + len(merged))

    def test_with_feeders_no_tip_edge_is_counted(self):
        case = fed(1, **CROSS_SETTINGS)
        result = case["result"]
        n_knots = len(result["knots"])
        counted = law_cross_connections(case)
        self.assertEqual(len(counted), law_describe(case)["cross_connections"]["count"])
        self.assertTrue(all(max(vertices) < n_knots for vertices in counted))
        # not vacuous: transverse capillary segments with two junction ends hold tips, the feeder at the tip
        # excluding them
        excluded = law_cross_connections(case, feeder_ends=False)
        self.assertGreater(sum(1 for vertices in excluded if max(vertices) >= n_knots), 0)


# --------------------------------------------------------------------------
# Tests 8, 9 and 12: the rules on the accepted edges, the written columns
# and the laws of the counters, at the defaults and at the settings below


# The settings the rule, column and law tests build at, each alone (a cube
# of side SIDE) and between the foam member's feeders: the defaults (along
# x), without links, with breaks, without sway, at a sway above w = 1.5
# (most offsets cut at w), along an oblique axis (every component
# non-zero, so that every slab bound and the "outside" outcome are met) and
# along z, across the feeders' growth direction y. Each meets F9's
# condition, which does not depend on the sway, the links or the breaks.
CHECKED_SETTINGS = (("defaults", {}), ("links off", {"link_spacing": None}), ("run 30", {"run": 30.0}),
                    ("sway 0", {"sway": 0.0}), ("sway 2", {"sway": 2.0}),
                    ("oblique axis", {"axis": [1.0, 2.0, 2.0]}), ("z axis", {"axis": [0.0, 0.0, 1.0]}))

# A setting beyond F9's condition: h - T = 4 against l tan(90 - 72.06 / 2)
# = 2.89, so that w = 2 lets two chords of one lane meet below the kin
# angle and min_angle. The development measurement on seeds 11-20 (beds
# alone, side 48, 30 lanes): lane edges refused 37-66 of 668-672 lane
# candidates (by the angle 3-12 and the clearance at a shared end 32-54
# over all candidates), 9-22 of 19-44 link sites without a feasible
# partner, links accepted 9-24 of 10-24 candidates, no chain dropped and
# no chord fallback. The refusals are asserted only to be there.
BEYOND_F9 = {"spacing": 6.0, "knot": 2.1, "sway": 2.0}

# foam's ten keys of 3.7, in order, and those a fibre bed appends
FOAM_RETURN_KEYS = ("nodes", "tree", "frame", "seeds", "greedy_edges", "kept", "tips", "tip_edges", "edges",
                    "chains")
FIBRE_RETURN_KEYS = ("axis", "lane_extent", "knots", "knot_lane", "knot_t", "knot_boundary", "links")
CHAIN_KEYS = ("vertices", "outcome", "first", "last", "closed")


def checked_cases(seed, settings=CHECKED_SETTINGS):
    """The builds of each setting at this seed: alone, then between the foam member's feeders."""
    cases = []
    for _, changes in settings:
        cases += [alone(seed, **changes), fed(seed, **changes)]
    return cases


def beyond_f9_cases(seed):
    return [alone(seed, **BEYOND_F9), fed(seed, **BEYOND_F9)]


@functools.lru_cache(maxsize=None)
def foam_build(seed=1, with_feeders=False):
    """
    A foam bed at the foam preset's settings: alone in a cube of side SIDE,
    or between the foam member's feeders in its domain and with its frame.
    """
    settings = main.FAMILIES["foam"]["bed"]
    if not with_feeders:
        return build_case(f"foam alone, side {SIDE:g}, seed {seed}", (np.zeros(3), np.full(3, SIDE)), settings,
                          seed=seed)
    call = foam_call(seed)
    return build_case(f"foam fed, foam R 4, seed {seed}", call["domain"], settings, seed=seed,
                      feeders=call["feeders"], frame=call["frame"])


def accepted_view(case):
    """
    The case with its kept edges replaced by every edge the greedy pass
    accepted (greedy_edges, those pruned later included), for check_rules:
    kept, tip_edges and knot_boundary are set from those edges' degrees so
    that check_rules' checks of the pruned bed (a removed knot has no edge,
    a kept interior knot two or more) hold by construction, and only the
    greedy pass's rules are tested (the degree caps, min_angle, the kin
    angle, chord to chord at least T and the girth).
    """
    result = case["result"]
    greedy = result["greedy_edges"]
    n_knots = len(result["knots"])
    degree = np.bincount(greedy.ravel(), minlength=n_knots + len(result["tips"]))
    view = dict(result, edges=greedy, kept=degree[:n_knots] > 0,
                tip_edges=degree[n_knots:].astype(np.int64),
                knot_boundary=result["knot_boundary"] | (degree[:n_knots] == 1))
    return dict(case, result=view)


def check_bitwise_degrees(test, case):
    """
    The written bed rebuilt from its coordinates with no tolerance
    (graph.build at tol 0, which joins only columns that compare equal):
    every kept knot is written, and is one vertex whose degree is its
    number of kept bed edges; every knot the bed removed is written
    nowhere; a tip with k kept bed edges has degree 1 + k. Returns the
    counts of what was checked.
    """
    result = case["result"]
    nodes = result["nodes"]
    built = graph.build(nodes, tol=0.0)
    canonical, degree = built["canonical"], built["degree"]
    vertex_of = {}
    for column in np.flatnonzero(canonical >= 0).tolist():
        vertex_of.setdefault(nodes[:3, column].tobytes(), set()).add(int(canonical[column]))
    X = vertex_points(case)
    n_knots = len(result["knots"])
    bed_degree = np.bincount(result["edges"].ravel(), minlength=len(X))
    counts = {"knots": 0, "junctions": 0, "tips with two": 0}
    for v in range(n_knots):
        found = vertex_of.get(X[v].tobytes())
        if not result["kept"][v]:
            test.assertIsNone(found)
            continue
        test.assertIsNotNone(found)
        test.assertEqual(len(found), 1)
        test.assertEqual(int(degree[found.pop()]), int(bed_degree[v]))
        counts["knots"] += 1
        counts["junctions"] += int(bed_degree[v] == 3)
    for t, column in enumerate(result["tips"].tolist()):
        k = int(result["tip_edges"][t])
        test.assertEqual(int(degree[canonical[column]]), 1 + k)
        counts["tips with two"] += int(k == 2)
    return counts


def refusals(case):
    """The greedy pass's refusals at a build, by candidate kind and by rule, from its counters."""
    ev = case["events"]
    tips_accepted = ev["bed_edges"] - ev["bed_lane_edges"] - ev["bed_links"]
    return {"lane edges": (ev["bed_lane_candidates"] - ev["bed_lane_edges"], ev["bed_lane_candidates"]),
            "tip edges": (ev["bed_candidates_tip"] - tips_accepted, ev["bed_candidates_tip"]),
            "links": (ev["bed_candidates_link"] - ev["bed_links"], ev["bed_candidates_link"]),
            "degree": ev["bed_refused_degree"], "angle": ev["bed_refused_angle"],
            "girth": ev["bed_refused_girth"], "clearance": ev["bed_refused_clearance"],
            "at a shared end": ev["bed_refused_clearance_at_end"]}


def print_refusals(cases):
    for case in cases:
        found = refusals(case)
        kinds = ", ".join(f"{name} {found[name][0]} of {found[name][1]}" for name in ("lane edges", "tip edges",
                                                                                     "links"))
        rules = ", ".join(f"{name} {found[name]}" for name in ("degree", "angle", "girth", "clearance",
                                                                 "at a shared end"))
        print(f"\n  {case['label']}: refused {kinds}; by rule {rules}; dropped chains "
              f"{case['events']['bed_segments_dropped']}", end="")
    print()


class RuleTests(unittest.TestCase):
    """
    Test 8: the greedy pass's rules re-evaluated from the record, F9 on beds
    alone where its condition holds, and the laws on a setting beyond it.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = checked_cases(1) + beyond_f9_cases(1)

    def test_the_rules_hold_on_the_kept_edges_alone_and_with_feeders(self):
        tip_pairs, girths = 0, []
        for case in self.cases:
            with self.subTest(build=case["label"]):
                counts = check_rules(self, case)
                self.assertGreater(counts["pairs"], 0)
                self.assertGreater(counts["chord pairs"], 0)
                tip_pairs += counts["tip pairs"]
                girths.append(counts["girth"])
                if case["feeders"] is None and case["settings"]["link_spacing"] is None:
                    # lanes without links are paths: the bed alone has no cycle
                    self.assertEqual(counts["girth"], math.inf)
        self.assertGreater(tip_pairs, 0)            # the feeder's inward direction was met at some tip
        self.assertLess(min(girths), math.inf)      # and the girth on some cycle

    def test_the_rules_hold_on_every_edge_the_greedy_pass_accepted(self):
        # greedy_edges, the edges peeled, removed as islands or dropped later
        # included: the rules bound the pass itself, not only what survives it
        pruned = 0
        for case in self.cases:
            with self.subTest(build=case["label"]):
                counts = check_rules(self, accepted_view(case))
                self.assertGreater(counts["pairs"], 0)
                self.assertGreaterEqual(counts["girth"], case["settings"]["girth"])
                pruned += len(case["result"]["greedy_edges"]) - len(case["result"]["edges"])
        self.assertGreater(pruned, 0)               # the view holds edges the kept bed does not

    @SLOW
    def test_the_rules_hold_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for case in checked_cases(seed) + beyond_f9_cases(seed):
                with self.subTest(build=case["label"]):
                    self.assertGreater(check_rules(self, case)["pairs"], 0)
                    check_rules(self, accepted_view(case))

    def test_f9_holds_on_beds_alone_where_its_condition_does(self):
        for _, changes in CHECKED_SETTINGS:
            case = alone(1, **changes)
            with self.subTest(build=case["label"]):
                self.check_f9(case)

    @SLOW
    def test_f9_holds_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for _, changes in CHECKED_SETTINGS:
                case = alone(seed, **changes)
                with self.subTest(build=case["label"]):
                    self.check_f9(case)

    def check_f9(self, case):
        ev = case["events"]
        self.assertIsNone(case["feeders"])
        self.assertTrue(f9_holds_by_condition(case["settings"], case["d_min"], case["margin"]))
        self.assertGreater(ev["bed_lane_candidates"], 0)
        self.assertEqual(ev["bed_lane_edges"], ev["bed_lane_candidates"])      # no lane edge refused
        self.assertEqual(ev["bed_segments_dropped"], 0)                         # no chain dropped
        self.assertEqual(ev["bed_edges_dropped"], 0)

    def test_a_setting_beyond_f9_s_condition_refuses_lane_edges_and_still_obeys_f1_to_f8_and_e3_to_e5(self):
        self.check_beyond_f9(beyond_f9_cases(1))

    @SLOW
    def test_a_setting_beyond_f9_s_condition_still_obeys_its_laws_on_seeds_one_to_ten(self):
        self.check_beyond_f9([case for seed in SEEDS for case in beyond_f9_cases(seed)])

    def check_beyond_f9(self, cases):
        settings = bed.parse_settings(dict(FIBRE, **BEYOND_F9))
        self.assertFalse(f9_holds_by_condition(settings, D_MIN, MARGIN))
        for case in cases:
            with self.subTest(build=case["label"]):
                # F9 is not asserted where its condition fails; F1-F8 and
                # E1-E5, E7-E9 are
                check_laws(self, case)
                if case["feeders"] is None:
                    # the setting does refuse lane edges, at the angle and at a shared end
                    found = refusals(case)
                    self.assertGreater(found["lane edges"][0], 0)
                    self.assertGreater(found["angle"], 0)
                    self.assertGreater(found["at a shared end"], 0)
        print_refusals(cases)


class ColumnTests(unittest.TestCase):
    """
    Test 9: the written bed. Its ends and knots are written bitwise, an
    attached tip has degree 1 + k, describe.clearance at the build's margin
    finds no violation, and without feeders every column is the bed's.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = checked_cases(1) + beyond_f9_cases(1)

    def test_the_ends_and_knots_are_written_bitwise_and_a_tip_with_k_edges_has_degree_1_plus_k(self):
        junctions = attached = tips_with_two = 0
        for case in self.cases:
            with self.subTest(build=case["label"]):
                attached += check_columns(self, case)["tips attached"]
                counts = check_bitwise_degrees(self, case)
                self.assertGreater(counts["knots"], 0)
                junctions += counts["junctions"]
                tips_with_two += counts["tips with two"]
        self.assertGreater(junctions, 0)            # chains end bitwise at shared knots of degree 3
        self.assertGreater(attached, 0)
        self.assertGreater(tips_with_two, 0)        # a tip ending two chains, of degree 3

    def test_no_clearance_is_violated_at_the_build_s_margin(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                self.assertEqual(describe.clearance(case["result"]["nodes"], case["margin"])["violations"], 0)

    def test_the_bed_s_columns_are_labelled_bridge_and_without_feeders_no_column_is_anything_else(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                result = case["result"]
                nodes, tree = result["nodes"], result["tree"]
                finite = np.isfinite(nodes[0])
                n_feeders = 0 if case["feeders"] is None else np.asarray(case["feeders"][0]).shape[1]
                # one separator, labelled -1, before each chain
                self.assertEqual(int(np.count_nonzero(~finite[n_feeders:])), len(result["chains"]))
                self.assertTrue(np.all(tree[n_feeders:][~finite[n_feeders:]] == -1))
                self.assertTrue(np.all(tree[n_feeders:][finite[n_feeders:]] == BRIDGE))
                if case["feeders"] is None:
                    self.assertEqual(sorted(set(tree[finite].tolist())), [BRIDGE])
                    self.assertEqual(library.root_columns(tree), [])
                else:
                    # the feeders keep their labels, and the bed adds no root
                    np.testing.assert_array_equal(tree[:n_feeders], case["feeders"][1])
                    self.assertEqual(library.root_columns(tree), library.root_columns(case["feeders"][1]))
                    self.assertEqual(len(library.root_columns(tree)), 2)

    @SLOW
    def test_the_columns_and_the_clearance_hold_on_seeds_one_to_ten(self):
        attached = 0
        for seed in SEEDS:
            for case in checked_cases(seed) + beyond_f9_cases(seed):
                with self.subTest(build=case["label"]):
                    attached += check_columns(self, case)["tips attached"]
                    check_bitwise_degrees(self, case)
                    self.assertEqual(describe.clearance(case["result"]["nodes"], case["margin"])["violations"], 0)
        self.assertGreater(attached, 0)


class LawTests(unittest.TestCase):
    """
    Test 12: F1-F9, E1-E5 and E7-E9 on fibre beds alone and with feeders;
    every counter appended for the fibre bed is 0 on foam beds.
    """

    def test_the_laws_hold_alone_and_with_feeders_at_each_setting(self):
        cases = checked_cases(1)
        for case in cases:
            with self.subTest(build=case["label"]):
                check_laws(self, case)
        # every term of F1-F7 is met at least once, so that no law is
        # checked only at 0 = 0
        for key in ("bed_proposals_outside", "bed_proposals_overlapping", "bed_knots_on_feeders",
                    "bed_lane_edges_on_feeders", "bed_breaks", "bed_link_sites_no_partner", "bed_link_sites_angle",
                    "bed_link_sites_on_feeders", "bed_candidates_tip", "bed_refused_degree", "bed_refused_angle",
                    "bed_refused_girth", "bed_refused_clearance", "bed_edges_dead_end", "bed_edges_island",
                    "bed_knots_dead_end", "bed_knots_island", "bed_links_kept"):
            with self.subTest(counter=key):
                self.assertGreater(max(case["events"][key] for case in cases), 0)

    @SLOW
    def test_the_laws_hold_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            for case in checked_cases(seed):
                with self.subTest(build=case["label"]):
                    check_laws(self, case)

    def test_on_foam_beds_every_appended_counter_is_zero(self):
        self.check_foam_counters(1)

    @SLOW
    def test_on_foam_beds_every_appended_counter_is_zero_on_seeds_one_to_ten(self):
        for seed in SEEDS:
            self.check_foam_counters(seed)

    def check_foam_counters(self, seed):
        appended = bed.EVENT_KEYS[bed.EVENT_KEYS.index("bed_proposals_outside"):]
        self.assertEqual(len(appended), 17)                 # the seventeen counters of the fibre bed's laws
        for with_feeders in (False, True):
            case = foam_build(seed, with_feeders)
            with self.subTest(build=case["label"]):
                ev = case["events"]
                self.assertEqual(list(ev), list(bed.EVENT_KEYS))
                self.assertEqual({key: ev[key] for key in appended}, dict.fromkeys(appended, 0))
                self.assertGreater(ev["bed_seeds"], 0)
                self.assertGreater(ev["bed_edges"], 0)
                if with_feeders:
                    self.assertGreater(ev["bed_tips_attached"], 0)


class ReturnTests(unittest.TestCase):
    """
    The keys bed.build returns: foam's ten of 3.7 exactly, and for a fibre
    bed the same ten followed by its seven, with the dtypes and shapes of
    the order's interface.
    """

    def test_a_fibre_build_returns_foam_s_ten_keys_then_its_seven(self):
        for case in (alone(1), fed(1), alone(1, link_spacing=None), alone(1, axis=[1.0, 2.0, 2.0]),
                     fed(1, axis=[0.0, 0.0, 1.0])):
            with self.subTest(build=case["label"]):
                result = case["result"]
                self.assertEqual(tuple(result), FOAM_RETURN_KEYS + FIBRE_RETURN_KEYS)
                n_lanes, n_knots = len(result["seeds"]), len(result["knots"])
                n_tips = len(result["tips"])
                self.assertGreater(n_lanes, 0)
                self.assertGreater(n_knots, 2 * n_lanes)
                if case["feeders"] is None:
                    self.assertEqual(n_tips, 0)
                else:
                    self.assertGreater(n_tips, 0)

                def check(name, dtype, shape):
                    value = result[name]
                    self.assertIsInstance(value, np.ndarray, name)
                    self.assertEqual(value.dtype, np.dtype(dtype), name)
                    self.assertEqual(value.shape, shape, name)

                n_columns = result["nodes"].shape[1]
                check("nodes", np.float64, (4, n_columns))
                check("tree", np.int8, (n_columns,))
                check("seeds", np.float64, (n_lanes, 3))
                check("greedy_edges", np.int64, (len(result["greedy_edges"]), 2))
                check("kept", np.bool_, (n_knots,))
                check("tips", np.int64, (n_tips,))
                check("tip_edges", np.int64, (n_tips,))
                check("edges", np.int64, (len(result["edges"]), 2))
                check("lane_extent", np.float64, (n_lanes, 2))
                check("knots", np.float64, (n_knots, 3))
                check("knot_lane", np.int64, (n_knots,))
                check("knot_t", np.float64, (n_knots,))
                check("knot_boundary", np.bool_, (n_knots,))
                check("links", np.int64, (len(result["links"]), 2))
                # the vertex ids index the knots, then the tips
                for name in ("greedy_edges", "edges", "links"):
                    if len(result[name]):
                        self.assertGreaterEqual(int(result[name].min()), 0)
                        self.assertLess(int(result[name].max()), n_knots + n_tips)
                        self.assertTrue(np.all(result[name][:, 0] < result[name][:, 1]))
                self.assertTrue(np.all(result["links"] < n_knots))
                # "links" is the kept edges that join two lanes, in their order among the kept edges
                np.testing.assert_array_equal(result["links"], result["edges"][link_edge_mask(case, result["edges"])])
                self.assertTrue(np.all((result["knot_lane"] >= 0) & (result["knot_lane"] < n_lanes)))
                self.assertTrue(np.all(result["lane_extent"][:, 0] < result["lane_extent"][:, 1]))
                # the axis, a list of three floats, of unit length
                axis = result["axis"]
                self.assertIsInstance(axis, list)
                self.assertEqual(len(axis), 3)
                self.assertTrue(all(type(v) is float for v in axis))
                self.assertAlmostEqual(math.sqrt(sum(v * v for v in axis)), 1.0, delta=1e-12)
                self.assertIsInstance(result["frame"], dict)
                self.assertIsInstance(result["chains"], list)
                for chain in result["chains"]:
                    self.assertEqual(tuple(chain), CHAIN_KEYS)
                if case["settings"]["link_spacing"] is None:
                    self.assertEqual(result["links"].shape, (0, 2))

    def test_a_foam_build_returns_exactly_its_ten_keys(self):
        for with_feeders in (False, True):
            case = foam_build(1, with_feeders)
            with self.subTest(build=case["label"]):
                self.assertEqual(tuple(case["result"]), FOAM_RETURN_KEYS)



class SettingsObjectTests(unittest.TestCase):
    """BedSettings as a value: replaced, hashed, immutable, and its axis normalised at any magnitude."""

    def test_replace_works_on_either_kind_and_hashes(self):
        for settings in (bed.BedSettings(), bed.BedSettings(kind="fibre", axis=X_AXIS)):
            with self.subTest(kind=settings.kind):
                changed = dataclasses.replace(settings, spacing=6.0)
                self.assertEqual(changed.spacing, 6.0)
                self.assertEqual(dataclasses.replace(changed, spacing=settings.spacing), settings)
                self.assertEqual(hash(settings), hash(bed.BedSettings.parse(settings.as_dict())))
        # a field the kind does not take holds None and is refused when given anything else
        with self.assertRaises(ValueError):
            bed.BedSettings(kind="foam", knot=4.0)
        with self.assertRaises(ValueError):
            bed.parse_settings({"kind": "fibre", "stretch": None})

    def test_the_axis_handed_out_is_a_copy(self):
        settings = bed.BedSettings(kind="fibre", axis=[0.0, 1.0, 0.0])
        handed = settings.as_dict()
        handed["axis"][1] = 7.0
        self.assertEqual(settings.as_dict()["axis"], [0.0, 1.0, 0.0])
        self.assertIsInstance(settings.as_dict()["axis"], list)
        with self.assertRaises((TypeError, AttributeError)):
            settings.axis[0] = 3.0

    def test_an_axis_of_any_finite_magnitude_is_normalised(self):
        for given, expected in (([1e308, 1e308, 0.0], [math.sqrt(0.5), math.sqrt(0.5), 0.0]),
                                ([1e-200, 0.0, 0.0], [1.0, 0.0, 0.0]), ([0.0, -3e-310, 0.0], [0.0, -1.0, 0.0])):
            with self.subTest(axis=given):
                axis = bed.parse_settings(dict(FIBRE, axis=given))["axis"]
                np.testing.assert_allclose(axis, expected, rtol=1e-15, atol=0.0)
                self.assertEqual(bed.planned_seeds([40.0] * 3, dict(FIBRE, axis=None), D_MIN, axis=given),
                                 bed.planned_seeds([40.0] * 3, dict(FIBRE, axis=expected), D_MIN))
        with self.assertRaises(ValueError):
            bed.parse_settings(dict(FIBRE, axis=[0.0, 0.0, 0.0]))

    def test_a_frame_that_is_not_a_record_is_refused_before_any_draw(self):
        for frame in ([1, 2], "axis", 3.0):
            with self.subTest(frame=frame), recorded_generators() as created:
                with self.assertRaises(ValueError):
                    bed.build((np.zeros(3), np.full(3, 30.0)), FIBRE, d_min=D_MIN, seed=1, collision_margin=MARGIN,
                              frame=frame)
                self.assertEqual(seeds_of(created), [])


if __name__ == "__main__":
    unittest.main()
