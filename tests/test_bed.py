"""
Tests of bed.py, the explicit capillary bed, and of the foam family that
grows it: the settings and every refusal before a bed generator exists (by
bed.parse_settings, bed.build, grow_network and the command line), the
seeds against a brute-force loop, the rules re-evaluated from the record
(degree, angle, girth, dead ends, islands) and the greedy pass at their
bounds, the columns, tips, walk tangents and clearance of the written bed,
the laws of its counters (forced drops included) and the independence of
its sub-streams, the early return that leaves grow_network's released body
alone, scale, the frame record and stretch, the sizing guard, and foam end
to end.

Run from the repository root with
    python -m unittest tests.test_bed
The fast tests use seed 1. Seeds 2-10, scale, the order parameter over
stretch, the whole seed grid and foam end to end run only with
VSYSTEM_SLOW_TESTS=1.

The builds the tests read are made once each and shared by the classes
that read them (`alone`, `foam`): a bed alone (feeders None) in a cube of
side 64 d_min, 296 seeds, and the foam member of ratio 4, grown through
library.grow_member in its cube of side 60 with 243 seeds; bed.build's
whole result is captured by patching the name main looks it up under,
main.build_bed, since patching bed.build would not reach it.

The development measurement over seeds 1-10, which every margin below was
set from (beds alone at d_min 1 and margin 1 in cubes of side 64, 73 and 80
at stretch 1, 1.5 and 2, about 300 seeds each, and the foam member of
ratio 4):
  capillary-class S about y, stretch 1 / 1.5 / 2, per seed:
    0.016 0.133 0.199 | -0.035 0.131 0.161 | 0.002 0.127 0.196 |
    -0.033 0.090 0.204 | -0.034 0.106 0.168 | 0.008 0.116 0.166 |
    0.010 0.118 0.194 | -0.005 0.113 0.182 | 0.018 0.098 0.168 |
    0.005 0.107 0.175;
    the smallest rise from 1 to 1.5 was 0.080 (seed 9) and from 1.5 to 2
    0.029 (seed 2);
  the guard's ratio (M6), seed-pair candidates built over
  planned_candidates: bed alone 0.594-0.616 (2876-2985 of 4844), bed alone
    at stretch 2 0.571-0.604 (2729-2884 of 4778), foam 0.565-0.596
    (2245-2369 of 3977);
  chord fallbacks: none on seed 1 of either build; 1 on seeds 2, 7, 8 and
    10 alone, 1-3 on seeds 2, 4, 6, 8 and 10 of foam; no chain dropped and
    no closed chain on any seed;
  foam tips: 4-8 eligible, 1-5 of them with two bed edges on every seed;
    one component on every seed, touching both trees on all but seed 2.
"""
import ast
import collections
import contextlib
import copy
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import anastomosis  # noqa: E402
import bed  # noqa: E402
import describe  # noqa: E402
import frames  # noqa: E402
import graph  # noqa: E402
import libGenerator  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from guidance import unguided_frame  # noqa: E402
from tortuosity import bridge_path  # noqa: E402
from vSystem import F  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")

REFERENCE_MAIN = os.path.join(ROOT, "tests", "fixtures", "reference_code_3_6", "main.py")
D_MIN = 1.0
MARGIN = 1.0
H = 7.5                                  # the default hard core, d_min 1
LATER_SEEDS = tuple(range(2, 11))
# the cube of a bed alone, about 300 seeds, at each stretch the tests grow
SIDES = {1.0: 64.0, 1.5: 73.0, 2.0: 80.0}

# The preset as approved (plan section 6), every bed key written out.
FOAM_PRESET = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "grow_in_volume": True,
               "bed": {"kind": "foam", "spacing": 7.5, "girth": 8, "min_angle": 60.0, "reach": 2.5,
                       "diameter": 1.0, "persistence": 8.0, "feeder_stop": 2.0, "tip_edges": 2,
                       "stretch": 1.0, "density": 0.5, "jitter": 0.25, "step": 0.5}}
FOAM = FOAM_PRESET["bed"]

# The margins, from the development measurement above: the smallest rise
# of S measured, rounded down to its first significant figure; and the
# guard's ratio, the largest measured, 0.616, rounded up to the next 0.05.
S_RISE_TO_1_5 = 0.07
S_RISE_TO_2 = 0.02
GUARD_RATIO = 0.65

GUIDANCE = [{"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0}]


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


def alone(seed, stretch=1.0, side=None):
    """
    A bed alone (feeders None) in a cube of about 300 seeds, or of the side
    given; above stretch 1 with main's unguided frame.
    """
    side = SIDES[stretch] if side is None else side

    def make():
        settings = bed.parse_settings(dict(FOAM, stretch=stretch))
        frame = None
        if stretch != 1.0:
            frame = unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR, (side / 2.0,) * 3)
        events = {}
        domain = (np.zeros(3), np.full(3, side))
        result = bed.build(domain, settings, d_min=D_MIN, seed=seed, collision_margin=MARGIN, frame=frame,
                           events=events)
        return {"label": f"alone, side {side:g}, stretch {stretch:g}, seed {seed}", "result": result,
                "events": events, "settings": settings, "d_min": D_MIN, "margin": MARGIN, "attempts": 10,
                "feeders": None, "extents": domain[1] - domain[0], "planned": True,
                "call": ((domain, settings), dict(d_min=D_MIN, seed=seed, collision_margin=MARGIN, frame=frame))}
    return _cached(("alone", seed, stretch, side), make)


def foam(seed):
    """
    The foam member of ratio 4, with bed.build's result captured where main
    calls it, the generators its growth created and the global states it
    left.
    """
    def make():
        calls = []

        def capture(*args, **kwargs):
            result = bed.build(*args, **kwargs)
            calls.append((args, kwargs, result))
            return result

        with recorded_generators() as created, mock.patch.object(main, "build_bed", side_effect=capture):
            member = library.grow_member("foam", 4.0, seed)
        states = global_states()
        (args, kwargs, result), = calls
        domain = args[0]
        call = (args, {key: value for key, value in kwargs.items() if key != "events"})
        return {"label": f"foam R 4, seed {seed}", "result": result, "events": member["grown"]["events"],
                "settings": args[1], "d_min": kwargs["d_min"], "margin": kwargs["collision_margin"],
                "attempts": kwargs["attempts"], "feeders": kwargs["feeders"],
                "extents": np.asarray(domain[1]) - np.asarray(domain[0]), "planned": True, "call": call,
                "member": member, "generators": seeds_of(created), "states": states}
    return _cached(("foam", seed), make)


@contextlib.contextmanager
def every_walk_refused():
    """bed._collides answers True to the first walk of every chain, so that each falls back to its chord."""
    real = bed._collides
    walked = set()

    def first_walk_collides(obstacles, points, chain, anchors, closed, self_check=True):
        if chain >= 0 and chain not in walked:
            walked.add(chain)
            return True
        return real(obstacles, points, chain, anchors, closed, self_check)

    with mock.patch.object(bed, "_collides", side_effect=first_walk_collides):
        yield


def chords_of(case):
    """The same bed with every walk refused (attempts 0), so that every chain is placed as its checked chord."""
    def make():
        (args, kwargs) = case["call"]
        kwargs = dict(kwargs, attempts=0)
        events = {}
        with every_walk_refused():
            result = bed.build(*args, events=events, **kwargs)
        return dict(case, label=case["label"] + ", every walk refused", result=result, events=events, attempts=0)
    return _cached(("chords", case["label"]), make)


# Constructed beds: seeds given through a patched bed.place_seeds, the
# shortest-first order without jitter. An octagon of side h closes a cycle
# of exactly the girth, 8 edges; its sides meet at 135 degrees and every
# chord across it is refused by the angle, so the greedy pass keeps the
# octagon.
CONSTRUCTED = bed.parse_settings({"jitter": 0.0})
R8 = H / (2.0 * math.sin(math.pi / 8.0))        # the circumradius of an octagon of side h


def octagon(centre, start):
    centre = np.asarray(centre, dtype=float)
    return [centre + R8 * np.array([math.cos(start + 2 * math.pi * k / 8), math.sin(start + 2 * math.pi * k / 8), 0.0])
            for k in range(8)]


def feeder_to(tip, centre, label):
    """A straight feeder of diameter 2 running radially in to `tip` from 20 d_min outside it, sampled every 0.5."""
    outward = (tip - centre) / np.linalg.norm(tip - centre)
    points = [tip + outward * (20.0 - 0.5 * k) for k in range(40)] + [tip]
    nodes = np.empty((4, len(points)))
    nodes[:3] = np.array(points).T
    nodes[3] = 2.0
    return nodes, np.full(len(points), label, dtype=np.int8)


@contextlib.contextmanager
def chain_refused(anchors, closed):
    """
    bed._collides answers True to every walk and to the chord of the chain
    whose anchors (the vertices it shares) are `anchors` and that is closed
    or not as given, so that the chain is dropped; every other test is
    answered as before.
    """
    real = bed._collides
    anchors = set(anchors)

    def refusing(obstacles, points, chain, shared, is_closed, self_check=True):
        if chain >= 0 and bool(is_closed) == closed and set(shared) == anchors:
            return True
        return real(obstacles, points, chain, shared, is_closed, self_check)

    with mock.patch.object(bed, "_collides", side_effect=refusing):
        yield


def constructed(name, seed=1, refuse_walks=False, drop=None):
    """
    "lassos": two octagons, each hanging from a junction seed J, joined by a
    path of two seeds, a bed alone; "lassos to a tip": the lassos with the
    second octagon's vertex opposite J_B the tip of a straight feeder in
    place of a seed, so that it is two chains from J_B to the tip; "tip
    lasso": an octagon through the tip of a straight feeder, the tip taking
    two bed edges; "island": the tip lasso and, out of its reach, a second
    octagon with a tail of one seed, which the peel and the island pass
    remove; "two tips": an octagon through the tips of two straight feeders
    (trees 0 and 1) at opposite vertices, each tip taking two bed edges that
    end two chains.
    With `refuse_walks`, every walk is refused (attempts 0), so that every
    chain is placed as its checked chord. With `drop`, (anchors, closed),
    the chain chain_refused names is dropped.
    """
    return _cached(("constructed", name, seed, refuse_walks, drop), lambda: _construct(name, seed, refuse_walks, drop))


def _construct(name, seed, refuse_walks, drop):
    """constructed's build, made afresh on every call."""
    feeders = None
    if name in ("lassos", "lassos to a tip"):
        centre_a = np.array([20.0, 25.0, 20.0])
        ring_a = octagon(centre_a, 0.0)                      # vertex 0 is J_A, seed 0
        path = [ring_a[0] + H * np.array([1.0, 0.0, 0.0]) * k for k in (1, 2)]
        junction_b = ring_a[0] + np.array([3 * H, 0.0, 0.0])
        centre_b = junction_b + R8 * np.array([1.0, 0.0, 0.0])
        ring_b = octagon(centre_b, math.pi)                  # vertex 0 is J_B, seed 10
        if name == "lassos":
            seeds = ring_a + path + ring_b
        else:
            seeds = ring_a + path + [ring_b[k] for k in (0, 1, 2, 3, 5, 6, 7)]
            feeders = feeder_to(ring_b[4], centre_b, 0)                         # the tip, vertex 17
        domain = (np.zeros(3), np.array([90.0, 50.0, 40.0]))
    else:
        centre = np.array([30.0, 30.0, 20.0])
        ring = octagon(centre, 0.0)
        domain = (np.zeros(3), np.full(3, 60.0))
        if name in ("tip lasso", "island"):
            seeds = ring[1:]
            feeders = feeder_to(ring[0], centre, 0)
            if name == "island":
                far = octagon(centre + np.array([0.0, 0.0, 25.0]), 0.0)       # 25 > reach h = 18.75 above
                seeds = seeds + far + [far[0] + np.array([0.0, 0.0, H])]
        else:
            seeds = [ring[k] for k in (1, 2, 3, 5, 6, 7)]
            first, first_tree = feeder_to(ring[0], centre, 0)
            second, second_tree = feeder_to(ring[4], centre, 1)
            feeders = (np.concatenate([first, np.full((4, 1), np.nan), second], axis=1),
                       np.concatenate([first_tree, [-1], second_tree]).astype(np.int8))
    given = np.array(seeds)

    def place(rng, low, high, **options):
        return given.copy(), np.zeros(len(given), dtype=np.int8)

    events = {}
    attempts = 0 if refuse_walks else 10
    with mock.patch.object(bed, "place_seeds", side_effect=place):
        with every_walk_refused() if refuse_walks else contextlib.nullcontext():
            with chain_refused(*drop) if drop is not None else contextlib.nullcontext():
                result = bed.build(domain, CONSTRUCTED, d_min=D_MIN, seed=seed, collision_margin=MARGIN,
                                   feeders=feeders, attempts=attempts, events=events)
    label = (f"constructed {name}, seed {seed}" + (", every walk refused" if refuse_walks else "")
             + (f", the chain anchored at {sorted(drop[0])} dropped" if drop is not None else ""))
    # the seeds are given, so E2's planned count does not apply
    return {"label": label, "result": result, "events": events, "settings": CONSTRUCTED, "d_min": D_MIN,
            "margin": MARGIN, "attempts": attempts, "feeders": feeders, "extents": domain[1] - domain[0],
            "planned": False}


CONSTRUCTIONS = ("lassos", "lassos to a tip", "tip lasso", "two tips")

# The forced drops, (construction, (anchors, closed), what follows), the
# chains walked in the order lasso A, the path, then the second octagon's:
# the path between the lassos, its two seeds peeled; the second lasso,
# after the path is placed, so that the peel takes the path back with J_B
# and the octagon's seeds; the path to the tip's octagon, after lasso A is
# placed, so that the island pass takes lasso A back; and lasso A of the
# same, first, so that the peel retires the path before it is walked. What
# follows: (bed_segments_dropped, bed_segments_removed, bed_edges_dropped,
# (bed_seeds_dead_end, bed_edges_dead_end), (bed_seeds_island,
# bed_edges_island), the chains placed, the seeds kept).
DROPS = (
    ("lassos", ((0, 10), False), (1, 0, 3, (2, 0), (0, 0), 2, [True] * 8 + [False] * 2 + [True] * 8)),
    ("lassos", ((10,), True), (1, 1, 8, (10, 3), (0, 0), 1, [True] * 8 + [False] * 10)),
    ("lassos to a tip", ((0, 10), False), (1, 1, 3, (2, 0), (8, 8), 2, [False] * 10 + [True] * 7)),
    ("lassos to a tip", ((0,), True), (1, 0, 8, (10, 3), (0, 0), 2, [False] * 10 + [True] * 7)),
)


def vertex_points(case):
    """The bed vertices' coordinates: the seeds, then the eligible tips' feeder columns."""
    result = case["result"]
    if case["feeders"] is None:
        return result["seeds"]
    return np.concatenate([result["seeds"], case["feeders"][0][:3, result["tips"]].T])


# --------------------------------------------------------------------------
# The checks, each re-evaluated from the record and the stored coordinates


def check_identities(test, case):
    """E1-E9 of the plan's section 3 (bed.py's docstring)."""
    ev, result = case["events"], case["result"]
    test.assertEqual(ev["bed_proposals"],
                     ev["bed_proposals_on_feeders"] + ev["bed_proposals_overlapping"] + ev["bed_seeds"])      # E1
    test.assertEqual(ev["bed_seeds"], len(result["seeds"]))                                                   # E2
    if case["planned"]:
        test.assertEqual(ev["bed_seeds"], bed.planned_seeds(case["extents"], case["settings"], case["d_min"]))
    test.assertEqual(ev["bed_candidates"],
                     ev["bed_refused_degree"] + ev["bed_refused_angle"] + ev["bed_refused_girth"]
                     + ev["bed_refused_clearance"] + ev["bed_edges"])                                         # E3
    test.assertLessEqual(ev["bed_candidates_tip"], ev["bed_candidates"])
    test.assertLessEqual(ev["bed_refused_clearance_at_end"], ev["bed_refused_clearance"])                     # E4
    test.assertEqual(ev["bed_edges"], len(result["greedy_edges"]))
    test.assertEqual(ev["bed_edges"], len(result["edges"]) + ev["bed_edges_dead_end"] + ev["bed_edges_island"]
                     + ev["bed_edges_dropped"])                                                               # E5
    test.assertEqual(ev["bed_seeds"],
                     int(result["kept"].sum()) + ev["bed_seeds_dead_end"] + ev["bed_seeds_island"])          # E6
    test.assertEqual(ev["bed_segments"],
                     ev["bed_segments_walked"] + ev["bed_segments_chord"] + ev["bed_segments_dropped"])      # E7
    test.assertEqual(len(result["chains"]),
                     ev["bed_segments_walked"] + ev["bed_segments_chord"] - ev["bed_segments_removed"])
    test.assertEqual(ev["bed_tip_edges"], int(result["tip_edges"].sum()))                                    # E8
    test.assertEqual(ev["bed_tips_attached"], int(np.count_nonzero(result["tip_edges"])))
    test.assertLessEqual(ev["bed_tips_attached"], ev["bed_tips"] - ev["bed_tips_inside_junction"])
    test.assertLessEqual(ev["bed_components_one_tree"], ev["bed_components"])                                # E9
    test.assertLessEqual(ev["bed_walk_redraws"], case["attempts"] * ev["bed_segments"])


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
    Degree, angle, girth, no dead end and no island, re-evaluated from the
    record, and the components counted; returns the counts of what was
    checked, so that a caller can assert there was something to check.
    """
    result, settings = case["result"], case["settings"]
    seeds, tips, edges, kept = result["seeds"], result["tips"], result["edges"], result["kept"]
    n, t_count = len(seeds), len(tips)
    X = vertex_points(case)
    test.assertGreater(len(edges), 0)
    degree = np.bincount(edges.ravel(), minlength=n + t_count)
    # degree: a kept seed has two or three bed edges (so no dead end), a removed one none
    test.assertTrue(np.all((degree[:n][kept] == 2) | (degree[:n][kept] == 3)))
    test.assertTrue(np.all(degree[:n][~kept] == 0))
    # a tip has 1 + k_t <= 1 + tip_edges, k_t counted from the kept edges
    np.testing.assert_array_equal(degree[n:], result["tip_edges"])
    test.assertTrue(np.all(degree[n:] <= settings["tip_edges"]))

    # angle: every pair of directions at a vertex, the feeder's inward one
    # included at a tip, stands at least min_angle apart; two bed-edge
    # directions at least KIN_ANGLE_DEG apart, and a bed edge and the
    # feeder only min_angle (counted when closer than KIN_ANGLE_DEG). A
    # tip's inward direction is bed._inward, which the build uses in place
    # of -anastomosis._tip_tangent (the same rule, normalised elementwise):
    # it is first held to that within 1e-12 at every eligible tip, so that
    # the build and this check cannot agree on a wrong one.
    cos_min = math.cos(math.radians(settings["min_angle"]))
    cos_kin = math.cos(math.radians(bed.KIN_ANGLE_DEG))
    neighbours = collections.defaultdict(list)
    for a, b in edges.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    counts = {"pairs": 0, "tip pairs": 0, "pairs at a tip": 0, "tip edges inside the kin angle": 0}
    if case["feeders"] is not None:
        fnodes = case["feeders"][0]
        polylines = graph.polylines(fnodes)
        polyline_of = graph.polyline_of_column(fnodes)
        for column in tips.tolist():
            columns = polylines[polyline_of[column]]
            mine = bed._inward(fnodes, columns, column)
            theirs = anastomosis._tip_tangent(fnodes, columns, column)
            test.assertIsNotNone(theirs)
            test.assertLessEqual(float(np.max(np.abs(mine + theirs))), 1e-12)
    for v, others in neighbours.items():
        directions = [(X[w] - X[v]) / bed._dist(X[w][None, :], X[v][None, :])[0] for w in others]
        inward = None
        if v >= n:
            column = int(tips[v - n])
            inward = bed._inward(fnodes, polylines[polyline_of[column]], column)
        for p in range(len(directions)):
            for q in range(p + 1, len(directions)):
                dot = float(bed._dot3(directions[p], directions[q]))
                test.assertLessEqual(dot, cos_min)
                test.assertLessEqual(dot, cos_kin)
                counts["pairs"] += 1
                counts["pairs at a tip"] += v >= n
            if inward is not None:
                dot = float(bed._dot3(inward, directions[p]))
                test.assertLessEqual(dot, cos_min)
                counts["tip pairs"] += 1
                counts["tip edges inside the kin angle"] += dot > cos_kin

    # girth, in bed-graph edges over the kept bed edges and the feeder
    # segment graph (one edge per graph.segments path), by the test's own
    # search; a tip is the feeder vertex it stands on
    adjacency = collections.defaultdict(set)

    def vertex(v):
        return v if v < n else n + int(tips[v - n])

    links = [(vertex(a), vertex(b)) for a, b in edges.tolist()]
    if case["feeders"] is not None:
        for path in graph.segments(fnodes, graph.build(fnodes)["edges"]):
            test.assertNotEqual(int(path[0]), int(path[-1]))          # the feeders are trees
            links.append((n + int(path[0]), n + int(path[-1])))
    test.assertEqual(len(set(tuple(sorted(link)) for link in links)), len(links))   # a simple graph
    for a, b in links:
        adjacency[a].add(b)
        adjacency[b].add(a)
    girth = shortest_cycle(adjacency)
    test.assertGreaterEqual(girth, settings["girth"])
    counts["girth"] = girth

    # no island: with feeders, every kept component holds a tip; and E9's
    # components, counted by the test's own union-find: those with an edge,
    # and those whose tips all belong to one feeder tree (none without feeders)
    root = list(range(n + t_count))

    def find(v):
        while root[v] != v:
            root[v] = root[root[v]]
            v = root[v]
        return v

    for a, b in edges.tolist():
        root[find(a)] = find(b)
    components = {find(a) for a, _ in edges.tolist()}
    one_tree = 0
    if case["feeders"] is not None:
        with_tip = {find(n + t) for t in range(t_count)}
        for a, b in edges.tolist():
            test.assertIn(find(a), with_tip)
        labels = collections.defaultdict(set)
        for t, column in enumerate(tips.tolist()):
            labels[find(n + t)].add(int(case["feeders"][1][column]))
        one_tree = sum(len(labels[component]) == 1 for component in components)
    test.assertEqual((case["events"]["bed_components"], case["events"]["bed_components_one_tree"]),
                     (len(components), one_tree))
    counts["components"] = len(components)
    return counts


def check_columns(test, case):
    """
    The written bed: bitwise ends and interior seeds, the tips' degree, the
    labels and the layout of the columns. Returns what was checked.
    """
    result, feeders = case["result"], case["feeders"]
    nodes, tree, chains = result["nodes"], result["tree"], result["chains"]
    X = vertex_points(case)
    test.assertGreater(len(chains), 0)
    n_feeders = 0 if feeders is None else feeders[0].shape[1]
    d = case["settings"]["diameter"] * case["d_min"]
    column = n_feeders - 1                  # the last column written so far
    closed = 0
    for chain in chains:
        first, last, vertices = chain["first"], chain["last"], chain["vertices"]
        # one separator, then the polyline, every chain after the one before
        test.assertEqual(first, column + 2)
        test.assertTrue(np.all(np.isnan(nodes[:, first - 1])))
        test.assertEqual(int(tree[first - 1]), -1)
        test.assertGreaterEqual(last - first + 1, 2)
        test.assertTrue(np.all(np.isfinite(nodes[:, first:last + 1])))
        test.assertTrue(np.all(tree[first:last + 1] == BRIDGE))
        test.assertTrue(np.all(nodes[3, first:last + 1] == d))
        column = last
        # bitwise ends: a seed row, or the tip's feeder column
        test.assertEqual(nodes[:3, first].tobytes(), X[vertices[0]].tobytes())
        test.assertEqual(nodes[:3, last].tobytes(), X[vertices[-1]].tobytes())
        written = {nodes[:3, k].tobytes() for k in range(first, last + 1)}
        for v in vertices[1:-1]:
            test.assertIn(X[v].tobytes(), written)
        test.assertEqual(chain["closed"], vertices[0] == vertices[-1])
        test.assertIn(chain["outcome"], ("walked", "chord"))
        closed += chain["closed"]
    test.assertEqual(column, nodes.shape[1] - 1)
    separators = np.flatnonzero(np.isnan(nodes[0]))
    test.assertTrue(np.all(tree[separators] == -1))
    test.assertTrue(np.all(np.isnan(nodes[:, separators])))
    attached = 0
    if feeders is not None:
        # the feeders' columns and labels unchanged, NaN included, and their roots
        test.assertEqual(nodes[:, :n_feeders].tobytes(), np.asarray(feeders[0], dtype=float).tobytes())
        np.testing.assert_array_equal(tree[:n_feeders], feeders[1])
        test.assertEqual(library.root_columns(tree), library.root_columns(feeders[1]))
        # a tip with k bed edges ends with degree 1 + k (graph.py's degree of its vertex)
        degree = graph.build(nodes)["degree"]
        tips = result["tips"]
        test.assertGreater(len(tips), 0)
        for t, tip in enumerate(tips.tolist()):
            test.assertEqual(int(degree[tip]), 1 + int(result["tip_edges"][t]))
            attached += int(result["tip_edges"][t]) == 2
    return {"chains": len(chains), "closed": closed, "tips with two bed edges": attached}


def check_fallbacks(test, case):
    """
    Every chain placed as its chord: the concatenation of its edges' stored
    samples, bridge_path(x_lo, x_hi, step d) with no tangents, reversed
    where the chain runs from the higher vertex to the lower, each joint
    once, in the order written; every point on its seed chord to 1e-12.
    Returns the number of chord chains checked.
    """
    result, settings = case["result"], case["settings"]
    nodes = result["nodes"]
    X = vertex_points(case)
    step = settings["step"] * settings["diameter"] * case["d_min"]
    checked = 0
    for chain in result["chains"]:
        if chain["outcome"] != "chord":
            continue
        vertices = chain["vertices"]
        pieces = []
        for a, b in zip(vertices[:-1], vertices[1:]):
            lo, hi = min(a, b), max(a, b)
            piece = bridge_path(X[lo], X[hi], step)
            along = X[hi] - X[lo]
            offset = piece - X[lo]
            t = offset @ along / (along @ along)
            off_chord = np.linalg.norm(offset - t[:, None] * along, axis=1)
            test.assertLessEqual(float(off_chord.max()), 1e-12)
            test.assertTrue(np.all((t >= -1e-12) & (t <= 1.0 + 1e-12)))
            pieces.append(piece if a < b else piece[::-1])
        joined = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
        written = np.ascontiguousarray(nodes[:3, chain["first"]:chain["last"] + 1].T)
        test.assertEqual(written.tobytes(), np.ascontiguousarray(joined).tobytes())
        checked += 1
    return checked


@contextlib.contextmanager
def walk_tangents():
    """
    Every walk piece bed.build asks of bridge_path (those given a generator),
    as {(start bytes, end bytes): [(start_tangent, end_tangent), ...]}, one
    entry per attempt.
    """
    real = bed.bridge_path
    calls = collections.defaultdict(list)

    def spy(*args, **kwargs):
        if kwargs.get("rng") is not None:
            start, end = (np.asarray(x, dtype=float) for x in args[:2])
            calls[(start.tobytes(), end.tobytes())].append((kwargs.get("start_tangent"), kwargs.get("end_tangent")))
        return real(*args, **kwargs)

    with mock.patch.object(bed, "bridge_path", side_effect=spy):
        yield calls


def check_tangents(test, case, calls):
    """
    The walk's tangents (D4): None at a chain's ends, at a closed chain its J
    (the vertex of degree other than two or the tip, else the ring's first),
    and at every interior seed the Catmull-Rom tangent unit(x_next - x_prev)
    computed as bed.py computes it, bitwise, on every attempt. Returns the
    interior tangents checked.
    """
    result = case["result"]
    X = vertex_points(case)
    n = len(result["seeds"])
    degree = np.bincount(result["edges"].ravel(), minlength=len(X))
    checked = 0
    for chain in result["chains"]:
        # every chain placed was walked at least once, a chord fallback too
        vertices = chain["vertices"]
        last = len(vertices) - 1
        if chain["closed"]:
            junctions = [v for v in vertices[:-1] if v >= n or degree[v] != 2]
            test.assertLessEqual(len(junctions), 1)
            ends = set(junctions) if junctions else {vertices[0]}
        for k in range(last):
            attempts = calls[(X[vertices[k]].tobytes(), X[vertices[k + 1]].tobytes())]
            test.assertGreater(len(attempts), 0)
            for start_tangent, end_tangent in attempts:
                for position, tangent in ((k, start_tangent), (k + 1, end_tangent)):
                    if chain["closed"]:
                        end = vertices[position] in ends
                        before = vertices[position - 1] if position > 0 else vertices[last - 1]
                        after = vertices[position + 1] if position < last else vertices[1]
                    else:
                        end = position in (0, last)
                        before, after = vertices[max(position - 1, 0)], vertices[min(position + 1, last)]
                    if end:
                        test.assertIsNone(tangent)
                        continue
                    delta = X[after] - X[before]
                    expected = delta / bed._dist(delta[None, :], np.zeros((1, 3)))[0]
                    test.assertIsNotNone(tangent)
                    test.assertEqual(np.asarray(tangent, dtype=float).tobytes(), expected.tobytes())
                    checked += 1
    return checked


# --------------------------------------------------------------------------
# Item 30: settings and refusals


def cli_argv(**changes):
    """
    A bed run's command line: a 48 cube at voxel size 1, d_min 1, the walk,
    d0 8, seed 3 and --bed {}. A change replaces an option's values; None
    drops it, and a string is written as --option=value.
    """
    options = {"--grow-in-volume": [], "--d-min": ["1"], "--volume": ["48", "48", "48"], "--fit": ["voxel_size"],
               "--voxel-size": ["1"], "--tortuosity": ["walk"], "--persistence": ["10"], "--d0": ["8", "0"],
               "--iterations": ["6", "6"], "--count": ["1"], "--seed": ["3"], "--bed": ["{}"]}
    for name, values in changes.items():
        options[name] = values
    argv = []
    for name, values in options.items():
        if values is None:
            continue
        if isinstance(values, str):
            argv.append(f"{name}={values}")
        else:
            argv += [name] + list(values)
    return argv


# grow_network's side of each refusal: the arguments of a bed run (a 48
# cube at voxel size 1, d0 8, seed 3) changed as given
GROW = {"niter": 6, "d0": 8.0, "tVol": (48, 48, 48), "fit": "voxel_size", "voxel_size": 1.0, "d_min": 1.0,
        "grow_in_volume": True, "tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "seed": 3,
        "bed": FOAM}

# The settings specs refused, as JSON text; spacing 1.5 is refused by
# check_settings (h <= d + margin), every other by parse_settings.
REFUSED_SPECS = ('{"kind":"lattice"}', '{"spacing":0}', '{"girth":2}', '{"girth":8.0}', '{"girth":17}',
                 '{"tip_edges":3}', '{"stretch":0.5}', '{"stretch":5}', '{"density":0.65}', '{"step":0.05}',
                 '{"reach":1}', '{"spacing":1.5}', '{"colour":1}')

# Every refusal made before a bed generator exists, paired: (name, the
# change to grow_network's arguments, the change to the command line, a
# word of grow_network's message, so that each case is refused for its own
# reason). The negative seed and the non-finite voxel size join the plan's
# list: the command line refuses them too.
REFUSALS = tuple((f"bed {spec}", {"bed": json.loads(spec)}, {"--bed": [spec]}, "bed") for spec in REFUSED_SPECS) + (
    ("with anastomose", {"anastomose": True}, {"--anastomose": []}, "anastomose"),
    ("with cross_connect", {"cross_connect": True}, {"--cross-connect": []}, "cross_connect"),
    ("with capillary generations", {"capillary_generations": 1}, {"--capillary-generations": ["1"]}, "capillary fill"),
    ("with capillary runs", {"capillary_runs": 2}, {"--capillary-runs": ["2"]}, "capillary fill"),
    ("with guidance", {"guidance": GUIDANCE}, {"--guidance": [json.dumps(GUIDANCE)]}, "guidance"),
    ("without grow_in_volume", {"grow_in_volume": False}, {"--grow-in-volume": None}, "grow_in_volume"),
    ("without d_min", {"d_min": None}, {"--d-min": None}, "needs d_min"),
    ("d0 below feeder_stop x d_min", {"d0": 3.0, "d_min": 2.0}, {"--d0": ["3", "0"], "--d-min": ["2"]},
     "feeder_stop"),
    # V_c = 9^3 = 729 < 2 h^3 = 844: N_t = 0, main.BedBoxTooSmall
    ("a box that holds no seed", {"tVol": (10, 10, 10)}, {"--volume": ["10", "10", "10"]}, "holds no seed"),
    ("a negative seed", {"seed": -1}, {"--seed": "-1"}, "seed"),
    ("an infinite voxel size", {"voxel_size": math.inf}, {"--voxel-size": ["inf"]}, "voxel_size"),
    ("a voxel size that is not a number", {"voxel_size": math.nan}, {"--voxel-size": ["nan"]}, "voxel_size"),
)


class SettingsTests(unittest.TestCase):
    SPECS = ({}, {"kind": "foam"}, {"girth": 10}, {"girth": 3, "tip_edges": 1, "min_angle": 0},
             {"spacing": 9, "stretch": 2, "density": 0.3, "jitter": 0, "step": 0.1, "reach": 1.5},
             {"girth": np.int64(5), "spacing": np.float64(8.0), "feeder_stop": 1}, FOAM)

    def test_the_defaults_are_the_whole_dict_in_key_order(self):
        settings = bed.parse_settings({"kind": "foam"})
        self.assertEqual(settings, FOAM)
        self.assertEqual(list(settings), list(bed.SETTING_KEYS))
        self.assertIs(type(settings["girth"]), int)
        self.assertIs(type(settings["tip_edges"]), int)
        self.assertEqual(bed.BedSettings().as_dict(), settings)

    def test_normalising_twice_changes_nothing_and_survives_json(self):
        for spec in self.SPECS + (bed.BedSettings(girth=12),):
            with self.subTest(spec=spec):
                settings = bed.parse_settings(spec)
                self.assertEqual(list(settings), list(bed.SETTING_KEYS))
                self.assertEqual(bed.parse_settings(settings), settings)
                self.assertEqual(json.loads(json.dumps(settings)), settings)
                self.assertEqual(bed.BedSettings.parse(spec).as_dict(), settings)
        self.assertEqual(bed.parse_settings({"girth": 10})["girth"], 10)
        self.assertEqual(bed.parse_settings({"girth": np.int64(5)})["girth"], 5)
        self.assertIs(type(bed.parse_settings({"spacing": 9})["spacing"]), float)

    def test_the_settings_refused_are_refused_by_parse_settings_and_check_settings(self):
        for spec in REFUSED_SPECS:
            with self.subTest(spec=spec):
                value = json.loads(spec)
                if spec == '{"spacing":1.5}':
                    # a valid spec on its own; seeds h apart would touch at d_min 1 and margin 1
                    self.assertEqual(bed.parse_settings(value)["spacing"], 1.5)
                else:
                    with self.assertRaises(ValueError):
                        bed.parse_settings(value)
                with self.assertRaises(ValueError):
                    bed.check_settings(value, d_min=1.0, collision_margin=1.0)
        for spec in ([], "foam", None, {"girth": True}, {"tip_edges": True}, {"spacing": math.nan},
                     {"jitter": -0.1}, {"feeder_stop": 0.5}, {"min_angle": 121}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                bed.parse_settings(spec)

    def test_every_refusal_before_a_bed_generator_exists_leaves_both_global_states_alone(self):
        self.assertGreater(len(REFUSALS), 0)
        for name, change, _, word in REFUSALS:
            with self.subTest(refusal=name):
                arguments = dict(GROW, **change)
                niter, d0, tVol = arguments.pop("niter"), arguments.pop("d0"), arguments.pop("tVol")
                random.seed(11)
                np.random.seed(11)
                before = global_states()
                with recorded_generators() as created:
                    with self.assertRaises(ValueError) as caught:
                        main.grow_network(niter, d0, None, tVol, **arguments)
                self.assertIn(word, str(caught.exception))
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)
                if name == "a box that holds no seed":
                    self.assertIsInstance(caught.exception, main.BedBoxTooSmall)

    def test_build_refuses_a_malformed_argument_before_any_generator_exists(self):
        # a bed of 28 seeds in a cube of side 30 attached to the tip of one
        # straight feeder; each change is refused for its own reason
        nodes, tree = feeder_to(np.array([15.0, 15.0, 15.0]), np.array([15.0, 15.0, 5.0]), 0)
        base = {"domain": (np.zeros(3), np.full(3, 30.0)), "settings": FOAM, "d_min": D_MIN, "seed": 1,
                "collision_margin": MARGIN, "feeders": (nodes, tree)}

        def changed(column=None, value=None, label=None, rows=4):
            """The feeders with a diameter or a label changed, or only their first rows."""
            n, t = nodes[:rows].copy(), tree.copy()
            if value is not None:
                n[3, column] = value
            if label is not None:
                t = t.astype(np.int16)
                t[column] = label
            return n, t

        separated = (np.concatenate([nodes, np.full((4, 1), np.nan), nodes + np.array([[1.0], [0.0], [0.0], [0.0]])],
                                    axis=1),
                     np.concatenate([tree, [0], tree]).astype(np.int8))
        cases = [
            ("feeders that are not a pair", {"feeders": (nodes,)}, "feeders must be"),
            ("feeder nodes of shape (3, N)", {"feeders": changed(rows=3)}, "(4, N) array"),
            ("a float tree", {"feeders": (nodes, tree.astype(float))}, "integer label"),
            ("a tree one label short", {"feeders": (nodes, tree[:-1])}, "integer label"),
            ("a label of 300 in an int16 tree", {"feeders": changed(0, label=300)}, "fit an int8"),
            ("a finite column unlabelled", {"feeders": changed(3, label=-1)}, "label every finite column"),
            ("a labelled separator", {"feeders": separated}, "label every finite column"),
            ("a diameter of nan", {"feeders": changed(5, math.nan)}, "finite diameter above 0"),
            ("a diameter of 0", {"feeders": changed(5, 0.0)}, "finite diameter above 0"),
            ("a diameter of -1", {"feeders": changed(5, -1.0)}, "finite diameter above 0"),
            ("a diameter of inf", {"feeders": changed(5, math.inf)}, "finite diameter above 0"),
            ("margin None", {"collision_margin": None}, "collision margin"),
            ("margin -1", {"collision_margin": -1.0}, "collision margin"),
            ("margin nan", {"collision_margin": math.nan}, "collision margin"),
            ("seed -1", {"seed": -1}, "needs a seed"),
            ("seed 1.5", {"seed": 1.5}, "needs a seed"),
            ("seed True", {"seed": True}, "needs a seed"),
            ("seed None", {"seed": None}, "needs a seed"),
            ("attempts -1", {"attempts": -1}, "walk attempts"),
            ("attempts 1.5", {"attempts": 1.5}, "walk attempts"),
            ("index_kind x", {"index_kind": "x"}, "index kind"),
            ("d_min 0", {"d_min": 0.0}, "needs d_min"),
            ("d_min None", {"d_min": None}, "needs d_min"),
            ("events not a dict", {"events": []}, "events must be"),
            ("timings not a dict", {"timings": []}, "timings must be"),
            ("a domain that is not a pair", {"domain": (np.zeros(3),)}, "must be (low, high)"),
            ("a domain with an infinite corner", {"domain": (np.zeros(3), np.array([30.0, math.inf, 30.0]))},
             "finite 3-vectors"),
            ("a domain no wider than d", {"domain": (np.zeros(3), np.array([30.0, 30.0, 1.0]))}, "wider than"),
            ("settings refused by parse_settings", {"settings": {"girth": 2}}, "girth"),
            ("settings refused by check_settings", {"settings": dict(FOAM, spacing=1.5)}, "seeds would touch"),
            ("stretch 2 without a frame", {"settings": dict(FOAM, stretch=2.0)}, "frame"),
            # V_c = 9^3 = 729 < 2 h^3 = 844: N_t = 0
            ("a box that holds no seed", {"domain": (np.zeros(3), np.full(3, 10.0))}, "holds no seed"),
        ]
        self.assertEqual(bed.planned_seeds((10.0, 10.0, 10.0), FOAM, D_MIN), 0)
        # the base call itself is a bed
        with recorded_generators() as created:
            arguments = dict(base)
            result = bed.build(arguments.pop("domain"), arguments.pop("settings"), **arguments)
        self.assertEqual(seeds_of(created), [[1, bed.RNG_TAG, k] for k in (1, 2, 3)])
        self.assertEqual(result["tip_edges"].tolist(), [2])
        for name, change, words in cases:
            with self.subTest(refusal=name):
                arguments = dict(base, events={"bed_seeds": 5})
                arguments.update(change)
                random.seed(13)
                np.random.seed(13)
                before = global_states()
                with recorded_generators() as created:
                    with self.assertRaises(ValueError) as caught:
                        bed.build(arguments.pop("domain"), arguments.pop("settings"), **arguments)
                self.assertNotIsInstance(caught.exception, bed.SeedsNotPlaced)
                self.assertIn(words, str(caught.exception))
                self.assertEqual(seeds_of(created), [])
                assert_same_states(self, global_states(), before)
                if isinstance(arguments["events"], dict):
                    self.assertEqual(arguments["events"], {"bed_seeds": 5})

    def test_a_box_too_small_under_the_isotropic_fit_is_refused_after_the_grammar_and_before_the_bed(self):
        # the box is known only once the first tree's free extent is: here 7.7 d_min
        random.seed(3)
        np.random.seed(3)
        with recorded_generators() as created:
            with self.assertRaises(main.BedBoxTooSmall):
                main.grow_network(2, 2.5, None, (48, 48, 48), d_min=1.0, grow_in_volume=True, tortuosity="walk",
                                  persistence=10.0, seed=3, bed=FOAM)
        self.assertEqual([seed for seed in seeds_of(created) if len(seed) == 3], [])
        self.assertIn([3, 1], seeds_of(created))

    def test_the_foam_preset_is_as_approved(self):
        self.assertEqual(json.loads(json.dumps(main.FAMILIES["foam"])), FOAM_PRESET)
        self.assertEqual(list(main.FAMILIES["foam"]["bed"]), list(bed.SETTING_KEYS))


class CommandLineTests(unittest.TestCase):
    @staticmethod
    def run_main(argv):
        """main.main(argv) with its output captured; returns (status, SystemExit or None, stdout)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                return main.main(argv), None, out.getvalue()
            except SystemExit as caught:
                return None, caught, out.getvalue()

    def test_every_refusal_exits_before_anything_is_written(self):
        cases = [(name, cli_argv(**change)) for name, _, change, _ in REFUSALS]
        # the box too small under the isotropic fit is refused once the grammar has drawn, naming the seed
        isotropic = {"--fit": None, "--voxel-size": None, "--d0": ["2.5", "0"], "--iterations": ["2", "2"]}
        cases.append(("a box too small under the isotropic fit", cli_argv(**isotropic)))
        self.assertEqual(len(cases), len(REFUSALS) + 1)
        for name, argv in cases:
            with self.subTest(refusal=name), tempfile.TemporaryDirectory() as directory:
                out = os.path.join(directory, "out")
                status, caught, _ = self.run_main(argv + ["--out", out])
                self.assertIsNone(status)
                self.assertNotIn(caught.code, (0, None))
                self.assertTrue(str(caught.code).startswith("--bed"))          # refused as a bed run
                self.assertTrue(not os.path.exists(out) or os.listdir(out) == [])
                if name == "a box too small under the isotropic fit":
                    self.assertIn("holds no seed", str(caught.code))
                    self.assertIn("(seed 3)", str(caught.code))

    def test_a_partial_spec_is_recorded_normalised(self):
        spec = {"kind": "foam", "girth": 10}
        normalised = bed.parse_settings(spec)
        self.assertEqual(len(normalised), len(bed.SETTING_KEYS))
        with tempfile.TemporaryDirectory() as out:
            argv = cli_argv(**{"--bed": [json.dumps(spec)], "--d0": ["6", "0"], "--seed": ["2"]})
            status, caught, _ = self.run_main(argv + ["--out", out])
            self.assertIsNone(caught)
            self.assertEqual(status, 0)
            with open(os.path.join(out, "Lnet_i6_s2.json")) as handle:
                sidecar = json.load(handle)
            metadata = main.load_network(os.path.join(out, "Lnet_i6_s2.npz"))["metadata"]
        for record in (sidecar, metadata):
            self.assertEqual(record["bed"], normalised)
            self.assertEqual(list(record["bed"]), list(bed.SETTING_KEYS))
            self.assertEqual(record["rng_streams"]["bed"], [2, bed.RNG_TAG])
        # a preset's partial spec reaches a member's kwargs normalised (the
        # growth is stubbed: the kwargs are made before it)
        preset = dict(main.FAMILIES["foam"], bed=dict(spec))
        with mock.patch.dict(main.FAMILIES, {"foam": preset}), \
                mock.patch.object(main, "grow_network", return_value={}) as grow:
            member = library.grow_member("foam", 4.0, 1)
        self.assertEqual(member["kwargs"]["bed"], normalised)
        self.assertEqual(list(member["kwargs"]["bed"]), list(bed.SETTING_KEYS))
        self.assertEqual(grow.call_args.kwargs["bed"], normalised)
        self.assertEqual(preset["bed"], spec)                       # the preset itself is left alone

    def test_bed_null_turns_the_preset_s_bed_off(self):
        argv = ["--family", "foam", "--bed", "null", "--volume", "48", "48", "48", "--fit", "voxel_size",
                "--voxel-size", "1", "--iterations", "4", "4", "--d0", "6", "0", "--count", "1", "--seed", "3"]
        with tempfile.TemporaryDirectory() as out:
            status, caught, _ = self.run_main(argv + ["--out", out])
            self.assertIsNone(caught)
            self.assertEqual(status, 0)
            with open(os.path.join(out, "Lnet_i4_s3.json")) as handle:
                sidecar = json.load(handle)
        self.assertIsNone(sidecar["bed"])
        self.assertNotIn("bed", sidecar["rng_streams"])
        self.assertEqual(sidecar["trees"], 1)
        self.assertTrue(all(value == 0 for key, value in sidecar["events"].items() if key.startswith("bed_")))


# --------------------------------------------------------------------------
# Item 31: the seeds


class _CapReached(Exception):
    """The brute-force loop's refusal: (seeds accepted, proposals considered)."""


def brute_force_seeds(rng, low, high, *, spacing, count, stretch=1.0, axis=None, blocked=None, block, cap):
    """
    Random sequential addition one proposal at a time, from the same blocks
    rng.random((block, 3)) place_seeds draws: each block compressed with
    bed._compress, each proposal tested against the blocked points and then
    against the accepted seeds as a (1, 3) - (m, 3) array through bed._dist.
    Returns (seeds, outcomes); raises _CapReached at `cap` proposals.
    """
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    seeds, compressed, outcomes = [], np.empty((count, 3)), []
    considered = 0
    while len(seeds) < count:
        if considered >= cap:
            raise _CapReached(len(seeds), considered)
        X = low + (high - low) * rng.random((block, 3))
        C = bed._compress(X, axis, stretch)
        for k in range(block):
            if considered >= cap or len(seeds) == count:
                break
            considered += 1
            if blocked is not None and np.any(bed._dist(X[k:k + 1], blocked[0]) < blocked[1]):
                outcomes.append(1)
                continue
            m = len(seeds)
            if m and np.any(bed._dist(C[k:k + 1], compressed[:m]) < spacing):
                outcomes.append(2)
                continue
            outcomes.append(0)
            seeds.append(X[k])
            compressed[m] = C[k]
    return np.array(seeds, dtype=float).reshape(-1, 3), np.array(outcomes, dtype=np.int8)


def seed_grid():
    """
    The plan's grid: cubes of side 40, 55 and 75 d_min (35-480 seeds);
    stretch 1, and 2 along x and along (1, 1, 1)/sqrt 3; with and without a
    blocked straight polyline of diameter 4 along the box's diagonal.
    """
    line = np.linspace(0.0, 1.0, 200)[:, None]
    grid = []
    for side in (40.0, 55.0, 75.0):
        low, high = np.full(3, 0.5), np.full(3, side - 0.5)       # the domain inset by d/2, as build does
        for stretch, axis in ((1.0, None), (2.0, np.array([1.0, 0.0, 0.0])), (2.0, np.ones(3) / math.sqrt(3.0))):
            for with_line in (False, True):
                blocked = None
                if with_line:
                    points = low + (high - low) * line
                    blocked = (points, np.full(len(points), 2.0 + 0.5 + 1.0))     # r_f + d/2 + margin
                count = bed.planned_seeds((side,) * 3, {"stretch": stretch}, D_MIN)
                grid.append({"side": side, "stretch": stretch, "axis": axis, "blocked": blocked, "low": low,
                             "high": high, "count": count})
    return grid


class SeedTests(unittest.TestCase):
    """
    place_seeds against brute_force_seeds. The fast test runs the eighteen
    cubes of the grid once each, seeds 1-5 in turn; the slow one runs every
    cube on every seed 1-5 (the whole grid, about 20 s).
    """

    BLOCKS = (bed.RSA_BLOCK, 7, 64)

    def check_cube(self, cube, seed):
        reference = None
        for block in self.BLOCKS:
            rng = np.random.default_rng([seed, bed.RNG_TAG, 1])
            seeds, outcomes = bed.place_seeds(rng, cube["low"], cube["high"], spacing=H, count=cube["count"],
                                              stretch=cube["stretch"], axis=cube["axis"], blocked=cube["blocked"],
                                              block=block)
            loop = np.random.default_rng([seed, bed.RNG_TAG, 1])
            expected, expected_outcomes = brute_force_seeds(
                loop, cube["low"], cube["high"], spacing=H, count=cube["count"], stretch=cube["stretch"],
                axis=cube["axis"], blocked=cube["blocked"], block=block, cap=math.inf)
            self.assertEqual(seeds.tobytes(), expected.tobytes())
            np.testing.assert_array_equal(outcomes, expected_outcomes)
            self.assertEqual(outcomes.dtype, np.int8)
            self.assertEqual(len(outcomes), len(expected_outcomes))          # the proposals considered
            self.assertEqual(rng.bit_generator.state, loop.bit_generator.state)
            if reference is None:
                reference = (seeds.tobytes(), outcomes.tobytes())
            self.assertEqual((seeds.tobytes(), outcomes.tobytes()), reference)   # whatever the block
        # every pair a hard core apart in the compressed metric, every seed clear, the planned count
        self.assertEqual(len(seeds), cube["count"])
        self.assertGreater(len(seeds), 0)
        C = bed._compress(seeds, cube["axis"], cube["stretch"])
        i, j = np.triu_indices(len(C), 1)
        self.assertTrue(np.all(bed._dist(C[i], C[j]) >= H))
        if cube["blocked"] is not None:
            points, radii = cube["blocked"]
            for x in seeds:
                self.assertTrue(np.all(bed._dist(x[None, :], points) >= radii))
            self.assertGreater(int(np.count_nonzero(outcomes == 1)), 0)
        self.assertGreater(int(np.count_nonzero(outcomes == 2)), 0)

    def test_the_vectorised_seeds_equal_a_loop_over_the_proposals(self):
        grid = seed_grid()
        self.assertEqual(len(grid), 18)
        for k, cube in enumerate(grid):
            seed = 1 + k % 5
            with self.subTest(side=cube["side"], stretch=cube["stretch"], axis=cube["axis"],
                              blocked=cube["blocked"] is not None, seed=seed):
                self.check_cube(cube, seed)

    @SLOW
    def test_the_vectorised_seeds_equal_a_loop_over_the_whole_grid(self):
        for cube in seed_grid():
            for seed in range(1, 6):
                with self.subTest(side=cube["side"], stretch=cube["stretch"], axis=cube["axis"],
                                  blocked=cube["blocked"] is not None, seed=seed):
                    self.check_cube(cube, seed)

    def test_a_target_above_jamming_is_refused_after_exactly_the_proposal_cap(self):
        side = 33.0
        cap = math.ceil(64.0 * side ** 3 / H ** 3)            # ceil(64 V_c / h^3) = 5452, not a whole block
        self.assertEqual(cap, 5452)
        low, high = np.zeros(3), np.full(3, side)
        rng = np.random.default_rng([2, bed.RNG_TAG, 1])
        with self.assertRaises(bed.SeedsNotPlaced) as caught:
            bed.place_seeds(rng, low, high, spacing=H, count=200)
        found = re.search(r"only (\d+) of 200 seeds within (\d+) proposals", str(caught.exception))
        self.assertIsNotNone(found)
        self.assertEqual(int(found.group(2)), cap)
        with self.assertRaises(_CapReached) as reached:
            brute_force_seeds(np.random.default_rng([2, bed.RNG_TAG, 1]), low, high, spacing=H, count=200,
                              block=bed.RSA_BLOCK, cap=cap)
        self.assertEqual(reached.exception.args, (int(found.group(1)), cap))
        fresh = np.random.default_rng([2, bed.RNG_TAG, 1])
        for _ in range(math.ceil(cap / bed.RSA_BLOCK)):
            fresh.random((bed.RSA_BLOCK, 3))
        self.assertEqual(rng.bit_generator.state, fresh.bit_generator.state)

    def test_a_count_of_zero_places_and_draws_nothing(self):
        for block in (bed.RSA_BLOCK, 7):
            rng = np.random.default_rng([1, bed.RNG_TAG, 1])
            seeds, outcomes = bed.place_seeds(rng, np.zeros(3), np.full(3, 30.0), spacing=H, count=0, block=block)
            self.assertEqual((seeds.shape, outcomes.shape), ((0, 3), (0,)))
            self.assertEqual(rng.bit_generator.state, np.random.default_rng([1, bed.RNG_TAG, 1]).bit_generator.state)
            loop = np.random.default_rng([1, bed.RNG_TAG, 1])
            seeds, outcomes = brute_force_seeds(loop, np.zeros(3), np.full(3, 30.0), spacing=H, count=0, block=block,
                                                cap=math.inf)
            self.assertEqual((seeds.shape, outcomes.shape), ((0, 3), (0,)))
            self.assertEqual(loop.bit_generator.state, np.random.default_rng([1, bed.RNG_TAG, 1]).bit_generator.state)

    def test_seeds_that_run_out_are_refused_once_the_seed_stream_has_drawn(self):
        # a feeder of diameter 29 along the middle of a 25 cube leaves too little room for its 16 seeds
        t = np.linspace(0.0, 1.0, 71)[:, None]
        points = np.array([-5.0, 12.5, 12.5]) + t * np.array([35.0, 0.0, 0.0])
        nodes = np.vstack([points.T, np.full(71, 29.0)])
        feeders = (nodes, np.zeros(71, dtype=np.int8))
        events = {"bed_seeds": 5}
        random.seed(4)
        np.random.seed(4)
        before = global_states()
        with recorded_generators() as created:
            with self.assertRaises(bed.SeedsNotPlaced):
                bed.build((np.zeros(3), np.full(3, 25.0)), FOAM, d_min=D_MIN, seed=4, collision_margin=MARGIN,
                          feeders=feeders, events=events)
        self.assertEqual(seeds_of(created), [[4, bed.RNG_TAG, 1]])
        self.assertEqual(events, {"bed_seeds": 5})
        assert_same_states(self, global_states(), before)
        # the command line names the seed
        refusal = bed.SeedsNotPlaced("the bed placed only 3 of 9 seeds within 99 proposals")
        with tempfile.TemporaryDirectory() as out, mock.patch.object(main, "build_bed", side_effect=refusal):
            argv = cli_argv(**{"--volume": ["40", "40", "40"], "--d0": ["5", "0"], "--iterations": ["4", "4"],
                               "--seed": ["5"]})
            status, caught, _ = CommandLineTests.run_main(argv + ["--out", out])
            self.assertIsNone(status)
            self.assertIn("placed only 3 of 9 seeds", str(caught.code))
            self.assertIn("(seed 5)", str(caught.code))
            self.assertEqual(os.listdir(out), [])

    def test_build_places_its_seeds_in_the_domain_inset_by_half_a_diameter(self):
        # the corners build hands place_seeds, on a rebuilt bed of 70 seeds
        case = alone(1, side=40.0)
        low, high = (np.asarray(corner, dtype=float) for corner in case["call"][0][0])
        d = case["settings"]["diameter"] * case["d_min"]
        with mock.patch.object(bed, "place_seeds", wraps=bed.place_seeds) as placed:
            result, _ = rebuild(case)
        self.assertEqual(result["seeds"].tobytes(), case["result"]["seeds"].tobytes())
        (_, inset_low, inset_high), _ = placed.call_args
        self.assertEqual(np.asarray(inset_low).tobytes(), (low + d / 2.0).tobytes())
        self.assertEqual(np.asarray(inset_high).tobytes(), (high - d / 2.0).tobytes())
        # and every seed of the grown beds lies inside it
        for case in (alone(1), foam(1), alone(1, stretch=2.0, side=48.0)):
            with self.subTest(build=case["label"]):
                low, high = (np.asarray(corner, dtype=float) for corner in case["call"][0][0])
                d = case["settings"]["diameter"] * case["d_min"]
                seeds = case["result"]["seeds"]
                self.assertGreater(len(seeds), 0)
                self.assertTrue(np.all(seeds >= low + d / 2.0))
                self.assertTrue(np.all(seeds < high - d / 2.0))


# --------------------------------------------------------------------------
# Item 32: the rules


NO_REFUSAL = {"degree": 0, "angle": 0, "girth": 0, "clearance": 0, "at_end": 0}


def h_away(degrees):
    """The point h from the origin in the xy plane, `degrees` from +x."""
    return [H * math.cos(math.radians(degrees)), H * math.sin(math.radians(degrees)), 0.0]


def greedy(points, pairs, inward=None, caps=None):
    """
    bed._greedy over the candidate pairs (i < j) in the order given, at the
    foam settings (girth 8, min_angle 60, T = d + margin = 2) and with no
    feeder obstacles; `inward` maps a tip vertex to its feeder's inward
    direction. Returns (the accepted edges as a list, the refusal counts).
    """
    points = np.asarray(points, dtype=float)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    caps = [3] * len(points) if caps is None else caps
    out = bed._greedy(points, pairs[:, 0], pairs[:, 1], np.arange(len(pairs)), caps=caps, inward=inward or {},
                      adjacency=[[] for _ in range(len(points))], girth=FOAM["girth"], min_angle=FOAM["min_angle"],
                      clearance=D_MIN * FOAM["diameter"] + MARGIN, step=FOAM["step"])
    return out["edges"].tolist(), out["counts"]


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [alone(1), foam(1)]

    def test_the_rules_hold_on_the_bed_alone_and_on_foam(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                counts = check_rules(self, case)
                self.assertGreater(counts["pairs"], 0)
                self.assertLess(counts["girth"], math.inf)
                if case["feeders"] is not None:
                    self.assertGreater(counts["tip pairs"], 0)
                    self.assertGreater(counts["pairs at a tip"], 0)         # a tip with two bed edges
                    # a tip edge closer to the feeder than KIN_ANGLE_DEG is
                    # kept (seed 1: at 65.0 and 66.5 degrees from the inward
                    # direction): only min_angle bounds it
                    self.assertGreater(counts["tip edges inside the kin angle"], 0)

    def test_the_rules_hold_on_the_constructed_beds(self):
        for name in CONSTRUCTIONS + ("island",):
            with self.subTest(build=name):
                case = constructed(name)
                counts = check_rules(self, case)
                self.assertEqual(counts["girth"], 8)                         # the octagons, exactly the girth
                self.assertGreater(counts["pairs"], 0)
                if case["feeders"] is not None:
                    self.assertGreater(counts["tip pairs"], 0)
        # the island pass bites: the far octagon goes, once the peel has taken its tail
        events = constructed("island")["events"]
        self.assertEqual((events["bed_seeds_dead_end"], events["bed_edges_dead_end"]), (1, 1))
        self.assertEqual((events["bed_seeds_island"], events["bed_edges_island"]), (8, 8))
        kept = constructed("island")["result"]["kept"]
        self.assertEqual(kept.tolist(), [True] * 7 + [False] * 9)          # the tip lasso's seven seeds stay

    @SLOW
    def test_the_rules_hold_on_seeds_two_to_ten(self):
        for seed in LATER_SEEDS:
            for case in (alone(seed), foam(seed)):
                with self.subTest(build=case["label"]):
                    counts = check_rules(self, case)
                    self.assertGreater(counts["pairs"], 0)

    def test_a_tip_edge_stands_min_angle_from_its_feeder_and_the_kin_angle_only_from_another_bed_edge(self):
        # a tip at the origin whose feeder comes in along -x, so that its
        # inward direction is +x; seeds h from it in the xy plane
        tip, inward = [0.0, 0.0, 0.0], np.array([1.0, 0.0, 0.0])
        self.assertTrue(60.0 < 65.0 < bed.KIN_ANGLE_DEG)
        # 65 degrees from the inward direction, inside the kin angle: accepted
        edges, counts = greedy([h_away(65.0), tip], [(0, 1)], inward={1: inward})
        self.assertEqual(edges, [[0, 1]])
        self.assertEqual(counts, dict(NO_REFUSAL))
        # 55 degrees: refused by the angle
        edges, counts = greedy([h_away(55.0), tip], [(0, 1)], inward={1: inward})
        self.assertEqual(edges, [])
        self.assertEqual(counts, dict(NO_REFUSAL, angle=1))
        # a second bed edge 70 degrees from the first (135 from the inward
        # direction): beyond min_angle, inside the kin angle, refused at the
        # shared end
        edges, counts = greedy([h_away(65.0), h_away(135.0), tip], [(0, 2), (1, 2)], inward={2: inward},
                               caps=[3, 3, 2])
        self.assertEqual(edges, [[0, 2]])
        self.assertEqual(counts, dict(NO_REFUSAL, clearance=1, at_end=1))

    def test_two_bed_edges_inside_the_kin_angle_are_refused_as_clearance_at_their_shared_end(self):
        # seed 0 at the origin, seeds 1 and 2 h from it at 0 and 65 degrees,
        # and only the two candidates from seed 0
        edges, counts = greedy([[0.0, 0.0, 0.0], h_away(0.0), h_away(65.0)], [(0, 1), (0, 2)])
        self.assertEqual(edges, [[0, 1]])
        self.assertEqual(counts, dict(NO_REFUSAL, clearance=1, at_end=1))
        # and the counter counts on the grown beds (seed 1: 153 alone, 127 on foam)
        for case in (alone(1), foam(1)):
            with self.subTest(build=case["label"]):
                self.assertGreater(case["events"]["bed_refused_clearance_at_end"], 0)

    def test_an_angle_of_exactly_min_angle_is_accepted(self):
        # the tip at the origin with inward direction +x and a seed at
        # (h, b, 0): b is stepped an ulp at a time from h tan 60 degrees until
        # the bed edge's unit x component at the tip, as _greedy computes it,
        # is exactly cos 60 degrees; the edge is then accepted, and refused
        # once b is the ulp below that brings the angle under min_angle
        cos_min = math.cos(math.radians(60.0))

        def at_tip(b):
            unit, _ = bed._unit_rows(np.array([[0.0, 0.0, 0.0]]) - np.array([[H, b, 0.0]]))
            return -float(unit[0, 0])

        tie = None
        for direction in (-math.inf, math.inf):
            b = H * math.tan(math.radians(60.0))
            for _ in range(5000):
                if at_tip(b) == cos_min:
                    tie = b
                    break
                b = math.nextafter(b, direction)
            if tie is not None:
                break
        self.assertIsNotNone(tie)
        inward = {1: np.array([1.0, 0.0, 0.0])}
        edges, counts = greedy([[H, tie, 0.0], [0.0, 0.0, 0.0]], [(0, 1)], inward=inward)
        self.assertEqual(edges, [[0, 1]])
        self.assertEqual(counts, dict(NO_REFUSAL))
        inside = tie
        while at_tip(inside) <= cos_min:
            inside = math.nextafter(inside, -math.inf)
        self.assertLess(tie - inside, 1e-12)
        edges, counts = greedy([[H, inside, 0.0], [0.0, 0.0, 0.0]], [(0, 1)], inward=inward)
        self.assertEqual(edges, [])
        self.assertEqual(counts, dict(NO_REFUSAL, angle=1))

    def test_a_chain_that_returns_to_its_junction_is_walked(self):
        # on the abstract graph: a ring of eight seeds hanging from J = 0 by a stem to a tip
        ring = [(k, (k + 1) % 8) for k in range(8)]
        edges = np.array(ring + [(0, 8), (8, 9)], dtype=np.int64)
        is_tip = np.zeros(10, dtype=bool)
        is_tip[9] = True
        chains = bed._chains(10, edges, is_tip)
        closed = [chain for chain in chains if chain[0] == chain[-1]]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0][0], 0)
        self.assertEqual(sorted(closed[0][:-1]), list(range(8)))
        self.assertTrue([0, 8, 9] in chains or [9, 8, 0] in chains)
        # grown: two lassos joined by a path, and a lasso through a tip
        for name, junctions in (("lassos", (0, 10)), ("tip lasso", (7,))):
            with self.subTest(build=name):
                case = constructed(name)
                result, events = case["result"], case["events"]
                n = len(result["seeds"])
                is_tip = np.zeros(n + len(result["tips"]), dtype=bool)
                is_tip[n:] = True
                lassos = [chain for chain in bed._chains(len(is_tip), result["edges"], is_tip)
                          if chain[0] == chain[-1]]
                self.assertEqual(sorted(chain[0] for chain in lassos), list(junctions))     # [J, ..., J]
                self.assertTrue(all(len(chain) == 9 for chain in lassos))
                records = [chain for chain in result["chains"] if chain["closed"]]
                self.assertEqual(len(records), len(junctions))
                for record in records:
                    # written from its middle seed, so that J is an interior column (bed.py's docstring)
                    vertices = record["vertices"]
                    self.assertEqual(vertices[0], vertices[-1])
                    self.assertEqual(len(vertices), 9)
                    self.assertIn(vertices[len(vertices) - 1 - len(vertices) // 2], junctions)
                    self.assertIn(record["outcome"], ("walked", "chord"))
                self.assertEqual(events["bed_segments_dropped"], 0)
                self.assertEqual(describe.clearance(result["nodes"], MARGIN)["violations"], 0)
                if name == "tip lasso":
                    self.assertEqual(result["tip_edges"].tolist(), [2])
                    self.assertEqual(int(graph.build(result["nodes"])["degree"][result["tips"][0]]), 3)

    def test_a_tip_with_two_bed_edges_ends_two_chains(self):
        case = constructed("two tips")
        result = case["result"]
        n = len(result["seeds"])
        self.assertEqual(result["tip_edges"].tolist(), [2, 2])
        degree = graph.build(result["nodes"])["degree"]
        for t, tip in enumerate(result["tips"].tolist()):
            with self.subTest(tip=tip):
                ending = [chain for chain in result["chains"] if n + t in (chain["vertices"][0], chain["vertices"][-1])]
                self.assertEqual(len(ending), 2)
                for chain in ending:
                    self.assertIn(chain["outcome"], ("walked", "chord"))
                    self.assertFalse(chain["closed"])
                    ends = (result["nodes"][:3, chain["first"]].tobytes(), result["nodes"][:3, chain["last"]].tobytes())
                    self.assertIn(result["nodes"][:3, tip].tobytes(), ends)
                self.assertEqual(int(degree[tip]), 3)
        self.assertEqual(case["events"]["bed_segments_dropped"], 0)


# --------------------------------------------------------------------------
# Item 33: columns, tips and clearance


def vee_archive(polylines):
    """An archive of the given (points, diameter) polylines, each after a NaN separator."""
    blocks = []
    for points, diameter in polylines:
        rows = np.empty((4, len(points)))
        rows[:3] = np.asarray(points).T
        rows[3] = diameter
        blocks += [np.full((4, 1), np.nan), rows]
    return np.concatenate(blocks, axis=1)


class ColumnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [alone(1), foam(1)] + [constructed(name) for name in CONSTRUCTIONS + ("island",)]

    def test_ends_and_interior_seeds_are_bitwise_and_the_columns_are_laid_out(self):
        closed = attached = 0
        for case in self.cases:
            with self.subTest(build=case["label"]):
                counts = check_columns(self, case)
                closed += counts["closed"]
                attached += counts["tips with two bed edges"]
        self.assertGreater(closed, 0)
        self.assertGreater(attached, 0)
        self.assertEqual(library.root_columns(foam(1)["result"]["tree"]), library.root_columns(foam(1)["feeders"][1]))
        self.assertEqual(len(library.root_columns(foam(1)["result"]["tree"])), 2)

    def test_no_clearance_is_violated(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                self.assertEqual(describe.clearance(case["result"]["nodes"], case["margin"])["violations"], 0)

    def test_the_fallback_is_the_checked_chord(self):
        # seed 1 places no chord of its own (see the module's docstring), so
        # every walk of the foam bed is refused and every chain falls back
        case = chords_of(foam(1))
        result = case["result"]
        self.assertEqual(case["events"]["bed_segments_chord"], len(result["chains"]))
        self.assertGreater(len(result["chains"]), 0)
        self.assertEqual(check_fallbacks(self, case), len(result["chains"]))
        n = len(result["seeds"])
        self.assertTrue(any(max(chain["vertices"][0], chain["vertices"][-1]) >= n
                            for chain in result["chains"]))                  # chords to tips among them
        check_columns(self, case)
        self.assertEqual(describe.clearance(result["nodes"], case["margin"])["violations"], 0)
        # and on the lassos, written from their middle seeds
        lassos = constructed("lassos", refuse_walks=True)
        self.assertEqual(lassos["events"]["bed_segments_chord"], 3)
        self.assertEqual(check_fallbacks(self, lassos), 3)
        check_columns(self, lassos)
        self.assertEqual(describe.clearance(lassos["result"]["nodes"], MARGIN)["violations"], 0)
        for case in (alone(1), foam(1)):
            self.assertEqual(check_fallbacks(self, case), case["events"]["bed_segments_chord"])

    def test_the_eligible_tips_are_those_anastomosis_finds(self):
        # anastomose on the same feeders, bridging none: the tips it asks
        # about (degree one, not a root) and those inside a junction's zone
        stubs_found = 0
        for case in (foam(1),) + tuple(constructed(name) for name in ("lassos to a tip", "tip lasso", "two tips")):
            with self.subTest(build=case["label"]):
                nodes, tree = case["feeders"]
                asked = []
                real = anastomosis._Network.inside_junction

                def inside_junction(network, column, margin):
                    stub = real(network, column, margin)
                    asked.append((int(column), bool(stub)))
                    return stub

                events = {}
                with mock.patch.object(anastomosis._Network, "inside_junction", inside_junction):
                    anastomosis.anastomose(nodes.copy(), np.random.default_rng(0), 0.0, 1.0, tree=tree.copy(),
                                           collision_margin=case["margin"], events=events)
                self.assertGreater(len(asked), 0)
                tips = sorted(column for column, stub in asked if not stub)
                stubs = sorted(column for column, stub in asked if stub)
                self.assertEqual(case["result"]["tips"].tolist(), tips)
                self.assertEqual(case["events"]["bed_tips"], events["anastomosis_tips"])
                self.assertEqual(case["events"]["bed_tips_inside_junction"], events["anastomosis_tips_inside_junction"])
                self.assertEqual((len(tips) + len(stubs), len(stubs)),
                                 (events["anastomosis_tips"], events["anastomosis_tips_inside_junction"]))
                self.assertEqual(set(tips) & set(library.root_columns(tree)), set())
                stubs_found += len(stubs)
        self.assertGreater(stubs_found, 0)                      # foam seed 1 has two stubs

    def test_the_walk_leaves_each_interior_seed_along_its_catmull_rom_tangent(self):
        checked = 0
        for name in ("foam", "lassos"):
            with self.subTest(build=name):
                with walk_tangents() as calls:
                    if name == "foam":
                        case = foam(1)
                        result, _ = rebuild(case)
                        self.assertEqual(result["nodes"].tobytes(), case["result"]["nodes"].tobytes())
                        case = dict(case, result=result)
                    else:
                        case = _construct("lassos", 1, False, None)
                self.assertGreater(len(calls), 0)
                checked += check_tangents(self, case, calls)
                if name == "lassos":
                    self.assertEqual(sum(chain["closed"] for chain in case["result"]["chains"]), 2)
        self.assertGreater(checked, 0)

    @SLOW
    def test_the_columns_hold_on_seeds_two_to_ten(self):
        chords = 0
        for seed in LATER_SEEDS:
            for case in (alone(seed), foam(seed)):
                with self.subTest(build=case["label"]):
                    check_columns(self, case)
                    self.assertEqual(describe.clearance(case["result"]["nodes"], case["margin"])["violations"], 0)
                    found = check_fallbacks(self, case)
                    self.assertEqual(found, case["events"]["bed_segments_chord"])
                    chords += found
        self.assertGreater(chords, 0)

    def test_the_bed_check_agrees_with_describe(self):
        d, step = 1.0, 0.5
        J = np.array([10.0, 10.0, 10.0])

        def towards(degrees):
            return np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees)), 0.0])

        verdicts = []
        # vees: chord A from J along x, chord B from J at 60-80 degrees to it, B written from J or to J
        for length in (7.5, 12.0, 18.0):
            for degrees in range(60, 81):
                for to_J in (False, True):
                    a = bridge_path(J, J + length * towards(0.0), step)
                    b = bridge_path(J, J + length * towards(degrees), step)
                    if to_J:
                        b = b[::-1].copy()
                    obstacles = bed._Obstacles(np.empty((4, 0)), d / 2.0, MARGIN)
                    obstacles.place(0, a, {100: 0})
                    mine = bed._collides(obstacles, b, 1, {100: len(b) - 1 if to_J else 0}, False)
                    theirs = describe.clearance(vee_archive([(a, d), (b, d)]), MARGIN)["violations"] > 0
                    verdicts.append((("vee", length, degrees, to_J), mine, theirs))
        # a symmetric Y
        for length in (7.5, 12.0):
            paths = [bridge_path(J, J + length * towards(120.0 * k), step) for k in range(3)]
            obstacles = bed._Obstacles(np.empty((4, 0)), d / 2.0, MARGIN)
            obstacles.place(0, paths[0], {100: 0})
            obstacles.place(1, paths[1], {100: 0})
            mine = bed._collides(obstacles, paths[2], 2, {100: 0}, False)
            theirs = describe.clearance(vee_archive([(path, d) for path in paths]), MARGIN)["violations"] > 0
            verdicts.append((("Y", length), mine, theirs))
        # chords from the tip of a feeder of diameter 2 that runs in along +x, at +-theta to +x
        tip = np.array([30.0, 20.0, 10.0])
        feeder = np.stack([tip + np.array([-0.5 * (40 - k), 0.0, 0.0]) for k in range(41)])
        feeder_nodes = vee_archive([(feeder, 2.0)])[:, 1:]
        for length in (7.5, 12.0):
            for degrees in (40.0, 60.0, 75.0):
                for k in (1, 2):
                    angles = [degrees] if k == 1 else [degrees, -degrees]
                    chords = [bridge_path(tip, tip + length * towards(angle), step) for angle in angles]
                    obstacles = bed._Obstacles(feeder_nodes, d / 2.0, MARGIN, tip_columns={7: 40})
                    if k == 2:
                        obstacles.place(0, chords[0], {7: 0})
                    mine = bed._collides(obstacles, chords[-1], k - 1, {7: 0}, False)
                    archive = np.concatenate([feeder_nodes, vee_archive([(chord, d) for chord in chords])], axis=1)
                    theirs = describe.clearance(archive, MARGIN)["violations"] > 0
                    verdicts.append((("tip", length, degrees, k), mine, theirs))
        colliding = sum(theirs for _, _, theirs in verdicts)
        self.assertGreater(colliding, 0)
        self.assertLess(colliding, len(verdicts))
        for construction, mine, theirs in verdicts:
            with self.subTest(construction=construction):
                self.assertEqual(mine, theirs)


# --------------------------------------------------------------------------
# Item 34: events and streams


@contextlib.contextmanager
def swapped_stream(k):
    """bed._stream with sub-stream k replaced by default_rng([99, 7, k])."""
    real = bed._stream

    def stream(seed, j):
        if j == k:
            return np.random.default_rng([99, bed.RNG_TAG, j])
        return real(seed, j)

    with mock.patch.object(bed, "_stream", side_effect=stream):
        yield


def rebuild(case):
    """bed.build again on a case's arguments, with fresh events."""
    args, kwargs = case["call"]
    events = {}
    result = bed.build(*args, events=events, **kwargs)
    return result, events


class StreamTests(unittest.TestCase):
    VOXEL = dict(fit="voxel_size", voxel_size=1.0)

    @classmethod
    def setUpClass(cls):
        cls.cases = ([alone(1), foam(1), chords_of(foam(1))] + [constructed(name) for name in CONSTRUCTIONS]
                     + [constructed("island"), constructed("lassos", refuse_walks=True)]
                     + [constructed(name, drop=drop) for name, drop, _ in DROPS])

    def test_the_counters_obey_their_laws(self):
        for case in self.cases:
            with self.subTest(build=case["label"]):
                check_identities(self, case)
        # every term of E5-E7 is met: the drops, the peel and the island pass
        for key in ("bed_segments_dropped", "bed_segments_removed", "bed_edges_dropped", "bed_edges_dead_end",
                    "bed_seeds_dead_end", "bed_edges_island", "bed_seeds_island", "bed_segments_chord"):
            with self.subTest(counter=key):
                self.assertGreater(max(case["events"][key] for case in self.cases), 0)

    def test_a_dropped_chain_is_peeled_again_and_its_counters_close(self):
        # bed._collides refuses one chain's every walk and its chord (DROPS)
        for name, drop, expected in DROPS:
            case = constructed(name, drop=drop)
            with self.subTest(build=case["label"]):
                ev, result = case["events"], case["result"]
                dropped, removed, edges, dead_end, island, placed, kept = expected
                self.assertEqual((ev["bed_segments_dropped"], ev["bed_segments_removed"], ev["bed_edges_dropped"]),
                                 (dropped, removed, edges))
                self.assertEqual((ev["bed_seeds_dead_end"], ev["bed_edges_dead_end"]), dead_end)
                self.assertEqual((ev["bed_seeds_island"], ev["bed_edges_island"]), island)
                self.assertEqual(len(result["chains"]), placed)
                self.assertEqual(ev["bed_segments_walked"] + ev["bed_segments_chord"], placed + removed)
                self.assertEqual(result["kept"].tolist(), kept)
                check_identities(self, case)
                check_rules(self, case)
                check_columns(self, case)
                self.assertEqual(describe.clearance(result["nodes"], MARGIN)["violations"], 0)
                # no chain left in the record uses a dropped or removed edge
                kept_edges = {tuple(edge) for edge in result["edges"].tolist()}
                for chain in result["chains"]:
                    for a, b in zip(chain["vertices"][:-1], chain["vertices"][1:]):
                        self.assertIn((min(a, b), max(a, b)), kept_edges)

    def test_the_components_and_those_touching_one_tree_are_counted(self):
        # (bed_components, bed_components_one_tree): a bed alone touches no
        # tree; the two tips' octagon touches trees 0 and 1
        expected = {"lassos": (1, 0), "lassos to a tip": (1, 1), "tip lasso": (1, 1), "two tips": (1, 0),
                    "island": (1, 1)}
        for name, pair in expected.items():
            with self.subTest(build=name):
                events = constructed(name)["events"]
                self.assertEqual((events["bed_components"], events["bed_components_one_tree"]), pair)
        # the lassos less their path are two components
        events = constructed("lassos", drop=DROPS[0][1])["events"]
        self.assertEqual((events["bed_components"], events["bed_components_one_tree"]), (2, 0))

    @SLOW
    def test_the_counters_obey_their_laws_on_seeds_two_to_ten(self):
        for seed in LATER_SEEDS:
            for case in (alone(seed), foam(seed)):
                with self.subTest(build=case["label"]):
                    check_identities(self, case)

    def test_the_walks_draw_from_their_own_sub_stream(self):
        # swap k = 3 on the foam bed of seed 1, which drops no chain either way
        case = foam(1)
        base = case["result"]
        self.assertEqual(case["events"]["bed_segments_dropped"], 0)
        with swapped_stream(3):
            result, events = rebuild(case)
        self.assertEqual(events["bed_segments_dropped"], 0)
        for key in ("seeds", "greedy_edges", "kept", "edges", "tips", "tip_edges"):
            self.assertEqual(result[key].tobytes(), base[key].tobytes(), key)
            self.assertEqual(result[key].shape, base[key].shape, key)
        for key in bed.EVENT_KEYS[:bed.EVENT_KEYS.index("bed_edges") + 1]:
            self.assertEqual(events[key], case["events"][key], key)
        n_feeders = case["feeders"][0].shape[1]
        self.assertEqual(result["nodes"][:, :n_feeders].tobytes(), base["nodes"][:, :n_feeders].tobytes())
        self.assertNotEqual(result["nodes"][:, n_feeders:].tobytes(), base["nodes"][:, n_feeders:].tobytes())

    def test_the_order_and_the_seeds_draw_from_their_own_sub_streams(self):
        case = alone(1, side=40.0)                           # 70 seeds
        base = case["result"]
        self.assertGreater(len(base["greedy_edges"]), 0)
        with swapped_stream(2):
            result, _ = rebuild(case)
        self.assertEqual(result["seeds"].tobytes(), base["seeds"].tobytes())
        self.assertFalse(np.array_equal(result["greedy_edges"], base["greedy_edges"]))       # the jitter is applied
        with swapped_stream(1):
            result, _ = rebuild(case)
        self.assertFalse(np.array_equal(result["seeds"], base["seeds"]))

    def test_a_bed_draws_nothing_global_beyond_the_feeders_grammar_and_creates_its_sub_streams(self):
        # under fit voxel_size, the foam member: its growth after its seeding and parameter sampling
        case = foam(1)
        member = case["member"]
        self.assertEqual(case["generators"], [[1, 1], [1, 7, 1], [1, 7, 2], [1, 7, 3]])
        args = main.build_parser("foam").parse_args(member["argv"])
        random.seed(1)
        np.random.seed(1)
        properties, d0, niter = main.sample_parameters(args)
        self.assertEqual((properties, d0, niter), (member["properties"], member["d0"], member["niter"]))
        libGenerator.setProperties(properties)
        F(niter, d0, 2.0 * D_MIN)
        F(niter, d0, 2.0 * D_MIN)
        assert_same_states(self, case["states"], global_states())
        # under the isotropic fit, a pair in the first tree's box of side 32.5 (37 seeds)
        random.seed(1)
        np.random.seed(1)
        with recorded_generators() as created:
            grown = main.grow_network(5, 4.0, None, (48, 48, 48), d_min=D_MIN, grow_in_volume=True,
                                      tortuosity="walk", persistence=10.0, avoid_collisions=True, seed=1, bed=FOAM)
        after = global_states()
        self.assertEqual(seeds_of(created), [[1, 1], [1, 1], [1, 7, 1], [1, 7, 2], [1, 7, 3]])
        self.assertGreater(grown["events"]["bed_segments_walked"], 0)
        random.seed(1)
        np.random.seed(1)
        libGenerator.setProperties(None)
        F(5, 4.0, 2.0 * D_MIN)
        F(5, 4.0, 2.0 * D_MIN)
        assert_same_states(self, after, global_states())

    def test_without_a_bed_nothing_of_the_bed_is_created_or_counted(self):
        random.seed(2)
        np.random.seed(2)
        with recorded_generators() as created:
            grown = main.grow_network(4, 6.0, None, (48, 48, 48), d_min=1.0, grow_in_volume=True, tortuosity="walk",
                                      persistence=10.0, avoid_collisions=True, anastomose=True,
                                      anastomose_mode="arteriovenous", seed=2, **self.VOXEL)
        self.assertEqual(seeds_of(created), [[2, 1], [2, 1], [2, 2]])
        counters = {key: value for key, value in grown["events"].items() if key.startswith("bed_")}
        self.assertEqual(sorted(counters), sorted(bed.EVENT_KEYS))
        self.assertTrue(all(value == 0 for value in counters.values()))


# --------------------------------------------------------------------------
# Item 35: the early return


def _function(path, name):
    with open(path, "rb") as handle:
        source = handle.read()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return source, tree, node


def _span(source, first, last):
    """The source bytes from the start of `first`'s line to the end of `last`."""
    lines = source.splitlines(keepends=True)
    text = b"".join(lines[first.lineno - 1:last.end_lineno - 1])
    return text + lines[last.end_lineno - 1][:last.end_col_offset]


def _named(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


class EarlyReturnTests(unittest.TestCase):
    """
    grow_network with a bed returns before anything 3.6 ran, so its released
    body runs unchanged without one. Phase 4 changes that body
    (anastomose(..., accept=)) and updates these tests against the 3.7
    reference then.
    """

    def test_the_body_after_the_bed_return_is_the_released_source(self):
        source, _, current = _function(os.path.join(ROOT, "main.py"), "grow_network")
        reference_source, _, reference = _function(REFERENCE_MAIN, "grow_network")
        bed_test = ast.dump(ast.parse("bed is not None", mode="eval").body)
        body = current.body
        # the docstring, setProperties, the bed block ending in its return, then the released body
        self.assertIsInstance(body[0], ast.Expr)
        self.assertIsInstance(body[0].value, ast.Constant)
        self.assertIsInstance(body[2], ast.If)
        self.assertEqual(ast.dump(body[2].test), bed_test)
        self.assertIsInstance(body[2].body[-1], ast.Return)
        self.assertEqual(body[2].orelse, [])
        self.assertEqual([k for k, node in enumerate(body) if isinstance(node, ast.If)
                          and ast.dump(node.test) == bed_test], [2])
        released = reference.body
        self.assertIsInstance(released[0], ast.Expr)
        self.assertEqual(_span(source, body[3], body[-1]), _span(reference_source, released[2], released[-1]))
        self.assertEqual(_span(source, body[1], body[1]), _span(reference_source, released[1], released[1]))
        self.assertIn(b"libGenerator.setProperties(properties)", _span(source, body[1], body[1]))

    def test_the_helpers_and_the_signature_are_released(self):
        _, tree, current = _function(os.path.join(ROOT, "main.py"), "grow_network")
        _, reference_tree, reference = _function(REFERENCE_MAIN, "grow_network")
        for name in ("_walk_settings", "_root_offsets", "_bind_guidance", "_free_extent", "_clear_root",
                     "RootOutsideBox"):
            with self.subTest(helper=name):
                self.assertEqual(ast.dump(_named(tree, name)), ast.dump(_named(reference_tree, name)))
        arguments = copy.deepcopy(current.args)
        self.assertEqual(arguments.args[-1].arg, "bed")
        self.assertIsInstance(arguments.defaults[-1], ast.Constant)
        self.assertIsNone(arguments.defaults[-1].value)
        arguments.args.pop()
        arguments.defaults.pop()
        self.assertEqual(ast.dump(arguments), ast.dump(reference.args))

    def test_the_bed_path_grows_the_feeders_the_regular_path_grows(self):
        def passthrough(domain, settings, **kwargs):
            nodes, tree = kwargs["feeders"]
            return {"nodes": nodes, "tree": tree, "frame": kwargs["frame"]}

        seed, delta = 6, 1.0
        for fit, extra, bed_streams in (("voxel_size", {"voxel_size": 1.0}, [[6, 1]]),
                                        ("isotropic", {}, [[6, 1], [6, 1]])):
            with self.subTest(fit=fit):
                common = dict(fit=fit, grow_in_volume=True, tortuosity="walk", persistence=10.0,
                              avoid_collisions=True, seed=seed, **extra)
                random.seed(seed)
                np.random.seed(seed)
                with recorded_generators() as created_bed, \
                        mock.patch.object(main, "build_bed", side_effect=passthrough):
                    a = main.grow_network(6, 5.0, None, (60, 60, 60), d_min=delta, bed=FOAM, **common)
                states_bed = global_states()
                random.seed(seed)
                np.random.seed(seed)
                with recorded_generators() as created_regular:
                    b = main.grow_network(6, 5.0, None, (60, 60, 60), d_min=2.0 * delta, anastomose=True,
                                          anastomose_mode="arteriovenous", anastomosis_fraction=0.0, **common)
                states_regular = global_states()
                self.assertEqual(a["nodes"].tobytes(), b["nodes"].tobytes())
                self.assertEqual(a["nodes"].shape, b["nodes"].shape)
                np.testing.assert_array_equal(a["tree"], b["tree"])
                self.assertEqual(a["programs"], b["programs"])
                self.assertEqual(a["root_positions_um"], b["root_positions_um"])
                self.assertEqual(a["growth_box_um"], b["growth_box_um"])
                self.assertEqual(a["frame"], b["frame"])
                assert_same_states(self, states_bed, states_regular)
                # the regular pair runs the free-extent pass under both fits, the bed path only when it sizes the box
                self.assertEqual(seeds_of(created_bed), bed_streams)
                self.assertEqual(seeds_of(created_regular), [[6, 1], [6, 1], [6, 2]])
                walk_bed = [generator for given, generator in created_bed if list(given) == [6, 1]][-1]
                walk_regular = [generator for given, generator in created_regular if list(given) == [6, 1]][-1]
                self.assertEqual(walk_bed.bit_generator.state, walk_regular.bit_generator.state)


# --------------------------------------------------------------------------
# Item 36: scale


class ScaleTests(unittest.TestCase):
    """
    Powers of two keep every comparison of bed.py exact (low + size u,
    h/2 + h/2, the sort key, the elementwise helpers, and step / d inside
    bridge_path), so the scaled network is the scaled copy within 1e-12
    relative. The feeders' growth rounds in its last bits under scaling,
    as every grown network does (tests/test_guidance.py), and the bed
    attached to their tips with them; a coordinate within a hair of a face
    then shows a large relative error, so the comparison has the absolute
    floor of tests/test_guidance.py, 1e-14 of scale x the box side. Over
    seeds 1-10 the largest difference was 2.8e-15 of scale x the box side
    for grow_network at scales 0.5 and 2 and 7.1e-16 for the member at
    d_min 2; away from the faces (beyond 1e-3 of the box side) the largest
    relative difference was 3.1e-13; the events were equal on every seed.
    """

    @staticmethod
    def grown(scale):
        random.seed(2)
        np.random.seed(2)
        return main.grow_network(6, 6.0 * scale, None, (48, 48, 48), fit="voxel_size", voxel_size=scale,
                                 d_min=scale, collision_margin=scale, grow_in_volume=True, tortuosity="walk",
                                 persistence=10.0, avoid_collisions=True, seed=2, bed=FOAM)

    @SLOW
    def test_foam_scales_with_d0_d_min_the_margin_and_the_voxel_size(self):
        reference = self.grown(1.0)
        self.assertGreater(reference["events"]["bed_segments_walked"], 0)
        floor = 1e-14 * max(reference["growth_box_um"])
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                scaled = self.grown(scale)
                self.assertEqual(scaled["nodes"].shape, reference["nodes"].shape)
                np.testing.assert_allclose(scaled["nodes"], scale * reference["nodes"], rtol=1e-12,
                                           atol=scale * floor)
                self.assertEqual(scaled["events"], reference["events"])

    @SLOW
    def test_a_foam_member_scales_with_the_unit(self):
        reference = foam(1)["member"]["grown"]
        twice = library.grow_member("foam", 4.0, 1, d_min=2.0, collision_margin=2.0)["grown"]
        self.assertEqual(twice["nodes"].shape, reference["nodes"].shape)
        floor = 1e-14 * max(reference["growth_box_um"])
        np.testing.assert_allclose(twice["nodes"], 2.0 * reference["nodes"], rtol=1e-12, atol=2.0 * floor)
        self.assertEqual(twice["events"], reference["events"])


# --------------------------------------------------------------------------
# Item 37: the frame and stretch


class FrameTests(unittest.TestCase):
    # a foam member of ratio 2.52 (59 seeds in a cube of side 37.8), grown
    # through the command line and the library writer in about two seconds
    RATIO = 2.52

    @classmethod
    def setUpClass(cls):
        cls.foam = foam(1)
        cls.stretched = alone(1, stretch=2.0, side=48.0)          # 61 seeds

    def test_foam_s_frame_is_the_unguided_record_alike_in_the_return_the_sidecar_and_the_metadata(self):
        frame = self.foam["member"]["grown"]["frame"]
        self.assertEqual((frame["kind"], frame["rules"], frame["origin"]), ("none", [], [30.0, 30.0, 30.0]))
        argv = ["--family", "foam"] + library.member_argv("foam", self.RATIO)[2:] + ["--count", "1", "--seed", "1"]
        with tempfile.TemporaryDirectory() as out:
            status, caught, _ = CommandLineTests.run_main(argv + ["--out", out])
            self.assertIsNone(caught)
            self.assertEqual(status, 0)
            stem = next(name[:-5] for name in os.listdir(out) if name.endswith(".json"))
            with open(os.path.join(out, stem + ".json")) as handle:
                sidecar = json.load(handle)
            archive = main.load_network(os.path.join(out, stem + ".npz"))
        # the library writer's member, its return kept by a spy on grow_member
        returned = []
        real = library.grow_member

        def spy(*args, **kwargs):
            returned.append(real(*args, **kwargs))
            return returned[-1]

        task = {"member": {"id": 0, "family": "foam", "family_index": 0, "position": 0, "seed": 1,
                           "ratio": self.RATIO, "bin": 0, "u": 0.0, "file": "net_00000_foam.npz"},
                "settings": {"d_min": 1.0, "collision_margin": 1.0, "mesh_box_c": 15.0, "iteration_cap": 64,
                             "avoid_collisions": True, "ratio_lo": self.RATIO, "ratio_hi": self.RATIO}}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(library, "grow_member", side_effect=spy):
            outcome = library.grow_and_write(dict(task, directory=directory))
            self.assertNotIn("failure", outcome)
            metadata = main.load_network(os.path.join(directory, "net_00000_foam.npz"))["metadata"]
        grown = returned[0]["grown"]
        self.assertEqual(archive["nodes"].tobytes(), grown["nodes"].tobytes())      # the same network
        frame = grown["frame"]
        self.assertEqual(frame["kind"], "none")
        self.assertEqual(frame["rules"], [])
        self.assertEqual(frame["origin"], [v / 2.0 for v in grown["growth_box_um"]])
        recorded = json.loads(json.dumps(frame))
        self.assertEqual(sidecar["frame"], recorded)
        self.assertEqual(archive["metadata"]["frame"], recorded)
        self.assertEqual(metadata["frame"], recorded)

    def test_a_stretched_bed_records_its_axis(self):
        frame = self.stretched["result"]["frame"]
        self.assertEqual(frame["kind"], "axis")
        self.assertEqual(frame["axis"], [0.0, 1.0, 0.0])
        self.assertEqual(frame["sense"], "nematic")
        self.assertIsNone(frame["tree_senses"])
        self.assertIsNone(frame["normal"])
        self.assertEqual(frame["rules"], [])
        Q = frames.align(frame, axis=[1, 0, 0])
        mapped = Q @ np.asarray(frame["axis"])
        self.assertLessEqual(abs(abs(float(mapped[0])) - 1.0), 1e-12)
        self.assertLessEqual(float(np.max(np.abs(mapped[1:]))), 1e-12)

    def test_a_stretch_without_a_frame_is_refused_before_any_draw(self):
        with recorded_generators() as created:
            with self.assertRaises(ValueError):
                bed.build((np.zeros(3), np.full(3, 80.0)), dict(FOAM, stretch=2.0), d_min=D_MIN, seed=1,
                          collision_margin=MARGIN, frame=None)
            with self.assertRaises(ValueError):
                bed.frame_record(None, {"stretch": 2.0})
        self.assertEqual(seeds_of(created), [])
        # at stretch 1 the record is the frame given, unchanged
        frame = unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR, (30.0, 30.0, 30.0))
        self.assertIs(bed.frame_record(frame, FOAM), frame)

    @SLOW
    def test_the_order_about_the_axis_rises_with_the_stretch(self):
        axis_frame = alone(1, stretch=2.0)["result"]["frame"]             # an axis frame about y
        for seed in range(1, 11):
            with self.subTest(seed=seed):
                order = {}
                for stretch in (1.0, 1.5, 2.0):
                    nodes = alone(seed, stretch)["result"]["nodes"]
                    report = describe.describe(nodes, frame=axis_frame, d_ref=D_MIN)
                    order[stretch] = report["frame_orientation"]["capillary"]["S"]
                self.assertGreaterEqual(order[1.5] - order[1.0], S_RISE_TO_1_5)
                self.assertGreaterEqual(order[2.0] - order[1.5], S_RISE_TO_2)


# --------------------------------------------------------------------------
# Item 38: the sizing guard


class GuardTests(unittest.TestCase):
    LIBRARY = ["--seed", "1", "--count", "1", "--families", "foam", "--workers", "1", "--ratio-range", "4", "4.2"]

    @classmethod
    def setUpClass(cls):
        cls.cases = [alone(1), foam(1), alone(1, stretch=2.0)]

    def test_the_planned_counts_are_the_closed_forms(self):
        # N_t = floor(density prod(extent - d) / stretch / h^3), and the
        # candidates ceil(N_t density (4 pi / 3) reach^3 / 2)
        for extents, settings, d_min, seeds in (((150.0, 150.0, 150.0), FOAM, 1.0, 3920),
                                                ((80.0, 80.0, 80.0), dict(FOAM, stretch=2.0), 1.0, 292),
                                                ((80.0, 80.0, 80.0), dict(FOAM, stretch=1.5), 1.0, 389),
                                                ((90.0, 60.0, 45.0), FOAM, 1.0, 273),
                                                ((160.0, 120.0, 90.0), dict(FOAM, stretch=2.0), 2.0, 121)):
            with self.subTest(extents=extents, stretch=settings["stretch"], d_min=d_min):
                h, d = settings["spacing"] * d_min, settings["diameter"] * d_min
                volume = (extents[0] - d) * (extents[1] - d) * (extents[2] - d)
                self.assertEqual(math.floor(0.5 * volume / settings["stretch"] / h ** 3), seeds)
                self.assertEqual(bed.planned_seeds(extents, settings, d_min), seeds)
                self.assertEqual(bed.planned_candidates(extents, settings, d_min),
                                 math.ceil(seeds * 0.5 * (4.0 * math.pi / 3.0) * 2.5 ** 3 / 2.0))

    def test_the_planned_candidates_bound_the_seed_pairs_built(self):
        # seed 1 built 0.616, 0.596 and 0.596 of the planned count (M6; the
        # ten seeds in the module's docstring)
        for case in self.cases:
            with self.subTest(build=case["label"]):
                planned = bed.planned_candidates(case["extents"], case["settings"], case["d_min"])
                built = case["events"]["bed_candidates"] - case["events"]["bed_candidates_tip"]
                self.assertGreater(built, 0)
                self.assertGreaterEqual(planned, built)
                self.assertLessEqual(built / planned, GUARD_RATIO)

    def test_a_member_over_the_cap_is_refused_before_anything_is_drawn(self):
        random.seed(17)
        np.random.seed(17)
        before = global_states()
        with recorded_generators() as created:
            with self.assertRaises(bed.BedTooLarge) as caught:
                library.grow_member("foam", 4.0, 1, bed_max_candidates=10)
        self.assertEqual(seeds_of(created), [])
        assert_same_states(self, global_states(), before)
        planned = bed.planned_candidates((60.0, 60.0, 60.0), FOAM, 1.0)        # the cube of side 15 x 4
        self.assertEqual(str(caught.exception), f"planned {planned} candidate edges exceed --bed-max-candidates "
                                                f"10; raise the cap or lower --box-c foam")

    def test_a_library_records_a_member_over_the_cap_as_a_failure_and_will_not_resume_with_another(self):
        members = library.plan_library(1, 1, ["foam"], [1.0], (4.0, 4.2))
        argv = library.member_argv("foam", members[0]["ratio"])
        args = main.build_parser("foam").parse_args(argv)
        planned = bed.planned_candidates(np.asarray(args.volume, dtype=float) * args.voxel_size,
                                         bed.parse_settings(args.bed), args.d_min)
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "lib")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = library.main(["--out", out, "--bed-max-candidates", "10"] + self.LIBRARY)
            self.assertEqual(status, 1)
            with open(os.path.join(out, "manifest.json")) as handle:
                content = json.load(handle)["content"]
            self.assertEqual(content["growth"]["bed_max_candidates"], 10)
            self.assertEqual(len(content["failures"]), 1)
            failure = {key: content["failures"][0][key] for key in ("id", "family", "seed", "ratio", "reason")}
            self.assertEqual(failure, {"id": 0, "family": "foam", "seed": members[0]["seed"],
                                       "ratio": members[0]["ratio"],
                                       "reason": f"BedTooLarge: planned {planned} candidate edges exceed "
                                                 f"--bed-max-candidates 10; raise the cap or lower --box-c foam"})
            self.assertEqual([name for name in os.listdir(out) if name.endswith(".npz")], [])
            # a resume with another cap is a different library
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    library.main(["--out", out, "--bed-max-candidates", "20"] + self.LIBRARY)
            self.assertIn("growth", str(caught.exception.code))

    def test_the_cap_is_a_whole_number_of_at_least_one(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                out = os.path.join(directory, "lib")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        library.main(["--out", out, f"--bed-max-candidates={value}"] + self.LIBRARY)
                self.assertNotIn(caught.exception.code, (0, None))
                self.assertFalse(os.path.exists(out))
        args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1",
                                                  "--bed-max-candidates", "2e6"])
        self.assertEqual(args.bed_max_candidates, 2000000)
        self.assertIs(type(args.bed_max_candidates), int)

    def test_a_library_without_a_bed_records_its_growth_as_before(self):
        for families in ([], ["--families", "tree", "mesh", "tumour"], ["--families", "capillary_bed"]):
            with self.subTest(families=families):
                args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1"] + families)
                settings = library.growth_settings(args)
                self.assertNotIn("bed_max_candidates", settings)
                if not families or families[1] == "tree":
                    self.assertEqual(settings, {"collision_margin": 1.0, "mesh_box_c": 15.0, "iteration_cap": 64,
                                                "avoid_collisions": True, "d_min": 1.0})


# --------------------------------------------------------------------------
# Item 39: foam end to end


class FoamFamilyTests(unittest.TestCase):
    """
    The share of seeds 1-10 whose bed has a component touching both feeder
    trees was 9 of 10 in development (all but seed 2); the connectivity
    check is asserted to pass on exactly those, and the seeds are not chosen
    for it.
    """

    SEEDS = (1, 2, 3)

    def test_foam_needs_d_min_and_refuses_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "out")
            status, caught, _ = CommandLineTests.run_main(
                ["--family", "foam", "--volume", "3", "3", "3", "--fit", "voxel_size", "--voxel-size", "20",
                 "--count", "1", "--seed", "1", "--out", out])
            self.assertIsNone(status)
            self.assertIn("--d-min", str(caught.code))
            self.assertFalse(os.path.exists(out))

    def test_foam_s_frame_is_none(self):
        self.assertEqual(foam(1)["member"]["grown"]["frame"]["kind"], "none")
        self.assertNotIn("guidance", main.FAMILIES["foam"])

    def test_the_connectivity_check_skips_a_root_below_feeder_stop_times_d_min(self):
        import check_connectivity
        argv = ["--family", "foam", "--d0", "3", "0", "--d-min", "2", "--volume", "3", "3", "3", "--fit", "voxel_size",
                "--voxel-size", "20", "--iterations", "4", "4", "--seed", "1"]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(check_connectivity.generate_and_check(1, argv), 0)
        self.assertIn("seed 1: the bed's feeder_stop x --d-min exceeds the sampled root diameter, skipped",
                      printed.getvalue())
        # without a bed the floor is --d-min itself, as before
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(check_connectivity.generate_and_check(1, ["--d0", "1.5", "0", "--d-min", "2",
                                                                       "--d0-min", "1", "--seed", "1"]), 0)
        self.assertIn("seed 1: --d-min exceeds the sampled root diameter, skipped", printed.getvalue())

    @SLOW
    def test_foam_runs_on_the_command_line_in_the_connectivity_check_and_as_a_member(self):
        import check_connectivity
        joined = []
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                argv = (["--family", "foam"] + library.member_argv("foam", 4.0)[2:]
                        + ["--count", "1", "--seed", str(seed)])
                with tempfile.TemporaryDirectory() as out:
                    status, caught, _ = CommandLineTests.run_main(argv + ["--out", out])
                    self.assertIsNone(caught)
                    self.assertEqual(status, 0)
                    stem = f"Lnet_i64_s{seed}"
                    self.assertEqual(sorted(os.listdir(out)), sorted(stem + ext for ext in (".json", ".npz", ".tiff")))
                    with open(os.path.join(out, stem + ".json")) as handle:
                        sidecar = json.load(handle)
                self.assertEqual(sidecar["bed"], bed.parse_settings(main.FAMILIES["foam"]["bed"]))
                self.assertEqual(sidecar["rng_streams"]["bed"], [seed, bed.RNG_TAG])
                ev = sidecar["events"]
                self.assertEqual(ev["bed_proposals"],
                                 ev["bed_proposals_on_feeders"] + ev["bed_proposals_overlapping"] + ev["bed_seeds"])
                self.assertEqual(ev["bed_candidates"],
                                 ev["bed_refused_degree"] + ev["bed_refused_angle"] + ev["bed_refused_girth"]
                                 + ev["bed_refused_clearance"] + ev["bed_edges"])
                self.assertLessEqual(ev["bed_candidates_tip"], ev["bed_candidates"])
                self.assertLessEqual(ev["bed_refused_clearance_at_end"], ev["bed_refused_clearance"])
                self.assertLessEqual(ev["bed_components_one_tree"], ev["bed_components"])
                self.assertLessEqual(ev["bed_walk_redraws"], sidecar["collision_attempts"] * ev["bed_segments"])
                with contextlib.redirect_stdout(io.StringIO()):
                    broken = check_connectivity.generate_and_check(1, argv)
                if ev["bed_components"] > ev["bed_components_one_tree"]:
                    # a component touches both trees, so the network is one piece
                    self.assertEqual(broken, 0)
                    joined.append(seed)
                member = library.grow_member("foam", 4.0, seed)["grown"]
                side = library.BOX_C["foam"] * 4.0
                self.assertEqual(member["growth_box_um"], [side, side, side])
                self.assertEqual(member["events"], ev)
                self.assertEqual(sidecar["frame"], json.loads(json.dumps(member["frame"])))
        # the connectivity assertion ran: seeds 1 and 3 join both trees in development
        self.assertGreater(len(joined), 0)


if __name__ == "__main__":
    unittest.main()
