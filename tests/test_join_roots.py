"""
Tests of root attachment in join.py: with attach_roots, a network's root
inside the field of view may attach, as a side branch, to a nearby vessel of
another network that is at least as thick.

Run from the repository root with
    python -m unittest tests.test_join_roots
The benchmark is slow and runs only with VSYSTEM_SLOW_TESTS=1; it prints the
timings, the root outcome mix, the attachment rate per root diameter, the
root bridges' chord and arc, their volume against the tip bridges' and the
free ends per unit length.
"""
import math
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import graph  # noqa: E402
import join  # noqa: E402
from join import EVENT_KEYS, TIP_OUTCOMES, crop_network, join_networks  # noqa: E402
from library import root_columns  # noqa: E402
from tests.test_join import bounding_box, build_forest, grow, outcome_counts, place  # noqa: E402
from tests.test_pinned_join import BRIDGE_FIELDS_3_3, EVENT_KEYS_3_3, SUMMARY_KEYS_3_3  # noqa: E402

OUTCOME = {name: code for code, name in enumerate(TIP_OUTCOMES)}
ROOT_CODES = {name for name in TIP_OUTCOMES if name == "root" or name.startswith("root_")}
TIP_OUTCOMES_3_3 = ("root", "cut_end", "stub", "not_selected", "bridged_source", "bridged_partner",
                    "no_partner", "collision_failed", "over_budget")

_FORESTS = {}


def with_roots(network):
    """The network with its root columns, the first finite column of each tree label."""
    return dict(network, roots=np.array(root_columns(network["tree"]), dtype=np.int64))


def place_root_at(network, matrix, root_xyz):
    """The network rotated by `matrix` and moved so that its root lies at `root_xyz`, with `roots` set."""
    network = with_roots(network)
    nodes = network["nodes"]
    root = int(network["roots"][0])
    centre = np.nanmean(nodes[:3], axis=1)
    offset = np.asarray(root_xyz, dtype=float) - matrix @ (nodes[:3, root] - centre)
    return place(network, matrix, offset)


def trunk_point(network, fraction):
    """A point of the root's polyline, `fraction` of the way along its columns."""
    nodes = network["nodes"]
    root = root_columns(network["tree"])[0]
    run = [run for run in graph.polylines(nodes) if run[0] == root][0]
    return nodes[:3, run[int(fraction * (len(run) - 1))]]


def thin_beside_thick(gap=12.0):
    """
    A thin tree (root diameter 2) whose root lies `gap` beside the middle of
    the trunk of a thick tree (root diameter 8), both growing along +z, so
    the thin root's upstream direction (-z) sees the trunk within the cone.
    """
    thick = place_root_at(grow(2, 4, d0=8.0), np.eye(3), np.zeros(3))
    thin = place_root_at(grow(1, 3, d0=2.0), np.eye(3), trunk_point(thick, 0.5) + np.array([gap, 0.0, 0.0]))
    return thin, thick


def mixed_forest(seed=31, count=36, side=200.0, inner=(20.0, 180.0), margin=1.0):
    """
    A cropped forest of trees whose root diameters are spread (2, 3, 5 and
    8) so that the ratio rule decides, with each network's root columns
    passed through the crop. Cached per setting.
    """
    key = (seed, count, side, inner, margin)
    if key not in _FORESTS:
        library = [grow(s, 4, d0=d0) for s, d0 in ((1, 3.0), (2, 5.0), (3, 8.0), (4, 2.0))]
        forest = build_forest(library, count, np.zeros(3), np.full(3, side), np.random.default_rng(seed))
        lo, hi = np.full(3, inner[0]), np.full(3, inner[1])
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        cropped = [crop_network(with_roots(n), lo, hi, r_max + margin) for n in forest]
        _FORESTS[key] = (cropped, lo, hi)
    cropped, lo, hi = _FORESTS[key]
    return [dict(n) for n in cropped], lo, hi


def mixed_settings(lo, hi, **overrides):
    settings = dict(fraction=0.9, collision_margin=1.0, box=(lo, hi), boundary_margin=2.0)
    settings.update(overrides)
    return settings


def row_of(result, network, column):
    tips = result["tips"]
    rows = tips[(tips["network"] == network) & (tips["column"] == column)]
    return rows


def outcome_of(result, network, column):
    rows = row_of(result, network, column)
    assert rows.shape == (1,), (network, column, rows)
    return TIP_OUTCOMES[int(rows["outcome"][0])]


def upstream_unit(nodes, column):
    """
    Unit vector from `column` to the first later point of its polyline that
    differs from it (or, failing that, the first earlier one); None when the
    polyline has no other distinct point.
    """
    n = nodes.shape[1]
    here = nodes[:3, column]
    for side in (range(column + 1, n), range(column - 1, -1, -1)):
        for other in side:
            if not np.isfinite(nodes[0, other]):
                break                                       # the polyline ends here
            delta = nodes[:3, other] - here
            length = float(np.linalg.norm(delta))
            if length > 0.0:
                return delta / length
    return None


def straight_vessel(diameter=4.0, z=-14.0):
    """A straight vessel along x at height z, not a root: an interior partner at every column but its ends."""
    x = np.arange(-30.0, 31.0, 1.0)
    nodes = np.vstack([x, np.zeros(x.size), np.full(x.size, z), np.full(x.size, diameter)])
    return {"nodes": nodes, "roots": np.zeros(0, dtype=np.int64)}


def short_polyline(points, diameter):
    nodes = np.vstack([np.asarray(points, dtype=float).T, np.full(len(points), diameter)])
    return {"nodes": nodes, "roots": np.zeros(0, dtype=np.int64)}


def settings_for(networks, **overrides):
    lo, hi = bounding_box(networks, 20.0)
    settings = dict(fraction=0.0, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0, attach_roots=True)
    settings.update(overrides)
    return settings


def fields_3_3(result):
    """What 3.3 returned: the bridges' 3.3 fields and geometry, the tips table, the 3.3 counters and summary."""
    return ([tuple(b[f] if f not in ("tip", "partner") else tuple(b[f]) for f in BRIDGE_FIELDS_3_3)
             for b in result["bridges"]],
            [b["geometry"].tobytes() for b in result["bridges"]],
            result["tips"].tobytes(),
            {k: result["events"][k] for k in EVENT_KEYS_3_3},
            {k: result["summary"][k] for k in SUMMARY_KEYS_3_3})


class RegistryTests(unittest.TestCase):
    def test_outcomes_and_counters_are_appended_after_those_of_3_3(self):
        self.assertEqual(TIP_OUTCOMES[:len(TIP_OUTCOMES_3_3)], TIP_OUTCOMES_3_3)
        self.assertEqual(TIP_OUTCOMES[len(TIP_OUTCOMES_3_3):],
                         ("root_attached", "root_not_selected", "root_no_partner", "root_collision_failed",
                          "root_over_budget"))
        self.assertEqual(EVENT_KEYS[:len(EVENT_KEYS_3_3)], EVENT_KEYS_3_3)
        self.assertEqual(EVENT_KEYS[len(EVENT_KEYS_3_3):],
                         ("join_root_eligible", "join_root_selected", "join_root_attached",
                          "join_root_not_selected", "join_root_no_partner", "join_root_collision_failed",
                          "join_root_over_budget"))

    def test_the_root_parameters_are_validated_whether_or_not_the_option_is_on(self):
        thin, thick = thin_beside_thick()
        lo, hi = bounding_box([thin, thick], 20.0)
        base = dict(fraction=1.0, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0)
        for attach in (False, True):
            for bad in (dict(root_fraction=-0.1), dict(root_fraction=1.5), dict(root_partner_min_ratio=0.9),
                        dict(root_radius=0.0), dict(root_radius=-1.0)):
                with self.subTest(attach_roots=attach, **bad):
                    with self.assertRaises(ValueError):
                        join_networks([thin, thick], np.random.default_rng(0), attach_roots=attach, **base, **bad)
        for attach in (False, True):
            for good in (dict(root_fraction=0.0), dict(root_fraction=1.0), dict(root_partner_min_ratio=1.0),
                         dict(root_radius=None), dict(root_radius=3.0)):
                join_networks([thin, thick], np.random.default_rng(0), attach_roots=attach, **base, **good)

    def test_a_separator_root_is_refused_with_the_option_on(self):
        thin, thick = thin_beside_thick()
        lo, hi = bounding_box([thin, thick], 20.0)
        nodes = np.concatenate([thin["nodes"], np.full((4, 1), np.nan), thin["nodes"]], axis=1)
        separator = thin["nodes"].shape[1]
        network = {"nodes": nodes, "roots": np.array([0, separator])}
        base = dict(fraction=0.0, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0)
        with self.assertRaisesRegex(ValueError, "separator"):
            join_networks([network, thick], np.random.default_rng(0), attach_roots=True, **base)
        # the option-off path accepts it, as 3.3 does
        join_networks([network, thick], np.random.default_rng(0), attach_roots=False, **base)


class SideBranchTests(unittest.TestCase):
    def test_a_root_near_a_thicker_vessel_attaches_as_a_side_branch(self):
        thin, thick = thin_beside_thick()
        lo, hi = bounding_box([thin, thick], 20.0)
        settings = dict(fraction=0.0, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0, merged=True)
        off = join_networks([thin, thick], np.random.default_rng(1), **settings)
        self.assertEqual(off["bridges"], [])
        self.assertEqual(outcome_of(off, 0, 0), "root")
        self.assertEqual(outcome_of(off, 1, 0), "root")
        self.assertEqual(off["summary"]["blunt_roots"], {"before": 2, "after": 2})
        on = join_networks([thin, thick], np.random.default_rng(1), attach_roots=True, **settings)
        self.assertEqual(len(on["bridges"]), 1)
        bridge = on["bridges"][0]
        d_root = float(thin["nodes"][3, 0])
        self.assertEqual(bridge["source_kind"], "root")
        self.assertEqual(bridge["tip"], (0, 0))
        self.assertEqual(bridge["partner"][0], 1)
        self.assertEqual(bridge["partner_kind"], "interior")
        self.assertEqual(bridge["diameter"], d_root)
        self.assertTrue(np.all(bridge["geometry"][3] == d_root))
        partner_column = bridge["partner"][1]
        self.assertGreaterEqual(float(thick["nodes"][3, partner_column]), d_root)
        np.testing.assert_array_equal(bridge["geometry"][:3, 0], thin["nodes"][:3, 0])
        np.testing.assert_array_equal(bridge["geometry"][:3, -1], thick["nodes"][:3, partner_column])
        self.assertEqual(outcome_of(on, 0, 0), "root_attached")
        self.assertEqual(outcome_of(on, 1, 0), "root_no_partner")          # nothing as thick as 8 nearby
        # the attached interior point is now a junction, and the two networks are one component
        merged = on["merged"]
        column = int(np.flatnonzero(merged["network"] == 1)[partner_column])
        self.assertEqual(int(merged["node_kind"][column]), graph.JUNCTION)
        self.assertEqual(int(thick["node_kind"][partner_column]), graph.INTERIOR)
        self.assertEqual(on["summary"]["components"], {"before": 2, "after": 1})
        self.assertEqual(on["summary"]["blunt_roots"], {"before": 2, "after": 1})
        self.assertEqual(on["summary"]["free_ends"]["after"], on["summary"]["free_tips"]["after"] + 1)
        events = on["events"]
        self.assertEqual((events["join_root_eligible"], events["join_root_selected"], events["join_root_attached"],
                          events["join_root_no_partner"]), (2, 2, 1, 1))
        self.assertEqual((events["join_bridges"], events["join_partner_interior"], events["join_components_joined"]),
                         (0, 1, 1))

    def test_a_root_never_attaches_to_a_thinner_vessel(self):
        # the thick tree's root beside the thin tree's trunk: only thinner vessels nearby
        thin = place_root_at(grow(1, 3, d0=2.0), np.eye(3), np.zeros(3))
        thick = place_root_at(grow(2, 4, d0=8.0), np.eye(3), trunk_point(thin, 0.5) + np.array([12.0, 0.0, 0.0]))
        lo, hi = bounding_box([thin, thick], 20.0)
        result = join_networks([thin, thick], np.random.default_rng(2), fraction=0.0, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0, attach_roots=True)
        self.assertEqual(result["bridges"], [])
        self.assertEqual(outcome_of(result, 1, 0), "root_no_partner")
        self.assertEqual(result["events"]["join_root_no_partner"], 2)
        # and on a forest, every root partner is at least root_partner_min_ratio root diameters
        # across, measured at the vertex, for the default and a stricter ratio
        cropped, lo, hi = mixed_forest()
        vertex_diameter = [graph.vertex_diameter(n["nodes"], graph.build(n["nodes"])["canonical"])
                           if n["nodes"].shape[1] else np.zeros(0) for n in cropped]
        seen = 0
        for ratio in (1.0, 2.0):
            result = join_networks(cropped, np.random.default_rng(3), attach_roots=True,
                                   root_partner_min_ratio=ratio, **mixed_settings(lo, hi))
            for bridge in result["bridges"]:
                if bridge["source_kind"] != "root":
                    continue
                seen += 1
                d_root = float(cropped[bridge["tip"][0]]["nodes"][3, bridge["tip"][1]])
                d_partner = float(vertex_diameter[bridge["partner"][0]][bridge["partner"][1]])
                self.assertGreaterEqual(d_partner, ratio * d_root)
                self.assertEqual(bridge["diameter"], d_root)
        self.assertGreater(seen, 0)
        strict = join_networks(cropped, np.random.default_rng(3), attach_roots=True, root_partner_min_ratio=1e6,
                               **mixed_settings(lo, hi))
        self.assertGreater(strict["events"]["join_root_eligible"], 0)
        self.assertEqual(strict["events"]["join_root_no_partner"], strict["events"]["join_root_selected"])
        self.assertEqual(strict["events"]["join_root_attached"], 0)
        self.assertTrue(all(b["source_kind"] == "tip" for b in strict["bridges"]))

    def test_root_bridges_depart_upstream(self):
        cropped, lo, hi = mixed_forest()
        result = join_networks(cropped, np.random.default_rng(4), attach_roots=True, **mixed_settings(lo, hi))
        checked = 0
        for bridge in result["bridges"]:
            if bridge["source_kind"] != "root":
                continue
            network, column = bridge["tip"]
            w = upstream_unit(cropped[network]["nodes"], column)
            if w is None:
                continue
            first = bridge["geometry"][:3, 1] - bridge["geometry"][:3, 0]
            self.assertLess(float(first @ w), 0.0)
            # and within 45 degrees of the upstream direction -w, as a tip bridge's first segment is of its tangent
            self.assertGreater(float(first @ -w) / float(np.linalg.norm(first)), math.cos(math.radians(45.0)))
            checked += 1
        self.assertGreater(checked, 2)

    def test_the_cone_is_taken_about_the_upstream_direction_and_the_search_radius_limits_it(self):
        thin, thick = thin_beside_thick()
        settings = settings_for([thin, thick])
        # the search radius is in root diameters: the nearest allowed trunk point lies 8.72 units from a root of diameter 2
        results = {}
        for root_radius, expected in ((4.0, "root_no_partner"), (4.5, "root_attached")):
            results[root_radius] = join_networks([thin, thick], np.random.default_rng(1), root_radius=root_radius, **settings)
            self.assertEqual(outcome_of(results[root_radius], 0, 0), expected, root_radius)
        nearest = results[4.5]["bridges"][0]
        self.assertLessEqual(nearest["chord"], 4.5 * float(thin["nodes"][3, 0]))
        root_xyz = thin["nodes"][:3, 0]
        ahead = thick["nodes"][:3, nearest["partner"][1]] - root_xyz
        ahead /= np.linalg.norm(ahead)

        def frame(axis):
            """A rotation taking +y, the direction a grown tree leaves its root in, to `axis`."""
            a = np.cross(axis, [0.0, 0.0, 1.0])
            a /= np.linalg.norm(a)
            return np.stack([a, axis, np.cross(a, axis)], axis=1)

        # the body laid along the direction to that point puts it 180 degrees from the upstream
        # direction, which points out of the vessel: behind, so no partner within reach;
        # the body laid the other way puts it straight ahead
        toward = place_root_at(grow(1, 3, d0=2.0), frame(ahead), root_xyz)
        away = place_root_at(grow(1, 3, d0=2.0), frame(-ahead), root_xyz)
        np.testing.assert_allclose(upstream_unit(toward["nodes"], 0), ahead, atol=1e-9)
        np.testing.assert_allclose(upstream_unit(away["nodes"], 0), -ahead, atol=1e-9)
        behind = join_networks([toward, thick], np.random.default_rng(1), root_radius=4.5, **settings)
        self.assertEqual(outcome_of(behind, 0, 0), "root_no_partner")
        self.assertGreater(behind["events"]["join_behind_skipped"], 0)
        in_front = join_networks([away, thick], np.random.default_rng(1), root_radius=4.5, **settings)
        self.assertEqual(outcome_of(in_front, 0, 0), "root_attached")
        self.assertEqual(in_front["bridges"][0]["partner"][0], 1)
        self.assertEqual(in_front["events"]["join_behind_skipped"], 0)

    def test_interior_points_rank_before_tips_and_an_attached_tip_is_consumed(self):
        thin = place_root_at(grow(1, 3, d0=2.0), np.eye(3), np.zeros(3))        # root at the origin, u = -y
        straight = straight_vessel()                                             # z = -14: fourteen units ahead
        straight["nodes"][1:3] = straight["nodes"][2:0:-1]                       # turn it: y = -14, z = 0
        stub = short_polyline([(4.0, -6.0, 0.0), (4.0, -7.0, 0.0), (4.0, -8.0, 0.0)], 4.0)   # its tip 7.2 away
        both = join_networks([thin, straight, stub], np.random.default_rng(1), **settings_for([thin, straight, stub]))
        self.assertEqual(outcome_of(both, 0, 0), "root_attached")
        bridge = both["bridges"][0]
        self.assertEqual((bridge["partner"][0], bridge["partner_kind"]), (1, "interior"))     # not the nearer tip
        self.assertAlmostEqual(bridge["chord"], 14.0)
        self.assertEqual(outcome_of(both, 2, 0), "not_selected")
        only_tip = join_networks([thin, stub], np.random.default_rng(1), **settings_for([thin, stub]))
        self.assertEqual(outcome_of(only_tip, 0, 0), "root_attached")
        bridge = only_tip["bridges"][0]
        self.assertEqual((bridge["partner"], bridge["partner_kind"], bridge["diameter"]), ((1, 0), "tip", 2.0))
        self.assertEqual(outcome_of(only_tip, 1, 0), "bridged_partner")
        events = only_tip["events"]
        self.assertEqual((events["join_partner_tips"], events["join_bridged_partners"], events["join_partner_interior"]),
                         (1, 1, 0))
        self.assertEqual(only_tip["summary"]["free_tips"]["after"], only_tip["summary"]["free_tips"]["before"] - 1)
        # a thinner tip, nearer still, is never used
        thinner = short_polyline([(3.0, -5.0, 0.0), (3.0, -6.0, 0.0), (3.0, -7.0, 0.0)], 1.5)
        result = join_networks([thin, thinner], np.random.default_rng(1), **settings_for([thin, thinner]))
        self.assertEqual(outcome_of(result, 0, 0), "root_no_partner")

    def test_a_root_in_a_junction_zone_or_without_a_tangent_still_attaches(self):
        separator = np.full((4, 1), np.nan)
        straight = straight_vessel()
        # a Y whose first junction lies within the overlap zone of its root: a stub as a tip, a root all the same
        stem = short_polyline([(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (-1, 0, 4), (-2, 0, 5), (-3, 0, 6)], 2.0)
        branch = short_polyline([(0, 0, 3), (1, 0, 4), (2, 0, 5), (3, 0, 6)], 2.0)
        y = {"nodes": np.concatenate([stem["nodes"], separator, branch["nodes"]], axis=1), "roots": np.array([0])}
        built = graph.build(y["nodes"])
        self.assertEqual(int(built["degree"][3]), 3)
        networks = [y, straight]
        off = join_networks(networks, np.random.default_rng(1), **settings_for(networks, attach_roots=False))
        self.assertEqual(outcome_of(off, 0, 0), "root")
        on = join_networks(networks, np.random.default_rng(1), **settings_for(networks))
        self.assertEqual(outcome_of(on, 0, 0), "root_attached")
        self.assertEqual(on["bridges"][0]["partner"][0], 1)
        # a root whose own polyline holds no other distinct point has no tangent: no cone, a chord departure
        lone = short_polyline([(0, 0, 0), (0, 0, 0)], 2.0)
        body = short_polyline([(0, 0, 0), (0, 0, 3), (0, 0, 6), (0, 0, 9)], 2.0)
        network = {"nodes": np.concatenate([lone["nodes"], separator, body["nodes"]], axis=1), "roots": np.array([0])}
        built = graph.build(network["nodes"])
        self.assertEqual(int(built["degree"][0]), 1)
        self.assertIsNone(upstream_unit(network["nodes"], 0))
        networks = [network, straight]
        result = join_networks(networks, np.random.default_rng(1), **settings_for(networks))
        self.assertEqual(outcome_of(result, 0, 0), "root_attached")
        bridge = result["bridges"][0]
        self.assertEqual((bridge["partner"][0], bridge["partner_kind"]), (1, "interior"))
        first = bridge["geometry"][:3, 1] - bridge["geometry"][:3, 0]
        chord = bridge["geometry"][:3, -1] - bridge["geometry"][:3, 0]
        self.assertGreater(float(first @ chord) / (np.linalg.norm(first) * np.linalg.norm(chord)),
                           math.cos(math.radians(45.0)))

    def test_cut_end_roots_never_attach(self):
        thick = place_root_at(grow(2, 4, d0=8.0), np.eye(3), np.zeros(3))
        thin = grow(1, 3, d0=2.0)
        beside = np.array([12.0, 0.0, 0.0])
        boundary = 3.0
        # roots beside the trunk: one eligible, one that a crop made (its kind is not a tip),
        # one within the boundary margin of the hi x face, one outside it
        eligible = place_root_at(thin, np.eye(3), trunk_point(thick, 0.5) + beside)
        not_a_tip = place_root_at(thin, np.eye(3), trunk_point(thick, 0.7) + beside)
        not_a_tip["node_kind"] = not_a_tip["node_kind"].copy()
        not_a_tip["node_kind"][int(not_a_tip["roots"][0])] = graph.INTERIOR
        near_face = place_root_at(thin, np.eye(3), trunk_point(thick, 0.1) + beside)
        face_x = float(near_face["nodes"][0, int(near_face["roots"][0])]) + 1.0
        outside = place_root_at(thin, np.eye(3), trunk_point(thick, 0.2) + beside)
        outside = place_root_at(thin, np.eye(3), [face_x + 2.0] + outside["nodes"][1:3, 0].tolist())
        networks = [thick, eligible, not_a_tip, near_face, outside]
        lo, hi = bounding_box(networks, 20.0)
        hi[0] = face_x
        for network, name in ((thick, "thick"), (eligible, "eligible"), (not_a_tip, "not_a_tip")):
            root = network["nodes"][:3, int(network["roots"][0])]
            self.assertTrue(np.all(root > lo + boundary) and np.all(root < hi - boundary), name)
        self.assertLess(float(near_face["nodes"][0, 0]), face_x)
        self.assertGreater(float(near_face["nodes"][0, 0]), face_x - boundary)
        self.assertGreater(float(outside["nodes"][0, 0]), face_x)
        settings = dict(fraction=0.0, collision_margin=0.5, box=(lo, hi), boundary_margin=boundary)
        off = join_networks(networks, np.random.default_rng(5), **settings)
        self.assertEqual([outcome_of(off, k, 0) for k in range(5)], ["root", "root", "root", "root", "cut_end"])
        on = join_networks(networks, np.random.default_rng(5), attach_roots=True, **settings)
        self.assertEqual([outcome_of(on, k, 0) for k in range(5)],
                         ["root_no_partner", "root_attached", "cut_end", "cut_end", "cut_end"])
        self.assertEqual([b["tip"] for b in on["bridges"]], [(1, 0)])
        self.assertEqual(on["events"]["join_root_eligible"], 2)
        self.assertEqual(on["events"]["join_cut_ends"], off["events"]["join_cut_ends"] + 2)   # the two that were root
        self.assertEqual(on["summary"]["blunt_roots"], {"before": 2, "after": 1})
        self.assertEqual(off["summary"]["blunt_roots"], {"before": 2, "after": 2})

    def test_root_partners_lie_on_other_networks_whatever_the_policy(self):
        thin, thick = thin_beside_thick()
        lo, hi = bounding_box([thin, thick], 20.0)
        settings = dict(fraction=0.0, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0,
                        policy="any", min_separation=1, attach_roots=True)
        two = join_networks([thin, thick], np.random.default_rng(6), **settings)
        self.assertEqual([b["tip"] for b in two["bridges"]], [(0, 0)])
        self.assertEqual(two["bridges"][0]["partner"][0], 1)
        # the same geometry as one network: the thick trunk is kin, so the root finds no partner
        nodes = np.concatenate([thin["nodes"], np.full((4, 1), np.nan), thick["nodes"]], axis=1)
        kind = np.concatenate([thin["node_kind"], [graph.SEPARATOR], thick["node_kind"]]).astype(np.int8)
        offset = thin["nodes"].shape[1] + 1
        one = {"nodes": nodes, "node_kind": kind, "roots": np.array([0, offset])}
        same = join_networks([one], np.random.default_rng(6), **settings)
        self.assertEqual(same["bridges"], [])
        self.assertEqual(outcome_of(same, 0, 0), "root_no_partner")
        self.assertEqual(outcome_of(same, 0, offset), "root_no_partner")
        # and on a forest under policy any, every root bridge crosses networks
        cropped, lo, hi = mixed_forest()
        result = join_networks(cropped, np.random.default_rng(7), attach_roots=True,
                               **mixed_settings(lo, hi, policy="any", min_separation=1))
        root_bridges = [b for b in result["bridges"] if b["source_kind"] == "root"]
        self.assertGreater(len(root_bridges), 0)
        for bridge in root_bridges:
            self.assertNotEqual(bridge["tip"][0], bridge["partner"][0])


class AccountingTests(unittest.TestCase):
    def check_partition(self, networks, result):
        tips = result["tips"]
        degree_one = []
        for k, network in enumerate(networks):
            if network["nodes"].shape[1] == 0:
                continue
            built = graph.build(network["nodes"])
            vertices = np.flatnonzero(built["canonical"] == np.arange(network["nodes"].shape[1]))
            degree_one.extend((k, int(v)) for v in vertices if built["degree"][v] == 1)
        self.assertEqual(sorted(degree_one), [(int(r["network"]), int(r["column"])) for r in tips])
        counts = outcome_counts(result)
        events = result["events"]
        self.assertEqual(sum(counts.values()), events["join_tips"])
        for name, key in join._OUTCOME_COUNTERS:
            self.assertEqual(counts[name], events[key], name)
        self.assertEqual(counts["root"], 0)
        root_rows = sum(counts[name] for name in ROOT_CODES)
        self.assertEqual(events["join_eligible"], events["join_tips"] - root_rows - counts["cut_end"] - counts["stub"])
        self.assertEqual(len(result["bridges"]), events["join_bridges"] + events["join_root_attached"])
        self.assertEqual(counts["bridged_source"], sum(b["source_kind"] == "tip" for b in result["bridges"]))
        self.assertEqual(counts["root_attached"], sum(b["source_kind"] == "root" for b in result["bridges"]))
        self.assertEqual(counts["bridged_partner"], events["join_partner_tips"])
        self.assertEqual(events["join_partner_tips"] + events["join_partner_interior"], len(result["bridges"]))
        self.assertEqual(events["join_root_eligible"],
                         events["join_root_selected"] + counts["root_not_selected"])
        self.assertEqual(events["join_root_selected"],
                         counts["root_attached"] + counts["root_no_partner"] + counts["root_collision_failed"]
                         + counts["root_over_budget"])
        summary = result["summary"]
        self.assertEqual(summary["free_tips"]["before"], events["join_tips"] - root_rows - counts["cut_end"])
        self.assertEqual(summary["free_tips"]["after"],
                         summary["free_tips"]["before"] - counts["bridged_source"] - counts["bridged_partner"])
        self.assertEqual(summary["blunt_roots"]["before"], events["join_root_eligible"])
        self.assertEqual(summary["blunt_roots"]["after"], events["join_root_eligible"] - counts["root_attached"])
        for when in ("before", "after"):
            self.assertEqual(summary["free_ends"][when], summary["free_tips"][when] + summary["blunt_roots"][when])
            length = summary["length"][when]
            self.assertAlmostEqual(summary["free_ends_per_length"][when], summary["free_ends"][when] / length)
            self.assertAlmostEqual(summary["blunt_roots_per_length"][when], summary["blunt_roots"][when] / length)
        return counts

    def test_outcomes_partition_the_degree_one_vertices_and_agree_with_events(self):
        cropped, lo, hi = mixed_forest()
        result = join_networks(cropped, np.random.default_rng(8), attach_roots=True, root_fraction=0.7,
                               **mixed_settings(lo, hi))
        counts = self.check_partition(cropped, result)
        self.assertGreater(counts["root_attached"], 0)
        self.assertGreater(counts["root_not_selected"], 0)
        self.assertGreater(counts["bridged_source"], 0)
        self.assertGreater(counts["cut_end"], 0)

    def test_a_mesh_pair_gives_each_of_its_four_roots_one_row(self):
        mesh = grow(3, 6, family="mesh")
        nodes = mesh["nodes"]
        height = float(np.nanmax(nodes[1]) - np.nanmin(nodes[1]))
        networks = [with_roots(place(mesh, np.eye(3), np.zeros(3))),
                    with_roots(place(mesh, np.diag([-1.0, -1.0, 1.0]), np.array([0.0, height + 10.0, 0.0])))]
        self.assertEqual([n["roots"].size for n in networks], [2, 2])
        lo, hi = bounding_box(networks, 20.0)
        result = join_networks(networks, np.random.default_rng(11), fraction=1.0, collision_margin=0.5,
                               box=(lo, hi), boundary_margin=0.0, attach_roots=True)
        counts = self.check_partition(networks, result)
        roots = {(k, int(c)) for k, n in enumerate(networks) for c in n["roots"]}
        for k, column in sorted(roots):
            self.assertIn(outcome_of(result, k, column), ROOT_CODES - {"root"})
        self.assertEqual(sum(counts[name] for name in ROOT_CODES), 4)
        self.assertEqual(result["events"]["join_root_eligible"], 4)
        for bridge in result["bridges"]:
            self.assertNotIn(tuple(bridge["partner"]), roots)
            if bridge["source_kind"] == "tip":
                self.assertNotIn(tuple(bridge["tip"]), roots)
            else:
                self.assertIn(tuple(bridge["tip"]), roots)
        self.assertGreater(len(result["bridges"]), 0)

    def test_identical_inputs_and_generator_state_give_identical_output(self):
        cropped, lo, hi = mixed_forest()
        runs = [join_networks(cropped, np.random.default_rng(9), attach_roots=True, root_fraction=0.8,
                              **mixed_settings(lo, hi)) for _ in range(2)]
        self.assertGreater(sum(b["source_kind"] == "root" for b in runs[0]["bridges"]), 0)
        self.assertEqual(len(runs[0]["bridges"]), len(runs[1]["bridges"]))
        for a, b in zip(runs[0]["bridges"], runs[1]["bridges"]):
            np.testing.assert_array_equal(a["geometry"], b["geometry"])
            self.assertEqual((a["tip"], a["partner"], a["source_kind"], a["redraws"]),
                             (b["tip"], b["partner"], b["source_kind"], b["redraws"]))
        np.testing.assert_array_equal(runs[0]["tips"], runs[1]["tips"])
        self.assertEqual(runs[0]["events"], runs[1]["events"])
        self.assertEqual(runs[0]["summary"], runs[1]["summary"])
        other = join_networks(cropped, np.random.default_rng(10), attach_roots=True, root_fraction=0.8,
                              **mixed_settings(lo, hi))
        self.assertNotEqual([b["tip"] for b in other["bridges"]], [b["tip"] for b in runs[0]["bridges"]])

    def test_the_result_is_invariant_under_a_change_of_unit(self):
        cropped, lo, hi = mixed_forest()
        settings = dict(fraction=0.9, collision_margin=1.0, boundary_margin=2.0, tol=graph.DEFAULT_TOL,
                        attach_roots=True, root_radius=6.0, root_partner_min_ratio=1.2)
        reference = join_networks(cropped, np.random.default_rng(12), box=(lo, hi), **settings)
        root_bridges = [b for b in reference["bridges"] if b["source_kind"] == "root"]
        self.assertGreater(len(root_bridges), 0)
        volume = sum(b["volume"] for b in reference["bridges"][:len(root_bridges) + 2])
        budgeted = join_networks(cropped, np.random.default_rng(12), box=(lo, hi),
                                 max_bridge_volume=volume * (1 + 1e-12), **settings)
        self.assertEqual(len(budgeted["bridges"]), len(root_bridges) + 2)
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                scaled = [dict(network, nodes=network["nodes"] * scale) for network in cropped]
                for original, budget in ((reference, None), (budgeted, volume * (1 + 1e-12) * scale ** 3)):
                    result = join_networks(scaled, np.random.default_rng(12), box=(lo * scale, hi * scale),
                                           fraction=0.9, collision_margin=1.0 * scale, boundary_margin=2.0 * scale,
                                           tol=graph.DEFAULT_TOL * scale, max_bridge_volume=budget,
                                           attach_roots=True, root_radius=6.0, root_partner_min_ratio=1.2)
                    np.testing.assert_array_equal(result["tips"], original["tips"])
                    self.assertEqual(result["events"], original["events"])
                    self.assertEqual(len(result["bridges"]), len(original["bridges"]))
                    for a, b in zip(result["bridges"], original["bridges"]):
                        self.assertEqual((a["tip"], a["partner"], a["source_kind"], a["partner_kind"]),
                                         (b["tip"], b["partner"], b["source_kind"], b["partner_kind"]))
                        np.testing.assert_allclose(a["geometry"], b["geometry"] * scale, rtol=1e-12, atol=0.0)
                        np.testing.assert_allclose(a["arc"], b["arc"] * scale, rtol=1e-12)
                        np.testing.assert_allclose(a["volume"], b["volume"] * scale ** 3, rtol=1e-12)
                    for key in ("blunt_roots", "free_ends"):
                        self.assertEqual(result["summary"][key], original["summary"][key])

    def test_root_and_tip_bridges_share_the_budget_with_roots_first(self):
        cropped, lo, hi = mixed_forest()
        settings = mixed_settings(lo, hi, fraction=1.0, attach_roots=True)
        full = join_networks(cropped, np.random.default_rng(13), **settings)
        kinds = [b["source_kind"] for b in full["bridges"]]
        n_roots = kinds.count("root")
        self.assertGreaterEqual(n_roots, 2)
        self.assertGreater(kinds.count("tip"), 2)
        self.assertEqual(kinds, ["root"] * n_roots + ["tip"] * (len(kinds) - n_roots))   # roots first
        volumes = [b["volume"] for b in full["bridges"]]
        # a budget spent during the root phase: the remaining selected roots are
        # over budget, and so is every selected tip that was not consumed
        m = n_roots // 2
        part = join_networks(cropped, np.random.default_rng(13), max_bridge_volume=sum(volumes[:m]) * (1 + 1e-12),
                             **settings)
        self.assertEqual(len(part["bridges"]), m)
        self.assertEqual([b["source_kind"] for b in part["bridges"]], ["root"] * m)
        for a, b in zip(part["bridges"], full["bridges"]):
            self.assertEqual((a["tip"], a["partner"]), (b["tip"], b["partner"]))
            np.testing.assert_array_equal(a["geometry"], b["geometry"])
        counts = outcome_counts(part)
        events = part["events"]
        self.assertEqual(counts["root_attached"], m)
        self.assertGreater(counts["root_over_budget"], 0)
        self.assertEqual(counts["root_attached"] + counts["root_no_partner"] + counts["root_collision_failed"]
                         + counts["root_over_budget"], events["join_root_selected"])
        self.assertEqual(events["join_bridges"], 0)
        self.assertEqual(counts["over_budget"], events["join_selected"] - events["join_source_consumed"])
        self.assertGreater(counts["over_budget"], 0)
        self.assertLessEqual(part["summary"]["bridge_volume"], sum(volumes[:m]) * (1 + 1e-12))
        # a budget spent during the tip phase: every root bridge is placed, and the tip bridges follow
        k = n_roots + 2
        later = join_networks(cropped, np.random.default_rng(13), max_bridge_volume=sum(volumes[:k]) * (1 + 1e-12),
                              **settings)
        self.assertEqual(len(later["bridges"]), k)
        for a, b in zip(later["bridges"], full["bridges"]):
            self.assertEqual((a["tip"], a["partner"], a["source_kind"]), (b["tip"], b["partner"], b["source_kind"]))
            np.testing.assert_array_equal(a["geometry"], b["geometry"])
        counts = outcome_counts(later)
        self.assertEqual(counts["root_over_budget"], 0)
        self.assertEqual(counts["root_attached"], n_roots)
        self.assertEqual(counts["bridged_source"], 2)
        self.assertGreater(counts["over_budget"], 0)

    def test_any_prefix_of_the_bridges_is_a_valid_result(self):
        cropped, lo, hi = mixed_forest(seed=35, count=12, side=130.0, inner=(15.0, 115.0))
        settings = mixed_settings(lo, hi, fraction=1.0, attach_roots=True)
        full = join_networks(cropped, np.random.default_rng(14), **settings)
        n_roots = sum(b["source_kind"] == "root" for b in full["bridges"])
        self.assertGreaterEqual(n_roots, 1)
        self.assertGreater(len(full["bridges"]), n_roots)
        for k in range(len(full["bridges"]) + 1):
            with self.subTest(k=k):
                budget = sum(b["volume"] for b in full["bridges"][:k])
                result = join_networks(cropped, np.random.default_rng(14), max_bridge_volume=budget * (1 + 1e-12),
                                       **settings)
                self.assertEqual(len(result["bridges"]), k)
                for a, b in zip(result["bridges"], full["bridges"][:k]):
                    self.assertEqual((a["tip"], a["partner"], a["source_kind"]), (b["tip"], b["partner"], b["source_kind"]))
                    np.testing.assert_array_equal(a["geometry"], b["geometry"])
                    self.assertEqual(outcome_of(result, *a["tip"]),
                                     "root_attached" if a["source_kind"] == "root" else "bridged_source")
                    if a["partner_kind"] == "tip":
                        self.assertEqual(outcome_of(result, *a["partner"]), "bridged_partner")
                counts = outcome_counts(result)
                if k < len(full["bridges"]):
                    self.assertGreater(counts["over_budget"] + counts["root_over_budget"], 0)
                if k < n_roots:
                    self.assertGreater(counts["root_over_budget"], 0)
                self.assertEqual(result["events"]["join_over_budget"], counts["over_budget"])
                self.assertEqual(result["events"]["join_root_over_budget"], counts["root_over_budget"])
                self.assertEqual(counts["root_attached"] + counts["bridged_source"], k)

    def test_in_box_roots_that_are_ineligible_change_only_their_own_rows(self):
        thick = place_root_at(grow(2, 4, d0=8.0), np.eye(3), np.zeros(3))
        thick["roots"] = np.zeros(0, dtype=np.int64)
        thin = grow(1, 3, d0=2.0)
        beside = np.array([12.0, 0.0, 0.0])
        not_a_tip = place_root_at(thin, np.eye(3), trunk_point(thick, 0.7) + beside)
        not_a_tip["node_kind"] = not_a_tip["node_kind"].copy()
        not_a_tip["node_kind"][0] = graph.INTERIOR
        near_face = place_root_at(thin, np.eye(3), trunk_point(thick, 0.1) + beside)
        networks = [thick, not_a_tip, near_face]
        lo, hi = bounding_box(networks, 20.0)
        hi[0] = float(near_face["nodes"][0, 0]) + 1.0
        settings = dict(fraction=1.0, collision_margin=0.5, box=(lo, hi), boundary_margin=3.0)
        rngs = [np.random.default_rng(16) for _ in range(2)]
        off = join_networks(networks, rngs[0], attach_roots=False, **settings)
        on = join_networks(networks, rngs[1], attach_roots=True, **settings)
        self.assertGreater(len(off["bridges"]), 0)
        self.assertEqual(on["events"]["join_root_eligible"], 0)
        self.assertEqual([outcome_of(off, k, 0) for k in (1, 2)], ["root", "root"])
        self.assertEqual([outcome_of(on, k, 0) for k in (1, 2)], ["cut_end", "cut_end"])
        bridges_on, geometry_on, tips_on, events_on, summary_on = fields_3_3(on)
        bridges_off, geometry_off, tips_off, events_off, summary_off = fields_3_3(off)
        self.assertEqual((bridges_on, geometry_on, summary_on), (bridges_off, geometry_off, summary_off))
        self.assertEqual(rngs[0].bit_generator.state, rngs[1].bit_generator.state)
        self.assertEqual({k: v for k, v in events_on.items() if k not in ("join_roots", "join_cut_ends")},
                         {k: v for k, v in events_off.items() if k not in ("join_roots", "join_cut_ends")})
        self.assertEqual((events_on["join_roots"], events_on["join_cut_ends"]),
                         (0, events_off["join_cut_ends"] + events_off["join_roots"]))
        other_rows = (on["tips"]["column"] != 0) | (on["tips"]["network"] == 0)
        np.testing.assert_array_equal(on["tips"][other_rows], off["tips"][other_rows])
        self.assertEqual(on["summary"]["blunt_roots"], {"before": 0, "after": 0})

    def test_with_no_eligible_root_the_option_draws_nothing(self):
        cropped, lo, hi = mixed_forest()
        rootless = [dict(network, roots=np.zeros(0, dtype=np.int64)) for network in cropped]
        rngs = [np.random.default_rng(15) for _ in range(2)]
        off = join_networks(rootless, rngs[0], attach_roots=False, **mixed_settings(lo, hi))
        on = join_networks(rootless, rngs[1], attach_roots=True, **mixed_settings(lo, hi))
        self.assertGreater(len(off["bridges"]), 0)
        self.assertEqual(fields_3_3(on), fields_3_3(off))
        self.assertEqual(rngs[0].bit_generator.state, rngs[1].bit_generator.state)
        self.assertEqual(on["events"]["join_root_eligible"], 0)
        self.assertEqual(on["summary"]["blunt_roots"], {"before": 0, "after": 0})
        self.assertTrue(all(b["source_kind"] == "tip" for b in on["bridges"]))
        # with eligible roots the option draws, so the tip bridges differ from the option off
        off = join_networks(cropped, np.random.default_rng(15), attach_roots=False, **mixed_settings(lo, hi))
        on = join_networks(cropped, np.random.default_rng(15), attach_roots=True, **mixed_settings(lo, hi))
        self.assertGreater(on["events"]["join_root_eligible"], 0)
        self.assertNotEqual([b["tip"] for b in on["bridges"] if b["source_kind"] == "tip"],
                            [b["tip"] for b in off["bridges"]])


@unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmark")
class BenchmarkTests(unittest.TestCase):
    """
    The typical forest of the joining benchmark (330 networks placed in a
    box of side 700 with a point-clearance rejection, cropped, fraction 0.9,
    boundary margin 2), grown with root diameters spread log-uniformly over
    [2, 8] at d_min = 1 so that the ratio rule decides, and joined with root
    attachment off and on. Printed: the timings, the root outcome mix, the
    attachment rate per root diameter, the root bridges' chord and arc in
    root diameters at the default and a smaller root_radius, their volume
    against the tip bridges' and the free ends per unit length.
    """

    def test_typical_forest_with_root_attachment(self):
        diameters = [2.0 * 4.0 ** (i / 5.0) for i in range(6)]
        library = [grow(seed, 5, d0=d0, subdivisions=2) for i, d0 in enumerate(diameters) for seed in (2 * i + 1, 2 * i + 2)]
        rng = np.random.default_rng(20)
        lo, hi = np.zeros(3), np.full(3, 700.0)
        started = time.perf_counter()
        forest = build_forest(library, 330, lo, hi, rng, clearance_gap=2.0, tries=60)
        placed = time.perf_counter() - started
        margin = 1.0
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        cropped = [crop_network(with_roots(n), lo, hi, r_max + margin) for n in forest]
        import resource
        settings = dict(fraction=0.9, collision_margin=margin, box=(lo, hi), boundary_margin=2.0)
        runs = {}
        for label, extra in (("off", {}), ("on", dict(attach_roots=True)),
                             ("on, root_radius 5", dict(attach_roots=True, root_radius=5.0))):
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            started = time.perf_counter()
            result = join_networks(cropped, np.random.default_rng(21), **settings, **extra)
            elapsed = time.perf_counter() - started
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            runs[label] = (result, elapsed, rss_before, rss_after)
        print(f"\ntypical forest: {len(cropped)} networks, {runs['off'][0]['events']['join_points']} points, "
              f"placing {placed:.1f} s; root diameters {', '.join(f'{d:.2f}' for d in diameters)}")
        for label, (result, elapsed, rss_before, rss_after) in runs.items():
            events = result["events"]
            counts = outcome_counts(result)
            summary = result["summary"]
            print(f"  {label}: joining {elapsed:.1f} s, peak RSS {rss_after:.0f} MB; "
                  f"eligible tips {events['join_eligible']}, selected {events['join_selected']}, "
                  f"tip bridges {events['join_bridges']}; eligible roots {events['join_root_eligible']}, "
                  f"selected {events['join_root_selected']}, attached {events['join_root_attached']}")
            print("    root outcomes: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v and k in ROOT_CODES))
            print(f"    components {summary['components']['before']} -> {summary['components']['after']}, "
                  f"free tips per length {summary['free_tips_per_length']['before']:.5f} -> "
                  f"{summary['free_tips_per_length']['after']:.5f}, "
                  f"blunt roots per length {summary['blunt_roots_per_length']['before']:.5f} -> "
                  f"{summary['blunt_roots_per_length']['after']:.5f}, "
                  f"free ends per length {summary['free_ends_per_length']['before']:.5f} -> "
                  f"{summary['free_ends_per_length']['after']:.5f}")
            root_bridges = [b for b in result["bridges"] if b["source_kind"] == "root"]
            tip_bridges = [b for b in result["bridges"] if b["source_kind"] == "tip"]
            if root_bridges:
                chord = np.array([b["chord"] / b["diameter"] for b in root_bridges])
                arc = np.array([b["arc"] / b["diameter"] for b in root_bridges])
                root_volume = sum(b["volume"] for b in root_bridges)
                tip_volume = sum(b["volume"] for b in tip_bridges)
                print(f"    root bridges {len(root_bridges)}: chord/d_root median {np.median(chord):.2f} "
                      f"(mean {chord.mean():.2f}, max {chord.max():.2f}), arc/d_root median {np.median(arc):.2f} "
                      f"(mean {arc.mean():.2f}); interior partners "
                      f"{sum(b['partner_kind'] == 'interior' for b in root_bridges)}; "
                      f"volume {root_volume:.0f} ({root_volume / len(root_bridges):.1f} per bridge) against "
                      f"tip bridges {tip_volume:.0f} ({tip_volume / max(len(tip_bridges), 1):.1f} per bridge)")
                tips = result["tips"]
                per_bin = {}
                for row in tips:
                    name = TIP_OUTCOMES[int(row["outcome"])]
                    if name in ROOT_CODES:
                        d_root = round(float(cropped[int(row["network"])]["nodes"][3, int(row["column"])]), 2)
                        eligible, attached = per_bin.get(d_root, (0, 0))
                        per_bin[d_root] = (eligible + 1, attached + (name == "root_attached"))
                print("    attachment rate per root diameter: " + ", ".join(
                    f"{d}: {attached}/{eligible}" for d, (eligible, attached) in sorted(per_bin.items())))
        self.assertLess(runs["off"][1], 30.0)
        self.assertLess(runs["on"][1], 60.0)
        self.assertLess(runs["on"][3], 2048.0)
        self.assertGreater(runs["on"][0]["events"]["join_root_attached"], 0)


if __name__ == "__main__":
    unittest.main()
