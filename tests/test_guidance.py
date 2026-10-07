"""
Tests of guidance.py and of the guided walk: the rules and their refusals,
the turn and bank of one step, the stationary laws on a long stem, the draw
count, the senses of polar rules, the frame record, root offsets, scale
invariance and the aligned family end to end.

Run from the repository root with
    python -m unittest tests.test_guidance
The tests that grow the aligned preset at R 5 over ten seeds, and the one
that grows twelve long stems to check that the order falls with the guidance
length, take minutes and run only with VSYSTEM_SLOW_TESTS=1; their margins
were set from the development measurement recorded beside each of them.
"""
import contextlib
import io
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

import graph  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from analyseGrammar import WalkSettings, branching_turtle_to_coords  # noqa: E402
from guidance import EVENT_KEYS, Guidance, bank_frame, guide_frame, parse_rules  # noqa: E402
from main import FAMILIES, RootOutsideBox, grow_network  # noqa: E402

PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
VOLUME = (48, 48, 24)
WALK = {"tortuosity": "walk", "persistence": 10.0}
X, Y, Z = np.eye(3)

AXIS_X = {"below": None, "field": "axis", "axis": [1, 0, 0], "length": 3.0, "onset": 0.0}
PLANE_Z = {"below": None, "field": "plane", "normal": [0, 0, 1], "length": 3.0, "onset": 0.0}
SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")


def grow(seed, niter=5, d0=20.0, properties=PROPERTIES, volume=VOLUME, **options):
    """One network from the grammar's seeded draws, with grow_network's options."""
    random.seed(seed)
    np.random.seed(seed)
    return grow_network(niter, d0, dict(properties), volume, seed=seed, **options)


def grow_preset(family, ratio, seed, **override):
    """A library member of `family` at root ratio `ratio`, grown as library.grow_member grows it."""
    argv = library.member_argv(family, ratio)
    args = main.build_parser(family).parse_args(argv)
    main.validate_shaping(args)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    properties, d0, niter = main.sample_parameters(args)
    kwargs = {"tVol": tuple(args.volume), "fit": args.fit, "clip_axes": tuple(args.clip_axes),
              "voxel_size": args.voxel_size, "subdivisions": args.subdivisions, "d_min": args.d_min,
              "grow_in_volume": args.grow_in_volume, "seed": seed}
    kwargs.update(main.shaping_options(args))
    kwargs.update(override)
    return grow_network(niter, d0, properties, **kwargs)


def edge_tangents(grown):
    """Unit tangents (lower to higher column), lengths, diameters and the higher column's tree label."""
    nodes, edges = grown["nodes"], grown["edges"]
    canonical = graph.canonical_columns(nodes)
    diameter = graph.vertex_diameter(nodes, canonical)
    vector = nodes[:3, edges[1]] - nodes[:3, edges[0]]
    length = np.linalg.norm(vector, axis=0)
    tangent = vector / np.where(length > 0.0, length, 1.0)
    mean = (diameter[edges[0]] + diameter[edges[1]]) / 2.0
    keep = (length > 0.0) & np.isfinite(mean)
    return tangent[:, keep], length[keep], mean[keep], grown["tree"][edges[1]][keep]


def order_parameter(grown, axis, mask=None):
    """Length-weighted S = <(3 (t . a)^2 - 1) / 2> over the edges `mask` selects."""
    tangent, length, _, _ = edge_tangents(grown)
    u = tangent.T @ np.asarray(axis, dtype=float)
    if mask is not None:
        u, length = u[mask], length[mask]
    return float(np.sum(length * (3.0 * u ** 2 - 1.0) / 2.0) / np.sum(length))


def stem_headings(rules, seed, moves=2000, persistence=10.0, direction=Y, perpendicular=Z, bound_to=None):
    """
    The unit step directions of one walked stem of `moves` sub-segments of
    length 1.4 and diameter 1, eight steps per sub-segment (step 0.175 d),
    guided by `rules`; the walk draws from default_rng([seed, 1]).
    """
    guide = Guidance(rules, 1.0).bind(direction if bound_to is None else bound_to)
    walk = WalkSettings(persistence, np.random.default_rng([seed, 1]), guidance=guide)
    rows = list(branching_turtle_to_coords("{" + "f(1.4,1.0)" * moves + "}", 1.0, direction=direction,
                                           perpendicular=perpendicular, walk=walk, subdivisions=3))
    points = np.array([row[:3] for row in rows if not math.isnan(row[0])])
    steps = np.diff(points, axis=0)
    length = np.linalg.norm(steps, axis=1)
    return steps[length > 0.0] / length[length > 0.0, None]      # the stem's first point is yielded twice


class RuleTests(unittest.TestCase):
    def test_rules_are_normalised_with_exactly_the_keys_of_their_field_in_order(self):
        rules = parse_rules(FAMILIES["aligned"]["guidance"], 1.0)
        self.assertEqual([list(rule) for rule in rules],
                         [["below", "field", "axis", "sense", "polarity", "length", "onset"],
                          ["below", "field", "normal", "length", "onset", "bank"]])
        self.assertEqual(rules[0], {"below": 2.0, "field": "axis", "axis": [1.0, 0.0, 0.0], "sense": "polar",
                                    "polarity": "partner", "length": 4.9, "onset": 2.0})
        self.assertEqual(rules[1], {"below": None, "field": "plane", "normal": [1.0, 0.0, 0.0], "length": 5.0,
                                    "onset": 2.0, "bank": False})
        nematic = parse_rules([{"below": None, "field": "axis", "axis": [3, 4, 0], "length": 2}], None)
        self.assertEqual(nematic, [{"below": None, "field": "axis", "axis": [0.6, 0.8, 0.0], "sense": "nematic",
                                    "length": 2.0, "onset": 2.0}])
        self.assertEqual(parse_rules([{"below": None, "field": None}], None), [{"below": None, "field": None}])
        self.assertEqual(parse_rules([{"below": 3, "field": None}, {"below": None, "field": "plane", "normal": [0, 0, 2],
                                       "length": 1, "bank": True}], 2.0),
                         [{"below": 3.0, "field": None}, {"below": None, "field": "plane", "normal": [0.0, 0.0, 1.0],
                                                           "length": 1.0, "onset": 2.0, "bank": True}])
        for rule in rules + nematic:
            for value in rule.values():
                self.assertNotIsInstance(value, np.generic)
        json.dumps(rules)

    def test_parsing_normalised_rules_returns_them_unchanged(self):
        specs = ([{"below": 1.5, "field": "axis", "axis": [1, 1, 1], "sense": "polar", "polarity": "fixed",
                   "length": 3.3, "onset": 1.0},
                  {"below": 4, "field": "plane", "normal": [0.1, 0.2, 0.97], "length": 2, "bank": True},
                  {"below": None, "field": None}],
                 FAMILIES["aligned"]["guidance"])
        for spec in specs:
            rules = parse_rules(spec, 1.0)
            self.assertEqual(parse_rules(rules, 1.0), rules)
            self.assertEqual(parse_rules(json.loads(json.dumps(rules)), 1.0), rules)

    def test_every_malformed_specification_is_refused(self):
        axis = {"below": None, "field": "axis", "axis": [1, 0, 0], "length": 3}
        plane = {"below": None, "field": "plane", "normal": [0, 0, 1], "length": 3}
        bad = {
            "not a list": "axis", "empty": [], "a dict": {"field": "axis"}, "a rule that is not a dict": [["axis"]],
            "unknown key": [dict(axis, colour="red")],
            "axis on a plane": [dict(plane, axis=[1, 0, 0])], "sense on a plane": [dict(plane, sense="polar")],
            "polarity on a plane": [dict(plane, polarity="root")],
            "normal on an axis": [dict(axis, normal=[0, 0, 1])], "bank on an axis": [dict(axis, bank=True)],
            "length on field None": [{"below": None, "field": None, "length": 3}],
            "onset on field None": [{"below": None, "field": None, "onset": 0}],
            "field missing": [{"below": None, "axis": [1, 0, 0], "length": 3}],
            "unknown field": [dict(axis, field="spiral")],
            "axis missing": [{"below": None, "field": "axis", "length": 3}],
            "normal missing": [{"below": None, "field": "plane", "length": 3}],
            "a two-vector": [dict(axis, axis=[1, 0])], "a non-numeric vector": [dict(axis, axis=["1", 0, 0])],
            "a non-finite vector": [dict(axis, axis=[1, float("nan"), 0])],
            "a zero vector": [dict(axis, axis=[0, 0, 0])], "a tiny vector": [dict(axis, axis=[1e-13, 0, 0])],
            "sense unknown": [dict(axis, sense="axial")],
            "polarity unknown": [dict(axis, sense="polar", polarity="tip")],
            "polarity on a nematic axis": [dict(axis, polarity="root")],
            "bank not a bool": [dict(plane, bank=1)],
            "length missing": [{"below": None, "field": "axis", "axis": [1, 0, 0]}],
            "length zero": [dict(axis, length=0)], "length negative": [dict(axis, length=-1)],
            "length a bool": [dict(axis, length=True)],
            "onset negative": [dict(axis, onset=-0.5)], "onset non-finite": [dict(axis, onset=float("inf"))],
            "below missing": [{"field": "axis", "axis": [1, 0, 0], "length": 3}],
            "below zero": [dict(axis, below=0.0)], "below negative": [dict(axis, below=-2)],
            "below a string": [dict(axis, below="2")],
            "bounds not increasing": [dict(axis, below=2.0), dict(plane, below=2.0)],
            "bounds decreasing": [dict(axis, below=3.0), dict(plane, below=2.0)],
            "None before the last rule": [dict(axis, below=None), dict(plane, below=2.0)],
            "None twice": [dict(axis, below=None), dict(plane, below=None)],
            "an integer too large for a float": [dict(axis, length=10 ** 400)],
            "a vector component too large for a float": [dict(axis, axis=[10 ** 400, 0, 0])],
        }
        for name, spec in bad.items():
            with self.subTest(refusal=name):
                with self.assertRaises(ValueError):
                    parse_rules(spec, 1.0)
        with self.assertRaises(ValueError):
            parse_rules((axis,), 1.0)                               # a tuple is not a list
        with self.assertRaises(ValueError):
            parse_rules([dict(axis, below=2.0)], None)              # a finite bound needs d_min
        parse_rules([dict(axis, below=2.0)], 1.0)
        parse_rules([axis], None)

    def test_every_malformed_specification_is_refused_by_growth_and_the_command_line_before_writing(self):
        axis = {"below": None, "field": "axis", "axis": [1, 0, 0], "length": 3}
        plane = {"below": None, "field": "plane", "normal": [0, 0, 1], "length": 3}
        bad = {
            "empty": [], "a dict": {"field": "axis"}, "a rule that is not a dict": [["axis"]],
            "unknown key": [dict(axis, colour="red")], "axis on a plane": [dict(plane, axis=[1, 0, 0])],
            "unknown field": [dict(axis, field="spiral")], "axis missing": [{"below": None, "field": "axis", "length": 3}],
            "a zero vector": [dict(axis, axis=[0, 0, 0])], "sense unknown": [dict(axis, sense="axial")],
            "polarity on a nematic axis": [dict(axis, polarity="root")], "bank not a bool": [dict(plane, bank=1)],
            "length missing": [{"below": None, "field": "axis", "axis": [1, 0, 0]}],
            "onset negative": [dict(axis, onset=-0.5)], "below zero": [dict(axis, below=0.0)],
            "bounds not increasing": [dict(axis, below=2.0), dict(plane, below=2.0)],
            "None before the last rule": [dict(axis, below=None), dict(plane, below=2.0)],
            "an integer too large for a float": [dict(axis, length=10 ** 400)],
        }
        for name, spec in bad.items():
            with self.subTest(refusal=name):
                with self.assertRaises(ValueError):
                    grow(3, niter=3, d_min=2.0, guidance=spec, **WALK)
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        main.main(["--count", "1", "--seed", "3", "--iterations", "3", "3", "--volume", "48", "48",
                                   "24", "--tortuosity", "walk", "--persistence", "10", "--d-min", "2",
                                   "--guidance", json.dumps(spec), "--out", out])
                    self.assertNotEqual(caught.exception.code, 0)
                    self.assertEqual(os.listdir(out), [])

    def test_large_vectors_and_numpy_numbers_are_normalised_like_any_other(self):
        axis = {"below": None, "field": "axis", "axis": [1, 0, 0], "length": 3, "onset": 0.0}
        # a component beyond the square root of the largest float would overflow an unscaled norm
        for vector, unit in (([1e200, 0, 0], [1.0, 0.0, 0.0]), ([1.7e308, 1.7e308, 0], [2 ** -0.5, 2 ** -0.5, 0.0])):
            rules = parse_rules([dict(axis, axis=vector)], None)
            np.testing.assert_allclose(rules[0]["axis"], unit, rtol=0.0, atol=1e-15)
            self.assertEqual(parse_rules(rules, None), rules)
        numpy_numbers = parse_rules([dict(axis, below=np.int64(2), axis=[np.int32(3), np.float32(4), 0],
                                          length=np.float32(3), onset=np.int64(0)),
                                     {"below": None, "field": None}], 1.0)
        self.assertEqual(numpy_numbers, parse_rules([dict(axis, below=2, axis=[3, 4, 0]),
                                                     {"below": None, "field": None}], 1.0))
        for value in numpy_numbers[0].values():
            self.assertNotIsInstance(value, np.generic)
        with self.assertRaises(ValueError):
            parse_rules([dict(axis, axis=[np.bool_(True), 0, 0])], None)


class TurnTests(unittest.TestCase):
    def frames(self, rng, count):
        for _ in range(count):
            heading = rng.normal(size=3)
            heading /= np.linalg.norm(heading)
            perp = rng.normal(size=3)
            perp -= (perp @ heading) * heading
            perp /= np.linalg.norm(perp)
            target = rng.normal(size=3)
            target /= np.linalg.norm(target)
            yield heading, perp, target

    def assert_orthonormal(self, heading, perp):
        self.assertAlmostEqual(float(np.linalg.norm(heading)), 1.0, delta=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(perp)), 1.0, delta=1e-12)
        self.assertAlmostEqual(float(heading @ perp), 0.0, delta=1e-12)

    def test_the_turn_never_exceeds_the_angle_to_the_target_and_keeps_the_frame_orthonormal(self):
        rng = np.random.default_rng(3)
        for heading, perp, target in self.frames(rng, 200):
            for axial in (True, False):
                for rate in (0.05, 0.5, 1.0):
                    theta = math.acos(np.clip(heading @ target, -1.0, 1.0))
                    new_heading, new_perp, omega = guide_frame(heading, perp, target, rate, axial)
                    self.assertLessEqual(omega, theta + 1e-12)
                    self.assert_orthonormal(new_heading, new_perp)
                    # the turn is about h x t, so the new heading stays in the plane of h and t
                    self.assertAlmostEqual(float(new_heading @ np.cross(heading, target)), 0.0, delta=1e-12)
                    after = math.acos(np.clip(new_heading @ target, -1.0, 1.0))
                    if axial and theta > math.pi / 2:
                        self.assertGreaterEqual(after, theta - 1e-12)      # drifts to the nearer end, -t
                    else:
                        self.assertAlmostEqual(after, theta - omega, delta=1e-9)

    def test_steering_without_noise_converges_monotonically_and_stops_at_the_target(self):
        def angle_to(h, target):
            return math.atan2(float(np.linalg.norm(np.cross(h, target))), float(h @ target))

        rng = np.random.default_rng(4)
        for heading, perp, target in self.frames(rng, 20):
            for axial in (True, False):
                if axial and heading @ target < 0.0:
                    target = -target
                angle = angle_to(heading, target)
                h, p = heading, perp
                for _ in range(400):
                    h, p, omega = guide_frame(h, p, target, 0.3, axial)
                    after = angle_to(h, target)
                    self.assertLessEqual(after, angle + 1e-12)
                    angle = after
                self.assertLess(angle, 1e-9)
        heading, perp = X.copy(), Y.copy()
        same_heading, same_perp, omega = guide_frame(heading, perp, X, 1.0, True)
        self.assertIs(same_heading, heading)                                    # theta = 0: the inputs come back
        self.assertIs(same_perp, perp)
        self.assertEqual(omega, 0.0)

    def test_a_polar_target_directly_behind_exerts_no_torque(self):
        heading, perp = Y.copy(), Z.copy()
        same_heading, same_perp, omega = guide_frame(heading, perp, -Y, 1.0, False)
        self.assertIs(same_heading, heading)
        self.assertIs(same_perp, perp)
        self.assertEqual(omega, 0.0)
        # a hair off, the heading turns back towards the target
        nearly = -Y + 1e-6 * X
        turned, _, omega = guide_frame(heading, perp, nearly / np.linalg.norm(nearly), 1.0, False)
        self.assertGreater(omega, 0.0)
        self.assertGreater(float(turned @ X), 0.0)

    def test_an_undefined_plane_target_is_counted_and_leaves_the_frame_alone(self):
        guide = Guidance([PLANE_Z], 1.0).bind(Y)
        events = {key: 0 for key in EVENT_KEYS}
        heading, perp = Z.copy(), X.copy()                                     # along the normal
        same_heading, same_perp = guide.steer(heading, perp, np.zeros(3), guide.rules[0], 1.0, 0.1, 5.0, events)
        self.assertIs(same_heading, heading)
        self.assertIs(same_perp, perp)
        self.assertEqual(events, dict({key: 0 for key in EVENT_KEYS}, guidance_undefined_steps=1))

    def test_the_bank_turns_the_perpendicular_to_the_normal_and_is_skipped_when_already_there(self):
        heading = Y.copy()
        tilted = np.array([0.0, 0.0, 1.0]) * math.cos(0.4) + np.array([1.0, 0.0, 0.0]) * math.sin(0.4)
        perp, omega = bank_frame(heading, tilted, Z, 1.0)
        self.assertGreater(omega, 0.0)
        self.assertAlmostEqual(float(perp @ heading), 0.0, delta=1e-12)
        self.assertGreater(float(perp @ Z), float(tilted @ Z))
        self.assertAlmostEqual(math.acos(np.clip(perp @ Z, -1, 1)), 0.4 - omega, delta=1e-9)
        # perp along the normal, or along minus the normal, is where the bank rests
        for resting in (Z, -Z):
            same, omega = bank_frame(heading, resting, Z, 1.0)
            self.assertIs(same, resting)
            self.assertEqual(omega, 0.0)
        # the heading along the normal leaves no component to bank towards
        same, omega = bank_frame(Z.copy(), X.copy(), Z, 1.0)
        self.assertEqual(omega, 0.0)
        # over random frames and normals the bank turns by at most the angle to
        # its target and keeps the frame orthonormal
        rng = np.random.default_rng(5)
        for heading, perp, normal in self.frames(rng, 200):
            q = normal - (normal @ heading) * heading
            target = q / np.linalg.norm(q)
            if perp @ target < 0.0:
                target = -target
            psi = math.atan2(float(np.linalg.norm(np.cross(perp, target))), float(perp @ target))
            for rate in (0.05, 0.5, 1.0):
                banked, omega = bank_frame(heading, perp, normal, rate)
                self.assertLessEqual(omega, psi + 1e-12)
                self.assert_orthonormal(heading, banked)
                after = math.atan2(float(np.linalg.norm(np.cross(banked, target))), float(banked @ target))
                self.assertAlmostEqual(after, psi - omega, delta=1e-9)
        # a skipped bank is not a bank step, a bank is
        rules = [dict(PLANE_Z, bank=True)]
        guide = Guidance(rules, 1.0).bind(Y)
        events = {key: 0 for key in EVENT_KEYS}
        guide.steer(Y.copy(), Z.copy(), np.zeros(3), guide.rules[0], 1.0, 0.1, 5.0, events)
        self.assertEqual(events["guidance_bank_steps"], 0)
        self.assertEqual(events["guided_steps"], 1)
        guide.steer(Y.copy(), tilted, np.zeros(3), guide.rules[0], 1.0, 0.1, 5.0, events)
        self.assertEqual(events["guidance_bank_steps"], 1)
        self.assertEqual(events["guided_steps"], 2)


class WalkTests(unittest.TestCase):
    def test_the_walk_without_guidance_is_what_it_was(self):
        reference = grow(5, avoid_collisions=True, **WALK)
        for options in ({"guidance": None}, {"guidance": None, "root_offsets": None}):
            with self.subTest(options=options):
                same = grow(5, avoid_collisions=True, **WALK, **options)
                np.testing.assert_array_equal(same["nodes"], reference["nodes"])
                self.assertEqual(same["events"], reference["events"])
                self.assertEqual(same["frame"], reference["frame"])
        self.assertEqual(reference["frame"]["kind"], "none")
        self.assertEqual(reference["frame"]["rules"], [])
        for key in EVENT_KEYS:
            self.assertEqual(reference["events"][key], 0)
        # a WalkSettings without a guidance and one given None draw the same and leave the generator alike
        program = reference["program"]
        rows = []
        states = []
        for settings in (WalkSettings(10.0, np.random.default_rng([5, 1])),
                         WalkSettings(10.0, np.random.default_rng([5, 1]), guidance=None)):
            rows.append(np.array(list(branching_turtle_to_coords(program, 20.0, walk=settings))))
            states.append(settings.rng.bit_generator.state)
        np.testing.assert_array_equal(rows[0], rows[1])
        self.assertEqual(states[0], states[1])

    def test_zero_root_offsets_leave_the_roots_where_they_are(self):
        options = dict(FAMILIES["mesh"], d_min=2.0)
        reference = grow(6, **options)
        same = grow(6, root_offsets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], **options)
        np.testing.assert_array_equal(same["nodes"], reference["nodes"])
        self.assertEqual(same["root_positions_um"], reference["root_positions_um"])
        self.assertEqual(same["events"], reference["events"])
        moved = grow(6, root_offsets=[[2.0, 0.0, 0.0], [0.0, 0.0, -2.0]], **options)
        self.assertEqual(moved["root_positions_um"][0][0], reference["root_positions_um"][0][0] + 4.0)
        self.assertEqual(moved["root_positions_um"][0][1:], reference["root_positions_um"][0][1:])
        self.assertEqual(moved["frame"]["origin"], reference["frame"]["origin"])     # the box is the same

    def test_guided_steps_draw_exactly_what_unguided_steps_draw(self):
        moves = 50
        for rules in ([AXIS_X], [PLANE_Z], [dict(PLANE_Z, bank=True)],
                      [dict(AXIS_X, sense="polar", polarity="fixed")]):
            with self.subTest(field=rules[0]["field"]):
                guide = Guidance(rules, 1.0).bind(Y)
                walk = WalkSettings(10.0, np.random.default_rng([9, 1]), guidance=guide)
                list(branching_turtle_to_coords("{" + "f(1.4,1.0)" * moves + "}", 1.0, walk=walk, subdivisions=3))
                fresh = np.random.default_rng([9, 1])
                for _ in range(moves * 8):
                    fresh.normal(0.0, 1.0)
                    fresh.uniform(0.0, 2.0 * math.pi)
                self.assertEqual(walk.rng.bit_generator.state, fresh.bit_generator.state)

    def test_the_counters_partition_the_steps_and_flips_follow_the_sense(self):
        for rules, flips in (([dict(AXIS_X, length=30.0)], "some"),
                             ([dict(AXIS_X, sense="polar", polarity="fixed", length=30.0)], "none")):
            with self.subTest(sense=rules[0].get("sense", "nematic")):
                guide = Guidance(rules, 1.0).bind(Y)
                calls = [0]
                steer = guide.steer

                def counting(*args, _steer=steer, _calls=calls):
                    _calls[0] += 1
                    return _steer(*args)

                guide.steer = counting
                walk = WalkSettings(10.0, np.random.default_rng([2, 1]), guidance=guide)
                events = {}
                list(branching_turtle_to_coords("{" + "f(1.4,1.0)" * 300 + "}", 1.0, walk=walk, events=events,
                                                subdivisions=3))
                self.assertEqual(calls[0], 2400)
                self.assertEqual(events["guided_steps"] + events["guidance_onset_steps"] + events["unguided_steps"]
                                 + events["guidance_undefined_steps"], calls[0])
                if flips == "some":
                    self.assertGreater(events["guidance_sense_flips"], 0)         # G / P = 3: a weak nematic pull
                else:
                    self.assertEqual(events["guidance_sense_flips"], 0)
        # an unguided class and an onset are counted as such
        rules = [{"below": 0.5, "field": "axis", "axis": [1, 0, 0], "length": 3.0, "onset": 3.0},
                 {"below": 1.5, "field": None}, dict(AXIS_X, onset=3.0)]
        guide = Guidance(rules, 1.0).bind(Y)
        walk = WalkSettings(10.0, np.random.default_rng([2, 1]), guidance=guide)
        events = {}
        list(branching_turtle_to_coords("{" + "f(1.4,1.0)" * 20 + "}[{" + "f(2.8,2.0)" * 20 + "}]", 1.0, walk=walk,
                                        events=events, subdivisions=3))
        self.assertEqual(events["unguided_steps"], 160)                           # the stem of diameter 1
        self.assertEqual(events["guidance_onset_steps"], 18)        # arc < 3 d = 6 on steps of 0.35: 0 .. 17
        self.assertEqual(events["guided_steps"], 160 - 18)

    def test_the_rule_is_chosen_from_the_first_move_and_held_over_an_anomaly(self):
        # a stem of diameter 1 whose third move narrows to 0.5 and widens to 1.5: all under the
        # rule of diameter 1 (the axis), never the unguided class below 0.6
        rules = [{"below": 0.6, "field": None}, dict(AXIS_X, onset=0.0)]
        guide = Guidance(rules, 1.0).bind(Y)
        walk = WalkSettings(10.0, np.random.default_rng([2, 1]), guidance=guide)
        events = {}
        list(branching_turtle_to_coords("{f(1.4,1.0)f(1.4,1.0)f(1.4,0.5)f(1.4,1.5)f(1.4,1.0)}", 1.0, walk=walk,
                                        events=events, subdivisions=3))
        self.assertEqual(events["unguided_steps"], 0)
        self.assertEqual(events["guided_steps"], 40)
        # a stem that starts narrow is unguided throughout
        events = {}
        walk = WalkSettings(10.0, np.random.default_rng([2, 1]), guidance=guide)
        list(branching_turtle_to_coords("{f(1.4,0.5)f(1.4,1.0)}", 1.0, walk=walk, events=events, subdivisions=3))
        self.assertEqual(events["unguided_steps"], 16)
        self.assertEqual(events["guided_steps"], 0)
        # the continuation of a stem after a branch chooses again, from its own first move
        events = {}
        walk = WalkSettings(10.0, np.random.default_rng([2, 1]), guidance=guide)
        list(branching_turtle_to_coords("{f(1.4,1.0)[f(1.4,1.0)]f(1.4,0.5)}", 1.0, walk=walk, events=events,
                                        subdivisions=3))
        self.assertEqual(events["guided_steps"], 8)
        self.assertEqual(events["unguided_steps"], 8)


class StationaryLawTests(unittest.TestCase):
    """
    One stem of 2000 sub-segments at P 10, step 0.175 d, starting perpendicular
    to the axis, measured after a burn-in of 10 G. Tolerances are provisional
    and only ever tightened from measurement.

    The development measurement at G 3, seeds 1-4 (seeds 1-10 in brackets):
    nematic <cos^2 theta> 0.8211, per seed 0.8135-0.8254 (0.8258,
    0.8135-0.8350), against 0.8297 from the Watson law; polar <cos theta>
    0.9219, per seed 0.9199-0.9230 (0.9229, 0.9199-0.9263), against 0.9250
    from the Fisher law; plane <sin^2 beta> 0.0777, per seed 0.0740-0.0811
    (0.0771, 0.0708-0.0824), against 0.0747 from the girdle law. The
    concentrations that reproduce the means of seeds 1-4 are 0.96 of the
    laws' (Watson K 6.39 against 6.67, Fisher kappa 12.80 against 13.33,
    girdle K 6.41 against 6.67), a bias of the order of step / (G d) = 0.058.
    """

    SEEDS = (1, 2, 3, 4)
    BURN_IN = 10

    def moments(self, rules, G):
        values = []
        for seed in self.SEEDS:
            headings = stem_headings(rules, seed)
            skip = int(round(self.BURN_IN * G / 0.175))
            values.append(headings[skip:])
        return values

    def test_a_nematic_axis_settles_into_the_watson_law(self):
        per_seed = [float(np.mean((h @ X) ** 2)) for h in self.moments([AXIS_X], 3.0)]
        for value in per_seed:
            self.assertAlmostEqual(value, 0.830, delta=0.035)                   # K = 2 P / G = 6.67
        self.assertAlmostEqual(float(np.mean(per_seed)), 0.830, delta=0.02)

    def test_a_polar_axis_settles_into_the_fisher_law(self):
        rules = [dict(AXIS_X, sense="polar", polarity="fixed")]
        per_seed = [float(np.mean(h @ X)) for h in self.moments(rules, 3.0)]
        kappa = 4.0 * 10.0 / 3.0
        expected = 1.0 / math.tanh(kappa) - 1.0 / kappa                          # 0.925
        self.assertAlmostEqual(expected, 0.925, delta=0.001)
        self.assertAlmostEqual(float(np.mean(per_seed)), expected, delta=0.015)

    def test_a_plane_settles_into_the_girdle_law(self):
        rules = [{"below": None, "field": "plane", "normal": [1, 0, 0], "length": 3.0, "onset": 0.0}]
        per_seed = [float(np.mean((h @ X) ** 2)) for h in self.moments(rules, 3.0)]
        self.assertAlmostEqual(float(np.mean(per_seed)), 0.0747, delta=0.015)    # K = 2 P / G = 6.67

    @SLOW
    def test_the_order_falls_as_the_guidance_length_grows(self):
        # twelve stems, so it is gated; S per seed 1-4 measured 0.915-0.917 at
        # G 1, 0.720-0.738 at G 3 and 0.249-0.353 at G 10
        orders = [[float(np.mean((3.0 * (h @ X) ** 2 - 1.0) / 2.0)) for h in self.moments([dict(AXIS_X, length=G)], G)]
                  for G in (1.0, 3.0, 10.0)]
        for seed, (at_1, at_3, at_10) in zip(self.SEEDS, zip(*orders)):
            with self.subTest(seed=seed):
                self.assertGreater(at_1, at_3)
                self.assertGreater(at_3, at_10)


class SenseTests(unittest.TestCase):
    def per_tree_alignment(self, grown, axis):
        tangent, length, _, label = edge_tangents(grown)
        u = tangent.T @ axis
        return [float(np.sum(length[label == k] * u[label == k]) / np.sum(length[label == k])) for k in (0, 1)]

    def test_a_root_rule_steers_each_tree_along_its_own_root_heading(self):
        rules = [{"below": None, "field": "axis", "axis": [0, 1, 0], "sense": "polar", "polarity": "root",
                  "length": 3.0, "onset": 2.0}]
        # measured on seeds 1-3: 0.81 to 0.83 on tree 0 and -0.78 to -0.80 on tree 1
        grown = grow(1, anastomose=True, anastomose_mode="arteriovenous", anastomosis_fraction=0.0, guidance=rules,
                     **WALK)
        along = self.per_tree_alignment(grown, Y)
        self.assertGreater(along[0], 0.7)
        self.assertLess(along[1], -0.7)
        self.assertEqual(grown["frame"]["tree_senses"], [1, -1])
        self.assertEqual(grown["frame"]["sense"], "polar")

    def test_a_partner_rule_steers_each_tree_towards_the_other_root(self):
        grown = grow_preset("aligned", 4.0, 2)
        self.assertEqual(grown["frame"]["tree_senses"], [1, -1])
        tangent, length, diameter, label = edge_tangents(grown)
        u = tangent.T @ X
        sense = np.where(label == 0, 1.0, np.where(label == 1, -1.0, 0.0))
        capillary = (diameter < 2.0) & (sense != 0.0)
        polar_order = float(np.sum(length[capillary] * sense[capillary] * u[capillary]) / np.sum(length[capillary]))
        self.assertGreater(polar_order, 0.0)
        flipped = grow_preset("aligned", 4.0, 2, root_offsets=[[15.0, 0.0, 0.0], [-15.0, 0.0, 0.0]])
        self.assertEqual(flipped["frame"]["tree_senses"], [-1, 1])

    def test_a_partner_rule_is_refused_without_a_partner_to_steer_towards(self):
        partner = [{"below": None, "field": "axis", "axis": [1, 0, 0], "sense": "polar", "polarity": "partner",
                    "length": 3.0}]
        box = dict(WALK, grow_in_volume=True, d_min=2.0)
        with self.assertRaisesRegex(ValueError, "partner"):
            grow(3, guidance=partner, **box)                                            # one tree
        pair = dict(box, anastomose=True, anastomose_mode="arteriovenous")
        with self.assertRaisesRegex(ValueError, "partner"):
            grow(3, guidance=partner, **pair)                                           # no root offsets
        with self.assertRaisesRegex(ValueError, "partner"):
            grow(3, guidance=partner, root_offsets=[[0, 0, 1], [0, 0, -1]], **pair)     # equal along the axis
        grown = grow(3, guidance=partner, root_offsets=[[-1, 0, 0], [1, 0, 0]], **pair)
        self.assertEqual(grown["frame"]["tree_senses"], [1, -1])
        for argv in (["--guidance", json.dumps(partner), "--tortuosity", "walk", "--persistence", "10", "--d-min", "2",
                      "--grow-in-volume"],
                     ["--guidance", json.dumps(partner), "--family", "mesh", "--d-min", "2"],
                     ["--guidance", json.dumps(partner), "--family", "mesh", "--d-min", "2",
                      "--root-offsets", "[[0, 0, 1], [0, 0, -1]]"]):
            with self.subTest(argv=argv):
                args = main.build_parser(argv[argv.index("--family") + 1] if "--family" in argv else "tree").parse_args(argv)
                with self.assertRaisesRegex(SystemExit, "partner"):
                    main.validate_shaping(args)


class GrowthTests(unittest.TestCase):
    def test_the_free_extent_is_that_of_the_guided_growth(self):
        rules = [dict(AXIS_X, sense="polar", polarity="root")]
        grown = grow(4, guidance=rules, **WALK)
        finite = np.isfinite(grown["nodes"][0])
        points = grown["nodes"][:3, finite]
        expected = points.max(axis=1) - points.min(axis=1)
        guide = Guidance(rules, None).bind(Y)
        walk = main._walk_settings("walk", 10.0, 4, guide)
        extent, along = main._free_extent(grown["program"], 20.0, Y, Z, walk)
        np.testing.assert_allclose(extent, expected, rtol=0.0, atol=0.0)
        self.assertEqual(along, float(points[1].max()))
        # and so the box of a guided network is sized to the guided tree, not to an unguided one
        plain = grow(4, **WALK)
        self.assertFalse(np.allclose(plain["nodes"][:3, np.isfinite(plain["nodes"][0])].max(axis=1)
                                     - plain["nodes"][:3, np.isfinite(plain["nodes"][0])].min(axis=1), expected))
        # grow_network sizes its box from that extent, and places a free second root at its reach
        boxed = grow(4, guidance=rules, grow_in_volume=True, **WALK)
        shape = np.asarray(VOLUME, dtype=float)
        np.testing.assert_allclose(boxed["growth_box_um"], shape * np.min(expected / shape), rtol=1e-12, atol=0.0)
        pair = grow(4, anastomose=True, anastomose_mode="arteriovenous", anastomosis_fraction=0.0, guidance=rules,
                    **WALK)
        np.testing.assert_array_equal(pair["root_positions_um"][1], along * Y)

    def test_growth_scales_with_the_unit(self):
        reference = grow_preset("aligned", 3.0, 3)
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                argv = library.member_argv("aligned", 3.0, d_min=scale, collision_margin=scale)
                args = main.build_parser("aligned").parse_args(argv)
                random.seed(3)
                np.random.seed(3)
                properties, d0, niter = main.sample_parameters(args)
                self.assertEqual(d0, 3.0 * scale)
                scaled = grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit,
                                      clip_axes=tuple(args.clip_axes), voxel_size=args.voxel_size,
                                      subdivisions=args.subdivisions, d_min=args.d_min,
                                      grow_in_volume=args.grow_in_volume, seed=3, **main.shaping_options(args))
                self.assertEqual(scaled["nodes"].shape, reference["nodes"].shape)
                # rounding of order 1e-16 of the box shows as a large relative error on a
                # coordinate within a hair of a face (up to 2.4e-10 over the seeds below), so
                # the comparison has an absolute floor. Over seeds 1-10 the largest difference
                # is 1.4e-15 of scale x the box side at R 3 and 2.2e-15 at R 5; the floor is 1e-14
                floor = 1e-14 * max(reference["growth_box_um"])
                np.testing.assert_allclose(scaled["nodes"], scale * reference["nodes"], rtol=1e-12,
                                           atol=scale * floor)
                self.assertEqual(scaled["events"], reference["events"])
                np.testing.assert_allclose(scaled["frame"]["origin"], scale * np.asarray(reference["frame"]["origin"]),
                                           rtol=1e-12)
                self.assertEqual(scaled["frame"]["rules"], reference["frame"]["rules"])

    def test_every_root_offset_refusal_fires(self):
        box = dict(WALK, grow_in_volume=True, d_min=2.0)
        pair = dict(box, anastomose=True, anastomose_mode="arteriovenous")
        refused = {
            "one vector for two trees": (pair, [[1, 0, 0]]),
            "two vectors for one tree": (box, [[1, 0, 0], [0, 0, 1]]),
            "a two-vector": (box, [[1, 0]]),
            "not a number": (box, [["1", 0, 0]]),
            "not finite": (box, [[float("nan"), 0, 0]]),
            "a string": (box, "[[1, 0, 0]]"),
            "without grow_in_volume": (dict(WALK, d_min=2.0), [[1, 0, 0]]),
            "without d_min": (dict(WALK, grow_in_volume=True), [[1, 0, 0]]),
            "without the walk": ({"grow_in_volume": True, "d_min": 2.0}, [[1, 0, 0]]),   # offsets need no walk
        }
        for name, (options, offsets) in refused.items():
            with self.subTest(refusal=name):
                if name == "without the walk":
                    grow(3, root_offsets=offsets, **options)                               # allowed
                    continue
                with self.assertRaises(ValueError):
                    grow(3, root_offsets=offsets, **options)
        with self.assertRaises(RootOutsideBox):
            grow(3, root_offsets=[[-1000.0, 0.0, 0.0]], **box)
        with self.assertRaises(ValueError):
            grow(3, root_offsets=[[10 ** 400, 0, 0]], **box)                              # too large for a float
        # numpy rows and numpy numbers are vectors and numbers like any other
        listed = grow(3, root_offsets=[[1.0, 0.0, 0.0]], **box)
        for offsets in ([np.array([1.0, 0.0, 0.0])], [[np.int64(1), np.float32(0), 0]], np.array([[1.0, 0.0, 0.0]])):
            np.testing.assert_array_equal(grow(3, root_offsets=offsets, **box)["nodes"], listed["nodes"])
        with self.assertRaises(ValueError):
            grow(3, guidance=[AXIS_X], grow_in_volume=True, d_min=2.0)                    # guidance needs the walk

    def test_the_command_line_refuses_what_grow_network_refuses_before_writing(self):
        base = ["--count", "1", "--seed", "3", "--volume", "48", "48", "24", "--iterations", "4", "4"]
        refused = (
            ["--guidance", json.dumps([AXIS_X])],                                       # without the walk
            ["--guidance", json.dumps([dict(AXIS_X, length=0)]), "--tortuosity", "walk", "--persistence", "10"],
            ["--guidance", json.dumps([dict(AXIS_X, below=2.0)]), "--tortuosity", "walk", "--persistence", "10"],
            ["--root-offsets", "[[1, 0, 0]]"],                                            # without the box and d_min
            ["--root-offsets", "[[1, 0, 0]]", "--grow-in-volume"],
            ["--root-offsets", "[[1, 0, 0], [0, 0, 1]]", "--grow-in-volume", "--d-min", "2"],
            ["--root-offsets", "[[-1000, 0, 0]]", "--grow-in-volume", "--d-min", "2", "--fit", "voxel_size",
             "--voxel-size", "2"],
            ["--root-offsets", "[[-1000, 0, 0]]", "--grow-in-volume", "--d-min", "2"],  # found once the box is known
            ["--family", "aligned"],                                                     # needs --d-min
        )
        for extra in refused:
            with self.subTest(argv=extra):
                with tempfile.TemporaryDirectory() as out:
                    with self.assertRaises(SystemExit) as caught:
                        with contextlib.redirect_stdout(io.StringIO()):
                            main.main(base + extra + ["--out", out])
                    self.assertNotEqual(caught.exception.code, 0)
                    self.assertEqual(os.listdir(out), [])
        for extra in (["--guidance", "not json"], ["--root-offsets", "[1, 0, 0"]):
            with self.subTest(argv=extra):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main.build_parser().parse_args(extra)

    def test_the_command_line_options_parse_inline_json(self):
        args = main.build_parser().parse_args(["--guidance", json.dumps([AXIS_X]), "--root-offsets", "[[1, 0, 0]]"])
        self.assertEqual(args.guidance, [AXIS_X])
        self.assertEqual(args.root_offsets, [[1, 0, 0]])
        self.assertEqual(main.shaping_options(args)["guidance"], [AXIS_X])
        self.assertEqual(main.shaping_options(args)["root_offsets"], [[1, 0, 0]])
        preset = main.build_parser("aligned").parse_args([])
        self.assertEqual(preset.guidance, FAMILIES["aligned"]["guidance"])
        self.assertEqual(preset.root_offsets, FAMILIES["aligned"]["root_offsets"])
        plain = main.build_parser("tree").parse_args([])
        self.assertIsNone(plain.guidance)
        self.assertIsNone(plain.root_offsets)


class FrameRecordTests(unittest.TestCase):
    def test_the_presets_without_guidance_record_no_frame_vector(self):
        for family in ("tree", "mesh", "tumour"):
            with self.subTest(family=family):
                properties = dict(PROPERTIES)
                options = dict(FAMILIES[family])
                for key in ("aneurysm_prob", "stenosis_prob"):
                    if key in options:
                        properties[key] = options.pop(key)
                options.pop("d0", None)
                grown = grow(2, properties=properties, **options)
                frame = grown["frame"]
                self.assertEqual(frame["kind"], "none")
                self.assertEqual(frame["rules"], [])
                self.assertEqual(frame["frame_version"], 1)
                for key in ("axis", "sense", "tree_senses", "normal"):
                    self.assertIsNone(frame[key])
                self.assertEqual(frame["grow_direction"], [0.0, 1.0, 0.0])
                self.assertEqual(frame["grow_perpendicular"], [0.0, 0.0, 1.0])
                if grown["growth_box_um"] is None:
                    self.assertEqual(frame["origin"], grown["root_positions_um"][0])
                else:
                    self.assertEqual(frame["origin"], [v / 2.0 for v in grown["growth_box_um"]])
                json.dumps(frame)

    def test_the_aligned_preset_records_its_axis_and_senses(self):
        grown = grow_preset("aligned", 3.0, 5)
        frame = grown["frame"]
        self.assertEqual(frame["kind"], "axis")
        self.assertEqual(frame["axis"], [1.0, 0.0, 0.0])
        self.assertEqual(frame["sense"], "polar")
        self.assertEqual(frame["tree_senses"], [1, -1])
        self.assertIsNone(frame["normal"])
        self.assertEqual(frame["rules"], parse_rules(FAMILIES["aligned"]["guidance"], 1.0))
        self.assertEqual(frame["origin"], [v / 2.0 for v in grown["growth_box_um"]])
        self.assertEqual(list(frame), ["frame_version", "kind", "axis", "sense", "tree_senses", "normal", "origin",
                                       "grow_direction", "grow_perpendicular", "rules"])
        # the recorded rule is the field rule of smallest bound: a plane first gives a plane frame
        planar = grow(5, guidance=[dict(PLANE_Z, below=3.0), dict(AXIS_X)], d_min=1.0, **WALK)
        self.assertEqual(planar["frame"]["kind"], "plane")
        self.assertEqual(planar["frame"]["normal"], [0.0, 0.0, 1.0])
        self.assertIsNone(planar["frame"]["axis"])
        self.assertEqual(len(planar["frame"]["rules"]), 2)
        unguided_class = grow(5, guidance=[{"below": None, "field": None}], **WALK)
        self.assertEqual(unguided_class["frame"]["kind"], "none")
        self.assertEqual(unguided_class["frame"]["rules"], [{"below": None, "field": None}])

    def test_the_record_is_the_same_in_the_return_the_sidecar_and_the_archive(self):
        argv = ["--family", "aligned", "--d-min", "2", "--count", "1", "--seed", "6", "--volume", "48", "48", "24",
                "--iterations", "4", "4", "--fit", "voxel_size", "--voxel-size", "2"]
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(argv + ["--out", out]), 0)
            stem = next(name[:-5] for name in os.listdir(out) if name.endswith(".json"))
            with open(os.path.join(out, stem + ".json")) as handle:
                sidecar = json.load(handle)
            network = main.load_network(os.path.join(out, stem + ".npz"))
        args = main.build_parser("aligned").parse_args(argv)
        random.seed(6)
        np.random.seed(6)
        properties, d0, niter = main.sample_parameters(args)
        grown = grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit, clip_axes=tuple(args.clip_axes),
                             voxel_size=args.voxel_size, subdivisions=args.subdivisions, d_min=args.d_min,
                             grow_in_volume=args.grow_in_volume, seed=6, **main.shaping_options(args))
        self.assertEqual(sidecar["frame"], grown["frame"])
        self.assertEqual(network["metadata"]["frame"], grown["frame"])
        self.assertEqual(sidecar["guidance"], FAMILIES["aligned"]["guidance"])
        self.assertEqual(sidecar["root_offsets"], FAMILIES["aligned"]["root_offsets"])
        self.assertEqual(sidecar["family"], "aligned")
        self.assertEqual(sidecar["trees"], 2)
        for key in EVENT_KEYS:
            self.assertIn(key, sidecar["events"])
        self.assertGreater(sidecar["events"]["guided_steps"], 0)


class FamilyTests(unittest.TestCase):
    def test_aligned_runs_end_to_end_on_the_command_line_and_in_the_connectivity_check(self):
        import check_connectivity
        argv = ["--family", "aligned", "--d-min", "2", "--count", "1", "--seed", "8", "--volume", "48", "48", "24",
                "--iterations", "5", "5", "--fit", "voxel_size", "--voxel-size", "2"]
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(main.main(argv + ["--out", out]), 0)
            names = sorted(os.listdir(out))
            self.assertEqual([os.path.splitext(name)[1] for name in names], [".json", ".npz", ".tiff"])
            network = main.load_network(os.path.join(out, names[1]))
        self.assertIn("guided_steps=", printed.getvalue())
        self.assertEqual(network["metadata"]["family"], "aligned")
        self.assertEqual(network["metadata"]["frame"]["kind"], "axis")
        self.assertEqual(sorted(set(network["tree"][network["tree"] >= 0].tolist())), [0, 1, 2])
        with contextlib.redirect_stdout(io.StringIO()):
            failed = check_connectivity.generate_and_check(1, argv)
        self.assertEqual(failed, 0)

    def test_a_family_without_a_preset_and_an_unknown_name_are_refused(self):
        with mock.patch.dict(FAMILIES, {"unoffered": None}):
            with tempfile.TemporaryDirectory() as out:
                with self.assertRaises(SystemExit) as caught:
                    main.main(["--family", "unoffered", "--count", "1", "--seed", "1", "--out", out])
                self.assertIn("unoffered", str(caught.exception))
                self.assertEqual(os.listdir(out), [])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main.main(["--family", "capillary", "--count", "1", "--seed", "1"])


class ClassTests(unittest.TestCase):
    """
    The aligned preset at R 5 over seeds 1-10, guided and with guidance None,
    both described with the guided run's frame. The margins below are half
    the mean differences of the development measurement: capillary-class
    S_x 0.141 guided against -0.002 unguided (difference 0.143, smallest
    0.098), capillary polar order 0.485 against 0.063 (0.421, smallest
    0.245) and larger-class S_x -0.165 against -0.043 (-0.123).
    """

    SEEDS = range(1, 11)
    MARGIN_CAPILLARY = 0.07
    MARGIN_POLAR = 0.2
    MARGIN_LARGER = 0.06

    @SLOW
    def test_guidance_orders_the_capillaries_and_flattens_the_feeders(self):
        from describe import describe
        guided, unguided = [], []
        for seed in self.SEEDS:
            grown = grow_preset("aligned", 5.0, seed)
            plain = grow_preset("aligned", 5.0, seed, guidance=None)
            for store, network in ((guided, grown), (unguided, plain)):
                report = describe(network["nodes"], network["edges"], tree=network["tree"], frame=grown["frame"],
                                  d_ref=1.0)
                store.append((report["frame_orientation"]["capillary"]["S"], report["polar_order"]["capillary"],
                              report["frame_orientation"]["larger"]["S"]))
        guided, unguided = np.array(guided), np.array(unguided)
        gain = guided.mean(axis=0) - unguided.mean(axis=0)
        print(f"\nclasses: capillary S guided {guided[:, 0].mean():.3f} unguided {unguided[:, 0].mean():.3f}, "
              f"polar order {guided[:, 1].mean():.3f} / {unguided[:, 1].mean():.3f}, "
              f"larger S {guided[:, 2].mean():.3f} / {unguided[:, 2].mean():.3f}")
        self.assertGreater(gain[0], self.MARGIN_CAPILLARY)
        self.assertGreater(gain[1], self.MARGIN_POLAR)
        self.assertLess(gain[2], -self.MARGIN_LARGER)


class BankTests(unittest.TestCase):
    """
    A plane field (normal z, G 2) on a walked tree with the default roll of
    70 degrees, bank off and on, over seeds 1-10. The margin is half the mean
    gain of the development measurement in the length-weighted in-plane
    fraction over all classes: 0.0377 (smallest 0.0233).
    """

    MARGIN = 0.018

    @SLOW
    def test_bank_raises_the_in_plane_fraction(self):
        from describe import describe
        gains = []
        for seed in range(1, 11):
            fractions = []
            for bank in (False, True):
                rules = [{"below": None, "field": "plane", "normal": [0, 0, 1], "length": 2.0, "onset": 2.0,
                          "bank": bank}]
                grown = grow(seed, niter=64, d0=5.0, properties=dict(PROPERTIES, roll_angle=70.0), d_min=1.0,
                             guidance=rules, **WALK)
                report = describe(grown["nodes"], grown["edges"], frame=grown["frame"], d_ref=1.0)
                fractions.append(report["frame_orientation"]["all"]["in_plane_fraction"])
            gains.append(fractions[1] - fractions[0])
        print(f"\nbank: mean gain in the in-plane fraction {np.mean(gains):.4f}, smallest {np.min(gains):.4f}")
        self.assertGreater(float(np.mean(gains)), self.MARGIN)


class OnsetTests(unittest.TestCase):
    """
    The aligned preset at R 5 over seeds 1-10 with the onset of both rules
    at 2 (as released) and at 0. The measure is the collision redraws per
    walk step, guided and onset steps together, averaged over the ten
    seeds (the mean of the per-seed ratios); it holds over the ten seeds
    taken together, not seed by seed.

    The development measurement: 0.0812 with onset 2 against 0.0902 with
    onset 0 (pooled over the seeds 0.0800 against 0.0885), lower in 6 seeds
    of 10; total redraws 416 against 515 and terminations 19.5 against 24.7
    per network, each lower in 9 seeds of 10; capillary S_x 0.141 against
    0.248. Steps within the first 2 diameters of their polyline are redrawn
    at 0.148 with onset 2 and 0.166 without, later steps at 0.034 and 0.031,
    and they are about 0.37 of all walk steps under either setting. Per
    guided step the ratio is 0.123 against 0.090, higher with the onset,
    because the redraws of the onset steps stay in the count while the onset
    steps leave the guided steps it is divided by; this test therefore
    measures per walk step. The margin is half the mean difference.
    """

    MARGIN = 0.004

    @SLOW
    def test_the_onset_lowers_the_redraws_per_walk_step(self):
        without_onset = [dict(rule, onset=0.0) for rule in FAMILIES["aligned"]["guidance"]]
        rows = []
        for seed in range(1, 11):
            pair = []
            for rules in (None, without_onset):
                grown = grow_preset("aligned", 5.0, seed) if rules is None else grow_preset("aligned", 5.0, seed,
                                                                                            guidance=rules)
                events = grown["events"]
                pair.append((events["collision_redraws"] / (events["guided_steps"] + events["guidance_onset_steps"]),
                             events["collision_terminations"], order_parameter(grown, X, edge_tangents(grown)[2] < 2.0)))
            rows.append(pair)
        rows = np.array(rows)
        print(f"\nonset: redraws per walk step {rows[:, 0, 0].mean():.4f} with onset 2, {rows[:, 1, 0].mean():.4f} "
              f"without; terminations {rows[:, 0, 1].mean():.1f} / {rows[:, 1, 1].mean():.1f}; capillary S_x "
              f"{rows[:, 0, 2].mean():.3f} / {rows[:, 1, 2].mean():.3f}")
        self.assertLess(rows[:, 0, 0].mean(), rows[:, 1, 0].mean() - self.MARGIN)


if __name__ == "__main__":
    unittest.main()
