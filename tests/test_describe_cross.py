"""
Tests of the descriptors describe.py appended in 3.8 for beds of capillaries
that run along an axis: "cross_connections" (the transverse capillary
segments between junctions of capillaries, the unbranched runs along the
axis and the kinds of junction) and "cross_section" (the arrangement of the
capillary crossings on the planes of "transverse_spacing").

Every network here is constructed, so that each quantity is known in
advance: a ladder of two straight capillaries along the axis with
transverse rungs, junctions of three straight branches, an H whose bar
stands at a chosen angle to the axis, square and hexagonal arrays of straight
lanes. The keys describe wrote before are held byte for byte to those of
the 3.7 modules (tests/fixtures/reference_code_3_7/describe.py) on the
fixture archives and on these networks; the 3.7 pin holds them on the
archives it writes.

Run from the repository root with
    python -m unittest tests.test_describe_cross
"""
import importlib.util
import json
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import describe  # noqa: E402
from describe import CROSS_ANGLE_DEG, describe_archive  # noqa: E402

FIXTURES = [os.path.join(ROOT, "tests", "fixtures", name) for name in ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz",
                                                                       "Lnet_i9_s3.npz")]
REFERENCE_DESCRIBE = os.path.join(ROOT, "tests", "fixtures", "reference_code_3_7", "describe.py")
NEW_KEYS_3_8 = ("cross_connections", "cross_section")


def frame(axis=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0), kind="axis"):
    """A frame record as the generator writes it, about an axis (or a plane of that normal), without rules."""
    return {"frame_version": 1, "kind": kind, "axis": [float(v) for v in axis] if kind == "axis" else None,
            "sense": "nematic" if kind == "axis" else None, "tree_senses": None,
            "normal": [float(v) for v in axis] if kind == "plane" else None,
            "origin": [float(v) for v in origin], "grow_direction": [0.0, 1.0, 0.0],
            "grow_perpendicular": [0.0, 0.0, 1.0], "rules": []}


def network(polylines):
    """(4, N) nodes of the polylines, each a list of (x, y, z, diameter) points, separated by NaN columns."""
    columns = []
    for points in polylines:
        if columns:
            columns.append(np.full((4, 1), np.nan))
        columns.append(np.array(points, dtype=float).T)
    return np.concatenate(columns, axis=1)


def line(start, end, points, diameter):
    """`points` evenly spaced points from start to end, each of the given diameter."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return [tuple(start + (end - start) * k / (points - 1)) + (diameter,) for k in range(points)]


# The ladder: two rails along x, W apart, with K rungs across them every L,
# sampled every L / 2 along the rails; diameter D, d_ref D.
K, L, W, D = 4, 8.0, 5.0, 0.5


def ladder(scale=1.0):
    rails = [line((0.0, y, 10.0), ((K + 1) * L, y, 10.0), 2 * (K + 1) + 1, D) for y in (0.0, W)]
    rungs = [line((L * (k + 1), 0.0, 10.0), (L * (k + 1), W, 10.0), 3, D) for k in range(K)]
    return network(rails + rungs) * scale


def star(directions, diameter=1.0, length=10.0, points=5):
    """Three straight branches of the given directions out of one junction at (20, 20, 20)."""
    centre = np.full(3, 20.0)
    return network([line(centre, centre + length * np.asarray(u, dtype=float), points, diameter) for u in directions])


def h_shape(degrees, feeder_diameter=None):
    """
    Two capillaries along x through (0, 0, 0) and through Q = (W cot t, W, 0),
    joined by a bar of four edges from the one to the other at t = `degrees`
    to x; diameter 1. With `feeder_diameter`, the second capillary is
    replaced by a feeder of that diameter ending at Q and a capillary leaving
    Q along +x, so that Q is the feeder's tip.
    """
    t = math.radians(degrees)
    q = np.array([W / math.tan(t), W, 0.0])
    polylines = [line((-10.0, 0.0, 0.0), (10.0, 0.0, 0.0), 5, 1.0), line((0.0, 0.0, 0.0), q, 5, 1.0)]
    if feeder_diameter is None:
        polylines.append(line(q - (10.0, 0.0, 0.0), q + (10.0, 0.0, 0.0), 5, 1.0))
    else:
        polylines += [line(q - (10.0, 0.0, 0.0), q, 5, feeder_diameter), line(q, q + (10.0, 0.0, 0.0), 5, 1.0)]
    return network(polylines)


def lanes(positions, z0=0.0, z1=10.0, points=5, diameter=1.0):
    """Straight lanes along z through the given (x, y) positions."""
    return [line((x, y, z0), (x, y, z1), points, diameter) for x, y in positions]


def unit_axis(degrees):
    t = math.radians(degrees)
    return (math.cos(t), math.sin(t), 0.0)


class CrossConnectionTests(unittest.TestCase):
    def test_a_ladder_is_measured_exactly(self):
        report = describe.describe(ladder(), frame=frame(origin=(20.0, 2.5, 10.0)), d_ref=D)
        cross = report["cross_connections"]
        rails, rungs = 2 * (K + 1) * L, K * W           # capillary length along the rails and in the rungs
        self.assertEqual(cross["count"], K)
        self.assertEqual(cross["per_length"], K / (rails + rungs))
        self.assertEqual(cross["per_length_d"], K / (rails + rungs) * D)
        self.assertEqual(cross["end_spacing_d"], rails / (2.0 * K * D))
        self.assertEqual(cross["chord_d"], {"median": W / D, "p10": W / D, "p90": W / D})
        self.assertEqual(cross["length_share"], rungs / (rails + rungs))
        # the runs of the rails between rungs and from each rail's ends to its first and last rung
        self.assertEqual(cross["axial_segments_d"],
                         {"count": 2 * (K + 1), "median": L / D, "mean": L / D, "p10": L / D, "p90": L / D})
        self.assertEqual(cross["junction_types"], {"count": 2 * K, "T": 1.0, "Y": 0.0, "other": 0.0})

    def test_the_kinds_of_junction(self):
        down = (-1.0, 0.0, 0.0)
        cases = (
            # a symmetric fork: the parent along -a, the daughters 30 degrees either side of +a
            ("fork", [down, unit_axis(30.0), unit_axis(-30.0)], "Y"),
            # an oblique merge: +a, -a and -a turned by 35 degrees
            ("merge", [(1.0, 0.0, 0.0), down, unit_axis(180.0 - 35.0)], "Y"),
            # a through capillary with a branch across it
            ("through", [(1.0, 0.0, 0.0), down, (0.0, 0.0, 1.0)], "T"),
            # three branches on one side of the transverse plane: 10, 40 and 70 degrees from +a
            ("one side", [unit_axis(10.0), unit_axis(40.0), unit_axis(70.0)], "other"),
            ("all along", [unit_axis(5.0), unit_axis(20.0), unit_axis(-15.0)], "other"),
        )
        for name, directions, kind in cases:
            with self.subTest(junction=name):
                types = describe.describe(star(directions), frame=frame(), d_ref=1.0)["cross_connections"][
                    "junction_types"]
                self.assertEqual(types["count"], 1)
                self.assertEqual(types[kind], 1.0)
                self.assertEqual(types["T"] + types["Y"] + types["other"], 1.0)
        # the ladder's rung junctions are T
        types = describe.describe(ladder(), frame=frame(), d_ref=D)["cross_connections"]["junction_types"]
        self.assertEqual((types["count"], types["T"]), (2 * K, 1.0))
        # a junction where a larger vessel meets is left out
        types = describe.describe(star([down, unit_axis(30.0), unit_axis(-30.0)], diameter=3.0), frame=frame(),
                                  d_ref=1.0)["cross_connections"]["junction_types"]
        self.assertEqual(types, {"count": 0, "T": None, "Y": None, "other": None})

    def test_a_bar_across_the_axis_is_a_cross_connection_and_a_bar_along_it_a_run(self):
        across = math.degrees(math.acos(math.cos(math.radians(CROSS_ANGLE_DEG))))
        for degrees, counted in ((90.0, True), (61.0, True), (59.0, False), (50.0, False)):
            with self.subTest(degrees=degrees):
                cross = describe.describe(h_shape(degrees), frame=frame(), d_ref=1.0)["cross_connections"]
                self.assertEqual(cross["count"], int(counted))
                # the bar's chord and the four runs of the two capillaries
                bar = W / math.sin(math.radians(degrees))
                self.assertEqual(cross["axial_segments_d"]["count"], 4 + (not counted))
                if counted:
                    self.assertAlmostEqual(cross["chord_d"]["median"], bar, places=12)
                    self.assertAlmostEqual(cross["length_share"], bar / (40.0 + bar), places=12)
                else:
                    self.assertEqual(cross["chord_d"], {"median": None, "p10": None, "p90": None})
                    self.assertIsNone(cross["end_spacing_d"])
                    self.assertEqual(cross["length_share"], 0.0)
        self.assertAlmostEqual(across, CROSS_ANGLE_DEG, places=9)

    def test_a_transverse_edge_at_a_feeder_tip_is_not_counted(self):
        # the bar's mean diameter, (2 + 1 + 1 + 1) / 4, keeps it capillary-class beside the feeder tip
        with_feeder = describe.describe(h_shape(90.0, feeder_diameter=3.0), frame=frame(), d_ref=1.0)
        self.assertEqual(with_feeder["degree_histogram"]["3"], 2)
        self.assertEqual(with_feeder["cross_connections"]["count"], 0)
        # the capillary's two runs, the bar and the capillary leaving the tip; the feeder is larger-class
        self.assertEqual((with_feeder["segments_by_class"]["capillary"]["count"],
                          with_feeder["segments_by_class"]["larger"]["count"]), (4, 1))
        # the same shape with a capillary in the feeder's place
        self.assertEqual(describe.describe(h_shape(90.0, feeder_diameter=1.0), frame=frame(),
                                           d_ref=1.0)["cross_connections"]["count"], 1)

    def test_both_keys_are_none_without_an_axis_frame_or_d_ref(self):
        nodes = ladder()
        for name, options in (("no frame", {"d_ref": D}), ("no d_ref", {"frame": frame()}),
                              ("a plane frame", {"frame": frame(kind="plane"), "d_ref": D}),
                              ("a frame of kind none", {"frame": dict(frame(), kind="none"), "d_ref": D}),
                              ("neither", {})):
            with self.subTest(case=name):
                report = describe.describe(nodes, **options)
                self.assertEqual(tuple(report)[-2:], NEW_KEYS_3_8)
                for key in NEW_KEYS_3_8:
                    self.assertIsNone(report[key], key)
        # an archive without a frame, with and without d_ref
        for options in ({}, {"d_ref": 1.0}):
            report = describe_archive(FIXTURES[0], **options)
            self.assertIsNone(report["frame"])
            for key in NEW_KEYS_3_8:
                self.assertIsNone(report[key], key)
        # an axis frame and d_ref but no capillary-class edge: nothing to count and no planes
        report = describe.describe(ladder(), frame=frame(), d_ref=D / 4.0)
        self.assertEqual(report["cross_connections"]["count"], 0)
        self.assertIsNone(report["cross_connections"]["per_length"])
        self.assertEqual(report["cross_connections"]["junction_types"]["count"], 0)
        self.assertIsNone(report["transverse_spacing"])
        self.assertIsNone(report["cross_section"])


class CrossSectionTests(unittest.TestCase):
    S, N = 2.0, 4                              # a square array of N x N lanes S apart, in a box N S wide

    def square(self):
        return [(self.S / 2.0 + self.S * i, self.S / 2.0 + self.S * j) for i in range(self.N) for j in range(self.N)]

    def describe_lanes(self, positions, box, d_ref=1.0, extra=()):
        nodes = network(lanes(positions, diameter=d_ref) + list(extra))
        return describe.describe(nodes, frame=frame(axis=(0.0, 0.0, 1.0), origin=(box[0] / 2.0, box[1] / 2.0, 5.0)),
                                 d_ref=d_ref, volume=(box[0], box[1], 10.0))

    def test_a_square_array_reads_one_and_a_hexagonal_one_sqrt_of_2_over_root_3(self):
        side = self.N * self.S
        report = self.describe_lanes(self.square(), (side, side))
        self.assertEqual(report["transverse_spacing"]["area_source"], "argument")
        self.assertEqual(report["transverse_spacing"]["mean"], self.S)
        self.assertEqual(report["cross_section"]["nn_ratio"], 1.0)
        self.assertEqual(report["cross_section"]["density_cv"], 0.0)
        # a hexagonal array of rows S apart along x and S sqrt(3) / 2 apart along y, in a box of an even
        # number of whole rows, so that each lane holds the same area
        rows, columns = 6, 6
        pitch = self.S * math.sqrt(3.0) / 2.0
        hexagonal = [((i + 0.25 + 0.5 * (r % 2)) * self.S, (r + 0.5) * pitch) for r in range(rows)
                     for i in range(columns)]
        report = self.describe_lanes(hexagonal, (columns * self.S, rows * pitch))
        self.assertAlmostEqual(report["cross_section"]["nn_ratio"], math.sqrt(2.0 / math.sqrt(3.0)), places=12)

    def test_the_densities_per_plane_are_as_constructed(self):
        side = self.N * self.S
        # four more lanes from z = 0 to 4.5: the planes at z = 1, 2, 3 and 4 cross them, those at 5 to 9 do not
        d_ref = 0.5
        extra = lanes([(2.0, 2.0), (2.0, 6.0), (6.0, 2.0), (6.0, 6.0)], z1=4.5, points=4, diameter=d_ref)
        report = self.describe_lanes(self.square(), (side, side), d_ref=d_ref, extra=extra)
        counts = [self.N ** 2 + 4] * 4 + [self.N ** 2] * 5
        expected = [count / side ** 2 * d_ref ** 2 for count in counts]
        section = report["cross_section"]
        self.assertEqual(section["per_plane_density_d"], expected)
        self.assertEqual(section["density_cv"], float(np.std(expected)) / float(np.mean(expected)))
        # the planes' total is transverse_spacing's density
        self.assertAlmostEqual(report["transverse_spacing"]["crossing_density_d"], float(np.mean(expected)),
                               places=15)
        # transverse_spacing itself is what it was without the new key
        self.assertEqual(report["transverse_spacing"]["count"], sum(counts))

    def test_a_plane_without_area_has_no_density(self):
        # a box placed beside the lanes cuts no plane of them
        nodes = network(lanes(self.square()))
        report = describe.describe(nodes, frame=frame(axis=(0.0, 0.0, 1.0), origin=(4.0, 4.0, 50.0)), d_ref=1.0,
                                   volume=(8.0, 8.0, 10.0))
        self.assertEqual(report["cross_section"]["per_plane_density_d"], [None] * 9)
        self.assertIsNone(report["cross_section"]["density_cv"])
        self.assertIsNone(report["cross_section"]["nn_ratio"])


def numeric_leaves(value, prefix=""):
    """Every number in a nested record, keyed by its path."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_leaves(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for k, item in enumerate(value):
            yield from numeric_leaves(item, f"{prefix}/{k}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


class InvarianceTests(unittest.TestCase):
    def test_both_keys_are_unit_free_but_the_length_per_unit(self):
        for scale in (0.5, 2.5):
            with self.subTest(scale=scale):
                base = describe.describe(ladder(), frame=frame(origin=(20.0, 2.5, 10.0)), d_ref=D,
                                         volume=(40.0, 10.0, 10.0))
                scaled = describe.describe(ladder(scale), frame=frame(origin=(20.0 * scale, 2.5 * scale,
                                                                              10.0 * scale)),
                                           d_ref=D * scale, volume=(40.0 * scale, 10.0 * scale, 10.0 * scale))
                for key in NEW_KEYS_3_8:
                    leaves = dict(numeric_leaves(base[key]))
                    self.assertGreater(len(leaves), 5)
                    for path, value in numeric_leaves(scaled[key]):
                        factor = 1.0 / scale if path == "/per_length" else 1.0
                        self.assertAlmostEqual(value, factor * leaves[path],
                                               delta=1e-12 * max(1.0, abs(factor * leaves[path])), msg=key + path)
                    self.assertEqual(json.loads(json.dumps(scaled[key], allow_nan=False)), scaled[key])


def reference_describe():
    """describe.py as released in 3.7, loaded under a name of its own."""
    spec = importlib.util.spec_from_file_location("describe_3_7", REFERENCE_DESCRIBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExistingKeysTests(unittest.TestCase):
    def test_the_keys_of_3_7_are_byte_identical(self):
        released = reference_describe()
        box_frame = frame(origin=(250.0, 250.0, 125.0))
        options = ({}, {"margin": 1.0}, {"frame": box_frame, "d_ref": 2.0},
                   {"frame": box_frame, "d_ref": 2.0, "volume": (500.0, 500.0, 250.0), "evd_spacing": 25.0})
        cases = [(os.path.basename(path), released.describe_archive, describe_archive, (path,), extra)
                 for path in FIXTURES for extra in options]
        constructed = (("ladder", ladder(), {"frame": frame(origin=(20.0, 2.5, 10.0)), "d_ref": D}),
                       ("H across", h_shape(90.0, feeder_diameter=3.0), {"frame": frame(), "d_ref": 1.0}),
                       ("fork", star([(-1.0, 0.0, 0.0), unit_axis(30.0), unit_axis(-30.0)]),
                        {"frame": frame(), "d_ref": 1.0, "evd_spacing": 2.0}))
        cases += [(name, released.describe, describe.describe, (nodes,), extra) for name, nodes, extra in constructed]
        for name, before, after, args, extra in cases:
            with self.subTest(case=name, options=sorted(extra)):
                old, new = before(*args, **extra), after(*args, **extra)
                self.assertEqual(tuple(new), tuple(old) + NEW_KEYS_3_8)
                self.assertEqual(json.dumps({key: new[key] for key in old}).encode(), json.dumps(old).encode())


if __name__ == "__main__":
    unittest.main()
