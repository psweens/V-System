"""
Tests for the rigid placement of a network by its frame record.

The rotations are checked for what a rotation must be, proper and
orthonormal to rounding, and for what each one promises to map where; the
transforms for carrying nodes and record together, which the length-weighted
order parameter about the frame's axis, computed here from the edges of a
synthetic polyline, must not notice.

Run from the repository root with
    python -m unittest tests.test_frames -v
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frames import (PARALLEL_TOL, ROTATION_TOL, align, random_rotation, rotation_about,  # noqa: E402
                    rotation_between, rotation_of_frames, transform_frame, transform_nodes)

IDENTITY = np.eye(3)
LAB_AXES = [np.array(row) for row in np.eye(3)]
OBLIQUE = [np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0), np.array([-1.0, 1.0, 0.5]) / 1.5,
           np.array([0.3, -0.4, 0.0]) / 0.5]

# The example records of the specification: an "aligned" axis frame, polar
# along x with a plane rule beside its axis rule, and a plane frame.
AXIS_FRAME = {"frame_version": 1, "kind": "axis", "axis": [1.0, 0.0, 0.0], "sense": "polar",
              "tree_senses": [1, -1], "normal": None, "origin": [37.5, 37.5, 37.5],
              "grow_direction": [0.0, 1.0, 0.0], "grow_perpendicular": [0.0, 0.0, 1.0],
              "rules": [{"below": 2.0, "field": "axis", "axis": [1.0, 0.0, 0.0], "sense": "polar",
                         "polarity": "partner", "length": 4.9, "onset": 2.0},
                        {"below": None, "field": "plane", "normal": [1.0, 0.0, 0.0], "length": 5.0,
                         "onset": 2.0, "bank": False}]}
PLANE_FRAME = {"frame_version": 1, "kind": "plane", "axis": None, "sense": None, "tree_senses": None,
               "normal": [0.0, 0.0, 1.0], "origin": [0.0, 0.0, 0.0], "grow_direction": [0.0, 1.0, 0.0],
               "grow_perpendicular": [0.0, 0.0, 1.0],
               "rules": [{"below": None, "field": "plane", "normal": [0.0, 0.0, 1.0], "length": 2.0,
                          "onset": 2.0, "bank": True}]}
NONE_FRAME = {"frame_version": 1, "kind": "none", "axis": None, "sense": None, "tree_senses": None,
              "normal": None, "origin": [0.0, 0.0, 0.0], "grow_direction": [0.0, 1.0, 0.0],
              "grow_perpendicular": [0.0, 0.0, 1.0], "rules": []}


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def random_unit(rng):
    return unit(rng.normal(size=3))


def perpendicular_to(u, rng):
    """A unit vector perpendicular to `u`, drawn at random."""
    p = rng.normal(size=3)
    return unit(p - (p @ u) * u)


def rotation_defect(q):
    """How far `q` is from a proper rotation: the larger of |det - 1| and max |Q^T Q - I|."""
    return max(abs(float(np.linalg.det(q)) - 1.0), float(np.max(np.abs(q.T @ q - IDENTITY))))


def rotation_angle(q):
    """The angle `q` turns by, acos((tr Q - 1) / 2), clipped against rounding."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(q)) - 1.0) / 2.0)))


def axis_frame(axis, sense, rng=None):
    """An axis frame like the specification's example, with its heading perpendicular to the axis."""
    frame = copy.deepcopy(AXIS_FRAME)
    axis = unit(axis)
    frame["axis"] = [float(v) for v in axis]
    frame["sense"] = sense
    frame["rules"][0]["axis"] = list(frame["axis"])
    frame["rules"][0]["sense"] = sense
    if sense == "nematic":
        del frame["rules"][0]["polarity"]
        frame["tree_senses"] = None
    if rng is not None:
        heading = perpendicular_to(axis, rng)
        frame["grow_direction"] = [float(v) for v in heading]
        frame["grow_perpendicular"] = [float(v) for v in unit(np.cross(axis, heading))]
    return frame


def plane_frame(normal):
    frame = copy.deepcopy(PLANE_FRAME)
    frame["normal"] = [float(v) for v in unit(normal)]
    frame["rules"][0]["normal"] = list(frame["normal"])
    return frame


def near_antiparallel_pairs(rng):
    """
    Pairs (u, w) with 1 + u . w stepping from 1e-3 down to 1e-15, each from
    the lab axes and from oblique directions.

    w = -cos(theta) u + sin(theta) p with cos(theta) = 1 - eps and p
    perpendicular to u; the smallest eps leaves |u x w| about 4.5e-8, still
    above PARALLEL_TOL, so every pair goes through the Rodrigues form.
    """
    pairs = []
    for eps in (1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-15):
        theta = math.acos(1.0 - eps)
        for u in LAB_AXES + [-a for a in LAB_AXES] + OBLIQUE + [random_unit(rng) for _ in range(5)]:
            p = perpendicular_to(u, rng)
            pairs.append((eps, u, -(math.cos(theta) * u) + math.sin(theta) * p))
    return pairs


def polyline_archive(rng, axis, n_polylines=4, n_points=150, bias=1.5, diameter=3.0):
    """
    A (4, N) archive of polylines whose steps lean along `axis`, with the
    generator's NaN separators, so that the order parameter about the axis is
    well away from zero and from one.
    """
    axis = unit(axis)
    columns = []
    for k in range(n_polylines):
        if k:
            columns.append([np.nan] * 4)
        point = rng.uniform(0.0, 50.0, size=3)
        for _ in range(n_points):
            step = unit(bias * axis * rng.choice([-1.0, 1.0]) + rng.normal(size=3)) * rng.uniform(0.5, 2.0)
            point = point + step
            columns.append([point[0], point[1], point[2], diameter * rng.uniform(0.5, 1.5)])
    return np.array(columns, dtype=float).T


def order_parameter(nodes, axis):
    """
    Length-weighted S = <(3 (t . a)^2 - 1) / 2> over the edges of the
    archive, an edge joining two consecutive finite columns.
    """
    axis = unit(axis)
    finite = np.isfinite(nodes[:3]).all(axis=0)
    linked = finite[:-1] & finite[1:]
    vector = (nodes[:3, 1:] - nodes[:3, :-1])[:, linked].T
    length = np.linalg.norm(vector, axis=1)
    tangent = vector / length[:, None]
    cosine = tangent @ axis
    return float(np.sum(length * (3.0 * cosine ** 2 - 1.0) / 2.0) / np.sum(length))


class RotationBetweenTests(unittest.TestCase):
    def assert_maps(self, u, w, q, nematic=False):
        """`q` is a proper rotation to 1e-12 taking u to w, to 1e-12 or to 1e-7 for a near-parallel pair."""
        self.assertLessEqual(rotation_defect(q), 1e-12)
        s = float(np.linalg.norm(np.cross(unit(u), unit(w))))
        tolerance = 1e-12 if s >= 1e-3 else 1e-7
        target = unit(w)
        if nematic and unit(u) @ target < 0.0:
            target = -target
        self.assertLessEqual(float(np.max(np.abs(q @ unit(u) - target))), tolerance)

    def test_random_pairs(self):
        rng = np.random.default_rng(11)
        for _ in range(300):
            u, w = rng.normal(size=3) * 3.0, rng.normal(size=3) * 0.2
            self.assert_maps(u, w, rotation_between(u, w))

    def test_near_antiparallel_pairs_stay_orthonormal(self):
        rng = np.random.default_rng(12)
        for eps, u, w in near_antiparallel_pairs(rng):
            with self.subTest(eps=eps, u=u.round(3).tolist()):
                q = rotation_between(u, w)
                self.assertLessEqual(rotation_defect(q), 1e-12)
                self.assert_maps(u, w, q)
                # the half turn is nearly whole: the rotation angle is within
                # a few of the pair's own angle of a straight angle
                self.assertGreaterEqual(rotation_angle(q), math.pi - 3.0 * math.acos(1.0 - eps) - 1e-9)

    def test_exactly_parallel_and_antiparallel_pairs(self):
        for u in LAB_AXES + [-a for a in LAB_AXES] + OBLIQUE:
            with self.subTest(u=u.round(3).tolist()):
                np.testing.assert_allclose(rotation_between(u, u), IDENTITY, atol=1e-15)
                np.testing.assert_allclose(rotation_between(u, 2.5 * u), IDENTITY, atol=1e-15)
                q = rotation_between(u, -u)
                self.assertLessEqual(rotation_defect(q), 1e-12)
                np.testing.assert_allclose(q @ u, -u, atol=1e-12)
                self.assertAlmostEqual(rotation_angle(q), math.pi, places=7)
                # a half turn is its own inverse
                np.testing.assert_allclose(q @ q, IDENTITY, atol=1e-12)

    def test_nematic_turns_by_the_nearer_end(self):
        rng = np.random.default_rng(13)
        for _ in range(200):
            u, w = random_unit(rng), random_unit(rng)
            q = rotation_between(u, w, nematic=True)
            self.assert_maps(u, w, q, nematic=True)
            self.assertLessEqual(rotation_angle(q), math.pi / 2.0 + 1e-12)
            # as arrows, the same pair is mapped the other way round when it faces back
            if u @ w < 0.0:
                np.testing.assert_allclose(rotation_between(u, w) @ u, w, atol=1e-12)
        # as lines, an exactly antiparallel pair is one line already, so nothing turns
        np.testing.assert_allclose(rotation_between(LAB_AXES[0], -LAB_AXES[0], nematic=True), IDENTITY, atol=1e-15)

    def test_inputs_are_normalised_first(self):
        q = rotation_between([0.0, 3.0, 0.0], [0.0, 0.0, -0.1])
        np.testing.assert_allclose(q @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-15)

    def test_refuses_zero_or_non_finite_directions(self):
        for bad in ([0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [np.inf, 0.0, 0.0], [1.0, 0.0], None, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    rotation_between(bad, [1.0, 0.0, 0.0])
                with self.assertRaises(ValueError):
                    rotation_between([1.0, 0.0, 0.0], bad)


class RotationAboutTests(unittest.TestCase):
    def test_quarter_turns_about_the_lab_axes(self):
        x, y, z = LAB_AXES
        np.testing.assert_allclose(rotation_about(z, math.pi / 2.0) @ x, y, atol=1e-15)
        np.testing.assert_allclose(rotation_about(x, math.pi / 2.0) @ y, z, atol=1e-15)
        np.testing.assert_allclose(rotation_about(y, math.pi / 2.0) @ z, x, atol=1e-15)
        np.testing.assert_allclose(rotation_about(z, math.pi) @ x, -x, atol=1e-15)
        np.testing.assert_allclose(rotation_about(z, 0.0), IDENTITY, atol=1e-15)
        # the axis is normalised, so its length does not change the turn
        np.testing.assert_allclose(rotation_about(5.0 * z, math.pi / 2.0) @ x, y, atol=1e-15)

    def test_matches_rodrigues(self):
        rng = np.random.default_rng(21)
        for _ in range(100):
            axis, angle = rng.normal(size=3), rng.uniform(-2.0 * math.pi, 2.0 * math.pi)
            u = unit(axis)
            skew = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
            expected = math.cos(angle) * IDENTITY + math.sin(angle) * skew + (1.0 - math.cos(angle)) * np.outer(u, u)
            q = rotation_about(axis, angle)
            np.testing.assert_allclose(q, expected, atol=1e-15)
            self.assertLessEqual(rotation_defect(q), 1e-12)
            # the axis is fixed and the turn is the angle, folded to 0..pi
            np.testing.assert_allclose(q @ u, u, atol=1e-15)
            self.assertAlmostEqual(rotation_angle(q), abs(math.remainder(angle, 2.0 * math.pi)), places=9)

    def test_the_length_of_the_axis_does_not_matter_from_tiny_to_huge(self):
        # a direction is scaled by its largest component before its norm is
        # taken, which would otherwise overflow above about 1e154 or lose
        # precision in subnormal squares below about 1e-154
        reference = rotation_about([1.0, 2.0, 2.0], 1.0)
        for scale in (1e-300, 1e-160, 1e155, 1e300):
            with self.subTest(scale=scale):
                q = rotation_about([scale, 2.0 * scale, 2.0 * scale], 1.0)
                self.assertLessEqual(rotation_defect(q), 1e-12)
                np.testing.assert_allclose(q, reference, atol=1e-15)
                np.testing.assert_allclose(rotation_between([scale, 0.0, 0.0], [0.0, scale, 0.0]) @ LAB_AXES[0],
                                           LAB_AXES[1], atol=1e-15)
                q = align(AXIS_FRAME, axis=[0.0, 0.0, scale], spin=0.5)
                self.assertLessEqual(rotation_defect(q), 1e-12)
                np.testing.assert_allclose(q @ np.asarray(AXIS_FRAME["axis"]), LAB_AXES[2], atol=1e-15)

    def test_refuses_a_zero_or_non_finite_axis_or_angle(self):
        for bad in ([0.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]):
            with self.assertRaises(ValueError):
                rotation_about(bad, 1.0)
        for bad in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                rotation_about([0.0, 0.0, 1.0], bad)


class RotationOfFramesTests(unittest.TestCase):
    def test_maps_both_directions(self):
        rng = np.random.default_rng(31)
        for _ in range(100):
            u1, u2, w1, w2 = (rng.normal(size=3) for _ in range(4))
            q = rotation_of_frames(u1, u2, w1, w2)
            self.assertLessEqual(rotation_defect(q), 1e-12)
            np.testing.assert_allclose(q @ unit(u1), unit(w1), atol=1e-12)
            # the second direction is mapped after Gram-Schmidt on the first
            u2p = unit(u2 - (u2 @ unit(u1)) * unit(u1))
            w2p = unit(w2 - (w2 @ unit(w1)) * unit(w1))
            np.testing.assert_allclose(q @ u2p, w2p, atol=1e-12)
            np.testing.assert_allclose(q @ np.cross(unit(u1), u2p), np.cross(unit(w1), w2p), atol=1e-12)

    def test_perpendicular_pairs_are_mapped_as_given(self):
        rng = np.random.default_rng(32)
        for _ in range(50):
            u1, w1 = random_unit(rng), random_unit(rng)
            u2, w2 = perpendicular_to(u1, rng), perpendicular_to(w1, rng)
            q = rotation_of_frames(u1, u2, w1, w2)
            np.testing.assert_allclose(q @ u2, w2, atol=1e-12)
        # the frame of the generator's default heading and perpendicular to itself is the identity
        np.testing.assert_allclose(rotation_of_frames([0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]), IDENTITY, atol=1e-15)

    def test_nearly_parallel_pairs_still_give_a_rotation_the_transforms_accept(self):
        # the second direction a hair from the first, but beyond PARALLEL_TOL:
        # one Gram-Schmidt pass would leave rounding of order 1e-16 / separation
        # along the first, and the basis would stop being orthonormal
        rng = np.random.default_rng(33)
        for separation in (1e-4, 1e-7, 2e-8):
            for _ in range(200):
                u1, w1, w2 = random_unit(rng), random_unit(rng), random_unit(rng)
                u2 = u1 + separation * perpendicular_to(u1, rng)
                q = rotation_of_frames(u1, u2, w1, w2)
                self.assertLessEqual(rotation_defect(q), 1e-12)
                np.testing.assert_allclose(q @ u1, w1, atol=1e-12)
                transform_nodes(np.zeros((4, 1)), q)

    def test_refuses_degenerate_pairs(self):
        with self.assertRaises(ValueError):
            rotation_of_frames([1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 1])
        with self.assertRaises(ValueError):
            rotation_of_frames([1, 0, 0], [0, 1, 0], [0, 1, 0], [0, -1, 0])
        with self.assertRaises(ValueError):
            rotation_of_frames([0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1])


class RandomRotationTests(unittest.TestCase):
    def test_reproduces_the_quaternion_formula_for_a_fixed_seed(self):
        q = np.random.default_rng(1234).normal(size=4)
        q = q / np.linalg.norm(q)
        a, b, c, d = q
        expected = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                             [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                             [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
        np.testing.assert_allclose(random_rotation(np.random.default_rng(1234)), expected, atol=1e-15)
        # exactly four normal draws are consumed, so the stream after a draw is predictable
        rng = np.random.default_rng(1234)
        random_rotation(rng)
        self.assertEqual(rng.normal(), np.random.default_rng(1234).normal(size=5)[4])

    def test_is_proper_and_orthonormal(self):
        rng = np.random.default_rng(41)
        for _ in range(500):
            self.assertLessEqual(rotation_defect(random_rotation(rng)), 1e-12)

    def test_has_no_preferred_direction(self):
        # a fixed vector sent through many draws has a mean near zero and a
        # mean squared component of a third on every axis, as a uniform
        # distribution over the sphere gives; 4000 draws put the standard
        # error of the mean at about 0.01 per component
        rng = np.random.default_rng(42)
        images = np.array([random_rotation(rng) @ LAB_AXES[2] for _ in range(4000)])
        self.assertLess(float(np.linalg.norm(images.mean(axis=0))), 0.06)
        np.testing.assert_allclose((images ** 2).mean(axis=0), [1 / 3] * 3, atol=0.04)


class AlignTests(unittest.TestCase):
    def test_nematic_turns_at_most_a_right_angle(self):
        rng = np.random.default_rng(51)
        for _ in range(300):
            frame = axis_frame(random_unit(rng), "nematic", rng)
            target = random_unit(rng)
            q = align(frame, axis=target)
            self.assertLessEqual(rotation_defect(q), 1e-12)
            self.assertLessEqual(rotation_angle(q), math.pi / 2.0 + 1e-12)
            image = q @ np.array(frame["axis"])
            self.assertLessEqual(float(np.max(np.abs(np.abs(image @ target) - 1.0))), 1e-12)

    def test_polar_maps_the_axis_exactly(self):
        rng = np.random.default_rng(52)
        for k in range(300):
            own = random_unit(rng)
            frame = axis_frame(own, "polar", rng)
            target = random_unit(rng)
            if k % 2:
                # half the cases face back: a_frame . A < 0 still goes to A itself
                target = -target if own @ target > 0.0 else target
            spin = 0.0 if k % 3 == 0 else rng.uniform(0.0, 2.0 * math.pi)
            q = align(frame, axis=target, spin=spin)
            self.assertLessEqual(rotation_defect(q), 1e-12)
            np.testing.assert_allclose(q @ own, target, atol=1e-12)
        # a polar frame facing exactly back is turned by a half turn, not left alone
        frame = axis_frame(LAB_AXES[0], "polar")
        np.testing.assert_allclose(align(frame, axis=-LAB_AXES[0]) @ LAB_AXES[0], -LAB_AXES[0], atol=1e-15)
        np.testing.assert_allclose(align(axis_frame(LAB_AXES[0], "nematic"), axis=-LAB_AXES[0]), IDENTITY, atol=1e-15)

    def test_spin_is_about_the_target(self):
        rng = np.random.default_rng(53)
        for sense in ("polar", "nematic"):
            frame = axis_frame(random_unit(rng), sense, rng)
            target = random_unit(rng)
            for spin in (0.0, 0.7, -2.0, math.pi):
                expected = rotation_about(target, spin) @ align(frame, axis=target)
                np.testing.assert_allclose(align(frame, axis=target, spin=spin), expected, atol=1e-15)
        # two networks aligned to one axis with different spins differ by a turn about that axis
        q0 = align(frame, axis=target, spin=0.3)
        q1 = align(frame, axis=target, spin=1.3)
        np.testing.assert_allclose(q1 @ q0.T, rotation_about(target, 1.0), atol=1e-12)
        # the given axis need not be unit
        np.testing.assert_allclose(align(frame, axis=3.0 * target), align(frame, axis=target), atol=1e-15)

    def test_plane_frames_align_their_normal_as_a_line(self):
        rng = np.random.default_rng(54)
        for _ in range(200):
            frame = plane_frame(random_unit(rng))
            target = random_unit(rng)
            spin = rng.uniform(0.0, 2.0 * math.pi)
            q = align(frame, normal=target, spin=spin)
            self.assertLessEqual(rotation_defect(q), 1e-12)
            self.assertLessEqual(rotation_angle(align(frame, normal=target)), math.pi / 2.0 + 1e-12)
            image = q @ np.array(frame["normal"])
            self.assertAlmostEqual(abs(float(image @ target)), 1.0, delta=1e-12)

    def test_refusals(self):
        axis, normal = LAB_AXES[0], LAB_AXES[2]
        with self.assertRaises(ValueError):
            align(None, axis=axis)                                     # no record, as before 3.5
        with self.assertRaises(ValueError):
            align(NONE_FRAME, axis=axis)                               # guidance off
        with self.assertRaises(ValueError):
            align(NONE_FRAME, normal=normal)
        with self.assertRaises(ValueError):
            align(PLANE_FRAME, axis=axis)                              # an axis of a plane frame
        with self.assertRaises(ValueError):
            align(AXIS_FRAME, normal=normal)                           # a normal of an axis frame
        with self.assertRaises(ValueError):
            align(AXIS_FRAME)                                          # neither
        with self.assertRaises(ValueError):
            align(AXIS_FRAME, axis=axis, normal=normal)                # both
        with self.assertRaises(ValueError):
            align(AXIS_FRAME, axis=[0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            align(AXIS_FRAME, axis=axis, spin=math.nan)
        with self.assertRaises(ValueError):
            align(dict(AXIS_FRAME, kind="helix"), axis=axis)
        with self.assertRaises(ValueError):
            align(dict(AXIS_FRAME, sense="twisted"), axis=axis)
        with self.assertRaises(ValueError):
            align("axis", axis=axis)


class TransformTests(unittest.TestCase):
    BAD_ROTATIONS = (np.diag([1.0, 1.0, -1.0]),                        # a reflection
                     np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),   # a shear
                     2.0 * IDENTITY, np.eye(2), np.ones(3), np.zeros((3, 3, 1)),
                     IDENTITY + 1e-8, np.full((3, 3), np.nan))

    def setUp(self):
        self.rng = np.random.default_rng(61)
        self.nodes = polyline_archive(self.rng, LAB_AXES[1], n_polylines=3, n_points=20)

    def test_nodes_follow_the_formula(self):
        q, t, scale = random_rotation(self.rng), [5.0, -2.0, 1.5], 0.25
        out = transform_nodes(self.nodes, q, t, scale)
        self.assertEqual(out.shape, self.nodes.shape)
        self.assertIsNot(out, self.nodes)
        finite = np.isfinite(self.nodes[0])
        np.testing.assert_allclose(out[:3, finite], scale * (q @ self.nodes[:3, finite]) + np.array(t)[:, None],
                                   atol=1e-12)
        np.testing.assert_allclose(out[3, finite], scale * self.nodes[3, finite])
        np.testing.assert_array_equal(np.isnan(out), np.isnan(self.nodes))
        # the defaults move nothing
        np.testing.assert_allclose(transform_nodes(self.nodes, IDENTITY), self.nodes)
        # the input is untouched and an integer or list input is accepted
        self.assertEqual(transform_nodes([[1, 2], [3, 4], [5, 6], [7, 8]], IDENTITY, scale=2).tolist(),
                         [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0], [14.0, 16.0]])
        self.assertEqual(transform_nodes(np.empty((4, 0)), IDENTITY).shape, (4, 0))

    def test_scaling(self):
        # from micrometres to millimetres: nodes, diameters and origin scale,
        # the separators stay and the vectors stay unit and unchanged
        nodes = np.array([[10.0, 20.0, np.nan, 30.0], [1.0, 2.0, np.nan, 3.0],
                          [0.5, 0.5, np.nan, 0.5], [4.0, 3.0, np.nan, 2.0]])
        out = transform_nodes(nodes, IDENTITY, scale=1e-3)
        np.testing.assert_allclose(out[:, [0, 1, 3]], nodes[:, [0, 1, 3]] * 1e-3)
        self.assertTrue(np.isnan(out[:, 2]).all())
        frame = transform_frame(AXIS_FRAME, IDENTITY, scale=1e-3)
        self.assertEqual(frame["origin"], [0.0375, 0.0375, 0.0375])
        for key in ("axis", "grow_direction", "grow_perpendicular"):
            self.assertEqual(frame[key], AXIS_FRAME[key])
            self.assertAlmostEqual(float(np.linalg.norm(frame[key])), 1.0, delta=1e-12)
        self.assertEqual(frame["rules"], AXIS_FRAME["rules"])
        # an oblique frame too: a pure rescale leaves every vector bitwise as
        # it was, which re-normalising would not, so the record differs from
        # the original in its origin alone
        oblique = axis_frame(OBLIQUE[0], "nematic", np.random.default_rng(62))
        rescaled = transform_frame(oblique, IDENTITY, scale=1e-3)
        self.assertEqual(dict(rescaled, origin=oblique["origin"]), oblique)
        # the translation is in the unit of the result, applied after the scale
        np.testing.assert_allclose(transform_nodes(nodes, IDENTITY, t=[1.0, 2.0, 3.0], scale=2.0)[:3, 0],
                                   [21.0, 4.0, 4.0])
        self.assertEqual(transform_frame(PLANE_FRAME, IDENTITY, t=[1.0, 2.0, 3.0], scale=2.0)["origin"],
                         [1.0, 2.0, 3.0])

    def test_frame_follows_the_nodes(self):
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            q, t, scale = random_rotation(rng), rng.normal(size=3) * 10.0, rng.uniform(0.1, 10.0)
            for original in (AXIS_FRAME, axis_frame(random_unit(rng), "nematic", rng), PLANE_FRAME, NONE_FRAME):
                before = copy.deepcopy(original)
                out = transform_frame(original, q, t, scale)
                self.assertEqual(original, before)                     # the input is untouched
                self.assertIsNot(out, original)
                self.assertEqual(set(out), set(original))
                np.testing.assert_allclose(out["origin"], scale * (q @ np.array(original["origin"])) + t, atol=1e-9)
                for key in ("axis", "normal", "grow_direction", "grow_perpendicular"):
                    if original[key] is None:
                        self.assertIsNone(out[key])
                        continue
                    np.testing.assert_allclose(out[key], q @ np.array(original[key]), atol=1e-12)
                    self.assertAlmostEqual(float(np.linalg.norm(out[key])), 1.0, delta=1e-12)
                    self.assertIsInstance(out[key], list)
                    self.assertTrue(all(type(v) is float for v in out[key]))
                for key in ("frame_version", "kind", "sense", "tree_senses"):
                    self.assertEqual(out[key], original[key])
                self.assertEqual(len(out["rules"]), len(original["rules"]))
                for rule, rule_before in zip(out["rules"], original["rules"]):
                    self.assertEqual(set(rule), set(rule_before))
                    for key, value in rule_before.items():
                        if key in ("axis", "normal") and value is not None:
                            np.testing.assert_allclose(rule[key], q @ np.array(value), atol=1e-12)
                            self.assertAlmostEqual(float(np.linalg.norm(rule[key])), 1.0, delta=1e-12)
                        else:
                            self.assertEqual(rule[key], value)
                # parse-equality: the record survives a JSON round trip unchanged
                self.assertEqual(json.loads(json.dumps(out, allow_nan=False)), out)
                # and a rigid move the other way restores it, origin included
                back = transform_frame(out, q.T, -(q.T @ t) / scale, 1.0 / scale)
                np.testing.assert_allclose(back["origin"], original["origin"], atol=1e-9)
                for key in ("axis", "normal", "grow_direction", "grow_perpendicular"):
                    if original[key] is not None:
                        np.testing.assert_allclose(back[key], original[key], atol=1e-12)

    def test_grow_frame_moves_with_the_box(self):
        # the box sides (grow_direction x grow_perpendicular, grow_direction,
        # grow_perpendicular) rotate with Q and stay a right-handed frame
        q = random_rotation(np.random.default_rng(71))
        out = transform_frame(AXIS_FRAME, q)
        direction, perpendicular = np.array(out["grow_direction"]), np.array(out["grow_perpendicular"])
        np.testing.assert_allclose(direction, q @ [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.cross(direction, perpendicular), q @ [1.0, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(float(direction @ perpendicular), 0.0, delta=1e-12)

    def test_rotations_within_tolerance_are_accepted(self):
        # a Q orthonormal to 1e-10 is accepted, and the vectors are Q v as
        # written, so they are unit to that and no better
        q = random_rotation(np.random.default_rng(72)) * (1.0 + 1e-10)
        out = transform_frame(AXIS_FRAME, q)
        for key in ("axis", "grow_direction", "grow_perpendicular"):
            np.testing.assert_allclose(out[key], q @ np.array(AXIS_FRAME[key]), atol=1e-15)
            self.assertAlmostEqual(float(np.linalg.norm(out[key])), 1.0, delta=ROTATION_TOL)
        self.assertEqual(transform_nodes(self.nodes, q).shape, self.nodes.shape)

    def test_refuses_what_is_not_a_rotation(self):
        for bad in self.BAD_ROTATIONS:
            with self.subTest(bad=np.asarray(bad).shape):
                with self.assertRaises(ValueError):
                    transform_nodes(self.nodes, bad)
                with self.assertRaises(ValueError):
                    transform_frame(AXIS_FRAME, bad)
        with self.assertRaises(ValueError):
            transform_nodes(self.nodes, "Q")
        # the accepted deviation is ROTATION_TOL, so a determinant of 1 - 2e-9 is out
        with self.assertRaises(ValueError):
            transform_nodes(self.nodes, IDENTITY * (1.0 - 2.0 * ROTATION_TOL))
        # the parallel threshold is what rotation_between documents
        self.assertEqual(PARALLEL_TOL, 1e-8)

    def test_refuses_bad_placements_and_records(self):
        for scale in (0.0, -1.0, math.inf, math.nan, "2", None, [1.0, 2.0]):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    transform_nodes(self.nodes, IDENTITY, scale=scale)
                with self.assertRaises(ValueError):
                    transform_frame(AXIS_FRAME, IDENTITY, scale=scale)
        for t in ([1.0, 2.0], [1.0, np.nan, 0.0], None):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    transform_nodes(self.nodes, IDENTITY, t=t)
                with self.assertRaises(ValueError):
                    transform_frame(AXIS_FRAME, IDENTITY, t=t)
        with self.assertRaises(ValueError):
            transform_nodes(np.zeros((3, 4)), IDENTITY)
        with self.assertRaises(ValueError):
            transform_nodes(np.zeros(4), IDENTITY)
        with self.assertRaises(ValueError):
            transform_nodes([[1.0, "x"], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], IDENTITY)
        with self.assertRaises(ValueError):
            transform_nodes([[1.0, object()], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], IDENTITY)
        with self.assertRaises(ValueError):
            transform_frame(None, IDENTITY)
        with self.assertRaises(ValueError):
            transform_frame(dict(AXIS_FRAME, kind="helix"), IDENTITY)
        with self.assertRaises(ValueError):
            transform_frame(dict(AXIS_FRAME, axis=[1.0, 0.0]), IDENTITY)
        with self.assertRaises(ValueError):
            transform_frame(dict(AXIS_FRAME, origin=[np.nan, 0.0, 0.0]), IDENTITY)
        with self.assertRaises(ValueError):
            transform_frame(dict(AXIS_FRAME, rules={"axis": [1.0, 0.0, 0.0]}), IDENTITY)


class OrderParameterTests(unittest.TestCase):
    def test_order_parameter_about_the_carried_axis_is_invariant(self):
        rng = np.random.default_rng(81)
        own = random_unit(rng)
        frame = axis_frame(own, "polar", rng)
        nodes = polyline_archive(rng, own)
        before = order_parameter(nodes, frame["axis"])
        # a biased walk is ordered, but not perfectly
        self.assertGreater(before, 0.2)
        self.assertLess(before, 0.95)
        q, t, scale = random_rotation(rng), rng.normal(size=3) * 100.0, 0.37
        moved_nodes = transform_nodes(nodes, q, t, scale)
        moved_frame = transform_frame(frame, q, t, scale)
        after = order_parameter(moved_nodes, moved_frame["axis"])
        self.assertAlmostEqual(after, before, delta=1e-9)
        # about the old axis the moved vessels are measured askew: Q turns the
        # axis through a wide angle here, and S about a direction at angle
        # theta to a perfectly ordered bundle falls to (3 cos^2 theta - 1) / 2
        turned = math.acos(abs(float(np.array(moved_frame["axis"]) @ own)))
        self.assertGreater(turned, 0.3)
        self.assertGreater(abs(order_parameter(moved_nodes, own) - before), 0.05)
        # alignment to a shared axis is such a move, and lands the order on that axis
        shared = LAB_AXES[2]
        q = align(frame, axis=shared, spin=rng.uniform(0.0, 2.0 * math.pi))
        aligned = transform_frame(frame, q)
        np.testing.assert_allclose(aligned["axis"], shared, atol=1e-12)
        self.assertAlmostEqual(order_parameter(transform_nodes(nodes, q), shared), before, delta=1e-9)


class DescriptorInvarianceTests(unittest.TestCase):
    """A moved archive, described against its moved frame, measures what it measured before."""

    def test_a_random_transform_leaves_the_frame_relative_descriptors_unchanged(self):
        import random
        import main
        from describe import describe
        argv = ["--family", "aligned", "--d-min", "1", "--iterations", "64", "64", "--d0", "4", "0", "--d0-min", "4",
                "--volume", "3", "3", "3", "--fit", "voxel_size", "--voxel-size", "20"]
        args = main.build_parser("aligned").parse_args(argv)
        random.seed(2)
        np.random.seed(2)
        properties, d0, niter = main.sample_parameters(args)
        grown = main.grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit,
                                  clip_axes=tuple(args.clip_axes), voxel_size=args.voxel_size,
                                  subdivisions=args.subdivisions, d_min=args.d_min,
                                  grow_in_volume=args.grow_in_volume, seed=2, **main.shaping_options(args))
        rng = np.random.default_rng(7)
        q = random_rotation(rng)
        t = rng.normal(size=3) * 100.0
        moved = transform_nodes(grown["nodes"], q, t, scale=2.5)
        frame = transform_frame(grown["frame"], q, t, scale=2.5)
        before = describe(grown["nodes"], grown["edges"], frame=grown["frame"], d_ref=1.0, tree=grown["tree"])
        after = describe(moved, grown["edges"], frame=frame, d_ref=2.5, tree=grown["tree"])
        for cls in ("all", "capillary", "larger"):
            for key in ("S", "mean_abs_cos", "crossing_ratio", "mean_angle_deg", "within_20_deg", "within_45_deg",
                        "watson_K", "fisher_axial_K", "length_d"):
                self.assertAlmostEqual(after["frame_orientation"][cls][key], before["frame_orientation"][cls][key],
                                       delta=1e-9 * max(1.0, abs(before["frame_orientation"][cls][key])), msg=(cls, key))
            self.assertAlmostEqual(after["polar_order"][cls], before["polar_order"][cls], delta=1e-9)
        for key in ("median_d", "mean_d", "p10_d", "p90_d", "crossing_density_d"):
            self.assertAlmostEqual(after["transverse_spacing"][key], before["transverse_spacing"][key],
                                   delta=1e-9 * max(1.0, abs(before["transverse_spacing"][key])), msg=key)
        self.assertEqual(after["transverse_spacing"]["count"], before["transverse_spacing"]["count"])
        self.assertEqual(after["frame"]["tree_senses"], before["frame"]["tree_senses"])
        # whereas the order about the old axis, which the move turned away from, is not what it was
        stale = describe(moved, grown["edges"], frame=grown["frame"], d_ref=2.5, tree=grown["tree"])
        self.assertNotAlmostEqual(stale["frame_orientation"]["capillary"]["S"],
                                  before["frame_orientation"]["capillary"]["S"], delta=0.01)


if __name__ == "__main__":
    unittest.main()
