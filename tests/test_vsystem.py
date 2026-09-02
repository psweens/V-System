"""
Regression tests for the V-System pipeline.

Run from the repository root with either
    python -m unittest discover -s tests -t .
or
    python -m pytest
"""
import importlib
import math
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from libGenerator import setProperties, calBifurcation, getLength  # noqa: E402
from vSystem import F  # noqa: E402
from analyseGrammar import branching_turtle_to_coords, tokenise  # noqa: E402
from utils import bezier_interpolation  # noqa: E402
from computeVoxel import process_network, rasterise_segments, fit_to_volume  # noqa: E402

PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate(seed, niter, d0=20.0, tVol=(64, 64, 32)):
    seed_all(seed)
    setProperties(PROPERTIES)
    program = F(niter, d0)
    nodes = bezier_interpolation(branching_turtle_to_coords(program, d0))
    return program, nodes, process_network(nodes, tVol=tVol)


class ImportTests(unittest.TestCase):
    def test_every_module_imports_on_the_installed_stack(self):
        for name in ("libGenerator", "vSystem", "analyseGrammar", "utils",
                     "computeVoxel", "preprocessing", "visuals"):
            with self.subTest(module=name):
                importlib.import_module(name)


class GrammarTests(unittest.TestCase):
    def test_generated_grammars_are_balanced_and_fully_tokenised(self):
        for seed, niter in [(0, 6), (1, 9), (2, 12)]:
            with self.subTest(seed=seed, niter=niter):
                seed_all(seed)
                setProperties(PROPERTIES)
                program = F(niter, 20.0)
                self.assertEqual(program.count("["), program.count("]"))
                tokens = list(tokenise(program))
                self.assertTrue(all(math.isfinite(v) for _, params in tokens for v in params))
                self.assertTrue(any(cmd == "f" for cmd, _ in tokens))

    def test_operands_are_parsed_as_numbers_not_commands(self):
        tokens = list(tokenise("f(3.859e-05,0.127)+(-37.46)/(-25.9)f(-0.3,2.1)"))
        self.assertEqual(tokens, [("f", (3.859e-05, 0.127)), ("+", (-37.46,)),
                                  ("/", (-25.9,)), ("f", (-0.3, 2.1))])

        def final_state(program):
            state = None
            for state in branching_turtle_to_coords(program, 5.0):
                pass
            return state

        self.assertAlmostEqual(final_state("+(-37.46)f(10,5)")[3], -37.46)
        state = final_state("/(-25.9)f(10,5)")
        self.assertAlmostEqual(state[3], 0.0)
        self.assertAlmostEqual(state[4], -25.9)

    def test_malformed_programs_raise(self):
        for program in ("+()", "f", "f(nan,1)", "+(1e400)", "f(1,2)x3", "f(abc)", "q(1)"):
            with self.subTest(program=program):
                with self.assertRaises(ValueError):
                    for _ in branching_turtle_to_coords(program, 5.0):
                        pass

    def test_deep_default_configuration_does_not_crash(self):
        seed_all(7)
        setProperties({"k": 3, "epsilon": 4.0, "randmarg": 0.3, "sigma": 5, "stochparams": True})
        program = F(14, 7.0)
        count = sum(1 for _ in branching_turtle_to_coords(program, 7.0))
        self.assertGreater(count, 0)


class ParameterTests(unittest.TestCase):
    def test_segment_lengths_stay_positive_and_proportional(self):
        setProperties({"k": 3, "epsilon": 4.0, "randmarg": 0.3, "sigma": 5, "stochparams": True})
        seed_all(0)
        for d0 in (20.0, 1.0, 0.1, 1e-3):
            lengths = np.array([getLength(d0) for _ in range(2000)])
            self.assertTrue(np.all(lengths > 0))
            self.assertTrue(np.all(lengths >= 0.7 * 4.0 * d0 - 1e-12))
            self.assertTrue(np.all(lengths <= 1.3 * 4.0 * d0 + 1e-12))

    def test_randmarg_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            setProperties({"k": 3, "epsilon": 7.0, "randmarg": 3, "sigma": 5, "stochparams": True})
        setProperties(PROPERTIES)

    def test_bifurcation_rejects_non_positive_diameter(self):
        setProperties(PROPERTIES)
        for d0 in (0.0, -1.0):
            with self.assertRaises(ValueError):
                calBifurcation(d0)

    def test_bifurcation_obeys_murray_law(self):
        setProperties(PROPERTIES)
        seed_all(3)
        for _ in range(100):
            p = calBifurcation(20.0)
            self.assertAlmostEqual(p["d1"] ** 3 + p["d2"] ** 3, 20.0 ** 3, places=8)
            self.assertGreater(p["th1"], 0.0)
            self.assertGreater(p["th2"], 0.0)


class VoxelTests(unittest.TestCase):
    def test_volume_is_binary_uint8_and_zero_initialised(self):
        tVol = (48, 48, 24)
        junk = np.full(tVol, 7, dtype=np.uint8)  # occupy a same-sized block, then free it
        del junk
        _, _, volume = generate(seed=1, niter=7, tVol=tVol)
        self.assertEqual(volume.dtype, np.uint8)
        self.assertEqual(volume.shape, tVol)
        self.assertTrue(set(np.unique(volume).tolist()) <= {0, 1})
        self.assertGreater(int(volume.sum()), 0)

    def test_capsule_voxel_count_matches_the_analytic_volume_at_three_orientations(self):
        length, radius = 40.0, 5.0
        ideal = math.pi * radius ** 2 * length + 4.0 / 3.0 * math.pi * radius ** 3  # cylinder plus end caps
        step2 = length / math.sqrt(2)
        step3 = length / math.sqrt(3)
        cases = {
            "axis": ((20, 40, 40), (60, 40, 40)),
            "xy-diagonal": ((20, 20, 40), (20 + step2, 20 + step2, 40)),
            "xyz-diagonal": ((20, 20, 20), (20 + step3, 20 + step3, 20 + step3)),
        }
        for name, (start, end) in cases.items():
            with self.subTest(orientation=name):
                points = np.array([start, end], dtype=float).T
                volume = rasterise_segments(points, np.array([radius, radius]), (80, 80, 80))
                self.assertAlmostEqual(int(volume.sum()) / ideal, 1.0, delta=0.05)

    def test_isotropic_fit_preserves_angles_and_the_length_to_diameter_ratio(self):
        # a 60 degree bifurcation with very unequal extents per axis
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([100.0, 0.0, 0.0])
        p2 = p1 + 50.0 * np.array([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
        p3 = p1 + 50.0 * np.array([math.cos(math.radians(-60)), math.sin(math.radians(-60)), 0.2])
        nan = np.full(4, np.nan)
        data = np.array([[*p0, 20.0], [*p1, 20.0], [*p2, 12.0], nan, [*p1, 20.0], [*p3, 12.0]]).T
        points, radii = fit_to_volume(data, (100, 60, 40), fit="isotropic", margin=1.0)
        a = points[:, 1] - points[:, 0]
        b = points[:, 2] - points[:, 1]
        angle = math.degrees(math.acos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        self.assertAlmostEqual(angle, 60.0, places=6)
        self.assertAlmostEqual(2 * radii[0] / np.linalg.norm(a), 20.0 / 100.0, places=9)
        # nothing outside the margin, and the largest radius still fits
        finite = ~np.isnan(points[0])
        self.assertTrue(np.all(points[:, finite] - radii[finite] >= 1.0 - 1e-9))
        self.assertTrue(np.all(points[:, finite] + radii[finite] <= np.array([[100], [60], [40]]) - 1.0 + 1e-9))

    def test_rendered_network_never_touches_the_volume_faces(self):
        _, _, volume = generate(seed=5, niter=7, tVol=(64, 48, 32))
        self.assertGreater(int(volume.sum()), 0)
        for face in (volume[0], volume[-1], volume[:, 0], volume[:, -1], volume[..., 0], volume[..., -1]):
            self.assertEqual(int(face.sum()), 0)

    def test_clipped_isotropic_fit_fills_the_kept_axes(self):
        _, nodes, _ = generate(seed=2, niter=6)
        tVol = (100, 80, 20)
        points, radii = fit_to_volume(nodes, tVol, fit="isotropic", margin=1.0, clip_axes=(2,))
        rmax = float(np.nanmax(radii))
        span = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
        available = np.array(tVol) - 2 * (1.0 + rmax)
        self.assertTrue(np.isclose(span[0], available[0]) or np.isclose(span[1], available[1]))
        self.assertTrue(np.all(span[:2] <= available[:2] + 1e-9))
        self.assertGreater(span[2], available[2])  # the depth axis is left to be clipped

    def test_stretch_mode_fills_every_axis(self):
        _, nodes, _ = generate(seed=2, niter=6)
        tVol = (100, 80, 60)
        points, radii = fit_to_volume(nodes, tVol, fit="stretch", margin=1.0)
        rmax = float(np.nanmax(radii))
        span = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
        self.assertTrue(np.allclose(span, np.array(tVol) - 2 * (1.0 + rmax)))


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_grammar_and_volume(self):
        program_a, _, volume_a = generate(seed=11, niter=8)
        program_b, _, volume_b = generate(seed=11, niter=8)
        self.assertEqual(program_a, program_b)
        self.assertTrue(np.array_equal(volume_a, volume_b))


if __name__ == "__main__":
    unittest.main()
