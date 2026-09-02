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
from computeVoxel import process_network, transversal  # noqa: E402

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

    LENGTH, DIAMETER = 40, 10
    IDEAL = math.pi * (DIAMETER / 2) ** 2 * LENGTH

    def rasterised_fraction(self, start, end):
        volume = np.zeros((80, 80, 80), dtype=np.uint8)
        transversal(*start, *end, self.DIAMETER, volume, (80, 80, 80))
        return int(volume.sum()) / self.IDEAL

    def test_axis_aligned_cylinder_voxel_count_matches_pi_r_squared_l(self):
        self.assertAlmostEqual(self.rasterised_fraction((20, 40, 40), (60, 40, 40)), 1.0, delta=0.05)

    @unittest.expectedFailure
    def test_oblique_cylinder_voxel_count_matches_pi_r_squared_l(self):
        # Known defect: cross-section discs are drawn perpendicular to the stepped
        # axis rather than to the segment, so oblique vessels come out thinner.
        # This test starts passing once the rasteriser draws true capsules.
        step = int(self.LENGTH / math.sqrt(2))
        self.assertAlmostEqual(self.rasterised_fraction((20, 20, 40), (20 + step, 20 + step, 40)), 1.0, delta=0.05)
        step = int(self.LENGTH / math.sqrt(3))
        self.assertAlmostEqual(self.rasterised_fraction((20, 20, 20), (20 + step,) * 3), 1.0, delta=0.05)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_grammar_and_volume(self):
        program_a, _, volume_a = generate(seed=11, niter=8)
        program_b, _, volume_b = generate(seed=11, niter=8)
        self.assertEqual(program_a, program_b)
        self.assertTrue(np.array_equal(volume_a, volume_b))


if __name__ == "__main__":
    unittest.main()
