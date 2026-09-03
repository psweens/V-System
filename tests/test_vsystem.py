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
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import libGenerator as lg  # noqa: E402
from libGenerator import setProperties, calBifurcation, getLength  # noqa: E402
from vSystem import F, I, A, example_grammar  # noqa: E402
from analyseGrammar import branching_turtle_to_coords, tokenise  # noqa: E402
from utils import interpolate_segments, bspline, rotate_about  # noqa: E402
from computeVoxel import (process_network, rasterise_segments, fit_to_volume,  # noqa: E402
                          normalise_axes, rasterise_line)
from main import generate_network, load_network, save_network  # noqa: E402

PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate(seed, niter, d0=20.0, tVol=(64, 64, 32), properties=PROPERTIES, **fit):
    seed_all(seed)
    setProperties(properties)
    program = F(niter, d0)
    nodes = interpolate_segments(branching_turtle_to_coords(program, d0))
    return program, nodes, process_network(nodes, tVol=tVol, **fit)


def final_row(program, d0=5.0):
    row = None
    for row in branching_turtle_to_coords(program, d0):
        pass
    return row


def connected_count(volume):
    """
    Returns (voxels set, voxels reachable from one of them under 26-connectivity),
    so that the two are equal exactly when the rendered network is one piece.
    """
    set_voxels = np.argwhere(volume)
    if len(set_voxels) == 0:
        return 0, 0
    shape = volume.shape
    neighbourhood = [(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)]
    seen = np.zeros(shape, dtype=bool)
    start = tuple(int(v) for v in set_voxels[0])
    seen[start] = True
    stack = [start]
    reached = 1
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in neighbourhood:
            a, b, c = x + dx, y + dy, z + dz
            if (0 <= a < shape[0] and 0 <= b < shape[1] and 0 <= c < shape[2]
                    and volume[a, b, c] and not seen[a, b, c]):
                seen[a, b, c] = True
                reached += 1
                stack.append((a, b, c))
    return len(set_voxels), reached


def angle_between(a, b):
    return math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)))


class ImportTests(unittest.TestCase):
    def test_every_module_imports_on_the_installed_stack(self):
        for name in ("libGenerator", "vSystem", "analyseGrammar", "utils",
                     "computeVoxel", "preprocessing", "visuals"):
            with self.subTest(module=name):
                importlib.import_module(name)


class TokeniserTests(unittest.TestCase):
    def test_operands_are_parsed_as_numbers_not_commands(self):
        tokens = list(tokenise("f(3.859e-05,0.127)+(-37.46)/(-25.9)*(5)f(-0.3,2.1)"))
        self.assertEqual(tokens, [("f", (3.859e-05, 0.127)), ("+", (-37.46,)), ("/", (-25.9,)),
                                  ("*", (5.0,)), ("f", (-0.3, 2.1))])

    def test_malformed_programs_raise(self):
        for program in ("+()", "f(nan,1)", "+(1e400)", "f(1,2)x3", "f(abc)", "]", "[f(1,1)"):
            with self.subTest(program=program):
                with self.assertRaises(ValueError):
                    for _ in branching_turtle_to_coords(program, 5.0):
                        pass


class TurtleTests(unittest.TestCase):
    """The interpreter of Galarreta-Valverde (2012), sections 3.1.1 and 3.3.1."""

    def setUp(self):
        setProperties(PROPERTIES)

    def test_initial_point_is_yielded_and_a_move_follows_the_direction(self):
        rows = list(branching_turtle_to_coords("f(10,5)", 5.0))
        self.assertEqual(rows[0][:4], (0.0, 0.0, 0.0, 5.0))
        np.testing.assert_allclose(rows[1][:3], (0.0, 10.0, 0.0))

    def test_turn_rotates_the_direction_about_the_perpendicular(self):
        # direction y rotated about perpendicular z by +90 degrees is -x
        np.testing.assert_allclose(final_row("+(90)f(10,5)")[:3], (-10.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(final_row("-(90)f(10,5)")[:3], (10.0, 0.0, 0.0), atol=1e-12)

    def test_roll_rotates_the_perpendicular_about_the_direction(self):
        # roll z about y by +90 degrees gives x; a turn about x then takes y to z
        np.testing.assert_allclose(final_row("/(90)+(90)f(10,5)")[:3], (0.0, 0.0, 10.0), atol=1e-12)
        np.testing.assert_allclose(final_row("*(90)+(90)f(10,5)")[:3], (0.0, 0.0, -10.0), atol=1e-12)

    def test_a_turn_followed_by_its_inverse_restores_the_heading(self):
        np.testing.assert_allclose(final_row("+(30)-(30)f(10,5)")[:3], (0.0, 10.0, 0.0), atol=1e-12)

    def test_bifurcation_angles_are_realised_exactly_on_opposite_sides(self):
        rows = [r for r in branching_turtle_to_coords("f(10,5)[+(30)f(10,5)][-(40)f(10,5)]", 5.0)
                if not math.isnan(r[0])]
        parent = np.array(rows[1][:3]) - np.array(rows[0][:3])
        first = np.array(rows[2][:3]) - np.array(rows[1][:3])
        second = np.array(rows[4][:3]) - np.array(rows[3][:3])
        self.assertAlmostEqual(angle_between(parent, first), 30.0, places=9)
        self.assertAlmostEqual(angle_between(parent, second), 40.0, places=9)
        self.assertAlmostEqual(angle_between(first, second), 70.0, places=9)

    def test_state_is_restored_by_a_pop(self):
        rows = list(branching_turtle_to_coords("f(10,5)[+(90)f(3,2)]f(10)", 5.0))
        self.assertTrue(math.isnan(rows[3][0]))
        self.assertEqual(rows[4][:4], rows[1][:4])            # restored point and diameter
        np.testing.assert_allclose(rows[5][:3], (0.0, 20.0, 0.0))  # heading restored too

    def test_bare_symbols_take_the_source_defaults(self):
        seed_all(0)
        rows = list(branching_turtle_to_coords("f", 4.0))
        length = np.linalg.norm(np.array(rows[1][:3]))
        self.assertGreaterEqual(length, 0.8 * lg.epsilon * 4.0 - 1e-9)
        self.assertLessEqual(length, 1.2 * lg.epsilon * 4.0 + 1e-9)
        self.assertEqual(rows[1][3], 4.0)  # a missing diameter keeps the current one
        seed_all(1)
        rows = list(branching_turtle_to_coords("+f(10,4)", 4.0))
        seed_all(1)
        expected = calBifurcation(4.0)["th1"]
        self.assertAlmostEqual(angle_between(np.array(rows[1][:3]), np.array([0.0, 1.0, 0.0])), expected, places=9)
        rows = list(branching_turtle_to_coords("/+(90)f(10,4)", 4.0))
        self.assertAlmostEqual(angle_between(np.array(rows[1][:3]), np.array([0.0, 0.0, 1.0])), 90.0 - lg.roll_angle, places=9)

    def test_braces_number_segments(self):
        rows = list(branching_turtle_to_coords("{f(1,1)f(1,1)}f(1,1){f(1,1)}", 1.0))
        self.assertEqual([r[4] for r in rows], [-1, 0, 0, 0, -1, 1, 1])


class InterpolationTests(unittest.TestCase):
    def test_bspline_starts_and_ends_on_the_control_points(self):
        control = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]])
        curve = bspline(control, subdivisions=3)
        np.testing.assert_allclose(curve[0], control[0])
        np.testing.assert_allclose(curve[-1], control[-1])
        self.assertEqual(len(curve), 8 * (len(control) + 1) + 1)

    def test_braced_segments_are_interpolated_and_others_kept(self):
        rows = list(branching_turtle_to_coords("{f(1,1)+(30)f(1,1)-(30)f(1,1)}f(1,1)", 1.0))
        nodes = interpolate_segments(rows, subdivisions=2)
        self.assertEqual(nodes.shape[0], 4)
        self.assertGreater(nodes.shape[1], len(rows))
        np.testing.assert_allclose(nodes[:3, 0], rows[0][:3])
        np.testing.assert_allclose(nodes[:3, -1], rows[-1][:3])
        self.assertFalse(np.isnan(nodes).any())

    def test_subdivision_zero_keeps_the_polyline(self):
        rows = list(branching_turtle_to_coords("{f(1,1)f(1,1)}[f(1,1)]", 1.0))
        nodes = interpolate_segments(rows, subdivisions=0)
        finite = ~np.isnan(nodes[0])
        self.assertEqual(int(finite.sum()), len([r for r in rows if not math.isnan(r[0])]))


class GrammarTests(unittest.TestCase):
    def setUp(self):
        setProperties(PROPERTIES)

    def test_generated_grammars_are_balanced_and_interpretable(self):
        for name, rule, niter in (("F", F, 8), ("I", I, 5), ("A", A, 5), ("example", example_grammar, 5)):
            with self.subTest(grammar=name):
                seed_all(3)
                program = rule(niter, 20.0)
                self.assertEqual(program.count("["), program.count("]"))
                self.assertEqual(program.count("{"), program.count("}"))
                rows = list(branching_turtle_to_coords(program, 20.0))
                self.assertGreater(len(rows), 1)
                self.assertTrue(all(math.isfinite(v) for _, params in tokenise(program) for v in params))

    def test_iterations_count_drawn_generations(self):
        seed_all(0)
        self.assertNotIn("f", F(0, 20.0))
        self.assertIn("f(", F(1, 20.0))
        one = F(1, 20.0)
        self.assertEqual(one.count("{"), 1)   # one drawn stem
        self.assertEqual(one.count("["), 2)   # two daughters left as non-terminals

    def test_daughters_turn_on_opposite_sides_and_roll_by_roll_angle(self):
        seed_all(0)
        setProperties({**PROPERTIES, "roll_angle": 70.0, "aneurysm_prob": 0.0, "stenosis_prob": 0.0})
        program = F(1, 20.0)
        self.assertRegex(program, r"\[\+\([0-9.]+\)/\(70\.0\)F\]\[-\([0-9.]+\)/\(70\.0\)F\]$")

    def test_anomaly_rules_change_the_diameter_locally(self):
        seed_all(0)
        setProperties({**PROPERTIES, "aneurysm_prob": 1.0, "stenosis_prob": 0.0, "aneurysm_factor": 1.5})
        diameters = [params[1] for cmd, params in tokenise(F(1, 20.0)) if cmd == "f"]
        self.assertEqual(sorted(set(diameters)), [20.0, 30.0])
        setProperties({**PROPERTIES, "aneurysm_prob": 0.0, "stenosis_prob": 1.0, "stenosis_factor": 0.5})
        diameters = [params[1] for cmd, params in tokenise(F(1, 20.0)) if cmd == "f"]
        self.assertEqual(sorted(set(diameters)), [10.0, 20.0])
        setProperties({**PROPERTIES, "aneurysm_prob": 0.0, "stenosis_prob": 0.0})
        diameters = [params[1] for cmd, params in tokenise(F(1, 20.0)) if cmd == "f"]
        self.assertEqual(set(diameters), {20.0})

    def test_deep_default_configuration_does_not_crash(self):
        seed_all(7)
        setProperties({"k": 3, "epsilon": 4.0, "randmarg": 0.3, "sigma": 5, "stochparams": True})
        program = F(12, 7.0)
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

    def test_properties_are_validated_and_defaulted(self):
        with self.assertRaises(ValueError):
            setProperties({"randmarg": 3})
        with self.assertRaises(ValueError):
            setProperties({"epsilon": 0})
        with self.assertRaises(ValueError):
            setProperties({"aneurysm_prob": 0.7, "stenosis_prob": 0.7})
        with self.assertRaises(ValueError):
            setProperties({"no_such_key": 1})
        setProperties({"epsilon": 6.0})
        self.assertEqual(lg.epsilon, 6.0)
        self.assertEqual(lg.roll_angle, lg.default["roll_angle"])
        setProperties(PROPERTIES)

    def test_bifurcation_rejects_non_positive_diameter(self):
        setProperties(PROPERTIES)
        for d0 in (0.0, -1.0):
            with self.assertRaises(ValueError):
                calBifurcation(d0)

    def test_bifurcation_obeys_murray_law_and_zamir_angles(self):
        setProperties(PROPERTIES)
        seed_all(3)
        for _ in range(100):
            p = calBifurcation(20.0)
            self.assertAlmostEqual(p["d1"] ** 3 + p["d2"] ** 3, 20.0 ** 3, places=8)
            self.assertGreater(p["th1"], 0.0)
            self.assertGreater(p["th2"], 0.0)
        setProperties({**PROPERTIES, "stochparams": False})
        p = calBifurcation(20.0)
        self.assertAlmostEqual(p["th1"], 37.4673, places=3)
        self.assertAlmostEqual(p["th2"], 37.4673, places=3)
        setProperties(PROPERTIES)

    def test_rotate_about_is_a_rotation(self):
        v = np.array([1.0, 2.0, 3.0])
        w = rotate_about(v, [0.0, 0.0, 1.0], 90.0)
        np.testing.assert_allclose(w, [-2.0, 1.0, 3.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(w), np.linalg.norm(v))


class VoxelTests(unittest.TestCase):
    def test_volume_is_binary_uint8_and_zero_initialised(self):
        tVol = (48, 48, 24)
        junk = np.full(tVol, 7, dtype=np.uint8)  # occupy a same-sized block, then free it
        del junk
        _, _, volume = generate(seed=1, niter=6, tVol=tVol)
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
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([100.0, 0.0, 0.0])
        p2 = p1 + 50.0 * np.array([math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0])
        p3 = p1 + 50.0 * np.array([math.cos(math.radians(-60)), math.sin(math.radians(-60)), 0.2])
        nan = np.full(4, np.nan)
        data = np.array([[*p0, 20.0], [*p1, 20.0], [*p2, 12.0], nan, [*p1, 20.0], [*p3, 12.0]]).T
        points, radii = fit_to_volume(data, (100, 60, 40), fit="isotropic", margin=1.0)
        a = points[:, 1] - points[:, 0]
        b = points[:, 2] - points[:, 1]
        self.assertAlmostEqual(angle_between(a, b), 60.0, places=6)
        self.assertAlmostEqual(2 * radii[0] / np.linalg.norm(a), 20.0 / 100.0, places=9)
        finite = ~np.isnan(points[0])
        self.assertTrue(np.all(points[:, finite] - radii[finite] >= 1.0 - 1e-9))
        self.assertTrue(np.all(points[:, finite] + radii[finite] <= np.array([[100], [60], [40]]) - 1.0 + 1e-9))

    def test_rendered_network_never_touches_the_volume_faces(self):
        _, _, volume = generate(seed=5, niter=6, tVol=(64, 48, 32))
        self.assertGreater(int(volume.sum()), 0)
        for face in (volume[0], volume[-1], volume[:, 0], volume[:, -1], volume[..., 0], volume[..., -1]):
            self.assertEqual(int(face.sum()), 0)

    def test_clipped_isotropic_fit_fills_the_kept_axes(self):
        _, nodes, _ = generate(seed=2, niter=5)
        tVol = (100, 80, 20)
        points, radii = fit_to_volume(nodes, tVol, fit="isotropic", margin=1.0, clip_axes=(2,))
        rmax = float(np.nanmax(radii))
        span = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
        available = np.array(tVol) - 2 * (1.0 + rmax)
        self.assertTrue(np.isclose(span[0], available[0]) or np.isclose(span[1], available[1]))
        self.assertTrue(np.all(span[:2] <= available[:2] + 1e-9))

    def test_stretch_mode_fills_every_axis(self):
        _, nodes, _ = generate(seed=2, niter=5)
        tVol = (100, 80, 60)
        points, radii = fit_to_volume(nodes, tVol, fit="stretch", margin=1.0)
        rmax = float(np.nanmax(radii))
        span = np.nanmax(points, axis=1) - np.nanmin(points, axis=1)
        self.assertTrue(np.allclose(span, np.array(tVol) - 2 * (1.0 + rmax)))


class CommandLineTests(unittest.TestCase):
    def test_main_writes_reproducible_uint8_volumes_with_sidecars(self):
        import json
        import tempfile
        import tifffile
        from main import main

        args = ["--count", "2", "--seed", "42", "--volume", "48", "40", "16", "--iterations", "3", "4"]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.assertEqual(main(args + ["--out", first]), 0)
            self.assertEqual(main(args + ["--out", second]), 0)
            tiffs = sorted(f for f in os.listdir(first) if f.endswith(".tiff"))
            self.assertEqual(len(tiffs), 2)
            first_seed = [f for f in tiffs if f.endswith("_s42.tiff")]
            self.assertEqual(len(first_seed), 1)
            volume = tifffile.imread(os.path.join(first, first_seed[0]))
            self.assertEqual(volume.dtype, np.uint8)
            self.assertEqual(volume.shape, (16, 40, 48))  # pages z, rows y, columns x
            self.assertEqual(set(np.unique(volume).tolist()), {0, 255})
            with open(os.path.join(first, first_seed[0][:-5] + ".json")) as handle:
                record = json.load(handle)
            self.assertEqual(record["seed"], 42)
            self.assertEqual(record["axis_order"], "zyx")
            for name in tiffs:
                with open(os.path.join(first, name), "rb") as a, open(os.path.join(second, name), "rb") as b:
                    self.assertEqual(a.read(), b.read())
                self.assertTrue(os.path.exists(os.path.join(first, name[:-5] + ".npz")))

    def test_the_sidecar_and_centreline_alone_reproduce_the_written_volume(self):
        import json
        import tifffile
        from main import main

        args = ["--count", "1", "--seed", "7", "--volume", "64", "64", "32",
                "--iterations", "4", "4", "--fit", "voxel_size", "--voxel-size", "6"]
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(main(args + ["--out", out]), 0)
            stem = next(n[:-5] for n in os.listdir(out) if n.endswith(".tiff"))
            with open(os.path.join(out, stem + ".json")) as handle:
                record = json.load(handle)
            self.assertEqual(record["units"], "um")
            self.assertEqual(record["voxel_size"], 6.0)
            self.assertIsNone(record["d_min"])

            network = load_network(os.path.join(out, stem + ".npz"))
            self.assertEqual(network["metadata"], record)
            self.assertEqual(network["nodes"].shape[0], 4)
            self.assertIn("f(", network["program"])
            volume = process_network(network["nodes"], tuple(record["volume"]), fit=record["fit"],
                                     voxel_size=record["voxel_size"], clip_axes=record["clip_axes"])
            self.assertGreater(int(volume.sum()), 0)
            written = tifffile.imread(os.path.join(out, stem + ".tiff"))
            np.testing.assert_array_equal(np.transpose(volume, (2, 1, 0)) * 255, written)

    def test_declared_units_and_d_min_reach_the_sidecar(self):
        import json
        from main import main

        args = ["--count", "1", "--seed", "3", "--volume", "48", "48", "24",
                "--iterations", "8", "8", "--units", "mm", "--d-min", "6"]
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(main(args + ["--out", out]), 0)
            stem = next(n[:-5] for n in os.listdir(out) if n.endswith(".tiff"))
            with open(os.path.join(out, stem + ".json")) as handle:
                record = json.load(handle)
            self.assertEqual(record["units"], "mm")
            self.assertEqual(record["d_min"], 6.0)
            nodes = load_network(os.path.join(out, stem + ".npz"))["nodes"]
            self.assertGreaterEqual(float(np.nanmin(nodes[3])), 6.0 * 0.5 - 1e-9)  # a stenosis halves it

    def test_a_non_positive_d_min_is_refused(self):
        from main import main
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit):
                main(["--count", "1", "--seed", "1", "--d-min", "0", "--out", out])


class ConnectivityTests(unittest.TestCase):
    """
    A capsule sets only the voxels whose centres it contains, so a vessel
    thinner than a voxel would rasterise as a dotted line and break the tree
    into fragments. Each segment is therefore also drawn as a connected digital
    line.
    """

    def setUp(self):
        setProperties(PROPERTIES)

    def test_a_sub_voxel_vessel_renders_as_one_unbroken_path(self):
        points = np.array([[10.0, 60.0], [10.0, 55.0], [10.0, 52.0]])  # oblique, so it misses centres
        radii = np.array([0.2, 0.2])
        tVol = (72, 72, 72)
        dotted = rasterise_segments(points, radii, tVol, connect=False)
        joined = rasterise_segments(points, radii, tVol, connect=True)
        self.assertGreater(int(joined.sum()), int(dotted.sum()))
        total, reached = connected_count(joined)
        self.assertEqual(total, reached)
        self.assertGreater(total, 50)
        dotted_total, dotted_reached = connected_count(dotted)
        self.assertLess(dotted_reached, dotted_total)  # the bare capsules really are broken

    def test_connecting_changes_nothing_once_a_vessel_fills_a_voxel(self):
        # every voxel the line sets is within sqrt(3)/2 of the centreline, so at
        # that radius and above the capsule already contains it
        rng = np.random.default_rng(0)
        tVol = (48, 48, 48)
        for radius in (math.sqrt(3) / 2, 1.0, 5.0):
            with self.subTest(radius=radius):
                for _ in range(12):
                    p0 = rng.uniform(15.0, 25.0, 3)
                    points = np.stack([p0, p0 + rng.uniform(-12.0, 12.0, 3)], axis=1)
                    radii = np.array([radius, radius])
                    np.testing.assert_array_equal(
                        rasterise_segments(points, radii, tVol, connect=False),
                        rasterise_segments(points, radii, tVol, connect=True))

    def test_a_rendered_network_is_one_piece_when_nothing_is_clipped(self):
        for seed, tVol, niter in ((3, (96, 96, 96), 7), (11, (96, 96, 96), 7)):
            with self.subTest(seed=seed):
                seed_all(seed)
                volume, _, _ = generate_network(niter, 20.0, PROPERTIES, tVol, clip_axes=())
                total, reached = connected_count(volume)
                self.assertGreater(total, 0)
                self.assertEqual(total, reached)

    def test_bare_capsules_break_the_same_network_apart(self):
        seed_all(3)
        tVol = (96, 96, 96)
        volume, _, _ = generate_network(7, 20.0, PROPERTIES, tVol, clip_axes=(), connect=False)
        total, reached = connected_count(volume)
        self.assertGreater(total, 0)
        self.assertLess(reached, total)

    def test_the_line_stays_inside_the_volume(self):
        volume = np.zeros((8, 8, 8), dtype=np.uint8)
        rasterise_line(volume, (-40.0, -40.0, -40.0), (40.0, 40.0, 40.0))
        self.assertGreater(int(volume.sum()), 0)
        rasterise_line(volume, (-40.0, -40.0, -40.0), (-30.0, -30.0, -30.0))  # wholly outside
        outside = np.zeros((8, 8, 8), dtype=np.uint8)
        rasterise_line(outside, (100.0, 100.0, 100.0), (200.0, 200.0, 200.0))
        self.assertEqual(int(outside.sum()), 0)

    def test_a_clipped_slab_may_still_render_several_pieces(self):
        # a branch that leaves a slab and re-enters is two vessels in the image,
        # as it would be in a real acquisition; connect does not change that
        seed_all(3)
        volume, _, _ = generate_network(8, 20.0, PROPERTIES, (128, 128, 24), clip_axes=(2,))
        total, reached = connected_count(volume)
        self.assertGreater(total, 0)
        self.assertLessEqual(reached, total)


class CentrelinePersistenceTests(unittest.TestCase):
    """
    The saved centreline is the source of truth and the volume is derived from
    it, so a reader can re-render a network at another resolution without
    regenerating it.
    """

    def setUp(self):
        setProperties(PROPERTIES)

    def test_saved_nodes_re_render_the_generated_volume(self):
        tVol = (64, 64, 32)
        cases = (("isotropic", {"clip_axes": (2,)}),
                 ("voxel_size", {"voxel_size": 6.0, "clip_axes": (2,)}))
        for fit, settings in cases:
            with self.subTest(fit=fit):
                seed_all(17)
                volume, program, nodes = generate_network(6, 20.0, PROPERTIES, tVol,
                                                          fit=fit, **settings)
                self.assertGreater(int(volume.sum()), 0)
                with tempfile.TemporaryDirectory() as out:
                    path = os.path.join(out, "net.npz")
                    save_network(path, nodes, program=program, metadata={"units": "um"})
                    network = load_network(path)
                np.testing.assert_array_equal(network["nodes"], nodes)
                self.assertEqual(network["program"], program)
                self.assertEqual(network["metadata"], {"units": "um"})
                again = process_network(network["nodes"], tVol, fit=fit, **settings)
                self.assertTrue(np.array_equal(again, volume))

    def test_a_centreline_saved_without_provenance_still_reads_back(self):
        _, nodes, _ = generate(seed=8, niter=4)
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "bare.npz")
            save_network(path, nodes)
            network = load_network(path)
        np.testing.assert_array_equal(network["nodes"], nodes)
        self.assertIsNone(network["program"])
        self.assertIsNone(network["metadata"])

    def test_nodes_do_not_depend_on_the_volume_they_were_rendered_into(self):
        seed_all(9)
        _, _, small = generate_network(5, 20.0, PROPERTIES, (64, 64, 32), fit="isotropic")
        seed_all(9)
        _, _, large = generate_network(5, 20.0, PROPERTIES, (256, 128, 64),
                                       fit="voxel_size", voxel_size=3.0)
        np.testing.assert_array_equal(small, large)

    def test_a_centreline_is_far_smaller_than_the_volume_it_renders(self):
        _, nodes, _ = generate(seed=6, niter=8)
        tVol = (256, 256, 128)
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "net.npz")
            save_network(path, nodes)
            self.assertLess(os.path.getsize(path), int(np.prod(tVol)) / 10)


class UnitConventionTests(unittest.TestCase):
    """
    Grammar units are physical: one unit is one micrometre, so voxel_size is a
    modality's voxel size and calibre in voxels goes as 1 / voxel_size.
    """

    def setUp(self):
        setProperties(PROPERTIES)

    def test_radii_in_voxels_scale_inversely_with_voxel_size(self):
        _, nodes, _ = generate(seed=4, niter=6)
        tVol = (64, 64, 32)
        _, coarse = fit_to_volume(nodes, tVol, fit="voxel_size", voxel_size=8.0)
        _, fine = fit_to_volume(nodes, tVol, fit="voxel_size", voxel_size=2.0)
        self.assertGreater(float(np.nanmax(coarse)), 0.0)
        np.testing.assert_allclose(fine, 4.0 * coarse)

    def test_rasterised_calibre_scales_inversely_with_voxel_size(self):
        diameter = 8.0  # grammar units; a single straight vessel along x
        nodes = np.array([[20.0, 80.0], [15.0, 15.0], [15.0, 15.0], [diameter, diameter]])
        tVol = (128, 48, 48)
        areas = {}
        for voxel_size in (1.0, 0.5):
            volume = process_network(nodes, tVol, fit="voxel_size", voxel_size=voxel_size)
            areas[voxel_size] = int(volume[tVol[0] // 2].sum())  # a mid-length cross-section
            ideal = math.pi * (diameter / 2.0 / voxel_size) ** 2
            self.assertAlmostEqual(areas[voxel_size] / ideal, 1.0, delta=0.05)
        self.assertAlmostEqual(math.sqrt(areas[0.5] / areas[1.0]), 2.0, delta=0.05)

    def test_a_non_positive_voxel_size_is_rejected(self):
        _, nodes, _ = generate(seed=4, niter=4)
        for voxel_size in (None, 0.0, -1.0):
            with self.subTest(voxel_size=voxel_size):
                with self.assertRaises(ValueError):
                    fit_to_volume(nodes, (64, 64, 32), fit="voxel_size", voxel_size=voxel_size)


class AxisSelectionTests(unittest.TestCase):
    def setUp(self):
        setProperties(PROPERTIES)

    def test_names_indices_and_bare_strings_all_name_axes(self):
        self.assertEqual(normalise_axes(()), ())
        self.assertEqual(normalise_axes("z"), (2,))
        self.assertEqual(normalise_axes(("Z", 0, "x")), (0, 2))
        self.assertEqual(normalise_axes([1, 1]), (1,))
        self.assertEqual(normalise_axes(np.array([2, 0])), (0, 2))

    def test_an_unknown_axis_raises_rather_than_being_ignored(self):
        for axes in (("w",), ("",), (3,), (-1,), (1.5,), (None,)):
            with self.subTest(axes=axes):
                with self.assertRaises(ValueError):
                    normalise_axes(axes)

    def test_a_named_clip_axis_clips(self):
        _, nodes, _ = generate(seed=2, niter=5)
        tVol = (100, 80, 20)
        named = fit_to_volume(nodes, tVol, fit="isotropic", clip_axes=("z",))
        indexed = fit_to_volume(nodes, tVol, fit="isotropic", clip_axes=(2,))
        unclipped = fit_to_volume(nodes, tVol, fit="isotropic", clip_axes=())
        for a, b in zip(named, indexed):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(named[1], unclipped[1]))
        with self.assertRaises(ValueError):
            fit_to_volume(nodes, tVol, fit="isotropic", clip_axes=("depth",))


class StoppingCriterionTests(unittest.TestCase):
    """`d_min` stops a branch once its diameter falls below a resolvable calibre."""

    def test_no_d_min_leaves_the_grammar_unchanged(self):
        setProperties(PROPERTIES)
        seed_all(5)
        without = F(6, 20.0)
        seed_all(5)
        explicit_none = F(6, 20.0, d_min=None)
        self.assertEqual(without, explicit_none)

    def test_no_vessel_is_drawn_below_d_min(self):
        setProperties({**PROPERTIES, "aneurysm_prob": 0.0, "stenosis_prob": 0.0})
        seed_all(5)
        unlimited = F(12, 20.0)
        seed_all(5)
        limited = F(12, 20.0, d_min=10.0)
        drawn = [p[1] for c, p in tokenise(limited) if c == "f" and len(p) > 1]
        self.assertTrue(drawn)
        self.assertGreaterEqual(min(drawn), 10.0)
        self.assertLess(len(limited), len(unlimited))

    def test_d_min_reaches_the_generation_count_it_implies(self):
        # deterministic daughters shrink by 2^(-1/k) a generation, so the last
        # drawn calibre lies in [d_min, d_min * 2^(1/k)) and the depth is about
        # log(d0 / d_min) / log(2^(1/k)).
        k = PROPERTIES["k"]
        setProperties({**PROPERTIES, "stochparams": False,
                       "aneurysm_prob": 0.0, "stenosis_prob": 0.0})
        seed_all(0)
        d0, d_min = 20.0, 3.0
        diameters = [p[1] for c, p in tokenise(F(100, d0, d_min=d_min)) if c == "f" and len(p) > 1]
        self.assertGreaterEqual(min(diameters), d_min)
        self.assertLess(min(diameters), d_min * 2 ** (1.0 / k))
        depth = math.log(d0 / min(diameters)) / math.log(2 ** (1.0 / k))
        self.assertAlmostEqual(depth, math.floor(math.log(d0 / d_min) / math.log(2 ** (1.0 / k))),
                               delta=1.0)

    def test_generate_network_honours_d_min(self):
        setProperties(PROPERTIES)
        seed_all(13)
        _, program, nodes = generate_network(12, 20.0, {**PROPERTIES, "aneurysm_prob": 0.0,
                                                        "stenosis_prob": 0.0},
                                             (64, 64, 32), d_min=8.0)
        self.assertGreaterEqual(float(np.nanmin(nodes[3])), 8.0 - 1e-9)
        self.assertIn("f(", program)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_grammar_and_volume(self):
        program_a, _, volume_a = generate(seed=11, niter=6)
        program_b, _, volume_b = generate(seed=11, niter=6)
        self.assertEqual(program_a, program_b)
        self.assertTrue(np.array_equal(volume_a, volume_b))


if __name__ == "__main__":
    unittest.main()
