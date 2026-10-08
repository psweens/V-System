"""
Tests of the capillary fill and the rungs as grow_network, the command line
and the library use them: the new options and their refusals, the rungs'
isolation from everything grown before them, determinism and scale, the
families aligned_bed and capillary_bed, and the free-extent pass the fill
skips. The grammar itself is tested in tests/test_vsystem.py and the rung
stage on its own in tests/test_connections.py.

Run from the repository root with
    python -m unittest tests.test_fill
The fill families end to end and as library members take minutes and run
only with VSYSTEM_SLOW_TESTS=1.
"""
import contextlib
import io
import json
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

import library  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from main import FAMILIES, grow_network  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")
# epsilon 10 gives capillary stems long enough to carry rung sites clear of their junctions
PROPERTIES = {"k": 3, "epsilon": 10.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
VOLUME = (48, 48, 24)
# a free walked tree with avoidance and anastomosis, below 2 d_min from its third generation
GROWTH = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
          "anastomose_mode": "any", "anastomosis_fraction": 0.5, "d_min": 1.0}
OFF = {"capillary_generations": 0, "capillary_runs": 1, "cross_connect": False, "rung_below": 2.0,
       "rung_spacing": 20.0, "rung_radius": 8.0, "rung_lateral_deg": 60.0, "rung_axis": None,
       "rung_min_separation": 2}


def grow(seed, niter=8, d0=4.0, properties=PROPERTIES, **options):
    """One network from the grammar's seeded draws, with grow_network's options."""
    random.seed(seed)
    np.random.seed(seed)
    return grow_network(niter, d0, dict(properties), VOLUME, seed=seed, **options)


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


def global_states():
    return random.getstate(), np.random.get_state()


class OptionTests(unittest.TestCase):
    def test_the_options_default_to_off_and_reach_the_sidecar_and_kwargs(self):
        args = main.build_parser("tree").parse_args([])
        self.assertEqual({key: main.shaping_options(args)[key] for key in OFF}, OFF)
        args = main.build_parser("tree").parse_args(
            ["--d-min", "1", "--capillary-generations", "2", "--capillary-runs", "3", "--cross-connect",
             "--rung-below", "1.5", "--rung-spacing", "30", "--rung-radius", "6", "--rung-lateral-deg", "45",
             "--rung-axis", "[1, 0, 0]", "--rung-min-separation", "3"])
        self.assertEqual({key: main.shaping_options(args)[key] for key in OFF},
                         {"capillary_generations": 2, "capillary_runs": 3, "cross_connect": True, "rung_below": 1.5,
                          "rung_spacing": 30.0, "rung_radius": 6.0, "rung_lateral_deg": 45.0, "rung_axis": [1, 0, 0],
                          "rung_min_separation": 3})
        # a preset's rungs are switched off on the command line
        bed = main.build_parser("capillary_bed").parse_args(["--no-cross-connect"])
        self.assertFalse(bed.cross_connect)
        self.assertEqual(bed.capillary_generations, 3)
        argv = ["--count", "1", "--seed", "4", "--volume", "48", "48", "24", "--iterations", "5", "5", "--d-min", "2",
                "--capillary-generations", "1", "--capillary-runs", "2"]
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main.main(argv + ["--out", out]), 0)
            stem = next(name[:-5] for name in os.listdir(out) if name.endswith(".json"))
            with open(os.path.join(out, stem + ".json")) as handle:
                sidecar = json.load(handle)
        self.assertEqual((sidecar["capillary_generations"], sidecar["capillary_runs"]), (1, 2))
        self.assertFalse(sidecar["cross_connect"])
        self.assertEqual(sidecar["rungs"], 0)
        self.assertEqual(sidecar["rng_streams"]["rungs"], [4, 5])

    def test_grow_network_and_the_command_line_refuse_the_same_settings_before_anything_is_written(self):
        refused = {
            "generations negative": {"capillary_generations": -1},
            "runs zero": {"capillary_runs": 0},
            "generations a float": {"capillary_generations": 1.0},
            "runs a bool": {"capillary_runs": True},
            "generations without d_min": {"capillary_generations": 2, "d_min": None},
            "rungs without d_min": {"cross_connect": True, "d_min": None},
            "rungs not a bool": {"cross_connect": 1},
            "spacing zero": {"rung_spacing": 0.0},
            "radius negative": {"rung_radius": -1.0},
            "below zero": {"rung_below": 0.0},
            "lateral beyond a right angle": {"rung_lateral_deg": 120.0},
            "a zero axis": {"rung_axis": [0, 0, 0]},
            "separation zero": {"rung_min_separation": 0},
        }
        for name, options in refused.items():
            with self.subTest(refusal=name):
                settings = dict(GROWTH, **options)
                with self.assertRaises(ValueError):
                    grow(3, niter=4, **settings)
        cli = {
            "generations negative": ["--d-min", "1", "--capillary-generations", "-1"],
            "runs zero": ["--d-min", "1", "--capillary-runs", "0"],
            "generations without d_min": ["--capillary-generations", "2"],
            "rungs without d_min": ["--cross-connect"],
            "spacing zero": ["--d-min", "1", "--cross-connect", "--rung-spacing", "0"],
            "a zero axis": ["--d-min", "1", "--cross-connect", "--rung-axis", "[0, 0, 0]"],
            "lateral beyond a right angle": ["--d-min", "1", "--cross-connect", "--rung-lateral-deg", "120"],
            "aligned_bed without d_min": ["--family", "aligned_bed"],
            "capillary_bed without d_min": ["--family", "capillary_bed"],
        }
        base = ["--count", "1", "--seed", "3", "--volume", "48", "48", "24", "--iterations", "4", "4"]
        for name, argv in cli.items():
            with self.subTest(refusal=name):
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        main.main(base + argv + ["--out", out])
                    self.assertNotEqual(caught.exception.code, 0)
                    self.assertEqual(os.listdir(out), [])


class IsolationTests(unittest.TestCase):
    """
    Order item 24 at the level of growth: the rungs run after anastomosis on
    a stream of their own, so turning them on changes nothing drawn before
    them. The development measurement over seeds 1-10 of GROWTH at d0 4:
    rungs 1, 2, 1, 7, 0, 4, 5, 1, 2, 2 from 13-38 sites, and 5-11
    anastomosis bridges; the seeds asserted, 1-3, each make a rung.
    """

    SEEDS = (1, 2, 3)

    def test_anastomosis_and_everything_before_it_are_identical_with_rungs_on_and_off(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                with recorded_generators() as created_off:
                    off = grow(seed, **GROWTH)
                states_off = global_states()
                with recorded_generators() as created_on:
                    on = grow(seed, cross_connect=True, **GROWTH)
                states_on = global_states()
                self.assertGreater(len(on["rungs"]), 0)
                self.assertEqual(off["rungs"], [])
                # the walk and anastomosis streams are created alike and end alike; the rungs add tag 5
                self.assertEqual([seed_ for seed_, _ in created_off], [[seed, 1], [seed, 2]])
                self.assertEqual([seed_ for seed_, _ in created_on], [[seed, 1], [seed, 2], [seed, 5]])
                for (_, a), (_, b) in zip(created_off, created_on):
                    self.assertEqual(a.bit_generator.state, b.bit_generator.state)
                # the bridges, the anastomosis counters and every column grown before the rungs
                self.assertEqual(on["bridges"], off["bridges"])
                for key in main.EVENT_KEYS:
                    if not key.startswith("rung_"):
                        self.assertEqual(on["events"][key], off["events"][key], key)
                columns = off["nodes"].shape[1]
                np.testing.assert_array_equal(on["nodes"][:, :columns], off["nodes"])
                np.testing.assert_array_equal(on["tree"][:columns], off["tree"])
                self.assertTrue(np.all((on["tree"][columns:] == BRIDGE) | (on["tree"][columns:] == -1)))
                self.assertEqual(on["programs"], off["programs"])
                self.assertEqual(states_on[0], states_off[0])
                for a, b in zip(states_on[1], states_off[1]):
                    np.testing.assert_array_equal(a, b)
                # the partition of the sites holds and the record matches the counter
                events = on["events"]
                self.assertEqual(events["rung_sites"],
                                 events["rung_sites_near_junction"] + events["rung_sites_consumed"]
                                 + events["rung_no_partner"] + events["rung_collision_failed"] + events["rung_bridges"])
                self.assertEqual(events["rung_bridges"], len(on["rungs"]))
                self.assertTrue(all(value == 0 for key, value in off["events"].items() if key.startswith("rung_")))

    def test_growth_with_rungs_is_deterministic_and_scales_with_the_unit(self):
        reference = grow(2, cross_connect=True, collision_margin=1.0, **GROWTH)
        self.assertEqual(grow(2, cross_connect=True, collision_margin=1.0, **GROWTH)["rungs"], reference["rungs"])
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                settings = dict(GROWTH, d_min=scale)
                scaled = grow(2, d0=4.0 * scale, cross_connect=True, collision_margin=scale, **settings)
                self.assertEqual(scaled["nodes"].shape, reference["nodes"].shape)
                self.assertEqual(scaled["events"], reference["events"])
                # rounding near zero coordinates, as in tests/test_guidance.py's scale test
                floor = 1e-14 * float(np.nanmax(np.abs(reference["nodes"][:3])))
                np.testing.assert_allclose(scaled["nodes"], scale * reference["nodes"], rtol=1e-12, atol=scale * floor)


class FamilyTests(unittest.TestCase):
    ALIGNED_BED = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                   "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                   "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                   "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                 "polarity": "partner", "length": 2.0, "onset": 0.0},
                                {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0, "onset": 0.0}],
                   "capillary_generations": 2, "capillary_runs": 4,
                   "cross_connect": True, "rung_below": 2.0, "rung_spacing": 40.0, "rung_radius": 8.0,
                   "rung_lateral_deg": 60.0, "rung_axis": [1, 0, 0], "rung_min_separation": 2}
    CAPILLARY_BED = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 1.0, "grow_in_volume": True,
                     "capillary_generations": 3, "capillary_runs": 2,
                     "cross_connect": True, "rung_below": 2.0, "rung_spacing": 20.0, "rung_radius": 8.0,
                     "rung_lateral_deg": 60.0, "rung_axis": None, "rung_min_separation": 2}

    def test_the_presets_are_as_approved_and_aligned_bed_builds_on_aligned_tight(self):
        self.assertEqual(FAMILIES["aligned_bed"], self.ALIGNED_BED)
        self.assertEqual(FAMILIES["capillary_bed"], self.CAPILLARY_BED)
        base = {key: value for key, value in self.ALIGNED_BED.items() if key not in OFF}
        self.assertEqual(base, FAMILIES["aligned_tight"])
        self.assertEqual((library.BOX_C["aligned_bed"], library.BOX_C["capillary_bed"]), (20.0, 15.0))

    def test_a_filled_pair_in_the_volume_skips_the_free_extent_pass_the_released_families_keep(self):
        # the pass interprets the first tree once more on a walk stream of its own; a box
        # grown at a fixed voxel size uses neither its extent nor its reach
        def walk_streams(family, **override):
            argv = library.member_argv(family, 2.52)
            args = main.build_parser(family).parse_args(argv)
            random.seed(5)
            np.random.seed(5)
            properties, d0, niter = main.sample_parameters(args)
            options = dict(main.shaping_options(args), capillary_generations=0, capillary_runs=1, cross_connect=False)
            options.update(override)
            with recorded_generators() as created:
                main.grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit, voxel_size=args.voxel_size,
                                  subdivisions=args.subdivisions, d_min=args.d_min,
                                  grow_in_volume=args.grow_in_volume, seed=5, **options)
            return [seed for seed, _ in created].count([5, 1])

        self.assertEqual(walk_streams("aligned_tight"), 2)
        self.assertEqual(walk_streams("aligned_tight", capillary_runs=2), 1)


@SLOW
class FillFamilyRunTests(unittest.TestCase):
    """The fill families end to end at the smallest default ratio."""

    def test_each_fill_family_runs_on_the_command_line_and_in_the_connectivity_check(self):
        import check_connectivity
        for family, kind in (("aligned_bed", "axis"), ("capillary_bed", "none")):
            with self.subTest(family=family):
                argv = ["--family", family] + library.member_argv(family, 2.52)[2:] + ["--count", "1", "--seed", "3"]
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main.main(argv + ["--out", out]), 0)
                    stem = next(name[:-5] for name in os.listdir(out) if name.endswith(".json"))
                    with open(os.path.join(out, stem + ".json")) as handle:
                        sidecar = json.load(handle)
                self.assertEqual(sidecar["family"], family)
                self.assertEqual(sidecar["frame"]["kind"], kind)
                self.assertGreater(sidecar["events"]["rung_sites"], 0)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(check_connectivity.generate_and_check(1, argv), 0)

    def test_a_fill_member_grows_in_its_cube_with_its_rungs_recorded_and_scales_with_the_unit(self):
        member = library.grow_member("capillary_bed", 2.52, 4)
        grown = member["grown"]
        side = library.BOX_C["capillary_bed"] * 2.52
        np.testing.assert_allclose(grown["growth_box_um"], [side, side, side], rtol=1e-12)
        row = library.describe_member(grown, 1.0)
        self.assertEqual(row["rungs"], len(grown["rungs"]))
        self.assertEqual(grown["events"]["rung_bridges"], len(grown["rungs"]))
        twice = library.grow_member("capillary_bed", 2.52, 4, d_min=2.0, collision_margin=2.0)["grown"]
        self.assertEqual(twice["events"], grown["events"])
        floor = 1e-14 * side
        np.testing.assert_allclose(twice["nodes"], 2.0 * grown["nodes"], rtol=1e-12, atol=2.0 * floor)


if __name__ == "__main__":
    unittest.main()
