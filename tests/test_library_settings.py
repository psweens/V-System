"""
Tests of a library's family settings, `vsystem-library --family-settings
FAMILY JSON`: their normalisation, the flags they add to a member's command
line, which stays a runnable `vsystem` command line growing the same
network, their records in the manifest and the archives, resuming, the
refusals, and the other families' members, which they leave unchanged. The
3.8 pin (tests/test_pinned_release_3_8.py) holds a library without them to
what 3.8 grows.

Run from the repository root with
    python -m unittest tests.test_library_settings
which takes about a minute. The benchmark of family settings runs only
with VSYSTEM_SLOW_TESTS=1 (see SettingsBenchmark).
"""
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bed  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from library import grow_member, member_argv, normalise_family_settings  # noqa: E402
from tests.test_pinned_release_3_8 import INDEX_COLUMNS_3_8  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")
FOAM_SPACING_5 = {"bed": {"spacing": 5}}
CAPILLARY_4 = {"capillary_generations": 4}
# the keys the order names, in order
KEYS = ("persistence", "anastomosis_fraction", "anastomosis_radius", "anastomosis_min_separation",
        "aneurysm_prob", "stenosis_prob", "root_offsets", "guidance", "capillary_generations", "capillary_runs",
        "cross_connect", "rung_below", "rung_spacing", "rung_radius", "rung_lateral_deg", "rung_axis",
        "rung_min_separation", "bed")
# every key on foam: refused together by the command line, but each parses back as given
EVERY_KEY = {"persistence": 6, "anastomosis_fraction": 0.75, "anastomosis_radius": 20, "anastomosis_min_separation": 4.0,
             "aneurysm_prob": 0.05, "stenosis_prob": -0.0001, "root_offsets": [[-5, 0, 0.5], [5.0, 0, 0]],
             "guidance": [{"below": 2.0, "field": "plane", "normal": [1, 0, 0], "length": 5.0}],
             "capillary_generations": 2, "capillary_runs": 3, "cross_connect": False, "rung_below": 1.5,
             "rung_spacing": 25, "rung_radius": 6.5, "rung_lateral_deg": 45, "rung_axis": [0, 0, 1],
             "rung_min_separation": 3, "bed": {"density": 0.3, "jitter": 0.0}}


def run_cli(argv):
    """Runs the library command line in this process, returning (status, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = library.main(argv)
        except SystemExit as exc:
            # as the interpreter does: a message is printed to stderr and the status is 1
            if isinstance(exc.code, str):
                err.write(exc.code + "\n")
                status = 1
            else:
                status = exc.code or 0
    return status, out.getvalue(), err.getvalue()


def run_main(argv):
    """main.main(argv) with its output captured; returns (status, SystemExit or None)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return main.main(argv), None
        except SystemExit as caught:
            return None, caught


def parsed(family, argv):
    return main.build_parser(family).parse_args(argv)


class NormalisationTests(unittest.TestCase):
    def test_the_keys_are_parser_dests_in_the_order_given(self):
        self.assertEqual(library.FAMILY_SETTING_KEYS, KEYS)
        dests = {action.dest for action in main.build_parser()._actions}
        self.assertEqual([key for key in KEYS if key not in dests], [])

    def test_a_bed_is_merged_over_the_preset_and_written_out_whole(self):
        foam = normalise_family_settings("foam", {"bed": {"density": 0.3}})
        self.assertEqual(list(foam["bed"]), list(bed.SETTING_KEYS))                       # 13 keys
        self.assertEqual(foam["bed"], dict(bed.parse_settings(main.FAMILIES["foam"]["bed"]), density=0.3))
        fibre = normalise_family_settings("fibre", {"bed": {"density": 0.3}})
        self.assertEqual(list(fibre["bed"]), list(bed.FIBRE_SETTING_KEYS))                # 21 keys
        self.assertEqual(fibre["bed"], dict(bed.parse_settings(main.FAMILIES["fibre"]["bed"]), density=0.3))
        # the preset's own kind and layout may be repeated, as a normalised bed does
        self.assertEqual(normalise_family_settings("fibre", {"bed": {"kind": "fibre", "layout": "sheets",
                                                                     "density": 0.3}}), fibre)
        # an empty bed object is the preset's bed
        self.assertEqual(normalise_family_settings("foam", {"bed": {}})["bed"],
                         bed.parse_settings(main.FAMILIES["foam"]["bed"]))

    def test_values_are_typed_and_keys_ordered(self):
        settings = normalise_family_settings("foam", dict(reversed(list(EVERY_KEY.items()))))
        self.assertEqual(list(settings), list(KEYS))
        self.assertEqual(settings["persistence"], 6.0)
        self.assertIsInstance(settings["persistence"], float)
        self.assertIsInstance(settings["anastomosis_radius"], float)
        self.assertEqual(settings["anastomosis_min_separation"], 4)
        self.assertIsInstance(settings["anastomosis_min_separation"], int)
        self.assertIs(settings["cross_connect"], False)
        self.assertEqual(settings["root_offsets"], [[-5, 0, 0.5], [5.0, 0, 0]])
        # an array is a copy, and a tuple given through the Python API becomes a list
        given = {"rung_axis": (0, 0, 1)}
        copied = normalise_family_settings("capillary_bed", given)
        self.assertEqual(copied, {"rung_axis": [0, 0, 1]})
        array = [1, 0, 0]
        copied = normalise_family_settings("capillary_bed", {"rung_axis": array})
        array[0] = 7
        self.assertEqual(copied["rung_axis"], [1, 0, 0])
        # numpy values through the Python API are the JSON values they stand for
        for given in (np.array([1, 0, 0]), [np.int64(1), 0, np.float64(0.0)]):
            normalised = normalise_family_settings("capillary_bed", {"rung_axis": given})
            self.assertEqual(normalised, {"rung_axis": [1, 0, 0.0]})
            self.assertEqual(json.dumps(normalised["rung_axis"]), "[1, 0, 0.0]" if isinstance(given, list)
                             else "[1, 0, 0]")

    def test_normalising_twice_changes_nothing(self):
        for family, spec in (("foam", FOAM_SPACING_5), ("capillary_bed", CAPILLARY_4), ("foam", EVERY_KEY),
                             ("fibre", {"bed": {"density": 0.3, "run": 100}, "persistence": 12}),
                             ("aligned", {"guidance": [{"below": None, "field": "plane", "normal": [1, 0, 0],
                                                        "length": 4.0}]})):
            with self.subTest(family=family):
                once = normalise_family_settings(family, spec)
                self.assertEqual(normalise_family_settings(family, once), once)
                self.assertEqual(json.loads(json.dumps(once, allow_nan=False)), once)

    def test_what_is_refused(self):
        refused = [
            ("foam", [], "JSON object"), ("foam", None, "JSON object"), ("foam", {}, "empty"),
            ("foam", {"spacing": 5}, "unknown setting"),
            # the library's own settings
            ("foam", {"d0": 3.0}, "the library fixes it"), ("foam", {"collision_margin": 2.0}, "the library fixes it"),
            ("tree", {"avoid_collisions": False}, "the library fixes it"),
            ("mesh", {"grow_in_volume": False}, "the library fixes it"),
            ("foam", {"iterations": [4, 4]}, "the library fixes it"), ("foam", {"d_min": 2.0}, "the library fixes it"),
            # what makes a family what it is
            ("foam", {"tortuosity": "stems"}, "changes what the family is"),
            ("mesh", {"anastomose": False}, "changes what the family is"),
            ("mesh", {"anastomose_mode": "any"}, "changes what the family is"),
            # the grammar's ranges and the numerics
            ("tree", {"epsilon": [3, 5]}, "same for every family"), ("tree", {"sigma": 4}, "same for every family"),
            ("tree", {"subdivisions": 2}, "same for every family"), ("tree", {"clip_axes": []}, "same for every family"),
            # the bed
            ("tree", {"bed": {"spacing": 5}}, "grows no bed"), ("capillary_bed", {"bed": {}}, "grows no bed"),
            ("foam", {"bed": None}, "turns the preset's bed off"), ("foam", {"bed": [1]}, "JSON object"),
            ("foam", {"bed": {"kind": "fibre"}}, "kind 'fibre' cannot replace"),
            ("fibre", {"bed": {"layout": "pair"}}, "layout 'pair' cannot replace"),
            ("foam", {"bed": {"layout": "pair"}}, "unknown bed setting 'layout'"),
            ("foam", {"bed": {"axis": [1, 0, 0]}}, "unknown bed setting 'axis'"),
            ("foam", {"bed": {"density": 0.7}}, "density"), ("foam", {"bed": {"girth": 8.0}}, "girth"),
            # types
            ("foam", {"persistence": True}, "finite number"), ("foam", {"persistence": "5"}, "finite number"),
            ("foam", {"persistence": math.nan}, "finite number"), ("foam", {"persistence": 10 ** 400}, "finite number"),
            ("capillary_bed", {"capillary_generations": 3.5}, "whole number"),
            ("capillary_bed", {"capillary_generations": True}, "whole number"),
            ("capillary_bed", {"cross_connect": 1}, "true or false"),
            ("aligned", {"guidance": None}, "JSON array"), ("aligned", {"root_offsets": {"x": 1}}, "JSON array"),
            ("capillary_bed", {"rung_axis": [math.inf, 0, 0]}, "finite numbers"),
            ("foam", {"bed": {1: 2}}, "unknown bed setting 1"),
            # a negative number argparse would read as an option
            ("tree", {"aneurysm_prob": -1e-05}, "read as an option"),
            ("nonesuch", {"persistence": 5}, "unknown family"),
        ]
        for family, spec, message in refused:
            with self.subTest(family=family, spec=spec):
                with self.assertRaises(ValueError) as caught:
                    normalise_family_settings(family, spec)
                self.assertIn(message, str(caught.exception))


class ArgvTests(unittest.TestCase):
    def test_without_settings_the_command_line_is_3_8_s(self):
        # 3.8's command lines, written out (the 3.8 pin compares every family under every option)
        foam = ["--family", "foam", "--d0", "4.0", "0", "--d0-min", "4.0", "--d-min", "1.0", "--iterations", "64",
                "64", "--collision-margin", "1.0", "--volume", "3", "3", "3", "--fit", "voxel_size", "--voxel-size",
                "20.0"]
        self.assertEqual(member_argv("foam", 4.0, settings=None), foam)
        self.assertEqual(member_argv("tree", 4.0, settings=None), foam[:1] + ["tree"] + foam[2:-1] + ["1.0",
                                                                                                       "--avoid-collisions"])

    def test_the_flags_follow_the_others_in_key_order_and_parse_back_as_normalised(self):
        settings = normalise_family_settings("foam", EVERY_KEY)
        plain = member_argv("foam", 4.0)
        argv = member_argv("foam", 4.0, settings=EVERY_KEY)
        self.assertEqual(argv[:len(plain)], plain)
        flags = [token for token in argv[len(plain):] if token.startswith("--")]
        self.assertEqual(flags, ["--persistence", "--anastomosis-fraction", "--anastomosis-radius",
                                 "--anastomosis-min-separation", "--aneurysm-prob", "--stenosis-prob",
                                 "--root-offsets", "--guidance", "--capillary-generations", "--capillary-runs",
                                 "--no-cross-connect", "--rung-below", "--rung-spacing", "--rung-radius",
                                 "--rung-lateral-deg", "--rung-axis", "--rung-min-separation", "--bed"])
        tail = argv[len(plain):]
        self.assertEqual(tail[tail.index("--persistence") + 1], "6.0")
        self.assertEqual(tail[tail.index("--anastomosis-min-separation") + 1], "4")
        self.assertEqual(tail[tail.index("--stenosis-prob") + 1], "-0.0001")
        self.assertEqual(tail[tail.index("--rung-axis") + 1], "[0,0,1]")
        self.assertEqual(json.loads(tail[tail.index("--bed") + 1]), settings["bed"])
        args = parsed("foam", argv)
        for key, value in settings.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(args, key), value)
        # and the bool the other way
        args = parsed("foam", member_argv("foam", 4.0, settings={"cross_connect": True}))
        self.assertIs(args.cross_connect, True)

    def test_an_array_replaces_the_preset_s_whole(self):
        rule = [{"below": None, "field": "plane", "normal": [1, 0, 0], "length": 4.0}]
        args = parsed("aligned", member_argv("aligned", 4.0, settings={"guidance": rule}))
        self.assertEqual(args.guidance, rule)
        self.assertEqual(args.root_offsets, main.FAMILIES["aligned"]["root_offsets"])
        # a bed object is merged instead: the other preset values stay
        args = parsed("foam", member_argv("foam", 4.0, settings={"bed": {"density": 0.3}}))
        released = bed.parse_settings(main.FAMILIES["foam"]["bed"])
        self.assertEqual({key: value for key, value in args.bed.items() if key != "density"},
                         {key: value for key, value in released.items() if key != "density"})
        self.assertEqual(args.bed["density"], 0.3)


class EquivalenceTests(unittest.TestCase):
    """A member with settings is what `vsystem` grows from the member's recorded command line and seed."""

    CASES = (("foam", 4.0, 5, FOAM_SPACING_5),
             ("capillary_bed", 2.52, 2, CAPILLARY_4),
             # every key but the bed: floats, whole numbers, a bool and JSON arrays
             ("mesh", 3.0, 4, {"persistence": 6.5, "anastomosis_fraction": 0.75, "anastomosis_radius": 20,
                               "anastomosis_min_separation": 4, "aneurysm_prob": 0.05, "stenosis_prob": 0.0,
                               "root_offsets": [[-5, 0, 0], [5, 0, 0]],
                               "guidance": [{"below": 2.0, "field": "plane", "normal": [0, 0, 1], "length": 5.0}],
                               "capillary_generations": 2, "capillary_runs": 2,
                               "cross_connect": True, "rung_below": 2.0, "rung_spacing": 15, "rung_radius": 8,
                               "rung_lateral_deg": 45, "rung_axis": [1, 0, 0], "rung_min_separation": 2}))

    def test_a_member_equals_the_command_line_output(self):
        for family, ratio, seed, settings in self.CASES:
            with self.subTest(family=family):
                member = grow_member(family, ratio, seed, settings=settings)
                self.assertEqual(member["family_settings"], normalise_family_settings(family, settings))
                with tempfile.TemporaryDirectory() as out:
                    status, caught = run_main(member["argv"] + ["--count", "1", "--seed", str(seed), "--out", out])
                    self.assertEqual((status, caught), (0, None))
                    stem = next(name[:-4] for name in os.listdir(out) if name.endswith(".npz"))
                    written = main.load_network(os.path.join(out, stem + ".npz"))
                np.testing.assert_array_equal(written["nodes"], member["grown"]["nodes"])
                self.assertEqual(written["program"], member["grown"]["program"])
                self.assertEqual(written["metadata"]["events"], member["grown"]["events"])
                for key, value in member["family_settings"].items():
                    recorded = member["properties"][key] if key in ("aneurysm_prob", "stenosis_prob") \
                        else member["kwargs"][key]
                    self.assertEqual(recorded, value, key)
                    sidecar = written["metadata"]["properties"] if key in ("aneurysm_prob", "stenosis_prob") \
                        else written["metadata"]
                    self.assertEqual(sidecar[key], value, key)
                if family == "capillary_bed":
                    # four generations draw more than the preset's three
                    plain = grow_member(family, ratio, seed)
                    self.assertGreater(len(member["grown"]["program"]), len(plain["grown"]["program"]))
                if family == "mesh":
                    self.assertGreater(member["grown"]["events"]["guided_steps"], 0)
                    self.assertGreater(member["grown"]["events"]["rung_bridges"], 0)

    def test_without_settings_a_member_is_unchanged(self):
        plain = grow_member("tree", 3.0, 5)
        again = grow_member("tree", 3.0, 5, settings=None)
        self.assertNotIn("family_settings", plain)
        self.assertEqual(sorted(plain), ["argv", "d0", "grown", "kwargs", "niter", "properties"])
        np.testing.assert_array_equal(plain["grown"]["nodes"], again["grown"]["nodes"])
        self.assertEqual(plain["argv"], again["argv"])


class CapTests(unittest.TestCase):
    """The lines printed before growth, and the cap and feeder_stop they report, which grow_member applies."""

    def test_the_largest_admitted_ratio_is_where_grow_member_s_cap_starts_to_refuse(self):
        settings = {"bed": {"spacing": 5}}
        cap = library.planned_bed_candidates(parsed("foam", member_argv("foam", 4.0, settings=settings)))
        limit = library.largest_admitted_ratio("foam", cap, settings=settings)
        self.assertGreaterEqual(limit, 4.0)
        self.assertLess(limit, 4.5)
        above = limit * (1.0 + 1e-8)
        with self.assertRaises(bed.BedTooLarge):
            grow_member("foam", above, 1, settings=settings, bed_max_candidates=cap)
        self.assertLessEqual(library.planned_bed_candidates(parsed("foam", member_argv("foam", limit,
                                                                                         settings=settings))), cap)
        # the default cap: every preset foam member up to R 31.33, as the README states
        self.assertAlmostEqual(library.largest_admitted_ratio("foam", library.DEFAULT_BED_MAX_CANDIDATES), 31.33,
                               delta=0.01)
        self.assertIsNone(library.largest_admitted_ratio("tree", 10))

    def test_the_lines_report_the_cap_at_r_hi_and_the_smallest_r_feeder_stop_admits(self):
        options = dict(collision_margin=1.0, mesh_box_c=None, iteration_cap=64, avoid_collisions=True, box_c={})
        family_settings = {"foam": normalise_family_settings("foam", {"bed": {"spacing": 5, "feeder_stop": 3}}),
                           "capillary_bed": normalise_family_settings("capillary_bed", CAPILLARY_4)}
        lines = library.check_family_settings(family_settings, (2.52, 8.0), 2000000, **options)
        planned = library.planned_bed_candidates(parsed("foam", member_argv("foam", 8.0,
                                                                            settings=family_settings["foam"])))
        self.assertEqual(lines, [f"--family-settings foam: the bed plans {planned} candidate edges at R 8 against "
                                 f"--bed-max-candidates 2000000, which admits R up to 20.90; a larger member is "
                                 f"recorded as a failure",
                                 "--family-settings foam: feeder_stop 3 admits R from 3; a smaller member is "
                                 "recorded as a failure"])
        # below feeder_stop the member is refused by grow_network and by the command line alike
        argv = member_argv("foam", 2.9, settings=family_settings["foam"])
        with self.assertRaises(ValueError):
            grow_member("foam", 2.9, 1, settings=family_settings["foam"])
        status, caught = run_main(argv + ["--count", "1", "--seed", "1", "--out", tempfile.mkdtemp()])
        self.assertIn("feeder_stop", str(caught))
        # the smallest R is printed exactly, so that it is never below the member check's bound
        exact = {"foam": normalise_family_settings("foam", {"bed": {"feeder_stop": 2.0000004}})}
        line = library.check_family_settings(exact, (1.5, 4.0), 2000000, **options)[1]
        self.assertIn("feeder_stop 2.0000004 admits R from 2.0000004;", line)
        # no line where feeder_stop is below R_LO
        lines = library.check_family_settings({"foam": family_settings["foam"]}, (3.0, 8.0), 2000000, **options)
        self.assertEqual(len(lines), 1)


class LibraryTests(unittest.TestCase):
    """A tiny library with settings for foam beside the same library without them."""

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        cls.plain_out = tempfile.mkdtemp()
        cls.base = ["--count", "3", "--seed", "2", "--families", "foam", "capillary_bed", "tree",
                    "--ratio-range", "2.52", "3.0", "--workers", "2"]
        cls.argv = ["--out", cls.out] + cls.base + ["--family-settings", "foam", json.dumps(FOAM_SPACING_5)]
        cls.status, cls.stdout, cls.stderr = run_cli(cls.argv)
        cls.plain_status, _, cls.plain_stderr = run_cli(["--out", cls.plain_out] + cls.base)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)
        shutil.rmtree(cls.plain_out, ignore_errors=True)

    def read(self, directory, name):
        with open(os.path.join(directory, name)) as handle:
            return json.load(handle)

    def metadata(self, directory, name):
        return main.load_network(os.path.join(directory, name))["metadata"]

    def test_the_settings_are_recorded_normalised_in_the_manifest_and_the_family_s_archives(self):
        self.assertEqual((self.status, self.plain_status), (0, 0), self.stderr + self.plain_stderr)
        settings = normalise_family_settings("foam", FOAM_SPACING_5)
        growth = self.read(self.out, "manifest.json")["content"]["growth"]
        self.assertEqual(growth["family_settings"], {"foam": settings})
        self.assertEqual(list(growth["family_settings"]["foam"]["bed"]), list(bed.SETTING_KEYS))
        self.assertEqual(normalise_family_settings("foam", growth["family_settings"]["foam"]),
                         growth["family_settings"]["foam"])
        # the rest of the growth block is the library's without settings
        plain = self.read(self.plain_out, "manifest.json")["content"]["growth"]
        self.assertEqual({key: value for key, value in growth.items() if key != "family_settings"}, plain)
        self.assertNotIn("family_settings", plain)
        foam = self.metadata(self.out, "net_00000_foam.npz")
        self.assertEqual(foam["family_settings"], settings)
        self.assertEqual(foam["grow_kwargs"]["bed"], settings["bed"])
        self.assertEqual(foam["argv"], member_argv("foam", foam["ratio"], settings=settings))
        self.assertEqual(foam["family"], "foam")                      # the family keeps its name
        for name in ("net_00001_capillary_bed.npz", "net_00002_tree.npz"):
            self.assertNotIn("family_settings", self.metadata(self.out, name))
        for name in os.listdir(self.plain_out):
            if name.endswith(".npz"):
                self.assertNotIn("family_settings", self.metadata(self.plain_out, name))
        # no index column is added
        for directory in (self.out, self.plain_out):
            with open(os.path.join(directory, "index.csv")) as handle:
                self.assertEqual(handle.readline().strip().split(","), list(library.INDEX_COLUMNS))
        self.assertEqual(library.INDEX_COLUMNS, INDEX_COLUMNS_3_8)

    def test_the_cap_line_is_printed_before_growth(self):
        lines = self.stdout.splitlines()
        self.assertTrue(lines[0].startswith("--family-settings foam: the bed plans "), lines[0])
        self.assertIn("at R 3 against --bed-max-candidates 2000000, which admits R up to 20.90", lines[0])
        self.assertTrue(lines[1].startswith("net_"), lines[1])

    def test_the_other_families_members_are_bitwise_equal_to_a_library_without_settings(self):
        self.assertEqual((self.status, self.plain_status), (0, 0), self.stderr + self.plain_stderr)
        rows = {row["file"]: row for row in self.read(self.out, "index.json")["networks"]}
        plain_rows = {row["file"]: row for row in self.read(self.plain_out, "index.json")["networks"]}
        self.assertEqual(sorted(rows), sorted(plain_rows))
        networks = self.read(self.out, "manifest.json")["content"]["networks"]
        plain_networks = self.read(self.plain_out, "manifest.json")["content"]["networks"]
        unpinned = ("seconds", "peak_rss_mb")
        for name in rows:
            with self.subTest(name=name):
                a = main.load_network(os.path.join(self.out, name))
                b = main.load_network(os.path.join(self.plain_out, name))
                # the plan: seed, ratio, bin and u
                for key in ("seed", "ratio", "bin", "u"):
                    self.assertEqual(rows[name][key], plain_rows[name][key], key)
                if name.endswith("_foam.npz"):
                    self.assertNotEqual(a["metadata"]["nodes_sha256"], b["metadata"]["nodes_sha256"])
                    continue
                np.testing.assert_array_equal(a["nodes"], b["nodes"])
                self.assertEqual(a["program"], b["program"])
                for array in ("edges", "node_kind", "tree"):
                    np.testing.assert_array_equal(a[array], b[array])
                self.assertEqual({key: value for key, value in a["metadata"].items() if key not in unpinned},
                                 {key: value for key, value in b["metadata"].items() if key not in unpinned})
                self.assertEqual({key: value for key, value in rows[name].items() if key not in unpinned},
                                 {key: value for key, value in plain_rows[name].items() if key not in unpinned})
                key = str(rows[name]["id"])
                self.assertEqual(networks[key], plain_networks[key])

    def test_a_rerun_with_the_same_settings_keeps_every_archive(self):
        self.assertEqual(self.status, 0, self.stderr)
        before = self.read(self.out, "manifest.json")
        # the same settings written differently normalise alike
        argv = ["--out", self.out] + self.base + ["--family-settings", "foam", '{"bed": {"spacing": 5.0}}']
        status, stdout, stderr = run_cli(argv)
        self.assertEqual(status, 0, stderr)
        after = self.read(self.out, "manifest.json")
        self.assertEqual(after["content"], before["content"])
        last = after["run"]["runs"][-1]
        self.assertEqual((last["kept"], last["grown"], last["failed"]), (3, 0, 0))
        self.assertNotIn(".npz:", stdout)

    def test_a_rerun_with_different_or_no_settings_is_refused(self):
        self.assertEqual(self.status, 0, self.stderr)
        with open(os.path.join(self.out, "manifest.json")) as handle:
            before = handle.read()
        for given in (["--family-settings", "foam", '{"bed": {"spacing": 6}}'], [],
                      ["--family-settings", "foam", json.dumps(FOAM_SPACING_5),
                       "--family-settings", "capillary_bed", json.dumps(CAPILLARY_4)]):
            with self.subTest(given=given):
                status, _, stderr = run_cli(["--out", self.out] + self.base + given)
                self.assertNotEqual(status, 0)
                self.assertIn("cannot resume, growth was", stderr)
        with open(os.path.join(self.out, "manifest.json")) as handle:
            self.assertEqual(handle.read(), before)

    def test_an_archive_is_checked_against_the_command_line_with_its_settings(self):
        self.assertEqual(self.status, 0, self.stderr)
        rows = {row["file"]: row for row in self.read(self.out, "index.json")["networks"]}
        member = dict(rows["net_00000_foam.npz"], file="net_00000_foam.npz")
        member["id"] = 0
        content = self.read(self.out, "manifest.json")["content"]
        settings = dict(content["growth"], ratio_lo=2.52, ratio_hi=3.0)
        record, reason = library.validate_existing(os.path.join(self.out, member["file"]), member, settings)
        self.assertIsNone(reason)
        without = {key: value for key, value in settings.items() if key != "family_settings"}
        record, reason = library.validate_existing(os.path.join(self.out, member["file"]), member, without)
        self.assertIsNone(record)
        self.assertTrue(reason.startswith("grown from"), reason)


class RefusalTests(unittest.TestCase):
    """Every refusal of --family-settings names the family and comes before anything is written."""

    CASES = (
        (["--families", "foam"], ["foam", '{"bogus": 1}'], "unknown setting"),
        (["--families", "foam"], ["foam", '{"d0": 3}'], "the library fixes it"),
        (["--families", "foam"], ["foam", '{"collision_margin": 2}'], "the library fixes it"),
        (["--families", "tree"], ["tree", '{"avoid_collisions": false}'], "the library fixes it"),
        (["--families", "mesh"], ["mesh", '{"grow_in_volume": false}'], "the library fixes it"),
        (["--families", "foam"], ["foam", '{"tortuosity": "stems"}'], "changes what the family is"),
        (["--families", "mesh"], ["mesh", '{"anastomose": false}'], "changes what the family is"),
        (["--families", "mesh"], ["mesh", '{"anastomose_mode": "any"}'], "changes what the family is"),
        (["--families", "foam"], ["capillary_bed", '{"capillary_generations": 4}'], "not a family of this library"),
        ([], ["foam", '{"persistence": 5}'], "not a family of this library"),
        (["--families", "foam"], ["foam", '{"persistence": 5}', "--family-settings", "foam", '{"persistence": 6}'],
         "given twice"),
        (["--families", "foam"], ["foam", "{persistence: 5}"], "not JSON"),
        (["--families", "foam"], ["foam", "[1, 2]"], "JSON object"),
        (["--families", "foam"], ["foam", "5"], "JSON object"),
        (["--families", "foam"], ["foam", "{}"], "empty"),
        (["--families", "tree"], ["tree", '{"bed": {"spacing": 5}}'], "grows no bed"),
        (["--families", "foam"], ["foam", '{"bed": null}'], "turns the preset's bed off"),
        (["--families", "foam"], ["foam", '{"bed": {"kind": "fibre"}}'], "kind 'fibre' cannot replace"),
        (["--families", "fibre"], ["fibre", '{"bed": {"layout": "pair"}}'], "layout 'pair' cannot replace"),
        (["--families", "foam"], ["foam", '{"bed": {"density": 0.7}}'], "density"),
        (["--families", "foam"], ["foam", '{"capillary_generations": 4}'], "cannot be combined with a capillary fill"),
        (["--families", "foam"], ["foam", '{"persistence": true}'], "finite number"),
        (["--families", "tree"], ["tree", '{"aneurysm_prob": -1e-05}'], "read as an option"),
        # what the command line refuses with the member's other arguments, at either end of the range
        (["--families", "fibre"], ["fibre", '{"bed": {"axis": null}}'], "the command line refuses the member"),
        (["--families", "aligned"], ["aligned", '{"root_offsets": [[-90, 0, 0], [90, 0, 0]]}'],
         "outside the growth box"),
        (["--families", "foam"], ["foam", '{"bed": {"spacing": 1.5}}'], "seeds would touch"),
        # refused at one end of the range only: offsets outside the cube at R_LO, too many feeder sheets at R_HI
        (["--families", "aligned"], ["aligned", '{"root_offsets": [[-25, 0, 0], [25, 0, 0]]}'],
         "refuses the member at R 2.52: --root-offsets"),
        (["--families", "fibre"], ["fibre", '{"bed": {"unit": 0.4}}'], "refuses the member at R 4: --bed"),
    )

    def test_each_refusal_names_the_family_before_anything_is_written(self):
        for families, given, message in self.CASES:
            with self.subTest(given=given), tempfile.TemporaryDirectory() as parent:
                out = os.path.join(parent, "lib")
                status, stdout, stderr = run_cli(["--out", out, "--count", "2", "--seed", "1",
                                                  "--ratio-range", "2.52", "4"] + families
                                                 + ["--family-settings"] + given)
                self.assertNotEqual(status, 0)
                self.assertIn(f"--family-settings {given[0]}:", stderr)
                self.assertIn(message, stderr)
                self.assertFalse(os.path.exists(out))


# --------------------------------------------------------------------------
# Benchmark

def _settings_member_task(task):
    """One member with settings in a fresh process: grown, timed, its peak RSS, and described."""
    from describe import describe
    from tests.test_topology_benchmark import own_peak_rss_mb
    started = time.perf_counter()
    try:
        result = grow_member(task["family"], task["ratio"], task["seed"], settings=task["settings"])
    except bed.BedTooLarge as error:
        return {"refused": str(error), "seconds": time.perf_counter() - started}
    seconds = time.perf_counter() - started
    peak = own_peak_rss_mb()
    grown = result["grown"]
    nodes, edges = grown["nodes"], grown["edges"]
    box = [float(v) for v in grown["growth_box_um"]]
    side = box[0]
    length = float(np.sum(np.linalg.norm(nodes[:3, edges[0]] - nodes[:3, edges[1]], axis=0)))
    started = time.perf_counter()
    report = describe(nodes, edges, metadata={"growth_box_um": box}, d_ref=1, evd_spacing=side / 64)
    tissue = report["tissue_distance"]
    args = main.build_parser(task["family"]).parse_args(result["argv"])
    return {"seconds": seconds, "peak_rss_mb": peak, "describe_seconds": time.perf_counter() - started,
            "points": report["points"], "columns": int(nodes.shape[1]),
            "planned_candidates": library.planned_bed_candidates(args),
            "length_density_d2": length / float(np.prod(box)),
            "tissue_mean_d": tissue["mean_d"], "tissue_p90_d": tissue["p90_d"], "tissue_max_d": tissue["max_d"],
            "tissue_spacing": tissue["spacing"], "capillary_length_share": report["calibre_shares"]["length"]["below_2"],
            "bed_seeds": grown["events"].get("bed_seeds", 0), "rung_bridges": grown["events"].get("rung_bridges", 0),
            "box_side": side, "generations": max(library.drawn_generations(p) for p in grown["programs"])}


def _ratios(name, default):
    text = os.environ.get(name)
    return default if not text else tuple(float(v) for v in text.split(","))


@SLOW
class SettingsBenchmark(unittest.TestCase):
    """
    Members grown through library.grow_member with settings, each in a fresh
    process of its own, one at a time, seed 1: foam at R {4, 8, 16} with bed
    spacing {7.5, 6, 5, 4.5} and density {0.5, 0.6} (7.5 and 0.5 are the
    preset's), and capillary_bed at R {4, 8} with capillary_generations {3,
    4} (3 is the preset's). Per member: growth time, peak RSS (VmHWM),
    points, planned candidate edges, length density over the growth box
    (total edge length / box volume, per d_min^2), the tissue distance of
    describe(nodes, edges, metadata={"growth_box_um": box}, d_ref=1,
    evd_spacing=side / 64) (mean, p90 and max, in d_min), the capillary
    length share (calibre_shares["length"]["below_2"]), and per (spacing,
    density) the largest R the default cap admits. A run that takes longer
    than VSYSTEM_SETTINGS_RUN_LIMIT_S (default 900) is stopped, reported and
    left out. The ratios can be narrowed for a quick check with
    VSYSTEM_SETTINGS_FOAM_RATIOS and VSYSTEM_SETTINGS_CAPILLARY_RATIOS
    ("4,8"). The tables are printed, and written as JSON to the path in
    VSYSTEM_BENCHMARK_JSON when that is set. Run with
        VSYSTEM_SLOW_TESTS=1 nice python -m unittest tests.test_library_settings.SettingsBenchmark
    """

    def test_the_benchmark(self):
        from tests.test_fibre_benchmark import run_fresh
        limit = float(os.environ.get("VSYSTEM_SETTINGS_RUN_LIMIT_S", 900))
        foam_ratios = _ratios("VSYSTEM_SETTINGS_FOAM_RATIOS", (4.0, 8.0, 16.0))
        capillary_ratios = _ratios("VSYSTEM_SETTINGS_CAPILLARY_RATIOS", (4.0, 8.0))
        tasks = []
        for ratio in foam_ratios:
            for spacing in (7.5, 6.0, 5.0, 4.5):
                for density in (0.5, 0.6):
                    tasks.append({"family": "foam", "ratio": ratio, "seed": 1, "label": f"spacing {spacing:g} "
                                  f"density {density:g}", "settings": {"bed": {"spacing": spacing, "density": density}}})
        for ratio in capillary_ratios:
            for generations in (3, 4):
                tasks.append({"family": "capillary_bed", "ratio": ratio, "seed": 1,
                              "label": f"capillary_generations {generations}",
                              "settings": {"capillary_generations": generations}})
        rows = []
        for task in tasks:
            message = run_fresh(_settings_member_task, task, limit)
            state = "ok" if "ok" in message else "timed out" if "timed_out" in message else "error"
            print(f"  [{state} in {message['wall_seconds']:.1f} s] {task['family']} R {task['ratio']:g} "
                  f"{task['label']}", flush=True)
            self.assertNotIn("error", message, message.get("error"))
            rows.append(dict(task, **message))
        limits = {}
        for spacing in (7.5, 6.0, 5.0, 4.5):
            for density in (0.5, 0.6):
                limits[f"{spacing:g} {density:g}"] = library.largest_admitted_ratio(
                    "foam", library.DEFAULT_BED_MAX_CANDIDATES, settings={"bed": {"spacing": spacing,
                                                                                  "density": density}})
        print("\nfamily         R   setting                     seconds  peak MB   points  planned  "
              "L/V d^-2  tissue mean/p90/max d  cap share")
        for row in rows:
            head = f"{row['family']:13s} {row['ratio']:3g}  {row['label']:26s}"
            if "timed_out" in row:
                print(f"{head} stopped after {row['timed_out']:.0f} s and left out")
                continue
            out = row["ok"]
            if "refused" in out:
                print(f"{head} refused: {out['refused']}")
                continue
            planned = "-" if out["planned_candidates"] is None else f"{out['planned_candidates']}"
            print(f"{head} {out['seconds']:8.1f} {out['peak_rss_mb']:8.0f} {out['points']:8d} {planned:>8s}  "
                  f"{out['length_density_d2']:8.4f}  {out['tissue_mean_d']:6.2f} {out['tissue_p90_d']:6.2f} "
                  f"{out['tissue_max_d']:6.2f}   {out['capillary_length_share']:.3f}")
        print("\nlargest R the default cap (2e6) admits, foam, by spacing and density")
        for key, value in limits.items():
            print(f"  spacing {key.split()[0]:>3s} density {key.split()[1]}: {value:.2f}")
        if os.environ.get("VSYSTEM_BENCHMARK_JSON"):
            with open(os.environ["VSYSTEM_BENCHMARK_JSON"], "w") as handle:
                json.dump({"rows": rows, "largest_admitted_ratio": limits}, handle, indent=2, default=str)
        # sanity only: every finished member grew, and a denser foam bed is longer per unit volume
        finished = {(row["family"], row["ratio"], row["label"]): row["ok"] for row in rows
                    if "ok" in row and "refused" not in row["ok"]}
        for key, out in finished.items():
            self.assertGreater(out["points"], 0, key)
        for ratio in foam_ratios:
            sparse = finished.get(("foam", ratio, "spacing 7.5 density 0.5"))
            dense = finished.get(("foam", ratio, "spacing 4.5 density 0.5"))
            if sparse and dense:
                self.assertGreater(dense["length_density_d2"], sparse["length_density_d2"], ratio)


if __name__ == "__main__":
    unittest.main()
