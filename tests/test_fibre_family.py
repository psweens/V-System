"""
The sheets layout, the fibre family and the library's fibre members (tests
14-17 of 3.8): the feeder sheets, their trees, labels, roles, roots and
programs, the plane rule's hold on each tree, every refusal through
grow_network and the command line, the preset end to end (archive, sidecar,
TIFF, the connectivity check, a library member), the frame record alike in
the return, the sidecar and the metadata, the member's scale with the unit,
the cap on planned candidates and the index columns 3.8 appends.
"""
import contextlib
import copy
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
sys.path.insert(0, ROOT)

import bed  # noqa: E402
import check_connectivity  # noqa: E402
import frames  # noqa: E402
import graph  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from describe import describe  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")
RATIO = 4.0
SIDE = library.BOX_C["fibre"] * RATIO          # the member's cube, 60 d_min at d_min 1
# the preset as the order approves it
FIBRE_PRESET = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "grow_in_volume": True,
                "bed": {"kind": "fibre", "spacing": 5.0, "girth": 8, "min_angle": 60.0, "reach": 2.0,
                        "diameter": 1.0, "persistence": 16.0, "feeder_stop": 2.0, "tip_edges": 2, "density": 0.5,
                        "jitter": 0.25, "step": 0.5, "axis": [1.0, 0.0, 0.0], "knot": 4.0, "sway": 1.0,
                        "link_spacing": 20.0, "link_reach": 1.75, "run": None, "layout": "sheets", "unit": 60.0,
                        "sheet_length": 2.0}}
# five sheets in the member's cube: floor(60 / 12 + 1/2)
FIVE_SHEETS = {"unit": 12.0}

_MEMBERS = {}


def preset_with(**bed_changes):
    preset = copy.deepcopy(main.FAMILIES["fibre"])
    preset["bed"].update(bed_changes)
    return preset


def member(seed=1, ratio=RATIO, **bed_changes):
    """library.grow_member("fibre", ratio, seed) with the preset's bed changed as given; made once."""
    key = (seed, ratio, tuple(sorted((k, json.dumps(v)) for k, v in bed_changes.items())))
    if key not in _MEMBERS:
        with mock.patch.dict(main.FAMILIES, {"fibre": preset_with(**bed_changes)}):
            _MEMBERS[key] = library.grow_member("fibre", ratio, seed)
    return _MEMBERS[key]


def global_states():
    return random.getstate(), np.random.get_state()[1].tobytes()


def drawn_programs(result, n):
    """n consecutive draws of F after the member's own seeding and parameter sampling."""
    args = main.build_parser("fibre").parse_args(result["argv"])
    random.seed(result["kwargs"]["seed"])
    np.random.seed(result["kwargs"]["seed"] % (2 ** 32))
    properties, d0, niter = main.sample_parameters(args)
    stop = result["kwargs"]["bed"]["feeder_stop"] * result["kwargs"]["d_min"]
    return [main.F(niter, d0, stop) for _ in range(n)]


def grow_kwargs(result, **changes):
    """The member's grow_network keywords, ready to call again, changed as given."""
    kwargs = dict(result["kwargs"])
    kwargs["tVol"] = tuple(kwargs["tVol"])
    kwargs["clip_axes"] = tuple(kwargs["clip_axes"])
    kwargs.update(changes)
    return kwargs


def regrow(result, **changes):
    """grow_network on the member's draws again (the globals seeded and the parameters sampled as it did)."""
    args = main.build_parser("fibre").parse_args(result["argv"])
    random.seed(result["kwargs"]["seed"])
    np.random.seed(result["kwargs"]["seed"] % (2 ** 32))
    properties, d0, niter = main.sample_parameters(args)
    return main.grow_network(niter, d0, properties, **grow_kwargs(result, **changes))


def sheet_rms(grown, label, j, sheet):
    """A feeder tree's length-weighted rms distance from its sheet plane x_j = sheet."""
    nodes, tree = grown["nodes"], grown["tree"]
    a, b = nodes[:3, :-1], nodes[:3, 1:]
    same = (tree[:-1] == label) & (tree[1:] == label) & np.isfinite(a[0]) & np.isfinite(b[0])
    length = np.sqrt(np.sum((b - a)[:, same] ** 2, axis=0))
    middle = (a[j, same] + b[j, same]) / 2.0 - sheet
    return math.sqrt(float(np.sum(length * middle * middle) / np.sum(length)))


def run_main(argv):
    """main.main(argv) with its output captured; returns (status, SystemExit or None)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            return main.main(argv), None
        except SystemExit as caught:
            return None, caught


def cli_argv(seed=1, ratio=RATIO):
    return (["--family", "fibre"] + library.member_argv("fibre", ratio)[2:]
            + ["--count", "1", "--seed", str(seed)])


# --------------------------------------------------------------------------
# Test 15: the sheets layout


class SheetsLayoutTests(unittest.TestCase):
    def test_the_sheet_count_is_the_closest_whole_number_of_units_and_at_least_two(self):
        settings = bed.parse_settings(FIBRE_PRESET["bed"])
        for side, n in ((60.0, 2), (89.9, 2), (90.0, 2), (150.0, 3), (210.0, 4), (240.0, 4), (20.0, 2)):
            with self.subTest(side=side):
                self.assertEqual(main._sheet_count(np.array([side, 60.0, 60.0]), 0, settings, 1.0), n)
        tight = dict(settings, unit=1.0)
        self.assertEqual(main._sheet_count(np.array([126.0, 1.0, 1.0]), 0, tight, 1.0), 126)
        with self.assertRaises(main.TooManySheets):
            main._sheet_count(np.array([127.0, 1.0, 1.0]), 0, tight, 1.0)
        self.assertEqual(main.SHEETS_MAX, 126)
        self.assertEqual(126 + (126 >= BRIDGE), np.iinfo(np.int8).max)

    def check_trees(self, result, j):
        grown = result["grown"]
        box = grown["growth_box_um"]
        settings = result["kwargs"]["bed"]
        n = max(2, math.floor(box[j] / (settings["unit"] * result["kwargs"]["d_min"]) + 0.5))
        trees = grown["feeder_trees"]
        self.assertEqual(len(trees), n)
        self.assertEqual(len(grown["programs"]), n)
        self.assertEqual(len(grown["root_positions_um"]), n)
        third = 2 if j == 0 else 0
        for k, entry in enumerate(trees):
            with self.subTest(tree=k):
                self.assertEqual(entry["label"], k + (k >= BRIDGE))
                self.assertEqual(entry["role"], "arterial" if k % 2 == 0 else "venous")
                self.assertEqual(entry["sheet"], (k + 0.5) * box[j] / n)
                self.assertEqual(entry["root"], grown["root_positions_um"][k])
                # arterial roots on the y = 0 face heading +y, venous ones on the opposite face
                self.assertEqual(entry["root"][1], 0.0 if k % 2 == 0 else box[1])
                if k == 0 or grown["events"]["root_relocations"] == 0:
                    self.assertEqual(entry["root"][j], entry["sheet"])
                    self.assertEqual(entry["root"][third], box[third] / 2.0)
                # the tree's columns carry its label, its root first
                columns = np.flatnonzero(grown["tree"] == entry["label"])
                self.assertGreater(columns.size, 1)
                self.assertEqual(grown["nodes"][:3, columns[0]].tolist(), entry["root"])
        labels = set(np.unique(grown["tree"]).tolist())
        self.assertEqual(labels - {-1, BRIDGE}, {entry["label"] for entry in trees})
        # the programs are n consecutive draws of F
        self.assertEqual(grown["programs"], drawn_programs(result, n))
        return grown

    def test_the_trees_labels_roles_roots_and_programs_along_x(self):
        grown = self.check_trees(member(**FIVE_SHEETS), 0)
        self.assertEqual([entry["label"] for entry in grown["feeder_trees"]], [0, 1, 3, 4, 5])

    def test_the_trees_labels_roles_roots_and_programs_along_z(self):
        self.check_trees(member(axis=[0.0, 0.0, 1.0], **FIVE_SHEETS), 2)

    def test_the_preset_s_two_sheets(self):
        grown = self.check_trees(member(), 0)
        self.assertEqual([(t["label"], t["role"], t["sheet"]) for t in grown["feeder_trees"]],
                         [(0, "arterial", 15.0), (1, "venous", 45.0)])

    def test_under_the_isotropic_fit_the_programs_are_still_consecutive_draws(self):
        result = member(**FIVE_SHEETS)
        grown = regrow(result, fit="isotropic", voxel_size=None)
        n = len(grown["feeder_trees"])
        self.assertGreaterEqual(n, 2)
        self.assertEqual(grown["programs"], drawn_programs(result, n))
        box = grown["growth_box_um"]
        self.assertEqual(n, max(2, math.floor(box[0] / 12.0 + 0.5)))

    def check_sheet_rms(self, seed):
        # the plane rule steers headings, not positions, so a tree grown
        # without it can stay as close to its sheet; at R 8 every tree of
        # seeds 1-3 kept closer with the rule in development
        result = member(seed=seed, ratio=8.0)
        guided = result["grown"]
        with mock.patch.object(main.Guidance, "bind", lambda self, heading, partner=None: None):
            free = regrow(result)
        self.assertEqual(free["programs"], guided["programs"])
        for entry in guided["feeder_trees"]:
            with self.subTest(seed=seed, tree=entry["label"]):
                self.assertLess(sheet_rms(guided, entry["label"], 0, entry["sheet"]),
                                sheet_rms(free, entry["label"], 0, entry["sheet"]))

    def test_each_tree_keeps_closer_to_its_sheet_than_without_the_plane_rule(self):
        self.check_sheet_rms(1)

    @SLOW
    def test_each_tree_keeps_closer_to_its_sheet_on_seeds_two_and_three(self):
        for seed in (2, 3):
            self.check_sheet_rms(seed)

    REFUSALS = (
        ("root offsets", {"root_offsets": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, {}, "root_offsets"),
        ("a null axis", {}, {"axis": None}, "axis"),
        ("the axis y", {}, {"axis": [0.0, 1.0, 0.0]}, "axis"),
        ("an oblique axis", {}, {"axis": [1.0, 1.0, 0.0]}, "axis"),
        ("the growth direction x", {"direction": (1.0, 0.0, 0.0), "perpendicular": (0.0, 0.0, 1.0)}, {},
         "growth direction"),
        ("127 sheets and more", {}, {"unit": 0.4}, "126"),
        ("no walk", {"tortuosity": "stems"}, {}, "walk"),
    )

    def test_every_refusal_fires_through_grow_network_before_any_draw(self):
        result = member()
        for name, changes, bed_changes, word in self.REFUSALS:
            with self.subTest(refusal=name):
                kwargs = grow_kwargs(result, **changes)
                kwargs["bed"] = dict(kwargs["bed"], **bed_changes)
                random.seed(5)
                np.random.seed(5)
                before = global_states()
                with self.assertRaises(ValueError) as caught:
                    main.grow_network(result["niter"], result["d0"], result["properties"], **kwargs)
                self.assertIn(word, str(caught.exception))
                self.assertEqual(global_states(), before)

    def test_every_refusal_but_the_growth_direction_fires_on_the_command_line(self):
        # the command line grows along main's DEFAULT_DIRECTION, +y, so its growth direction is never refused
        for name, changes, bed_changes, word in self.REFUSALS:
            if "direction" in changes:
                continue
            with self.subTest(refusal=name):
                argv = cli_argv()
                if "root_offsets" in changes:
                    argv += ["--root-offsets", json.dumps(changes["root_offsets"])]
                if "tortuosity" in changes:
                    argv += ["--tortuosity", changes["tortuosity"]]
                if bed_changes:
                    argv += ["--bed", json.dumps(dict(FIBRE_PRESET["bed"], **bed_changes))]
                with tempfile.TemporaryDirectory() as out:
                    status, caught = run_main(argv + ["--out", os.path.join(out, "run")])
                    self.assertIsNone(status)
                    self.assertIsNotNone(caught)
                    self.assertIn("--bed", str(caught.code))
                    self.assertIn(word.replace("root_offsets", "--root-offsets"), str(caught.code))
                    self.assertFalse(os.path.exists(os.path.join(out, "run")))


# --------------------------------------------------------------------------
# Test 16: the family


class FamilyTests(unittest.TestCase):
    def test_the_preset_is_as_approved(self):
        self.assertEqual(main.FAMILIES["fibre"], FIBRE_PRESET)
        self.assertEqual(list(main.FAMILIES["fibre"]["bed"]), list(bed.FIBRE_SETTING_KEYS))
        self.assertEqual(bed.parse_settings(main.FAMILIES["fibre"]["bed"]), FIBRE_PRESET["bed"])
        self.assertEqual(json.loads(json.dumps(main.FAMILIES["fibre"])), FIBRE_PRESET)

    def test_it_runs_end_to_end_alike_on_the_command_line_and_as_a_library_member(self):
        argv = cli_argv()
        with tempfile.TemporaryDirectory() as out:
            status, caught = run_main(argv + ["--out", out])
            self.assertIsNone(caught)
            self.assertEqual(status, 0)
            stem = "Lnet_i64_s1"
            self.assertEqual(sorted(os.listdir(out)), sorted(stem + ext for ext in (".json", ".npz", ".tiff")))
            with open(os.path.join(out, stem + ".json")) as handle:
                sidecar = json.load(handle)
            archive = main.load_network(os.path.join(out, stem + ".npz"))
        grown = member()["grown"]
        self.assertEqual(archive["nodes"].tobytes(), grown["nodes"].tobytes())
        self.assertEqual(grown["growth_box_um"], [SIDE] * 3)
        self.assertEqual(sidecar["trees"], len(grown["feeder_trees"]))
        self.assertEqual(sidecar["feeder_trees"], json.loads(json.dumps(grown["feeder_trees"])))
        self.assertEqual(sidecar["bed"], FIBRE_PRESET["bed"])
        self.assertEqual(sidecar["events"], grown["events"])
        # test 14: the record alike in the return, the sidecar and the metadata
        returned = []
        real = library.grow_member

        def spy(*args, **kwargs):
            returned.append(real(*args, **kwargs))
            return returned[-1]

        task = {"member": {"id": 0, "family": "fibre", "family_index": 0, "position": 0, "seed": 1,
                           "ratio": RATIO, "bin": 0, "u": 0.0, "file": "net_00000_fibre.npz"},
                "settings": {"d_min": 1.0, "collision_margin": 1.0, "mesh_box_c": 15.0, "iteration_cap": 64,
                             "avoid_collisions": True, "ratio_lo": RATIO, "ratio_hi": RATIO}}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(library, "grow_member", side_effect=spy):
            outcome = library.grow_and_write(dict(task, directory=directory))
            self.assertNotIn("failure", outcome)
            metadata = main.load_network(os.path.join(directory, "net_00000_fibre.npz"))["metadata"]
        frame = returned[0]["grown"]["frame"]
        self.assertEqual(frame, grown["frame"])
        self.assertEqual((frame["kind"], frame["axis"], frame["sense"]), ("axis", [1.0, 0.0, 0.0], "nematic"))
        self.assertEqual(sidecar["frame"], json.loads(json.dumps(frame)))
        self.assertEqual(metadata["frame"], json.loads(json.dumps(frame)))
        self.assertEqual(metadata["feeder_trees"], sidecar["feeder_trees"])
        # frames.align and transform_frame take the record
        Q = frames.align(frame, axis=[0.0, 0.0, 1.0])
        moved = frames.transform_frame(frame, Q)
        self.assertAlmostEqual(abs(moved["axis"][2]), 1.0, delta=1e-12)
        # the connectivity check runs it; it passes when the network is one piece
        with contextlib.redirect_stdout(io.StringIO()):
            failed = check_connectivity.generate_and_check(1, argv)
        pieces = graph.betti(grown["edges"], graph.canonical_columns(grown["nodes"]))["components"]
        if pieces == 1:
            self.assertEqual(failed, 0)
        self.assertIn(failed, (0, 1))

    def test_it_needs_d_min_and_bed_null_turns_its_bed_off(self):
        argv = cli_argv()
        at = argv.index("--d-min")
        with tempfile.TemporaryDirectory() as out:
            status, caught = run_main(argv[:at] + argv[at + 2:] + ["--out", os.path.join(out, "run")])
            self.assertIsNotNone(caught)
            self.assertIn("--d-min", str(caught.code))
            self.assertFalse(os.path.exists(os.path.join(out, "run")))
        with tempfile.TemporaryDirectory() as out:
            status, caught = run_main(argv + ["--bed", "null", "--iterations", "4", "4", "--out", out])
            self.assertIsNone(caught)
            self.assertEqual(status, 0)
            with open(os.path.join(out, "Lnet_i4_s1.json")) as handle:
                sidecar = json.load(handle)
        self.assertIsNone(sidecar["bed"])
        self.assertEqual(sidecar["trees"], 1)
        self.assertNotIn("feeder_trees", sidecar)
        self.assertNotIn("bed", sidecar["rng_streams"])


# --------------------------------------------------------------------------
# Test 17: the library


class LibraryTests(unittest.TestCase):
    def test_a_fibre_member_scales_with_the_unit(self):
        reference = member()["grown"]
        twice = library.grow_member("fibre", RATIO, 1, d_min=2.0, collision_margin=2.0)["grown"]
        self.assertEqual(twice["nodes"].shape, reference["nodes"].shape)
        floor = 1e-14 * max(reference["growth_box_um"])
        np.testing.assert_allclose(twice["nodes"], 2.0 * reference["nodes"], rtol=1e-12, atol=2.0 * floor)
        self.assertEqual(twice["events"], reference["events"])

    def test_the_cap_is_the_closed_form_and_a_member_over_it_is_refused_before_any_draw(self):
        settings = bed.parse_settings(FIBRE_PRESET["bed"])
        x = SIDE - settings["diameter"]
        n = math.floor(settings["density"] * x * x / settings["spacing"] ** 2)
        intervals = math.ceil(x / settings["knot"])
        planned = n * (intervals + 1) + math.ceil(n * intervals * settings["knot"] / (2.0 * settings["link_spacing"]))
        self.assertEqual(bed.planned_candidates([SIDE] * 3, settings, 1.0), planned)
        # the preset's axis is given, so the growth direction the library passes changes nothing
        self.assertEqual(bed.planned_candidates([SIDE] * 3, dict(settings, axis=None), 1.0, axis=[0.0, 1.0, 0.0]),
                         bed.planned_candidates([SIDE] * 3, dict(settings, axis=[0.0, 1.0, 0.0]), 1.0))
        random.seed(5)
        np.random.seed(5)
        before = global_states()
        with self.assertRaises(bed.BedTooLarge):
            library.grow_member("fibre", RATIO, 1, bed_max_candidates=planned - 1)
        self.assertEqual(global_states(), before)
        # an axis left null plans along the growth direction, y, rather than being refused
        with mock.patch.dict(main.FAMILIES, {"fibre": preset_with(axis=None, layout="pair")}):
            with self.assertRaises(bed.BedTooLarge) as caught:
                library.grow_member("fibre", RATIO, 1, bed_max_candidates=1)
        self.assertIn(str(bed.planned_candidates([SIDE] * 3, dict(settings, axis=None), 1.0,
                                                 axis=main.DEFAULT_DIRECTION)), str(caught.exception))

    def test_the_new_index_columns_are_describe_s_over_the_growth_box(self):
        grown = member()["grown"]
        row = library.describe_member(grown, 1.0, 1.0)
        report = describe(grown["nodes"], grown["edges"], metadata={"growth_box_um": grown["growth_box_um"]},
                          margin=1.0, tree=grown["tree"], frame=grown["frame"], d_ref=1.0)
        self.assertEqual(report["transverse_spacing"]["area_source"], "growth_box_um")
        self.assertEqual(row["capillary_crossing_density"], report["transverse_spacing"]["crossing_density_d"])
        self.assertEqual(row["capillary_crossing_ratio"], report["frame_orientation"]["capillary"]["crossing_ratio"])
        self.assertEqual(row["cross_connections_per_length"], report["cross_connections"]["per_length_d"])
        for column in ("capillary_crossing_density", "capillary_crossing_ratio", "cross_connections_per_length"):
            self.assertGreater(row[column], 0.0)
        self.assertEqual(list(row)[-3:], list(library.INDEX_COLUMNS[-3:]))
        # a tree has no frame axis: the three columns are empty
        plain = library.describe_member(library.grow_member("tree", 3.0, 3)["grown"], 1.0)
        for column in ("capillary_crossing_density", "capillary_crossing_ratio", "cross_connections_per_length"):
            self.assertIsNone(plain[column])


if __name__ == "__main__":
    unittest.main()
