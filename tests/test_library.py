"""
Tests of library.py: the network-library generator and its weights.

Run from the repository root with
    python -m unittest tests.test_library
The command-line runs grow tiny libraries (small ratio ranges) in worker
processes, so this module takes about a minute.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
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
from analyseGrammar import tokenise  # noqa: E402
from library import grow_member, library_weights, plan_library, plan_ratios  # noqa: E402

TINY = ["--ratio-range", "2.52", "3.5"]


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


class PlanningTests(unittest.TestCase):
    def test_rng_tags_do_not_collide_with_main(self):
        self.assertFalse(set(library.RNG_STREAMS.values()) & set(main.RNG_STREAMS.values()))

    def test_family_counts_by_largest_remainder_with_ties_to_the_first_family(self):
        self.assertEqual(library.family_counts(2000, ["tree", "mesh", "tumour"], [1, 1, 1]), [667, 667, 666])
        self.assertEqual(library.family_counts(10, ["tree", "mesh", "tumour"], [1, 1, 1]), [4, 3, 3])
        self.assertEqual(library.family_counts(7, ["a", "b"], [2, 1]), [5, 2])
        self.assertEqual(library.family_counts(0, ["a", "b"], [1, 1]), [0, 0])

    def test_ids_are_contiguous_blocks_in_family_order_with_seeds_from_the_library_seed(self):
        plan = plan_library(7, 10, ["tree", "mesh", "tumour"], [1, 1, 1], (2.52, 25.0))
        self.assertEqual([m["id"] for m in plan], list(range(10)))
        self.assertEqual([m["family"] for m in plan], ["tree"] * 4 + ["mesh"] * 3 + ["tumour"] * 3)
        for member in plan:
            self.assertEqual(member["seed"], int(np.random.SeedSequence([7, member["id"]]).generate_state(1)[0]))
            self.assertEqual(member["file"], f"net_{member['id']:05d}_{member['family']}.npz")

    def test_stratification_covers_the_range_with_one_network_per_bin(self):
        lo, hi = 2.52, 25.0
        for family_index, count in ((0, 12), (1, 7), (2, 1)):
            with self.subTest(count=count):
                ratios = plan_ratios(3, family_index, count, lo, hi)
                self.assertEqual(sorted(r[0] for r in ratios), list(range(count)))      # a permutation of the bins
                for bin_, u, ratio in ratios:
                    self.assertTrue(0.0 <= u < 1.0)
                    edge_lo = lo * (hi / lo) ** (bin_ / count)
                    edge_hi = lo * (hi / lo) ** ((bin_ + 1) / count)
                    self.assertTrue(edge_lo <= ratio < edge_hi)
                    self.assertTrue(lo <= ratio < hi)
                # the law written out
                rng = np.random.default_rng([3, library.RNG_STREAMS["ratio"], family_index])
                u = rng.random(count)
                perm = rng.permutation(count)
                self.assertEqual([r[2] for r in ratios],
                                 [lo * (hi / lo) ** ((perm[k] + u[k]) / count) for k in range(count)])


class GrowthTests(unittest.TestCase):
    def test_growth_is_deterministic(self):
        a = grow_member("tumour", 3.0, 11)
        b = grow_member("tumour", 3.0, 11)
        np.testing.assert_array_equal(a["grown"]["nodes"], b["grown"]["nodes"])
        self.assertEqual(a["grown"]["program"], b["grown"]["program"])
        self.assertEqual(a["grown"]["events"], b["grown"]["events"])

    def test_presets_take_effect(self):
        mesh = grow_member("mesh", 5.0, 5)
        self.assertEqual(len(mesh["grown"]["programs"]), 2)
        self.assertEqual(mesh["kwargs"]["anastomose_mode"], "arteriovenous")
        self.assertTrue(mesh["kwargs"]["grow_in_volume"])
        self.assertGreater(mesh["grown"]["events"]["anastomosis_bridges_arteriovenous"], 0)
        side = library.DEFAULT_MESH_BOX_C * 5.0
        np.testing.assert_allclose(mesh["grown"]["growth_box_um"], [side, side, side], rtol=1e-12)
        tumour = grow_member("tumour", 3.0, 5)
        self.assertEqual(tumour["properties"]["aneurysm_prob"], 0.1)
        self.assertEqual(tumour["properties"]["stenosis_prob"], 0.1)
        self.assertEqual(tumour["kwargs"]["anastomose_mode"], "any")
        self.assertEqual(tumour["kwargs"]["anastomosis_fraction"], 0.8)
        self.assertEqual(tumour["kwargs"]["persistence"], 3.0)
        self.assertEqual(tumour["d0"], 3.0)                                      # the preset's d0 is replaced
        tree = grow_member("tree", 3.0, 5)
        self.assertTrue(tree["kwargs"]["avoid_collisions"])
        self.assertFalse(tree["kwargs"]["anastomose"])
        plain = grow_member("tree", 3.0, 5, avoid_collisions=False)
        self.assertFalse(plain["kwargs"]["avoid_collisions"])
        self.assertEqual(plain["grown"]["program"], tree["grown"]["program"])    # the same grammar draws

    def test_every_drawn_diameter_is_at_least_d_min_but_a_stenosis_middle(self):
        for family, ratio in (("tree", 4.0), ("tumour", 4.0), ("mesh", 4.0)):
            with self.subTest(family=family):
                grown = grow_member(family, ratio, 21)["grown"]
                for program in grown["programs"]:
                    moves = [params for command, params in tokenise(program) if command == "f" and len(params) > 1]
                    self.assertGreater(len(moves), 0)
                    for k, params in enumerate(moves):
                        diameter = params[1]
                        if diameter < 1.0:
                            # the middle move of a stenosis: half the sub-segment's diameter
                            self.assertGreater(k, 0)
                            self.assertAlmostEqual(diameter, 0.5 * moves[k - 1][1], places=12)
                            self.assertGreaterEqual(moves[k - 1][1], 1.0)

    def test_growth_scales_with_the_unit(self):
        for family in ("tree", "tumour", "mesh"):
            with self.subTest(family=family):
                one = grow_member(family, 3.0, 9, d_min=1.0, collision_margin=1.0, mesh_box_c=15.0)
                two = grow_member(family, 3.0, 9, d_min=2.0, collision_margin=2.0, mesh_box_c=15.0)
                self.assertEqual(two["d0"], 2.0 * one["d0"])
                self.assertEqual(one["grown"]["nodes"].shape, two["grown"]["nodes"].shape)
                np.testing.assert_allclose(two["grown"]["nodes"], 2.0 * one["grown"]["nodes"], rtol=1e-12, atol=0.0)
                self.assertEqual(one["grown"]["events"], two["grown"]["events"])

    def test_a_library_network_equals_the_command_line_output(self):
        for family, ratio, seed in (("tree", 3.0, 4), ("mesh", 4.0, 4), ("tumour", 3.0, 4)):
            with self.subTest(family=family):
                member = grow_member(family, ratio, seed)
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = main.main(member["argv"] + ["--count", "1", "--seed", str(seed), "--out", out])
                    self.assertEqual(status, 0)
                    stem = next(n[:-4] for n in os.listdir(out) if n.endswith(".npz"))
                    written = main.load_network(os.path.join(out, stem + ".npz"))
                np.testing.assert_array_equal(written["nodes"], member["grown"]["nodes"])
                self.assertEqual(written["program"], member["grown"]["program"])
                self.assertEqual(written["metadata"]["d0"], ratio)
                self.assertEqual(written["metadata"]["events"], member["grown"]["events"])

    def test_tree_and_tumour_growth_do_not_depend_on_the_volume(self):
        for family in ("tree", "tumour"):
            with self.subTest(family=family):
                member = grow_member(family, 3.0, 6)
                kwargs = dict(member["kwargs"])
                kwargs["tVol"] = (512, 512, 140)
                kwargs["voxel_size"] = 2.5
                properties = member["properties"]
                import random
                random.seed(6)
                np.random.seed(6)
                main.sample_parameters(main.build_parser(family).parse_args(member["argv"]))
                again = main.grow_network(member["niter"], member["d0"], properties, **kwargs)
                np.testing.assert_array_equal(again["nodes"], member["grown"]["nodes"])


class WeightTests(unittest.TestCase):
    def test_weights_reproduce_a_power_law(self):
        lo, hi, count, exponent = 2.52, 25.0, 20000, 3.0
        rows = [{"ratio": r, "ratio_law": "log-uniform"} for _, _, r in plan_ratios(1, 0, count, lo, hi)]
        weights = library_weights(rows, exponent=exponent)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        ratio = np.array([row["ratio"] for row in rows])
        order = np.argsort(ratio)
        empirical = np.cumsum(weights[order])
        # the target: density proportional to R ** -exponent per unit log R, so its
        # cumulative distribution in R is (lo ** -e - R ** -e) / (lo ** -e - hi ** -e)
        target = (lo ** -exponent - ratio[order] ** -exponent) / (lo ** -exponent - hi ** -exponent)
        self.assertLess(float(np.max(np.abs(empirical - target))), 0.01)          # a Kolmogorov distance
        self.assertAlmostEqual(library.effective_sample_size(weights) / count, 0.29, delta=0.02)
        masked = library_weights(rows, exponent=exponent, mask=ratio > 5.0)
        self.assertTrue(np.all(masked[ratio <= 5.0] == 0.0))
        self.assertAlmostEqual(float(masked.sum()), 1.0, places=12)
        with self.assertRaises(ValueError):
            library_weights([{"ratio": 3.0, "ratio_law": "uniform"}])


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        cls.argv = ["--out", cls.out, "--count", "3", "--seed", "5", "--workers", "2"] + TINY
        cls.status, cls.stdout, cls.stderr = run_cli(cls.argv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_a_tiny_library_is_written_with_its_index_and_manifest(self):
        self.assertEqual(self.status, 0, self.stderr)
        names = sorted(os.listdir(self.out))
        self.assertEqual(names, ["index.csv", "index.json", "manifest.json",
                                 "net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        with open(os.path.join(self.out, "index.json")) as handle:
            index = json.load(handle)
        self.assertEqual(index["units"], "d_min")
        self.assertEqual(index["ratio_law"], "log-uniform")
        self.assertEqual([row["id"] for row in index["networks"]], [0, 1, 2])
        for row in index["networks"]:
            for key in ("family", "ratio", "seed", "file", "points", "tips", "cycles", "components", "total_length",
                        "diameter_p50", "diameter_v50", "diameter_min", "diameter_max", "clearance_violations",
                        "events", "seconds", "peak_rss_mb", "nodes_sha256", "program_sha256", "generations"):
                self.assertIn(key, row)
            self.assertGreater(row["points"], 0)
            self.assertGreater(row["seconds"], 0.0)
            self.assertGreater(row["peak_rss_mb"], 0.0)
            self.assertTrue(2.52 <= row["ratio"] < 3.5)
            network = main.load_network(os.path.join(self.out, row["file"]))
            record = network["metadata"]
            self.assertEqual(record["units"], "d_min")
            self.assertEqual(record["family"], row["family"])
            self.assertEqual(record["ratio"], row["ratio"])
            self.assertEqual(record["seed"], row["seed"])
            self.assertEqual(record["ratio_law"], "log-uniform")
            self.assertEqual(record["grow_kwargs"]["d_min"], 1.0)
            self.assertEqual(len(record["root_columns"]), 2 if row["family"] == "mesh" else 1)
            self.assertEqual(record["root_columns"][0], 0)
            self.assertIn("avoid_collisions", record)
            self.assertAlmostEqual(record["min_point_diameter"], float(np.nanmin(network["nodes"][3])))
            self.assertEqual(record["nodes_sha256"], row["nodes_sha256"])
            self.assertEqual(set(record["events"]), set(main.EVENT_KEYS))
        with open(os.path.join(self.out, "index.csv")) as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0].split(","), list(library.INDEX_COLUMNS))
        self.assertEqual(len(lines), 4)
        with open(os.path.join(self.out, "manifest.json")) as handle:
            manifest = json.load(handle)
        content = manifest["content"]
        self.assertEqual(content["family_counts"], {"tree": 1, "mesh": 1, "tumour": 1})
        self.assertEqual(content["ratio_law"], "log-uniform")
        self.assertEqual(content["code_sha256"], library.code_hash())
        self.assertEqual(content["failures"], [])
        self.assertEqual(set(content["networks"]), {"0", "1", "2"})
        self.assertEqual(manifest["content_sha256"], library.content_hash(content))
        for key in ("host", "python", "numpy", "runs", "per_network"):
            self.assertIn(key, manifest["run"])
        self.assertEqual(manifest["run"]["runs"][0]["grown"], 3)

    def test_describe_samples_a_member_s_tissue_over_its_growth_box(self):
        from describe import describe_archive
        self.assertEqual(self.status, 0, self.stderr)
        with open(os.path.join(self.out, "index.json")) as handle:
            rows = json.load(handle)["networks"]
        for row in rows:
            with self.subTest(family=row["family"]):
                path = os.path.join(self.out, row["file"])
                box = main.load_network(path)["metadata"]["growth_box"]
                self.assertEqual(box is None, row["family"] != "mesh")
                tissue = describe_archive(path, evd_spacing=1.0)["tissue_distance"]
                if box is None:
                    self.assertEqual(tissue["domain"]["source"], "bounding_box")
                else:
                    self.assertEqual(tissue["domain"], {"min": [0.0, 0.0, 0.0], "max": box, "source": "growth_box"})

    def test_rerunning_resumes_without_regrowing_and_the_manifest_hash_is_stable(self):
        with open(os.path.join(self.out, "manifest.json")) as handle:
            before = json.load(handle)
        status, stdout, stderr = run_cli(self.argv)
        self.assertEqual(status, 0, stderr)
        with open(os.path.join(self.out, "manifest.json")) as handle:
            after = json.load(handle)
        self.assertEqual(after["content_sha256"], before["content_sha256"])
        self.assertEqual(after["content"], before["content"])
        last = after["run"]["runs"][-1]
        self.assertEqual((last["kept"], last["grown"], last["failed"]), (3, 0, 0))
        self.assertNotIn(".npz:", stdout)                                       # nothing was grown
        # a fresh run of the same library elsewhere gives the same content hash
        with tempfile.TemporaryDirectory() as other:
            argv = ["--out", other] + self.argv[2:]
            status, _, stderr = run_cli(argv)
            self.assertEqual(status, 0, stderr)
            with open(os.path.join(other, "manifest.json")) as handle:
                fresh = json.load(handle)
        self.assertEqual(fresh["content_sha256"], before["content_sha256"])

    def test_a_different_library_or_code_is_refused(self):
        for change in (["--count", "4"], ["--ratio-range", "2.52", "4.0"], ["--no-avoid-collisions"], ["--seed", "6"]):
            with self.subTest(change=change):
                argv = list(self.argv)
                if change[0] in argv:
                    at = argv.index(change[0])
                    width = 3 if change[0] == "--ratio-range" else 2
                    argv[at:at + width] = change
                else:
                    argv += change
                status, _, stderr = run_cli(argv)
                self.assertNotEqual(status, 0)
                self.assertIn("cannot resume", stderr)
        manifest_path = os.path.join(self.out, "manifest.json")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        altered = dict(manifest)
        altered["content"] = dict(manifest["content"], code_sha256="0" * 64)
        with open(manifest_path, "w") as handle:
            json.dump(altered, handle)
        try:
            status, _, stderr = run_cli(self.argv)
            self.assertNotEqual(status, 0)
            self.assertIn("code hash", stderr)
        finally:
            with open(manifest_path, "w") as handle:
                json.dump(manifest, handle)

    def test_a_missing_archive_is_regrown_on_resume_and_a_foreign_one_refused(self):
        path = os.path.join(self.out, "net_00000_tree.npz")
        with open(path, "rb") as handle:
            original = handle.read()
        try:
            os.remove(path)
            status, stdout, stderr = run_cli(self.argv)
            self.assertEqual(status, 0, stderr)
            self.assertIn("net_00000_tree.npz:", stdout)
            with open(path, "rb") as handle:
                regrown = handle.read()
            with np.load(path) as handle:
                np.load(io.BytesIO(original))
                nodes_again = handle["nodes"]
            with np.load(io.BytesIO(original)) as handle:
                np.testing.assert_array_equal(handle["nodes"], nodes_again)
            self.assertNotEqual(regrown, b"")
            # an archive of another network under a planned name is not silently kept
            shutil.copyfile(os.path.join(self.out, "net_00001_mesh.npz"), path)
            status, _, stderr = run_cli(self.argv)
            self.assertNotEqual(status, 0)
            self.assertIn("not the planned network", stderr)
        finally:
            with open(path, "wb") as handle:
                handle.write(original)

    def test_unoffered_and_unknown_families_are_refused_before_growth(self):
        with mock.patch.dict(main.FAMILIES, {"unoffered": None}), tempfile.TemporaryDirectory() as out:
            for families in (["unoffered"], ["tree", "unoffered"], ["capillary"], ["tree", "tree"]):
                with self.subTest(families=families):
                    status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--families"] + families)
                    self.assertNotEqual(status, 0)
                    self.assertIn("twice" if families == ["tree", "tree"] else families[-1], stderr)
                    self.assertEqual(os.listdir(out), [])

    def test_the_console_script_entry_point_runs(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "library.py"), "--help"],
                                capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ratio-range", result.stdout)


class AlignedMemberTests(unittest.TestCase):
    """The aligned family in the library: its box, its frame, its index columns and its scaling."""

    @classmethod
    def setUpClass(cls):
        cls.member = grow_member("aligned", 4.0, 3)

    def test_the_aligned_member_grows_in_its_box_with_its_frame(self):
        grown = self.member["grown"]
        side = library.BOX_C["aligned"] * 4.0
        np.testing.assert_allclose(grown["growth_box_um"], [side, side, side], rtol=1e-12)
        self.assertEqual(self.member["kwargs"]["guidance"], main.FAMILIES["aligned"]["guidance"])
        self.assertEqual(self.member["kwargs"]["root_offsets"], main.FAMILIES["aligned"]["root_offsets"])
        self.assertEqual(grown["frame"]["kind"], "axis")
        self.assertEqual(grown["frame"]["tree_senses"], [1, -1])
        self.assertEqual(len(grown["programs"]), 2)
        self.assertGreater(grown["events"]["guided_steps"], 0)
        self.assertIn("--voxel-size", self.member["argv"])
        self.assertEqual(self.member["argv"][self.member["argv"].index("--voxel-size") + 1], repr(side / 3.0))

    def test_the_index_row_carries_what_describe_measures_against_the_frame(self):
        from describe import describe
        grown = self.member["grown"]
        row = library.describe_member(grown, 1.0, 1.0)
        report = describe(grown["nodes"], grown["edges"], frame=grown["frame"], d_ref=1.0, tree=grown["tree"])
        self.assertEqual(row["frame_kind"], "axis")
        self.assertEqual(row["capillary_order"], report["frame_orientation"]["capillary"]["S"])
        self.assertEqual(row["larger_order"], report["frame_orientation"]["larger"]["S"])
        self.assertEqual(row["capillary_polar_order"], report["polar_order"]["capillary"])
        self.assertEqual(row["capillary_length_share"], report["calibre_shares"]["length"]["below_2"])
        self.assertEqual(row["capillary_volume_share"], report["calibre_shares"]["volume"]["below_2"])
        self.assertEqual(row["capillary_segment_median"], report["segments_by_class"]["capillary"]["median"])
        self.assertEqual(row["transverse_spacing_median"], report["transverse_spacing"]["median"])
        self.assertGreater(row["capillary_polar_order"], 0.0)
        self.assertGreater(row["transverse_spacing_median"], 0.0)
        self.assertEqual(list(row)[-9:], list(library.INDEX_COLUMNS[36:]))
        # the columns 3.5 and 3.6 added, by name and in order, after the 3.4 ones (which the 3.4 pin checks)
        self.assertEqual(list(library.INDEX_COLUMNS[36:]),
                         ["frame_kind", "capillary_order", "larger_order", "capillary_polar_order",
                          "capillary_length_share", "capillary_volume_share", "capillary_segment_median",
                          "transverse_spacing_median", "rungs"])
        self.assertEqual(row["rungs"], 0)
        # a tree has a frame of kind "none": the frame columns are empty, the class columns are not
        plain = library.describe_member(grow_member("tree", 3.0, 3)["grown"], 1.0)
        self.assertEqual(plain["frame_kind"], "none")
        for column in ("capillary_order", "larger_order", "capillary_polar_order", "transverse_spacing_median"):
            self.assertIsNone(plain[column])
        self.assertGreater(plain["capillary_length_share"], 0.0)
        self.assertGreater(plain["capillary_segment_median"], 0.0)

    def test_growth_scales_with_the_unit(self):
        one = self.member["grown"]
        two = grow_member("aligned", 4.0, 3, d_min=2.0, collision_margin=2.0)["grown"]
        self.assertEqual(one["nodes"].shape, two["nodes"].shape)
        # rounding beside a face shows as a large relative error (up to 9e-11 over the seeds
        # below), so the comparison has an absolute floor. Over seeds 1-10 at R 4 the largest
        # difference is 3.8e-15 of twice the box side; the floor is 1e-14
        floor = 1e-14 * max(one["growth_box_um"])
        np.testing.assert_allclose(two["nodes"], 2.0 * one["nodes"], rtol=1e-12, atol=2.0 * floor)
        self.assertEqual(one["events"], two["events"])
        self.assertEqual(two["frame"]["rules"], one["frame"]["rules"])
        np.testing.assert_allclose(two["frame"]["origin"], 2.0 * np.asarray(one["frame"]["origin"]), rtol=1e-12)


class BoxConstantTests(unittest.TestCase):
    def test_the_box_constants_merge_over_the_defaults_and_refuse_a_mesh_given_twice(self):
        self.assertEqual(library.box_constants(), library.BOX_C)
        self.assertEqual(library.box_constants(12.0), dict(library.BOX_C, mesh=12.0))
        self.assertEqual(library.box_constants(None, {"aligned": 20.0, "mesh": 11.0}),
                         dict(library.BOX_C, mesh=11.0, aligned=20.0))
        self.assertEqual(library.BOX_C, {"mesh": 15.0, "aligned": 15.0, "aligned_tight": 20.0, "aligned_bed": 20.0,
                                         "capillary_bed": 15.0})
        with self.assertRaises(ValueError):
            library.box_constants(12.0, {"mesh": 11.0})
        with self.assertRaises(ValueError):
            library.box_constants(None, {"aligned": 0.0})
        # a family that grows free has no box to size
        for family in ("tree", "tumour", "capillary"):
            with self.assertRaises(ValueError):
                library.box_constants(None, {family: 5.0})
        argv = library.member_argv("aligned", 4.0, box_c={"aligned": 10.0})
        self.assertEqual(argv[argv.index("--voxel-size") + 1], repr(10.0 * 4.0 / 3.0))
        argv = library.member_argv("mesh", 4.0, mesh_box_c=12.0)
        self.assertEqual(argv[argv.index("--voxel-size") + 1], repr(12.0 * 4.0 / 3.0))
        self.assertEqual(library.member_argv("tree", 4.0, box_c={"aligned": 10.0}),
                         library.member_argv("tree", 4.0))

    def test_the_command_line_refuses_a_mesh_box_given_twice_and_records_other_boxes(self):
        with tempfile.TemporaryDirectory() as out:
            status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--mesh-box-c", "12",
                                         "--box-c", "mesh", "11"] + TINY)
            self.assertNotEqual(status, 0)
            self.assertIn("twice", stderr)
            status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--box-c", "capillary", "11"] + TINY)
            self.assertNotEqual(status, 0)
            self.assertIn("capillary", stderr)
            status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--box-c", "tree", "5"] + TINY)
            self.assertNotEqual(status, 0)
            self.assertIn("not a box family", stderr)
            status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--box-c", "aligned", "-3"] + TINY)
            self.assertNotEqual(status, 0)
            self.assertEqual(os.listdir(out), [])
        # a default library records no box_c; one listing aligned records aligned's constant only
        args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1"])
        self.assertEqual(library.growth_settings(args),
                         {"collision_margin": 1.0, "mesh_box_c": 15.0, "iteration_cap": 64, "avoid_collisions": True,
                          "d_min": 1.0})
        args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1", "--families", "aligned",
                                                  "mesh", "--box-c", "aligned", "12", "--box-c", "mesh", "9"])
        box_c = {family: float(c) for family, c in args.box_c}
        self.assertEqual(library.growth_settings(args, box_c)["box_c"], {"aligned": 12.0})
        self.assertEqual(library.growth_settings(args, box_c)["mesh_box_c"], 9.0)
        args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1", "--families", "tree",
                                                  "--box-c", "aligned", "12"])
        self.assertNotIn("box_c", library.growth_settings(args, {"aligned": 12.0}))


class AlignedFailureTests(unittest.TestCase):
    def test_a_member_whose_root_offset_leaves_its_box_is_recorded_as_a_failure(self):
        # at c 10 and R below 3 the cube is narrower than the roots' 30 d_min
        # separation, so the command line's refusal of the offset is this
        # member's failure, listed in the manifest, and the run ends
        with self.assertRaises(ValueError):
            library.grow_member("aligned", 2.6, 1, box_c={"aligned": 10.0})
        with tempfile.TemporaryDirectory() as out:
            status, _, stderr = run_cli(["--out", out, "--count", "1", "--seed", "1", "--families", "aligned",
                                         "--box-c", "aligned", "10", "--ratio-range", "2.52", "3.0", "--workers", "1"])
            self.assertEqual(status, 1)
            with open(os.path.join(out, "manifest.json")) as handle:
                failures = json.load(handle)["content"]["failures"]
            self.assertEqual(len(failures), 1)
            self.assertIn("outside the growth box", failures[0]["reason"])


class AlignedLibraryTests(unittest.TestCase):
    """A tiny library of aligned networks: written, described and resumed with its frame."""

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        cls.argv = ["--out", cls.out, "--count", "2", "--seed", "3", "--families", "aligned", "--box-c", "aligned", "12",
                    "--ratio-range", "2.52", "3.0"]
        cls.status, cls.stdout, cls.stderr = run_cli(cls.argv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_the_archives_carry_the_frame_and_describe_reads_d_ref_from_them(self):
        from describe import describe_archive
        self.assertEqual(self.status, 0, self.stderr)
        with open(os.path.join(self.out, "manifest.json")) as handle:
            content = json.load(handle)["content"]
        self.assertEqual(content["growth"]["box_c"], {"aligned": 12.0})
        self.assertEqual(content["library_version"], library.LIBRARY_VERSION)
        with open(os.path.join(self.out, "index.json")) as handle:
            rows = json.load(handle)["networks"]
        for row in rows:
            network = main.load_network(os.path.join(self.out, row["file"]))
            record = network["metadata"]
            self.assertEqual(record["frame"]["kind"], "axis")
            self.assertEqual(record["d_min"], 1.0)
            self.assertEqual(row["frame_kind"], "axis")
            report = describe_archive(os.path.join(self.out, row["file"]))
            self.assertEqual(report["classes"], {"bound": 2.0, "d_ref": 1.0, "d_ref_source": "d_min"})
            self.assertEqual(report["frame"], record["frame"])
            self.assertEqual(row["capillary_order"], report["frame_orientation"]["capillary"]["S"])
            self.assertEqual(row["capillary_polar_order"], report["polar_order"]["capillary"])
        with open(os.path.join(self.out, "index.csv")) as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header, list(library.INDEX_COLUMNS))

    def test_a_resumed_row_is_rebuilt_with_every_new_column(self):
        index_path = os.path.join(self.out, "index.json")
        with open(index_path) as handle:
            index = json.load(handle)
        before = {row["id"]: row for row in index["networks"]}
        try:
            index["networks"] = [row for row in index["networks"] if row["id"] != 0]
            with open(index_path, "w") as handle:
                json.dump(index, handle)
            status, stdout, stderr = run_cli(self.argv)
            self.assertEqual(status, 0, stderr)
            self.assertNotIn(".npz:", stdout)                                    # nothing was grown again
            with open(index_path) as handle:
                after = {row["id"]: row for row in json.load(handle)["networks"]}
            for column in library.INDEX_COLUMNS:
                if column not in ("seconds", "peak_rss_mb"):
                    self.assertEqual(after[0][column], before[0][column], column)
        finally:
            with open(index_path, "w") as handle:
                json.dump({"units": index["units"], "ratio_law": index["ratio_law"],
                           "networks": [before[k] for k in sorted(before)]}, handle)


if __name__ == "__main__":
    unittest.main()
