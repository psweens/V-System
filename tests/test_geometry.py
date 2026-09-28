"""
Tests of the geometry stages: the explicit graph in the archive, smooth
tortuosity, collision avoidance, anastomosis, the family presets and the
descriptor tool, together with the guarantees they must not break: the default
command line reproduces the reference centrelines bit for bit, the archive and
rasteriser contracts are unchanged, and computeVoxel imports under Python 3.9.

Run from the repository root with
    python -m unittest tests.test_geometry
The default-volume TIFF check is slow and runs only with VSYSTEM_SLOW_TESTS=1.
"""
import contextlib
import hashlib
import inspect
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import graph  # noqa: E402
import libGenerator as lg  # noqa: E402
from analyseGrammar import branching_turtle_to_coords, tokenise  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from collisions import CollisionAvoider  # noqa: E402
from computeVoxel import process_network  # noqa: E402
from describe import describe  # noqa: E402
from main import (FAMILIES, build_parser, generate_network, grow_network, load_network,  # noqa: E402
                  main, save_network)
from spatial import GridIndex, have_scipy  # noqa: E402
from tortuosity import bridge_path  # noqa: E402
from utils import interpolate_segments  # noqa: E402
from vSystem import F  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
QUIET = {"aneurysm_prob": 0.0, "stenosis_prob": 0.0}
SMALL = (48, 48, 24)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def grammar_tip_count(program):
    """
    Tips of a tree grammar counted from its string alone: a stem ends in a tip
    when neither of the two brackets that follow it draws anything, and the
    root is a tip. A stem with one drawn and one terminated daughter continues
    as a vessel and is not a tip.
    """
    stack = [[]]                          # per open bracket: emptiness of its child brackets so far
    leaves = 0
    for command, _ in tokenise(program):
        if command == "{":
            for level in stack:
                level.append("stem")
        elif command == "[":
            stack.append([])
        elif command == "]":
            content = stack.pop()
            stack[-1].append("empty" if "stem" not in content else "drawn")
    # each stem is followed by two brackets; a leaf stem by two empty ones
    for level in stack:
        pass
    tokens = [c for c, _ in tokenise(program)]
    depth_children = {}
    depth = 0
    open_children = []
    for c in tokens:
        if c == "[":
            depth += 1
            open_children.append(False)
        elif c == "{":
            for i in range(len(open_children)):
                open_children[i] = True
            depth_children.setdefault(depth, []).append([])
        elif c == "]":
            drew = open_children.pop()
            depth -= 1
            if depth_children.get(depth):
                depth_children[depth][-1].append(drew)
    for level in depth_children.values():
        for stem in level:
            if len(stem) >= 2 and not any(stem[:2]):
                leaves += 1
    return leaves + 1


def vertex_kinds(nodes, node_kind):
    canonical = graph.canonical_columns(nodes)
    vertices = np.unique(canonical[canonical >= 0])
    return node_kind[vertices]


def grow(seed, niter=6, d0=20.0, properties=None, **options):
    seed_all(seed)
    merged = dict(PROPERTIES, **QUIET)
    if properties:
        merged.update(properties)
    return grow_network(niter, d0, merged, SMALL, seed=seed, **options)


def connected_count(volume, connectivity=6):
    """(voxels set, voxels reachable from the first one) under 6- or 26-connectivity."""
    set_voxels = np.argwhere(volume)
    if len(set_voxels) == 0:
        return 0, 0
    shape = volume.shape
    neighbourhood = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)
                     and (connectivity == 26 or abs(dx) + abs(dy) + abs(dz) == 1)]
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


class ReferenceFixtureTests(unittest.TestCase):
    """
    The default command line must reproduce the centrelines written before the
    geometry stages existed, byte for byte in `nodes` and `program`.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(FIXTURES, "reference_hashes.json")) as handle:
            cls.reference = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}

    def test_default_command_line_reproduces_the_reference_centrelines(self):
        self.assertGreaterEqual(len(self.reference), 3)
        for name, record in self.reference.items():
            with self.subTest(fixture=name):
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stdout(io.StringIO()):
                        status = main(["--count", "1", "--seed", str(record["seed"]),
                                       "--volume", "64", "64", "32", "--out", out])
                    self.assertEqual(status, 0)
                    written = load_network(os.path.join(out, name))
                self.assertEqual(sha256(written["nodes"].tobytes()), record["nodes_sha256"])
                self.assertEqual(sha256(written["program"].encode()), record["program_sha256"])
                self.assertEqual(list(written["nodes"].shape), record["nodes_shape"])
                for key, value in record["metadata"].items():
                    self.assertEqual(written["metadata"][key], value, key)   # old keys, old values
                if record.get("fixture_stored"):
                    stored = load_network(os.path.join(FIXTURES, name))
                    np.testing.assert_array_equal(stored["nodes"], written["nodes"])
                    self.assertEqual(stored["program"], written["program"])

    @unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to render the default volume")
    def test_default_volume_tiff_is_byte_identical(self):
        for name, record in self.reference.items():
            if "default_volume_tiff_sha256" not in record:
                continue
            with self.subTest(fixture=name):
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(["--count", "1", "--seed", str(record["seed"]), "--out", out])
                    with open(os.path.join(out, record["default_volume_tiff_name"]), "rb") as handle:
                        self.assertEqual(sha256(handle.read()), record["default_volume_tiff_sha256"])

    def test_archives_written_before_the_graph_upgrade_on_load(self):
        for name in os.listdir(FIXTURES):
            if not name.endswith(".npz"):
                continue
            with self.subTest(fixture=name):
                with np.load(os.path.join(FIXTURES, name)) as handle:
                    self.assertNotIn("edges", handle.files)
                network = load_network(os.path.join(FIXTURES, name))
                self.assertIsNotNone(network["edges"])
                self.assertEqual(network["edges"].shape[0], 2)
                self.assertEqual(network["node_kind"].shape, (network["nodes"].shape[1],))
                betti = graph.betti(network["edges"], graph.canonical_columns(network["nodes"]))
                self.assertEqual(betti["components"], 1)
                self.assertEqual(betti["cycles"], 0)
                bare = load_network(os.path.join(FIXTURES, name), upgrade=False)
                self.assertIsNone(bare["edges"])

    def test_the_default_path_draws_no_random_numbers_after_the_grammar(self):
        # interpretation, interpolation and the graph must leave both generators untouched
        seed_all(4)
        lg.setProperties(PROPERTIES)
        program = F(5, 20.0)
        state = (random.getstate(), np.random.get_state()[1].copy())
        nodes = interpolate_segments(branching_turtle_to_coords(program, 20.0))
        graph.build(nodes)
        self.assertEqual(random.getstate(), state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])


class ContractTests(unittest.TestCase):
    def test_process_network_signature_is_unchanged(self):
        parameters = inspect.signature(process_network).parameters
        self.assertEqual(list(parameters), ["data", "tVol", "fit", "voxel_size", "margin", "clip_axes", "connect"])
        self.assertEqual(parameters["connect"].default, True)

    def test_generate_network_keeps_its_return_value_and_accepts_the_options(self):
        seed_all(2)
        volume, program, nodes = generate_network(5, 20.0, PROPERTIES, SMALL)
        seed_all(2)
        shaped = generate_network(5, 20.0, PROPERTIES, SMALL, tortuosity="walk", persistence=6.0,
                                  avoid_collisions=True, anastomose=True, seed=2)
        self.assertEqual(volume.dtype, np.uint8)
        self.assertEqual(shaped[1], program)                                  # same grammar
        self.assertEqual(shaped[0].shape, volume.shape)
        self.assertGreater(int(shaped[0].sum()), 0)
        self.assertEqual(shaped[2].shape[0], 4)

    def test_nodes_layout_and_units_are_unchanged_for_a_shaped_network(self):
        grown = grow(3, tortuosity="walk", persistence=8.0, avoid_collisions=True, anastomose=True)
        nodes = grown["nodes"]
        self.assertEqual(nodes.shape[0], 4)
        self.assertEqual(nodes.dtype, np.float64)
        separators = np.isnan(nodes[0])
        self.assertTrue(np.all(np.isnan(nodes[:, separators])))          # all-NaN columns
        self.assertFalse(np.any(np.isnan(nodes[:, ~separators])))
        self.assertTrue(np.all(nodes[3, ~separators] > 0))

    def test_metadata_records_every_geometry_flag_and_every_counter(self):
        args = ["--count", "1", "--seed", "3", "--volume", "48", "48", "24", "--iterations", "5", "5",
                "--tortuosity", "walk", "--persistence", "7.5", "--avoid-collisions",
                "--collision-margin", "0.5", "--anastomose", "--anastomosis-fraction", "0.4"]
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--out", out]), 0)
            stem = next(n[:-5] for n in os.listdir(out) if n.endswith(".json"))
            with open(os.path.join(out, stem + ".json")) as handle:
                record = json.load(handle)
            network = load_network(os.path.join(out, stem + ".npz"))
        self.assertEqual(record["units"], "um")
        self.assertEqual(record["tortuosity"], "walk")
        self.assertEqual(record["persistence"], 7.5)
        self.assertTrue(record["avoid_collisions"])
        self.assertEqual(record["collision_margin"], 0.5)
        self.assertTrue(record["anastomose"])
        self.assertEqual(record["anastomosis_fraction"], 0.4)
        self.assertEqual(record["anastomose_mode"], "any")
        self.assertEqual(record["family"], "tree")
        self.assertEqual(record["archive_version"], 2)
        self.assertEqual(record["rng_streams"]["walk"], [3, 1])
        for key in ("bound_terminations", "collision_redraws", "collision_terminations",
                    "anastomosis_tips", "anastomosis_bridges", "anastomosis_no_partner"):
            self.assertIn(key, record["events"])
        self.assertEqual(network["metadata"], record)
        self.assertIsNotNone(network["tree"])
        self.assertEqual(set(np.unique(network["tree"]).tolist()) - {-1}, {0, BRIDGE} if record["bridges"] else {0})

    @unittest.skipUnless(shutil.which("python3.9") or os.environ.get("VSYSTEM_PYTHON39"),
                         "no Python 3.9 interpreter found (set VSYSTEM_PYTHON39)")
    def test_computeVoxel_and_the_archive_reader_import_under_python39(self):
        interpreter = os.environ.get("VSYSTEM_PYTHON39") or shutil.which("python3.9")
        probe = subprocess.run([interpreter, "-c", "import numpy"], capture_output=True, text=True)
        if probe.returncode:
            self.skipTest(f"{interpreter} has no numpy; point VSYSTEM_PYTHON39 at a 3.9 with numpy")
        code = "import computeVoxel, graph, spatial, describe; print('ok')"
        result = subprocess.run([interpreter, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_no_module_uses_syntax_newer_than_python39(self):
        # the whole import chain is compiled with the 3.9 grammar
        import ast
        for name in os.listdir(ROOT):
            if name.endswith(".py"):
                with self.subTest(module=name):
                    with open(os.path.join(ROOT, name)) as handle:
                        ast.parse(handle.read(), feature_version=(3, 9))


class GraphArchiveTests(unittest.TestCase):
    def test_edges_round_trip_through_the_archive(self):
        grown = grow(5)
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "net.npz")
            save_network(path, grown["nodes"], edges=grown["edges"], node_kind=grown["node_kind"],
                         tree=grown["tree"])
            network = load_network(path)
        np.testing.assert_array_equal(network["edges"], grown["edges"])
        np.testing.assert_array_equal(network["node_kind"], grown["node_kind"])
        np.testing.assert_array_equal(network["tree"], grown["tree"])
        np.testing.assert_array_equal(graph.edges_from_nodes(network["nodes"]), grown["edges"])

    def test_node_kind_tip_count_matches_the_grammar(self):
        for seed, niter, d_min in ((1, 4, None), (2, 6, None), (3, 7, 6.0), (4, 5, None)):
            with self.subTest(seed=seed, niter=niter):
                grown = grow(seed, niter=niter, d_min=d_min)
                kinds = vertex_kinds(grown["nodes"], grown["node_kind"])
                self.assertEqual(int(np.sum(kinds == graph.TIP)), grammar_tip_count(grown["program"]))
                if d_min is None:
                    self.assertEqual(int(np.sum(kinds == graph.TIP)), 2 ** (niter - 1) + 1)
                self.assertEqual(int(np.sum(kinds == graph.JUNCTION)), int(np.sum(kinds == graph.TIP)) - 2)
                canonical = graph.canonical_columns(grown["nodes"])
                betti = graph.betti(grown["edges"], canonical)
                self.assertEqual((betti["components"], betti["cycles"]), (1, 0))

    def test_a_malformed_edges_array_is_refused(self):
        grown = grow(2, niter=3)
        with tempfile.TemporaryDirectory() as out:
            for bad in (np.zeros((3, 4), dtype=int), np.array([[0], [grown["nodes"].shape[1]]])):
                with self.assertRaises(ValueError):
                    save_network(os.path.join(out, "bad.npz"), grown["nodes"], edges=bad)


class TortuosityTests(unittest.TestCase):
    def test_the_walk_keeps_the_grammar_arc_length_and_branching(self):
        for seed in (1, 2):
            with self.subTest(seed=seed):
                plain = grow(seed)
                walked = grow(seed, tortuosity="walk", persistence=10.0)
                self.assertEqual(plain["program"], walked["program"])           # same grammar draws
                lengths = []
                for grown in (plain, walked):
                    nodes = grown["nodes"]
                    finite = ~np.isnan(nodes[0])
                    steps = np.linalg.norm(np.diff(nodes[:3], axis=1), axis=0)
                    lengths.append(float(steps[finite[:-1] & finite[1:]].sum()))
                grammar = sum(p[0] for c, p in tokenise(plain["program"]) if c == "f")
                self.assertAlmostEqual(lengths[1], grammar, delta=1e-6 * grammar)
                self.assertEqual(int(np.sum(vertex_kinds(walked["nodes"], walked["node_kind"]) == graph.TIP)),
                                 int(np.sum(vertex_kinds(plain["nodes"], plain["node_kind"]) == graph.TIP)))
                self.assertFalse(np.array_equal(plain["nodes"].shape, walked["nodes"].shape)
                                 and np.array_equal(plain["nodes"], walked["nodes"], equal_nan=True))

    def test_the_walk_is_reproducible_from_the_seed(self):
        a = grow(5, tortuosity="walk", persistence=6.0)
        b = grow(5, tortuosity="walk", persistence=6.0)
        np.testing.assert_array_equal(a["nodes"], b["nodes"])
        c = grow(6, tortuosity="walk", persistence=6.0)
        self.assertFalse(a["nodes"].shape == c["nodes"].shape and np.array_equal(a["nodes"], c["nodes"], equal_nan=True))

    def test_arc_chord_rises_as_persistence_falls(self):
        ratios = []
        for persistence in (50.0, 20.0, 10.0, 5.0, 2.0):
            grown = grow(7, niter=7, tortuosity="walk", persistence=persistence)
            ratios.append(describe(grown["nodes"], grown["edges"])["arc_chord"]["mean"])
        self.assertTrue(all(b > a for a, b in zip(ratios, ratios[1:])), ratios)
        self.assertGreater(ratios[-1], 1.05)

    def test_the_walk_needs_a_persistence_and_a_seed(self):
        with self.assertRaises(ValueError):
            grow(1, tortuosity="walk")
        with self.assertRaises(ValueError):
            seed_all(1)
            grow_network(4, 20.0, PROPERTIES, SMALL, tortuosity="walk", persistence=5.0)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit) as caught:
                main(["--count", "1", "--seed", "1", "--tortuosity", "walk", "--out", out])
            self.assertIn("--persistence", str(caught.exception))

    def test_a_walk_stays_inside_the_growth_box(self):
        grown = grow(3, niter=8, tortuosity="walk", persistence=5.0, grow_in_volume=True)
        box = np.array(grown["growth_box_um"])
        finite = ~np.isnan(grown["nodes"][0])
        points = grown["nodes"][:3, finite]
        self.assertTrue(np.all(points >= -1e-9))
        self.assertTrue(np.all(points <= box[:, None] + 1e-9))
        self.assertGreater(grown["events"]["walk_bound_redraws"] + grown["events"]["bound_terminations"], 0)

    def test_a_bridge_starts_and_ends_exactly_where_asked(self):
        rng = np.random.default_rng(0)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([30.0, 10.0, -5.0])
        for kwargs in ({}, {"rng": rng, "persistence": 5.0, "diameter": 4.0}):
            path = bridge_path(start, end, 1.0, start_tangent=(0.0, 1.0, 0.0), **kwargs)
            np.testing.assert_array_equal(path[0], start)
            np.testing.assert_array_equal(path[-1], end)
            self.assertGreater(len(path), 20)
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            self.assertTrue(np.all(steps > 0))


class CollisionTests(unittest.TestCase):
    def test_avoidance_that_shortens_nothing_leaves_the_geometry_identical(self):
        found = False
        for seed in range(1, 12):
            plain = grow(seed)
            avoided = grow(seed, avoid_collisions=True, collision_margin=0.0)
            if avoided["events"]["collision_truncated_stems"] == 0:
                np.testing.assert_array_equal(plain["nodes"], avoided["nodes"])
                found = True
            else:
                self.assertLess(avoided["nodes"].shape[1], plain["nodes"].shape[1])
        self.assertTrue(found, "no seed without a collision among the first eleven")

    def test_avoidance_reproduces_the_default_interpretation_of_pinned_stems(self):
        # branches opened inside a stem, bare moves and nested stems: the shaped
        # interpreter must yield exactly what the default path yields once
        # utils.interpolate_segments has smoothed it, when nothing collides
        for program in ("{f(1,0.4)[+(90)f(1,0.4)]f(1,0.4)}",
                        "{f(1,0.4)f(1,0.4)[+(90){f(1,0.4)f(1,0.4)}]f(1,0.4)f(1,0.4)}",
                        "{f(1,0.4)[+(90)f(1,0.4)}]f(1,0.4)f(1,0.4)}",
                        "f(2,0.4)f(2,0.4){f(1,0.4)f(1,0.4)f(1,0.4)}[+(30)f(1,0.4)]f(1,0.4)",
                        "{f(1,0.4)f(1,0.4)f(1,0.4)[+(90){f(1,0.4)f(1,0.4)}[-(90)f(1,0.4)]f(1,0.4)]f(1,0.4)f(1,0.4)}"):
            with self.subTest(program=program):
                plain = interpolate_segments(branching_turtle_to_coords(program, 0.4))
                events = {}
                avoider = CollisionAvoider(GridIndex(cell_size=3.0), margin=0.0, attempts=3)
                shaped = interpolate_segments(branching_turtle_to_coords(program, 0.4, avoid=avoider,
                                                                         events=events))
                self.assertEqual(events["collision_truncated_stems"], 0)
                np.testing.assert_array_equal(plain, shaped)

    def test_default_trees_collide_and_avoided_trees_do_not(self):
        # branches of the plain grammar cross once the tree is deep enough for
        # distant sub-trees to meet: from about ten generations at the shorter
        # segment lengths of the default --epsilon range
        deep = {"epsilon": 4.0}
        collided = 0
        for seed in (1, 2):
            plain = grow(seed, niter=10, properties=deep)
            report = describe(plain["nodes"], plain["edges"], margin=0.0)
            collided += report["clearance_um"]["violations"]
        self.assertGreater(collided, 0)                     # the plain grammar lets branches cross
        for seed in (1, 2):
            for tortuosity, persistence in (("stems", None), ("walk", 8.0)):
                with self.subTest(seed=seed, tortuosity=tortuosity):
                    grown = grow(seed, niter=10, properties=deep, avoid_collisions=True,
                                 collision_margin=1.0, tortuosity=tortuosity, persistence=persistence)
                    report = describe(grown["nodes"], grown["edges"], margin=1.0)
                    self.assertEqual(report["clearance_um"]["violations"], 0)
                    if report["clearance_um"]["min"] is not None:
                        self.assertGreaterEqual(report["clearance_um"]["min"], 1.0 - 1e-9)
                    events = grown["events"]
                    self.assertGreater(events["collision_truncated_stems"] + events["collision_terminations"]
                                       + events["collision_redraws"], 0)

    def test_the_two_indexes_give_the_same_network(self):
        if not have_scipy():
            self.skipTest("scipy not installed")
        a = grow(1, niter=7, avoid_collisions=True, collision_index="grid", tortuosity="walk", persistence=8.0)
        b = grow(1, niter=7, avoid_collisions=True, collision_index="kdtree", tortuosity="walk", persistence=8.0)
        np.testing.assert_array_equal(a["nodes"], b["nodes"])
        self.assertEqual(a["events"], b["events"])

    def test_the_avoider_excuses_only_the_junction_neighbourhood(self):
        index = GridIndex(cell_size=5.0)
        avoider = CollisionAvoider(index, margin=0.5, attempts=3)
        parent = avoider.open_stem(None, np.zeros(3), 2.0)
        along = np.linspace(0.0, 20.0, 21)
        avoider.commit(np.stack([np.zeros(21), along, np.zeros(21)], axis=1), np.ones(21), parent, along)
        child = avoider.open_stem(parent, np.array([0.0, 20.0, 0.0]), 2.0)
        # a child point inside the junction sphere overlaps the parent's end legitimately
        near = np.array([[0.5, 20.5, 0.0]])
        self.assertFalse(avoider.conflicts(near, [1.0], child, [0.7])[0])
        # a child point that curls back onto the parent's middle is a collision
        far = np.array([[0.0, 10.0, 1.0]])
        self.assertTrue(avoider.conflicts(far, [1.0], child, [15.0])[0])
        # an unrelated stem is not excused at the junction either
        stranger = avoider.open_stem(None, np.array([50.0, 50.0, 50.0]), 2.0)
        self.assertTrue(avoider.conflicts(near, [1.0], stranger, [0.7])[0])

    def test_counts_reach_the_sidecar(self):
        args = ["--count", "1", "--seed", "1", "--volume", "48", "48", "24", "--iterations", "7", "7",
                "--avoid-collisions", "--collision-margin", "0"]
        with tempfile.TemporaryDirectory() as out:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--out", out]), 0)
            stem = next(n[:-5] for n in os.listdir(out) if n.endswith(".json"))
            with open(os.path.join(out, stem + ".json")) as handle:
                record = json.load(handle)
        self.assertTrue(record["avoid_collisions"])
        self.assertIn(record["collision_index"], ("grid", "kdtree"))
        self.assertGreaterEqual(record["events"]["collision_truncated_stems"], 0)


class AnastomosisTests(unittest.TestCase):
    def setUp(self):
        self.plain = grow(2, niter=7)
        self.looped = grow(2, niter=7, anastomose=True, anastomosis_fraction=0.5)

    def test_cycles_appear_and_tip_density_falls(self):
        before = describe(self.plain["nodes"], self.plain["edges"])
        after = describe(self.looped["nodes"], self.looped["edges"])
        self.assertEqual(before["cycles"], 0)
        self.assertGreater(after["cycles"], 0)
        self.assertEqual(after["cycles"], self.looped["events"]["anastomosis_bridges"]
                         - (before["components"] - after["components"]))
        self.assertLess(after["tips"]["count"], before["tips"]["count"])
        self.assertLess(after["tips"]["per_mm"], before["tips"]["per_mm"])
        self.assertEqual(after["components"], 1)

    def test_bridges_are_new_polylines_joined_bitwise_at_both_ends(self):
        nodes = self.looped["nodes"]
        tree = self.looped["tree"]
        n_plain = self.plain["nodes"].shape[1]
        np.testing.assert_array_equal(nodes[:, :n_plain], self.plain["nodes"])   # the tree is untouched
        bridges = graph.polylines(nodes)
        bridge_runs = [run for run in bridges if tree[run[0]] == BRIDGE]
        self.assertEqual(len(bridge_runs), self.looped["events"]["anastomosis_bridges"])
        plain_keys = {nodes[:3, j].tobytes() for j in range(n_plain) if not np.isnan(nodes[0, j])}
        for run in bridge_runs:
            # a bridge leaves a tip of the tree and lands on the tree or on an earlier bridge
            others = {nodes[:3, j].tobytes() for j in range(nodes.shape[1])
                      if not np.isnan(nodes[0, j]) and j not in set(run.tolist())}
            self.assertIn(nodes[:3, run[0]].tobytes(), plain_keys)
            self.assertIn(nodes[:3, run[-1]].tobytes(), others)
            self.assertTrue(np.all(nodes[3, run] == nodes[3, run[0]]))            # one diameter per bridge

    def test_anastomosis_is_reproducible_and_counted(self):
        again = grow(2, niter=7, anastomose=True, anastomosis_fraction=0.5)
        np.testing.assert_array_equal(again["nodes"], self.looped["nodes"])
        events = self.looped["events"]
        self.assertEqual(events["anastomosis_tips"], 2 ** 6 + 1 - 1)      # every tip but the root
        self.assertEqual(events["anastomosis_selected"],
                         events["anastomosis_bridges"] + events["anastomosis_no_partner"]
                         + events["anastomosis_source_consumed"] + events["anastomosis_collision_failed"])

    def test_bridges_never_join_a_tip_to_its_sister_by_default(self):
        # a sister tip is two segments away; the default separation of three
        # rules it out, a separation of two lets it back in
        for separation, expected in ((3, 0), (2, None)):
            grown = grow(2, niter=7, anastomose=True, anastomosis_fraction=1.0,
                         anastomosis_min_separation=separation)
            nodes = grown["nodes"]
            n_plain = self.plain["nodes"].shape[1]
            canonical = graph.canonical_columns(nodes)
            segments = graph.segments(self.plain["nodes"], self.plain["edges"], graph.canonical_columns(self.plain["nodes"]))
            junction_of_tip = {}
            for path in segments:
                for end, other in ((path[0], path[-1]), (path[-1], path[0])):
                    junction_of_tip[int(end)] = int(other)
            sisters = 0
            for bridge in grown["bridges"]:
                tip, partner = bridge["tip"], bridge["partner"]
                if partner < n_plain and bridge["partner_kind"] == "tip":
                    a = junction_of_tip.get(int(canonical[tip]))
                    b = junction_of_tip.get(int(canonical[partner]))
                    sisters += int(a is not None and a == b)
            if expected == 0:
                self.assertEqual(sisters, 0)
                self.assertGreater(grown["events"]["anastomosis_kin_skipped"], 0)
                self.assertGreater(grown["events"]["anastomosis_bridges"], 0)
            else:
                self.assertGreater(sisters, 0)

    def test_bridges_lead_forward_and_carry_curvature(self):
        # a partner lies within the forward cone of its tip, and a bridge is a
        # curve rather than a strut: its arc exceeds its chord
        from anastomosis import FORWARD_CONE_DEG
        cone = math.cos(math.radians(FORWARD_CONE_DEG))
        nodes = self.looped["nodes"]
        runs = graph.polylines(nodes)
        polyline_of = graph.polyline_of_column(nodes)
        for bridge in self.looped["bridges"]:
            tip, partner = bridge["tip"], bridge["partner"]
            own = runs[polyline_of[tip]]
            before = [c for c in own if c != tip and not np.array_equal(nodes[:3, c], nodes[:3, tip])]
            tangent = nodes[:3, tip] - nodes[:3, before[-1]]
            tangent /= np.linalg.norm(tangent)
            chord = nodes[:3, partner] - nodes[:3, tip]
            self.assertGreaterEqual(float(chord @ tangent), cone * np.linalg.norm(chord) - 1e-9)
            self.assertGreater(bridge["arc_um"], 1.02 * bridge["chord_um"])
        self.assertGreater(self.looped["events"]["anastomosis_behind_skipped"], 0)

    def test_a_zero_fraction_changes_nothing(self):
        nothing = grow(2, niter=7, anastomose=True, anastomosis_fraction=0.0)
        np.testing.assert_array_equal(nothing["nodes"], self.plain["nodes"])

    def test_arteriovenous_mode_grows_two_trees_and_joins_them(self):
        grown = grow(3, niter=7, anastomose=True, anastomose_mode="arteriovenous",
                     anastomosis_fraction=0.7, grow_in_volume=True)
        labels = set(np.unique(grown["tree"]).tolist()) - {-1}
        self.assertEqual(len(grown["programs"]), 2)
        self.assertIn(0, labels)
        self.assertIn(1, labels)
        events = grown["events"]
        self.assertGreater(events["anastomosis_bridges"], 0)
        self.assertGreater(events["anastomosis_bridges_arteriovenous"], 0)
        report = describe(grown["nodes"], grown["edges"], tree=grown["tree"])
        self.assertEqual(report["components"], 1)                        # the two trees are one network
        self.assertGreater(report["cycles"], 0)
        # the roots of both trees are still tips: neither inlet nor outlet was bridged
        self.assertGreaterEqual(report["tips"]["count"], 2)

    def test_avoided_bridges_respect_the_margin(self):
        grown = grow(4, niter=7, tortuosity="walk", persistence=8.0, avoid_collisions=True,
                     collision_margin=1.0, anastomose=True, anastomosis_fraction=0.6)
        self.assertGreater(grown["events"]["anastomosis_bridges"], 0)
        report = describe(grown["nodes"], grown["edges"], margin=1.0)
        self.assertEqual(report["clearance_um"]["violations"], 0)

    def test_bridges_never_cross_the_tree_even_without_avoidance(self):
        # the plain 7-generation tree has no crossing of its own, so every
        # sub-margin pair after anastomosis would be a bridge through a vessel
        before = describe(self.plain["nodes"], self.plain["edges"], margin=1.0)
        after = describe(self.looped["nodes"], self.looped["edges"], margin=1.0)
        self.assertEqual(before["clearance_um"]["violations"], 0)
        self.assertEqual(after["clearance_um"]["violations"], 0)
        self.assertGreater(self.looped["events"]["anastomosis_bridges"], 0)

    def test_a_cycle_containing_network_renders_through_the_rasteriser(self):
        # what a downstream reader does: load the archive, check the layout, rasterise with connect
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "net.npz")
            save_network(path, self.looped["nodes"], metadata={"units": "um"},
                         edges=self.looped["edges"])
            with np.load(path, allow_pickle=False) as handle:
                nodes = handle["nodes"]
                metadata = json.loads(handle["metadata"].item())
        self.assertEqual(nodes.shape[0], 4)
        self.assertEqual(metadata["units"], "um")
        self.assertTrue(np.isnan(nodes[:, np.isnan(nodes[0])]).all())
        extent = np.nanmax(nodes[:3], axis=1) - np.nanmin(nodes[:3], axis=1)
        shape = tuple(int(v) for v in np.ceil(extent / 4.0) + 8)
        volume = process_network(nodes, shape, fit="voxel_size", voxel_size=4.0, connect=True)
        self.assertGreater(int(volume.sum()), 0)
        total, reached = connected_count(volume, 6)
        self.assertEqual(total, reached)


class FamilyTests(unittest.TestCase):
    def test_presets_set_defaults_that_explicit_options_override(self):
        mesh = build_parser("mesh").parse_args([])
        self.assertEqual(mesh.tortuosity, "walk")
        self.assertTrue(mesh.avoid_collisions)
        self.assertEqual(mesh.anastomose_mode, "arteriovenous")
        self.assertTrue(mesh.grow_in_volume)
        overridden = build_parser("mesh").parse_args(["--tortuosity", "stems", "--anastomosis-fraction", "0.1"])
        self.assertEqual(overridden.tortuosity, "stems")
        self.assertEqual(overridden.anastomosis_fraction, 0.1)
        self.assertTrue(overridden.avoid_collisions)
        tumour = build_parser("tumour").parse_args([])
        self.assertEqual(tuple(tumour.d0), (20.0, 10.0))
        self.assertGreater(tumour.aneurysm_prob, lg.default["aneurysm_prob"])
        tree = build_parser("tree").parse_args([])
        self.assertEqual(tree.tortuosity, "stems")
        self.assertFalse(tree.anastomose)

    def test_aligned_is_refused_with_an_explanation(self):
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit) as caught:
                main(["--family", "aligned", "--count", "1", "--seed", "1", "--out", out])
        self.assertIn("aligned", str(caught.exception))
        self.assertIsNone(FAMILIES["aligned"])

    def test_every_available_family_runs_end_to_end(self):
        for family in ("tree", "mesh", "tumour"):
            with self.subTest(family=family):
                args = ["--family", family, "--count", "1", "--seed", "8", "--volume", "48", "48", "24",
                        "--iterations", "6", "6"]
                with tempfile.TemporaryDirectory() as out:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(args + ["--out", out]), 0)
                    stem = next(n[:-5] for n in os.listdir(out) if n.endswith(".json"))
                    with open(os.path.join(out, stem + ".json")) as handle:
                        record = json.load(handle)
                    network = load_network(os.path.join(out, stem + ".npz"))
                self.assertEqual(record["family"], family)
                report = describe(network["nodes"], network["edges"], metadata=record, tree=network["tree"])
                if family == "tree":
                    self.assertEqual(report["cycles"], 0)
                else:
                    self.assertGreater(report["cycles"], 0)
                    self.assertEqual(report["clearance_um"]["violations"], 0)
                if family == "mesh":
                    self.assertEqual(record["trees"], 2)


class DescriptorTests(unittest.TestCase):
    def test_describe_runs_on_a_generated_network_and_serialises(self):
        grown = grow(1, niter=6)
        report = describe(grown["nodes"], grown["edges"])
        text = json.dumps(report)
        self.assertIn("arc_chord", text)
        self.assertEqual(report["components"], 1)
        self.assertEqual(report["cycles"], 0)
        self.assertEqual(report["tips"]["count"], 2 ** 5 + 1)
        self.assertGreater(report["total_length_mm"], 0)
        self.assertGreater(report["arc_chord"]["mean"], 1.0)         # the spline of a zig-zag is not straight
        self.assertAlmostEqual(sum(report["orientation"]["eigenvalues"]), 1.0, places=6)

    def test_describe_matches_the_generator_counts(self):
        grown = grow(2, niter=6, anastomose=True, anastomosis_fraction=0.5)
        report = describe(grown["nodes"], grown["edges"], metadata={"events": grown["events"]})
        self.assertEqual(report["events"]["anastomosis_bridges"], grown["events"]["anastomosis_bridges"])
        self.assertEqual(report["tips"]["count"],
                         int(np.sum(vertex_kinds(grown["nodes"], grown["node_kind"]) == graph.TIP)))

    def test_the_command_line_tool_prints_json(self):
        grown = grow(1, niter=5)
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "net.npz")
            save_network(path, grown["nodes"], metadata={"units": "um"}, edges=grown["edges"])
            result = subprocess.run([sys.executable, os.path.join(ROOT, "describe.py"), path],
                                    capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn("tips", report if "tips" in report else next(iter(report.values())))


if __name__ == "__main__":
    unittest.main()
