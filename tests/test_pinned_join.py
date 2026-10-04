"""
Pins what join_networks draws, as 3.3 released it.

The 3.3 module (tests/fixtures/reference_code_3_3/join.py, the join.py of
3.3.0, whose imports graph, anastomosis and collisions are the current
modules) and the current one are run in the same environment on two small
forests, and the sha256 of the bridges (geometry bytes and the 3.3 fields),
the tips table, the 3.3 counters, the 3.3 summary entries and the generator
state after the call are compared. Running both in one environment is the
only portable comparison: the bridge walk and the collision checks go through
BLAS kernels whose last bits differ between machines. The hashes recorded on
one machine (tests/fixtures/reference_hashes_3_3.json) are checked too, and
skipped only when the reference code itself no longer reproduces them here.

With root attachment off (the default), anything a later version adds to the
result must be inert: every counter appended to EVENT_KEYS is zero, every
bridge is sourced at a tip, and the summary entries added count what 3.3
counted without changing it.

Run from the repository root with
    python -m unittest tests.test_pinned_join
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
REFERENCE = os.path.join(FIXTURES, "reference_code_3_3")
RECORDED = os.path.join(FIXTURES, "reference_hashes_3_3.json")

# The counters, summary entries and bridge fields of 3.3, listed literally so
# that a later version's additions never enter the comparison.
EVENT_KEYS_3_3 = ("join_networks", "join_points", "join_isolated", "join_tips",
                  "join_roots", "join_cut_ends", "join_stubs", "join_eligible", "join_selected",
                  "join_not_selected", "join_bridges", "join_bridged_partners", "join_no_partner",
                  "join_collision_failed", "join_over_budget",
                  "join_partner_tips", "join_partner_interior", "join_source_consumed",
                  "join_redraws", "join_box_redraws", "join_kin_skipped", "join_behind_skipped",
                  "join_components_joined")
SUMMARY_KEYS_3_3 = ("components", "length", "free_tips", "free_tips_per_length", "bridge_volume")
BRIDGE_FIELDS_3_3 = ("tip", "partner", "partner_kind", "chord", "arc", "diameter", "volume", "redraws")

FORESTS = ("pair", "cropped")

# Runs in a subprocess with one join.py first on the path (its imports come
# from the repository root, second on the path) and prints the hashes of what
# join_networks draws for the two forests, plus what the result holds beyond
# the 3.3 fields.
DRIVER = r"""
import hashlib, json, random, sys
code_dir, root, fields = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sys.path.insert(0, root)
sys.path.insert(0, code_dir)
import os
import numpy as np
import main
import join
from anastomosis import BRIDGE
from join import TIP_OUTCOMES, crop_network, join_networks
from spatial import GridIndex

if os.path.dirname(os.path.abspath(join.__file__)) != os.path.abspath(code_dir):
    raise SystemExit(f"join was imported from {join.__file__}, not from {code_dir}")

PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True,
              "aneurysm_prob": 0.0, "stenosis_prob": 0.0}


def grow(seed, niter):
    random.seed(seed)
    np.random.seed(seed)
    preset = dict(main.FAMILIES["tree"])
    properties = dict(PROPERTIES)
    for key in ("aneurysm_prob", "stenosis_prob"):
        if key in preset:
            properties[key] = preset.pop(key)
    preset.pop("d0", None)
    grown = main.grow_network(niter, 8.0, properties, (48, 48, 24), d_min=1.0, seed=seed, subdivisions=3, **preset)
    tree = grown["tree"]
    roots = [int(np.flatnonzero(tree == label)[0]) for label in np.unique(tree[tree >= 0]) if label != BRIDGE]
    return {"nodes": grown["nodes"], "node_kind": grown["node_kind"], "tree": tree,
            "roots": np.array(roots, dtype=np.int64)}


def rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def place(network, matrix, offset):
    nodes = network["nodes"].copy()
    finite = np.isfinite(nodes[0])
    centre = np.nanmean(nodes[:3], axis=1)
    nodes[:3, finite] = matrix @ (nodes[:3, finite] - centre[:, None]) + np.asarray(offset, dtype=float)[:, None]
    return dict(network, nodes=nodes)


def build_forest(library, count, lo, hi, rng, clearance_gap=2.0, tries=40):
    index = GridIndex(cell_size=20.0)
    placed = []
    for i in range(count):
        network = library[i % len(library)]
        for _ in range(tries):
            candidate = place(network, rotation(rng), rng.uniform(lo, hi))
            finite = np.isfinite(candidate["nodes"][0])
            points = candidate["nodes"][:3, finite].T
            radii = candidate["nodes"][3, finite] / 2.0
            if index.n:
                qi, _ = index.query(points, radii, clearance_gap)
                if len(qi):
                    continue
            index.add(points, radii, np.full(len(points), i))
            placed.append(candidate)
            break
    return placed


def forests():
    # (a) two facing copies of one tree, uncropped, no budget, policy cross
    tree = grow(1, 4)
    del tree["roots"]                                   # the default roots
    nodes = tree["nodes"]
    height = float(np.nanmax(nodes[1]) - np.nanmin(nodes[1]))
    pair = [place(tree, np.eye(3), np.zeros(3)),
            place(tree, np.diag([-1.0, -1.0, 1.0]), np.array([0.0, height + 15.0, 0.0]))]
    lo = np.min([np.nanmin(n["nodes"][:3], axis=1) for n in pair], axis=0) - 20.0
    hi = np.max([np.nanmax(n["nodes"][:3], axis=1) for n in pair], axis=0) + 20.0
    yield "pair", pair, dict(fraction=0.8, collision_margin=0.5, box=(lo, hi), boundary_margin=0.0,
                             policy="cross", max_bridge_volume=None)
    # (b) a placed forest cropped to an inner box, roots passed through the
    # crop, a boundary margin, and a budget that stops part-way
    library = [grow(seed, 5) for seed in (1, 2, 3)]
    forest = build_forest(library, 20, np.zeros(3), np.full(3, 240.0), np.random.default_rng(21))
    lo, hi = np.full(3, 40.0), np.full(3, 200.0)
    margin = 1.0
    r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
    cropped = [crop_network(n, lo, hi, r_max + margin) for n in forest]
    yield "cropped", cropped, dict(fraction=0.9, collision_margin=margin, box=(lo, hi), boundary_margin=2.0,
                                   policy="cross", max_bridge_volume=3000.0)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


out = {}
for name, networks, settings in forests():
    rng = np.random.default_rng(22)
    result = join_networks(networks, rng, **settings)
    bridges = [dict({field: bridge[field] for field in fields["bridge"]},
                    geometry=sha(np.ascontiguousarray(bridge["geometry"]).tobytes()))
               for bridge in result["bridges"]]
    outcomes = result["tips"]["outcome"]
    out[name] = {
        "hashes": {
            "bridges": len(bridges),
            "bridges_sha256": sha(canonical(bridges)),
            "tips_sha256": sha(np.ascontiguousarray(result["tips"]).tobytes()),
            "events_sha256": sha(canonical({key: result["events"][key] for key in fields["events"]})),
            "summary_sha256": sha(canonical({key: result["summary"][key] for key in fields["summary"]})),
            "rng_state_sha256": sha(canonical(rng.bit_generator.state)),
        },
        "extras": {
            "over_budget": int(np.count_nonzero(outcomes == TIP_OUTCOMES.index("over_budget"))),
            "appended_events": {key: value for key, value in result["events"].items()
                                if key not in fields["events"]},
            "source_kinds": sorted({bridge.get("source_kind", "tip") for bridge in result["bridges"]}),
            "summary": result["summary"],
        },
    }
print(json.dumps(out))
"""


def run_driver(code_dir):
    """Hashes and extras of what the join.py in `code_dir` draws for each forest."""
    fields = {"events": EVENT_KEYS_3_3, "summary": SUMMARY_KEYS_3_3, "bridge": BRIDGE_FIELDS_3_3}
    result = subprocess.run([sys.executable, "-c", DRIVER, code_dir, ROOT, json.dumps(fields)],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode:
        raise AssertionError(f"{code_dir}: {result.stderr}")
    return json.loads(result.stdout)


class PinnedJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = run_driver(ROOT)
        cls.reference = run_driver(REFERENCE)
        with open(RECORDED) as handle:
            cls.recorded = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}

    def test_join_networks_draws_what_the_3_3_module_draws_in_this_environment(self):
        for name in FORESTS:
            with self.subTest(forest=name):
                self.assertGreater(self.reference[name]["hashes"]["bridges"], 0)
                self.assertEqual(self.current[name]["hashes"], self.reference[name]["hashes"])
        # the cropped forest's budget stops part-way, so the over_budget path is pinned too
        self.assertGreater(self.reference["cropped"]["extras"]["over_budget"], 0)
        self.assertGreater(self.reference["cropped"]["hashes"]["bridges"], 1)

    def test_join_networks_matches_the_recorded_3_3_hashes(self):
        self.assertEqual(set(self.recorded), set(FORESTS))
        for name in FORESTS:
            with self.subTest(forest=name):
                if self.current[name]["hashes"] == self.recorded[name]:
                    continue
                if self.reference[name]["hashes"] != self.recorded[name]:
                    self.skipTest("this environment's rounding differs from the recording machine's; "
                                  "the same-environment comparison above still holds")
                self.assertEqual(self.current[name]["hashes"], self.recorded[name])

    def test_with_root_attachment_off_nothing_beyond_the_3_3_fields_is_live(self):
        for name in FORESTS:
            with self.subTest(forest=name):
                extras = self.current[name]["extras"]
                for key, value in extras["appended_events"].items():
                    self.assertEqual(value, 0, key)
                self.assertLessEqual(set(extras["source_kinds"]), {"tip"})
                summary = extras["summary"]
                added = {key: value for key, value in summary.items() if key not in SUMMARY_KEYS_3_3}
                for key, entry in added.items():
                    self.assertEqual(set(entry), {"before", "after"}, key)
                # no root is attached, so the blunt roots are what they were; the
                # free ends and the per-length rates follow the tip bridges as 3.3's do
                if "blunt_roots" in added:
                    self.assertEqual(added["blunt_roots"]["after"], added["blunt_roots"]["before"])
                if "free_ends" in added:
                    for when in ("before", "after"):
                        self.assertEqual(added["free_ends"][when],
                                         summary["free_tips"][when] + added["blunt_roots"][when])
                for key in ("blunt_roots", "free_ends"):
                    if key + "_per_length" in added:
                        for when in ("before", "after"):
                            length = summary["length"][when]
                            self.assertAlmostEqual(added[key + "_per_length"][when], added[key][when] / length)


if __name__ == "__main__":
    unittest.main()
