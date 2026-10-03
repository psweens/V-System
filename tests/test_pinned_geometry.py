"""
Pins the geometry grow_network draws for the mesh and tumour families.

The 3.2 modules (tests/fixtures/reference_code_3_2, the import closure of
main.grow_network as released) are run in the same environment as the current
ones, on the same seeds, and the sha256 of `nodes`, `tree` and the bridges
list are compared. Running both in one environment is the only portable
comparison: the B-spline's matrix products go through BLAS kernels whose last
bits differ between machines. The hashes recorded on one machine
(tests/fixtures/reference_hashes_3_2.json) are checked too, and skipped only
when the reference code itself no longer reproduces them here.

Run from the repository root with
    python -m unittest tests.test_pinned_geometry
"""
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
REFERENCE = os.path.join(FIXTURES, "reference_code_3_2")
RECORDED = os.path.join(FIXTURES, "reference_hashes_3_2.json")

# (family, seed, generations): small networks with a handful of bridges each
CASES = (("mesh", 3, 7), ("mesh", 5, 7), ("tumour", 1, 6), ("tumour", 2, 6))

# Runs in a subprocess with the modules of one code directory on the path and
# prints the hashes of what grow_network draws for one case.
DRIVER = r"""
import hashlib, json, random, sys
code_dir, family, seed, niter = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, code_dir)
import numpy as np
from main import FAMILIES, grow_network
properties = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
options = dict(FAMILIES[family])
for key in ("aneurysm_prob", "stenosis_prob"):
    if key in options:
        properties[key] = options.pop(key)
options.pop("d0", None)
random.seed(seed)
np.random.seed(seed)
grown = grow_network(niter, 20.0, properties, (48, 48, 24), seed=seed, **options)
sha = lambda data: hashlib.sha256(data).hexdigest()
print(json.dumps({
    "columns": int(grown["nodes"].shape[1]),
    "bridges": len(grown["bridges"]),
    "nodes_sha256": sha(np.ascontiguousarray(grown["nodes"]).tobytes()),
    "tree_sha256": sha(np.ascontiguousarray(grown["tree"]).tobytes()),
    "bridges_sha256": sha(json.dumps(grown["bridges"], sort_keys=True).encode()),
}))
"""


def hashes(code_dir, family, seed, niter):
    """Hashes of what the modules in `code_dir` draw for one case."""
    result = subprocess.run([sys.executable, "-c", DRIVER, code_dir, family, str(seed), str(niter)],
                            capture_output=True, text=True, cwd=code_dir)
    if result.returncode:
        raise AssertionError(f"{code_dir}: {result.stderr}")
    return json.loads(result.stdout)


def case_key(family, seed, niter):
    return f"{family}_s{seed}_i{niter}"


class PinnedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = {case_key(*case): hashes(ROOT, *case) for case in CASES}
        cls.reference = {case_key(*case): hashes(REFERENCE, *case) for case in CASES}
        with open(RECORDED) as handle:
            cls.recorded = {k: v for k, v in json.load(handle).items() if not k.startswith("_")}

    def test_grow_network_draws_what_the_3_2_modules_draw_in_this_environment(self):
        for case in CASES:
            key = case_key(*case)
            with self.subTest(case=key):
                self.assertGreater(self.reference[key]["bridges"], 0)       # the pin covers bridges
                self.assertEqual(self.current[key], self.reference[key])

    def test_grow_network_matches_the_recorded_3_2_hashes(self):
        self.assertEqual(set(self.recorded), {case_key(*case) for case in CASES})
        for case in CASES:
            key = case_key(*case)
            with self.subTest(case=key):
                if self.current[key] == self.recorded[key]:
                    continue
                if self.reference[key] != self.recorded[key]:
                    self.skipTest("this environment's rounding differs from the recording machine's; "
                                  "the same-environment comparison above still holds")
                self.assertEqual(self.current[key], self.recorded[key])


if __name__ == "__main__":
    unittest.main()
