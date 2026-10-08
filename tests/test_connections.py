"""
Tests of connections.py, the rungs that join neighbouring capillaries: two
parallel lines 400 d long and 5 d apart (chords, site gaps, components and
cycles, the partition of the sites, junction zones and clearance), partners
that touch, lines of unequal diameter (the search radius and the rung's
diameter), joints shared by two polylines, the lateral rule about the
tangent and about an axis and its counter, the ranking of partners, the tree
separation, the refusals, and the stage on its own: deterministic, drawing
only from its generator in a fixed order, with gaps and walk steps set by
the vessel, making each rung an obstacle, leaving its input and every
existing column alone, scaling with the unit, and adding no clearance
violation to a grown network.

Run from the repository root with
    python -m unittest tests.test_connections
The fast tests use seeds 1-3. The development measurement over seeds 1-10,
which every tolerance below was set from, runs only with
VSYSTEM_SLOW_TESTS=1.
"""
import math
import os
import random
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import anastomosis  # noqa: E402
import connections  # noqa: E402
import graph  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from collisions import KIN_REACH  # noqa: E402
from connections import EVENT_KEYS, RUNG_OUTCOMES, check_settings, cross_connect  # noqa: E402
from describe import clearance  # noqa: E402
from main import grow_network  # noqa: E402
from spatial import make_index  # noqa: E402
from tortuosity import bridge_path  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the slow tests")

SEEDS = (1, 2, 3)
TEN_SEEDS = tuple(range(1, 11))
MARGIN = 1.0
SETTINGS = {"d_min": 1.0, "collision_margin": MARGIN, "persistence": 10.0}
PARTITION = ("rung_sites_near_junction", "rung_sites_consumed", "rung_no_partner", "rung_collision_failed",
             "rung_bridges")
OUTCOME_EVENTS = {"bridge": "rung_bridges", "near_junction": "rung_sites_near_junction",
                  "consumed": "rung_sites_consumed", "no_partner": "rung_no_partner",
                  "collision_failed": "rung_collision_failed"}

# The development measurement on the two parallel lines (400 d long, 5 d
# apart, d 1, points every 0.2 d), default settings with d_min 1, margin 1 and
# persistence 10, seeds 1-10:
#   sites 37 34 36 49 33 36 33 34 52 39; rungs 29 28 30 37 29 29 26 29 38 30;
#   near_junction 7 5 6 12 4 6 7 4 14 9; consumed 1 1 0 0 0 1 0 1 0 0;
#   no_partner and collision_failed 0; redraws 1 1 2 0 2 1 2 1 4 1;
#   every chord 5 d exactly and at 90 degrees to x (the two lines share their
#   x samples, so the nearest partner is straight across); every site within
#   0.0998 of its vertex (half the sampling, 0.1, bounds it); clearance
#   violations 0; KS p of the site gaps against Exp(20), per seed:
KS_P_TEN_SEEDS = (0.37, 0.40, 0.46, 0.23, 0.18, 0.48, 0.25, 0.38, 0.12, 0.98)
# numpy's exact KS agreed with scipy's within 5.6e-17 on D and 1.8e-15 on p
# (2.3e-15 on the d 1.5 lines below), so the cross-check allows 1e-15 and
# 1e-14. Rotated by 45 degrees about z, the chords straight across came within
# 4.1e-14 of 5 d (under one ulp of the rotated coordinates, 5.7e-14), so that
# test allows 1e-13; the chords stood at least 2.5 off the tangent bound and
# 0.051 off the axis bound, and the axis's chords 0.016 longer than 5 d /
# cos 15 degrees, so those bounds need no slack.
# On two lines of d 1.5 (still below 2 d_min), 5 apart and sampled every 0.25
# (not 0.2 d), seeds 1-10 got 20 21 21 25 16 24 17 24 29 21 rungs, every site
# within 0.1247 of its vertex (half the sampling, 0.125), and KS p of the
# gaps against Exp(30) 0.92 0.26 0.90 0.17 0.10 0.36 0.17 0.86 0.40 0.84.
# Lines of d 1 and d 1.5, margin 1: 5 apart, 20-34 rungs from sites on both
# lines; 10 apart, which only 8 x 1.5 reaches, 7-14 rungs, every one from the
# d 1.5 line; every rung 1.0 wide. Lines of d 1, 1.5 apart with margin 1: no
# rung, every free site no_partner (straight across touches, and anything
# further along stands less than 60 degrees off x); without a margin 32 rungs
# on seed 1; 2.5 apart with margin 1, 26-38 rungs, every chord 2.5.
# On the bent vessel (bent()), seeds 1-10 joined C to Q at 65 degrees to 0
# difference for phi +-65 and refused both sites by the angle for +-55; the
# angle test allows 1e-13, a few ulps of 65 for the trigonometry's last bits.
# On the cut line (split_lines()), seeds 1-10 put no site or rung end within
# a zone of a joint, while 2-12 sites a seed fell inside one.
# On walked trees (d0 4, d_min 1, margin 1, anastomosis "any" 0.5, default
# properties), seeds 1-10 got 3 6 3 5 3 3 5 7 1 3 rungs at the default
# spacing, with no clearance violation after them; 37 of their 39 rungs, and
# 46 of 49 at spacing 10, joined vertices of unequal diameter.
# The tests assert chords within 0.2 d of 5 d and at least 60 degrees to
# x, and KS p above 0.01 (seeds 1-3). Scaling by 0.5 and 2 reproduced
# every column and every chord to 0 difference on seeds 1-10, on the lines
# and on the walked trees, so the scale test asks for equality.
SCALE_ATOL = 0.0


def lines(*specs, length=400.0, spacing=0.2):
    """Straight polylines along x, one per (y, diameter, label), in archive order, sampled every `spacing`."""
    n = int(round(length / spacing)) + 1
    x = np.linspace(0.0, length, n)
    blocks, labels = [], []
    for y, diameter, label in specs:
        if blocks:
            blocks.append(np.full((4, 1), np.nan))
            labels.append(np.array([-1]))
        blocks.append(np.stack([x, np.full(n, float(y)), np.zeros(n), np.full(n, float(diameter))]))
        labels.append(np.full(n, label))
    return np.concatenate(blocks, axis=1), np.concatenate(labels).astype(np.int8)


def parallel_lines(gap=5.0, diameter=1.0):
    """Two lines along x, `gap` apart along y, trees 0 and 1."""
    return lines((0.0, diameter, 0), (gap, diameter, 1))


def polyline(points, spacing=0.2):
    """Columns sampled about every `spacing` along the straight pieces between `points`, diameter 1."""
    points = np.asarray(points, dtype=float)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        steps = max(1, int(round(np.linalg.norm(b - a) / spacing)))
        t = np.arange(1, steps + 1)[:, None] / steps
        out.append(a + t * (b - a))
    xyz = np.concatenate(out)
    return np.vstack([xyz.T, np.ones(len(xyz))])


def fork(gap=5.0, length=200.0):
    """
    One tree: a parent along x to J = (20, 0, 0), where two daughters part
    and then run parallel along x, `gap` apart, to x = `length`. The parent
    and the first daughter are one polyline and the second daughter another
    starting at J, as the interpreter writes them.
    """
    first = polyline([(0, 0, 0), (20, 0, 0), (25, gap / 2, 0), (length, gap / 2, 0)])
    second = polyline([(20, 0, 0), (25, -gap / 2, 0), (length, -gap / 2, 0)])
    nodes = np.concatenate([first, np.full((4, 1), np.nan), second], axis=1)
    tree = np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8)
    return nodes, tree


def u_bend(stubs):
    """
    A polyline along x at y 0 from x 200 to 0, up to y 5 and back along x to
    200: one vessel folded on itself. With `stubs`, a short daughter leaves
    each corner along -x, so that the corners become junctions and the fold
    three vessels: the two arms and the bar between them, the arms three
    segments apart.
    """
    pieces = [polyline([(200, 0, 0), (0, 0, 0), (0, 5, 0), (200, 5, 0)])]
    if stubs:
        pieces += [polyline([(0, 0, 0), (-3, 0, 0)]), polyline([(0, 5, 0), (-3, 5, 0)])]
    blocks = []
    for piece in pieces:
        blocks += [np.full((4, 1), np.nan), piece] if blocks else [piece]
    nodes = np.concatenate(blocks, axis=1)
    return nodes, np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8)


def wide_lines():
    """Two lines of d 1.5, still capillaries below 2 d_min, 5 apart and sampled every 0.25, which is not 0.2 d."""
    return lines((0.0, 1.5, 0), (5.0, 1.5, 1), spacing=0.25)


def split_lines(every=40.0):
    """
    The two parallel lines with the first cut into polylines `every` long,
    each starting with a bitwise copy of the last column of the one before:
    one vessel still, whose joints have degree two but are shared by two
    polylines. Returns (nodes, tree, joints), joints the x of the cuts.
    """
    whole, _ = parallel_lines()
    n = whole.shape[1] // 2
    first, second = whole[:, :n], whole[:, n + 1:]
    cut = int(round(every / 0.2))
    blocks = [first[:, start:start + cut + 1] for start in range(0, n - 1, cut)] + [second]
    labels = [np.zeros(block.shape[1]) for block in blocks[:-1]] + [np.ones(n)]
    for j in range(len(blocks) - 1, 0, -1):
        blocks.insert(j, np.full((4, 1), np.nan))
        labels.insert(j, np.array([-1]))
    joints = first[0, cut:n - 1:cut]
    return np.concatenate(blocks, axis=1), np.concatenate(labels).astype(np.int8), joints


def bent(phi_deg, theta_deg=20.0):
    """
    A vessel of three vertices, bent by 2 theta at its middle C, which every
    site snaps to: it comes in at -theta to x and leaves at +theta, so its
    tangent there, from the two neighbours, is x, while the incoming and the
    outgoing directions lie theta either side. Its only partner Q lies 5 from
    C at phi to x, on a vessel tilted 30 degrees off the chord out of the
    plane, so that Q's own chord back to C is refused by Q's tangent.
    """
    t, p, tilt = math.radians(theta_deg), math.radians(phi_deg), math.radians(30.0)
    incoming = np.array([math.cos(t), -math.sin(t), 0.0])
    outgoing = np.array([math.cos(t), math.sin(t), 0.0])
    chord = np.array([math.cos(p), math.sin(p), 0.0])
    across = math.cos(tilt) * chord + math.sin(tilt) * np.array([0.0, 0.0, 1.0])
    site_vessel = np.stack([-50.0 * incoming, np.zeros(3), 50.0 * outgoing], axis=1)
    q = 5.0 * chord
    partner_vessel = np.stack([q - 50.0 * across, q, q + 50.0 * across], axis=1)
    nodes = np.concatenate([np.vstack([site_vessel, np.ones(3)]), np.full((4, 1), np.nan),
                            np.vstack([partner_vessel, np.ones(3)])], axis=1)
    return nodes, np.array([0, 0, 0, -1, 1, 1, 1], dtype=np.int8)


def run(nodes, tree, seed, **settings):
    """cross_connect on its own generator: (nodes, tree, rungs, events, sites, generator)."""
    options = dict(SETTINGS)
    options.update(settings)
    events, sites = {}, []
    rng = np.random.default_rng([seed, 5])
    out, out_tree, rungs = cross_connect(nodes, tree, rng, events=events, sites=sites, **options)
    return out, out_tree, rungs, events, sites, rng


def rung_runs(out, n_input):
    """The (first, last) columns of every polyline appended after the first n_input columns."""
    return [(int(run[0]), int(run[-1])) for run in graph.polylines(out) if run[0] >= n_input]


def grown_tree(seed):
    """A walked tree of capillaries closed by anastomosis: d0 4, d_min 1, margin 1, default properties."""
    random.seed(seed)
    np.random.seed(seed)
    return grow_network(64, 4.0, None, (64, 64, 64), d_min=1.0, tortuosity="walk", persistence=10.0,
                        avoid_collisions=True, collision_margin=MARGIN, anastomose=True,
                        anastomosis_fraction=0.5, anastomose_mode="any", seed=seed)


def _matrix_power(matrix, exponent, n):
    """matrix ** n with a decimal exponent carried apart, as Marsaglia, Tsang and Wang (2003) do."""
    if n == 1:
        return matrix.copy(), exponent
    half, e_half = _matrix_power(matrix, exponent, n // 2)
    out, e_out = half @ half, 2 * e_half
    if n % 2:
        out, e_out = matrix @ out, e_out + exponent
    centre = out.shape[0] // 2
    if out[centre, centre] > 1e140:
        out, e_out = out * 1e-140, e_out + 140
    return out, e_out


def kolmogorov_cdf(n, d):
    """P(D_n < d) for the two-sided one-sample statistic, exactly (Marsaglia, Tsang and Wang 2003)."""
    if d <= 0.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    i, j = np.indices((m, m))
    H = np.where(i - j + 1 >= 0, 1.0, 0.0)
    H[:, 0] -= h ** np.arange(1, m + 1)
    H[m - 1, :] -= h ** np.arange(m, 0, -1)
    if 2.0 * h - 1.0 > 0.0:
        H[m - 1, 0] += (2.0 * h - 1.0) ** m
    factorial = np.array([float(math.factorial(v)) for v in range(m + 1)])
    H = np.where(i - j + 1 > 0, H / factorial[np.clip(i - j + 1, 0, m)], H)
    Q, e_q = _matrix_power(H, 0, n)
    s = Q[k - 1, k - 1]
    for step in range(1, n + 1):
        s = s * step / n
        if s < 1e-140:
            s, e_q = s * 1e140, e_q - 140
    return s * 10.0 ** e_q


def ks_exponential(sample, scale):
    """(D, p) of the one-sample Kolmogorov-Smirnov test of `sample` against Exp(scale), p exact."""
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    cdf = -np.expm1(-x / scale)
    rank = np.arange(1, n + 1)
    d = float(max(np.max(rank / n - cdf), np.max(cdf - (rank - 1) / n)))
    return d, 1.0 - kolmogorov_cdf(n, d)


def site_gaps(nodes, sites):
    """The gaps between consecutive snapped sites along each line, from x = 0, pooled over the lines."""
    gaps = []
    for unit in sorted({record["unit"] for record in sites}):
        x = np.sort([nodes[0, record["column"]] for record in sites if record["unit"] == unit])
        gaps.append(np.diff(np.concatenate([[0.0], x])))
    return np.concatenate(gaps)


class ParallelLineTests(unittest.TestCase):
    """Two parallel lines along x, 400 d long and 5 d apart."""

    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.tree = parallel_lines()
        cls.runs = {seed: run(cls.nodes, cls.tree, seed) for seed in SEEDS}

    def test_chords_stand_at_least_sixty_degrees_off_x_and_span_the_gap(self):
        for seed, (out, _, rungs, _, _, _) in self.runs.items():
            with self.subTest(seed=seed):
                self.assertGreater(len(rungs), 0)
                for (first, last), rung in zip(rung_runs(out, self.nodes.shape[1]), rungs):
                    chord = out[:3, last] - out[:3, first]
                    length = np.linalg.norm(chord)
                    angle = math.degrees(math.acos(min(1.0, abs(chord[0]) / length)))
                    self.assertGreaterEqual(angle, 60.0)
                    self.assertLessEqual(abs(length - 5.0), 0.2)
                    self.assertEqual(rung["chord_um"], length)
                    self.assertGreaterEqual(rung["angle_deg"], 60.0)

    def test_site_gaps_are_exponential_with_a_mean_of_twenty_diameters(self):
        # and of 20 x 1.5 = 30 on lines of d 1.5, whose median diameter sets the gaps
        wide = wide_lines()
        cases = [(self.nodes, seed, sites, 20.0, 0.1) for seed, (_, _, _, _, sites, _) in self.runs.items()]
        cases += [(wide[0], seed, run(*wide, seed)[4], 30.0, 0.125) for seed in SEEDS]
        for nodes, seed, sites, mean, half in cases:
            with self.subTest(mean=mean, seed=seed):
                for record in sites:
                    # every interior point of a bare line can take a site: it snaps to the nearest
                    self.assertLessEqual(abs(nodes[0, record["column"]] - record["arc"]), half)
                gaps = site_gaps(nodes, sites)
                d, p = ks_exponential(gaps, mean)
                self.assertGreater(p, 0.01)
                try:
                    from scipy import stats
                except ImportError:
                    continue
                reference = stats.kstest(gaps, "expon", args=(0.0, mean), method="exact")
                self.assertAlmostEqual(d, reference.statistic, delta=1e-15)
                self.assertAlmostEqual(p, reference.pvalue, delta=1e-14)

    def test_the_first_rung_joins_the_lines_and_each_later_rung_adds_one_cycle(self):
        for seed, (out, _, rungs, _, _, _) in self.runs.items():
            with self.subTest(seed=seed):
                n = self.nodes.shape[1]
                before = graph.build(out[:, :n])
                counts = graph.betti(before["edges"], before["canonical"])
                self.assertEqual((counts["components"], counts["cycles"]), (2, 0))
                for j, (_, last) in enumerate(rung_runs(out, n), start=1):
                    built = graph.build(out[:, :last + 1])
                    counts = graph.betti(built["edges"], built["canonical"])
                    self.assertEqual((counts["components"], counts["cycles"]), (1, j - 1))
                self.assertEqual(j, len(rungs))

    def test_rung_ends_are_bitwise_copies_of_degree_two_points_that_become_junctions(self):
        degree_before = graph.build(self.nodes)["degree"]
        for seed, (out, out_tree, rungs, _, _, _) in self.runs.items():
            with self.subTest(seed=seed):
                built = graph.build(out)
                for (first, last), rung in zip(rung_runs(out, self.nodes.shape[1]), rungs):
                    for end, column in ((first, rung["site"]), (last, rung["partner"])):
                        self.assertEqual(out[:3, end].tobytes(), self.nodes[:3, column].tobytes())
                        self.assertEqual(degree_before[column], 2)
                        self.assertEqual(built["degree"][built["canonical"][end]], 3)
                    self.assertTrue(np.all(out_tree[first:last + 1] == BRIDGE))
                    self.assertTrue(np.all(out[3, first:last + 1] == rung["diameter_um"]))
                    self.assertEqual(rung["diameter_um"], 1.0)
                    self.assertEqual((rung["site_tree"], rung["partner_tree"]),
                                     (int(self.tree[rung["site"]]), int(self.tree[rung["partner"]])))

    def test_no_rung_end_lies_in_a_junction_zone_and_no_clearance_is_violated(self):
        reach = KIN_REACH * (0.5 + 0.5 + MARGIN)
        for seed, (out, _, rungs, _, _, _) in self.runs.items():
            with self.subTest(seed=seed):
                # a bare line's junctions are its rung ends, and its tips are not centres
                for label in (0, 1):
                    ends = [r[key] for r in rungs for key in ("site", "partner") if self.tree[r[key]] == label]
                    xyz = self.nodes[:3, ends].T
                    distance = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
                    np.fill_diagonal(distance, np.inf)
                    self.assertGreaterEqual(float(distance.min()), reach)
                self.assertEqual(clearance(out, margin=MARGIN)["violations"], 0)

    def test_every_site_has_one_outcome_and_the_partition_holds(self):
        for seed, (out, _, rungs, events, sites, _) in self.runs.items():
            with self.subTest(seed=seed):
                self.assertEqual(sorted(events), sorted(EVENT_KEYS))
                self.assertEqual(events["rung_sites"], len(sites))
                self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
                for outcome in RUNG_OUTCOMES:
                    self.assertEqual(events[OUTCOME_EVENTS[outcome]],
                                     sum(record["outcome"] == outcome for record in sites))
                self.assertEqual(events["rung_bridges"], len(rungs))
                self.assertEqual(len(rung_runs(out, self.nodes.shape[1])), len(rungs))
                self.assertEqual(events["rung_bridges_cross_tree"], len(rungs))       # one line per tree
                self.assertGreater(events["rung_sites_near_junction"], 0)
        self.assertGreater(sum(r[3]["rung_sites_consumed"] for r in self.runs.values()), 0)

    def test_lines_out_of_reach_get_no_rung_and_every_free_site_no_partner(self):
        nodes, tree = parallel_lines(gap=10.0)          # beyond 8 site diameters
        for seed in SEEDS:
            with self.subTest(seed=seed):
                out, _, rungs, events, sites, _ = run(nodes, tree, seed)
                self.assertEqual(rungs, [])
                self.assertIs(out, nodes)
                self.assertEqual(events["rung_no_partner"], events["rung_sites"] - events["rung_sites_consumed"])
                self.assertGreater(events["rung_no_partner"], 0)

    def test_partners_already_touching_the_site_are_refused(self):
        # 1.5 apart with margin 1, the point straight across touches the site,
        # (1 + 1) / 2 + 1 = 2 > 1.5, and every point far enough along not to
        # touch stands less than 60 degrees off x; 2.5 apart nothing touches
        nodes, tree = parallel_lines(gap=1.5)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, _, _ = run(nodes, tree, seed)
                self.assertEqual(rungs, [])
                self.assertEqual(events["rung_no_partner"], events["rung_sites"] - events["rung_sites_consumed"])
                self.assertGreater(events["rung_no_partner"], 0)
                # without a margin the same points no longer touch
                self.assertGreater(len(run(nodes, tree, seed, collision_margin=None)[2]), 0)
                _, _, rungs, _, _, _ = run(*parallel_lines(gap=2.5), seed)
                self.assertGreater(len(rungs), 0)
                self.assertTrue(all(rung["chord_um"] == 2.5 for rung in rungs))

    def test_a_vessel_in_the_way_fails_every_rung(self):
        # a vessel of diameter 3 halfway across: never a site or a partner, since
        # its median diameter is not below 2 d_min, but in the way of every chord
        nodes, tree = lines((0.0, 1.0, 0), (4.5, 3.0, 0), (9.0, 1.0, 1), length=100.0)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                out, _, rungs, events, sites, _ = run(nodes, tree, seed, radius=12.0, attempts=2)
                self.assertEqual(rungs, [])
                self.assertGreater(events["rung_collision_failed"], 0)
                self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
                self.assertEqual(events["rung_bridge_redraws"], 3 * 5 * events["rung_collision_failed"])
                n = nodes.shape[1] // 3
                self.assertTrue(all(record["column"] < n or record["column"] > 2 * n for record in sites))


class VesselDiameterTests(unittest.TestCase):
    """Lines of unequal diameter: the search reaches radius site diameters, and a rung is as wide as its thinner end."""

    def test_the_search_reaches_radius_times_the_site_diameter(self):
        # 10 apart, beyond 8 x 1 but within 8 x 1.5: only the wider line's sites find partners
        nodes, tree = lines((0.0, 1.0, 0), (10.0, 1.5, 1))
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, _, sites, _ = run(nodes, tree, seed)
                self.assertGreater(len(rungs), 0)
                self.assertTrue(all(tree[rung["site"]] == 1 for rung in rungs))
                searched = [record["outcome"] for record in sites if tree[record["column"]] == 0
                            and record["outcome"] not in ("near_junction", "consumed")]
                self.assertGreater(len(searched), 0)
                self.assertEqual(set(searched), {"no_partner"})

    def test_a_rung_is_as_wide_as_its_narrower_end(self):
        for gap in (5.0, 10.0):
            nodes, tree = lines((0.0, 1.0, 0), (gap, 1.5, 1))
            for seed in SEEDS:
                with self.subTest(gap=gap, seed=seed):
                    out, _, rungs, _, _, _ = run(nodes, tree, seed)
                    self.assertGreater(len(rungs), 0)
                    for (first, last), rung in zip(rung_runs(out, nodes.shape[1]), rungs):
                        self.assertEqual(rung["diameter_um"], 1.0)
                        self.assertTrue(np.all(out[3, first:last + 1] == 1.0))
                    if gap == 5.0:
                        # sites on both lines, so the narrower end is the site's as often as the partner's
                        self.assertEqual({int(tree[rung["site"]]) for rung in rungs}, {0, 1})


class SharedVertexTests(unittest.TestCase):
    def test_a_joint_shared_by_two_polylines_is_a_centre_and_never_an_end(self):
        nodes, tree, joints = split_lines()
        self.assertEqual(len(connections._Vessels(nodes, tree, graph.DEFAULT_TOL, 1.0, 2.0).segments), 2)
        joint_columns = np.flatnonzero((tree == 0) & np.isin(nodes[0], joints)).tolist()
        self.assertEqual(len(joint_columns), 2 * len(joints))
        reach = KIN_REACH * (0.5 + 0.5 + MARGIN)
        tested = 0
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, sites, _ = run(nodes, tree, seed)
                self.assertGreater(len(rungs), 0)
                self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
                for rung in rungs:
                    for end in (rung["site"], rung["partner"]):
                        if tree[end] == 0:
                            self.assertGreaterEqual(float(np.min(np.abs(nodes[0, end] - joints))), reach)
                for record in sites:
                    self.assertNotIn(record["column"], joint_columns)
                    # a site within a joint's zone, on either line, would otherwise join near the joint
                    if np.min(np.abs(nodes[0, record["column"]] - joints)) < reach:
                        tested += 1
                        if tree[record["column"]] == 0:
                            self.assertIn(record["outcome"], ("near_junction", "consumed"))
        self.assertGreater(tested, 0)


class LateralRuleTests(unittest.TestCase):
    def test_without_an_axis_the_chord_stands_off_the_vessel_tangent(self):
        # the lines run at 45 degrees to x: a chord straight across stands 90
        # degrees off their tangent but only 45 degrees off x, so about x only
        # the partners far enough along the other line remain
        nodes, tree = parallel_lines()
        c = math.sqrt(0.5)
        rotation = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
        nodes = nodes.copy()
        nodes[:3] = rotation @ nodes[:3]
        for seed in SEEDS:
            with self.subTest(seed=seed):
                for axis, reference, straight in ((None, rotation[:, 0], True), ([1, 0, 0], np.eye(3)[0], False)):
                    out, _, rungs, events, _, _ = run(nodes, tree, seed, axis=axis)
                    self.assertGreater(len(rungs), 0)
                    for first, last in rung_runs(out, nodes.shape[1]):
                        chord = out[:3, last] - out[:3, first]
                        length = np.linalg.norm(chord)
                        self.assertLessEqual(abs(chord @ reference), 0.5 * length)
                        if straight:
                            self.assertAlmostEqual(length, 5.0, delta=1e-13)
                        else:
                            self.assertGreater(length, 5.0 / math.cos(math.radians(15.0)))

    def test_the_tangent_runs_from_the_site_s_neighbour_before_to_its_neighbour_after(self):
        # the site vessel bends by 40 degrees at its only interior vertex: the
        # tangent there is x, while the outgoing direction lies 20 degrees above
        # x and the incoming one 20 degrees below. A chord 65 degrees off x
        # passes, though it stands 45 degrees off one of the other two; a chord
        # 55 degrees off x fails, though it stands 75 degrees off one of them
        for phi, joined in ((65.0, True), (-65.0, True), (55.0, False), (-55.0, False)):
            nodes, tree = bent(phi)
            for seed in SEEDS:
                with self.subTest(phi=phi, seed=seed):
                    _, _, rungs, events, _, _ = run(nodes, tree, seed, collision_margin=None)
                    if joined:
                        self.assertEqual([(rung["site"], rung["partner"]) for rung in rungs], [(1, 5)])
                        self.assertAlmostEqual(rungs[0]["angle_deg"], 65.0, delta=1e-13)
                    else:
                        self.assertEqual(rungs, [])
                        self.assertEqual(events["rung_angle_skipped"], 2)

    def test_the_angle_counter_counts_the_partners_the_angle_alone_refuses(self):
        # a brute-force reading on the parallel lines: each site that reaches
        # the search counts the points of the other line within 8 d, not yet a
        # rung end, that stand less than 60 degrees off x; the points of its own
        # line, all along x, are excluded before the angle and not counted
        nodes, tree = parallel_lines()
        runs = graph.polylines(nodes)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, sites, _ = run(nodes, tree, seed)
                rng = np.random.default_rng([seed, 5])
                for _ in runs:
                    position = rng.exponential(20.0)
                    while position <= 400.0:
                        position += rng.exponential(20.0)
                placed, ends, count = iter(rungs), set(), 0
                for o in rng.permutation(len(sites)).tolist():
                    record = sites[o]
                    if record["outcome"] in ("near_junction", "consumed"):
                        continue
                    other = runs[1 - record["unit"]][1:-1]
                    chord = np.linalg.norm(nodes[:3, other].T - nodes[:3, record["column"]], axis=1)
                    along = np.abs(nodes[0, other] - nodes[0, record["column"]])
                    count += int(np.count_nonzero((chord <= 8.0) & ~np.isin(other, list(ends)) & (along > 0.5 * chord)))
                    if record["outcome"] == "bridge":
                        rung = next(placed)
                        ends.update((rung["site"], rung["partner"]))
                self.assertGreater(count, 0)
                self.assertEqual(events["rung_angle_skipped"], count)

    def test_an_axis_replaces_the_tangent(self):
        nodes, tree = parallel_lines()
        for seed in SEEDS:
            with self.subTest(seed=seed):
                tangent = run(nodes, tree, seed)
                # the lines' own tangent is x exactly, so axis x changes nothing
                along_x = run(nodes, tree, seed, axis=[1, 0, 0])
                np.testing.assert_array_equal(along_x[0], tangent[0])
                self.assertEqual(along_x[2], tangent[2])
                # every chord runs along y, so axis y leaves no partner and axis z every one
                along_y = run(nodes, tree, seed, axis=[0, 1, 0])
                self.assertEqual(along_y[2], [])
                self.assertEqual(along_y[3]["rung_no_partner"],
                                 along_y[3]["rung_sites"] - along_y[3]["rung_sites_consumed"])
                # each of those sites saw up to 63 points across within 8 d, every one refused by the angle
                self.assertGreaterEqual(along_y[3]["rung_angle_skipped"], along_y[3]["rung_no_partner"])
                self.assertLessEqual(along_y[3]["rung_angle_skipped"], 63 * along_y[3]["rung_no_partner"])
                self.assertEqual(tangent[3]["rung_angle_skipped"], along_x[3]["rung_angle_skipped"])
                along_z = run(nodes, tree, seed, axis=[0, 0, 5])
                self.assertEqual(along_z[2], tangent[2])


class RankingTests(unittest.TestCase):
    """
    Without a margin nothing collides, so every rung goes to its site's
    first-ranked partner, which a brute-force reading of the rule finds from
    the rung ends placed before it.
    """

    @staticmethod
    def ranked(nodes, site, placed):
        """The (chord, column) of every partner of `site` on straight lines along x, d 1, without a margin, in order."""
        runs = graph.polylines(nodes)
        own = next(run for run in runs if site in run)
        found = []
        for run in runs:
            if run is own:
                continue
            interior = run[1:-1]
            chord = np.linalg.norm(nodes[:3, interior].T - nodes[:3, site], axis=1)
            ends = np.array([e for e in placed if e in run], dtype=np.int64)
            zoned = np.zeros(interior.size, dtype=bool)
            if ends.size:
                zoned = np.min(np.abs(nodes[0, interior][:, None] - nodes[0, ends][None, :]), axis=1) < KIN_REACH
            keep = ((chord <= 8.0) & (chord > 1.0) & ~np.isin(interior, list(placed)) & ~zoned
                    & (np.abs(nodes[0, interior] - nodes[0, site]) <= math.cos(math.radians(60.0)) * chord))
            found.extend((float(chord[j]), int(interior[j])) for j in np.flatnonzero(keep))
        return sorted(found)

    def assert_ranked(self, specs):
        nodes, tree = lines(*specs)
        ties = 0
        for seed in SEEDS:
            with self.subTest(specs=specs, seed=seed):
                _, _, rungs, _, _, _ = run(nodes, tree, seed, collision_margin=None)
                self.assertGreater(len(rungs), 0)
                placed = set()
                for rung in rungs:
                    ranked = self.ranked(nodes, rung["site"], placed)
                    self.assertEqual((rung["chord_um"], rung["partner"]), ranked[0])
                    ties += int(len(ranked) > 1 and ranked[1][0] == ranked[0][0])
                    placed.update((rung["site"], rung["partner"]))
        return ties

    def test_partners_rank_by_chord_then_by_column(self):
        # equally near lines on both sides of line 0, in either archive order,
        # so its sites meet ties that only the column breaks
        for specs in ([(-3.0, 1.0, 3), (0.0, 1.0, 0), (3.0, 1.0, 1)], [(3.0, 1.0, 1), (0.0, 1.0, 0), (-3.0, 1.0, 3)]):
            self.assertGreater(self.assert_ranked(specs), 0)
        # and an unequal pair, the nearer of which comes later in the archive
        self.assert_ranked([(-4.0, 1.0, 3), (0.0, 1.0, 0), (3.0, 1.0, 1)])
        self.assert_ranked([(0.0, 1.0, 0), (5.0, 1.0, 1)])


class SeparationTests(unittest.TestCase):
    def test_vessels_that_meet_are_two_apart(self):
        # the two daughters of the fork share their junction: two segments
        # apart, so the default of 2 joins them and 3 excludes them as kin
        nodes, tree = fork()
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, _, _ = run(nodes, tree, seed)
                self.assertGreater(len(rungs), 0)
                self.assertEqual(events["rung_kin_skipped"], 0)
                for rung in rungs:
                    # every rung joins the two daughters, one on each side of the parent's line
                    self.assertLess(nodes[1, rung["site"]] * nodes[1, rung["partner"]], 0.0)
                _, _, rungs, events, _, _ = run(nodes, tree, seed, min_separation=3)
                self.assertEqual(rungs, [])
                self.assertGreater(events["rung_kin_skipped"], 0)

    def test_a_vessel_folded_on_itself_gets_no_rung_across_the_fold(self):
        nodes, tree = u_bend(stubs=False)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, _, _ = run(nodes, tree, seed)
                self.assertEqual(rungs, [])
                self.assertGreater(events["rung_no_partner"], 0)
                self.assertEqual(events["rung_kin_skipped"], 0)

    def test_vessels_joined_through_a_third_are_three_apart(self):
        nodes, tree = u_bend(stubs=True)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                _, _, rungs, events, _, _ = run(nodes, tree, seed, min_separation=3)
                self.assertGreater(len(rungs), 0)
                for rung in rungs:
                    self.assertEqual(sorted((nodes[1, rung["site"]], nodes[1, rung["partner"]])), [0.0, 5.0])
                    self.assertGreater(min(nodes[0, rung["site"]], nodes[0, rung["partner"]]), 0.0)
                _, _, rungs, events, _, _ = run(nodes, tree, seed, min_separation=4)
                self.assertEqual(rungs, [])
                self.assertGreater(events["rung_kin_skipped"], 0)

    def test_two_trees_are_never_kin(self):
        nodes, tree = parallel_lines()
        for seed in SEEDS:
            with self.subTest(seed=seed):
                deep = run(nodes, tree, seed, min_separation=50)
                plain = run(nodes, tree, seed)
                np.testing.assert_array_equal(deep[0], plain[0])
                self.assertEqual(deep[3]["rung_kin_skipped"], 0)


class StageTests(unittest.TestCase):
    """The stage on its own: its draws, its input, its scale and its clearance."""

    @classmethod
    def setUpClass(cls):
        cls.lines = parallel_lines()
        # a walked tree of seed 2 grows 5444 columns with 10 anastomosis bridges
        # and gets 6 rungs at the default spacing and 8 at spacing 10 (measured)
        cls.grown = grown_tree(2)

    def test_the_stage_is_deterministic(self):
        nodes, tree = self.grown["nodes"], self.grown["tree"]
        first = run(nodes, tree, 2)
        second = run(nodes, tree, 2)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])
        self.assertEqual(first[4], second[4])
        self.assertEqual(first[5].bit_generator.state, second[5].bit_generator.state)
        self.assertGreater(len(first[2]), 0)

    def test_the_stage_draws_the_gaps_then_the_permutation_then_the_walks(self):
        # on the parallel lines, and on lines of d 1.5 sampled every 0.25, whose
        # gaps are drawn at 20 x 1.5 and whose walks step 0.25, not 0.2 d
        for (nodes, tree), diameter in ((self.lines, 1.0), (wide_lines(), 1.5)):
            step = float(np.median(np.diff(nodes[0, graph.polylines(nodes)[0]])))
            for seed in SEEDS:
                with self.subTest(diameter=diameter, seed=seed):
                    out, _, rungs, _, sites, _ = run(nodes, tree, seed)
                    rng = np.random.default_rng([seed, 5])
                    expected = []
                    for unit in (0, 1):                     # each line is one vessel, in archive order
                        position = rng.exponential(20.0 * diameter)
                        while position <= 400.0:
                            expected.append((unit, position))
                            position += rng.exponential(20.0 * diameter)
                    self.assertEqual([(record["unit"], record["arc"]) for record in sites], expected)
                    order = rng.permutation(len(sites))
                    # nothing is placed before the first site's turn, so its rung comes first
                    rung = rungs[0]
                    self.assertEqual(sites[order[0]]["column"], rung["site"])
                    for _ in range(rung["redraws"] + 1):
                        path = bridge_path(nodes[:3, rung["site"]], nodes[:3, rung["partner"]], step, rng=rng,
                                           persistence=10.0, diameter=diameter)
                    first, last = rung_runs(out, nodes.shape[1])[0]
                    np.testing.assert_array_equal(out[:3, first:last + 1], path.T)

    def test_the_input_is_left_alone_and_no_existing_column_moves(self):
        nodes, tree = self.grown["nodes"], self.grown["tree"]
        nodes_before, tree_before = nodes.copy(), tree.copy()
        state = (random.getstate(), np.random.get_state())
        out, out_tree, rungs, _, _, _ = run(nodes, tree, 2)
        self.assertGreater(len(rungs), 0)
        np.testing.assert_array_equal(nodes, nodes_before)
        np.testing.assert_array_equal(tree, tree_before)
        self.assertEqual(random.getstate(), state[0])
        after = np.random.get_state()
        self.assertEqual(after[0], state[1][0])
        np.testing.assert_array_equal(after[1], state[1][1])
        self.assertEqual(after[2:], state[1][2:])
        n = nodes.shape[1]
        np.testing.assert_array_equal(out[:, :n], nodes)
        np.testing.assert_array_equal(out_tree[:n], tree)
        self.assertTrue(np.all(out_tree[n:][np.isfinite(out[0, n:])] == BRIDGE))
        # roots, the first column of each tree label, stay where they were
        for label in (0, 1):
            self.assertEqual(np.flatnonzero(out_tree == label)[:1].tolist(), np.flatnonzero(tree == label)[:1].tolist())

    def test_a_network_without_capillaries_draws_nothing(self):
        nodes, tree = parallel_lines(diameter=2.0)       # median diameter not below 2 d_min
        out, out_tree, rungs, events, sites, rng = run(nodes, tree, 1)
        self.assertIs(out, nodes)
        np.testing.assert_array_equal(out_tree, tree)
        self.assertEqual((rungs, sites), ([], []))
        self.assertEqual(set(events.values()), {0})
        self.assertEqual(rng.bit_generator.state, np.random.default_rng([1, 5]).bit_generator.state)

    def test_the_stage_scales_with_the_unit(self):
        cases = [(self.lines, seed) for seed in SEEDS] + [((self.grown["nodes"], self.grown["tree"]), 2)]
        for (nodes, tree), seed in cases:
            base = run(nodes, tree, seed)
            self.assertGreater(len(base[2]), 0)
            for scale in (0.5, 2.0):
                with self.subTest(columns=nodes.shape[1], seed=seed, scale=scale):
                    scaled = run(scale * nodes, tree, seed, d_min=scale, collision_margin=scale * MARGIN,
                                 tol=scale * graph.DEFAULT_TOL)
                    np.testing.assert_allclose(scaled[0], scale * base[0], rtol=0.0, atol=SCALE_ATOL)
                    np.testing.assert_array_equal(scaled[1], base[1])
                    self.assertEqual(scaled[3], base[3])
                    self.assertEqual([r["outcome"] for r in scaled[4]], [r["outcome"] for r in base[4]])
                    self.assertEqual(len(scaled[2]), len(base[2]))
                    for mine, theirs in zip(scaled[2], base[2]):
                        self.assertEqual((mine["site"], mine["partner"]), (theirs["site"], theirs["partner"]))
                        self.assertEqual(mine["chord_um"], scale * theirs["chord_um"])

    def test_rungs_add_no_clearance_violation_to_a_grown_tree(self):
        nodes, tree = self.grown["nodes"], self.grown["tree"]
        self.assertEqual(clearance(nodes, margin=MARGIN)["violations"], 0)
        vertex = graph.vertex_diameter(nodes, graph.canonical_columns(nodes, graph.DEFAULT_TOL))
        for spacing in (20.0, 10.0):
            with self.subTest(spacing=spacing):
                out, _, rungs, events, _, _ = run(nodes, tree, 2, spacing=spacing)
                self.assertGreater(len(rungs), 0)
                self.assertGreater(events["rung_bridge_redraws"], 0)
                self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
                self.assertEqual(clearance(out, margin=MARGIN)["violations"], 0)
                # each rung is as wide as the narrower of the two vertices it joins, which mostly differ
                for (first, last), rung in zip(rung_runs(out, nodes.shape[1]), rungs):
                    self.assertEqual(rung["diameter_um"], min(vertex[rung["site"]], vertex[rung["partner"]]))
                    self.assertTrue(np.all(out[3, first:last + 1] == rung["diameter_um"]))
                self.assertGreater(sum(vertex[r["site"]] != vertex[r["partner"]] for r in rungs), 0)

    def test_every_check_gives_the_verdict_of_anastomosis_on_its_bridges(self):
        # before the first rung the two checks see the same network, and every
        # check the stage makes then must agree with anastomosis' own rule
        nodes, tree = self.grown["nodes"], self.grown["tree"]
        reference = anastomosis._Network(nodes, tree, graph.DEFAULT_TOL)
        vertices = np.array(sorted(reference.degree), dtype=np.int64)
        index = make_index("grid", cell_size=max(reference.diameter.values()))
        index.add(nodes[:3, vertices].T, np.array([reference.diameter[int(v)] for v in vertices]) / 2.0, vertices)
        verdicts = []
        check = connections._collides

        def both(net, obstacles, path, radius, site, partner, margin):
            mine = check(net, obstacles, path, radius, site, partner, margin)
            if obstacles.rungs is None:
                verdicts.append((mine, anastomosis._bridge_collides(reference, index, path, radius, site, partner,
                                                                    margin)))
            return mine

        with mock.patch.object(connections, "_collides", side_effect=both):
            for spacing in (20.0, 10.0):
                run(nodes, tree, 2, spacing=spacing)
        self.assertGreater(len(verdicts), 20)
        self.assertIn((False, False), verdicts)
        self.assertTrue(all(mine == theirs for mine, theirs in verdicts))

    def test_the_stage_makes_each_placed_rung_an_obstacle(self):
        # with a margin, the interior of every rung joins the obstacles at the
        # rung's radius as it is placed, so the checks after it see it
        cases = [(self.lines, seed, {}) for seed in SEEDS]
        cases.append(((self.grown["nodes"], self.grown["tree"]), 2, {"spacing": 10.0}))
        original = connections._Obstacles.add_rung
        for (nodes, tree), seed, settings in cases:
            with self.subTest(columns=nodes.shape[1], seed=seed):
                calls = []

                def record(obstacles, points, radius):
                    calls.append((np.array(points), radius))
                    return original(obstacles, points, radius)

                with mock.patch.object(connections._Obstacles, "add_rung", autospec=True, side_effect=record):
                    out, _, rungs, _, _, _ = run(nodes, tree, seed, **settings)
                self.assertGreater(len(rungs), 0)
                self.assertEqual(len(calls), len(rungs))
                for (first, last), rung, (points, radius) in zip(rung_runs(out, nodes.shape[1]), rungs, calls):
                    np.testing.assert_array_equal(points, out[:3, first + 1:last].T)
                    self.assertEqual(radius, rung["diameter_um"] / 2.0)

    def test_a_placed_rung_is_an_obstacle_and_never_a_partner(self):
        nodes, tree = self.lines
        net = connections._Vessels(nodes, tree, graph.DEFAULT_TOL, 1.0, 2.0)
        obstacles = connections._Obstacles("grid", nodes[:3, net.vertices].T, net.diameter[net.vertices] / 2.0,
                                           net.vertices, MARGIN, 1.0)
        site, partner = 500, 2002 + 500                 # straight across at x 100
        crossing = bridge_path(nodes[:3, site], nodes[:3, partner], 0.2)
        self.assertFalse(connections._collides(net, obstacles, crossing, 0.5, site, partner, MARGIN))
        placed = bridge_path(nodes[:3, 490], nodes[:3, 2002 + 510], 0.2)     # from x 98 to x 102
        obstacles.add_rung(placed[1:-1], 0.5)
        self.assertTrue(connections._collides(net, obstacles, crossing, 0.5, site, partner, MARGIN))
        out, _, rungs, _, _, _ = run(nodes, tree, 1)
        self.assertTrue(all(rung["partner"] < nodes.shape[1] for rung in rungs))

    def test_a_rung_that_comes_back_alongside_itself_collides(self):
        # a U of two legs `apart` and `height` tall: facing points 1.5 apart are
        # closer than 2 r + margin = 2, and the longest arc between two such
        # points of the interior is 2 (height - 0.2) + 1.5, which collides once
        # it exceeds 2 (2 r + margin) = 4: at heights 2.8 (6.7) and 2.0 (5.1),
        # not at 1.2 (3.5); legs 3 apart are never that close
        nodes, tree = self.lines
        net = connections._Vessels(nodes, tree, graph.DEFAULT_TOL, 1.0, 2.0)
        obstacles = connections._Obstacles("grid", nodes[:3, net.vertices].T, net.diameter[net.vertices] / 2.0,
                                           net.vertices, MARGIN, 1.0)
        site = 500                                       # x 100 on the first line
        for apart, height, collides in ((1.5, 2.8, True), (3.0, 2.8, False), (1.5, 2.0, True), (1.5, 1.2, False)):
            with self.subTest(apart=apart, height=height):
                partner = site + int(round(apart / 0.2))
                corners = [(100.0, 0.0, 0.0), (100.0, height, 0.0), (100.0 + apart, height, 0.0),
                           (100.0 + apart, 0.0, 0.0)]
                path = polyline(corners)[:3].T
                path[0], path[-1] = nodes[:3, site], nodes[:3, partner]
                self.assertEqual(connections._collides(net, obstacles, path, 0.5, site, partner, MARGIN), collides)

    def test_a_column_s_own_diameter_excuses_as_describe_reads_it(self):
        # P, 1.5 d along the site's vessel, holds a second column of diameter 3:
        # its vertex is 3 wide but describe reads its first column's 1. A rung
        # leaving at 60 degrees passes P at clearance 0 where describe no longer
        # excuses the pair, while anastomosis' rule, which excuses with the
        # vertex's radius, lets it through
        h = 6.0 * math.sin(math.radians(60.0))
        site_line = np.array([[-1.0, 0.0, 1.5, 1.5, 3.0, 10.0], [0.0] * 6, [0.0] * 6, [1.0, 1.0, 1.0, 3.0, 1.0, 1.0]])
        partner_line = np.array([[0.0, 3.0, 6.0], [h] * 3, [0.0] * 3, [1.0] * 3])
        nodes = np.concatenate([site_line, np.full((4, 1), np.nan), partner_line], axis=1)
        tree = np.where(np.isnan(nodes[0]), -1, 0).astype(np.int8)
        site, partner = 1, 8
        path = bridge_path(nodes[:3, site], nodes[:3, partner], 0.2)
        net = connections._Vessels(nodes, tree, graph.DEFAULT_TOL, 1.0, 2.0)
        obstacles = connections._Obstacles("grid", nodes[:3, net.vertices].T, net.diameter[net.vertices] / 2.0,
                                           net.vertices, MARGIN, 1.0)
        self.assertTrue(connections._collides(net, obstacles, path, 0.5, site, partner, MARGIN))
        reference = anastomosis._Network(nodes, tree, graph.DEFAULT_TOL)
        vertices = np.array(sorted(reference.degree), dtype=np.int64)
        index = make_index("grid", cell_size=3.0)
        index.add(nodes[:3, vertices].T, np.array([reference.diameter[int(v)] for v in vertices]) / 2.0, vertices)
        self.assertFalse(anastomosis._bridge_collides(reference, index, path, 0.5, site, partner, MARGIN))
        rows = np.vstack([path.T, np.full(len(path), 1.0)])
        joined = np.concatenate([nodes, np.full((4, 1), np.nan), rows], axis=1)
        self.assertGreater(clearance(joined, margin=MARGIN)["violations"], 0)
        # with P a single column of diameter 1, anastomosis' rule refuses the same pass:
        # only the wider column made it excuse what describe counts
        plain = np.delete(nodes, 3, axis=1)
        reference = anastomosis._Network(plain, np.delete(tree, 3), graph.DEFAULT_TOL)
        vertices = np.array(sorted(reference.degree), dtype=np.int64)
        index = make_index("grid", cell_size=3.0)
        index.add(plain[:3, vertices].T, np.array([reference.diameter[int(v)] for v in vertices]) / 2.0, vertices)
        self.assertTrue(anastomosis._bridge_collides(reference, index, path, 0.5, site, partner - 1, MARGIN))

    def test_without_a_margin_rungs_go_unchecked(self):
        nodes, tree = self.lines
        out, _, rungs, events, _, _ = run(nodes, tree, 1, collision_margin=None)
        self.assertGreater(len(rungs), 0)
        self.assertEqual(events["rung_bridge_redraws"], 0)
        self.assertTrue(all(rung["redraws"] == 0 for rung in rungs))


class ValidationTests(unittest.TestCase):
    def test_the_axis_comes_back_as_a_unit_list(self):
        self.assertIsNone(check_settings())
        self.assertEqual(check_settings(axis=[0, 2, 0]), [0.0, 1.0, 0.0])
        self.assertEqual(check_settings(axis=np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])
        unit = [0.6, 0.0, 0.8]
        self.assertEqual(check_settings(axis=unit), unit)
        self.assertEqual(check_settings(d_min=1.0, below=2, spacing=40.0, radius=8, lateral_deg=0,
                                        min_separation=np.int64(3), max_candidates=1, attempts=0,
                                        collision_margin=0, persistence=10), None)
        # the spacing's floor is one diameter, and each index kind is accepted
        for kind in ("auto", "grid", "kdtree"):
            self.assertIsNone(check_settings(spacing=1, index_kind=kind))

    def test_malformed_settings_are_refused_before_any_draw(self):
        nodes, tree = parallel_lines()
        refused = [{"d_min": 0.0}, {"d_min": -1.0}, {"d_min": float("nan")}, {"d_min": True},
                   {"below": 0.0}, {"below": float("inf")}, {"spacing": 0.0}, {"spacing": -20.0},
                   {"spacing": "20"}, {"radius": 0.0}, {"lateral_deg": -1.0}, {"lateral_deg": 90.5},
                   {"lateral_deg": float("nan")}, {"axis": [0, 0, 0]}, {"axis": [1, 0]}, {"axis": [1, float("nan"), 0]},
                   {"axis": "x"}, {"min_separation": 0}, {"min_separation": 2.0}, {"min_separation": True},
                   {"max_candidates": 0}, {"max_candidates": 5.0}, {"attempts": -1}, {"attempts": 1.5},
                   {"collision_margin": -0.5}, {"collision_margin": float("inf")}, {"persistence": 0.0},
                   {"persistence": -10.0}, {"spacing": 0.999}, {"spacing": 1e-9}, {"index_kind": "bogus"},
                   {"index_kind": None}, {"index_kind": ["grid"]}]
        for bad in refused:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    check_settings(**bad)
                rng = np.random.default_rng([1, 5])
                with self.assertRaises(ValueError):
                    cross_connect(nodes, tree, rng, **dict(SETTINGS, **bad))
                self.assertEqual(rng.bit_generator.state, np.random.default_rng([1, 5]).bit_generator.state)
        # a network without capillaries never builds an index, and is refused a bad kind all the same
        for bad_call in (lambda rng: cross_connect(nodes, tree, rng, d_min=None),
                         lambda rng: cross_connect(nodes, tree[:-1], rng, d_min=1.0),
                         lambda rng: cross_connect(nodes[:3], tree, rng, d_min=1.0),
                         lambda rng: cross_connect(*parallel_lines(diameter=2.0), rng, d_min=1.0, index_kind="bogus")):
            rng = np.random.default_rng([1, 5])
            with self.assertRaises(ValueError):
                bad_call(rng)
            self.assertEqual(rng.bit_generator.state, np.random.default_rng([1, 5]).bit_generator.state)

    def test_the_outcomes_and_counters_are_append_only(self):
        self.assertEqual(RUNG_OUTCOMES, ("bridge", "near_junction", "consumed", "no_partner", "collision_failed"))
        self.assertEqual(EVENT_KEYS, ("rung_sites", "rung_sites_near_junction", "rung_sites_consumed",
                                      "rung_no_partner", "rung_collision_failed", "rung_bridges",
                                      "rung_bridges_cross_tree", "rung_kin_skipped", "rung_angle_skipped",
                                      "rung_bridge_redraws"))

    def test_a_missing_tree_labels_every_column_zero(self):
        nodes, tree = parallel_lines()
        labelled = run(nodes, np.where(tree >= 0, 0, -1).astype(np.int8), 1)
        unlabelled = run(nodes, None, 1)
        np.testing.assert_array_equal(unlabelled[0], labelled[0])
        np.testing.assert_array_equal(unlabelled[1], labelled[1])
        self.assertEqual(unlabelled[3]["rung_bridges_cross_tree"], 0)


class DevelopmentMeasurementTests(unittest.TestCase):
    @SLOW
    def test_the_parallel_lines_and_grown_trees_over_ten_seeds(self):
        nodes, tree = parallel_lines()
        for seed, recorded in zip(TEN_SEEDS, KS_P_TEN_SEEDS):
            out, _, rungs, events, sites, _ = run(nodes, tree, seed)
            chords = [np.linalg.norm(out[:3, last] - out[:3, first]) for first, last in rung_runs(out, nodes.shape[1])]
            p = ks_exponential(site_gaps(nodes, sites), 20.0)[1]
            violations = clearance(out, margin=MARGIN)["violations"]
            print(f"lines seed {seed}: sites {len(sites)}, rungs {len(rungs)}, "
                  f"outcomes {[events[key] for key in PARTITION]}, redraws {events['rung_bridge_redraws']}, "
                  f"chord {min(chords):.6f}-{max(chords):.6f}, min angle {min(r['angle_deg'] for r in rungs):.4f}, "
                  f"KS p {p:.3f}, violations {violations}")
            self.assertAlmostEqual(p, recorded, delta=0.005)
            self.assertLessEqual(max(abs(c - 5.0) for c in chords), 0.2)
            self.assertGreaterEqual(min(r["angle_deg"] for r in rungs), 60.0)
            self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
            self.assertEqual(violations, 0)
            for scale in (0.5, 2.0):
                scaled = run(scale * nodes, tree, seed, d_min=scale, collision_margin=scale * MARGIN,
                             tol=scale * graph.DEFAULT_TOL)
                self.assertEqual(float(np.nanmax(np.abs(scaled[0] - scale * out))), SCALE_ATOL)
        for seed in TEN_SEEDS:
            grown = grown_tree(seed)
            out, _, rungs, events, _, _ = run(grown["nodes"], grown["tree"], seed)
            violations = clearance(out, margin=MARGIN)["violations"]
            print(f"grown tree seed {seed}: columns {grown['nodes'].shape[1]}, rungs {len(rungs)}, "
                  f"outcomes {[events[key] for key in PARTITION]}, violations {violations}")
            self.assertEqual(events["rung_sites"], sum(events[key] for key in PARTITION))
            self.assertEqual(violations, 0)
            for scale in (0.5, 2.0):
                scaled = run(scale * grown["nodes"], grown["tree"], seed, d_min=scale,
                             collision_margin=scale * MARGIN, tol=scale * graph.DEFAULT_TOL)
                self.assertEqual(float(np.nanmax(np.abs(scaled[0] - scale * out))), SCALE_ATOL)

    @SLOW
    def test_the_tolerances_over_ten_seeds(self):
        nodes, tree = wide_lines()
        snap, p_values = 0.0, []
        for seed in TEN_SEEDS:
            sites = run(nodes, tree, seed)[4]
            snap = max(snap, max(abs(nodes[0, record["column"]] - record["arc"]) for record in sites))
            p_values.append(ks_exponential(site_gaps(nodes, sites), 30.0)[1])
        print(f"d 1.5 lines: snap {snap:.4f}, KS p against Exp(30) {' '.join(f'{p:.2f}' for p in p_values)}")
        self.assertLessEqual(snap, 0.125)
        self.assertGreater(min(p_values), 0.01)
        try:
            from scipy import stats
        except ImportError:
            stats = None
        if stats is not None:
            worst_d = worst_p = 0.0
            for (lines_nodes, lines_tree), mean in ((parallel_lines(), 20.0), (wide_lines(), 30.0)):
                for seed in TEN_SEEDS:
                    gaps = site_gaps(lines_nodes, run(lines_nodes, lines_tree, seed)[4])
                    d, p = ks_exponential(gaps, mean)
                    reference = stats.kstest(gaps, "expon", args=(0.0, mean), method="exact")
                    worst_d = max(worst_d, abs(d - reference.statistic))
                    worst_p = max(worst_p, abs(p - reference.pvalue))
            print(f"KS against scipy: D within {worst_d:.2g}, p within {worst_p:.2g}")
            self.assertLessEqual(worst_d, 1e-15)
            self.assertLessEqual(worst_p, 1e-14)
        c = math.sqrt(0.5)
        rotation = np.array([[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]])
        rotated, rotated_tree = parallel_lines()
        rotated = rotated.copy()
        rotated[:3] = rotation @ rotated[:3]
        chord_error, margins, excess = 0.0, [], np.inf
        for seed in TEN_SEEDS:
            for axis, reference in ((None, rotation[:, 0]), ([1, 0, 0], np.eye(3)[0])):
                out = run(rotated, rotated_tree, seed, axis=axis)[0]
                for first, last in rung_runs(out, rotated.shape[1]):
                    chord = out[:3, last] - out[:3, first]
                    length = float(np.linalg.norm(chord))
                    margins.append(0.5 * length - abs(float(chord @ reference)))
                    if axis is None:
                        chord_error = max(chord_error, abs(length - 5.0))
                    else:
                        excess = min(excess, length - 5.0 / math.cos(math.radians(15.0)))
        print(f"rotated lines: chord within {chord_error:.2g} of 5, {min(margins):.3f} inside the angle bound, "
              f"{excess:.3f} longer than 5 / cos 15")
        self.assertLessEqual(chord_error, 1e-13)
        self.assertGreaterEqual(min(margins), 0.0)
        self.assertGreater(excess, 0.0)
        angle_error = 0.0
        for phi in (65.0, -65.0):
            bent_nodes, bent_tree = bent(phi)
            for seed in TEN_SEEDS:
                rungs = run(bent_nodes, bent_tree, seed, collision_margin=None)[2]
                angle_error = max(angle_error, abs(rungs[0]["angle_deg"] - 65.0))
        print(f"bent vessel: angle within {angle_error:.2g} of 65")
        self.assertLessEqual(angle_error, 1e-13)


if __name__ == "__main__":
    unittest.main()
