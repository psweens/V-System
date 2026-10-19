"""
Tests of the anti-aliased voxeliser, computeVoxel.rasterise_segments(...,
antialias=True), and of the default it leaves unchanged.

Default. With antialias left False, rasterise_segments and process_network
draw bit for bit what the 3.9.0 voxeliser draws. The 3.9.0 computeVoxel is
byte-identical to tests/fixtures/reference_code_3_8/computeVoxel.py (the
module has not changed since 3.2; its sha256 is recorded with the 3.8
hashes and checked here); both run in this process on random polylines and
on the network fixtures.

Anti-aliased. Measured as the CHANGELOG states the acceptance criteria:
  - straight tubes of diameter d = 1.5-8 voxels, axis-aligned, along face
    and body diagonals and in random directions, at random sub-voxel
    offsets: the cross-section of a tube is the number of its voxels in a
    lattice plane across its dominant axis k, times |u_k|, over the planes
    of its central part. For d >= 2 every tube's mean is within 10 % of
    pi r^2; for d = 2-3 the mean coefficient of variation along the tubes is
    at most 0.10;
  - short necks: thin voxels are those in no all-foreground 2 x 2 x 2 cube; a
    neck is a 26-connected run of thin voxels 26-adjacent to two or more
    26-components of the 2 x 2 x 2 opening, short if it has at most 5 voxels.
    Two-voxel tubes, straight at oblique angles and tortuous, have at most
    0.4 short necks per mm at 2 um voxels (3.9 has several on the tortuous
    ones);
  - topology: the network fixtures have as many 26-components as under 3.9;
    two vessels 2 voxels or more apart surface to surface never touch, and
    those more than sqrt(3) apart touch only where 3.9 has them touch;
  - tapers and junctions render as one face-connected piece without an
    enclosed cavity or a plane across a taper far below its cross-section;
  - every voxel drawn lies within half a voxel of a capsule, on a chain or
    between six drawn neighbours; a polyline thinner than 2.45 voxels holds
    no 2 x 2 x 2 cube; the result is deterministic and independent of the
    order of the polylines.

The comparison of 3.9 with 3.10 that the CHANGELOG reports (area bias, CV,
short necks per mm, components, runtime on a foam and a capillary_bed
network grown at d_min 4 um and rendered at 2 um in 256^3) is slow and runs
only with VSYSTEM_SLOW_TESTS=1; it prints its tables and writes them as JSON
to the path in VSYSTEM_BENCHMARK_JSON when that is set.

Run from the repository root with
    python -m unittest tests.test_antialias
    VSYSTEM_SLOW_TESTS=1 python -m unittest tests.test_antialias.AntialiasBenchmark
The neck and component measures use scipy.ndimage and are skipped where scipy
is not installed.
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import computeVoxel  # noqa: E402
from computeVoxel import process_network, rasterise_segments  # noqa: E402
from spatial import have_scipy  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
REFERENCE = os.path.join(FIXTURES, "reference_code_3_8", "computeVoxel.py")
ARCHIVES = ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz", "Lnet_i9_s3.npz")
SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmark")
NEEDS_SCIPY = unittest.skipUnless(have_scipy(), "scipy not installed")
UM_PER_VOXEL = 2.0          # the neck rates are per mm of centre line at 2 um voxels
SHORT = 5                   # a short neck has at most this many voxels


def reference_voxeliser():
    """The 3.9.0 computeVoxel, loaded under a name of its own."""
    with open(REFERENCE, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(FIXTURES, "reference_hashes_3_8.json")) as handle:
        recorded = json.load(handle)["_reference_files"]["computeVoxel.py"]
    if digest != recorded:
        raise AssertionError(f"{REFERENCE} is not the released module: sha256 {digest}")
    spec = importlib.util.spec_from_file_location("computeVoxel_3_9", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def antialiased(points, radii, shape, connect=True):
    return rasterise_segments(points, radii, shape, connect=connect, antialias=True)


def default(points, radii, shape, connect=True):
    return rasterise_segments(points, radii, shape, connect=connect)


def directions(n_random, seed):
    """Axis-aligned, face- and body-diagonal and random unit vectors, named."""
    named = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "xy": (1, 1, 0), "xz": (1, 0, -1),
             "yz": (0, 1, 1), "xyz": (1, 1, 1), "x-yz": (1, -1, 1)}
    out = [(name, np.array(v, float) / np.linalg.norm(v)) for name, v in named.items()]
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        v = rng.normal(size=3)
        out.append((f"random{i}", v / np.linalg.norm(v)))
    return out


def tube(u, d, length, offset, margin=6):
    """A straight tube through the volume's centre: points (3, 2), radii and the shape."""
    r = d / 2.0
    shape = tuple(int(math.ceil(e)) + 1 for e in np.abs(u) * length + 2 * (r + margin))
    centre = (np.array(shape) - 1) / 2.0 + np.asarray(offset)
    return np.stack([centre - u * length / 2, centre + u * length / 2], axis=1), np.array([r, r]), shape


def plane_areas(volume, points, u, trim):
    """The tube's cross-section in each lattice plane across its dominant axis, times |u_k|."""
    k = int(np.argmax(np.abs(u)))
    p0, p1 = points[:, 0], points[:, 1]
    counts = np.bincount(np.nonzero(volume)[k], minlength=volume.shape[k])
    length = float(np.linalg.norm(p1 - p0))
    areas = []
    for c in range(volume.shape[k]):
        t = (c - p0[k]) / u[k]
        if trim <= t <= length - trim:
            areas.append(counts[c] * abs(u[k]))
    return np.array(areas)


def curve(seed, d, steps=400, step=0.5, persistence=25.0):
    """A tortuous centre line (a persistent random walk): points, radii and shape."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    p = np.zeros(3)
    points = [p.copy()]
    for _ in range(steps):
        u = u + rng.normal(size=3) / math.sqrt(persistence)
        u /= np.linalg.norm(u)
        p = p + step * u
        points.append(p.copy())
    points = np.array(points).T
    points = points - points.min(axis=1)[:, None] + d / 2.0 + 6.0
    shape = tuple(int(math.ceil(v + d / 2.0 + 7.0)) for v in points.max(axis=1))
    return points, np.full(points.shape[1], d / 2.0), shape


def measure_tubes(render, d, n_random, offsets, length, seed=0):
    """(direction, area bias, CV along the tube) of straight tubes of diameter d."""
    rng = np.random.default_rng(seed + int(10 * d))
    rows = []
    for name, u in directions(n_random, seed):
        for _ in range(offsets):
            points, radii, shape = tube(u, d, length, rng.uniform(-0.5, 0.5, 3))
            areas = plane_areas(render(points, radii, shape), points, u, d / 2 + 3)
            rows.append((name, areas.mean() / (math.pi * d * d / 4) - 1, areas.std() / areas.mean()))
    return rows


def centre_line_length(points):
    return float(np.nansum(np.linalg.norm(np.diff(points, axis=1), axis=0)))


def cube_corners(volume):
    """The lowest corner of every all-foreground 2 x 2 x 2 cube."""
    v = volume.astype(bool)
    n = tuple(s - 1 for s in v.shape)
    full = np.ones(n, dtype=bool)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                full &= v[i:i + n[0], j:j + n[1], k:k + n[2]]
    return full


def opening(volume):
    """The voxels of all-foreground 2 x 2 x 2 cubes."""
    full = cube_corners(volume)
    out = np.zeros(volume.shape, dtype=bool)
    n = full.shape
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                out[i:i + n[0], j:j + n[1], k:k + n[2]] |= full
    return out


def short_necks(volume, mask=None):
    """The short necks of a volume, counting only those with a voxel in `mask`."""
    from scipy import ndimage
    v = volume.astype(bool)
    opened = opening(v)
    thin = v & ~opened
    s26 = np.ones((3, 3, 3), dtype=bool)
    olab, nopen = ndimage.label(opened, s26)
    tlab, nthin = ndimage.label(thin, s26)
    if nopen < 2 or nthin == 0:
        return 0
    # (thin run, opening component) for each pair of 26-adjacent voxels
    pad = np.pad(olab, 1)
    where = np.nonzero(thin)
    pairs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                near = pad[where[0] + 1 + dx, where[1] + 1 + dy, where[2] + 1 + dz]
                hit = near > 0
                pairs.append(np.stack([tlab[where][hit], near[hit]], axis=1))
    pairs = np.unique(np.concatenate(pairs), axis=0)
    touched = np.bincount(pairs[:, 0], minlength=nthin + 1)
    size = np.bincount(tlab.ravel(), minlength=nthin + 1)
    neck = (touched >= 2) & (size <= SHORT)
    neck[0] = False
    if mask is not None:
        neck &= np.bincount(tlab[mask & thin], minlength=nthin + 1) > 0
    return int(neck.sum())


def components(volume, connectivity=26):
    from scipy import ndimage
    structure = np.ones((3, 3, 3), dtype=bool) if connectivity == 26 else None
    return ndimage.label(volume, structure)[1]


def load_fixture(name):
    return np.load(os.path.join(FIXTURES, name), allow_pickle=True)["nodes"]


def random_polylines(rng, shape, count):
    """NaN-separated polylines with repeats, sub-voxel to thick radii, tapers and clipping."""
    columns, radii = [], []
    for _ in range(count):
        n = int(rng.integers(1, 8))
        p = rng.uniform(-4.0, np.array(shape) + 4.0, size=(n, 3)).T
        r = rng.choice([0.0, 0.2, 0.6, 1.0, 1.4, 2.3, 4.0], size=n) * rng.uniform(0.8, 1.2, size=n)
        if n > 2 and rng.random() < 0.3:
            p[:, 1] = p[:, 0]                      # a repeated point with its own radius
        columns += [p, np.full((3, 1), np.nan)]
        radii += [r, [np.nan]]
    return np.hstack(columns), np.concatenate(radii)


class DefaultTests(unittest.TestCase):
    """antialias=False draws what 3.9 draws."""

    @classmethod
    def setUpClass(cls):
        cls.reference = reference_voxeliser()

    def test_the_default_draws_what_3_9_draws_on_random_polylines(self):
        rng = np.random.default_rng(39)
        for case in range(40):
            shape = tuple(int(s) for s in rng.integers(6, 30, size=3))
            points, radii = random_polylines(rng, shape, int(rng.integers(1, 6)))
            for connect in (True, False):
                with self.subTest(case=case, connect=connect):
                    expected = self.reference.rasterise_segments(points, radii, shape, connect=connect)
                    np.testing.assert_array_equal(rasterise_segments(points, radii, shape, connect=connect), expected)
                    np.testing.assert_array_equal(
                        rasterise_segments(points, radii, shape, connect=connect, antialias=False), expected)

    def test_process_network_draws_what_3_9_draws_on_the_fixtures(self):
        for name in ARCHIVES[:2]:
            nodes = load_fixture(name)
            for kwargs in ({"fit": "voxel_size", "voxel_size": 4.0}, {"fit": "isotropic", "clip_axes": (2,)},
                           {"fit": "stretch"}, {"fit": "isotropic", "connect": False}):
                with self.subTest(archive=name, **kwargs):
                    expected = self.reference.process_network(nodes, (96, 80, 48), **kwargs)
                    np.testing.assert_array_equal(process_network(nodes, (96, 80, 48), **kwargs), expected)
                    np.testing.assert_array_equal(process_network(nodes, (96, 80, 48), antialias=False, **kwargs),
                                                  expected)

    def test_antialias_must_be_a_bool(self):
        points, radii = np.array([[1.0, 5.0], [2.0, 3.0], [2.0, 4.0]]), np.array([1.0, 1.0])
        for value in (0.5, 1, "yes", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rasterise_segments(points, radii, (8, 8, 8), antialias=value)
        for value in (True, False, np.bool_(True)):
            self.assertEqual(rasterise_segments(points, radii, (8, 8, 8), antialias=value).dtype, np.uint8)


class StraightTubeTests(unittest.TestCase):
    """Cross-sectional area and its variation along straight tubes."""

    def measure(self, render, d):
        return measure_tubes(render, d, n_random=6, offsets=2, length=60.0)

    def test_every_tube_keeps_its_area_and_the_cross_section_is_steady(self):
        for d in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
            rows = self.measure(antialiased, d)
            bias = np.array([row[1] for row in rows])
            cv = np.array([row[2] for row in rows])
            with self.subTest(d=d):
                if d >= 2.0:
                    self.assertLessEqual(abs(bias.mean()), 0.10)
                    self.assertLessEqual(np.abs(bias).max(), 0.10, rows)
                if 2.0 <= d <= 3.0:
                    self.assertLessEqual(cv.mean(), 0.10)

    def test_the_default_does_not_keep_the_area_at_every_offset(self):
        # what the mode is for: 3.9's area swings with the sub-voxel offset
        for d in (2.0, 2.5):
            bias = np.array([row[1] for row in self.measure(default, d)])
            with self.subTest(d=d):
                self.assertGreater(np.abs(bias).max(), 0.15)

    def test_the_cross_section_is_steadier_than_the_default(self):
        for d in (2.5, 3.0):
            with self.subTest(d=d):
                aa = np.mean([row[2] for row in self.measure(antialiased, d)])
                old = np.mean([row[2] for row in self.measure(default, d)])
                self.assertLess(aa, 0.75 * old)


@NEEDS_SCIPY
class NeckTests(unittest.TestCase):
    """Two-voxel vessels without short necks."""

    def test_oblique_two_voxel_tubes_have_no_short_necks(self):
        rng = np.random.default_rng(7)
        necks, length = 0, 0.0
        for name, u in directions(8, 3)[3:]:
            points, radii, shape = tube(u, 2.0, 120.0, rng.uniform(-0.5, 0.5, 3))
            trim = 5.0
            index = np.indices(shape).reshape(3, -1).T
            t = (index - points[:, 0]) @ u
            mask = ((t >= trim) & (t <= 120.0 - trim)).reshape(shape)
            necks += short_necks(antialiased(points, radii, shape), mask)
            length += 120.0 - 2 * trim
        self.assertLessEqual(necks / (length * UM_PER_VOXEL / 1000.0), 0.4)

    def test_tortuous_two_voxel_vessels_have_no_short_necks(self):
        counts = {"antialiased": 0, "default": 0}
        length = 0.0
        for seed in range(4):
            points, radii, shape = curve(seed, 2.0)
            counts["antialiased"] += short_necks(antialiased(points, radii, shape))
            counts["default"] += short_necks(default(points, radii, shape))
            length += centre_line_length(points)
        per_mm = {k: v / (length * UM_PER_VOXEL / 1000.0) for k, v in counts.items()}
        self.assertLessEqual(per_mm["antialiased"], 0.4)
        self.assertGreater(per_mm["default"], 0.4)              # the test sees what the mode removes

    def test_a_polyline_thinner_than_the_cube_threshold_holds_no_cube(self):
        for seed in range(3):
            for d in (1.8, 2.0, 2.2, 2.4):
                points, radii, shape = curve(seed + 10, d, steps=200)
                with self.subTest(seed=seed, d=d):
                    self.assertFalse(np.any(cube_corners(antialiased(points, radii, shape))))


@NEEDS_SCIPY
class TopologyTests(unittest.TestCase):
    """Components, merges, holes and seams."""

    def pair(self, rng, d1, d2, gap):
        """Two polylines whose surfaces come within `gap` of each other, parallel or skew."""
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        w = np.cross(u, rng.normal(size=3))
        w /= np.linalg.norm(w)
        v = rng.normal(size=3) if rng.random() < 0.5 else u.copy()
        v -= (v @ w) * w
        v /= np.linalg.norm(v)
        separation = d1 / 2 + d2 / 2 + gap
        length = 30.0
        shape = (int(2 * length + 2 * separation + 12),) * 3
        centre = (np.array(shape) - 1) / 2.0 + rng.uniform(-0.5, 0.5, 3)
        n = int(rng.integers(30, 100))
        first = np.linspace(centre - u * length / 2, centre + u * length / 2, n + 1).T
        second = np.linspace(centre + w * separation - v * length / 2, centre + w * separation + v * length / 2,
                             n + 1).T
        points = np.hstack([first, np.full((3, 1), np.nan), second])
        radii = np.concatenate([np.full(n + 1, d1 / 2), [np.nan], np.full(n + 1, d2 / 2)])
        return points, radii, shape

    def test_vessels_two_voxels_apart_never_touch(self):
        rng = np.random.default_rng(2)
        for case in range(60):
            d1, d2 = rng.choice([1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 4.0], size=2)
            points, radii, shape = self.pair(rng, d1, d2, 2.0)
            with self.subTest(case=case, d1=d1, d2=d2):
                self.assertEqual(components(antialiased(points, radii, shape)), 2)

    def test_vessels_more_than_sqrt3_apart_touch_only_where_the_default_has_them_touch(self):
        rng = np.random.default_rng(3)
        for case in range(60):
            d1, d2 = rng.choice([1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 4.0], size=2)
            points, radii, shape = self.pair(rng, d1, d2, 1.74)
            with self.subTest(case=case, d1=d1, d2=d2):
                if components(antialiased(points, radii, shape)) < 2:
                    self.assertLess(components(default(points, radii, shape)), 2)

    def test_the_fixtures_have_the_components_they_have_by_default(self):
        # the whole network in the volume: the same number of 26-components
        for name in ARCHIVES:
            nodes = load_fixture(name)
            extent = np.nanmax(nodes[:3], axis=1) - np.nanmin(nodes[:3], axis=1)
            for voxel_size in (2.0, 4.0, 8.0):
                shape = tuple(int(e) for e in np.ceil(extent / voxel_size) + 8)
                with self.subTest(archive=name, voxel_size=voxel_size):
                    kwargs = {"fit": "voxel_size", "voxel_size": voxel_size}
                    self.assertEqual(components(process_network(nodes, shape, antialias=True, **kwargs)),
                                     components(process_network(nodes, shape, **kwargs)))
            for shape in ((64, 64, 32), (128, 128, 64), (512, 512, 140)):
                with self.subTest(archive=name, shape=shape):
                    self.assertEqual(components(process_network(nodes, shape, antialias=True)),
                                     components(process_network(nodes, shape)))

    def test_where_a_face_cuts_vessels_only_the_pieces_it_cuts_can_differ(self):
        # a vessel running just outside a slab shows its edge in the face plane,
        # which the default can break in two where the anti-aliased mode does not;
        # the pieces that touch no face are as many as by default
        from scipy import ndimage

        def inner_pieces(volume):
            labels, n = ndimage.label(volume, np.ones((3, 3, 3), dtype=bool))
            touching = set()
            for axis in range(3):
                touching.update(np.unique(np.take(labels, 0, axis=axis)).tolist())
                touching.update(np.unique(np.take(labels, -1, axis=axis)).tolist())
            return n - len(touching - {0})

        for name in ARCHIVES:
            nodes = load_fixture(name)
            for shape in ((128, 128, 40), (256, 256, 70), (320, 320, 90), (512, 512, 140)):
                with self.subTest(archive=name, shape=shape):
                    self.assertEqual(inner_pieces(process_network(nodes, shape, clip_axes=(2,), antialias=True)),
                                     inner_pieces(process_network(nodes, shape, clip_axes=(2,))))

    def test_tapers_and_junctions_have_no_holes_or_seams(self):
        from scipy import ndimage
        rng = np.random.default_rng(5)
        cases = []
        for r0, r1 in ((0.5, 2.5), (1.0, 3.0), (3.0, 1.0), (1.5, 4.0), (0.8, 1.2)):
            for _ in range(4):
                u = rng.normal(size=3)
                cases.append(("taper", r0, r1, u / np.linalg.norm(u)))
        for d in (2.0, 3.0, 4.0):
            for _ in range(4):
                cases.append(("junction", d, d, rng.normal(size=(3, 3))))
        for kind, r0, r1, g in cases:
            if kind == "taper":
                length = 40.0
                shape = tuple(int(math.ceil(abs(c) * length + 2 * max(r0, r1) + 12)) + 1 for c in g)
                centre = (np.array(shape) - 1) / 2.0 + rng.uniform(-0.5, 0.5, 3)
                # sampled like a network, every voxel or so
                t = np.linspace(-0.5, 0.5, 41)
                points = centre[:, None] + g[:, None] * t[None] * length
                radii = r0 + (r1 - r0) * (t + 0.5)
            else:
                arms = g / np.linalg.norm(g, axis=1, keepdims=True)
                length = 20.0
                shape = (int(2 * length + 2 * r0 + 14),) * 3
                centre = (np.array(shape) - 1) / 2.0 + rng.uniform(-0.5, 0.5, 3)
                columns, rr = [], []
                for arm in arms:              # three polylines leaving one shared vertex
                    columns += [centre[:, None] + arm[:, None] * np.linspace(0, length, 21)[None],
                                np.full((3, 1), np.nan)]
                    rr += [np.full(21, r0 / 2), [np.nan]]
                points, radii = np.hstack(columns), np.concatenate(rr)
            volume = antialiased(points, radii, shape).astype(bool)
            with self.subTest(kind=kind, r0=r0, r1=r1):
                self.assertEqual(ndimage.label(volume)[1], 1)
                self.assertEqual(int(ndimage.binary_fill_holes(volume).sum()), int(volume.sum()))
                if kind == "taper":
                    u = g
                    k = int(np.argmax(np.abs(u)))
                    counts = np.bincount(np.nonzero(volume)[k], minlength=shape[k])
                    p0 = points[:, 0]
                    total = float(np.linalg.norm(points[:, -1] - p0))
                    for c in range(shape[k]):
                        t = (c - p0[k]) / u[k]
                        r = r0 + (r1 - r0) * t / total
                        if max(r0, r1) + 2 <= t <= total - max(r0, r1) - 2 and r >= 1.0:
                            self.assertGreaterEqual(counts[c] * abs(u[k]), 0.6 * math.pi * r * r)


class PropertyTests(unittest.TestCase):
    """What every anti-aliased volume satisfies."""

    def test_every_voxel_lies_near_a_capsule_on_a_chain_or_between_drawn_neighbours(self):
        rng = np.random.default_rng(11)
        reference = reference_voxeliser()
        for case in range(12):
            shape = (24, 22, 20)
            points, radii = random_polylines(rng, shape, 4)
            volume = antialiased(points, radii, shape).astype(bool)
            # within 0.5 of a capsule: inside the capsules grown by half a voxel
            grown = reference.rasterise_segments(points, radii + 0.5, shape, connect=True).astype(bool)
            v = np.pad(volume, 1)
            enclosed = (v[:-2, 1:-1, 1:-1] & v[2:, 1:-1, 1:-1] & v[1:-1, :-2, 1:-1] & v[1:-1, 2:, 1:-1]
                        & v[1:-1, 1:-1, :-2] & v[1:-1, 1:-1, 2:])
            with self.subTest(case=case):
                self.assertFalse(np.any(volume & ~grown & ~enclosed))

    def test_the_chain_is_drawn_and_sub_voxel_vessels_render_as_by_default(self):
        rng = np.random.default_rng(12)
        for case in range(20):
            shape = (20, 20, 20)
            points = rng.uniform(2, 18, size=(3, 6))
            radii = rng.uniform(0.0, 0.3, size=6)
            with self.subTest(case=case):
                np.testing.assert_array_equal(antialiased(points, radii, shape), default(points, radii, shape))
            points, radii = random_polylines(rng, shape, 3)
            chain = np.zeros(shape, dtype=np.uint8)
            for i in range(points.shape[1] - 1):
                if not (np.isnan(points[0, i]) or np.isnan(points[0, i + 1])):
                    computeVoxel.rasterise_line(chain, points[:, i], points[:, i + 1])
            with self.subTest(case=case, chain=True):
                self.assertFalse(np.any(chain.astype(bool) & ~antialiased(points, radii, shape).astype(bool)))

    def test_deterministic_and_independent_of_the_order_of_polylines(self):
        rng = np.random.default_rng(13)
        shape = (40, 36, 30)
        points, radii = random_polylines(rng, shape, 12)
        first = antialiased(points, radii, shape)
        np.testing.assert_array_equal(antialiased(points, radii, shape), first)
        # the same polylines in reverse order, each keeping its own point order
        breaks = np.flatnonzero(np.isnan(points[0]))
        pieces = np.split(np.arange(points.shape[1]), breaks + 1)
        order = np.concatenate([piece for piece in pieces[::-1] if len(piece)])
        np.testing.assert_array_equal(antialiased(points[:, order], radii[order], shape), first)

    def test_reaches_the_volume_from_a_window_of_a_larger_network(self):
        # a network passed whole with a small volume draws only what lies in it
        nodes = load_fixture("Lnet_i5_s1.npz")
        full = process_network(nodes, (240, 210, 180), fit="voxel_size", voxel_size=2.0, antialias=True)
        points, radii = computeVoxel.fit_to_volume(nodes, (240, 210, 180), fit="voxel_size", voxel_size=2.0)
        window = antialiased(points - np.array([[90.0], [80.0], [60.0]]), radii, (64, 64, 64))
        crop = full[90:154, 80:144, 60:124]
        self.assertGreater(int(window.sum()), 0)
        self.assertLessEqual(int((window != crop).sum()), 0.01 * int(crop.sum()))


@SLOW
@NEEDS_SCIPY
class AntialiasBenchmark(unittest.TestCase):
    """3.9 against 3.10: the tables the CHANGELOG reports."""

    def test_benchmark(self):
        report = {"straight": {}, "curved": {}, "networks": {}, "fixtures": {}}
        renders = {"3.9": default, "3.10": antialiased}
        for d in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
            row = {}
            for label, render in renders.items():
                rows = measure_tubes(render, d, n_random=12, offsets=3, length=120.0)
                bias = np.array([r[1] for r in rows])
                cv = np.array([r[2] for r in rows])
                row[label] = {"bias_mean": float(bias.mean()), "bias_abs_max": float(np.abs(bias).max()),
                              "cv_mean": float(cv.mean()), "cv_max": float(cv.max())}
            for label, render in renders.items():
                necks, length = 0, 0.0
                rng = np.random.default_rng(int(100 * d))
                for name, u in directions(12, 0)[3:]:
                    points, radii, shape = tube(u, d, 120.0, rng.uniform(-0.5, 0.5, 3))
                    index = np.indices(shape).reshape(3, -1).T
                    t = (index - points[:, 0]) @ u
                    mask = ((t >= d / 2 + 4) & (t <= 120.0 - d / 2 - 4)).reshape(shape)
                    necks += short_necks(render(points, radii, shape), mask)
                    length += 120.0 - d - 8
                row[label]["short_necks_per_mm_oblique"] = necks / (length * UM_PER_VOXEL / 1000.0)
            report["straight"][str(d)] = row
        for d in (2.0, 2.2, 2.5):
            row = {}
            for label, render in renders.items():
                necks, length = 0, 0.0
                for seed in range(8):
                    points, radii, shape = curve(seed, d)
                    necks += short_necks(render(points, radii, shape))
                    length += centre_line_length(points)
                row[label] = necks / (length * UM_PER_VOXEL / 1000.0)
            report["curved"][str(d)] = row
        with tempfile.TemporaryDirectory() as out:
            for family in ("foam", "capillary_bed"):
                subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--count", "1", "--family", family,
                                "--d-min", "4", "--fit", "voxel_size", "--voxel-size", "2", "--volume", "256", "256",
                                "256", "--seed", "1", "--out", os.path.join(out, family)],
                               check=True, capture_output=True)
                stem = [n for n in os.listdir(os.path.join(out, family)) if n.endswith(".npz")][0]
                nodes = np.load(os.path.join(out, family, stem), allow_pickle=True)["nodes"]
                report["networks"][family] = self.network(nodes, (256, 256, 256), 2.0)
        for name in ARCHIVES:
            report["fixtures"][name] = self.network(load_fixture(name), (256, 256, 256), 2.0)
        print(json.dumps(report, indent=1))
        if os.environ.get("VSYSTEM_BENCHMARK_JSON"):
            with open(os.environ["VSYSTEM_BENCHMARK_JSON"], "w") as handle:
                json.dump(report, handle, indent=1)

    def network(self, nodes, shape, voxel_size):
        points, radii = computeVoxel.fit_to_volume(nodes, shape, fit="voxel_size", voxel_size=voxel_size)
        mid = 0.5 * (points[:, :-1] + points[:, 1:])
        inside = np.all((mid >= -0.5) & (mid < np.array(shape)[:, None] - 0.5), axis=0)
        length = float(np.nansum(np.linalg.norm(np.diff(points, axis=1), axis=0)[inside]))
        out = {"segments": int(np.count_nonzero(~np.isnan(np.diff(points[0])))), "centre_line_mm":
               length * voxel_size / 1000.0}
        for label, render in (("3.9", default), ("3.10", antialiased)):
            start = time.perf_counter()
            volume = render(points, radii, shape)
            seconds = time.perf_counter() - start
            out[label] = {"seconds": seconds, "voxels": int(volume.sum()), "components26": components(volume),
                          "components6": components(volume, 6),
                          "short_necks_per_mm": short_necks(volume) / (length * voxel_size / 1000.0)}
        out["time_ratio"] = out["3.10"]["seconds"] / out["3.9"]["seconds"]
        return out


if __name__ == "__main__":
    unittest.main()
