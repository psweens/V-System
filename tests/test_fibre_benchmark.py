"""
Benchmark of the fibre bed, bed.py's kind "fibre": a bed alone over its
settings (B1), and a bed filled between the feeders of a mesh member and of
a packed forest, with stretched foam beside it on the same feeders (B3);
with the cost of a dense bed alone in a cube of 256 d_min. The fibre
family, which grows its own feeders (B2) and a library of it (B4), is not
part of this file.

Slow: every run is a bed.build call of seconds to minutes, so the benchmark
runs only with VSYSTEM_SLOW_TESTS=1. Each bed is built and described in a
fresh process of its own, one at a time, so that its time is its own and
its peak RSS is the process's own high-water mark (VmHWM, as
tests/test_topology_benchmark.own_peak_rss_mb reads it; ru_maxrss is
recorded beside it, but a spawned process's ru_maxrss starts at its
parent's peak), reported also less the peak of a process that only
imported the same modules. A run that takes longer than its limit is
stopped, reported and left out. The benchmark prints its tables and writes
them as JSON to the path in VSYSTEM_BENCHMARK_JSON when that is set. It
asserts sanity only: the counters keep the laws of bed.py
(tests/test_fibre.check_laws for a fibre bed, E1-E7 and zero fibre
counters for foam), and the counts sway does not draw (the lanes, their
knots, the link sites and candidates) are the same at every sway.

Every bed alone is described as
    describe(nodes, graph.build(nodes)["edges"], volume=high - low,
             frame=result["frame"], d_ref=d_min, evd_spacing=side / 64),
its area_source "argument"; a bed with feeders is described so on its own
columns (those after the feeders') and again with the feeders and their
labels. Every density is printed with its area_source.

Run from the repository root, under nice, with
    VSYSTEM_SLOW_TESTS=1 nice python -m unittest tests.test_fibre_benchmark
which runs every block but the cost case. The blocks run in parts:
    VSYSTEM_FIBRE_BLOCKS    the blocks, comma-separated: grid, sway, links,
                            persistence, run, dense (B1); member, foam,
                            forest (B3); cost256. Default: every block but
                            cost256. A block may name its own seeds after
                            "@", ranges with "-" and lists with "+":
                            dense@1, member@1+3, grid@1-3.
    VSYSTEM_FIBRE_SEEDS     the seeds of every block that names none,
                            "1,2,3" or "1-3"; default 1-3, and 1-2 for the
                            forest.
    VSYSTEM_FIBRE_GRID_SPACINGS, VSYSTEM_FIBRE_GRID_DENSITIES
                            a subset of the grid's spacings and densities,
                            comma-separated; default all of them.
    VSYSTEM_FIBRE_SIDE      B1's cube side in d_min, default 96 (a smaller
                            side only for a quick check of the benchmark).
    VSYSTEM_FIBRE_RUN_LIMIT_S
                            seconds after which a run is stopped and left
                            out, default 900; the 256 cube has
                            VSYSTEM_FIBRE_COST_LIMIT_S, default 3600.
For example, B1 at spacing {5, 3.2} x density {0.5, 0.6} on seeds 1-3 and
the dense block at seed 1, then the cost case:
    VSYSTEM_SLOW_TESTS=1 VSYSTEM_FIBRE_BLOCKS="grid,dense@1" VSYSTEM_FIBRE_SEEDS=1-3 \\
        VSYSTEM_FIBRE_GRID_SPACINGS=5,3.2 VSYSTEM_FIBRE_GRID_DENSITIES=0.5,0.6 \\
        nice python -m unittest tests.test_fibre_benchmark
    VSYSTEM_SLOW_TESTS=1 VSYSTEM_FIBRE_BLOCKS=cost256 nice python -m unittest tests.test_fibre_benchmark
A single block can also be named as a test:
    VSYSTEM_SLOW_TESTS=1 nice python -m unittest \\
        tests.test_fibre_benchmark.FibreBedAloneBenchmark.test_the_dense_block

The blocks. B1, a bed alone in a cube of side 96 d_min along x, d_min 1,
margin 1: "grid" spacing {6, 5, 4, 3.2} x density {0.4, 0.5, 0.6} at the
defaults; "sway" sway {0, 0.5, 1, 1.5, 2} x knot {4, 2.5} at spacing 5 and
6; "links" link_spacing {null, 40, 20, 10}; "persistence" {8, 16, 32}; "run"
{null, 100, 50}; "dense" spacing {3.2, 3.7} x density 0.6 x knot {4, 2.5,
2.1} x sway {w / 2, w} x link_spacing {20, 10}, w = (h - T) / 2. A run that
two blocks share is built once. B3: "member" a mesh member of ratio 8
(library.grow_member, its growth box the domain), a fibre bed along x at
(spacing, density) {(5, 0.5), (4, 0.6), (3.2, 0.6)} between its feeders and
the same without them, each passed the frame
guidance.unguided_frame((1, 0, 0), (0, 0, 1), box / 2); "foam" foam at
stretch 4 and density 0.5 on the same domain, feeders and frame; "forest"
the typical forest of tests/test_join.py's BenchmarkTests (tree networks
grow(seed, 5, subdivisions=2) of seeds 1-8, placed by build_forest at its
packing of 330 per 700^3 d_min, here in a cube of side 192 with the
placement generator default_rng(seed), cropped by
join.crop_network(n, lo, hi, r_max + 1)) passed as one feeder array of
label 0, fibre beds along x at {(5, 0.5), (4, 0.6)}, the bed's seed the
forest's. "cost256" the bed alone at spacing 3.2 and density 0.6, links on,
in cubes of side 128 and 256 on seed 1, and foam's BedCostBenchmark case of
10^4 seeds (seed 1) for the cost per placed chain length; the 128 cube
gives the extrapolation should the 256 cube pass its limit.

What each run reports, per the order's B1 and B3 lists: the counters;
crossing_density_d against N / area (area the domain's section across the
axis) and its excess over N / area against l / (8 lambda) x the links
over the sites (and beside it the kept links over the sites), the sites
without a feasible partner and the link acceptance (bed_links /
bed_candidates_link), with the chord graph's crossings counted exactly on
describe's nine planes (lanes, links and tip edges apart); the
nearest-neighbour median and mean, nn_ratio and density_cv; the capillary
crossing ratio; J(beta) (bed.chord_crossing_ratio), a Monte Carlo of the
cut law (two Gaussian offsets per chord cut radially at w, the ratio of
mean chord length to l) and the interior lane chords' ratio measured on the
knots; S, watson_K and fisher_axial_K_exact; within_20_deg and
within_45_deg beside the fractions a Fisher-axial law exp(K |u|) at the
run's fisher_axial_K_exact and a Watson law exp(K u^2) at its watson_K
predict (Gauss-Legendre integration over u); cross_connections
per_length_d, end_spacing_d against link_spacing, length_share, junction
types and the axial segments' median and mean; the segments' arc/chord;
tissue_distance mean, p90 and max at evd_spacing side / 64; the capillary
length density in d_min^-2; chains walked, chord and dropped, and redraws;
the surviving lane-length fraction against r^2 / (r + lambda_e)^2, lambda_e
the run's end_spacing_d; the seconds of each stage and the peak RSS. With
feeders also bed_knots_on_feeders, the shares of the planned lane length
(the chords of every pair of consecutive knots of a lane) cut by feeders,
refused (or broken), peeled and removed as islands by the first prune,
removed after the walk and kept, the components, the tips eligible and
attached and the cost per 10^6 d_min^3 of bed volume; for foam its crossing
ratio and its seed count against floor(density V / (stretch h^3)).
"""
import hashlib
import json
import math
import multiprocessing
import os
import resource
import sys
import time
import traceback
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bed  # noqa: E402
import graph  # noqa: E402
import join  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from describe import describe  # noqa: E402
from guidance import unguided_frame  # noqa: E402
# the checks of the fibre tests and the helpers of the other benchmarks, by name only, so that no test class
# of theirs is collected here
from tests.test_fibre import (D_MIN, FIBRE, MARGIN, build_case, check_laws, f9_holds_by_condition,  # noqa: E402
                              interior_count, lane_edge_mask, link_edge_mask, vertex_points)
from tests.test_join import build_forest, grow  # noqa: E402
from tests.test_topology_benchmark import bed_cube, bed_laws, own_peak_rss_mb, tissue_summary  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmark")

B1_BLOCKS = ("grid", "sway", "links", "persistence", "run", "dense")
B3_BLOCKS = ("member", "foam", "forest")
COST_BLOCKS = ("cost256",)
BLOCKS = B1_BLOCKS + B3_BLOCKS + COST_BLOCKS
DEFAULT_SEEDS = (1, 2, 3)
FOREST_SEEDS = (1, 2)

# B1: a bed alone in a cube along x
SIDE = 96.0
EVD_DIVISIONS = 64                        # evd_spacing = side / 64
GRID_SPACINGS = (6.0, 5.0, 4.0, 3.2)
GRID_DENSITIES = (0.4, 0.5, 0.6)
SWAYS = (0.0, 0.5, 1.0, 1.5, 2.0)
SWAY_KNOTS = (4.0, 2.5)
SWAY_SPACINGS = (5.0, 6.0)
LINK_SPACINGS = (None, 40.0, 20.0, 10.0)
PERSISTENCES = (8.0, 16.0, 32.0)
RUNS = (None, 100.0, 50.0)
DENSE_SPACINGS = (3.2, 3.7)
DENSE_DENSITY = 0.6
DENSE_KNOTS = (4.0, 2.5, 2.1)
DENSE_LINK_SPACINGS = (20.0, 10.0)

# B3: the mesh member, foam beside it, and the packed forest
MEMBER_RATIO = 8.0
MEMBER_FILLS = ((5.0, 0.5), (4.0, 0.6), (3.2, 0.6))
FOAM_STRETCH, FOAM_DENSITY = 4.0, 0.5
FOREST_SIDE = 192.0
FOREST_PACKING = (330, 700.0)             # tests/test_join.py's typical forest: 330 networks per 700^3 d_min
FOREST_LIBRARY_SEEDS = tuple(range(1, 9))
FOREST_FILLS = ((5.0, 0.5), (4.0, 0.6))

# cost256: a dense bed alone, and foam's BedCostBenchmark case for the cost per placed chain length
COST_SIDES = (128.0, 256.0)
COST_SETTINGS = {"spacing": 3.2, "density": 0.6}
COST_FOAM_SEEDS = 10000
CHAIN_LENGTH_UNIT = 1e5                   # seconds per 10^5 d_min of placed chain length
VOLUME_UNIT = 1e6                         # seconds per 10^6 d_min^3 of bed volume

RUN_LIMIT_S = 900.0
COST_LIMIT_S = 3600.0

# The Monte Carlo of the cut law: this many chords, on its own generator
CUT_LAW_CHORDS = 400000
CUT_LAW_SEED = 20251

# The counters a sway does not draw: on a bed alone, the lanes, their knots,
# the link sites and the sites without a neighbour are the same at every
# sway (which neighbours are feasible partners moves with the knots); with
# them the lanes' anchors and the knots' t, compared bitwise
SWAY_INVARIANT = ("bed_proposals", "bed_proposals_outside", "bed_proposals_overlapping", "bed_seeds",
                  "bed_knots", "bed_knots_boundary", "bed_lane_candidates", "bed_breaks", "bed_link_sites",
                  "bed_link_sites_no_partner", "bed_candidates_tip")
FIBRE_COUNTERS = bed.EVENT_KEYS[bed.EVENT_KEYS.index("bed_proposals_outside"):]

_results = {}


def _record(section, key, value):
    _results.setdefault(section, {})[key] = value


def tearDownModule():
    # whatever ran, so that one block run on its own still writes its tables
    path = os.environ.get("VSYSTEM_BENCHMARK_JSON")
    if path and _results:
        with open(path, "w") as handle:
            json.dump(_results, handle, indent=1)


# --------------------------------------------------------------------------
# What runs: the blocks and seeds named in the environment


def seed_list(text, separator=","):
    """Seeds from "1,2,3", "1-3" or a mix ("1-2,5"); with separator "+" for a block's own."""
    seeds = []
    for part in text.replace(" ", "").split(separator):
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def selected_blocks():
    """{block: seeds} of the blocks VSYSTEM_FIBRE_BLOCKS names (every block but cost256 by default)."""
    named = os.environ.get("VSYSTEM_FIBRE_BLOCKS")
    common = os.environ.get("VSYSTEM_FIBRE_SEEDS")
    out = {}
    if named and named.strip():
        entries = [entry for entry in named.replace(" ", "").split(",") if entry]
    else:
        entries = [block for block in BLOCKS if block not in COST_BLOCKS]
    for entry in entries:
        block, _, own = entry.partition("@")
        if block not in BLOCKS:
            raise ValueError(f"VSYSTEM_FIBRE_BLOCKS names {block!r}; the blocks are {', '.join(BLOCKS)}")
        if own:
            seeds = seed_list(own, "+")
        elif common:
            seeds = seed_list(common)
        else:
            seeds = FOREST_SEEDS if block == "forest" else DEFAULT_SEEDS
        out[block] = seeds
    return out


def numbers(name, default):
    """A tuple of floats from a comma-separated environment variable, else `default`."""
    text = os.environ.get(name)
    return tuple(float(v) for v in text.replace(" ", "").split(",") if v) if text else default


def b1_side():
    return float(os.environ.get("VSYSTEM_FIBRE_SIDE", SIDE))


def run_limit():
    return float(os.environ.get("VSYSTEM_FIBRE_RUN_LIMIT_S", RUN_LIMIT_S))


def cost_limit():
    return float(os.environ.get("VSYSTEM_FIBRE_COST_LIMIT_S", COST_LIMIT_S))


def sway_cut(spacing, diameter=FIBRE["diameter"], d_min=D_MIN, margin=MARGIN):
    """w = (h - T) / 2, in d_min, at the benchmark's margin."""
    return (spacing * d_min - (diameter * d_min + margin)) / 2.0 / d_min


def b1_settings(block):
    """The fibre settings of a B1 block, in the order the order lists them."""
    if block == "grid":
        spacings = numbers("VSYSTEM_FIBRE_GRID_SPACINGS", GRID_SPACINGS)
        densities = numbers("VSYSTEM_FIBRE_GRID_DENSITIES", GRID_DENSITIES)
        return [dict(FIBRE, spacing=s, density=d) for s in spacings for d in densities]
    if block == "sway":
        return [dict(FIBRE, spacing=s, knot=k, sway=v) for s in SWAY_SPACINGS for k in SWAY_KNOTS for v in SWAYS]
    if block == "links":
        return [dict(FIBRE, link_spacing=v) for v in LINK_SPACINGS]
    if block == "persistence":
        return [dict(FIBRE, persistence=v) for v in PERSISTENCES]
    if block == "run":
        return [dict(FIBRE, run=v) for v in RUNS]
    if block == "dense":
        out = []
        for spacing in DENSE_SPACINGS:
            w = sway_cut(spacing)
            for knot in DENSE_KNOTS:
                for sway in (w / 2.0, w):
                    for link_spacing in DENSE_LINK_SPACINGS:
                        out.append(dict(FIBRE, spacing=spacing, density=DENSE_DENSITY, knot=knot, sway=sway,
                                        link_spacing=link_spacing))
        return out
    raise ValueError(block)


def settings_label(settings):
    """A short name for a setting: kind, spacing, density, knot, sway, link spacing, persistence and run."""
    if settings["kind"] == "foam":
        return f"foam h{settings['spacing']:g} rho{settings['density']:g} st{settings['stretch']:g}"

    def show(value):
        return "-" if value is None else f"{value:.4g}"

    return (f"h{show(settings['spacing'])} rho{show(settings['density'])} l{show(settings['knot'])} "
            f"sw{show(settings['sway'])} L{show(settings['link_spacing'])} P{show(settings['persistence'])} "
            f"run{show(settings['run'])}")


# --------------------------------------------------------------------------
# Laws the order states with a closed form, computed here


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(512)


def _integral(f, low, high):
    """Gauss-Legendre over [low, high], 512 points."""
    x = (high - low) / 2.0 * _GAUSS_NODES + (low + high) / 2.0
    return (high - low) / 2.0 * float(np.sum(_GAUSS_WEIGHTS * f(x)))


def law_within(K, degrees, law):
    """
    The fraction of directions within `degrees` of the axis under an axial
    law of concentration K on the sphere, u = |cos| uniform under isotropy:
    int_c^1 f(u) du / int_0^1 f(u) du, c = cos(degrees), f = exp(K u) for
    the Fisher-axial law and exp(K u^2) for the Watson law, integrated
    numerically; None without K.
    """
    if K is None:
        return None
    c = math.cos(math.radians(degrees))
    shift = max(float(K), 0.0)          # exp(K (u - 1)) or exp(K (u^2 - 1)) for K > 0, so nothing overflows
    if law == "fisher_axial":
        def f(u):
            return np.exp(K * u - shift)
    else:
        def f(u):
            return np.exp(K * u * u - shift)
    return _integral(f, c, 1.0) / _integral(f, 0.0, 1.0)


_CUT_LAW = {}


def cut_law_monte_carlo(sway, knot, w, chords=CUT_LAW_CHORDS):
    """
    The chord law under the radial cut: the length-weighted crossing ratio
    of chords of axial length `knot` between two knots whose transverse
    offsets are two-dimensional Gaussians of standard deviation `sway` per
    component, each scaled radially to `w` when longer, i.e. mean chord
    length over `knot`; with its standard error. Cached by its arguments.
    """
    key = (float(sway), float(knot), float(w), int(chords))
    if key not in _CUT_LAW:
        rng = np.random.default_rng([CUT_LAW_SEED, chords])
        offsets = sway * rng.standard_normal((2, chords, 2))
        radius = np.sqrt((offsets ** 2).sum(axis=2))
        wide = radius > w
        offsets[wide] *= (w / radius[wide])[:, None]
        delta = offsets[1] - offsets[0]
        length = np.sqrt(knot ** 2 + (delta ** 2).sum(axis=1))
        _CUT_LAW[key] = (float(length.mean() / knot), float(length.std() / math.sqrt(chords) / knot))
    return _CUT_LAW[key]


# --------------------------------------------------------------------------
# One run, in a fresh process


def _plain(value):
    """numpy scalars and arrays as Python numbers and lists, NaN and infinities as None."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _ru_maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _dot(points, a):
    """points . a, the three products summed in order."""
    return points[..., 0] * a[0] + points[..., 1] * a[1] + points[..., 2] * a[2]


def _norm(vectors):
    return np.sqrt((vectors ** 2).sum(axis=-1))


def _chain_length(bed_nodes):
    """The summed arc lengths of the written bed polylines."""
    return float(sum(_norm(np.diff(bed_nodes[:3, run].T, axis=0)).sum() for run in graph.polylines(bed_nodes)))


def _laws_error(case):
    """None when the laws hold, else what failed: check_laws for a fibre bed, the foam laws otherwise."""
    if case["settings"]["kind"] == "fibre":
        try:
            check_laws(unittest.TestCase(), case)
        except AssertionError as error:
            return f"check_laws: {error}"
        return None
    events, result = case["events"], case["result"]
    low, high = case["domain"]
    laws = bed_laws(events, bed.planned_seeds(high - low, case["settings"], case["d_min"]), case["attempts"])
    laws["E5"] = events["bed_edges"] == (len(result["edges"]) + events["bed_edges_dead_end"]
                                         + events["bed_edges_island"] + events["bed_edges_dropped"])
    laws["E6"] = events["bed_seeds"] == (int(result["kept"].sum()) + events["bed_seeds_dead_end"]
                                         + events["bed_seeds_island"])
    laws["fibre counters 0"] = all(events[key] == 0 for key in FIBRE_COUNTERS)
    failed = [name for name, holds in laws.items() if not holds]
    return f"foam laws fail: {', '.join(failed)}" if failed else None


def descriptor_items(report, extents, d_min):
    """The order's B1 descriptors of one describe report, flat, in d_min."""
    spacing = report["transverse_spacing"] or {}
    section = report["cross_section"] or {}
    orientation = (report["frame_orientation"] or {}).get("capillary") or {}
    connections = report["cross_connections"] or {}
    junctions = connections.get("junction_types") or {}
    axial = connections.get("axial_segments_d") or {}
    tissue = report["tissue_distance"] or {}
    arc = report["arc_chord"]
    volume = float(np.prod(extents))
    length = orientation.get("length")
    K_f, K_w = orientation.get("fisher_axial_K_exact"), orientation.get("watson_K")
    density = report["length_density_mm_per_mm3"]
    return {
        "crossing_density_d": spacing.get("crossing_density_d"), "area_source": spacing.get("area_source"),
        "nn_median_d": spacing.get("median_d"), "nn_mean_d": spacing.get("mean_d"),
        "nn_ratio": section.get("nn_ratio"), "density_cv": section.get("density_cv"),
        "crossing_ratio": orientation.get("crossing_ratio"), "S": orientation.get("S"),
        "watson_K": K_w, "fisher_axial_K_exact": K_f,
        "within_20_deg": orientation.get("within_20_deg"), "within_45_deg": orientation.get("within_45_deg"),
        "within_20_fisher": law_within(K_f, 20.0, "fisher_axial"),
        "within_45_fisher": law_within(K_f, 45.0, "fisher_axial"),
        "within_20_watson": law_within(K_w, 20.0, "watson"), "within_45_watson": law_within(K_w, 45.0, "watson"),
        "cc_count": connections.get("count"), "cc_per_length_d": connections.get("per_length_d"),
        "end_spacing_d": connections.get("end_spacing_d"), "cc_length_share": connections.get("length_share"),
        "junctions_count": junctions.get("count"), "junction_T": junctions.get("T"),
        "junction_Y": junctions.get("Y"), "junction_other": junctions.get("other"),
        "axial_median_d": axial.get("median"), "axial_mean_d": axial.get("mean"),
        "arc_chord_mean": arc["mean"], "arc_chord_median": arc["median"], "arc_chord_p90": arc["p90"],
        "arc_chord_max": arc["max"],
        "tissue_mean_d": tissue.get("mean_d"), "tissue_p90_d": tissue.get("p90_d"),
        "tissue_max_d": tissue.get("max_d"), "tissue_spacing": tissue.get("spacing"),
        "tissue_source": (tissue.get("domain") or {}).get("source"),
        "length_density_d2": length / volume * d_min ** 2 if length else None,
        "length_density_all_d2": density * 1e-6 * d_min ** 2 if density is not None else None,
        "volume_source": report["volume_source"], "components": report["components"],
    }


def _describe_excerpt(report):
    """The parts of a report the JSON keeps."""
    orientation = report["frame_orientation"] or {}
    out = {key: report[key] for key in ("transverse_spacing", "cross_section", "cross_connections", "arc_chord",
                                        "junctions", "branch_angles_deg", "components", "cycles",
                                        "volume_source", "length_density_mm_per_mm3")}
    out.update({"frame_orientation_capillary": orientation.get("capillary"),
                "segments_capillary": (report["segments_by_class"] or {}).get("capillary"),
                "tissue_distance": tissue_summary(report["tissue_distance"])})
    return out


def _describe_planes(bed_nodes, edges, a):
    """describe's nine plane levels c_k on the axis a, from the end points of its edges of positive length."""
    points = bed_nodes[:3].T
    length = _norm(points[edges[1]] - points[edges[0]])
    ends = np.unique(np.concatenate([edges[0][length > 0.0], edges[1][length > 0.0]]))
    s = _dot(points[ends], a)
    s_min, s_max = float(s.min()), float(s.max())
    return [s_min + (0.1 + 0.1 * k) * (s_max - s_min) for k in range(9)]


def fibre_items(case, bed_nodes, edges, report):
    """
    What the order asks of a fibre bed beyond describe's keys: the planned
    counts, the link sites and acceptance, the chord graph's crossings on
    describe's planes, the chord law, the surviving lane length and the
    shares of the planned lane length lost on the way.
    """
    result, events, settings = case["result"], case["events"], case["settings"]
    d_min, margin = case["d_min"], case["margin"]
    low, high = case["domain"]
    extents = high - low
    a = np.asarray(result["axis"], dtype=float)
    area = bed._section([float(v) for v in extents], a)[0]
    n = len(result["seeds"])
    m = interior_count(case)
    h, d = settings["spacing"] * d_min, settings["diameter"] * d_min
    knot, sway = settings["knot"] * d_min, settings["sway"] * d_min
    w = (h - (d + margin)) / 2.0
    lam = None if settings["link_spacing"] is None else settings["link_spacing"] * d_min
    q = knot / (2.0 * lam) if lam else 0.0
    out = {"lanes": n, "knots": events["bed_knots"], "interior_knots": m,
           "planned_seeds": bed.planned_seeds(extents, settings, d_min, axis=a),
           "planned_candidates": bed.planned_candidates(extents, settings, d_min, axis=a),
           "candidates": events["bed_candidates"], "w_d": w / d_min,
           "theta_max_deg": math.degrees(math.atan2(2.0 * w, knot)),
           "f9_condition": f9_holds_by_condition(settings, d_min, margin),
           "n_over_area_d": n / area * d_min ** 2, "section_area_d2": area / d_min ** 2}
    out["candidates_over_planned"] = out["candidates"] / out["planned_candidates"] if out["planned_candidates"] \
        else None
    sites, candidates_link = events["bed_link_sites"], events["bed_candidates_link"]
    out.update({"link_sites": sites, "link_sites_planned": q * m,
                "link_sites_over_planned": sites / (q * m) if q * m else None,
                "link_sites_angle": events["bed_link_sites_angle"],
                "link_candidates": candidates_link, "links": events["bed_links"], "links_kept": events["bed_links_kept"],
                "link_acceptance": events["bed_links"] / candidates_link if candidates_link else None,
                "links_of_sites": events["bed_links"] / sites if sites else None,
                "links_kept_of_sites": events["bed_links_kept"] / sites if sites else None})
    per_lane_plane = knot / (8.0 * lam) if lam else 0.0
    out["predicted_excess_rel"] = per_lane_plane * (out["links_of_sites"] or 0.0)
    out["predicted_excess_rel_kept"] = per_lane_plane * (out["links_kept_of_sites"] or 0.0)
    density = report["transverse_spacing"]["crossing_density_d"] if report["transverse_spacing"] else None
    out["excess_d"] = density - out["n_over_area_d"] if density is not None else None
    out["excess_rel"] = out["excess_d"] / out["n_over_area_d"] if density is not None else None
    out["walked_crossings"] = int(round(density * 9 * area / d_min ** 2)) if density is not None else None

    # the chord graph on describe's planes: lanes, links and tip edges apart
    planes = _describe_planes(bed_nodes, edges, a)
    X = vertex_points(case)
    kept = result["edges"]
    s0, s1 = _dot(X[kept[:, 0]], a), _dot(X[kept[:, 1]], a)
    lane, link = lane_edge_mask(case, kept), link_edge_mask(case, kept)
    crossings = {"lane": [], "link": [], "tip": []}
    for c in planes:
        crosses = (s0 < c) != (s1 < c)
        crossings["lane"].append(int(np.count_nonzero(crosses & lane)))
        crossings["link"].append(int(np.count_nonzero(crosses & link)))
        crossings["tip"].append(int(np.count_nonzero(crosses & ~lane & ~link)))
    total = sum(sum(values) for values in crossings.values())
    out.update({"chord_crossings": total, "chord_lane_crossings": sum(crossings["lane"]),
                "chord_link_crossings": sum(crossings["link"]), "chord_tip_crossings": sum(crossings["tip"]),
                "chord_graph_density_d": total / (9 * area) * d_min ** 2,
                "chord_link_share": sum(crossings["link"]) / total if total else None,
                "chord_links_per_lane_plane": sum(crossings["link"]) / (9 * n),
                "density_identity": all(count == n for count in crossings["lane"]),
                "chord_crossings_by_plane": crossings, "plane_levels": planes})

    # the chord law, on consecutive interior knots of a lane
    knots, knot_lane, knot_t, boundary = (result[key] for key in ("knots", "knot_lane", "knot_t", "knot_boundary"))
    pair = (knot_lane[1:] == knot_lane[:-1]) & ~boundary[1:] & ~boundary[:-1]
    delta = knots[1:][pair] - knots[:-1][pair]
    axial = np.abs(_dot(delta, a))
    out["chord_interior"] = float(_norm(delta).sum() / axial.sum()) if axial.size else None
    out["chord_J"] = bed.chord_crossing_ratio(settings)
    out["chord_mc"], out["chord_mc_se"] = cut_law_monte_carlo(sway, knot, w)

    # the surviving lane length, along the axis, against r^2 / (r + lambda_e)^2
    kept_lane = kept[lane]
    planned_axial = float((result["lane_extent"][:, 1] - result["lane_extent"][:, 0]).sum())
    out["lane_length_kept_fraction"] = float(np.abs(knot_t[kept_lane[:, 1]] - knot_t[kept_lane[:, 0]]).sum()) \
        / planned_axial
    end_spacing = (report["cross_connections"] or {}).get("end_spacing_d")
    out["end_spacing_over_link_spacing"] = end_spacing / settings["link_spacing"] \
        if end_spacing is not None and settings["link_spacing"] else None
    if settings["run"] is not None and end_spacing is not None:
        r, lam_e = settings["run"] * d_min, end_spacing * d_min
        out["run_prediction"] = r ** 2 / (r + lam_e) ** 2
    else:
        out["run_prediction"] = 1.0 if settings["run"] is None else None

    # the planned lane length (every pair of consecutive knots of a lane) and where it went
    lo = np.flatnonzero(knot_lane[1:] == knot_lane[:-1])
    pair_length = _norm(knots[lo + 1] - knots[lo])
    planned = float(pair_length.sum())
    on = np.zeros(len(knots), dtype=bool)
    if case["feeders"] is not None:
        fnodes, ftree = bed._feeders(case["feeders"])
        blocked = bed._feeder_parts(fnodes, ftree, True, margin, d / 2.0)[4]
        if blocked is not None:
            on = bed._Blocked(*blocked).mask(knots)
    out["on_feeders_agrees"] = int(np.count_nonzero(on)) == events["bed_knots_on_feeders"]
    cut = on[lo] | on[lo + 1]
    greedy = result["greedy_edges"]
    n_knots, n_tips = len(knots), len(result["tips"])
    n_vertices = n_knots + n_tips
    # the first prune replayed: the peel alone, then the island pass
    is_tip = np.zeros(n_vertices, dtype=bool)
    is_tip[n_knots:] = True
    kept_vertex = np.ones(n_vertices, dtype=bool)
    kept_vertex[:n_knots] = ~on
    anchored = is_tip.copy()
    anchored[:n_knots] |= boundary
    peel = np.ones(len(greedy), dtype=bool)
    bed._prune(n_vertices, greedy, peel, is_tip, kept_vertex, False, anchored=anchored)
    island = peel.copy()
    bed._prune(n_vertices, greedy, island, is_tip, kept_vertex, case["feeders"] is not None, anchored=anchored)

    def code(e):
        return e[:, 0].astype(np.int64) * n_vertices + e[:, 1]

    final = np.isin(code(greedy), code(kept)) if len(greedy) else np.zeros(0, dtype=bool)
    greedy_lane = lane_edge_mask(case, greedy)
    where = np.searchsorted(lo, greedy[greedy_lane, 0])
    fate = np.full(lo.size, "refused", dtype=object)
    fate[cut] = "feeders"
    fate[where] = np.where(~peel[greedy_lane], "peeled", np.where(~island[greedy_lane], "islands",
                                                                  np.where(final[greedy_lane], "kept", "after_walk")))
    for name in ("feeders", "refused", "peeled", "islands", "after_walk", "kept"):
        out[f"lane_share_{name}"] = float(pair_length[fate == name].sum()) / planned if planned else None
    return out


def _bed_task(task):
    """
    One bed in a fresh process: bed.build timed by stage, the peak RSS once
    it is built, its laws, and, unless the task is for cost only, describe
    on its own columns and, with feeders, on the whole network, timed; with
    the order's items flat in "row".
    """
    settings = bed.parse_settings(task["settings"])
    low, high = (np.asarray(corner, dtype=float) for corner in task["domain"])
    extents = high - low
    d_min, margin = float(task.get("d_min", D_MIN)), float(task.get("margin", MARGIN))
    feeders = task.get("feeders")
    started = time.perf_counter()
    case = build_case(task["label"], (low, high), settings, seed=task["seed"], feeders=feeders,
                      frame=task.get("frame"), d_min=d_min, margin=margin)
    seconds = time.perf_counter() - started
    peak, ru_peak = own_peak_rss_mb(), _ru_maxrss_mb()
    result, events = case["result"], case["events"]
    laws = _laws_error(case)
    n_feeders = 0 if feeders is None else np.asarray(feeders[0]).shape[1]
    bed_nodes = result["nodes"][:, n_feeders:]
    volume_d3 = float(np.prod(extents)) / d_min ** 3
    chain_length = _chain_length(bed_nodes)
    row = {"seconds": seconds, "peak_rss_mb": peak, "ru_maxrss_mb": ru_peak,
           "chain_length_d": chain_length / d_min,
           "seconds_per_1e5_chain_d": seconds / (chain_length / d_min / CHAIN_LENGTH_UNIT) if chain_length else None,
           "seconds_per_1e6_volume_d3": seconds / (volume_d3 / VOLUME_UNIT),
           "volume_d3": volume_d3, "columns": int(result["nodes"].shape[1]), "bed_columns": int(bed_nodes.shape[1]),
           "walked": events["bed_segments_walked"], "chord": events["bed_segments_chord"],
           "dropped": events["bed_segments_dropped"], "removed": events["bed_segments_removed"],
           "redraws": events["bed_walk_redraws"], "segments": events["bed_segments"],
           "tips_eligible": events["bed_tips"] - events["bed_tips_inside_junction"],
           "tips_attached": events["bed_tips_attached"], "tip_edges": events["bed_tip_edges"],
           "bed_components": events["bed_components"], "bed_components_one_tree": events["bed_components_one_tree"]}
    row.update({f"seconds_{stage}": value for stage, value in case["timings"].items()})
    out = {"label": task["label"], "seed": task["seed"], "settings": settings, "domain": [low, high],
           "events": events, "timings": case["timings"], "laws": laws, "frame": result["frame"]}
    if settings["kind"] == "fibre":
        out["invariant"] = dict({key: events[key] for key in SWAY_INVARIANT},
                                anchors=_digest(result["seeds"]), knot_t=_digest(result["knot_t"]))
        row["on_feeders_knots"] = events["bed_knots_on_feeders"]
    else:
        V = float(np.prod(extents - settings["diameter"] * d_min))
        h = settings["spacing"] * d_min
        row.update({"seeds": events["bed_seeds"],
                    "seeds_formula": int(math.floor(settings["density"] * V / (settings["stretch"] * h ** 3))),
                    "planned_seeds": bed.planned_seeds(extents, settings, d_min),
                    "kept_seeds": int(result["kept"].sum())})
        row["seeds_agree"] = row["seeds"] == row["seeds_formula"]
    if task.get("describe", True):
        evd = float(extents.max()) / EVD_DIVISIONS
        started = time.perf_counter()
        edges = graph.build(bed_nodes)["edges"]
        report = describe(bed_nodes, edges, volume=[float(v) for v in extents], frame=result["frame"], d_ref=d_min,
                          evd_spacing=evd)
        row["describe_seconds"] = time.perf_counter() - started
        row.update(descriptor_items(report, extents, d_min))
        out["describe"] = _describe_excerpt(report)
        if settings["kind"] == "fibre":
            row.update(fibre_items(case, bed_nodes, edges, report))
        if feeders is not None:
            started = time.perf_counter()
            whole = describe(result["nodes"], graph.build(result["nodes"])["edges"],
                             volume=[float(v) for v in extents], frame=result["frame"], d_ref=d_min,
                             tree=result["tree"], evd_spacing=evd)
            row["whole_describe_seconds"] = time.perf_counter() - started
            row.update({f"whole_{key}": value for key, value in descriptor_items(whole, extents, d_min).items()})
            out["describe_whole"] = _describe_excerpt(whole)
    elif settings["kind"] == "fibre":
        row.update({"lanes": len(result["seeds"]), "knots": events["bed_knots"],
                    "planned_candidates": bed.planned_candidates(extents, settings, d_min,
                                                                 axis=np.asarray(result["axis"]))})
    row["peak_rss_mb_with_describe"] = own_peak_rss_mb()
    out["row"] = row
    return _plain(out)


def _baseline_task(_):
    """The peak RSS of a fresh process that has imported what a run's process imports, and done nothing else."""
    return own_peak_rss_mb()


def _child(function, task, connection):
    try:
        connection.send({"ok": function(task)})
    except BaseException:                   # reported to the parent, which fails the run
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def run_fresh(function, task, limit):
    """
    function(task) in a fresh spawned process, waited on for at most `limit`
    seconds: {"ok": its result}, {"error": the traceback} or {"timed_out":
    limit}, with "wall_seconds".
    """
    for name in library.THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(function, task, sending))
    started = time.perf_counter()
    process.start()
    sending.close()
    try:
        message = receiving.recv() if receiving.poll(limit) else {"timed_out": limit}
    except EOFError:
        message = {"error": "the process ended without a result"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receiving.close()
    if "error" in message and process.exitcode not in (0, None):
        message["error"] += f" (exit code {process.exitcode})"
    message["wall_seconds"] = time.perf_counter() - started
    return message


_RUNS = {}
_BASELINE = {}


def baseline_rss_mb():
    if "mb" not in _BASELINE:
        _BASELINE["mb"] = run_fresh(_baseline_task, None, 600.0).get("ok")
    return _BASELINE["mb"]


def _task_key(task):
    tag = task.get("tag", "alone")
    return json.dumps({"tag": tag, "domain": _plain(task["domain"]), "settings": bed.parse_settings(task["settings"]),
                       "seed": task["seed"], "describe": task.get("describe", True)}, sort_keys=True)


def run_tasks(tasks, limit=None):
    """Each task's message, built once per distinct run and one process at a time; printed as each ends."""
    out = []
    for task in tasks:
        key = _task_key(task)
        if key not in _RUNS:
            message = run_fresh(_bed_task, task, run_limit() if limit is None else limit)
            if "timed_out" in message:
                extents = np.asarray(task["domain"][1], dtype=float) - np.asarray(task["domain"][0], dtype=float)
                settings = bed.parse_settings(task["settings"])
                axis = settings.get("axis") or [1.0, 0.0, 0.0]
                message["planned_seeds"] = bed.planned_seeds(extents, settings, D_MIN, axis=axis)
                message["planned_candidates"] = bed.planned_candidates(extents, settings, D_MIN, axis=axis)
            _RUNS[key] = message
            state = ("ok" if "ok" in message else "timed out" if "timed_out" in message else "error")
            print(f"  [{state} in {message['wall_seconds']:.1f} s] {task['label']} seed {task['seed']}", flush=True)
        out.append(dict(_RUNS[key], task=task))
    return out


# --------------------------------------------------------------------------
# Reporting


def _fmt(value, width=7, digits=3):
    if isinstance(value, bool):
        return f"{str(value):>{width}s}"
    if isinstance(value, (int, float)):
        return f"{value:{width}.{digits}f}" if isinstance(value, float) else f"{value:{width}d}"
    return f"{'-' if value is None else str(value):>{width}s}"


def summary_over_seeds(rows):
    """mean, sd (population), min, max and n of every numeric item of the rows, over the rows that have it."""
    out = {}
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if values:
            array = np.array(values)
            out[key] = {"mean": float(array.mean()), "sd": float(array.std()), "min": float(array.min()),
                        "max": float(array.max()), "n": len(values)}
    for key in ("within_20", "within_45"):
        # the measured fraction less the Fisher-axial and Watson predictions at each run's own K
        for law in ("fisher", "watson"):
            gaps = [row[f"{key}_deg"] - row[f"{key}_{law}"] for row in rows
                    if row.get(f"{key}_deg") is not None and row.get(f"{key}_{law}") is not None]
            if gaps:
                out[f"{key}_minus_{law}"] = {"mean": float(np.mean(gaps)), "sd": float(np.std(gaps)), "n": len(gaps)}
    return out


B1_LINES = (
    ("counts", ("lanes", "knots", "candidates", "planned_candidates", "link_sites_angle", "link_candidates", "links",
                "links_kept", "link_acceptance", "links_of_sites", "links_kept_of_sites", "link_sites_over_planned")),
    ("density", ("crossing_density_d", "area_source", "n_over_area_d", "chord_graph_density_d", "density_identity",
                 "excess_rel", "predicted_excess_rel", "predicted_excess_rel_kept", "chord_link_share",
                 "nn_median_d", "nn_mean_d", "nn_ratio", "density_cv")),
    ("orientation", ("crossing_ratio", "chord_J", "chord_mc", "chord_interior", "S", "watson_K",
                     "fisher_axial_K_exact", "within_20_deg", "within_20_fisher", "within_20_watson",
                     "within_45_deg", "within_45_fisher", "within_45_watson", "theta_max_deg")),
    ("connections", ("cc_per_length_d", "end_spacing_d", "end_spacing_over_link_spacing", "cc_length_share", "junctions_count", "junction_T",
                     "junction_Y", "junction_other", "axial_median_d", "axial_mean_d", "arc_chord_mean",
                     "arc_chord_median")),
    ("tissue", ("tissue_mean_d", "tissue_p90_d", "tissue_max_d", "tissue_spacing", "tissue_source",
                "length_density_d2", "volume_source", "lane_length_kept_fraction", "run_prediction")),
    ("walk and cost", ("walked", "chord", "dropped", "redraws", "seconds", "seconds_greedy", "seconds_walk",
                       "describe_seconds", "peak_rss_mb", "peak_rss_mb_with_describe")),
)
B3_LINES = (
    ("feeders", ("on_feeders_knots", "on_feeders_agrees", "lane_share_feeders", "lane_share_refused",
                 "lane_share_peeled", "lane_share_islands", "lane_share_after_walk", "lane_share_kept",
                 "bed_components", "bed_components_one_tree", "whole_components", "tips_eligible", "tips_attached",
                 "tip_edges", "seconds_per_1e6_volume_d3")),
    ("whole network", ("whole_crossing_density_d", "whole_area_source", "whole_crossing_ratio", "whole_nn_ratio",
                       "whole_cc_per_length_d", "whole_tissue_mean_d", "whole_tissue_p90_d", "whole_tissue_max_d",
                       "whole_length_density_d2", "whole_describe_seconds")),
)
FOAM_LINES = (
    ("seeds", ("seeds", "seeds_formula", "seeds_agree", "planned_seeds", "kept_seeds", "tips_eligible",
               "tips_attached", "bed_components", "bed_components_one_tree", "whole_components")),
    ("bed", ("crossing_ratio", "crossing_density_d", "area_source", "S", "watson_K", "fisher_axial_K_exact",
             "within_20_deg", "within_45_deg", "arc_chord_mean", "chain_length_d", "length_density_d2",
             "tissue_mean_d", "tissue_p90_d", "tissue_max_d")),
    ("whole network", ("whole_crossing_ratio", "whole_crossing_density_d", "whole_area_source",
                       "whole_length_density_d2", "whole_tissue_mean_d")),
    ("cost", ("walked", "chord", "dropped", "redraws", "seconds", "seconds_greedy", "seconds_walk",
              "seconds_per_1e6_volume_d3", "describe_seconds", "peak_rss_mb")),
)


def print_runs(title, entries, lines, baseline):
    print(f"\n{title} (peak RSS less {baseline or 0.0:.0f} MB of imports in the tables' net figures)")
    for entry in entries:
        task = entry["task"]
        if "ok" not in entry:
            print(f"  {task['label']} seed {task['seed']}: "
                  + (f"stopped after {entry['timed_out']:g} s, planned lanes or seeds {entry['planned_seeds']}, "
                     f"planned candidates {entry['planned_candidates']}" if "timed_out" in entry
                     else "error\n" + entry["error"]))
            continue
        row = entry["ok"]["row"]
        laws = entry["ok"]["laws"]
        print(f"  {task['label']} seed {task['seed']}: laws {'hold' if laws is None else laws}")
        print("    counters      " + " ".join(f"{key[4:]}={value}" for key, value in entry["ok"]["events"].items()
                                              if value))
        for name, keys in lines:
            print(f"    {name:13s} " + " ".join(f"{key}={_fmt(row.get(key), 1, 4).strip()}" for key in keys))


def print_summary(title, groups, keys, entries):
    sources = sorted({str(entry["ok"]["row"].get(key)) for entry in entries if "ok" in entry
                      for key in ("area_source", "whole_area_source") if entry["ok"]["row"].get(key)})
    print(f"\n{title}: mean (sd) over seeds; crossing densities per d_min^2 over areas from {', '.join(sources)}")
    print("  setting" + " " * 52 + " ".join(f"{key[:14]:>16s}" for key in keys))
    for label, summary in groups.items():
        cells = []
        for key in keys:
            item = summary.get(key)
            cells.append(f"{item['mean']:9.4g} ({item['sd']:.2g})" if item else " " * 15 + "-")
        print(f"  {label:58s} " + " ".join(f"{cell:>16s}" for cell in cells))


SUMMARY_KEYS = ("crossing_density_d", "n_over_area_d", "excess_rel", "predicted_excess_rel", "nn_ratio",
                "crossing_ratio", "chord_mc", "chord_interior", "fisher_axial_K_exact", "watson_K",
                "within_20_minus_fisher", "within_45_minus_fisher", "end_spacing_d", "cc_per_length_d",
                "axial_mean_d", "tissue_mean_d", "length_density_d2", "link_acceptance", "seconds", "peak_rss_mb")


def grouped(entries):
    """{setting label: summary over its seeds} of the runs that finished."""
    groups = {}
    for entry in entries:
        if "ok" in entry:
            label = entry["task"]["label"]
            groups.setdefault(label, []).append(entry["ok"]["row"])
    return {label: summary_over_seeds(rows) for label, rows in groups.items()}


def _entry_record(entry):
    task = entry["task"]
    record = {"label": task["label"], "seed": task["seed"], "wall_seconds": entry["wall_seconds"]}
    if "ok" in entry:
        record.update(entry["ok"])
    else:
        record.update({key: entry[key] for key in entry if key not in ("task", "wall_seconds")})
    return record


# --------------------------------------------------------------------------
# The benchmarks


class _Blocks:
    """What every block does with its runs (a mixin of the test classes): the laws, sway's invariance, the tables
    and the JSON."""

    def block_seeds(self, block):
        blocks = selected_blocks()
        if block not in blocks:
            self.skipTest(f"block {block} not named in VSYSTEM_FIBRE_BLOCKS")
        return blocks[block]

    def check_and_record(self, section, block, entries, lines):
        baseline = baseline_rss_mb()
        print_runs(f"{section} {block}", entries, lines, baseline)
        for entry in entries:
            with self.subTest(run=entry["task"]["label"], seed=entry["task"]["seed"]):
                # a run that passes its limit is reported and left out; any other failure fails
                self.assertNotIn("error", entry, entry.get("error"))
                if "ok" in entry:
                    self.assertIsNone(entry["ok"]["laws"])
                    entry["ok"]["row"]["peak_rss_mb_net"] = entry["ok"]["row"]["peak_rss_mb"] - (baseline or 0.0)
        summaries = grouped(entries)
        _record(section, block, {"runs": {f"{e['task']['label']} s{e['task']['seed']}": _entry_record(e)
                                          for e in entries},
                                 "summary": summaries, "baseline_rss_mb": baseline})
        return summaries

    def check_sway_invariance(self, entries):
        """The counts sway does not draw agree across the sways of each setting and seed."""
        groups = {}
        for entry in entries:
            if "ok" not in entry:
                continue
            settings = dict(entry["ok"]["settings"])
            settings.pop("sway")
            key = json.dumps(settings, sort_keys=True) + f" seed {entry['task']['seed']}"
            groups.setdefault(key, []).append(entry["ok"])
        compared = 0
        for key, runs in groups.items():
            for run in runs[1:]:
                with self.subTest(setting=key, sway=run["settings"]["sway"]):
                    self.assertEqual(run["invariant"], runs[0]["invariant"])
                    compared += 1
        print(f"  counts independent of sway: {compared} runs compared with the first sway of their setting")
        return compared


def _alone_task(settings, seed, side):
    return {"label": settings_label(bed.parse_settings(settings)), "settings": settings, "seed": seed,
            "domain": [[0.0, 0.0, 0.0], [side] * 3], "tag": "alone"}


@SLOW
class FibreBedAloneBenchmark(_Blocks, unittest.TestCase):
    """B1: a bed alone in a cube of side 96 d_min along x, d_min 1 and margin 1, block by block."""

    def b1(self, block):
        seeds = self.block_seeds(block)
        side = b1_side()
        tasks = [_alone_task(settings, seed, side) for settings in b1_settings(block) for seed in seeds]
        print(f"\nB1 {block}: {len(tasks)} runs in a cube of side {side:g}")
        entries = run_tasks(tasks)
        summaries = self.check_and_record("b1", block, entries, B1_LINES)
        print_summary(f"B1 {block}", summaries, SUMMARY_KEYS, entries)
        return entries

    def test_the_grid_of_spacing_and_density(self):
        self.b1("grid")

    def test_sway_and_knot(self):
        entries = self.b1("sway")
        self.assertGreater(self.check_sway_invariance(entries), 0)

    def test_link_spacing(self):
        self.b1("links")

    def test_persistence(self):
        self.b1("persistence")

    def test_run(self):
        self.b1("run")

    def test_the_dense_block(self):
        entries = self.b1("dense")
        self.assertGreater(self.check_sway_invariance(entries), 0)


_MEMBERS = {}
_FORESTS = {}
_FOREST_LIBRARY = []


def mesh_member(seed):
    """The mesh member of ratio 8: its growth box from the origin, its nodes and labels, and the frame along x."""
    if seed not in _MEMBERS:
        grown = library.grow_member("mesh", MEMBER_RATIO, seed)["grown"]
        box = np.asarray(grown["growth_box_um"], dtype=float)
        _MEMBERS[seed] = {"domain": [[0.0, 0.0, 0.0], box.tolist()], "feeders": (grown["nodes"], grown["tree"]),
                          "frame": unguided_frame((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), box / 2.0)}
    return _MEMBERS[seed]


def packed_forest(seed):
    """
    tests/test_join.py's typical forest in a cube of side FOREST_SIDE at its
    packing, placed with default_rng(seed), cropped with r_max + 1 and
    joined into one feeder array of label 0.
    """
    if seed not in _FORESTS:
        if not _FOREST_LIBRARY:
            _FOREST_LIBRARY.extend(grow(s, 5, subdivisions=2) for s in FOREST_LIBRARY_SEEDS)
        lo, hi = np.zeros(3), np.full(3, FOREST_SIDE)
        count = int(round(FOREST_PACKING[0] * (FOREST_SIDE / FOREST_PACKING[1]) ** 3))
        forest = build_forest(_FOREST_LIBRARY, count, lo, hi, np.random.default_rng(seed), clearance_gap=2.0,
                              tries=60)
        r_max = max(float(np.nanmax(n["nodes"][3])) for n in forest) / 2.0
        cropped = [join.crop_network(n, lo, hi, r_max + MARGIN)["nodes"] for n in forest]
        pieces = []
        for nodes in cropped:
            if nodes.shape[1]:
                if pieces:
                    pieces.append(np.full((4, 1), np.nan))
                pieces.append(nodes)
        nodes = np.concatenate(pieces, axis=1)
        tree = np.where(np.isfinite(nodes[:3]).all(axis=0), 0, -1).astype(np.int8)
        _FORESTS[seed] = {"domain": [lo.tolist(), hi.tolist()], "feeders": (nodes, tree), "frame": None,
                          "placed": len(forest), "count": count, "r_max": r_max, "columns": int(nodes.shape[1])}
    return _FORESTS[seed]


@SLOW
class FibreFieldFillBenchmark(_Blocks, unittest.TestCase):
    """B3: a fibre bed filled between the feeders of a mesh member and of a packed forest, and foam beside it."""

    def test_member_feeders(self):
        seeds = self.block_seeds("member")
        tasks = []
        for seed in seeds:
            member = mesh_member(seed)
            for spacing, density in MEMBER_FILLS:
                settings = dict(FIBRE, spacing=spacing, density=density)
                label = settings_label(bed.parse_settings(settings))
                tasks.append({"label": f"member R {MEMBER_RATIO:g}, {label}", "settings": settings, "seed": seed,
                              "domain": member["domain"], "feeders": member["feeders"], "frame": member["frame"],
                              "tag": f"member {seed}"})
                tasks.append({"label": f"member box alone, {label}", "settings": settings, "seed": seed,
                              "domain": member["domain"], "frame": member["frame"], "tag": "member box alone"})
        print(f"\nB3 member feeders: {len(tasks)} runs, mesh members of ratio {MEMBER_RATIO:g} on seeds {seeds}")
        entries = run_tasks(tasks)
        summaries = self.check_and_record("b3", "member", entries, B1_LINES + B3_LINES)
        print_summary("B3 member feeders", summaries, SUMMARY_KEYS + ("lane_share_feeders", "lane_share_peeled",
                                                                       "lane_share_islands", "tips_attached"), entries)

    def test_foam_beside_the_member(self):
        seeds = self.block_seeds("foam")
        settings = dict(main.FAMILIES["foam"]["bed"], stretch=FOAM_STRETCH, density=FOAM_DENSITY)
        tasks = []
        for seed in seeds:
            member = mesh_member(seed)
            tasks.append({"label": f"member R {MEMBER_RATIO:g}, {settings_label(bed.parse_settings(settings))}",
                          "settings": settings, "seed": seed, "domain": member["domain"],
                          "feeders": member["feeders"], "frame": member["frame"], "tag": f"member {seed}"})
        print(f"\nB3 foam beside the member: {len(tasks)} runs")
        entries = run_tasks(tasks)
        summaries = self.check_and_record("b3", "foam", entries, FOAM_LINES)
        print_summary("B3 foam beside the member", summaries,
                      ("crossing_ratio", "seeds", "seeds_formula", "crossing_density_d", "chain_length_d",
                       "tips_attached", "seconds", "peak_rss_mb"), entries)

    def test_a_packed_forest(self):
        seeds = self.block_seeds("forest")
        tasks = []
        for seed in seeds:
            forest = packed_forest(seed)
            print(f"  forest seed {seed}: {forest['placed']} of {forest['count']} networks placed, "
                  f"{forest['columns']} columns after the crop, r_max {forest['r_max']:.2f}")
            _record("b3", f"forest_{seed}", {key: forest[key] for key in ("placed", "count", "columns", "r_max")})
            for spacing, density in FOREST_FILLS:
                settings = dict(FIBRE, spacing=spacing, density=density)
                tasks.append({"label": f"forest, {settings_label(bed.parse_settings(settings))}",
                              "settings": settings, "seed": seed, "domain": forest["domain"],
                              "feeders": forest["feeders"], "frame": None, "tag": f"forest {seed}"})
        print(f"\nB3 packed forest: {len(tasks)} runs in a cube of side {FOREST_SIDE:g}")
        entries = run_tasks(tasks)
        summaries = self.check_and_record("b3", "forest", entries, B1_LINES + B3_LINES)
        print_summary("B3 packed forest", summaries, SUMMARY_KEYS + ("lane_share_feeders", "lane_share_peeled",
                                                                      "lane_share_islands", "tips_attached"), entries)


@SLOW
class FibreCostBenchmark(_Blocks, unittest.TestCase):
    """The cost of a dense bed alone (spacing 3.2, density 0.6, links on) in cubes of side 128 and 256 d_min."""

    def test_a_dense_bed_in_a_256_cube(self):
        self.block_seeds("cost256")
        baseline = baseline_rss_mb()
        settings = dict(FIBRE, **COST_SETTINGS)
        rows = {}
        for side in COST_SIDES:
            task = dict(_alone_task(settings, 1, side), describe=False, tag="cost")
            task["label"] = f"cost {side:g}^3, {task['label']}"
            limit = cost_limit() if side == max(COST_SIDES) else run_limit()
            rows[f"fibre_{side:g}"] = run_tasks([task], limit=limit)[0]
        foam = bed.parse_settings(main.FAMILIES["foam"]["bed"])
        side = bed_cube(COST_FOAM_SEEDS, foam)
        task = {"label": f"cost foam {COST_FOAM_SEEDS} seeds, side {side:.2f}", "settings": foam, "seed": 1,
                "domain": [[0.0, 0.0, 0.0], [side] * 3], "describe": False, "tag": "cost"}
        rows["foam"] = run_tasks([task])[0]
        print(f"\ncost of a bed alone, one process at a time (peak RSS less {baseline or 0.0:.0f} MB of imports):")
        print("  run                                                        seconds   rss_MB  net_MB  knots/seeds  "
              "chain_d  s/1e5_chain_d  stages")
        record = {}
        for name, entry in rows.items():
            with self.subTest(run=name):
                self.assertNotIn("error", entry, entry.get("error"))
            if "ok" not in entry:
                print(f"  {entry['task']['label']}: stopped after {entry.get('timed_out')} s, planned lanes "
                      f"{entry.get('planned_seeds')}, planned candidates {entry.get('planned_candidates')}")
                record[name] = _entry_record(entry)
                continue
            with self.subTest(run=name):
                self.assertIsNone(entry["ok"]["laws"])
            row = entry["ok"]["row"]
            row["peak_rss_mb_net"] = row["peak_rss_mb"] - (baseline or 0.0)
            count = row.get("knots", entry["ok"]["events"]["bed_seeds"])
            print(f"  {entry['task']['label'][:58]:58s} {row['seconds']:8.1f} {row['peak_rss_mb']:8.0f} "
                  f"{row['peak_rss_mb_net']:7.0f} {count:12d} {row['chain_length_d']:9.0f} "
                  f"{_fmt(row['seconds_per_1e5_chain_d'], 13, 2)}  "
                  + " ".join(f"{stage}={value:.1f}" for stage, value in entry["ok"]["timings"].items()))
            record[name] = _entry_record(entry)
        largest, smaller = rows[f"fibre_{max(COST_SIDES):g}"], rows[f"fibre_{min(COST_SIDES):g}"]
        if "ok" not in largest and "ok" in smaller:
            # linear in the planned candidate edges, from the smaller cube's seconds
            extrapolated = smaller["ok"]["row"]["seconds"] * largest["planned_candidates"] \
                / smaller["ok"]["row"]["planned_candidates"]
            record["extrapolated_seconds_256"] = extrapolated
            print(f"  the {max(COST_SIDES):g} cube, extrapolated linearly in planned candidates from the "
                  f"{min(COST_SIDES):g} cube: {extrapolated:.0f} s")
        if "ok" in largest and "ok" in rows["foam"]:
            ratio = largest["ok"]["row"]["seconds_per_1e5_chain_d"] / rows["foam"]["ok"]["row"]["seconds_per_1e5_chain_d"]
            record["seconds_per_chain_length_over_foam"] = ratio
            print(f"  seconds per 1e5 d_min of placed chain length, fibre {max(COST_SIDES):g}^3 over foam: {ratio:.2f}")
        record["baseline_rss_mb"] = baseline
        _record("cost", "cost256", record)


if __name__ == "__main__":
    unittest.main()
