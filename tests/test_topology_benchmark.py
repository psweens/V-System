"""
Benchmark of the families against each other: growth cost and the
descriptors of every available family at three root ratios, the aligned
preset over its guidance length and onset, the explicit bed of each foam
network with the attachment of its feeders' tips, the cost of a bed against
its size, and a projection of the cost of a 2000-network library.

Slow: it grows several dozen networks, some of them large, and runs only
with VSYSTEM_SLOW_TESTS=1. It prints its tables and writes them as JSON to
the path in VSYSTEM_BENCHMARK_JSON when that is set. It asserts sanity only:
that the capillaries of `aligned` are more ordered about its frame axis than
those of a tree described against the same axis, that the counters
partition what they count, that a bed's counters keep the laws bed.py
states and are zero for every family without a bed, that no vertex of a
foam network has more than three vessels, and that a bed has no cycle
shorter than its girth.

Run from the repository root with
    VSYSTEM_SLOW_TESTS=1 python -m unittest tests.test_topology_benchmark
The families that cost far more per network run in invocations of their
own, foam with the cost of a bed against its size:
    VSYSTEM_SLOW_TESTS=1 VSYSTEM_BENCHMARK_FAMILIES=foam python -m unittest tests.test_topology_benchmark
"""
import json
import math
import multiprocessing
import os
import random
import resource
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bed  # noqa: E402
import graph  # noqa: E402
import library  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from describe import describe, describe_archive, load_archive  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmark")
RATIOS = (4.0, 8.0, 16.0)
COST_RATIO = 25.0
# the families filled with capillary trees or joined by an explicit bed cost
# far more per network, so they are benchmarked in their own invocation, R by
# R: VSYSTEM_BENCHMARK_FAMILIES names the families (default: every available
# one but these) and VSYSTEM_BENCHMARK_RATIOS the ratios, the last one grown
# for cost only on one seed (default: 4 8 16 25); the cost of a bed against
# its size (BedCostBenchmark) runs when the families named include foam
SEPARATE_FAMILIES = ("aligned_bed", "capillary_bed", "foam")


def benchmark_families():
    named = os.environ.get("VSYSTEM_BENCHMARK_FAMILIES")
    if named:
        return named.replace(",", " ").split()
    return [family for family in library.available_families() if family not in SEPARATE_FAMILIES]


def benchmark_ratios():
    named = os.environ.get("VSYSTEM_BENCHMARK_RATIOS")
    if named:
        values = [float(value) for value in named.replace(",", " ").split()]
        return tuple(values[:-1]), values[-1]
    return RATIOS, COST_RATIO
SEEDS = (1, 2, 3)
PERSISTENCE = main.FAMILIES["aligned"]["persistence"]
LENGTHS = (2.0, 4.9, 12.3, 16.4)            # kappa = 4 P / G of about 20, 8.2, 3.3 and 2.4
ONSETS = (0.0, 2.0)
# the side of the central cube a bed's tissue distance is also sampled over, in
# d_min: describe's 64^3 points at a spacing of 1 d_min
TISSUE_CUBE = 64.0
# the classes of a feeder tip's diameter, in d_min: the grammar stops below 2,
# so a tip of the first class would be a stenosis middle
TIP_CLASSES = (("<2", 0.0, 2.0), ("[2,2.52)", 2.0, 2.52), ("[2.52,4)", 2.52, 4.0), (">=4", 4.0, math.inf))
# the cost of one network above which a per-family ratio cap is worth having
NETWORK_SECONDS_LIMIT = 3600.0
NETWORK_RSS_MB_LIMIT = 8192.0
# BedCostBenchmark: bed.build alone, with the foam settings, on cubes planned
# to hold these numbers of seeds, over SEEDS up to BED_ALL_SEEDS_UP_TO and on
# the first seed above it; and its girth and its density swept at
# BED_SWEEP_SEEDS on the first seed
BED_SIZES = (1000, 3000, 10000, 30000, 100000)
BED_ALL_SEEDS_UP_TO = 30000
BED_SWEEP_SEEDS = 10000
BED_GIRTHS = (6, 8, 10)
BED_DENSITIES = (0.4, 0.5, 0.6)
# the longest cycle through a bed edge that BedCostBenchmark looks for, in
# edges, as describe looks for loops of up to 17 segments
BED_LONGEST_CYCLE = 17

_results = {}


def _record(section, key, value):
    _results.setdefault(section, {})[key] = value


def grow_preset(family, ratio, seed, **override):
    """A member grown in this process, as library.grow_member grows it, with grow_network overrides."""
    argv = library.member_argv(family, ratio)
    args = main.build_parser(family).parse_args(argv)
    main.validate_shaping(args)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    properties, d0, niter = main.sample_parameters(args)
    kwargs = {"tVol": tuple(args.volume), "fit": args.fit, "clip_axes": tuple(args.clip_axes),
              "voxel_size": args.voxel_size, "subdivisions": args.subdivisions, "d_min": args.d_min,
              "grow_in_volume": args.grow_in_volume, "seed": seed}
    kwargs.update(main.shaping_options(args))
    kwargs.update(override)
    started = time.perf_counter()
    grown = main.grow_network(niter, d0, properties, **kwargs)
    grown["seconds"] = time.perf_counter() - started
    return grown


def capillary_summary(report):
    """The frame-relative and class descriptors of a report, flattened for a table."""
    out = {}
    orientation = report["frame_orientation"]
    if orientation:
        for cls in ("capillary", "larger"):
            entry = orientation[cls]
            if entry:
                for key in ("S", "mean_abs_cos", "crossing_ratio", "watson_K", "fisher_axial_K",
                            "fisher_axial_K_exact", "within_45_deg"):
                    out[f"{cls}_{key}"] = entry.get(key)
    polar = report["polar_order"]
    if polar:
        out["capillary_polar_order"] = polar["capillary"]
    shares = report["calibre_shares"]
    if shares:
        out["length_below_1.5"] = shares["length"]["below_1.5"]
        out["length_below_2"] = shares["length"]["below_2"]
        out["volume_below_1.5"] = shares["volume"]["below_1.5"]
        out["volume_below_2"] = shares["volume"]["below_2"]
    segments = report["segments_by_class"]
    if segments and segments["capillary"]["count"]:
        out["capillary_segment_median"] = segments["capillary"]["median"]
        out["capillary_segment_cv"] = segments["capillary"]["cv"]
    spacing = report["transverse_spacing"]
    if spacing:
        out["spacing_median"] = spacing["median"]
        out["crossing_density"] = spacing["crossing_density"]
        out["crossings"] = spacing["count"]
    out["arc_chord_mean"] = report["arc_chord"]["mean"]
    junctions = report.get("junctions")
    if junctions and junctions["count"]:
        for key in ("count", "degree_3", "degree_4", "degree_5_plus", "mean_degree", "segments_per_junction"):
            out[f"junctions_{key}"] = junctions[key]
    angles = report.get("branch_angles_deg")
    if angles and angles["count"]:
        for key in ("min", "median", "max"):
            out[f"branch_angle_{key}"] = angles[key]
    loops = report.get("loops")
    if loops:
        for kind in ("by_segment", "by_node"):
            for key in ("found", "none_fraction", "median", "mean"):
                out[f"loops_{kind}_{key}"] = loops[kind].get(key)
        out["loops_by_node_length_median_d"] = loops["by_node"].get("length_median_d")
        out["cycles_per_length_d"] = loops["by_segment"].get("cycles_per_length_d")
    variation = report.get("segment_diameter_variation")
    if variation:
        for cls in ("all", "capillary"):
            if variation.get(cls):
                out[f"diameter_variation_{cls}_median"] = variation[cls]["median"]
                out[f"diameter_variation_{cls}_p90"] = variation[cls]["p90"]
    tissue = report.get("tissue_distance")
    if tissue:
        for key in ("spacing", "inside_fraction", "mean_d", "median_d", "p90_d", "max_d"):
            out[f"tissue_{key}"] = tissue.get(key)
        out["tissue_domain"] = tissue["domain"]["source"]
    return out


def bridge_geometry(grown):
    """
    The bridges of a grown network: for each, the chord's angle to x, whether
    its first and last segments point the same way along x (a through
    connection runs within 45 degrees of x without a reversal), and whether
    it joins the two trees.
    """
    nodes, tree = grown["nodes"], grown["tree"]
    rows = []
    finite = np.isfinite(nodes[0])
    labels = tree == BRIDGE
    start = None
    for k in range(nodes.shape[1] + 1):
        inside = k < nodes.shape[1] and finite[k] and labels[k]
        if inside and start is None:
            start = k
        elif not inside and start is not None:
            points = nodes[:3, start:k]
            if points.shape[1] >= 3:
                chord = points[:, -1] - points[:, 0]
                first = points[:, 1] - points[:, 0]
                last = points[:, -1] - points[:, -2]
                angle = math.degrees(math.acos(min(1.0, abs(chord[0]) / max(np.linalg.norm(chord), 1e-300))))
                rows.append({"angle_to_x_deg": angle, "same_way": bool(np.sign(first[0]) == np.sign(last[0])),
                             "chord": float(np.linalg.norm(chord))})
            start = None
    # the anastomosis bridges come first, then the rungs, each a polyline of its own
    for row, bridge in zip(rows, grown["bridges"]):
        row["kind"] = "bridge"
        row["cross_tree"] = bridge["tip_tree"] != bridge["partner_tree"]
    for row, rung in zip(rows[len(grown["bridges"]):], grown.get("rungs", [])):
        row["kind"] = "rung"
        row["cross_tree"] = rung["site_tree"] != rung["partner_tree"]
    return rows


def capillary_reach(grown):
    """How far the capillaries of each tree reach along x from their root, in d_min."""
    nodes, edges, tree = grown["nodes"], grown["edges"], grown["tree"]
    diameter = graph.vertex_diameter(nodes, graph.canonical_columns(nodes))
    mean = (diameter[edges[0]] + diameter[edges[1]]) / 2.0
    reach = []
    for k, sense in enumerate(grown["frame"]["tree_senses"]):
        root_x = grown["root_positions_um"][k][0]
        columns = edges[1][(mean < 2.0) & (tree[edges[1]] == k)]
        if columns.size:
            reach.append(float(np.max(sense * (nodes[0, columns] - root_x))))
        else:
            reach.append(None)
    return reach


def has_bed(family):
    """Whether a family's preset grows an explicit bed."""
    return (main.FAMILIES.get(family) or {}).get("bed") is not None


def statistics(values):
    """count, mean, median, p90 and max of a list of numbers, None where there is none."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max())}


def bed_laws(events, planned, attempts):
    """
    The laws of bed.py that a network's counters can be held to on their
    own, True where each holds: E1, E2 against the planned seed count, E3,
    E4's inequality, the first half of E7, and E9 with the walk's attempts
    per chain. Counters missing from `events` (an index row keeps those that
    are not zero) count 0.
    """
    count = {key: events.get(key, 0) for key in bed.EVENT_KEYS}
    refused = ("bed_refused_degree", "bed_refused_angle", "bed_refused_girth", "bed_refused_clearance", "bed_edges")
    return {
        "E1": count["bed_proposals"] == (count["bed_proposals_on_feeders"] + count["bed_proposals_overlapping"]
                                         + count["bed_seeds"]),
        "E2": count["bed_seeds"] == planned,
        "E3": (count["bed_candidates"] == sum(count[key] for key in refused)
               and count["bed_candidates_tip"] <= count["bed_candidates"]),
        "E4": count["bed_refused_clearance_at_end"] <= count["bed_refused_clearance"],
        "E7": count["bed_segments"] == (count["bed_segments_walked"] + count["bed_segments_chord"]
                                        + count["bed_segments_dropped"]),
        "E9": (count["bed_components_one_tree"] <= count["bed_components"]
               and count["bed_walk_redraws"] <= attempts * count["bed_segments"]),
    }


def bed_only(nodes, tree):
    """
    The bed of an archive on its own: a copy of the nodes with every finite
    column not labelled BRIDGE set to NaN, and the labels with those columns
    marked as separators. describe drops the empty runs; the bed's polylines
    keep their bitwise ends, and only the feeder side of the junctions where
    the bed attaches is lost.
    """
    tree = np.asarray(tree)
    feeder = np.isfinite(nodes[:3]).all(axis=0) & (tree != BRIDGE)
    view = np.array(nodes, dtype=float, copy=True)
    view[:, feeder] = np.nan
    return view, np.where(feeder, -1, tree).astype(tree.dtype)


def bed_polylines(nodes, tree):
    """The polylines of an archive's bed: the runs of columns labelled BRIDGE, one per chain placed."""
    tree = np.asarray(tree)
    return [run for run in graph.polylines(nodes) if tree[run[0]] == BRIDGE]


def bed_arc_chord(nodes, tree):
    """The arc over the chord of the bed's open polylines, walked or straight, and the number closed."""
    ratios, closed = [], 0
    for run in bed_polylines(nodes, tree):
        points = nodes[:3, run]
        chord = float(np.linalg.norm(points[:, -1] - points[:, 0]))
        if chord > 0.0:
            ratios.append(float(np.linalg.norm(np.diff(points, axis=1), axis=0).sum()) / chord)
        else:
            closed += 1
    return dict(statistics(ratios), closed=closed)


def bed_excursions(nodes, tree, box, d_min):
    """How far the bed's points lie outside the growth box from 0 to `box`, in d_min: a walk is not confined to it."""
    on_bed = np.isfinite(nodes[:3]).all(axis=0) & (np.asarray(tree) == BRIDGE)
    points = nodes[:3, on_bed].T
    beyond = np.maximum(np.maximum(-points, points - np.asarray(box, dtype=float)), 0.0).max(axis=1)
    outside = int(np.count_nonzero(beyond > 0.0))
    return {"points": int(points.shape[0]), "outside": outside,
            "share": outside / points.shape[0] if points.shape[0] else None,
            "max_d": float(beyond.max()) / d_min if beyond.size else 0.0}


def tissue_summary(tissue):
    """A tissue distance's figures in d_min, with the spacing and the box it was sampled over."""
    if not tissue:
        return None
    entry = {key: tissue.get(key) for key in ("spacing", "raised", "inside_fraction", "mean_d", "median_d", "p90_d",
                                               "max_d")}
    entry["source"] = tissue["domain"]["source"]
    return entry


def feeder_attachment(nodes, tree, collision_margin, d_min):
    """
    How an archive's bed attaches to the tips of its feeders, recomputed
    from the archive alone. The eligible tips and the stubs are those
    bed._feeder_tips finds on the feeder columns, the bed's columns set
    aside. An edge of the bed joins the vertices of two consecutive columns
    of a bed polyline, and a tip's k_t is the number of bed edges at it.
    Counted over the eligible tips, by tree and by the class of the tip's
    diameter (TIP_CLASSES); with the components of the bed graph (its edges
    and the tips they reach) and those whose attached tips belong to both
    trees, the bed polylines whose two ends are tips, the bed junctions (a
    vertex of three or more bed edges that is no feeder column) all of whose
    polylines end on tips, and the tips attached within a component that
    holds a bed junction.
    """
    tree = np.asarray(tree)
    n = nodes.shape[1]
    on_bed = np.isfinite(nodes[:3]).all(axis=0) & (tree == BRIDGE)
    feeders = np.array(nodes, dtype=float, copy=True)
    feeders[:, on_bed] = np.nan
    tips, stubs = bed._feeder_tips(feeders, np.where(on_bed, -1, tree), collision_margin)
    canonical = graph.canonical_columns(nodes)
    pairs = np.flatnonzero(on_bed[:-1] & on_bed[1:])
    low = np.minimum(canonical[pairs], canonical[pairs + 1])
    high = np.maximum(canonical[pairs], canonical[pairs + 1])
    code = np.unique(low[low != high] * n + high[low != high])
    edges = np.stack([code // n, code % n]).astype(np.int64)
    degree = np.bincount(edges.ravel(), minlength=n)
    labels, _ = graph.components(edges, canonical)
    k = degree[tips]
    attached = k > 0
    tip_tree = tree[tips].astype(np.int64)
    diameter = nodes[3, tips] / d_min
    trees = sorted(int(label) for label in np.unique(tree[(tree >= 0) & (tree != BRIDGE)]))

    runs = bed_polylines(nodes, tree)
    first = canonical[np.array([run[0] for run in runs], dtype=np.int64)]
    last = canonical[np.array([run[-1] for run in runs], dtype=np.int64)]
    tip_to_tip = np.isin(first, tips) & np.isin(last, tips)
    junctions = np.flatnonzero((degree >= 3) & (tree == BRIDGE))
    ends, far = np.concatenate([first, last]), np.concatenate([last, first])
    inner = [run[1:-1] for run in runs if len(run) > 2]
    inner = canonical[np.concatenate(inner)] if inner else np.zeros(0, dtype=np.int64)
    # a junction inside a polyline is the junction of a lasso, which ends on itself
    only_tips = ~np.isin(junctions, ends[~np.isin(far, tips)]) & ~np.isin(junctions, inner)
    in_junction_component = attached & np.isin(labels[tips], labels[junctions])
    trees_of = {}
    for label, label_tree in zip(labels[tips][attached].tolist(), tip_tree[attached].tolist()):
        trees_of.setdefault(label, set()).add(label_tree)

    def split(mask):
        return {"eligible": int(np.count_nonzero(mask)), "attached": int(np.count_nonzero(attached & mask)),
                "tip_edges": int(k[mask].sum())}

    return {"eligible": int(tips.size), "stubs": int(stubs.size), "attached": int(np.count_nonzero(attached)),
            "tip_edges": int(k.sum()),
            "k_t": {str(value): int(np.count_nonzero(k == value))
                    for value in range(max(2, int(k.max(initial=0))) + 1)},
            "by_tree": {str(label): split(tip_tree == label) for label in trees},
            "by_calibre": {name: split((diameter >= lo) & (diameter < hi)) for name, lo, hi in TIP_CLASSES},
            "components": int(np.unique(labels[degree > 0]).size),
            "components_both_trees": sum(1 for found in trees_of.values() if len(found) > 1),
            "tip_to_tip_polylines": int(np.count_nonzero(tip_to_tip)),
            "tip_to_tip_polylines_two_trees": int(np.count_nonzero(tip_to_tip & (tree[first] != tree[last]))),
            "bed_junctions": int(junctions.size), "bed_junctions_tips_only": int(np.count_nonzero(only_tips)),
            "attached_in_junction_component": int(np.count_nonzero(in_junction_component))}


def pooled_attachment(entries):
    """
    The attachment counts of several members summed, with the attachment
    rate, the same counted only for tips attached within a component that
    holds a bed junction, and the members whose bed joins both trees.
    """
    total = {}

    def add(into, part):
        for key, value in part.items():
            if isinstance(value, dict):
                add(into.setdefault(key, {}), value)
            else:
                into[key] = into.get(key, 0) + value

    for entry in entries:
        add(total, entry["attachment"])
    eligible = total.get("eligible", 0)
    total["rate"] = total["attached"] / eligible if eligible else None
    total["rate_in_junction_component"] = total["attached_in_junction_component"] / eligible if eligible else None
    total["members"] = len(entries)
    total["members_joining_both_trees"] = sum(1 for entry in entries if entry["events"]["bed_components"]
                                              > entry["events"]["bed_components_one_tree"])
    total["share_joining_both_trees"] = total["members_joining_both_trees"] / len(entries) if entries else None
    return total


def guard_trip_ratio(family, cap=library.DEFAULT_BED_MAX_CANDIDATES):
    """
    The least root ratio at which a library member of a family with a bed
    plans more candidate edges than `cap`, in its cube of BOX_C R, to a
    relative 1e-6; None when that lies above R 1000.
    """
    settings, side = main.FAMILIES[family]["bed"], library.BOX_C[family]

    def over(ratio):
        return bed.planned_candidates(np.full(3, side * ratio), settings, 1.0) > cap

    low, high = library.DEFAULT_RATIO_RANGE[0], 1000.0
    if over(low):
        return low
    if not over(high):
        return None
    while high - low > 1e-6 * high:
        middle = (low + high) / 2.0
        if over(middle):
            high = middle
        else:
            low = middle
    return high


def bed_cube(count, settings, d_min=1.0):
    """The side of a cube in which a bed plans `count` seeds, midway between those planning one less and one more."""
    parsed = bed.parse_settings(settings)
    h, d = parsed["spacing"] * d_min, parsed["diameter"] * d_min
    side = d + ((count + 0.5) * parsed["stretch"] * h ** 3 / parsed["density"]) ** (1.0 / 3.0)
    planned = bed.planned_seeds(np.full(3, side), parsed, d_min)
    if planned != count:
        raise AssertionError(f"a cube of side {side!r} plans {planned} seeds, not {count}")
    return side


def bed_cycle_lengths(n_vertices, edges, longest=BED_LONGEST_CYCLE):
    """
    The shortest cycle through each edge of a bed grown alone, in edges: one
    more than the shortest path between its ends that does not use it, found
    by a breadth-first search from each end, longest // 2 levels deep; 0 when
    there is no cycle of at most `longest` edges.
    """
    adjacency = [[] for _ in range(n_vertices)]
    for a, b in np.asarray(edges).tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    depth = longest // 2

    def ball(source, u, v):
        distance, frontier = {source: 0}, [source]
        for level in range(1, depth + 1):
            following = []
            for x in frontier:
                for y in adjacency[x]:
                    if (x == u and y == v) or (x == v and y == u) or y in distance:
                        continue
                    distance[y] = level
                    following.append(y)
            frontier = following
            if not frontier:
                break
        return distance

    lengths = []
    for u, v in np.asarray(edges).tolist():
        near, far = ball(u, u, v), ball(v, u, v)
        if len(near) > len(far):
            near, far = far, near
        path = min((steps + far[w] for w, steps in near.items() if w in far), default=None)
        lengths.append(path + 1 if path is not None and path + 1 <= longest else 0)
    return lengths


def histogram(values):
    """{"value": count} over a list of integers, in increasing order of value."""
    values = [int(value) for value in values]
    return {str(value): values.count(value) for value in sorted(set(values))}


def own_peak_rss_mb():
    """
    This process's own peak resident set size, in MB: VmHWM where /proc
    gives it, ru_maxrss elsewhere. On Linux a spawned process's ru_maxrss
    starts at its parent's peak (exec folds the high-water mark of the
    memory it replaces into the process's), so a child of a large parent
    reads the parent's peak there.
    """
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _grow_task(task):
    result = library.grow_and_write(task)
    if "row" in result:
        # beside the row's peak_rss_mb, the library's ru_maxrss, which starts at this benchmark's peak
        result["row"]["own_peak_rss_mb"] = own_peak_rss_mb()
    return result


def _baseline_task(_):
    """The peak RSS of a fresh process that has imported what a bed's process imports, and done nothing else."""
    return own_peak_rss_mb()


def _bed_task(task):
    """
    One bed grown alone in a fresh process: bed.build in a cube from the
    origin, timed by stage, with the peak RSS once it is built; then
    graph.build and describe on its nodes, timed, and what the task asks to
    measure beside: the bed's cycle lengths in edges and describe's loops
    ("loops"), its capillary segments and its tissue distance over the
    central cube ("tissue").
    """
    settings = bed.parse_settings(task["settings"])
    side = float(task["side"])
    events, timings = {}, {}
    started = time.perf_counter()
    result = bed.build((np.zeros(3), np.full(3, side)), settings, d_min=1.0, seed=task["seed"],
                       collision_margin=1.0, events=events, timings=timings)
    seconds = time.perf_counter() - started
    peak_rss_mb = own_peak_rss_mb()
    nodes = result["nodes"]
    started = time.perf_counter()
    built = graph.build(nodes)
    graph_seconds = time.perf_counter() - started
    started = time.perf_counter()
    report = describe(nodes, built["edges"], d_ref=1.0)
    describe_seconds = time.perf_counter() - started
    out = {"count": task["count"], "seed": task["seed"], "settings": settings, "side": side, "seconds": seconds,
           "timings": timings, "peak_rss_mb": peak_rss_mb, "graph_seconds": graph_seconds,
           "describe_seconds": describe_seconds, "seeds": int(len(result["seeds"])),
           "planned_seeds": bed.planned_seeds(np.full(3, side), settings, 1.0),
           "planned_candidates": bed.planned_candidates(np.full(3, side), settings, 1.0),
           "kept_seeds": int(np.count_nonzero(result["kept"])), "greedy_edges": int(len(result["greedy_edges"])),
           "edges": int(len(result["edges"])), "chains": int(len(result["chains"])),
           "columns": int(nodes.shape[1]), "events": events}
    if "loops" in task["measure"]:
        out["cycles_in_edges"] = histogram(bed_cycle_lengths(len(result["seeds"]), result["edges"]))
        out["loops"] = {kind: {key: report["loops"][kind][key] for key in ("sampled", "found", "median", "histogram")}
                        for kind in ("by_segment", "by_node")}
    if "tissue" in task["measure"]:
        segments = report["segments_by_class"]["capillary"]
        out["capillary_segments"] = {key: segments[key] for key in ("count", "median_d", "p10_d", "p90_d", "cv")}
        out["tissue_distance"] = tissue_summary(describe(nodes, built["edges"], d_ref=1.0, volume=[TISSUE_CUBE] * 3,
                                                         evd_spacing=1.0)["tissue_distance"])
    out["peak_rss_mb_with_describe"] = own_peak_rss_mb()
    return out


def tearDownModule():
    # whatever ran, so that one class run on its own still writes its tables
    path = os.environ.get("VSYSTEM_BENCHMARK_JSON")
    if path and _results:
        with open(path, "w") as handle:
            json.dump(_results, handle, indent=1)


@SLOW
class FamilyBenchmark(unittest.TestCase):
    """Every available family at R 4, 8 and 16 over three seeds, grown as the library grows them."""

    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.mkdtemp()
        families = benchmark_families()
        cls.ratios, cost_ratio = benchmark_ratios()
        members = []
        for family in families:
            for ratio in cls.ratios + (cost_ratio,):
                for seed in SEEDS if ratio != cost_ratio else SEEDS[:1]:
                    members.append({"id": len(members), "family": family, "ratio": ratio, "bin": 0, "u": 0.0,
                                    "seed": seed, "file": f"bench_{len(members):03d}_{family}.npz"})
        settings = dict(library.growth_settings(library.build_parser().parse_args(
            ["--out", cls.out, "--count", "1", "--seed", "1", "--families"] + families)),
            ratio_lo=min(cls.ratios + (cost_ratio,)), ratio_hi=cost_ratio)
        tasks = [{"member": member, "settings": settings, "directory": cls.out} for member in members]
        for name in library.THREAD_VARIABLES:
            os.environ.setdefault(name, "1")
        context = multiprocessing.get_context("spawn")
        cls.rows = {}
        cls.failures = []
        with context.Pool(processes=max(1, min(4, os.cpu_count() or 1)), maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_grow_task, tasks):
                if "failure" in result:
                    cls.failures.append(result["failure"])
                else:
                    cls.rows[result["id"]] = result["row"]
        cls.members = {member["id"]: member for member in members}
        cls.descriptions = {}

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.out, ignore_errors=True)

    @classmethod
    def described(cls, ident):
        """
        A member's description, its tissue distance over its growth box
        included, with the seconds describe took without and with it;
        described once and kept for every test that reads it.
        """
        if ident not in cls.descriptions:
            path = os.path.join(cls.out, cls.members[ident]["file"])
            started = time.perf_counter()
            report = describe_archive(path)
            describe_seconds = time.perf_counter() - started
            started = time.perf_counter()
            tissue = describe_archive(path, evd_spacing=1.0)["tissue_distance"]   # over a member's growth box
            tissue_seconds = time.perf_counter() - started
            report["tissue_distance"] = tissue
            cls.descriptions[ident] = (report, describe_seconds, tissue_seconds)
        return cls.descriptions[ident]

    def test_families_grow_and_describe(self):
        # a member whose bed plans more candidate edges than the library's cap
        # is refused before it is grown, and reported; any other failure fails
        refused = [failure for failure in self.failures if failure["reason"].startswith("BedTooLarge")]
        if refused:
            _record("families", "bed_too_large", refused)
            print("\nrefused by the cap on a bed's candidate edges: "
                  + "; ".join(f"{f['family']} R {f['ratio']:g} seed {f['seed']}" for f in refused))
        others = [failure for failure in self.failures if failure not in refused]
        self.assertEqual(others, [], others)
        table = {}
        print("\nfamily   R  seed   points  seconds  rss_MB  cap_S  polar  len<2  vol<2  seg_med  spacing  density  "
              "deg3  loop_n  angle  evd_mean  events")
        for ident, row in sorted(self.rows.items()):
            member = self.members[ident]
            path = os.path.join(self.out, member["file"])
            report, describe_seconds, tissue_seconds = self.described(ident)
            summary = capillary_summary(report)
            events = {k: v for k, v in row["events"].items() if v}
            program = str(main.load_network(path)["program"])
            entry = {"family": member["family"], "ratio": member["ratio"], "seed": member["seed"],
                     "points": row["points"], "seconds": row["seconds"], "peak_rss_mb": row["peak_rss_mb"],
                     "describe_seconds": describe_seconds, "tissue_seconds": tissue_seconds,
                     "archive_bytes": os.path.getsize(path), "program_characters": len(program),
                     "program_moves": program.count("f("), "rungs": row.get("rungs"),
                     "events": events, "descriptors": summary, "cycles": row["cycles"], "tips": row["tips"]}
            table[f"{member['family']}_r{member['ratio']:g}_s{member['seed']}"] = entry
            if has_bed(member["family"]) and report["junctions"]["count"]:
                # a seed takes at most three bed edges and a feeder tip its stem and at most two
                self.assertEqual((report["junctions"]["degree_4"], report["junctions"]["degree_5_plus"]), (0.0, 0.0),
                                 ident)

            def fmt(key, width=6, digits=3):
                value = summary.get(key)
                return f"{value:{width}.{digits}f}" if isinstance(value, (int, float)) else " " * (width - 1) + "-"

            print(f"{member['family']:8s} {member['ratio']:3g} {member['seed']:3d} {row['points']:8d} "
                  f"{row['seconds']:8.1f} {row['peak_rss_mb']:7.0f} {fmt('capillary_S')} "
                  f"{fmt('capillary_polar_order')} {fmt('length_below_2')} {fmt('volume_below_2')} "
                  f"{fmt('capillary_segment_median', 7, 2)} {fmt('spacing_median', 7, 2)} "
                  f"{fmt('crossing_density', 8, 4)} {fmt('junctions_degree_3', 5, 2)} {fmt('loops_by_node_median', 6, 1)} "
                  f"{fmt('branch_angle_median', 6, 1)} {fmt('tissue_mean_d', 8, 2)}  "
                  f"{' '.join(f'{k}={v}' for k, v in sorted(events.items()))}")
        _record("families", "members", table)
        largest = max(table.values(), key=lambda entry: entry["points"])
        _record("families", "describe_largest", {key: largest[key] for key in ("family", "ratio", "seed", "points",
                                                                            "describe_seconds")})
        print(f"describe_archive on the largest network ({largest['family']} R {largest['ratio']:g} seed "
              f"{largest['seed']}, {largest['points']} points): {largest['describe_seconds']:.1f} s")
        # sanity: on the same R and seed the capillaries of aligned are more ordered about its axis
        # than a tree's described against that axis
        present = {member["family"] for member in self.members.values()}
        for ratio in self.ratios if {"aligned", "tree"} <= present else ():
            for seed in SEEDS:
                aligned = table[f"aligned_r{ratio:g}_s{seed}"]
                tree_member = next(m for m in self.members.values()
                                   if m["family"] == "tree" and m["ratio"] == ratio and m["seed"] == seed)
                archive = main.load_network(os.path.join(self.out, tree_member["file"]))
                frame = describe_archive(os.path.join(self.out, next(
                    m["file"] for m in self.members.values()
                    if m["family"] == "aligned" and m["ratio"] == ratio and m["seed"] == seed)))["frame"]
                against = describe(archive["nodes"], archive["edges"], frame=frame, d_ref=1.0, tree=archive["tree"])
                tree_order = against["frame_orientation"]["capillary"]["S"]
                self.assertGreater(aligned["descriptors"]["capillary_S"], tree_order, (ratio, seed))
        # the counters partition what they count
        for ident, row in self.rows.items():
            events = row["events"]
            selected = events.get("anastomosis_selected", 0)
            self.assertEqual(selected, events.get("anastomosis_bridges", 0) + events.get("anastomosis_source_consumed", 0)
                             + events.get("anastomosis_no_partner", 0) + events.get("anastomosis_collision_failed", 0))
            self.assertEqual(events.get("rung_sites", 0),
                             sum(events.get(key, 0) for key in ("rung_sites_near_junction", "rung_sites_consumed",
                                                                 "rung_no_partner", "rung_collision_failed",
                                                                 "rung_bridges")))
            if not main.FAMILIES[self.members[ident]["family"]].get("cross_connect"):
                self.assertEqual(events.get("rung_sites", 0), 0)
            steps = sum(events.get(key, 0) for key in ("guided_steps", "guidance_onset_steps", "unguided_steps",
                                                        "guidance_undefined_steps"))
            if main.FAMILIES[self.members[ident]["family"]].get("guidance"):
                self.assertGreater(steps, 0)
                self.assertGreater(events.get("guided_steps", 0), 0)
            else:
                self.assertEqual(steps, 0)
            # the bed's counters keep its laws, and count nothing in a family without a bed
            if has_bed(self.members[ident]["family"]):
                metadata = main.load_network(os.path.join(self.out, self.members[ident]["file"]))["metadata"]
                kwargs = metadata["grow_kwargs"]
                planned = bed.planned_seeds(metadata["growth_box"], kwargs["bed"], metadata["d_min"])
                laws = bed_laws(events, planned, kwargs["collision_attempts"])
                self.assertEqual(laws, dict.fromkeys(laws, True), ident)
            else:
                self.assertEqual({key: events.get(key, 0) for key in bed.EVENT_KEYS}, dict.fromkeys(bed.EVENT_KEYS, 0),
                                 ident)

    def bed_member(self, ident):
        """
        What the benchmark reports of a member with a bed, in d_min: its cost,
        its bed's counters and their laws, its seeds, edges and segments, the
        descriptors of all its vessels and of its bed on its own (bed_only),
        with the tissue distance over its growth box and over the central
        cube, the arc over the chord of the bed's polylines, how far they
        leave the box, and the attachment of the feeders' tips.
        """
        member, row = self.members[ident], self.rows[ident]
        path = os.path.join(self.out, member["file"])
        report, describe_seconds, tissue_seconds = self.described(ident)
        archive = load_archive(path)
        nodes, edges, tree, metadata = archive["nodes"], archive["edges"], archive["tree"], archive["metadata"]
        d_min = float(metadata["d_min"])
        kwargs = metadata["grow_kwargs"]
        box = [float(v) for v in metadata["growth_box"]]
        events = {key: row["events"].get(key, 0) for key in bed.EVENT_KEYS}
        planned = bed.planned_seeds(box, kwargs["bed"], d_min)
        # a library archive's growth box is read by the length density only as growth_box_um
        boxed = dict(metadata, growth_box_um=box)
        cube = [TISSUE_CUBE * d_min] * 3
        alone, alone_tree = bed_only(nodes, tree)
        started = time.perf_counter()
        whole_box = describe(nodes, edges, metadata=boxed, tree=tree, evd_spacing=d_min)
        whole_cube = describe(nodes, edges, metadata=metadata, volume=cube, tree=tree, evd_spacing=d_min)
        alone_box = describe(alone, metadata=boxed, tree=alone_tree, evd_spacing=d_min)
        alone_cube = describe(alone, metadata=metadata, volume=cube, tree=alone_tree, evd_spacing=d_min)
        views_seconds = time.perf_counter() - started
        attachment = feeder_attachment(nodes, tree, kwargs["collision_margin"], d_min)
        segments, shares = report["segments_by_class"]["capillary"], report["calibre_shares"]

        def angles(described):
            return {key: described["branch_angles_deg"][key] for key in ("count", "min", "median", "max")}

        def loops(described):
            return {kind: {key: described["loops"][kind][key] for key in ("sampled", "found", "median", "histogram")}
                    for kind in ("by_segment", "by_node")}

        placed = events["bed_segments_walked"] + events["bed_segments_chord"] - events["bed_segments_removed"]
        return {
            "family": member["family"], "ratio": member["ratio"], "seed": member["seed"], "points": row["points"],
            "seconds": row["seconds"], "peak_rss_mb": row["peak_rss_mb"],
            "own_peak_rss_mb": row.get("own_peak_rss_mb"), "archive_bytes": os.path.getsize(path),
            "describe_seconds": describe_seconds, "describe_with_tissue_seconds": tissue_seconds,
            "describe_views_seconds": views_seconds,
            "events": events, "laws": bed_laws(events, planned, kwargs["collision_attempts"]),
            "seeds": {"planned": planned, "placed": events["bed_seeds"],
                      "kept": events["bed_seeds"] - events["bed_seeds_dead_end"] - events["bed_seeds_island"]},
            "proposals_on_feeders_share": (events["bed_proposals_on_feeders"] / events["bed_proposals"]
                                           if events["bed_proposals"] else None),
            "edges": {"greedy": events["bed_edges"],
                      "kept": events["bed_edges"] - events["bed_edges_dead_end"] - events["bed_edges_island"]
                      - events["bed_edges_dropped"]},
            "segments": {"attempted": events["bed_segments"], "walked": events["bed_segments_walked"],
                         "chord": events["bed_segments_chord"], "dropped": events["bed_segments_dropped"],
                         "removed": events["bed_segments_removed"], "placed": placed,
                         "polylines": len(bed_polylines(nodes, tree))},
            "capillary_segments": {key: segments[key] for key in ("count", "median_d", "p10_d", "p90_d", "cv")},
            "branch_angles_deg": {"all": angles(report), "bed": angles(alone_box)},
            "calibre_shares": {kind: {key: shares[kind][key] for key in ("below_1.5", "below_2")}
                               for kind in ("length", "volume")},
            "length_density_d2": whole_box["length_density_mm_per_mm3"] * 1e-6 * d_min ** 2,
            "length_density_volume": whole_box["volume_source"],
            "tissue_distance": {"all_growth_box": tissue_summary(whole_box["tissue_distance"]),
                                "all_cube": tissue_summary(whole_cube["tissue_distance"]),
                                "bed_growth_box": tissue_summary(alone_box["tissue_distance"]),
                                "bed_cube": tissue_summary(alone_cube["tissue_distance"])},
            "junctions": {"all": report["junctions"], "bed": alone_box["junctions"]},
            "loops": {"all": loops(report), "bed": loops(alone_box)},
            "arc_chord": {"bed_polylines": bed_arc_chord(nodes, tree), "bed_segments": alone_box["arc_chord"]},
            "excursions": bed_excursions(nodes, tree, box, d_min),
            "attachment": attachment,
            # the tips recomputed from the archive are the ones the bed counted
            "attachment_agrees_with_counters": (
                attachment["eligible"] == events["bed_tips"] - events["bed_tips_inside_junction"]
                and attachment["stubs"] == events["bed_tips_inside_junction"]
                and attachment["attached"] == events["bed_tips_attached"]
                and attachment["tip_edges"] == events["bed_tip_edges"]
                and attachment["components"] == events["bed_components"]
                and attachment["components_both_trees"]
                == events["bed_components"] - events["bed_components_one_tree"]),
        }

    def test_the_bed_of_each_member_and_the_attachment_of_its_feeders(self):
        idents = [ident for ident in sorted(self.rows) if has_bed(self.members[ident]["family"])]
        if not idents:
            self.skipTest("no family benchmarked here grows an explicit bed (VSYSTEM_BENCHMARK_FAMILIES=foam)")
        table = {}
        print("\nfamily    R seed   points seconds  rss_MB planned placed  kept  edges   kept  segs walked chord drop "
              "polyl  cap_med    cv  angle  bed_ang   L/V  evd_all evd_bed  cube_all cube_bed  tips att k2 joined "
              "on_feeders")

        def fmt(value, width=7, digits=2):
            return f"{value:{width}.{digits}f}" if isinstance(value, (int, float)) else " " * (width - 1) + "-"

        def shares(parts):
            return " ".join(f"{name}: {part['attached']}/{part['eligible']}" for name, part in parts.items())

        for ident in idents:
            entry = self.bed_member(ident)
            table[f"{entry['family']}_r{entry['ratio']:g}_s{entry['seed']}"] = entry
            seeds, edges, segments = entry["seeds"], entry["edges"], entry["segments"]
            tissue = {name: (value or {}).get("mean_d") for name, value in entry["tissue_distance"].items()}
            attachment, events = entry["attachment"], entry["events"]
            segment_lengths, calibres = entry["capillary_segments"], entry["calibre_shares"]
            angles, loops = entry["branch_angles_deg"], entry["loops"]["all"]
            print(f"{entry['family']:8s} {entry['ratio']:3g} {entry['seed']:3d} {entry['points']:8d} "
                  f"{entry['seconds']:7.1f} {fmt(entry['own_peak_rss_mb'], 7, 0)} {seeds['planned']:7d} "
                  f"{seeds['placed']:6d} {seeds['kept']:5d} {edges['greedy']:6d} {edges['kept']:6d} "
                  f"{segments['attempted']:5d} {segments['walked']:6d} {segments['chord']:5d} "
                  f"{segments['dropped']:4d} {segments['polylines']:5d} {fmt(segment_lengths['median_d'], 8)} "
                  f"{fmt(segment_lengths['cv'], 5)} {fmt(angles['all']['median'], 6, 1)} "
                  f"{fmt(angles['bed']['median'], 8, 1)} {fmt(entry['length_density_d2'], 6, 4)} "
                  f"{fmt(tissue['all_growth_box'], 8)} {fmt(tissue['bed_growth_box'], 7)} "
                  f"{fmt(tissue['all_cube'], 9)} {fmt(tissue['bed_cube'], 8)} "
                  f"{attachment['eligible']:5d} {attachment['attached']:3d} {attachment['k_t'].get('2', 0):2d} "
                  f"{events['bed_components'] - events['bed_components_one_tree']:6d} "
                  f"{fmt(entry['proposals_on_feeders_share'], 10, 4)}")
            print(f"    archive {entry['archive_bytes'] / 1e6:.1f} MB; describe {entry['describe_seconds']:.1f} s, "
                  f"{entry['describe_with_tissue_seconds']:.1f} s with the tissue distance, "
                  f"{entry['describe_views_seconds']:.1f} s the four views; segments p10/p90 "
                  f"{fmt(segment_lengths['p10_d'], 1)}/{fmt(segment_lengths['p90_d'], 1)}; below 1.5 and 2 d_min, "
                  f"length {fmt(calibres['length']['below_1.5'], 1, 3)} {fmt(calibres['length']['below_2'], 1, 3)}, "
                  f"volume {fmt(calibres['volume']['below_1.5'], 1, 3)} {fmt(calibres['volume']['below_2'], 1, 3)}; "
                  f"angles min/max {fmt(angles['all']['min'], 1, 1)}/{fmt(angles['all']['max'], 1, 1)}, "
                  f"bed {fmt(angles['bed']['min'], 1, 1)}/{fmt(angles['bed']['max'], 1, 1)}; loops by segment "
                  f"{loops['by_segment']['histogram']}, by node {loops['by_node']['histogram']}; arc/chord "
                  f"{fmt(entry['arc_chord']['bed_polylines']['median'], 1, 3)} median; outside the box "
                  f"{entry['excursions']['outside']} of {entry['excursions']['points']} bed points, "
                  f"up to {fmt(entry['excursions']['max_d'], 1)} d_min")
        _record("bed", "members", table)
        # the feeders' tips and the bed, over every member and at each ratio
        pooled = {}
        for family in sorted({entry["family"] for entry in table.values()}):
            entries = [entry for entry in table.values() if entry["family"] == family]
            pooled[family] = {"all": pooled_attachment(entries)}
            for ratio in sorted({entry["ratio"] for entry in entries}):
                pooled[family][f"r{ratio:g}"] = pooled_attachment([entry for entry in entries
                                                                    if entry["ratio"] == ratio])
        _record("bed", "attachment", pooled)
        for family, scopes in pooled.items():
            print(f"\nfeeder tips of {family} and its bed, by scope: eligible (stubs) attached rate | k_t | by tree "
                  f"| by tip calibre in d_min | members whose bed joins both trees | tip-to-tip polylines (two trees) "
                  f"| bed junctions with tips only | rate within a component holding a bed junction")
            for scope, total in scopes.items():
                print(f"  {scope:5s} {total['eligible']:5d} ({total['stubs']}) {total['attached']:5d} "
                      f"{fmt(total['rate'], 5, 3)} | {total['k_t']} | {shares(total['by_tree'])} | "
                      f"{shares(total['by_calibre'])} | {total['members_joining_both_trees']}/{total['members']} | "
                      f"{total['tip_to_tip_polylines']} ({total['tip_to_tip_polylines_two_trees']}) | "
                      f"{total['bed_junctions_tips_only']}/{total['bed_junctions']} | "
                      f"{fmt(total['rate_in_junction_component'], 5, 3)}")
        disagreeing = [key for key, entry in table.items() if not entry["attachment_agrees_with_counters"]]
        if disagreeing:
            print(f"tips recomputed from the archive differ from the bed's counters in: {', '.join(disagreeing)}")
        self.assertGreater(len(table), 0)

    @classmethod
    def cost_model(cls, family):
        """
        Power laws in R fitted to a family's measured seconds, archive bytes
        and own peak RSS (own_peak_rss_mb), with its largest row peak RSS.
        """
        ratios = np.array([cls.members[k]["ratio"] for k in cls.rows if cls.members[k]["family"] == family])
        seconds = np.array([cls.rows[k]["seconds"] for k in cls.rows if cls.members[k]["family"] == family])
        size = np.array([os.path.getsize(os.path.join(cls.out, cls.members[k]["file"]))
                         for k in cls.rows if cls.members[k]["family"] == family])
        rss = np.array([cls.rows[k]["peak_rss_mb"] for k in cls.rows if cls.members[k]["family"] == family])
        own = np.array([cls.rows[k].get("own_peak_rss_mb", cls.rows[k]["peak_rss_mb"])
                        for k in cls.rows if cls.members[k]["family"] == family])
        return {"seconds": tuple(np.polyfit(np.log(ratios), np.log(seconds), 1)),
                "bytes": tuple(np.polyfit(np.log(ratios), np.log(size), 1)),
                "own_peak_rss_mb": tuple(np.polyfit(np.log(ratios), np.log(own), 1)),
                "largest_rss_mb": float(rss.max())}

    @classmethod
    def project(cls, families):
        """
        The cost of a 2000-network library of `families` in equal shares
        over the default ratio range: each family's power laws evaluated on
        its planned ratios, summed. Returns the projection and its seconds.
        """
        projection = {}
        plan = library.plan_library(1, 2000, families, [1.0] * len(families), library.DEFAULT_RATIO_RANGE)
        weights = library.library_weights([{"ratio": m["ratio"], "ratio_law": "log-uniform"} for m in plan])
        total_seconds = total_bytes = 0.0
        peak = 0.0
        for family in families:
            model = cls.cost_model(family)
            # a power law in R fitted to the measured cost, evaluated on the planned ratios
            slope, intercept = model["seconds"]
            size_slope, size_intercept = model["bytes"]
            rss = model["largest_rss_mb"]
            planned = np.array([m["ratio"] for m in plan if m["family"] == family])
            family_seconds = float(np.sum(np.exp(intercept) * planned ** slope))
            family_bytes = float(np.sum(np.exp(size_intercept) * planned ** size_slope))
            projection[family] = {"networks": int(planned.size), "seconds_exponent": float(slope),
                                  "cpu_hours": family_seconds / 3600.0, "disk_gb": family_bytes / 1e9,
                                  "peak_rss_mb": float(rss)}
            total_seconds += family_seconds
            total_bytes += family_bytes
            peak = max(peak, rss)
        projection["total"] = {"networks": len(plan), "cpu_hours": total_seconds / 3600.0,
                               "disk_gb": total_bytes / 1e9, "peak_rss_mb": peak,
                               "effective_sample_size_under_library_weights":
                                   library.effective_sample_size(weights)}
        return projection, total_seconds

    def test_a_library_of_2000_networks_is_projected_from_the_cost_per_ratio(self):
        families = sorted({self.members[k]["family"] for k in self.rows}, key=benchmark_families().index)
        projection, total_seconds = self.project(families)
        _record("projection", "library_2000", projection)
        print("\n2000-network library over the default ratio range, projected from the cost per ratio:")
        for family, entry in projection.items():
            print(f"  {family:8s} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in entry.items()))
        # with a family that grows a bed: 2000 networks of it alone, the same
        # library without it, and whether its cost per network calls for a
        # cap on its ratio: at the cost ratio as measured, and as projected
        # at the ratio where the library's cap on planned candidates refuses it
        with_bed = [family for family in families if has_bed(family)]
        others = [family for family in families if not has_bed(family)]
        if with_bed:
            libraries = {f"{family}_alone": self.project([family])[0] for family in with_bed}
            if others:
                libraries["without_" + "_".join(with_bed)] = self.project(others)[0]
            cost_ratio = benchmark_ratios()[1]
            for name, entry in libraries.items():
                _record("projection", f"library_2000_{name}", entry)
                print(f"  2000 networks, {name.replace('_', ' ')}: " + " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in entry["total"].items()))
            for family in with_bed:
                model = self.cost_model(family)
                at_cost = [k for k in self.rows
                           if self.members[k]["family"] == family and self.members[k]["ratio"] == cost_ratio]
                trip = guard_trip_ratio(family)

                def at_trip(fit):
                    return float(np.exp(fit[1]) * trip ** fit[0]) if trip else None

                criterion = {"cost_ratio": cost_ratio,
                             "seconds_at_cost_ratio": max((self.rows[k]["seconds"] for k in at_cost), default=None),
                             "peak_rss_mb_at_cost_ratio": max((self.rows[k].get("own_peak_rss_mb",
                                                                                self.rows[k]["peak_rss_mb"])
                                                               for k in at_cost), default=None),
                             "guard_trip_ratio": trip, "projected_seconds_at_trip": at_trip(model["seconds"]),
                             "projected_peak_rss_mb_at_trip": at_trip(model["own_peak_rss_mb"]),
                             "seconds_limit": NETWORK_SECONDS_LIMIT, "peak_rss_mb_limit": NETWORK_RSS_MB_LIMIT}
                seconds = [criterion[key] for key in ("seconds_at_cost_ratio", "projected_seconds_at_trip")]
                memory = [criterion[key] for key in ("peak_rss_mb_at_cost_ratio", "projected_peak_rss_mb_at_trip")]
                criterion["crosses_a_limit"] = (any(v is not None and v > NETWORK_SECONDS_LIMIT for v in seconds)
                                                or any(v is not None and v > NETWORK_RSS_MB_LIMIT for v in memory))
                _record("projection", f"{family}_ratio_cap", criterion)
                print(f"  {family} per network: " + " ".join(
                    f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in criterion.items()))
        self.assertGreater(total_seconds, 0.0)


@SLOW
class AlignedBenchmark(unittest.TestCase):
    """The aligned preset at R 8 over the capillary guidance length and the onset, and its cost."""

    def test_the_capillary_order_follows_the_guidance_length_and_onset(self):
        print("\nG    onset seed  cap_S  polar  fisherK  watsonK  4P/G  2P/G  reach0 reach1  bridges cross through  spacing  density")
        table = {}
        for length in LENGTHS:
            for onset in ONSETS:
                for seed in SEEDS:
                    rules = [dict(rule) for rule in main.FAMILIES["aligned"]["guidance"]]
                    rules[0].update(length=length, onset=onset)
                    rules[1].update(onset=onset)
                    grown = grow_preset("aligned", 8.0, seed, guidance=rules)
                    report = describe(grown["nodes"], grown["edges"], frame=grown["frame"], d_ref=1.0,
                                      tree=grown["tree"])
                    summary = capillary_summary(report)
                    bridges = bridge_geometry(grown)
                    cross = [b for b in bridges if b["cross_tree"]]
                    through = [b for b in bridges if b["angle_to_x_deg"] <= 45.0 and b["same_way"]]
                    reach = capillary_reach(grown)
                    entry = {"length": length, "onset": onset, "seed": seed, "kappa": 4.0 * PERSISTENCE / length,
                             "watson_K_expected": 2.0 * PERSISTENCE / length, "descriptors": summary,
                             "bridges": len(bridges), "cross_tree_share": len(cross) / len(bridges) if bridges else None,
                             "through_fraction": len(through) / len(bridges) if bridges else None,
                             "reversals": sum(1 for b in bridges if not b["same_way"]),
                             "chords": [b["chord"] for b in bridges], "reach": reach,
                             "events": {k: v for k, v in grown["events"].items() if v}, "seconds": grown["seconds"]}
                    table[f"G{length:g}_onset{onset:g}_s{seed}"] = entry

                    def fmt(value, width=6, digits=3):
                        return f"{value:{width}.{digits}f}" if isinstance(value, (int, float)) else " " * (width - 1) + "-"

                    print(f"{length:4g} {onset:5g} {seed:4d} {fmt(summary.get('capillary_S'))} "
                          f"{fmt(summary.get('capillary_polar_order'))} {fmt(summary.get('capillary_fisher_axial_K'), 8, 2)} "
                          f"{fmt(summary.get('capillary_watson_K'), 8, 2)} {entry['kappa']:5.1f} "
                          f"{entry['watson_K_expected']:5.1f} {fmt(reach[0], 6, 1)} {fmt(reach[1], 6, 1)} "
                          f"{len(bridges):7d} {fmt(entry['cross_tree_share'], 6, 2)} {fmt(entry['through_fraction'], 7, 2)} "
                          f"{fmt(summary.get('spacing_median'), 8, 2)} {fmt(summary.get('crossing_density'), 8, 4)}")
        _record("aligned", "guidance_sweep", table)
        self.assertGreater(len(table), 0)

    def test_the_cost_of_guidance(self):
        rows = []
        print("\nguidance cost at R 8: seed  guided_s  unguided_s  guided_points  unguided_points")
        for seed in SEEDS:
            guided = grow_preset("aligned", 8.0, seed)
            unguided = grow_preset("aligned", 8.0, seed, guidance=None)
            rows.append({"seed": seed, "guided_seconds": guided["seconds"], "unguided_seconds": unguided["seconds"],
                         "guided_points": int(guided["nodes"].shape[1]), "unguided_points": int(unguided["nodes"].shape[1]),
                         "guided_events": {k: v for k, v in guided["events"].items() if v},
                         "unguided_events": {k: v for k, v in unguided["events"].items() if v}})
            print(f"  {seed:4d} {guided['seconds']:9.1f} {unguided['seconds']:11.1f} {guided['nodes'].shape[1]:14d} "
                  f"{unguided['nodes'].shape[1]:16d}")
        _record("aligned", "guidance_cost", rows)
        started = time.perf_counter()
        grown = grow_preset("aligned", 16.0, 1)
        describe(grown["nodes"], grown["edges"], frame=grown["frame"], d_ref=1.0, tree=grown["tree"])
        seconds = time.perf_counter() - started - grown["seconds"]
        _record("aligned", "describe_seconds_r16", {"points": int(grown["nodes"].shape[1]), "seconds": seconds})
        print(f"  describe on the R 16 aligned network ({grown['nodes'].shape[1]} points): {seconds:.1f} s")
        self.assertGreater(len(rows), 0)

    def test_the_cost_of_a_guided_step(self):
        # one walked stem of 2000 sub-segments, eight steps each, with no bounds
        # or avoider, so that the time is the walk's own and the guidance's;
        # every setting draws the same numbers, so the steps are the same count
        from analyseGrammar import WalkSettings, branching_turtle_to_coords
        from guidance import Guidance
        y, z = np.eye(3)[1], np.eye(3)[2]
        program = "{" + "f(1.4,1.0)" * 2000 + "}"
        steps = 2000 * 8
        settings = {
            "none": None,
            "nematic axis": [{"below": None, "field": "axis", "axis": [1, 0, 0], "length": 3.0, "onset": 0.0}],
            "polar axis": [{"below": None, "field": "axis", "axis": [1, 0, 0], "sense": "polar", "polarity": "fixed",
                            "length": 3.0, "onset": 0.0}],
            "plane": [{"below": None, "field": "plane", "normal": [1, 0, 0], "length": 3.0, "onset": 0.0}],
            "plane with bank": [{"below": None, "field": "plane", "normal": [1, 0, 0], "length": 3.0, "onset": 0.0,
                                 "bank": True}],
        }
        table = {}
        print("\nguidance cost per walk step (one stem of 16 000 steps, median of 5 runs x seeds 1-3):")
        for name, rules in settings.items():
            times = []
            for _ in range(5):
                for seed in SEEDS:
                    guide = None if rules is None else Guidance(rules, 1.0).bind(y)
                    walk = WalkSettings(10.0, np.random.default_rng([seed, 1]), guidance=guide)
                    started = time.perf_counter()
                    for _ in branching_turtle_to_coords(program, 1.0, direction=y, perpendicular=z, walk=walk,
                                                        subdivisions=3):
                        pass
                    times.append(time.perf_counter() - started)
            table[name] = {"median_seconds": float(np.median(times)), "per_step_us": 1e6 * float(np.median(times)) / steps}
            print(f"  {name:16s} {table[name]['per_step_us']:7.1f} us per step"
                  + ("" if rules is None else f", {table[name]['per_step_us'] - table['none']['per_step_us']:6.1f} "
                                              f"us more than the walk alone"))
        _record("aligned", "guidance_step_cost", table)
        self.assertGreater(table["nematic axis"]["per_step_us"], 0.0)


@SLOW
class BedCostBenchmark(unittest.TestCase):
    """
    The cost of an explicit bed against its size: bed.build alone with the
    foam settings, d_min 1 and margin 1, on cubes planned to hold BED_SIZES
    seeds, each run in a fresh process one at a time, its peak RSS less that
    of a process that only imported the same modules; and, reported only,
    its girth and its density swept at BED_SWEEP_SEEDS. Runs when the
    families benchmarked include foam.
    """

    @classmethod
    def setUpClass(cls):
        if "foam" not in benchmark_families():
            raise unittest.SkipTest("the cost of a bed runs with VSYSTEM_BENCHMARK_FAMILIES naming foam")
        cls.settings = bed.parse_settings(main.FAMILIES["foam"]["bed"])
        runs = {}

        def add(count, seed, girth, density, measure):
            key = (count, seed, girth, density)
            if key not in runs:
                settings = dict(cls.settings, girth=girth, density=density)
                runs[key] = {"count": count, "seed": seed, "settings": settings, "side": bed_cube(count, settings),
                             "measure": []}
            runs[key]["measure"] += [name for name in measure if name not in runs[key]["measure"]]

        girth, density = cls.settings["girth"], cls.settings["density"]
        for count in BED_SIZES:
            for seed in SEEDS if count <= BED_ALL_SEEDS_UP_TO else SEEDS[:1]:
                add(count, seed, girth, density, ())
        for value in BED_GIRTHS:
            add(BED_SWEEP_SEEDS, SEEDS[0], value, density, ("loops",))
        for value in BED_DENSITIES:
            add(BED_SWEEP_SEEDS, SEEDS[0], girth, value, ("tissue",))
        for name in library.THREAD_VARIABLES:
            os.environ.setdefault(name, "1")
        context = multiprocessing.get_context("spawn")
        # one process at a time, so that each is timed alone, and a fresh one for each run
        with context.Pool(processes=1, maxtasksperchild=1) as pool:
            cls.baseline_rss_mb = pool.map(_baseline_task, [None], chunksize=1)[0]
            results = pool.map(_bed_task, list(runs.values()), chunksize=1)
        cls.runs = dict(zip(runs, results))

    def run_of(self, count, seed, girth=None, density=None):
        girth = self.settings["girth"] if girth is None else girth
        density = self.settings["density"] if density is None else density
        return self.runs[(count, seed, girth, density)]

    def test_the_cost_of_a_bed_against_its_size(self):
        stages = ("seeds", "candidates", "greedy", "prune", "walk", "assemble")
        table = {}
        print(f"\nbed alone against its size (peak RSS less {self.baseline_rss_mb:.0f} MB of imports):")
        print("    seeds seed  total_s " + " ".join(f"{stage:>10s}" for stage in stages)
              + "  rss_MB  planned_cand  candidates   edges    kept  chains  columns graph_s describe_s")
        for count in BED_SIZES:
            for seed in SEEDS if count <= BED_ALL_SEEDS_UP_TO else SEEDS[:1]:
                run = self.run_of(count, seed)
                events = run["events"]
                entry = dict(run, peak_rss_mb=run["peak_rss_mb"] - self.baseline_rss_mb,
                             peak_rss_mb_with_describe=run["peak_rss_mb_with_describe"] - self.baseline_rss_mb,
                             seed_pair_candidates=events["bed_candidates"] - events["bed_candidates_tip"],
                             laws=bed_laws(events, count, 10))
                table[f"n{count}_s{seed}"] = entry
                _record("bed_cost", f"n{count}_s{seed}", entry)
                print(f"{count:9d} {seed:4d} {run['seconds']:8.2f} "
                      + " ".join(f"{run['timings'][stage]:10.2f}" for stage in stages)
                      + f" {entry['peak_rss_mb']:7.0f} {run['planned_candidates']:13d} {events['bed_candidates']:11d} "
                      f"{run['greedy_edges']:7d} {run['edges']:7d} {run['chains']:7d} {run['columns']:8d} "
                      f"{run['graph_seconds']:7.2f} {run['describe_seconds']:10.2f}")
                # the counters keep their laws, at the planned seed count and build's 10 attempts per chain
                self.assertEqual(entry["laws"], dict.fromkeys(entry["laws"], True), (count, seed))
                self.assertEqual(run["seeds"], count)
        # a power law in the seed count per stage, and the counts at which a
        # network would pass an hour or 8 GB, against the count the library's
        # cap on planned candidates admits
        sizes = np.log([entry["count"] for entry in table.values()])
        fits = {}
        for stage in stages + ("total",):
            seconds = [entry["seconds"] if stage == "total" else entry["timings"][stage] for entry in table.values()]
            fits[stage] = (tuple(float(v) for v in np.polyfit(sizes, np.log(np.maximum(seconds, 1e-9)), 1))
                           if len(set(sizes.tolist())) > 1 else None)
        rss = np.array([entry["peak_rss_mb"] for entry in table.values()])
        positive = rss > 0.0
        fits["peak_rss_mb"] = (tuple(float(v) for v in np.polyfit(sizes[positive], np.log(rss[positive]), 1))
                               if len(set(sizes[positive].tolist())) > 1 else None)

        def reached(fit, limit):
            return float(math.exp((math.log(limit) - fit[1]) / fit[0])) if fit and fit[0] > 0.0 else None

        # planned candidates are ceil(N density (4 pi / 3) reach^3 / 2), so the
        # largest seed count the cap admits lies at or below this one
        cap = library.DEFAULT_BED_MAX_CANDIDATES
        per_seed = self.settings["density"] * (4.0 * math.pi / 3.0) * self.settings["reach"] ** 3 / 2.0
        cap_seeds = int(cap / per_seed) + 1
        while bed.planned_candidates(np.full(3, bed_cube(cap_seeds, self.settings)), self.settings, 1.0) > cap:
            cap_seeds -= 1
        limits = {"exponents": {stage: fit[0] if fit else None for stage, fit in fits.items()}, "fits": fits,
                  "seeds_at_seconds_limit": reached(fits["total"], NETWORK_SECONDS_LIMIT),
                  "seeds_at_rss_limit": reached(fits["peak_rss_mb"], NETWORK_RSS_MB_LIMIT),
                  "seeds_at_default_cap": cap_seeds, "seconds_limit": NETWORK_SECONDS_LIMIT,
                  "peak_rss_mb_limit": NETWORK_RSS_MB_LIMIT}
        _record("bed_cost", "limits", limits)
        print("exponents in the seed count: " + " ".join(
            f"{stage}={exponent:.2f}" if exponent is not None else f"{stage}=-"
            for stage, exponent in limits["exponents"].items()))
        print(f"seeds at {NETWORK_SECONDS_LIMIT:g} s: {limits['seeds_at_seconds_limit']}, at "
              f"{NETWORK_RSS_MB_LIMIT:g} MB: {limits['seeds_at_rss_limit']}, admitted by the default cap of "
              f"{library.DEFAULT_BED_MAX_CANDIDATES} planned candidates: {cap_seeds}")
        self.assertGreater(len(table), 0)

    def test_the_girth_and_the_density_of_a_bed(self):
        # reported only: the bed's cycles in edges, which the girth bounds,
        # against describe's loops in segments, which merge a chain's edges
        print(f"\ngirth at {BED_SWEEP_SEEDS} seeds: the shortest cycle through each kept edge, in edges "
              f"(0: none of at most {BED_LONGEST_CYCLE}), and describe's loops, in segments")
        girths = {}
        for girth in BED_GIRTHS:
            run = self.run_of(BED_SWEEP_SEEDS, SEEDS[0], girth=girth)
            cycles = {int(length): count for length, count in run["cycles_in_edges"].items()}
            found = {length: count for length, count in cycles.items() if length}
            by_segment = {int(size): count for size, count in run["loops"]["by_segment"]["histogram"].items()}
            entry = {"girth": girth, "cycles_in_edges": run["cycles_in_edges"], "loops": run["loops"],
                     "shortest_cycle_in_edges": min(found) if found else None,
                     "edges_on_a_cycle_below_girth": sum(count for length, count in found.items() if length < girth),
                     "segments_on_a_loop_below_girth_share": (sum(count for size, count in by_segment.items()
                                                                  if size < girth) / sum(by_segment.values())
                                                              if by_segment else None),
                     "seconds": run["seconds"], "events": run["events"]}
            girths[str(girth)] = entry
            print(f"  girth {girth:2d}: in edges {run['cycles_in_edges']}; by segment "
                  f"{run['loops']['by_segment']['histogram']} (below the girth "
                  f"{entry['segments_on_a_loop_below_girth_share']}); by node {run['loops']['by_node']['histogram']}")
            # the girth is a rule of the construction: no cycle in edges shorter than it
            self.assertEqual(entry["edges_on_a_cycle_below_girth"], 0, girth)
        _record("bed_cost", "girth_sweep", girths)
        print(f"density at {BED_SWEEP_SEEDS} seeds: capillary segments and tissue distance over the central "
              f"{TISSUE_CUBE:g}^3 cube, in d_min")
        densities = {}
        for density in BED_DENSITIES:
            run = self.run_of(BED_SWEEP_SEEDS, SEEDS[0], density=density)
            densities[f"{density:g}"] = {"density": density, "side": run["side"],
                                         "capillary_segments": run["capillary_segments"],
                                         "tissue_distance": run["tissue_distance"], "seconds": run["seconds"],
                                         "events": run["events"]}
            segments, tissue = run["capillary_segments"], run["tissue_distance"] or {}
            print(f"  density {density:g}: cube side {run['side']:.1f}, segments {segments['count']} median "
                  f"{segments['median_d']} p10 {segments['p10_d']} p90 {segments['p90_d']} cv {segments['cv']}; "
                  f"tissue mean {tissue.get('mean_d')} median {tissue.get('median_d')} p90 {tissue.get('p90_d')} "
                  f"max {tissue.get('max_d')}")
        _record("bed_cost", "density_sweep", densities)
        self.assertEqual(len(densities), len(BED_DENSITIES))


if __name__ == "__main__":
    unittest.main()
