"""
Benchmark of the families against each other: growth cost and the Phase 1
descriptors of every available family at three root ratios, the aligned
preset over its guidance length and onset, and a projection of the cost of
a 2000-network library.

Slow: it grows several dozen networks, some of them large, and runs only
with VSYSTEM_SLOW_TESTS=1. It prints its tables and writes them as JSON to
the path in VSYSTEM_BENCHMARK_JSON when that is set. It asserts sanity only:
that the capillaries of `aligned` are more ordered about its frame axis than
those of a tree described against the same axis, and that the counters
partition what they count.

Run from the repository root with
    VSYSTEM_SLOW_TESTS=1 python -m unittest tests.test_topology_benchmark
"""
import json
import math
import multiprocessing
import os
import random
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import library  # noqa: E402
import main  # noqa: E402
from anastomosis import BRIDGE  # noqa: E402
from describe import describe, describe_archive  # noqa: E402

SLOW = unittest.skipUnless(os.environ.get("VSYSTEM_SLOW_TESTS"), "set VSYSTEM_SLOW_TESTS=1 to run the benchmark")
RATIOS = (4.0, 8.0, 16.0)
COST_RATIO = 25.0
SEEDS = (1, 2, 3)
PERSISTENCE = main.FAMILIES["aligned"]["persistence"]
LENGTHS = (2.0, 4.9, 12.3, 16.4)            # kappa = 4 P / G of about 20, 8.2, 3.3 and 2.4
ONSETS = (0.0, 2.0)

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
                for key in ("S", "mean_abs_cos", "crossing_ratio", "watson_K", "fisher_axial_K", "within_45_deg"):
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
    for row, bridge in zip(rows, grown["bridges"]):
        row["cross_tree"] = bridge["tip_tree"] != bridge["partner_tree"]
    return rows


def capillary_reach(grown):
    """How far the capillaries of each tree reach along x from their root, in d_min."""
    nodes, edges, tree = grown["nodes"], grown["edges"], grown["tree"]
    import graph
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


def _grow_task(task):
    return library.grow_and_write(task)


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
        families = library.available_families()
        members = []
        for family in families:
            for ratio in RATIOS + (COST_RATIO,):
                for seed in SEEDS if ratio != COST_RATIO else SEEDS[:1]:
                    members.append({"id": len(members), "family": family, "ratio": ratio, "bin": 0, "u": 0.0,
                                    "seed": seed, "file": f"bench_{len(members):03d}_{family}.npz"})
        settings = dict(library.growth_settings(library.build_parser().parse_args(
            ["--out", cls.out, "--count", "1", "--seed", "1", "--families"] + families)),
            ratio_lo=min(RATIOS), ratio_hi=COST_RATIO)
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

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_families_grow_and_describe(self):
        self.assertEqual(self.failures, [], self.failures)
        table = {}
        print("\nfamily   R  seed   points  seconds  rss_MB  cap_S  polar  len<2  vol<2  seg_med  spacing  density  events")
        for ident, row in sorted(self.rows.items()):
            member = self.members[ident]
            started = time.perf_counter()
            report = describe_archive(os.path.join(self.out, member["file"]))
            describe_seconds = time.perf_counter() - started
            summary = capillary_summary(report)
            events = {k: v for k, v in row["events"].items() if v}
            entry = {"family": member["family"], "ratio": member["ratio"], "seed": member["seed"],
                     "points": row["points"], "seconds": row["seconds"], "peak_rss_mb": row["peak_rss_mb"],
                     "describe_seconds": describe_seconds,
                     "archive_bytes": os.path.getsize(os.path.join(self.out, member["file"])),
                     "events": events, "descriptors": summary, "cycles": row["cycles"], "tips": row["tips"]}
            table[f"{member['family']}_r{member['ratio']:g}_s{member['seed']}"] = entry

            def fmt(key, width=6, digits=3):
                value = summary.get(key)
                return f"{value:{width}.{digits}f}" if isinstance(value, (int, float)) else " " * (width - 1) + "-"

            print(f"{member['family']:8s} {member['ratio']:3g} {member['seed']:3d} {row['points']:8d} "
                  f"{row['seconds']:8.1f} {row['peak_rss_mb']:7.0f} {fmt('capillary_S')} "
                  f"{fmt('capillary_polar_order')} {fmt('length_below_2')} {fmt('volume_below_2')} "
                  f"{fmt('capillary_segment_median', 7, 2)} {fmt('spacing_median', 7, 2)} "
                  f"{fmt('crossing_density', 8, 4)}  {' '.join(f'{k}={v}' for k, v in sorted(events.items()))}")
        _record("families", "members", table)
        largest = max(table.values(), key=lambda entry: entry["points"])
        _record("families", "describe_largest", {key: largest[key] for key in ("family", "ratio", "seed", "points",
                                                                            "describe_seconds")})
        print(f"describe_archive on the largest network ({largest['family']} R {largest['ratio']:g} seed "
              f"{largest['seed']}, {largest['points']} points): {largest['describe_seconds']:.1f} s")
        # sanity: on the same R and seed the capillaries of aligned are more ordered about its axis
        # than a tree's described against that axis
        for ratio in RATIOS:
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
            steps = sum(events.get(key, 0) for key in ("guided_steps", "guidance_onset_steps", "unguided_steps",
                                                        "guidance_undefined_steps"))
            if main.FAMILIES[self.members[ident]["family"]].get("guidance"):
                self.assertGreater(steps, 0)
                self.assertGreater(events.get("guided_steps", 0), 0)
            else:
                self.assertEqual(steps, 0)

    def test_a_library_of_2000_networks_is_projected_from_the_cost_per_ratio(self):
        families = library.available_families()
        projection = {}
        plan = library.plan_library(1, 2000, families, [1.0] * len(families), library.DEFAULT_RATIO_RANGE)
        weights = library.library_weights([{"ratio": m["ratio"], "ratio_law": "log-uniform"} for m in plan])
        total_seconds = total_bytes = 0.0
        peak = 0.0
        for family in families:
            ratios = np.array([self.members[k]["ratio"] for k in self.rows if self.members[k]["family"] == family])
            seconds = np.array([self.rows[k]["seconds"] for k in self.rows if self.members[k]["family"] == family])
            size = np.array([os.path.getsize(os.path.join(self.out, self.members[k]["file"]))
                             for k in self.rows if self.members[k]["family"] == family])
            rss = max(self.rows[k]["peak_rss_mb"] for k in self.rows if self.members[k]["family"] == family)
            # a power law in R fitted to the measured cost, evaluated on the planned ratios
            slope, intercept = np.polyfit(np.log(ratios), np.log(seconds), 1)
            size_slope, size_intercept = np.polyfit(np.log(ratios), np.log(size), 1)
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
        _record("projection", "library_2000", projection)
        print("\n2000-network library over the default ratio range, projected from the cost per ratio:")
        for family, entry in projection.items():
            print(f"  {family:8s} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in entry.items()))
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


if __name__ == "__main__":
    unittest.main()
