"""
Descriptor sweep over the geometry options.

Generates networks for a fixed set of seeds under each configuration below,
describes every one with describe.describe, and writes the per-configuration
means (and standard deviations over seeds) as a markdown table and as JSON:

    python docs/geometry/sweep.py --out docs/geometry --seeds 1 2 3 4 5 --workers 4

Configurations are expressed as command-line options of main.py, so that every
row of the table can be reproduced with `python main.py` and the same seed.
The per-network parameters (d0, epsilon, randmarg, generations) are drawn
exactly as main.py draws them for the seed, so all configurations of one seed
share the same grammar draws and differ only in the geometry stage under test.
"""
import argparse
import json
import multiprocessing
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from describe import describe  # noqa: E402
from main import FAMILIES, build_parser, grow_network, sample_parameters, shaping_options  # noqa: E402

# Fixed generation count for the comparisons, so that rows differ by the
# geometry stage and not by the depth the seed happened to draw.
DEPTH = ["--iterations", "8", "8"]
LSM = ["--d0", "25", "5", "--d-min", "1"]

CONFIGURATIONS = [
    ("default CLI (4-12 generations)", []),
    ("default, 8 generations", DEPTH),
    ("LSM calibration (d0 25, d_min 1), default depth", LSM),
    ("LSM calibration, 8 generations", LSM + DEPTH),
]
for persistence in (2, 3, 5, 8, 12, 20, 40):
    CONFIGURATIONS.append((f"walk, persistence {persistence}",
                           DEPTH + ["--tortuosity", "walk", "--persistence", str(persistence)]))
CONFIGURATIONS += [
    ("stems + collision avoidance (margin 1)", DEPTH + ["--avoid-collisions"]),
    ("walk p=8 + collision avoidance (margin 1)",
     DEPTH + ["--tortuosity", "walk", "--persistence", "8", "--avoid-collisions"]),
]
for mode in ("any", "arteriovenous"):
    for fraction in (0.25, 0.5, 0.75):
        options = DEPTH + ["--tortuosity", "walk", "--persistence", "8", "--avoid-collisions",
                           "--anastomose", "--anastomose-mode", mode,
                           "--anastomosis-fraction", str(fraction)]
        if mode == "arteriovenous":
            options += ["--grow-in-volume"]
        CONFIGURATIONS.append((f"anastomosis {mode}, fraction {fraction}", options))
for radius in (10, 20):
    CONFIGURATIONS.append((f"anastomosis arteriovenous, fraction 1.0, radius {radius}",
                           DEPTH + ["--tortuosity", "walk", "--persistence", "8", "--avoid-collisions",
                                    "--grow-in-volume", "--anastomose", "--anastomose-mode", "arteriovenous",
                                    "--anastomosis-fraction", "1.0", "--anastomosis-radius", str(radius)]))
for family in ("tree", "mesh", "tumour"):
    if FAMILIES.get(family) is not None:
        CONFIGURATIONS.append((f"family {family}", DEPTH + ["--family", family]))

COLUMNS = [
    ("points", lambda r: r["points"]),
    ("length mm", lambda r: r["total_length_mm"]),
    ("d P50", lambda r: r["diameter_um"]["p50"]),
    ("d P90", lambda r: r["diameter_um"]["p90"]),
    ("d P99", lambda r: r["diameter_um"]["p99"]),
    ("arc/chord mean", lambda r: r["arc_chord"]["mean"]),
    ("arc/chord P90", lambda r: r["arc_chord"]["p90"]),
    ("curvature 1/um", lambda r: r["curvature_per_um"]["mean_abs"]),
    ("tips", lambda r: r["tips"]["count"]),
    ("tips/mm", lambda r: r["tips"]["per_mm"]),
    ("junctions", lambda r: sum(v for k, v in r["degree_histogram"].items() if int(k) >= 3)),
    ("b0", lambda r: r["components"]),
    ("b1", lambda r: r["cycles"]),
    ("FA", lambda r: r["orientation"]["fractional_anisotropy"]),
    ("min clearance um", lambda r: r["clearance_um"]["min"]),
    ("violations", lambda r: r["clearance_um"]["violations"]),
    ("length density mm/mm3", lambda r: r["length_density_mm_per_mm3"]),
]
EVENT_COLUMNS = ["bound_terminations", "walk_bound_terminations", "collision_truncated_stems",
                 "collision_terminations", "anastomosis_bridges", "anastomosis_bridges_arteriovenous",
                 "anastomosis_no_partner", "anastomosis_collision_failed"]


def run_one(task):
    """Generates and describes one network; returns a flat record."""
    name, options, seed = task
    peel = argparse.ArgumentParser(add_help=False)
    peel.add_argument("--family", default="tree")
    family = peel.parse_known_args(options)[0].family
    args = build_parser(family).parse_args(options)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    properties, d0, niter = sample_parameters(args)
    started = time.time()
    grown = grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit, clip_axes=args.clip_axes,
                         voxel_size=args.voxel_size, subdivisions=args.subdivisions, d_min=args.d_min,
                         grow_in_volume=args.grow_in_volume, seed=seed, **shaping_options(args))
    metadata = {"events": grown["events"], "collision_margin": args.collision_margin,
                "avoid_collisions": args.avoid_collisions, "growth_box_um": grown["growth_box_um"],
                "volume": list(args.volume), "voxel_size": args.voxel_size, "fit": args.fit}
    report = describe(grown["nodes"], grown["edges"], metadata=metadata, tree=grown["tree"],
                      margin=args.collision_margin if args.avoid_collisions else 0.0)
    values = {label: getter(report) for label, getter in COLUMNS}
    values.update({key: grown["events"].get(key, 0) for key in EVENT_COLUMNS})
    return {"configuration": name, "options": options, "seed": seed, "generations": niter, "d0": d0,
            "seconds": time.time() - started, "values": values}


def summarise(records):
    """Mean and standard deviation of every column per configuration, in configuration order."""
    order = []
    grouped = {}
    for record in records:
        if record["configuration"] not in grouped:
            order.append(record["configuration"])
            grouped[record["configuration"]] = []
        grouped[record["configuration"]].append(record)
    summary = []
    for name in order:
        rows = grouped[name]
        stats = {}
        for key in list(rows[0]["values"]):
            values = np.array([r["values"][key] for r in rows if r["values"][key] is not None], dtype=float)
            stats[key] = {"mean": float(values.mean()) if values.size else None,
                          "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                          "n": int(values.size)}
        summary.append({"configuration": name, "options": rows[0]["options"], "seeds": [r["seed"] for r in rows],
                        "generations": [r["generations"] for r in rows], "seconds": sum(r["seconds"] for r in rows),
                        "stats": stats})
    return summary


def _fmt(stat, digits):
    if stat["mean"] is None:
        return "-"
    if stat["n"] > 1 and stat["sd"] > 0:
        return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"
    return f"{stat['mean']:.{digits}f}"


def markdown(summary, seeds):
    lines = [f"Means ± standard deviations over seeds {', '.join(map(str, seeds))}; every row reproducible with "
             f"`python main.py --count 1 --seed S <options>`. Definitions: describe.py.", ""]
    head = ["configuration"] + [label for label, _ in COLUMNS]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    digits = {"points": 0, "tips": 0, "junctions": 0, "b0": 1, "b1": 1, "violations": 1, "curvature 1/um": 4,
              "min clearance um": 2}
    for row in summary:
        cells = [row["configuration"]]
        for label, _ in COLUMNS:
            cells.append(_fmt(row["stats"][label], digits.get(label, 2)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Counted events (mean over seeds):", ""]
    head = ["configuration"] + EVENT_COLUMNS
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for row in summary:
        cells = [row["configuration"]] + [_fmt(row["stats"][key], 1) for key in EVENT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Options per row:", ""]
    for row in summary:
        lines.append(f"- {row['configuration']}: `{' '.join(row['options']) or '(defaults)'}`")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--only", nargs="*", help="substrings selecting configurations to run")
    args = parser.parse_args(argv)
    chosen = [(name, options) for name, options in CONFIGURATIONS
              if not args.only or any(s in name for s in args.only)]
    tasks = [(name, options, seed) for name, options in chosen for seed in args.seeds]
    started = time.time()
    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            records = pool.map(run_one, tasks, chunksize=1)
    else:
        records = [run_one(task) for task in tasks]
    summary = summarise(records)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "descriptors.json"), "w") as handle:
        json.dump({"records": records, "summary": summary}, handle, indent=1)
    with open(os.path.join(args.out, "descriptors.md"), "w") as handle:
        handle.write(markdown(summary, args.seeds))
    print(f"{len(records)} networks in {time.time() - started:.0f} s -> {args.out}/descriptors.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
