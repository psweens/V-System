"""
A reproducible library of networks in relative units.

A library is a fixed set of networks grown once and sampled from afterwards,
rendered at whatever scale a use needs. V-System is scale free: growth at
(lambda d0, lambda d_min, lambda margin, same seed) gives lambda times the
nodes up to float rounding. A library is therefore grown in relative units,
with the smallest drawn vessel diameter d_min = 1 unit, and rescaled on use:
a network of root ratio R = d0 / d_min stands for a vessel tree whose finest
drawn vessel is one unit across and whose root is R units across, at any
physical unit.

    vsystem-library --out lib --count 2000 --seed 1 --workers 8

grows `--count` networks of the families listed (`--families`, in the shares
of `--family-shares`), each from its own seed derived from `--seed` and its
id, with a root ratio drawn log-uniformly and stratified over
`--ratio-range`, and writes one archive per network (`net_{id:05d}_{family}
.npz`, written atomically), an index of descriptors (`index.json`,
`index.csv`) and a manifest (`manifest.json`) whose `content` part names
every parameter, the ratio law, the code the library was grown with and the
hash of every network's nodes and program, so that a library can be checked
and resumed. The growth of each network goes through main's own argument
parser, parameter sampling and grow_network, so a library network equals
what `vsystem` writes for the same settings and seed.

Root ratios. For family f at index j in `--families`, with n_f networks,
rng = default_rng([seed, RNG_STREAMS["ratio"], j]) draws u = rng.random(n_f)
and perm = rng.permutation(n_f), and the k-th network of f gets
R = R_LO (R_HI / R_LO) ** ((perm[k] + u[k]) / n_f): one network per bin of
equal width in log R, in a random order. The bins depend on n_f, so a
library of a different size or share is a new library, not an extension.

Families. Only the presets of main.FAMILIES that exist are accepted; a
family listed without a preset, or an unknown name, is refused before
anything is grown. The default library holds DEFAULT_FAMILIES (tree, mesh
and tumour), so a family added later does not change what `--families`
grows by default. `--avoid-collisions` (the default) affects the tree family
only, the mesh, tumour and aligned presets avoiding collisions already;
plain trees cross themselves from about R = 10, and the setting is recorded
per network.

Mesh and aligned networks grow in a cube of side c times R, with c from
BOX_C (`--mesh-box-c` for mesh, `--box-c FAMILY C` for any family), which
keeps the two trees within reach of each other; tree and tumour networks
grow freely, and the volume the parser is given is unused by their growth.

Every archive carries the network's frame record (see guidance.py) and its
d_min, and the index gives the orientation, polar order, capillary shares,
capillary segment length and transverse spacing that describe.py measures
against that frame.

Resuming. A run into a directory that holds a manifest refuses a different
parameter set or a different code hash, records a different Python or numpy
version with a warning, keeps every archive that loads and matches its plan
(seed, family, ratio and growth settings), regrows the rest, and never
drops a failure silently: failures are listed in the manifest and reported.
Archives are not compared byte for byte, since zip entries carry
timestamps and BLAS rounding differs between machines.

Parallelism uses processes: the grammar draws from the global `random` and
`numpy.random` generators and keeps its parameters in libGenerator's module
globals, so threads would share state. Every network grows in a fresh
process (spawned, with the BLAS thread counts set to one), so the peak
resident set recorded per network is that network's own.

Weights. `library_weights` gives the weight to draw each network with so
that the drawn root ratios follow a power law R ** -exponent per unit log R
rather than the library's log-uniform law.

Python 3.9 compatible; numpy and tifffile (through main) are the only
dependencies.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import resource
import socket
import sys
import tempfile
import time

import numpy as np

import graph
import main as cli
from describe import describe

# Tag of the ratio stream; it shares no tag with main.RNG_STREAMS.
RNG_STREAMS = {"ratio": 4}

# The version of the library format and generator, the release it belongs to.
LIBRARY_VERSION = "3.5.0"

RATIO_LAW = "log-uniform"
UNITS = "d_min"
DEFAULT_RATIO_RANGE = (2.52, 25.0)      # 2 ** (4 / 3) gives at least about four generations
DEFAULT_MESH_BOX_C = 15.0
DEFAULT_ITERATION_CAP = 64
DEFAULT_COLLISION_MARGIN = 1.0

# The families a library holds unless --families says otherwise.
DEFAULT_FAMILIES = ("tree", "mesh", "tumour")

# The families grown in a cube, with the side of the cube in root diameters:
# c R keeps the two trees of a pair within reach of each other.
BOX_C = {"mesh": DEFAULT_MESH_BOX_C, "aligned": 15.0}

# The parser is given this volume: tree and tumour growth never read it, and
# a mesh grows in a cube of side 3 x voxel_size, the voxel size being set to
# c R / 3 so that the cube's side is c R. Three voxels is the smallest volume
# main's rasteriser accepts, which keeps `vsystem` runnable on a library
# network's own arguments.
VOLUME = (3, 3, 3)

# Modules whose source bytes make up the code hash: this one and main's
# import closure.
CODE_MODULES = ("library", "main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance")

# Columns of index.csv, in order. Append-only: the columns from frame_kind on
# were added in 3.5 (orientation about the frame axis, the polar order and
# the capillary shares, segment length and spacing, in d_min; empty where a
# network has no frame axis or no polar sense).
INDEX_COLUMNS = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points", "polylines",
                 "tips", "junctions", "cycles", "components", "total_length",
                 "diameter_p50", "diameter_p90", "diameter_p99",
                 "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min", "diameter_max",
                 "min_point_diameter", "clearance_min", "clearance_violations", "bridges", "events",
                 "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                 "nodes_sha256", "program_sha256",
                 "frame_kind", "capillary_order", "larger_order", "capillary_polar_order",
                 "capillary_length_share", "capillary_volume_share", "capillary_segment_median",
                 "transverse_spacing_median")

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def available_families():
    """The family presets that exist, in main.FAMILIES order."""
    return [name for name, preset in cli.FAMILIES.items() if preset is not None]


def check_families(families):
    """Refuses a family that is not a usable preset, before anything is grown."""
    usable = available_families()
    for family in families:
        if family not in usable:
            if family in cli.FAMILIES:
                raise SystemExit(f"--families {family}: the preset is listed but not available; "
                                 f"choose from {' '.join(usable)}")
            raise SystemExit(f"--families {family}: unknown family; choose from {' '.join(usable)}")
    if len(set(families)) != len(families):
        raise SystemExit("--families lists a family twice")
    return list(families)


def family_counts(count, families, shares):
    """
    Networks per family by largest remainder over `shares`, ties going to the
    family listed first: 2000 at shares 1 1 1 gives 667, 667, 666.
    """
    if count < 0:
        raise ValueError("count cannot be negative")
    if len(shares) != len(families):
        raise ValueError("one share per family is needed")
    shares = [float(s) for s in shares]
    if any(s < 0.0 for s in shares) or sum(shares) <= 0.0:
        raise ValueError("shares must be non-negative and not all zero")
    total = sum(shares)
    exact = [count * s / total for s in shares]
    counts = [int(math.floor(e)) for e in exact]
    remainder = count - sum(counts)
    order = sorted(range(len(families)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def network_seed(seed, index):
    """The seed of network `index` of a library seeded with `seed`."""
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])


def plan_ratios(seed, family_index, count, lo, hi):
    """
    Stratified log-uniform root ratios for one family: (bin, u, ratio) for
    each of its `count` networks in library order.
    """
    rng = np.random.default_rng([int(seed), RNG_STREAMS["ratio"], int(family_index)])
    u = rng.random(count)
    perm = rng.permutation(count)
    return [(int(perm[k]), float(u[k]), float(lo * (hi / lo) ** ((perm[k] + u[k]) / count)))
            for k in range(count)]


def plan_library(seed, count, families, shares, ratio_range):
    """
    The members of a library: one dict per network with its id, family,
    seed, ratio, bin and u, ids running 0 .. count - 1 in contiguous family
    blocks in `families` order.
    """
    lo, hi = (float(v) for v in ratio_range)
    if not 0.0 < lo <= hi:
        raise ValueError("the ratio range must be 0 < lo <= hi")
    counts = family_counts(count, families, shares)
    members = []
    for j, (family, n) in enumerate(zip(families, counts)):
        for k, (bin_, u, ratio) in enumerate(plan_ratios(seed, j, n, lo, hi)):
            index = len(members)
            members.append({"id": index, "family": family, "family_index": j, "position": k,
                            "seed": network_seed(seed, index), "ratio": ratio, "bin": bin_, "u": u,
                            "file": f"net_{index:05d}_{family}.npz"})
    return members


def box_constants(mesh_box_c=None, box_c=None):
    """
    The side of each box family's cube in root diameters: BOX_C, with
    `mesh_box_c` for mesh and the entries of `box_c` over that.

    Raises:
        ValueError: for a mesh entry in `box_c` given together with
        `mesh_box_c`, a family in `box_c` that is not a box family (not in
        BOX_C), or a constant that is not positive.
    """
    boxes = dict(BOX_C)
    for family in box_c or {}:
        if family not in BOX_C:
            raise ValueError(f"{family!r} is not a box family; the box families are {sorted(BOX_C)}")
    if mesh_box_c is not None:
        if box_c and "mesh" in box_c:
            raise ValueError("the mesh box is given twice, as mesh_box_c and in box_c")
        boxes["mesh"] = float(mesh_box_c)
    for family, c in (box_c or {}).items():
        boxes[family] = float(c)
    for family, c in boxes.items():
        if not c > 0.0:
            raise ValueError(f"the box constant of {family} must be positive, got {c!r}")
    return boxes


def member_argv(family, ratio, *, d_min=1.0, collision_margin=DEFAULT_COLLISION_MARGIN,
                mesh_box_c=None, iteration_cap=DEFAULT_ITERATION_CAP, avoid_collisions=True, box_c=None):
    """
    The `vsystem` command line that grows one library network (without
    --count, --seed or --out). A family in BOX_C grows in a cube of side c R:
    the voxel size is set to c d0 / 3 over the three-voxel volume.
    """
    d0 = ratio * d_min
    boxes = box_constants(mesh_box_c, box_c)
    voxel_size = boxes[family] * d0 / VOLUME[0] if family in boxes else 1.0
    argv = ["--family", family, "--d0", repr(float(d0)), "0", "--d0-min", repr(float(d0)),
            "--d-min", repr(float(d_min)), "--iterations", str(int(iteration_cap)), str(int(iteration_cap)),
            "--collision-margin", repr(float(collision_margin)),
            "--volume"] + [str(v) for v in VOLUME] + ["--fit", "voxel_size", "--voxel-size", repr(float(voxel_size))]
    if family == "tree" and avoid_collisions:
        argv.append("--avoid-collisions")
    return argv


def grow_member(family, ratio, seed, *, d_min=1.0, collision_margin=DEFAULT_COLLISION_MARGIN,
                mesh_box_c=None, iteration_cap=DEFAULT_ITERATION_CAP, avoid_collisions=True, box_c=None):
    """
    Grows one library network exactly as `vsystem` would from the same
    arguments and seed: main's parser with the family preset, its parameter
    sampling under the seeded global generators, then grow_network.

    Returns:
        dict: "grown" (the result of grow_network), "argv", "kwargs" (every
        keyword passed to grow_network, JSON-ready), "niter", "d0" and
        "properties".
    """
    argv = member_argv(family, ratio, d_min=d_min, collision_margin=collision_margin, mesh_box_c=mesh_box_c,
                       iteration_cap=iteration_cap, avoid_collisions=avoid_collisions, box_c=box_c)
    args = cli.build_parser(family).parse_args(argv)
    try:
        cli.validate_shaping(args)
    except SystemExit as exc:
        # a refusal of the command line, such as a root offset outside the
        # member's box, is an error of this member, which the library records
        raise ValueError(str(exc)) from None
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    properties, d0, niter = cli.sample_parameters(args)
    if d0 != ratio * d_min:
        raise AssertionError(f"the sampled root diameter {d0!r} is not {ratio * d_min!r}")
    kwargs = {"tVol": tuple(args.volume), "fit": args.fit, "clip_axes": tuple(args.clip_axes),
              "voxel_size": args.voxel_size, "subdivisions": args.subdivisions, "d_min": args.d_min,
              "grow_in_volume": args.grow_in_volume, "seed": seed}
    kwargs.update(cli.shaping_options(args))
    grown = cli.grow_network(niter, d0, properties, **kwargs)
    record = dict(kwargs)
    record["tVol"] = list(record["tVol"])
    record["clip_axes"] = list(record["clip_axes"])
    return {"grown": grown, "argv": argv, "kwargs": record, "niter": niter, "d0": d0, "properties": properties}


def _weighted_percentile(values, weights, fraction):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = cumulative[-1] if cumulative.size else 0.0
    if not total > 0.0:
        return None
    slot = int(np.searchsorted(cumulative, fraction * total, side="left"))
    return float(values[order][min(slot, values.size - 1)])


def _frame_columns(report):
    """The index columns measured against the network's frame, None where there is none."""
    frame = report["frame"]
    orientation = report["frame_orientation"]
    polar = report["polar_order"]
    shares = report["calibre_shares"]
    segments = report["segments_by_class"]
    spacing = report["transverse_spacing"]

    def order(entry):
        return entry["S"] if entry and "S" in entry else None

    return {
        "frame_kind": frame["kind"] if frame else None,
        "capillary_order": order(orientation["capillary"]) if orientation else None,
        "larger_order": order(orientation["larger"]) if orientation else None,
        "capillary_polar_order": polar["capillary"] if polar else None,
        "capillary_length_share": shares["length"]["below_2"] if shares else None,
        "capillary_volume_share": shares["volume"]["below_2"] if shares else None,
        "capillary_segment_median": segments["capillary"]["median"] if segments else None,
        "transverse_spacing_median": spacing["median"] if spacing else None,
    }


def describe_member(grown, collision_margin, d_min=1.0):
    """
    The index row of a grown network: counts, lengths and diameter
    percentiles in units of d_min (describe's _um and _mm names refer to
    micrometres, so its quantities are renamed and the length un-scaled),
    and the frame-relative and calibre-class quantities measured with the
    network's frame record and d_min as the reference diameter.
    "generations" is filled in by the caller from the program.
    """
    nodes = grown["nodes"]
    edges = grown["edges"]
    report = describe(nodes, edges, margin=collision_margin, tree=grown["tree"], frame=grown.get("frame"),
                      d_ref=d_min)
    canonical = graph.canonical_columns(nodes)
    vertex_diameter = graph.vertex_diameter(nodes, canonical)
    edge_diameter = (vertex_diameter[edges[0]] + vertex_diameter[edges[1]]) / 2.0 if edges.shape[1] else np.zeros(0)
    length = np.linalg.norm(nodes[:3, edges[0]] - nodes[:3, edges[1]], axis=0) if edges.shape[1] else np.zeros(0)
    known = np.isfinite(edge_diameter)
    volume_weight = edge_diameter[known] ** 2 * length[known]
    finite = np.isfinite(nodes[3])
    diameters = report["diameter_um"]
    row = {
        "generations": None,
        "points": report["points"], "polylines": report["polylines"],
        "tips": report["tips"]["count"],
        "junctions": sum(v for d, v in report["degree_histogram"].items() if int(d) >= 3),
        "cycles": report["cycles"], "components": report["components"],
        "total_length": report["total_length_mm"] * 1000.0,
        "diameter_p50": diameters["p50"], "diameter_p90": diameters["p90"], "diameter_p99": diameters["p99"],
        "diameter_v50": _weighted_percentile(edge_diameter[known], volume_weight, 0.5),
        "diameter_v90": _weighted_percentile(edge_diameter[known], volume_weight, 0.9),
        "diameter_v99": _weighted_percentile(edge_diameter[known], volume_weight, 0.99),
        "diameter_min": diameters["min"], "diameter_max": diameters["max"],
        "min_point_diameter": float(np.min(nodes[3][finite])) if finite.any() else None,
        "clearance_min": report["clearance_um"]["min"],
        "clearance_violations": report["clearance_um"]["violations"],
        "bridges": len(grown["bridges"]),
        "events": {k: v for k, v in grown["events"].items() if v},
    }
    row.update(_frame_columns(report))
    return row


def drawn_generations(program):
    """
    Generations drawn in a program: the deepest bracket nesting, since every
    generation's daughters are bracketed inside their parent's and the last
    drawn generation's brackets hold only the terminal symbol. With d_min
    deciding where branches stop, this is the depth the network reached
    rather than the iteration cap it was given.
    """
    depth = deepest = 0
    for character in program:
        if character == "[":
            depth += 1
            deepest = max(deepest, depth)
        elif character == "]":
            depth -= 1
    return deepest


def root_columns(tree):
    """The first finite column of each tree label other than BRIDGE."""
    from anastomosis import BRIDGE
    tree = np.asarray(tree)
    return [int(np.flatnonzero(tree == label)[0]) for label in np.unique(tree[tree >= 0]) if label != BRIDGE]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def member_hashes(grown):
    return {"nodes_sha256": sha256_bytes(np.ascontiguousarray(grown["nodes"]).tobytes()),
            "program_sha256": sha256_bytes(grown["program"].encode())}


def save_member_archive(path, grown, metadata):
    """Writes the archive atomically: to a temporary file beside it, then renamed into place."""
    directory = os.path.dirname(os.path.abspath(path))
    handle, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=directory)
    os.close(handle)
    try:
        cli.save_network(temporary, grown["nodes"], program=grown["program"], metadata=metadata,
                         edges=grown["edges"], node_kind=grown["node_kind"], tree=grown["tree"])
        mask = os.umask(0)                    # mkstemp's private mode is not what a library wants
        os.umask(mask)
        os.chmod(temporary, 0o666 & ~mask)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def code_hash(root=None):
    """sha256 of the source bytes of library.py and main's import closure, in a fixed order."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in CODE_MODULES:
        with open(os.path.join(root, name + ".py"), "rb") as handle:
            digest.update(name.encode() + b"\n")
            digest.update(handle.read())
    return digest.hexdigest()


def growth_settings(args, box_c=None):
    """
    The growth parameters of a library, as the manifest records them. The
    mesh box constant is recorded as before; the constants of the other box
    families listed in `--families` are recorded under "box_c", which is
    absent when none is listed, so that a library of the default families
    records what it always did.
    """
    boxes = box_constants(args.mesh_box_c, box_c)
    families = args.families if args.families else list(DEFAULT_FAMILIES)
    settings = {"collision_margin": float(args.collision_margin), "mesh_box_c": boxes["mesh"],
                "iteration_cap": int(args.iteration_cap), "avoid_collisions": bool(args.avoid_collisions),
                "d_min": 1.0}
    listed = {family: boxes[family] for family in families if family in boxes and family != "mesh"}
    if listed:
        settings["box_c"] = listed
    return settings


def grow_and_write(task):
    """
    Grows one network, writes its archive and returns its index row, or a
    failure record. Runs in a worker process of its own.
    """
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    member, settings, directory = task["member"], task["settings"], task["directory"]
    started = time.perf_counter()
    try:
        result = grow_member(member["family"], member["ratio"], member["seed"], d_min=settings["d_min"],
                             collision_margin=settings["collision_margin"], mesh_box_c=settings["mesh_box_c"],
                             iteration_cap=settings["iteration_cap"], avoid_collisions=settings["avoid_collisions"],
                             box_c=settings.get("box_c"))
        grown = result["grown"]
        row = describe_member(grown, settings["collision_margin"], settings["d_min"])
        row["generations"] = max(drawn_generations(p) for p in grown["programs"])
        hashes = member_hashes(grown)
        seconds = time.perf_counter() - started
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        metadata = {
            "units": UNITS, "library_id": member["id"], "family": member["family"], "ratio": member["ratio"],
            "bin": member["bin"], "u": member["u"], "seed": member["seed"],
            "ratio_law": RATIO_LAW, "ratio_range": [settings["ratio_lo"], settings["ratio_hi"]],
            "argv": result["argv"], "niter": result["niter"], "generations": row["generations"], "d0": result["d0"],
            "properties": result["properties"], "grow_kwargs": result["kwargs"],
            "avoid_collisions": bool(result["kwargs"]["avoid_collisions"]),
            "events": grown["events"], "root_columns": root_columns(grown["tree"]),
            "growth_box": grown["growth_box_um"], "root_positions": grown["root_positions_um"],
            "bridges": grown["bridges"], "min_point_diameter": row["min_point_diameter"],
            "programs": grown["programs"], "archive_version": cli.ARCHIVE_VERSION,
            "seconds": seconds, "peak_rss_mb": peak_rss_mb,
            "frame": grown["frame"], "d_min": settings["d_min"],
        }
        metadata.update(hashes)
        save_member_archive(os.path.join(directory, member["file"]), grown, metadata)
        row.update({"id": member["id"], "family": member["family"], "ratio": member["ratio"], "bin": member["bin"],
                    "u": member["u"], "seed": member["seed"], "file": member["file"],
                    "avoid_collisions": bool(result["kwargs"]["avoid_collisions"]), "ratio_law": RATIO_LAW,
                    "ratio_lo": settings["ratio_lo"], "ratio_hi": settings["ratio_hi"],
                    "seconds": seconds, "peak_rss_mb": peak_rss_mb})
        row.update(hashes)
        return {"id": member["id"], "row": row}
    except Exception as exc:                      # the failure is reported, never swallowed
        return {"id": member["id"], "failure": {"id": member["id"], "family": member["family"],
                                                "seed": member["seed"], "ratio": member["ratio"],
                                                "reason": f"{type(exc).__name__}: {exc}",
                                                "seconds": time.perf_counter() - started}}


def validate_existing(path, member, settings):
    """
    Whether an archive on disk is the planned network: it loads and its
    metadata carries the member's seed, family and ratio and the library's
    growth settings. Returns (row, None) when it is, (None, reason) when it
    is not (a reason starting with "unreadable" means regrow it).
    """
    try:
        network = cli.load_network(path)
    except Exception as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    record = network["metadata"] or {}
    expected = {"library_id": member["id"], "family": member["family"], "seed": member["seed"],
                "ratio": member["ratio"], "bin": member["bin"], "u": member["u"], "units": UNITS,
                "ratio_law": RATIO_LAW, "ratio_range": [settings["ratio_lo"], settings["ratio_hi"]]}
    for key, value in expected.items():
        if record.get(key) != value:
            return None, f"{key} is {record.get(key)!r}, planned {value!r}"
    argv = member_argv(member["family"], member["ratio"], d_min=settings["d_min"],
                       collision_margin=settings["collision_margin"], mesh_box_c=settings["mesh_box_c"],
                       iteration_cap=settings["iteration_cap"], avoid_collisions=settings["avoid_collisions"],
                       box_c=settings.get("box_c"))
    if record.get("argv") != argv:
        return None, f"grown from {record.get('argv')!r}, planned {argv!r}"
    if "nodes_sha256" not in record or record["nodes_sha256"] != sha256_bytes(np.ascontiguousarray(network["nodes"]).tobytes()):
        return None, "unreadable: the nodes do not match their recorded hash"
    return record, None


def manifest_content(seed, count, families, shares, ratio_range, settings, workers_unused, members, rows, failures,
                     code):
    """The reproducible part of the manifest; its canonical JSON is hashed."""
    counts = family_counts(count, families, shares)
    return {
        "version": cli.ARCHIVE_VERSION, "library_version": LIBRARY_VERSION, "units": UNITS,
        "seed": int(seed), "count": int(count), "families": list(families),
        "family_shares": [float(s) for s in shares], "family_counts": dict(zip(families, counts)),
        "ratio_law": RATIO_LAW, "ratio_range": [float(v) for v in ratio_range],
        "growth": settings, "rng_streams": dict(RNG_STREAMS), "code_sha256": code,
        "networks": {str(m["id"]): {"family": m["family"], "seed": m["seed"], "ratio": m["ratio"],
                                    "bin": m["bin"], "u": m["u"], "file": m["file"],
                                    "nodes_sha256": rows[m["id"]]["nodes_sha256"] if m["id"] in rows else None,
                                    "program_sha256": rows[m["id"]]["program_sha256"] if m["id"] in rows else None}
                     for m in members},
        "failures": sorted(failures, key=lambda f: f["id"]),
    }


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(content):
    return sha256_bytes(canonical_json(content).encode())


def write_index(directory, rows):
    ordered = [rows[k] for k in sorted(rows)]
    index = {"units": UNITS, "ratio_law": RATIO_LAW, "networks": ordered}
    with open(os.path.join(directory, "index.json"), "w") as handle:
        json.dump(index, handle, indent=2, allow_nan=False)
    with open(os.path.join(directory, "index.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        for row in ordered:
            values = []
            for column in INDEX_COLUMNS:
                value = row.get(column)
                if isinstance(value, dict):
                    value = " ".join(f"{k}={v}" for k, v in sorted(value.items()))
                values.append("" if value is None else value)
            writer.writerow(values)
    return index


def load_index(path):
    with open(path) as handle:
        return json.load(handle)


def library_weights(index, exponent=3.0, mask=None):
    """
    Weights to draw the networks of a library with, so that the drawn root
    ratios follow p_target(R) proportional to R ** -exponent per unit log R.

    w_i is proportional to p_target(R_i) / p_library(R_i), with p_library
    read from the recorded law: constant per unit log R for log-uniform.
    The weights are normalised to sum one over `mask` and are zero elsewhere.

    Args:
        index (dict or sequence): index.json as loaded, or its "networks" rows.
        exponent (float): the power of the target law.
        mask (sequence or None): boolean, which networks may be drawn.

    Returns:
        ndarray: one weight per network, in index order.
    """
    rows = index["networks"] if isinstance(index, dict) else list(index)
    laws = {row.get("ratio_law", RATIO_LAW) for row in rows}
    if laws - {RATIO_LAW}:
        raise ValueError(f"unknown ratio law {sorted(laws - {RATIO_LAW})}")
    ratio = np.array([float(row["ratio"]) for row in rows])
    if np.any(ratio <= 0.0):
        raise ValueError("ratios must be positive")
    # p_library is constant per unit log R, so the ratio of the two densities is the target law itself
    weight = ratio ** (-float(exponent))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ratio.shape:
            raise ValueError("mask must hold one entry per network")
        weight = np.where(mask, weight, 0.0)
    total = weight.sum()
    if not total > 0.0:
        raise ValueError("no network can be drawn")
    return weight / total


def effective_sample_size(weights):
    """(sum w)^2 / sum w^2, the number of equally weighted draws the weights are worth."""
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Grow a reproducible library of networks in relative units (d_min = 1), with an index "
                    "and a manifest.")
    parser.add_argument("--out", required=True, metavar="DIR", help="output directory; a run into a "
                        "directory holding a manifest resumes it")
    parser.add_argument("--count", type=int, required=True, metavar="N", help="number of networks")
    parser.add_argument("--families", nargs="+", default=None, metavar="FAMILY",
                        help=f"families to grow, in id order (default: {' '.join(DEFAULT_FAMILIES)})")
    parser.add_argument("--family-shares", type=float, nargs="+", default=None, metavar="SHARE",
                        help="relative share of each family (default: equal)")
    parser.add_argument("--ratio-range", type=float, nargs=2, default=DEFAULT_RATIO_RANGE, metavar=("R_LO", "R_HI"),
                        help=f"range of the root ratio d0 / d_min, log-uniform and stratified "
                             f"(default {DEFAULT_RATIO_RANGE[0]} {DEFAULT_RATIO_RANGE[1]})")
    parser.add_argument("--seed", type=int, required=True, help="library seed")
    parser.add_argument("--workers", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--collision-margin", type=float, default=DEFAULT_COLLISION_MARGIN,
                        help=f"clearance between vessel surfaces, in units of d_min (default {DEFAULT_COLLISION_MARGIN})")
    parser.add_argument("--mesh-box-c", type=float, default=None,
                        help=f"a mesh grows in a cube of side this many times its root diameter (default {DEFAULT_MESH_BOX_C})")
    parser.add_argument("--box-c", nargs=2, action="append", default=None, metavar=("FAMILY", "C"),
                        help="the side of the cube a family grows in, in root diameters, for any family grown in a "
                             "box (repeatable; defaults: " + ", ".join(f"{f} {c:g}" for f, c in BOX_C.items())
                             + "; not together with --mesh-box-c for mesh)")
    parser.add_argument("--iteration-cap", type=int, default=DEFAULT_ITERATION_CAP,
                        help=f"generations allowed; d_min stops growth first (default {DEFAULT_ITERATION_CAP})")
    parser.add_argument("--avoid-collisions", dest="avoid_collisions", action="store_true", default=True,
                        help="grow trees with collision avoidance (default; mesh and tumour always do)")
    parser.add_argument("--no-avoid-collisions", dest="avoid_collisions", action="store_false",
                        help="grow plain trees, which cross themselves from about R = 10")
    return parser


def _warn(message):
    print("warning: " + message, file=sys.stderr, flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    families = check_families(args.families if args.families else list(DEFAULT_FAMILIES))
    box_c = {}
    for family, c in args.box_c or ():
        if family not in BOX_C:
            raise SystemExit(f"--box-c {family}: not a box family; choose from {' '.join(sorted(BOX_C))}")
        try:
            box_c[family] = float(c)
        except ValueError:
            raise SystemExit(f"--box-c {family} {c}: the constant must be a number")
        if not box_c[family] > 0.0:
            raise SystemExit(f"--box-c {family} {c}: the constant must be positive")
    if args.mesh_box_c is not None and "mesh" in box_c:
        raise SystemExit("the mesh box is given twice: use --mesh-box-c or --box-c mesh, not both")
    shares = args.family_shares if args.family_shares else [1.0] * len(families)
    if len(shares) != len(families):
        raise SystemExit("--family-shares needs one share per family")
    if args.count < 0:
        raise SystemExit("--count cannot be negative")
    lo, hi = args.ratio_range
    if not 0.0 < lo <= hi:
        raise SystemExit("--ratio-range must satisfy 0 < R_LO <= R_HI")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.collision_margin < 0.0:
        raise SystemExit("--collision-margin cannot be negative")
    if (args.mesh_box_c is not None and args.mesh_box_c <= 0.0) or args.iteration_cap < 1:
        raise SystemExit("--mesh-box-c must be positive and --iteration-cap at least 1")

    settings = growth_settings(args, box_c)
    settings.update({"ratio_lo": float(lo), "ratio_hi": float(hi)})
    members = plan_library(args.seed, args.count, families, shares, (lo, hi))
    code = code_hash()
    content_now = manifest_content(args.seed, args.count, families, shares, (lo, hi), growth_settings(args, box_c),
                                   None, members, {}, [], code)
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    run_record = {"host": socket.gethostname(), "platform": platform.platform(),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "started": datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs": []}
    rows = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            previous = json.load(handle)
        before = previous.get("content", {})
        for key in ("seed", "count", "families", "family_shares", "ratio_range", "growth", "ratio_law", "units"):
            if before.get(key) != content_now[key]:
                raise SystemExit(f"{manifest_path}: cannot resume, {key} was {before.get(key)!r} and is now "
                                 f"{content_now[key]!r}; use another --out for a different library")
        if before.get("code_sha256") != code:
            raise SystemExit(f"{manifest_path}: cannot resume, the code hash was {before.get('code_sha256')} "
                             f"and is now {code}; the generator changed, so this would be a different library")
        run_before = previous.get("run", {})
        for key in ("python", "numpy"):
            if run_before.get(key) != run_record[key]:
                _warn(f"resuming with {key} {run_record[key]}, the library was started with {run_before.get(key)}; "
                      f"last bits of the geometry may differ between the two")
        run_record["runs"] = run_before.get("runs", [])
        if os.path.exists(os.path.join(args.out, "index.json")):
            for row in load_index(os.path.join(args.out, "index.json"))["networks"]:
                rows[int(row["id"])] = row

    # keep every archive that is the planned network, regrow the rest
    pending = []
    kept = 0
    for member in members:
        path = os.path.join(args.out, member["file"])
        if os.path.exists(path):
            record, reason = validate_existing(path, member, settings)
            if record is not None and member["id"] in rows:
                kept += 1
                continue
            if record is not None:
                # a valid archive without an index row: its row is rebuilt from the archive
                network = cli.load_network(path)
                grown = {"nodes": network["nodes"], "edges": network["edges"], "tree": network["tree"],
                         "bridges": record.get("bridges", []), "events": record.get("events", {}),
                         "program": network["program"], "frame": record.get("frame")}
                row = describe_member(grown, settings["collision_margin"], settings["d_min"])
                row.update({"id": member["id"], "family": member["family"], "ratio": member["ratio"],
                            "bin": member["bin"], "u": member["u"], "seed": member["seed"], "file": member["file"],
                            "generations": record.get("generations"), "avoid_collisions": record.get("avoid_collisions"),
                            "ratio_law": RATIO_LAW, "ratio_lo": settings["ratio_lo"], "ratio_hi": settings["ratio_hi"],
                            "seconds": record.get("seconds"), "peak_rss_mb": record.get("peak_rss_mb"),
                            "nodes_sha256": record.get("nodes_sha256"), "program_sha256": record.get("program_sha256")})
                rows[member["id"]] = row
                kept += 1
                continue
            if reason.startswith("unreadable"):
                _warn(f"{member['file']}: {reason}; it is grown again")
            else:
                raise SystemExit(f"{path}: not the planned network ({reason}); use another --out")
        pending.append({"member": member, "settings": settings, "directory": args.out})

    failures = []
    started = time.perf_counter()
    if pending:
        for name in THREAD_VARIABLES:
            os.environ[name] = "1"
        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(grow_and_write, pending):
                if "failure" in result:
                    failures.append(result["failure"])
                    print(f"net {result['id']:05d}: FAILED {result['failure']['reason']}", file=sys.stderr, flush=True)
                    continue
                row = result["row"]
                rows[result["id"]] = row
                print(f"{row['file']}: {row['family']} R {row['ratio']:.2f}, {row['generations']} generations, "
                      f"{row['points']} points, {row['tips']} tips, {row['cycles']} cycles, "
                      f"{row['clearance_violations']} clearance violations, {row['seconds']:.1f} s, "
                      f"peak RSS {row['peak_rss_mb']:.0f} MB", flush=True)
    elapsed = time.perf_counter() - started

    content = manifest_content(args.seed, args.count, families, shares, (lo, hi), growth_settings(args, box_c),
                               None, members, rows, failures, code)
    run_record["runs"].append({"started": run_record.pop("started"),
                               "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               "workers": args.workers, "grown": len(pending) - len(failures),
                               "kept": kept, "failed": len(failures), "seconds": elapsed})
    run_record["per_network"] = {str(k): {"seconds": rows[k].get("seconds"), "peak_rss_mb": rows[k].get("peak_rss_mb")}
                                 for k in sorted(rows)}
    manifest = {"content": content, "content_sha256": content_hash(content), "run": run_record}
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    write_index(args.out, rows)
    print(f"{len(rows)} of {len(members)} networks in {args.out} ({kept} kept, {len(pending) - len(failures)} grown, "
          f"{len(failures)} failed); manifest {manifest['content_sha256'][:16]}", flush=True)
    if failures:
        for failure in failures:
            print(f"net {failure['id']:05d} ({failure['family']}, seed {failure['seed']}): {failure['reason']}",
                  file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
