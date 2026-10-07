"""
Pins what 3.4 draws, writes, measures and joins.

The 3.4 modules (tests/fixtures/reference_code_3_4: main's import closure as
released, with describe, join and library) and the current ones each run one
driver in a subprocess, in the same environment, and the sha256 of everything
the driver produces are compared:

    grow_network   the plain tree (stems, with avoidance, walked, walked with
                   avoidance and anastomosis), the mesh and tumour presets, a
                   mesh under the voxel_size fit, and the layouts and options
                   those leave out: a planar tree whose stems avoidance
                   shortens, growth confined to the volume with and without
                   avoidance, bridges on an unwalked tree, a second tree
                   placed where a free first one ends, the default properties;
    library        grow_member for tree, mesh and tumour at two root ratios,
                   with its argv, kwargs and index row, and two library plans;
    describe       the three fixture archives, one of them also at a margin,
                   the archives the command line writes and a joined forest;
    main.main      one tree and one mesh run and a run of two networks with
                   other options: archive, sidecar, TIFF pixels and TIFF file;
    join_networks  a cropped forest of the library members: the crops, the
                   joining with root attachment off and on and under the
                   policy "any", and the merged archive;
    library.main   a three-network library: the files written, manifest
                   content, index rows (index.json and index.csv), archives
                   and their metadata.

Arrays, programs and bridges are compared whole. Every record (events,
sidecars, metadata, kwargs, describe output, index rows, join output) is
first restricted, recursively, to the keys 3.4 wrote, which are listed
literally below, so that what a later version appends never enters the
comparison while everything 3.4 wrote must still be there, unchanged. The
global generators' states after each case, the states of the generators
each stage created and the joining generator's state are compared too; a
stage added later must therefore create no generator while its option is off.
With every later option off, whatever a record holds beyond its 3.4 keys
must be inert: every counter appended to the events is zero.

Running both in one environment is the only portable comparison: the
B-spline, the walk and the collision checks go through kernels whose last
bits differ between machines. Hashes recorded on a machine are checked as
well: tests/fixtures/reference_hashes_3_4.json on macOS arm64 and
reference_hashes_3_4_linux.json on Linux x86_64, the platform of CI. A case
passes when it matches a recording, and the check is skipped only when the
reference code itself reproduces none of them here. The
reference directory is an unchanged copy of the release: the sha256 of each
of its files is recorded with the hashes and checked, and the drivers write
nothing but to a temporary directory outside the repository. The two drivers
run side by side, each in its own process and its own output directory,
which halves the time the pin takes.

Run from the repository root with
    python -m unittest tests.test_pinned_release_3_4
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
REFERENCE = os.path.join(FIXTURES, "reference_code_3_4")
RECORDINGS = "reference_hashes_3_4"           # tests/fixtures/reference_hashes_3_4*.json, one per recording

REFERENCE_MODULES = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial",
                     "describe", "join", "library")
ARCHIVES = ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz", "Lnet_i9_s3.npz")

# ---------------------------------------------------------------------------
# The keys 3.4 wrote, listed literally so that a later version's additions
# never enter the comparison.
# ---------------------------------------------------------------------------
EVENT_KEYS_3_4 = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
                  "collision_redraws", "collision_terminations", "collision_truncated_stems",
                  "root_relocations", "root_collisions",
                  "anastomosis_tips", "anastomosis_tips_inside_junction", "anastomosis_kin_skipped",
                  "anastomosis_behind_skipped", "anastomosis_selected", "anastomosis_bridges",
                  "anastomosis_bridges_arteriovenous", "anastomosis_bridges_same_tree",
                  "anastomosis_partner_tips", "anastomosis_partner_interior",
                  "anastomosis_source_consumed", "anastomosis_no_partner",
                  "anastomosis_collision_failed", "anastomosis_bridge_redraws")
SHAPING_KEYS_3_4 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                    "collision_attempts", "collision_index", "anastomose", "anastomosis_radius",
                    "anastomosis_fraction", "anastomose_mode", "anastomosis_min_separation")
KWARGS_KEYS_3_4 = ("tVol", "fit", "clip_axes", "voxel_size", "subdivisions", "d_min",
                   "grow_in_volume", "seed") + SHAPING_KEYS_3_4
PROPERTY_KEYS_3_4 = ("k", "epsilon", "randmarg", "sigma", "stochparams", "roll_angle", "stem_angle",
                     "aneurysm_prob", "stenosis_prob")
# what grow_network returned beside its arrays, programs, bridges and events
GROWN_KEYS_3_4 = ("program", "growth_box_um", "root_positions_um", "clip_axes", "collision_index")
SIDECAR_KEYS_3_4 = ("seed", "iterations", "d0", "d_min", "volume", "axis_order", "units", "fit",
                    "clip_axes", "connect", "grow_in_volume", "voxel_size", "subdivisions",
                    "archive_version", "family", "trees", "growth_box_um", "root_positions_um",
                    "bridges", "bridge_persistence") + SHAPING_KEYS_3_4
RNG_STREAMS_3_4 = ("walk", "anastomosis")
MEMBER_KEYS_3_4 = ("argv", "niter", "d0")
PLAN_KEYS_3_4 = ("id", "family", "family_index", "position", "seed", "ratio", "bin", "u", "file")
# index.csv's columns; a run's seconds and peak memory are left out of the comparison
INDEX_COLUMNS_3_4 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points",
                     "polylines", "tips", "junctions", "cycles", "components", "total_length",
                     "diameter_p50", "diameter_p90", "diameter_p99",
                     "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min", "diameter_max",
                     "min_point_diameter", "clearance_min", "clearance_violations", "bridges", "events",
                     "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                     "nodes_sha256", "program_sha256")
UNPINNED_COLUMNS = ("seconds", "peak_rss_mb")
METADATA_KEYS_3_4 = ("units", "library_id", "family", "ratio", "bin", "u", "seed", "ratio_law",
                     "ratio_range", "argv", "niter", "generations", "d0", "avoid_collisions",
                     "root_columns", "growth_box", "root_positions", "bridges", "min_point_diameter",
                     "programs", "archive_version", "nodes_sha256", "program_sha256")
UNPINNED_MANIFEST_KEYS = ("code_sha256", "library_version")
FLAG_KEYS_3_4 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                 "anastomose", "anastomosis_fraction", "anastomosis_radius", "anastomose_mode",
                 "family", "seed", "iterations", "d0", "d_min")
DESCRIBE_KEYS_3_4 = ("units", "points", "polylines", "vertices", "edges", "components", "cycles",
                     "total_length_mm", "degree_histogram", "length_density_mm_per_mm3", "volume_mm3",
                     "volume_source")
DESCRIBE_GROUPS_3_4 = {
    "diameter_um": ("p50", "p90", "p99", "min", "max"),
    "arc_chord": ("mean", "median", "p90", "max", "n_segments", "n_closed"),
    "curvature_per_um": ("mean_abs", "n_pairs"),
    "tips": ("count", "per_mm"),
    "orientation": ("eigenvalues", "principal_axis", "fractional_anisotropy", "polar_histogram_deg",
                    "azimuth_histogram_deg"),
    "bounding_box_um": ("min", "max"),
    "clearance_um": ("min", "min_columns", "min_is_lower_bound", "search_reach_um", "violations",
                     "pairs_checked", "margin"),
}
PER_TREE_KEYS_3_4 = ("points", "polylines", "tips", "length_mm")
JOIN_EVENT_KEYS_3_4 = ("join_networks", "join_points", "join_isolated", "join_tips",
                       "join_roots", "join_cut_ends", "join_stubs", "join_eligible", "join_selected",
                       "join_not_selected", "join_bridges", "join_bridged_partners", "join_no_partner",
                       "join_collision_failed", "join_over_budget",
                       "join_partner_tips", "join_partner_interior", "join_source_consumed",
                       "join_redraws", "join_box_redraws", "join_kin_skipped", "join_behind_skipped",
                       "join_components_joined",
                       "join_root_eligible", "join_root_selected", "join_root_attached",
                       "join_root_not_selected", "join_root_no_partner", "join_root_collision_failed",
                       "join_root_over_budget")
JOIN_SUMMARY_KEYS_3_4 = ("components", "length", "free_tips", "free_tips_per_length", "bridge_volume",
                         "blunt_roots", "blunt_roots_per_length", "free_ends", "free_ends_per_length")
JOIN_SUMMARY_ENTRY_KEYS_3_4 = ("before", "after")        # of every summary entry but bridge_volume, a number
JOIN_MERGED_KEYS_3_4 = ("nodes", "edges", "node_kind", "network", "tree")
ARCHIVE_ARRAYS_3_4 = ("nodes", "program", "metadata", "edges", "node_kind", "tree")
INDEX_KEYS_3_4 = ("units", "ratio_law", "networks")
JOIN_BRIDGE_FIELDS_3_4 = ("tip", "partner", "source_kind", "partner_kind", "chord", "arc", "diameter",
                          "volume", "redraws")


def _whole(keys):
    return {key: None for key in keys}


# How each record is restricted before it is hashed. None keeps a value whole,
# a dict keeps the listed keys and restricts each value in turn, and {"*": rule}
# keeps every key (a label or an id, not a field) and restricts each value.
RULES = {
    "events": _whole(EVENT_KEYS_3_4),
    "grown": _whole(GROWN_KEYS_3_4),
    "kwargs": _whole(KWARGS_KEYS_3_4),
    "member": dict(_whole(MEMBER_KEYS_3_4), properties=_whole(PROPERTY_KEYS_3_4)),
    "plan": _whole(PLAN_KEYS_3_4),
    "sidecar": dict(_whole(SIDECAR_KEYS_3_4), properties=_whole(PROPERTY_KEYS_3_4),
                    rng_streams=_whole(RNG_STREAMS_3_4), events=_whole(EVENT_KEYS_3_4)),
    "index_row": dict(_whole(column for column in INDEX_COLUMNS_3_4 if column not in UNPINNED_COLUMNS),
                      events=_whole(EVENT_KEYS_3_4)),
    "metadata": dict(_whole(METADATA_KEYS_3_4), properties=_whole(PROPERTY_KEYS_3_4),
                     grow_kwargs=_whole(KWARGS_KEYS_3_4), events=_whole(EVENT_KEYS_3_4)),
    "describe": dict(_whole(DESCRIBE_KEYS_3_4), events=_whole(EVENT_KEYS_3_4), flags=_whole(FLAG_KEYS_3_4),
                     per_tree={"*": _whole(PER_TREE_KEYS_3_4)},
                     **{group: _whole(keys) for group, keys in DESCRIBE_GROUPS_3_4.items()}),
    "join_events": _whole(JOIN_EVENT_KEYS_3_4),
    "join_summary": dict({key: _whole(JOIN_SUMMARY_ENTRY_KEYS_3_4) for key in JOIN_SUMMARY_KEYS_3_4},
                         bridge_volume=None),
    "join_bridge": _whole(JOIN_BRIDGE_FIELDS_3_4),
    "index": _whole(INDEX_KEYS_3_4),
}

# Plain lists the driver reads beside the rules.
LISTS = {
    "modules": list(REFERENCE_MODULES),
    "archives": list(ARCHIVES),
    "archive_arrays": list(ARCHIVE_ARRAYS_3_4),
    "join_merged": list(JOIN_MERGED_KEYS_3_4),
    "index_columns": list(INDEX_COLUMNS_3_4),
    "unpinned_columns": list(UNPINNED_COLUMNS),
    "unpinned_manifest": list(UNPINNED_MANIFEST_KEYS),
}

# Runs in a subprocess with the modules of one code directory first on the
# path and that directory as the working directory, and prints, per case,
# the hashes of what those modules produce ("hashes"), a few plain counts of
# the same output ("facts") and whatever the output holds beyond the 3.4 keys
# ("extras"). It uses nothing but what 3.4 offers.
DRIVER = r"""
import contextlib, csv, hashlib, importlib, io, json, os, random, sys
code_dir, out_dir, fixtures = sys.argv[1], sys.argv[2], sys.argv[3]
rules, lists = json.loads(sys.argv[4]), json.loads(sys.argv[5])
sys.path.insert(0, code_dir)
import numpy as np
import tifffile
import describe as describe_module
import join as join_module
import library
import main
from spatial import GridIndex

for name in lists["modules"]:
    module = importlib.import_module(name)
    if os.path.realpath(os.path.dirname(module.__file__)) != os.path.realpath(code_dir):
        raise SystemExit(f"{name} was imported from {module.__file__}, not from {code_dir}")

# every generator a stage creates is recorded, with the seed it was given, so
# that its state after the case can be compared
default_rng = np.random.default_rng
created = []


def recording_rng(*args, **kwargs):
    generator = default_rng(*args, **kwargs)
    created.append((args, generator))
    return generator


np.random.default_rng = recording_rng


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=plain).encode()


def record_sha(value):
    return sha(canonical(value))


def array_sha(array):
    array = np.ascontiguousarray(array)
    return sha(f"{array.dtype.str} {array.shape} ".encode() + array.tobytes())


def restrict(value, rule):
    # the 3.4 part of a record; see RULES in the test
    if rule is None or not isinstance(value, dict):
        return value
    if "*" in rule:
        return {key: restrict(item, rule["*"]) for key, item in value.items()}
    return {key: restrict(value[key], sub) for key, sub in rule.items() if key in value}


def beyond(value, rule):
    # what a record holds beyond its 3.4 keys, one level deep
    return {key: item for key, item in value.items() if key not in rule}


def global_generators():
    version, internal, gauss = random.getstate()
    name, keys, position, has_gauss, cached = np.random.get_state()
    return record_sha([version, list(internal), gauss, name, keys.tolist(), int(position), int(has_gauss),
                       float(cached)])


def stage_generators():
    states = [{"seed": list(args), "state": generator.bit_generator.state} for args, generator in created]
    del created[:]
    return record_sha(states)


def grown_hashes(grown):
    return {"nodes": array_sha(grown["nodes"]), "tree": array_sha(grown["tree"]),
            "edges": array_sha(grown["edges"]), "node_kind": array_sha(grown["node_kind"]),
            "programs": record_sha(grown["programs"]), "bridges": record_sha(grown["bridges"]),
            "events": record_sha(restrict(grown["events"], rules["events"])),
            "record": record_sha(restrict(grown, rules["grown"])),
            "global_generators": global_generators(), "stage_generators": stage_generators()}


def grown_facts(grown):
    return {"columns": int(grown["nodes"].shape[1]), "bridges": len(grown["bridges"]),
            "trees": len(grown["programs"]),
            "events": {key: grown["events"][key] for key in rules["events"] if grown["events"].get(key)}}


def grown_extras(grown):
    known = set(rules["grown"]) | {"nodes", "tree", "edges", "node_kind", "programs", "bridges", "events"}
    return {"events": beyond(grown["events"], rules["events"]),
            "returned": {key: grown[key] for key in sorted(set(grown) - known)}}


def archive_hashes(path, metadata_rule):
    # the arrays 3.4 stored, as stored (read with numpy, not with the loader under test)
    with np.load(path, allow_pickle=False) as handle:
        stored = {name: handle[name] for name in handle.files}
    entry = {}
    for name in lists["archive_arrays"]:
        if name == "program":
            entry[name] = record_sha(stored[name].item())
        elif name == "metadata":
            entry[name] = record_sha(restrict(json.loads(stored[name].item()), rules[metadata_rule]))
        else:
            entry[name] = array_sha(stored[name])
    return entry, sorted(set(stored) - set(lists["archive_arrays"])), json.loads(stored["metadata"].item())


def describe_case(name, report):
    hashes[name] = {"report": record_sha(restrict(report, rules["describe"]))}
    facts[name] = {"points": report["points"], "components": report["components"], "cycles": report["cycles"],
                   "volume_source": report["volume_source"], "violations": report["clearance_um"]["violations"],
                   "events": len(restrict(report["events"], rules["events"])),
                   "flags": sorted(restrict(report["flags"], rules["describe"]["flags"])),
                   "trees": sorted(report["per_tree"] or {})}
    extras[name] = {"report": sorted(beyond(report, rules["describe"])),
                    "events": beyond(report["events"], rules["events"])}


hashes, facts, extras = {}, {}, {}

# --- grow_network ----------------------------------------------------------
PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
VOLUME = (48, 48, 24)
WALK = {"tortuosity": "walk", "persistence": 8.0}


def preset(family):
    # a family's options as tests/test_pinned_geometry.py passes them
    properties = dict(PROPERTIES)
    options = dict(main.FAMILIES[family])
    for key in ("aneurysm_prob", "stenosis_prob"):
        if key in options:
            properties[key] = options.pop(key)
    options.pop("d0", None)
    return properties, options


GROW_CASES = (
    ("tree_stems", 11, 6, PROPERTIES, {}),
    ("tree_stems_avoidance", 12, 6, PROPERTIES, {"avoid_collisions": True}),
    ("tree_walk", 13, 6, PROPERTIES, dict(WALK)),
    ("tree_walk_avoidance_anastomosis", 14, 6, PROPERTIES,
     dict(WALK, avoid_collisions=True, anastomose=True, anastomose_mode="any", anastomosis_min_separation=2)),
    ("mesh", 3, 6) + preset("mesh"),
    ("tumour", 1, 6) + preset("tumour"),
    # a box tight enough that a walk is terminated at its faces and the second root is moved
    ("mesh_voxel_size", 6, 6, preset("mesh")[0], dict(preset("mesh")[1], fit="voxel_size", voxel_size=3.0)),
    # what the cases above leave out. A planar tree, which crosses itself, so
    # that avoidance shortens stems (the plain tree of this size never collides)
    ("tree_stems_avoidance_planar", 3, 6, dict(PROPERTIES, roll_angle=0.0), {"avoid_collisions": True}),
    # the default interpreter and the shaped one confined to the volume, with d_min
    ("tree_stems_in_volume", 16, 6, PROPERTIES, {"grow_in_volume": True, "d_min": 4.0}),
    ("tree_stems_avoidance_in_volume", 16, 6, PROPERTIES,
     {"avoid_collisions": True, "grow_in_volume": True, "d_min": 4.0}),
    # bridges on an unwalked tree, which take the default bridge persistence
    ("tree_stems_anastomosis", 14, 6, PROPERTIES, {"anastomose": True}),
    # a second tree placed where a free first one ends
    ("tree_walk_arteriovenous_free", 15, 5, PROPERTIES,
     dict(WALK, anastomose=True, anastomose_mode="arteriovenous")),
    # libGenerator's own defaults and another sampling depth, stems and walked
    ("tree_default_properties", 3, 5, {}, {"subdivisions": 1}),
    ("tree_walk_subdivisions", 4, 5, {}, dict(WALK, subdivisions=2, avoid_collisions=True)),
)
for name, seed, niter, properties, options in GROW_CASES:
    random.seed(seed)
    np.random.seed(seed)
    grown = main.grow_network(niter, 20.0, dict(properties), VOLUME, seed=seed, **options)
    hashes[name], facts[name], extras[name] = grown_hashes(grown), grown_facts(grown), grown_extras(grown)

# --- library members, their index rows and a plan ---------------------------
# seeds on both sides of 2 ** 31, as a library's own seeds are
MEMBERS = (("tree", 3.0, 101), ("tree", 5.0, 3796490668), ("mesh", 3.0, 103), ("mesh", 5.0, 2147483649),
           ("tumour", 3.0, 4026531839), ("tumour", 5.0, 106))
members = []
for family, ratio, seed in MEMBERS:
    name = f"member_{family}_r{ratio:g}"
    result = library.grow_member(family, ratio, seed)
    grown = result["grown"]
    row = library.describe_member(grown, 1.0)
    hashes[name] = dict(grown_hashes(grown),
                        member=record_sha(restrict(result, rules["member"])),
                        kwargs=record_sha(restrict(result["kwargs"], rules["kwargs"])),
                        row=record_sha(restrict(row, rules["index_row"])))
    facts[name] = dict(grown_facts(grown), argv=result["argv"], avoid_collisions=result["kwargs"]["avoid_collisions"])
    extras[name] = dict(grown_extras(grown), kwargs=beyond(result["kwargs"], rules["kwargs"]),
                        row=sorted(beyond(row, rules["index_row"])))
    members.append({"nodes": grown["nodes"], "node_kind": grown["node_kind"], "tree": grown["tree"],
                    "roots": np.array(library.root_columns(grown["tree"]), dtype=np.int64)})

# a plan whose count divides evenly between the families and one that leaves a remainder
for name, seed, count in (("plan", 5, 9), ("plan_uneven", 7, 10)):
    plan = library.plan_library(seed, count, ("tree", "mesh", "tumour"), (1, 1, 1), (2.52, 25.0))
    hashes[name] = {"members": record_sha([restrict(member, rules["plan"]) for member in plan]),
                    "stage_generators": stage_generators()}
    facts[name] = {"members": len(plan), "families": [member["family"] for member in plan]}
    extras[name] = {"members": sorted({key for member in plan for key in beyond(member, rules["plan"])})}

# --- describe on the fixture archives ---------------------------------------
# (the fixtures are version-1 archives: no tree labels, no events, few flags and
# no growth box; the archives the command line writes below carry all of them)
for archive in lists["archives"]:
    describe_case("describe_" + archive, describe_module.describe_archive(os.path.join(fixtures, archive)))
# at a margin the network was not grown to keep, so that violations are counted
describe_case("describe_margin", describe_module.describe_archive(os.path.join(fixtures, lists["archives"][-1]),
                                                                  margin=1.0))

# --- the command line -------------------------------------------------------
CLI_CASES = (("cli_tree", ["--count", "1", "--seed", "7", "--volume", "48", "48", "24", "--iterations", "4", "6"]),
             ("cli_mesh", ["--family", "mesh", "--count", "1", "--seed", "8", "--volume", "48", "48", "24",
                           "--iterations", "6", "6"]),
             # two networks, so that the second takes seed + 1, under the options the two runs
             # above leave at their defaults; seed 8's first root diameter falls below --d0-min
             ("cli_options", ["--count", "2", "--seed", "7", "--volume", "48", "48", "24", "--iterations", "4", "4",
                              "--fit", "voxel_size", "--voxel-size", "8", "--deterministic", "--d0-min", "22",
                              "--subdivisions", "2", "--no-connect"]))
for name, argv in CLI_CASES:
    directory = os.path.join(out_dir, name)
    with contextlib.redirect_stdout(io.StringIO()):
        status = main.main(argv + ["--out", directory])
    written = sorted(os.listdir(directory))
    hashes[name] = {"status": status, "files": written}
    facts[name], extras[name] = {}, {}
    for stem in sorted(entry[:-4] for entry in written if entry.endswith(".npz")):
        entry, unknown_arrays, metadata = archive_hashes(os.path.join(directory, stem + ".npz"), "sidecar")
        with open(os.path.join(directory, stem + ".json")) as handle:
            sidecar = json.load(handle)
        with open(os.path.join(directory, stem + ".tiff"), "rb") as handle:
            tiff_file = sha(handle.read())
        pixels = tifffile.imread(os.path.join(directory, stem + ".tiff"))
        hashes[name][stem] = dict(entry, sidecar=record_sha(restrict(sidecar, rules["sidecar"])),
                                  tiff=array_sha(pixels), tiff_file=tiff_file)
        facts[name][stem] = {"voxels": int(np.count_nonzero(pixels)), "trees": sidecar["trees"],
                             "bridges": sidecar["bridges"], "d0": sidecar["d0"], "seed": sidecar["seed"]}
        extras[name][stem] = {"sidecar": beyond(sidecar, rules["sidecar"]),
                              "metadata": beyond(metadata, rules["sidecar"]),
                              "events": beyond(sidecar["events"], rules["events"]), "arrays": unknown_arrays}
    hashes[name].update(global_generators=global_generators(), stage_generators=stage_generators())

# --- describe on the archives just written --------------------------------------
for name in ("cli_tree", "cli_mesh"):
    stem = next(entry[:-4] for entry in hashes[name]["files"] if entry.endswith(".npz"))
    describe_case("describe_" + name, describe_module.describe_archive(os.path.join(out_dir, name, stem + ".npz")))

# --- joining a cropped forest of the library members -------------------------
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


def build_forest(networks, count, lo, hi, rng, clearance_gap=2.0, tries=40):
    index = GridIndex(cell_size=20.0)
    placed = []
    for i in range(count):
        network = networks[i % len(networks)]
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


FIELD = 240.0
forest = build_forest(members, 24, np.zeros(3), np.full(3, FIELD), default_rng(21))
lo, hi = np.full(3, 40.0), np.full(3, FIELD - 40.0)
margin = 1.0
r_max = max(float(np.nanmax(network["nodes"][3])) for network in forest) / 2.0
cropped = [join_module.crop_network(network, lo, hi, r_max + margin) for network in forest]
hashes["crop"] = {str(k): {key: array_sha(network[key]) for key in ("nodes", "columns", "node_kind", "tree", "roots")}
                  for k, network in enumerate(cropped)}
facts["crop"] = {"networks": len(cropped), "columns": [int(network["nodes"].shape[1]) for network in cropped],
                 "roots": [int(len(network["roots"])) for network in cropped]}
extras["crop"] = {"returned": sorted({key for network in cropped for key in network}
                                     - {"nodes", "columns", "node_kind", "tree", "roots"})}
JOIN_CASES = (("join_cropped", dict(policy="cross", attach_roots=False)),
              ("join_cropped_roots", dict(policy="cross", attach_roots=True, merged=True)),
              ("join_cropped_any", dict(policy="any")))
for name, settings in JOIN_CASES:
    rng = default_rng(22)
    result = join_module.join_networks(cropped, rng, fraction=0.9, collision_margin=margin, box=(lo, hi),
                                       boundary_margin=2.0, **settings)
    bridges = [dict(restrict(bridge, rules["join_bridge"]), geometry=array_sha(bridge["geometry"]))
               for bridge in result["bridges"]]
    hashes[name] = {"bridges": record_sha(bridges), "bridge_nodes": array_sha(result["bridge_nodes"]),
                    "tips": array_sha(result["tips"]),
                    "events": record_sha(restrict(result["events"], rules["join_events"])),
                    "summary": record_sha(restrict(result["summary"], rules["join_summary"])),
                    "generator": record_sha(rng.bit_generator.state)}
    facts[name] = {"networks": len(cropped), "bridges": len(bridges),
                   "root_bridges": sum(1 for bridge in bridges if bridge["source_kind"] == "root"),
                   "events": {key: value for key, value in restrict(result["events"], rules["join_events"]).items()
                              if value}}
    extras[name] = {"events": beyond(result["events"], rules["join_events"]),
                    "summary": sorted(beyond(result["summary"], rules["join_summary"])),
                    "result": sorted(set(result) - {"bridges", "bridge_nodes", "tips", "events", "summary", "merged"})}
    if "merged" in result:
        merged = result["merged"]
        hashes[name]["merged"] = {key: array_sha(merged[key]) for key in lists["join_merged"]}
        extras[name]["merged"] = sorted(set(merged) - set(lists["join_merged"]))
        # the joined forest is a network of several components with loops between them
        describe_case("describe_forest", describe_module.describe(merged["nodes"], merged["edges"], margin=margin,
                                                                  tree=merged["tree"]))

# --- a small library ---------------------------------------------------------
directory = os.path.join(out_dir, "library")
with contextlib.redirect_stdout(io.StringIO()):
    status = library.main(["--out", directory, "--count", "3", "--seed", "1", "--ratio-range", "2.52", "3.5",
                           "--workers", "1"])
with open(os.path.join(directory, "manifest.json")) as handle:
    content = json.load(handle)["content"]
with open(os.path.join(directory, "index.json")) as handle:
    index = json.load(handle)
with open(os.path.join(directory, "index.csv"), newline="") as handle:
    table = list(csv.reader(handle))
# index.csv restricted to the 3.4 columns, which a later version keeps first and in order
pinned = [k for k, column in enumerate(table[0])
          if column in lists["index_columns"] and column not in lists["unpinned_columns"]]
entry = {"status": status, "files": sorted(os.listdir(directory)),
         "content": record_sha({key: value for key, value in content.items()
                                if key not in lists["unpinned_manifest"]}),
         "index": record_sha(dict(restrict(index, rules["index"]),
                                  networks=[restrict(row, rules["index_row"]) for row in index["networks"]])),
         "index_csv": record_sha([[line[k] for k in pinned] for line in table]),
         "stage_generators": stage_generators()}
extra = {"rows": sorted({key for row in index["networks"] for key in beyond(row, rules["index_row"])}),
         "index": sorted(beyond(index, rules["index"])),
         "columns": table[0][len(lists["index_columns"]):],
         "library_version": content["library_version"], "metadata": {}, "arrays": {}, "events": {}}
for row in index["networks"]:
    entry[row["file"]], extra["arrays"][row["file"]], metadata = archive_hashes(
        os.path.join(directory, row["file"]), "metadata")
    extra["metadata"][row["file"]] = sorted(beyond(metadata, rules["metadata"]))
    extra["events"][row["file"]] = beyond(metadata["events"], rules["events"])
    report = describe_module.describe_archive(os.path.join(directory, row["file"]))
    # from 3.5 the archive's metadata carries d_min at its top level, which the
    # report's flags show; the flags are compared without it so that the rest of
    # the report is pinned against the 3.4 archive, which has no such entry
    report["flags"].pop("d_min", None)
    describe_case("describe_" + row["file"], report)
hashes["library"] = entry
facts["library"] = {"files": [row["file"] for row in index["networks"]], "families": content["families"],
                    "columns": table[0][:len(lists["index_columns"])], "rows": len(table) - 1,
                    "content_keys": sorted(content), "failures": len(content["failures"]),
                    "weights": library.library_weights(index).tolist()}
extras["library"] = extra

print(json.dumps({"hashes": hashes, "facts": facts, "extras": extras}, default=plain))
"""


def start_driver(code_dir, out_dir):
    """Starts the driver on the modules in `code_dir`, writing its files under `out_dir`."""
    os.makedirs(out_dir)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")      # nothing is written beside the modules
    return subprocess.Popen([sys.executable, "-c", DRIVER, code_dir, out_dir, FIXTURES,
                             json.dumps(RULES), json.dumps(LISTS)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=code_dir,
                            env=environment)


def finish_driver(process, code_dir):
    stdout, stderr = process.communicate()
    if process.returncode:
        raise AssertionError(f"{code_dir}: {stderr}")
    return json.loads(stdout)


def run_drivers(scratch):
    """
    What the 3.4 modules and the current ones produce, run side by side:
    (reference, current), each {"hashes", "facts", "extras"} by case. The
    files they write are left under `scratch` in "reference" and "current".
    """
    if os.path.commonpath([os.path.realpath(scratch), os.path.realpath(ROOT)]) == os.path.realpath(ROOT):
        raise AssertionError(f"the scratch directory {scratch} lies inside the repository")
    runs = [(code_dir, start_driver(code_dir, os.path.join(scratch, name)))
            for name, code_dir in (("reference", REFERENCE), ("current", ROOT))]
    return tuple(finish_driver(process, code_dir) for code_dir, process in runs)


def _restrict(value, rule):
    """The 3.4 part of a record, as the driver's restrict takes it."""
    if rule is None or not isinstance(value, dict):
        return value
    if "*" in rule:
        return {key: _restrict(item, rule["*"]) for key, item in value.items()}
    return {key: _restrict(value[key], sub) for key, sub in rule.items() if key in value}


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class PinnedRelease34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.scratch = scratch.name
        cls.reference, cls.current = run_drivers(cls.scratch)
        cls.recordings = {}
        for name in sorted(os.listdir(FIXTURES)):
            if name.startswith(RECORDINGS) and name.endswith(".json"):
                with open(os.path.join(FIXTURES, name)) as handle:
                    cls.recordings[name] = json.load(handle)

    def test_the_reference_directory_is_an_unchanged_copy_of_the_release(self):
        # nothing but the modules: hidden files and a bytecode cache a stray import leaves are not the copy
        present = sorted(name for name in os.listdir(REFERENCE)
                         if not name.startswith(".") and name != "__pycache__")
        self.assertEqual(present, sorted(name + ".py" for name in REFERENCE_MODULES))
        self.assertGreater(len(self.recordings), 0)
        for recording, recorded in self.recordings.items():
            self.assertEqual(sorted(recorded["_reference_files"]), present, recording)
            for name in present:
                with self.subTest(recording=recording, module=name):
                    self.assertEqual(file_sha256(os.path.join(REFERENCE, name)), recorded["_reference_files"][name])

    def test_the_current_modules_produce_what_the_3_4_modules_produce_in_this_environment(self):
        self.assertEqual(sorted(self.current["hashes"]), sorted(self.reference["hashes"]))
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                self.assertEqual(self.current["hashes"][case], self.reference["hashes"][case])
                self.assertEqual(self.current["facts"][case], self.reference["facts"][case])

    def test_the_cases_reach_the_stages_they_pin(self):
        facts = self.reference["facts"]
        events = {case: entry["events"] for case, entry in facts.items() if "events" in entry}
        # one tree and no bridge; the plain tree never collides, the planar one has stems shortened
        for case in ("tree_stems", "tree_stems_avoidance", "tree_walk", "tree_stems_in_volume",
                     "tree_stems_avoidance_planar", "tree_stems_avoidance_in_volume", "tree_default_properties"):
            self.assertEqual(facts[case]["trees"], 1, case)
            self.assertEqual(facts[case]["bridges"], 0, case)
        self.assertEqual(events["tree_stems_avoidance"], {})
        for case in ("tree_stems_avoidance_planar", "member_tree_r3", "member_tree_r5"):
            self.assertGreater(events[case]["collision_truncated_stems"], 0, case)
        # the bounds end branches on the default interpreter, on the shaped one and on the walk
        for case in ("tree_stems_in_volume", "tree_stems_avoidance_in_volume"):
            self.assertGreater(events[case]["bound_terminations"], 0, case)
        for case in ("mesh", "mesh_voxel_size"):
            self.assertGreater(events[case]["walk_bound_redraws"], 0, case)
        self.assertGreater(events["mesh_voxel_size"]["walk_bound_terminations"], 0)
        self.assertGreater(events["mesh_voxel_size"]["root_relocations"], 0)
        # avoidance redraws walk steps and ends branches; anastomosis places bridges, walked or not
        for case in ("tree_walk_avoidance_anastomosis", "tumour", "member_tumour_r5", "tree_walk_subdivisions"):
            self.assertEqual(facts[case]["trees"], 1, case)
            self.assertGreater(events[case]["collision_redraws"], 0, case)
        for case in ("tree_walk_avoidance_anastomosis", "tumour", "member_tumour_r3", "member_tumour_r5",
                     "tree_stems_anastomosis"):
            self.assertGreater(facts[case]["bridges"], 0, case)
        for case in ("mesh", "mesh_voxel_size", "tree_walk_arteriovenous_free", "member_mesh_r3", "member_mesh_r5"):
            self.assertEqual(facts[case]["trees"], 2, case)
            self.assertGreater(events[case]["anastomosis_bridges_arteriovenous"], 0, case)
        # the library's tree grows with avoidance; the plans hold nine and ten networks
        for ratio in ("3", "5"):
            self.assertTrue(facts[f"member_tree_r{ratio}"]["avoid_collisions"])
            self.assertIn("--avoid-collisions", facts[f"member_tree_r{ratio}"]["argv"])
        self.assertEqual(facts["plan"]["families"], ["tree"] * 3 + ["mesh"] * 3 + ["tumour"] * 3)
        self.assertEqual(facts["plan_uneven"]["families"], ["tree"] * 4 + ["mesh"] * 3 + ["tumour"] * 3)
        # the command line: one network each, then two whose second takes the next seed
        (tree,), (mesh,) = facts["cli_tree"].values(), facts["cli_mesh"].values()
        self.assertEqual((tree["trees"], tree["seed"], mesh["trees"], mesh["seed"]), (1, 7, 2, 8))
        self.assertGreater(mesh["bridges"], 0)
        options = facts["cli_options"]
        self.assertEqual(sorted(entry["seed"] for entry in options.values()), [7, 8])
        for case in ("cli_tree", "cli_mesh", "cli_options"):
            for stem, entry in facts[case].items():
                self.assertGreater(entry["voxels"], 0, stem)
                self.assertIn(f"_s{entry['seed']}", stem)
        # describe meets tree labels, events, every flag, a growth box, violations and several components
        for case in ("describe_cli_tree", "describe_cli_mesh"):
            self.assertEqual(facts[case]["events"], len(EVENT_KEYS_3_4), case)
            self.assertEqual(len(facts[case]["flags"]), len(FLAG_KEYS_3_4), case)
        self.assertEqual(facts["describe_cli_mesh"]["trees"], ["0", "1", "2"])
        self.assertEqual(facts["describe_cli_mesh"]["volume_source"], "growth_box_um")
        self.assertGreater(facts["describe_cli_mesh"]["cycles"], 0)
        self.assertGreater(facts["describe_margin"]["violations"], 0)
        self.assertGreater(facts["describe_forest"]["components"], 1)
        self.assertGreater(facts["describe_forest"]["cycles"], 0)
        # joining: cut ends, tip bridges, root bridges only when asked for, kin skipped under "any"
        self.assertGreater(facts["crop"]["networks"], 10)
        self.assertGreater(sum(facts["crop"]["roots"]), 0)
        self.assertGreater(facts["join_cropped"]["bridges"], 1)
        self.assertEqual(facts["join_cropped"]["root_bridges"], 0)
        self.assertGreater(facts["join_cropped"]["events"]["join_cut_ends"], 0)
        self.assertGreater(facts["join_cropped_roots"]["root_bridges"], 0)
        self.assertGreater(facts["join_cropped_any"]["events"]["join_kin_skipped"], 0)
        # the library: three networks of the default families and their index
        self.assertEqual(facts["library"]["files"],
                         ["net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        self.assertEqual(facts["library"]["columns"], list(INDEX_COLUMNS_3_4))
        self.assertEqual((facts["library"]["rows"], facts["library"]["failures"]), (3, 0))

    def test_with_the_later_options_off_the_frame_is_none_and_the_new_kwargs_are_none(self):
        # what 3.5 appends while its options are off: a frame of kind "none" in
        # every return, sidecar and archive, and guidance and root offsets of None
        extras = self.current["extras"]
        grown = [case for case in extras if isinstance(extras[case].get("returned"), dict)]
        self.assertGreater(len(grown), 10)
        for case in grown:
            with self.subTest(case=case):
                returned = extras[case]["returned"]
                self.assertEqual(set(returned), {"frame"})
                self.assertEqual(returned["frame"]["kind"], "none")
                self.assertEqual(returned["frame"]["rules"], [])
                if "kwargs" in extras[case]:
                    self.assertEqual(extras[case]["kwargs"], {"guidance": None, "root_offsets": None})
        for case in ("cli_tree", "cli_mesh", "cli_options"):
            for stem, entry in extras[case].items():
                with self.subTest(case=case, stem=stem):
                    for record in (entry["sidecar"], entry["metadata"]):
                        self.assertEqual(set(record), {"guidance", "root_offsets", "frame"})
                        self.assertIsNone(record["guidance"])
                        self.assertIsNone(record["root_offsets"])
                        self.assertEqual(record["frame"]["kind"], "none")
                    self.assertEqual(entry["arrays"], [])
        for name, keys in extras["library"]["metadata"].items():
            with self.subTest(archive=name):
                self.assertEqual(keys, ["d_min", "frame", "peak_rss_mb", "seconds"])
        self.assertEqual(extras["library"]["arrays"], {name: [] for name in extras["library"]["arrays"]})
        self.assertEqual(extras["library"]["index"], [])
        self.assertEqual(extras["library"]["library_version"], "3.5.0")

    def test_a_3_4_library_loads_and_describes_as_it_did_with_the_new_keys_inert(self):
        # the library the 3.4 modules wrote in the reference run, read by the current modules
        import numpy as np
        import library
        from describe import describe_archive
        directory = os.path.join(self.scratch, "reference", "library")
        index = library.load_index(os.path.join(directory, "index.json"))
        # the weights the 3.4 modules gave this index in the reference run, in this environment
        self.assertEqual(library.library_weights(index).tolist(), self.reference["facts"]["library"]["weights"])
        ratio = np.array([row["ratio"] for row in index["networks"]])
        np.testing.assert_allclose(library.library_weights(index), ratio ** -3.0 / np.sum(ratio ** -3.0),
                                   rtol=1e-15)
        for row in index["networks"]:
            with self.subTest(archive=row["file"]):
                report = describe_archive(os.path.join(directory, row["file"]))
                # every 3.4 key as the 3.4 modules reported it on the same archive
                self.assertEqual(hashlib.sha256(json.dumps(_restrict(report, RULES["describe"]), sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest(),
                                 self.reference["hashes"]["describe_" + row["file"]]["report"])
                self.assertIsNone(report["frame"])
                for key in ("frame_orientation", "polar_order", "transverse_spacing"):
                    self.assertIsNone(report[key], key)
                self.assertEqual(report["classes"], {"bound": 2.0, "d_ref": 1.0, "d_ref_source": "grow_kwargs"})
                for key in ("calibre_shares", "segments_by_class", "orientation_by_class"):
                    self.assertIsNotNone(report[key], key)
                self.assertGreater(report["calibre_shares"]["length"]["below_2"], 0.0)

    def test_with_the_later_options_off_every_appended_counter_is_zero(self):
        def appended(value):
            # the "events" entries of a case's extras, at any depth
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "events" and all(not isinstance(count, dict) for count in item.values()):
                        yield from item.items()
                    else:
                        yield from appended(item)

        for case, extras in self.current["extras"].items():
            with self.subTest(case=case):
                for key, count in appended(extras):
                    self.assertEqual(count, 0, key)

    def test_the_recorded_3_4_hashes_are_matched(self):
        # a case passes when it matches a recording; a recording that the 3.4
        # modules themselves reproduce here must be matched; one they do not
        # reproduce was made where rounding differs, and is skipped
        for recording, recorded in self.recordings.items():
            hashes = {key: value for key, value in recorded.items() if not key.startswith("_")}
            self.assertEqual(sorted(hashes), sorted(self.reference["hashes"]), recording)
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                recorded = [entry[case] for entry in self.recordings.values()]
                if self.current["hashes"][case] in recorded:
                    continue
                if self.reference["hashes"][case] not in recorded:
                    self.skipTest("this environment's rounding differs from every recording machine's; "
                                  "the same-environment comparison above still holds")
                self.assertIn(self.current["hashes"][case], recorded)


if __name__ == "__main__":
    unittest.main()
