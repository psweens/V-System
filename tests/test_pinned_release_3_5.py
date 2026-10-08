"""
Pins what 3.5 draws, writes, measures, places and joins.

The 3.5 modules (tests/fixtures/reference_code_3_5: main's import closure as
released, guidance included, with describe, join, library and frames) and the
current ones each run one driver in a subprocess, in the same environment,
and the sha256 of everything the driver produces are compared. The 3.4 pin
(tests/test_pinned_release_3_4.py) covers what 3.4 drew in full; this one
keeps a few of its cases, so that what 3.5 added to unguided growth is pinned
on each kind of growth, and adds the cases 3.5 opened:

    grow_network   a stem tree, a walked tree with avoidance and anastomosis,
                   the mesh and tumour presets and a tree in the volume; the
                   aligned preset with its released kwargs written out
                   literally; a nematic axis, a polar axis whose sense follows
                   the root on a free pair, calibre classes with a fixed polar
                   axis, an unguided class and a banked plane, a partner rule
                   between offset roots, and root offsets without guidance;
    library        grow_member for tree, mesh, tumour and aligned, with argv,
                   kwargs and index row, and plans with and without aligned;
    describe       a fixture archive at a margin, an aligned member against
                   its frame at two class bounds, a banked plane against its
                   frame, a moved member, and the archives below;
    frames         rotations, align and the transforms on an aligned member;
    main.main      a tree run, an aligned run and a run with --guidance and
                   --root-offsets: archive, sidecar, TIFF pixels and file;
    join_networks  a cropped forest of the library members, aligned among
                   them, joined with root attachment into a merged archive;
    library.main   the default three-network library, and one of tree and
                   aligned with a box constant of its own.

Every record (events, sidecars, metadata, kwargs, frame records and their
rules, describe output, index rows, join output) is first restricted,
recursively, to the keys 3.5 wrote, which are listed literally below, so that
what a later version appends never enters the comparison while everything 3.5
wrote must still be there, unchanged; a list of dicts, such as a frame's
rules, is restricted entry by entry. The global generators' states after each
case, the states of the generators each stage created and the joining
generator's state are compared too, so a stage added later must create no
generator while its option is off, and every counter appended to the events
must be zero.

Running both in one environment is the only portable comparison: the
B-spline, the walk and the collision checks go through kernels whose last
bits differ between machines. Hashes recorded on a machine
(tests/fixtures/reference_hashes_3_5*.json, one file per recording) are
checked as well: a case passes when it matches a recording, and the check is
skipped only when the reference code itself reproduces none of them here.
The reference directory is an unchanged copy of the release: the sha256 of
each of its files is recorded with the hashes and checked, and the drivers
write nothing but to a temporary directory outside the repository. The two
drivers run side by side, each in its own process and output directory.

Run from the repository root with
    python -m unittest tests.test_pinned_release_3_5
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
REFERENCE = os.path.join(FIXTURES, "reference_code_3_5")
RECORDINGS = "reference_hashes_3_5"           # tests/fixtures/reference_hashes_3_5*.json, one per recording

REFERENCE_MODULES = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance",
                     "describe", "join", "library", "frames")
ARCHIVES = ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz", "Lnet_i9_s3.npz")

# ---------------------------------------------------------------------------
# The keys 3.5 wrote, listed literally so that a later version's additions
# never enter the comparison.
# ---------------------------------------------------------------------------
EVENT_KEYS_3_5 = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
                  "collision_redraws", "collision_terminations", "collision_truncated_stems",
                  "root_relocations", "root_collisions",
                  "anastomosis_tips", "anastomosis_tips_inside_junction", "anastomosis_kin_skipped",
                  "anastomosis_behind_skipped", "anastomosis_selected", "anastomosis_bridges",
                  "anastomosis_bridges_arteriovenous", "anastomosis_bridges_same_tree",
                  "anastomosis_partner_tips", "anastomosis_partner_interior",
                  "anastomosis_source_consumed", "anastomosis_no_partner",
                  "anastomosis_collision_failed", "anastomosis_bridge_redraws",
                  "guided_steps", "guidance_onset_steps", "unguided_steps", "guidance_undefined_steps",
                  "guidance_sense_flips", "guidance_bank_steps")
SHAPING_KEYS_3_5 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                    "collision_attempts", "collision_index", "anastomose", "anastomosis_radius",
                    "anastomosis_fraction", "anastomose_mode", "anastomosis_min_separation",
                    "guidance", "root_offsets")
KWARGS_KEYS_3_5 = ("tVol", "fit", "clip_axes", "voxel_size", "subdivisions", "d_min",
                   "grow_in_volume", "seed") + SHAPING_KEYS_3_5
PROPERTY_KEYS_3_5 = ("k", "epsilon", "randmarg", "sigma", "stochparams", "roll_angle", "stem_angle",
                     "aneurysm_prob", "stenosis_prob")
# what grow_network returned beside its arrays, programs, bridges and events
GROWN_KEYS_3_5 = ("program", "growth_box_um", "root_positions_um", "clip_axes", "collision_index", "frame")
# the frame record and the keys of a rule of any field, as given or normalised
FRAME_KEYS_3_5 = ("frame_version", "kind", "axis", "sense", "tree_senses", "normal", "origin",
                  "grow_direction", "grow_perpendicular", "rules")
RULE_KEYS_3_5 = ("below", "field", "axis", "normal", "sense", "polarity", "length", "onset", "bank")
SIDECAR_KEYS_3_5 = ("seed", "iterations", "d0", "d_min", "volume", "axis_order", "units", "fit",
                    "clip_axes", "connect", "grow_in_volume", "voxel_size", "subdivisions",
                    "archive_version", "family", "trees", "growth_box_um", "root_positions_um",
                    "bridges", "bridge_persistence", "frame") + SHAPING_KEYS_3_5
RNG_STREAMS_3_5 = ("walk", "anastomosis")
MEMBER_KEYS_3_5 = ("argv", "niter", "d0")
PLAN_KEYS_3_5 = ("id", "family", "family_index", "position", "seed", "ratio", "bin", "u", "file")
# index.csv's columns; a run's seconds and peak memory are left out of the comparison
INDEX_COLUMNS_3_5 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points",
                     "polylines", "tips", "junctions", "cycles", "components", "total_length",
                     "diameter_p50", "diameter_p90", "diameter_p99",
                     "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min", "diameter_max",
                     "min_point_diameter", "clearance_min", "clearance_violations", "bridges", "events",
                     "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                     "nodes_sha256", "program_sha256",
                     "frame_kind", "capillary_order", "larger_order", "capillary_polar_order",
                     "capillary_length_share", "capillary_volume_share", "capillary_segment_median",
                     "transverse_spacing_median")
UNPINNED_COLUMNS = ("seconds", "peak_rss_mb")
METADATA_KEYS_3_5 = ("units", "library_id", "family", "ratio", "bin", "u", "seed", "ratio_law",
                     "ratio_range", "argv", "niter", "generations", "d0", "avoid_collisions",
                     "root_columns", "growth_box", "root_positions", "bridges", "min_point_diameter",
                     "programs", "archive_version", "nodes_sha256", "program_sha256", "d_min", "frame")
UNPINNED_MANIFEST_KEYS = ("code_sha256", "library_version")
FLAG_KEYS_3_5 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                 "anastomose", "anastomosis_fraction", "anastomosis_radius", "anastomose_mode",
                 "family", "seed", "iterations", "d0", "d_min")
DESCRIBE_KEYS_3_5 = ("units", "points", "polylines", "vertices", "edges", "components", "cycles",
                     "total_length_mm", "degree_histogram", "length_density_mm_per_mm3", "volume_mm3",
                     "volume_source")
ORIENTATION_KEYS_3_5 = ("eigenvalues", "principal_axis", "fractional_anisotropy", "polar_histogram_deg",
                        "azimuth_histogram_deg")
DESCRIBE_GROUPS_3_5 = {
    "diameter_um": ("p50", "p90", "p99", "min", "max"),
    "arc_chord": ("mean", "median", "p90", "max", "n_segments", "n_closed"),
    "curvature_per_um": ("mean_abs", "n_pairs"),
    "tips": ("count", "per_mm"),
    "orientation": ORIENTATION_KEYS_3_5,
    "bounding_box_um": ("min", "max"),
    "clearance_um": ("min", "min_columns", "min_is_lower_bound", "search_reach_um", "violations",
                     "pairs_checked", "margin"),
    "classes": ("bound", "d_ref", "d_ref_source"),
    "polar_order": ("all", "capillary", "larger"),
    "transverse_spacing": ("planes", "count", "median", "mean", "p10", "p90", "median_d", "mean_d", "p10_d",
                           "p90_d", "crossing_density", "crossing_density_d", "area_source"),
}
CLASSES_3_5 = ("all", "capillary", "larger")
# an entry of frame_orientation: about an axis, or about a plane
FRAME_ENTRY_KEYS_3_5 = ("S", "mean_abs_cos", "crossing_ratio", "mean_angle_deg", "within_20_deg", "within_45_deg",
                        "watson_K", "fisher_axial_K", "in_plane_fraction", "S_n", "length", "length_d")
SEGMENT_KEYS_3_5 = ("count", "median", "p10", "p90", "cv", "median_d", "p10_d", "p90_d")
SHARE_KEYS_3_5 = ("below_1.5", "below_2", "below_3")
PER_TREE_KEYS_3_5 = ("points", "polylines", "tips", "length_mm")
JOIN_EVENT_KEYS_3_5 = ("join_networks", "join_points", "join_isolated", "join_tips",
                       "join_roots", "join_cut_ends", "join_stubs", "join_eligible", "join_selected",
                       "join_not_selected", "join_bridges", "join_bridged_partners", "join_no_partner",
                       "join_collision_failed", "join_over_budget",
                       "join_partner_tips", "join_partner_interior", "join_source_consumed",
                       "join_redraws", "join_box_redraws", "join_kin_skipped", "join_behind_skipped",
                       "join_components_joined",
                       "join_root_eligible", "join_root_selected", "join_root_attached",
                       "join_root_not_selected", "join_root_no_partner", "join_root_collision_failed",
                       "join_root_over_budget")
JOIN_SUMMARY_KEYS_3_5 = ("components", "length", "free_tips", "free_tips_per_length", "bridge_volume",
                         "blunt_roots", "blunt_roots_per_length", "free_ends", "free_ends_per_length")
JOIN_SUMMARY_ENTRY_KEYS_3_5 = ("before", "after")        # of every summary entry but bridge_volume, a number
JOIN_MERGED_KEYS_3_5 = ("nodes", "edges", "node_kind", "network", "tree")
ARCHIVE_ARRAYS_3_5 = ("nodes", "program", "metadata", "edges", "node_kind", "tree")
INDEX_KEYS_3_5 = ("units", "ratio_law", "networks")
JOIN_BRIDGE_FIELDS_3_5 = ("tip", "partner", "source_kind", "partner_kind", "chord", "arc", "diameter",
                          "volume", "redraws")
# the presets as 3.5.0 released them, passed literally so that they stay pinned
MESH_3_5 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
            "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.5, "grow_in_volume": True}
TUMOUR_3_5 = {"tortuosity": "walk", "persistence": 3.0, "avoid_collisions": True, "anastomose": True,
              "anastomose_mode": "any", "anastomosis_fraction": 0.8, "d0": [20.0, 10.0], "aneurysm_prob": 0.1,
              "stenosis_prob": 0.1}
ALIGNED_3_5 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
               "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
               "root_offsets": [[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
               "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                             "polarity": "partner", "length": 4.9, "onset": 2.0},
                            {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0}]}


def _whole(keys):
    return {key: None for key in keys}


# How each record is restricted before it is hashed. None keeps a value whole,
# a dict keeps the listed keys and restricts each value in turn, {"*": rule}
# keeps every key (a label or an id, not a field) and restricts each value,
# and {"[]": rule} restricts every entry of a list.
RULE_RULE = {"[]": _whole(RULE_KEYS_3_5)}
FRAME_RULE = dict(_whole(FRAME_KEYS_3_5), rules=RULE_RULE)
SHAPING_RULE = dict(_whole(SHAPING_KEYS_3_5), guidance=RULE_RULE)
KWARGS_RULE = dict(_whole(KWARGS_KEYS_3_5), guidance=RULE_RULE)
RULES = {
    "events": _whole(EVENT_KEYS_3_5),
    "grown": dict(_whole(GROWN_KEYS_3_5), frame=FRAME_RULE),
    "kwargs": KWARGS_RULE,
    "frame": FRAME_RULE,
    "member": dict(_whole(MEMBER_KEYS_3_5), properties=_whole(PROPERTY_KEYS_3_5)),
    "plan": _whole(PLAN_KEYS_3_5),
    "sidecar": dict(_whole(SIDECAR_KEYS_3_5), properties=_whole(PROPERTY_KEYS_3_5),
                    rng_streams=_whole(RNG_STREAMS_3_5), events=_whole(EVENT_KEYS_3_5), frame=FRAME_RULE,
                    guidance=RULE_RULE),
    "index_row": dict(_whole(column for column in INDEX_COLUMNS_3_5 if column not in UNPINNED_COLUMNS),
                      events=_whole(EVENT_KEYS_3_5)),
    "metadata": dict(_whole(METADATA_KEYS_3_5), properties=_whole(PROPERTY_KEYS_3_5),
                     grow_kwargs=KWARGS_RULE, events=_whole(EVENT_KEYS_3_5), frame=FRAME_RULE),
    "describe": dict(_whole(DESCRIBE_KEYS_3_5), events=_whole(EVENT_KEYS_3_5), flags=_whole(FLAG_KEYS_3_5),
                     per_tree={"*": _whole(PER_TREE_KEYS_3_5)}, frame=FRAME_RULE,
                     frame_orientation=dict({cls: _whole(FRAME_ENTRY_KEYS_3_5) for cls in CLASSES_3_5}, kind=None),
                     orientation_by_class={cls: _whole(ORIENTATION_KEYS_3_5 + ("S_max", "planarity"))
                                           for cls in CLASSES_3_5},
                     calibre_shares={"length": _whole(SHARE_KEYS_3_5), "volume": _whole(SHARE_KEYS_3_5)},
                     segments_by_class={cls: _whole(SEGMENT_KEYS_3_5) for cls in ("capillary", "larger")},
                     **{group: _whole(keys) for group, keys in DESCRIBE_GROUPS_3_5.items()}),
    "join_events": _whole(JOIN_EVENT_KEYS_3_5),
    "join_summary": dict({key: _whole(JOIN_SUMMARY_ENTRY_KEYS_3_5) for key in JOIN_SUMMARY_KEYS_3_5},
                         bridge_volume=None),
    "join_bridge": _whole(JOIN_BRIDGE_FIELDS_3_5),
    "index": _whole(INDEX_KEYS_3_5),
}

# Plain lists the driver reads beside the rules.
LISTS = {
    "modules": list(REFERENCE_MODULES),
    "archives": list(ARCHIVES),
    "archive_arrays": list(ARCHIVE_ARRAYS_3_5),
    "join_merged": list(JOIN_MERGED_KEYS_3_5),
    "index_columns": list(INDEX_COLUMNS_3_5),
    "unpinned_columns": list(UNPINNED_COLUMNS),
    "unpinned_manifest": list(UNPINNED_MANIFEST_KEYS),
}
SETTINGS = {"mesh": MESH_3_5, "tumour": TUMOUR_3_5, "aligned": ALIGNED_3_5}

# Runs in a subprocess with the modules of one code directory first on the
# path and that directory as the working directory, and prints, per case,
# the hashes of what those modules produce ("hashes"), a few plain counts of
# the same output ("facts") and whatever the output holds beyond the 3.5 keys
# ("extras"). It uses nothing but what 3.5 offers.
DRIVER = r"""
import contextlib, csv, hashlib, importlib, io, json, os, random, sys
code_dir, out_dir, fixtures = sys.argv[1], sys.argv[2], sys.argv[3]
rules, lists, settings = json.loads(sys.argv[4]), json.loads(sys.argv[5]), json.loads(sys.argv[6])
sys.path.insert(0, code_dir)
import numpy as np
import tifffile
import describe as describe_module
import frames
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
    # the 3.5 part of a record; see RULES in the test
    if rule is None:
        return value
    if "[]" in rule:
        return [restrict(item, rule["[]"]) for item in value] if isinstance(value, list) else value
    if not isinstance(value, dict):
        return value
    if "*" in rule:
        return {key: restrict(item, rule["*"]) for key, item in value.items()}
    return {key: restrict(value[key], sub) for key, sub in rule.items() if key in value}


def beyond(value, rule):
    # what a record holds beyond its 3.5 keys, one level deep
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
    # the arrays 3.5 stored, as stored (read with numpy, not with the loader under test)
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
    # a family's options as tests/test_pinned_geometry.py passes them, as released (written out in the test)
    properties = dict(PROPERTIES)
    options = json.loads(json.dumps(settings[family]))
    for key in ("aneurysm_prob", "stenosis_prob"):
        if key in options:
            properties[key] = options.pop(key)
    options.pop("d0", None)
    return properties, options


GROW_CASES = (
    # a few of the 3.4 pin's cases, which that pin covers in full, so that the
    # records 3.5 added to unguided growth (a frame of kind "none", guidance and
    # root offsets of None) are pinned on each kind of growth
    ("tree_stems", 11, 6, PROPERTIES, {}),
    ("tree_walk_avoidance_anastomosis", 14, 6, PROPERTIES,
     dict(WALK, avoid_collisions=True, anastomose=True, anastomose_mode="any", anastomosis_min_separation=2)),
    ("mesh", 3, 6) + preset("mesh"),
    ("tumour", 1, 6) + preset("tumour"),
    ("tree_stems_in_volume", 16, 6, PROPERTIES, {"grow_in_volume": True, "d_min": 4.0}),
    # what 3.5 added: the aligned preset as released, grown in the volume with d_min
    ("aligned", 5, 6, preset("aligned")[0], dict(preset("aligned")[1], d_min=4.0)),
    # a nematic axis on a walked tree with avoidance, whose nearer end flips
    ("guided_nematic", 17, 6, PROPERTIES,
     dict(WALK, avoid_collisions=True,
          guidance=[{"below": None, "field": "axis", "axis": [1, 0, 0], "length": 2.0, "onset": 0.0}])),
    # a polar axis whose sense follows each root's heading, on a free arteriovenous pair
    ("guided_root_pair", 18, 5, PROPERTIES,
     dict(WALK, anastomose=True, anastomose_mode="arteriovenous",
          guidance=[{"below": None, "field": "axis", "axis": [0, 1, 0], "sense": "polar", "polarity": "root",
                     "length": 3.0}])),
    # calibre classes: a fixed polar axis below 2 d_min, an unguided class below 4 and a banked plane above
    ("guided_classes", 19, 6, dict(PROPERTIES, roll_angle=70.0),
     dict(WALK, d_min=1.0,
          guidance=[{"below": 2.0, "field": "axis", "axis": [0, 0, 1], "sense": "polar", "polarity": "fixed",
                     "length": 2.0, "onset": 1.0},
                    {"below": 4.0, "field": None},
                    {"below": None, "field": "plane", "normal": [0, 0, 1], "length": 3.0, "bank": True}])),
    # a partner rule between roots offset along x, in the volume
    ("guided_partner", 20, 5, PROPERTIES,
     dict(WALK, anastomose=True, anastomose_mode="arteriovenous", grow_in_volume=True, d_min=2.0,
          root_offsets=[[-3.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
          guidance=[{"below": None, "field": "axis", "axis": [1, 0, 0], "sense": "polar", "polarity": "partner",
                     "length": 3.0, "onset": 1.0}])),
    # root offsets without guidance, on the default interpreter
    ("root_offsets_stems", 16, 6, PROPERTIES, {"grow_in_volume": True, "d_min": 4.0, "root_offsets": [[2.0, 0.0, 1.0]]}),
)
for name, seed, niter, properties, options in GROW_CASES:
    random.seed(seed)
    np.random.seed(seed)
    grown = main.grow_network(niter, 20.0, dict(properties), VOLUME, seed=seed, **options)
    hashes[name], facts[name], extras[name] = grown_hashes(grown), grown_facts(grown), grown_extras(grown)

# --- library members, their index rows and a plan ---------------------------
# seeds on both sides of 2 ** 31, as a library's own seeds are
MEMBERS = (("tree", 3.0, 101), ("mesh", 3.0, 2147483649), ("tumour", 3.0, 4026531839), ("aligned", 3.0, 107),
           ("aligned", 4.0, 3221225473))
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
                    "roots": np.array(library.root_columns(grown["tree"]), dtype=np.int64),
                    "edges": grown["edges"], "frame": grown["frame"], "family": family})

# a plan whose count divides evenly between the families and one that leaves a remainder
PLANS = (("plan", 5, 9, ("tree", "mesh", "tumour"), (1, 1, 1)),
         ("plan_aligned", 9, 11, ("tree", "mesh", "tumour", "aligned"), (1, 1, 1, 2)))
for name, seed, count, families, shares in PLANS:
    plan = library.plan_library(seed, count, families, shares, (2.52, 25.0))
    hashes[name] = {"members": record_sha([restrict(member, rules["plan"]) for member in plan]),
                    "stage_generators": stage_generators()}
    facts[name] = {"members": len(plan), "families": [member["family"] for member in plan]}
    extras[name] = {"members": sorted({key for member in plan for key in beyond(member, rules["plan"])})}

# --- describe on the fixture archives ---------------------------------------
# (the fixtures are version-1 archives: no tree labels, no events, few flags,
# no growth box and no frame; the archives the command line writes below carry all of them)
# (the 3.4 pin describes all three; one, at a margin it was not grown to keep, so that violations are counted)
describe_case("describe_margin", describe_module.describe_archive(os.path.join(fixtures, lists["archives"][-1]),
                                                                  margin=1.0))
# what 3.5 added: an aligned member against its frame, at the default class bound and another
aligned_member = next(member for member in members if member["family"] == "aligned")
for name, bound in (("describe_aligned_member", 2.0), ("describe_aligned_member_bound", 3.0)):
    describe_case(name, describe_module.describe(aligned_member["nodes"], aligned_member["edges"],
                                                 tree=aligned_member["tree"], frame=aligned_member["frame"],
                                                 d_ref=1.0, class_bound=bound))
# a banked plane against its frame, with d_ref from the argument
random.seed(19)
np.random.seed(19)
banked = main.grow_network(6, 20.0, dict(PROPERTIES, roll_angle=70.0), VOLUME, seed=19, d_min=1.0,
                           guidance=[{"below": None, "field": "plane", "normal": [0, 0, 1], "length": 2.0,
                                      "bank": True}], **WALK)
stage_generators()
describe_case("describe_banked_plane", describe_module.describe(banked["nodes"], banked["edges"], tree=banked["tree"],
                                                                frame=banked["frame"], d_ref=1.0))

# --- frames: rotations, align and the transforms on an aligned member ----------
frame = aligned_member["frame"]
rotations = {"about": frames.rotation_about([1.0, 2.0, 2.0], 0.7),
             "between": frames.rotation_between([0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
             "between_antiparallel": frames.rotation_between([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]),
             "between_nematic": frames.rotation_between([0.0, 1.0, 0.0], [0.2, -1.0, 0.1], nematic=True),
             "of_frames": frames.rotation_of_frames([0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]),
             "random": frames.random_rotation(default_rng(31)),
             "align_axis": frames.align(frame, axis=[0.0, 0.0, 1.0], spin=0.4),
             "align_normal": frames.align(banked["frame"], normal=[1.0, 1.0, 0.0], spin=1.1)}
moved = frames.transform_nodes(aligned_member["nodes"], rotations["align_axis"], t=(5.0, -2.0, 1.0), scale=2.0)
moved_frame = frames.transform_frame(frame, rotations["align_axis"], t=(5.0, -2.0, 1.0), scale=2.0)
hashes["frames"] = {"rotations": {key: array_sha(value) for key, value in rotations.items()},
                    "nodes": array_sha(moved), "frame": record_sha(restrict(moved_frame, rules["frame"]))}
facts["frames"] = {"rotations": sorted(rotations), "columns": int(moved.shape[1]), "kind": moved_frame["kind"]}
extras["frames"] = {"frame": beyond(moved_frame, rules["frame"])}
describe_case("describe_moved_member", describe_module.describe(moved, aligned_member["edges"],
                                                                tree=aligned_member["tree"], frame=moved_frame,
                                                                d_ref=2.0))

# --- the command line -------------------------------------------------------
CLI_CASES = (("cli_tree", ["--count", "1", "--seed", "7", "--volume", "48", "48", "24", "--iterations", "4", "6"]),
             # what 3.5 added: the aligned family, and guidance and root offsets given inline
             ("cli_aligned", ["--family", "aligned", "--d-min", "2", "--count", "1", "--seed", "9", "--volume", "48",
                              "48", "24", "--iterations", "6", "6"]),
             ("cli_guidance", ["--count", "1", "--seed", "10", "--volume", "48", "48", "24", "--iterations", "5", "5",
                               "--tortuosity", "walk", "--persistence", "8", "--grow-in-volume", "--d-min", "2",
                               "--guidance", json.dumps([{"below": 3.0, "field": "axis", "axis": [0, 0, 1],
                                                          "length": 2.0},
                                                         {"below": None, "field": "plane", "normal": [1, 0, 0],
                                                          "length": 4.0, "bank": True}]),
                               "--root-offsets", "[[1.5, 0.0, -2.0]]"]))
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
                              "rng_streams": beyond(sidecar["rng_streams"], rules["sidecar"]["rng_streams"]),
                              "events": beyond(sidecar["events"], rules["events"]), "arrays": unknown_arrays}
    hashes[name].update(global_generators=global_generators(), stage_generators=stage_generators())

# --- describe on the archives just written --------------------------------------
for name in ("cli_tree", "cli_aligned", "cli_guidance"):
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
forest = build_forest([{key: member[key] for key in ("nodes", "node_kind", "tree", "roots")} for member in members],
                      12, np.zeros(3), np.full(3, FIELD), default_rng(21))
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
JOIN_CASES = (("join_cropped_roots", dict(policy="cross", attach_roots=True, merged=True)),)
for name, join_options in JOIN_CASES:
    rng = default_rng(22)
    result = join_module.join_networks(cropped, rng, fraction=0.9, collision_margin=margin, box=(lo, hi),
                                       boundary_margin=2.0, **join_options)
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

# --- small libraries ----------------------------------------------------------
LIBRARIES = (("library", ["--count", "3", "--seed", "1", "--ratio-range", "2.52", "3.5", "--workers", "1"]),
             # what 3.5 added: aligned grown in a cube of its own constant beside a tree
             ("library_aligned", ["--count", "2", "--seed", "2", "--families", "tree", "aligned",
                                  "--box-c", "aligned", "12", "--ratio-range", "2.52", "3.5", "--workers", "1"]))
for case, argv in LIBRARIES:
    directory = os.path.join(out_dir, case)
    with contextlib.redirect_stdout(io.StringIO()):
        status = library.main(["--out", directory] + argv)
    with open(os.path.join(directory, "manifest.json")) as handle:
        content = json.load(handle)["content"]
    with open(os.path.join(directory, "index.json")) as handle:
        index = json.load(handle)
    with open(os.path.join(directory, "index.csv"), newline="") as handle:
        table = list(csv.reader(handle))
    # index.csv restricted to the 3.5 columns, which a later version keeps first and in order
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
             "manifest": sorted(set(content) - {"code_sha256", "library_version"}),
             "row_values": [{key: value for key, value in beyond(row, rules["index_row"]).items()
                             if key not in lists["unpinned_columns"]} for row in index["networks"]],
             "library_version": content["library_version"], "metadata": {}, "metadata_values": {}, "arrays": {},
             "events": {}}
    for row in index["networks"]:
        entry[row["file"]], extra["arrays"][row["file"]], metadata = archive_hashes(
            os.path.join(directory, row["file"]), "metadata")
        extra["metadata"][row["file"]] = sorted(beyond(metadata, rules["metadata"]))
        extra["metadata_values"][row["file"]] = {key: value for key, value in beyond(metadata, rules["metadata"]).items()
                                                 if key not in ("seconds", "peak_rss_mb", "events")}
        extra["events"][row["file"]] = beyond(metadata["events"], rules["events"])
        describe_case(f"describe_{case}_" + row["file"], describe_module.describe_archive(
            os.path.join(directory, row["file"])))
    hashes[case] = entry
    facts[case] = {"files": [row["file"] for row in index["networks"]], "families": content["families"],
                   "columns": table[0][:len(lists["index_columns"])], "rows": len(table) - 1,
                   "growth": content["growth"], "failures": len(content["failures"]),
                   "weights": library.library_weights(index).tolist()}
    extras[case] = extra

print(json.dumps({"hashes": hashes, "facts": facts, "extras": extras}, default=plain))
"""


def start_driver(code_dir, out_dir):
    """Starts the driver on the modules in `code_dir`, writing its files under `out_dir`."""
    os.makedirs(out_dir)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")      # nothing is written beside the modules
    return subprocess.Popen([sys.executable, "-c", DRIVER, code_dir, out_dir, FIXTURES,
                             json.dumps(RULES), json.dumps(LISTS), json.dumps(SETTINGS)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=code_dir,
                            env=environment)


def finish_driver(process, code_dir):
    stdout, stderr = process.communicate()
    if process.returncode:
        raise AssertionError(f"{code_dir}: {stderr}")
    return json.loads(stdout)


def run_drivers(scratch):
    """
    What the 3.5 modules and the current ones produce, run side by side:
    (reference, current), each {"hashes", "facts", "extras"} by case. The
    files they write are left under `scratch` in "reference" and "current".
    """
    if os.path.commonpath([os.path.realpath(scratch), os.path.realpath(ROOT)]) == os.path.realpath(ROOT):
        raise AssertionError(f"the scratch directory {scratch} lies inside the repository")
    runs = [(code_dir, start_driver(code_dir, os.path.join(scratch, name)))
            for name, code_dir in (("reference", REFERENCE), ("current", ROOT))]
    return tuple(finish_driver(process, code_dir) for code_dir, process in runs)


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def load_recordings():
    """Every recording of the 3.5 hashes, by file name: tests/fixtures/reference_hashes_3_5*.json."""
    recordings = {}
    for name in sorted(os.listdir(FIXTURES)):
        if name.startswith(RECORDINGS) and name.endswith(".json"):
            with open(os.path.join(FIXTURES, name)) as handle:
                recordings[name] = json.load(handle)
    return recordings


class PinnedRelease35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.scratch = scratch.name
        cls.reference, cls.current = run_drivers(cls.scratch)
        cls.recordings = load_recordings()

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

    def test_the_current_modules_produce_what_the_3_5_modules_produce_in_this_environment(self):
        self.assertEqual(sorted(self.current["hashes"]), sorted(self.reference["hashes"]))
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                self.assertEqual(self.current["hashes"][case], self.reference["hashes"][case])
                self.assertEqual(self.current["facts"][case], self.reference["facts"][case])

    def test_the_cases_reach_the_stages_they_pin(self):
        facts = self.reference["facts"]
        events = {case: entry["events"] for case, entry in facts.items() if "events" in entry}
        # the cases kept from the 3.4 pin still reach what they reached there
        self.assertGreater(events["tree_stems_in_volume"]["bound_terminations"], 0)
        self.assertGreater(events["mesh"]["walk_bound_redraws"], 0)
        for case in ("tree_walk_avoidance_anastomosis", "tumour", "member_tumour_r3"):
            self.assertGreater(events[case]["collision_redraws"], 0, case)
        for case in ("mesh", "member_mesh_r3", "aligned", "member_aligned_r3", "member_aligned_r4", "guided_root_pair",
                     "guided_partner"):
            self.assertEqual(facts[case]["trees"], 2, case)
        for case in ("mesh", "aligned", "member_aligned_r3", "member_aligned_r4"):
            self.assertGreater(events[case]["anastomosis_bridges_arteriovenous"], 0, case)
        # guidance: every guided case steers; the onset, the flips of a nematic axis and the bank are reached
        for case in ("aligned", "guided_nematic", "guided_root_pair", "guided_classes", "guided_partner",
                     "member_aligned_r3", "member_aligned_r4"):
            self.assertGreater(events[case]["guided_steps"], 0, case)
        for case in ("aligned", "guided_classes", "guided_partner", "guided_root_pair"):
            self.assertGreater(events[case]["guidance_onset_steps"], 0, case)
        self.assertGreater(events["guided_nematic"]["guidance_sense_flips"], 0)
        self.assertGreater(events["guided_classes"]["unguided_steps"], 0)
        self.assertGreater(events["guided_classes"]["guidance_bank_steps"], 0)
        for case in ("tree_stems", "mesh", "tumour", "root_offsets_stems", "member_tree_r3"):
            self.assertEqual(events[case].get("guided_steps", 0), 0, case)
        # the plans: the aligned one holds every family, aligned at twice the share of each other
        self.assertEqual(facts["plan"]["families"], ["tree"] * 3 + ["mesh"] * 3 + ["tumour"] * 3)
        self.assertEqual(facts["plan_aligned"]["families"].count("aligned"), 5)        # 11 x 2/5 by largest remainder
        # the command line: aligned and inline guidance were grown, with voxels
        for case in ("cli_tree", "cli_aligned", "cli_guidance"):
            for stem, entry in facts[case].items():
                self.assertGreater(entry["voxels"], 0, stem)
                self.assertIn(f"_s{entry['seed']}", stem)
        (aligned,), (guided,) = facts["cli_aligned"].values(), facts["cli_guidance"].values()
        self.assertEqual((aligned["trees"], guided["trees"]), (2, 1))
        # describe meets every key of 3.5: tree labels, events, flags and the frame-relative keys
        for case in ("describe_cli_tree", "describe_cli_aligned"):
            self.assertEqual(facts[case]["events"], len(EVENT_KEYS_3_5), case)
            self.assertEqual(len(facts[case]["flags"]), len(FLAG_KEYS_3_5), case)
        self.assertGreater(facts["describe_margin"]["violations"], 0)
        self.assertGreater(facts["describe_forest"]["components"], 1)
        self.assertEqual(facts["frames"]["kind"], "axis")
        # joining and the libraries
        self.assertGreater(facts["join_cropped_roots"]["bridges"], 1)
        self.assertGreater(facts["join_cropped_roots"]["root_bridges"], 0)
        self.assertEqual(facts["library"]["files"],
                         ["net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        self.assertEqual(facts["library_aligned"]["files"], ["net_00000_tree.npz", "net_00001_aligned.npz"])
        self.assertEqual(facts["library_aligned"]["growth"]["box_c"], {"aligned": 12.0})
        for case in ("library", "library_aligned"):
            self.assertEqual(facts[case]["columns"], list(INDEX_COLUMNS_3_5), case)
            self.assertEqual(facts[case]["failures"], 0, case)

    def test_the_frame_records_reach_every_kind(self):
        # the frame-relative descriptors were computed, about an axis and about a plane
        for case in ("describe_aligned_member", "describe_aligned_member_bound", "describe_banked_plane",
                     "describe_moved_member", "describe_cli_aligned"):
            self.assertIn(case, self.reference["hashes"])
        self.assertEqual(self.reference["facts"]["frames"]["rotations"],
                         sorted(["about", "between", "between_antiparallel", "between_nematic", "of_frames", "random",
                                 "align_axis", "align_normal"]))

    def test_every_counter_appended_after_3_5_is_zero(self):
        def appended(value):
            # the "events" entries of a case's extras, at any depth; a library's are kept by archive
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "events" and all(not isinstance(count, dict) for count in item.values()):
                        yield from item.items()
                    elif key == "events":
                        for per_archive in item.values():
                            yield from per_archive.items()
                    else:
                        yield from appended(item)

        for case, extras in self.current["extras"].items():
            with self.subTest(case=case):
                for key, count in appended(extras):
                    self.assertEqual(count, 0, key)

    def test_with_the_later_options_off_what_3_6_added_is_off(self):
        # 3.6 appends the capillary fill and the rungs: with their options off
        # every one of their settings is at its off value, no rung is made, and
        # the rung stream is listed but never drawn (stage_generators above)
        off = {"capillary_generations": 0, "capillary_runs": 1, "cross_connect": False, "rung_below": 2.0,
               "rung_spacing": 20.0, "rung_radius": 8.0, "rung_lateral_deg": 60.0, "rung_axis": None,
               "rung_min_separation": 2}
        extras = self.current["extras"]
        grown = [case for case in extras if isinstance(extras[case].get("returned"), dict)]
        self.assertGreater(len(grown), 10)
        for case in grown:
            with self.subTest(case=case):
                self.assertEqual(extras[case]["returned"].get("rungs"), [])
                if "kwargs" in extras[case]:
                    self.assertEqual({key: extras[case]["kwargs"].get(key) for key in off}, off)
        for case in ("cli_tree", "cli_aligned", "cli_guidance"):
            for stem, entry in extras[case].items():
                with self.subTest(case=case, stem=stem):
                    for record in (entry["sidecar"], entry["metadata"]):
                        self.assertEqual({key: record.get(key) for key in off}, off)
                        self.assertEqual(record.get("rungs"), 0)
                    seed = self.current["facts"][case][stem]["seed"]
                    self.assertEqual(entry["rng_streams"], {"rungs": [seed, 5]})
                    self.assertEqual(entry["arrays"], [])
        for case in ("library", "library_aligned"):
            with self.subTest(case=case):
                for name, values in extras[case]["metadata_values"].items():
                    self.assertEqual(values.get("rungs"), [], name)
                for row in extras[case]["row_values"]:
                    self.assertEqual(row.get("rungs"), 0)
                self.assertEqual(extras[case]["columns"][:1], ["rungs"])
                self.assertEqual(extras[case]["arrays"], {name: [] for name in extras[case]["arrays"]})

    def test_the_released_presets_are_unchanged(self):
        import main
        for family, released in (("tree", {}), ("mesh", MESH_3_5), ("tumour", TUMOUR_3_5), ("aligned", ALIGNED_3_5)):
            with self.subTest(family=family):
                self.assertEqual(json.loads(json.dumps(main.FAMILIES[family])), released)

    def test_the_recorded_3_5_hashes_are_matched(self):
        # a case passes when it matches a recording; a recording that the 3.5
        # modules themselves reproduce here must be matched; one they do not
        # reproduce was made where rounding differs, and is skipped
        for recording, recorded in self.recordings.items():
            hashes = {key: value for key, value in recorded.items() if not key.startswith("_")}
            self.assertEqual(sorted(hashes), sorted(self.reference["hashes"]), recording)
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                recorded = [{key: value for key, value in entry.items() if not key.startswith("_")}[case]
                            for entry in self.recordings.values()]
                if self.current["hashes"][case] in recorded:
                    continue
                if self.reference["hashes"][case] not in recorded:
                    self.skipTest("this environment's rounding differs from every recording machine's; "
                                  "the same-environment comparison above still holds")
                self.assertIn(self.current["hashes"][case], recorded)


if __name__ == "__main__":
    unittest.main()
