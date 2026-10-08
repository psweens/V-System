"""
Pins what 3.6 draws, writes, measures, places and joins.

The 3.6 modules (tests/fixtures/reference_code_3_6: main's import closure as
released in 3.6.1, connections included, with describe, join, library and
frames) and the current ones each run one driver in a subprocess, in the same
environment, and the sha256 of everything the driver produces are compared.
The 3.4 and 3.5 pins (tests/test_pinned_release_3_4.py and _3_5.py) cover
what those releases drew in full; this one keeps one case of each kind from
the 3.5 pin, so that what 3.6 added is pinned on each kind of growth, and
adds the cases 3.6 opened:

    grow_network   a stem tree, a walked tree with avoidance and anastomosis,
                   the mesh, tumour, aligned and aligned_tight presets with
                   their released kwargs written out literally, and calibre
                   classes with a fixed polar axis, an unguided class and a
                   banked plane;
                   a stem tree with and without the capillary fill, a walked
                   tree filled and cross-connected, rungs kept from a fixed
                   axis without the fill, and a filled pair grown free;
    library        grow_member for tree, mesh and aligned and for the
                   aligned_bed, aligned_tight and capillary_bed presets, with
                   argv, kwargs and index row, and a plan of every family 3.6
                   offers;
    describe       a fixture archive at a margin, an aligned_bed member
                   against its frame at two class bounds, with the tissue
                   distance over its growth box (read before a fixed voxel
                   size) and over a box given, a banked plane against its
                   frame, a moved member, straight vessels nearly along and
                   nearly across that member's axis, a ring of junctions
                   longer than the loop search's depth, and the archives
                   below, so that the tissue distance is taken from every
                   source it can have;
    frames         rotations, align and the transforms on an aligned member;
    main.main      a tree run, a filled and cross-connected run at a fixed
                   voxel size with every capillary and rung option given, and
                   an aligned_tight run: archive, sidecar, TIFF pixels and
                   file;
    join_networks  a cropped forest of the library members, the rungs of
                   aligned_bed among them, joined with root attachment into a
                   merged archive;
    library.main   the default three-network library, and one of tree and
                   capillary_bed with a box constant of its own, whose
                   capillary_bed archive carries rungs.

Every record (events, sidecars, metadata, kwargs, frame records and their
rules, rung records, describe output, index rows, join output) is first
restricted, recursively, to the keys 3.6 wrote, which are listed literally
below, so that what a later version appends never enters the comparison while
everything 3.6 wrote must still be there, unchanged; a list of dicts, such as
a frame's rules or the rungs, is restricted entry by entry. The global
generators' states after each case, the states of the generators each stage
created and the joining generator's state are compared too, so a stage added
later must create no generator while its option is off, and every counter
appended to the events must be zero.

Running both in one environment is the only portable comparison: the
B-spline, the walk and the collision checks go through kernels whose last
bits differ between machines. Hashes recorded on a machine
(tests/fixtures/reference_hashes_3_6*.json, one file per recording: Linux
x86_64 with OpenBLAS's SkylakeX kernels and numpy's AVX-512 loops, and with
OpenBLAS's Zen kernels and numpy's AVX2 loops (_zen)) are checked as well: a
case passes when it matches a recording, and the check is
skipped only when the reference code itself reproduces none of them here.
The reference directory is an unchanged copy of the release: the sha256 of
each of its files is recorded with the hashes and checked, and the drivers
write nothing but to a temporary directory outside the repository. The two
drivers run side by side, each in its own process and output directory.

Run from the repository root with
    python -m unittest tests.test_pinned_release_3_6
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
REFERENCE = os.path.join(FIXTURES, "reference_code_3_6")
RECORDINGS = "reference_hashes_3_6"           # tests/fixtures/reference_hashes_3_6*.json, one per recording

REFERENCE_MODULES = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance", "connections",
                     "describe", "join", "library", "frames")
ARCHIVES = ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz", "Lnet_i9_s3.npz")

# ---------------------------------------------------------------------------
# The keys 3.6 wrote, listed literally so that a later version's additions
# never enter the comparison.
# ---------------------------------------------------------------------------
EVENT_KEYS_3_6 = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
                  "collision_redraws", "collision_terminations", "collision_truncated_stems",
                  "root_relocations", "root_collisions",
                  "anastomosis_tips", "anastomosis_tips_inside_junction", "anastomosis_kin_skipped",
                  "anastomosis_behind_skipped", "anastomosis_selected", "anastomosis_bridges",
                  "anastomosis_bridges_arteriovenous", "anastomosis_bridges_same_tree",
                  "anastomosis_partner_tips", "anastomosis_partner_interior",
                  "anastomosis_source_consumed", "anastomosis_no_partner",
                  "anastomosis_collision_failed", "anastomosis_bridge_redraws",
                  "guided_steps", "guidance_onset_steps", "unguided_steps", "guidance_undefined_steps",
                  "guidance_sense_flips", "guidance_bank_steps",
                  "rung_sites", "rung_sites_near_junction", "rung_sites_consumed", "rung_no_partner",
                  "rung_collision_failed", "rung_bridges", "rung_bridges_cross_tree",
                  "rung_kin_skipped", "rung_angle_skipped", "rung_bridge_redraws")
SHAPING_KEYS_3_6 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                    "collision_attempts", "collision_index", "anastomose", "anastomosis_radius",
                    "anastomosis_fraction", "anastomose_mode", "anastomosis_min_separation",
                    "guidance", "root_offsets",
                    "capillary_generations", "capillary_runs", "cross_connect", "rung_below", "rung_spacing",
                    "rung_radius", "rung_lateral_deg", "rung_axis", "rung_min_separation")
KWARGS_KEYS_3_6 = ("tVol", "fit", "clip_axes", "voxel_size", "subdivisions", "d_min",
                   "grow_in_volume", "seed") + SHAPING_KEYS_3_6
PROPERTY_KEYS_3_6 = ("k", "epsilon", "randmarg", "sigma", "stochparams", "roll_angle", "stem_angle",
                     "aneurysm_prob", "stenosis_prob")
# what grow_network returned beside its arrays, programs, bridges, rungs and events
GROWN_KEYS_3_6 = ("program", "growth_box_um", "root_positions_um", "clip_axes", "collision_index", "frame")
# a rung, as grow_network returns it and a library archive's metadata stores it
RUNG_FIELDS_3_6 = ("site", "partner", "site_tree", "partner_tree", "chord_um", "arc_um", "diameter_um", "points",
                   "redraws", "angle_deg")
# the frame record and the keys of a rule of any field, as given or normalised
FRAME_KEYS_3_6 = ("frame_version", "kind", "axis", "sense", "tree_senses", "normal", "origin",
                  "grow_direction", "grow_perpendicular", "rules")
RULE_KEYS_3_6 = ("below", "field", "axis", "normal", "sense", "polarity", "length", "onset", "bank")
# a sidecar's "rungs" is the number of rungs made
SIDECAR_KEYS_3_6 = ("seed", "iterations", "d0", "d_min", "volume", "axis_order", "units", "fit",
                    "clip_axes", "connect", "grow_in_volume", "voxel_size", "subdivisions",
                    "archive_version", "family", "trees", "growth_box_um", "root_positions_um",
                    "bridges", "bridge_persistence", "frame", "rungs") + SHAPING_KEYS_3_6
RNG_STREAMS_3_6 = ("walk", "anastomosis", "rungs")
MEMBER_KEYS_3_6 = ("argv", "niter", "d0")
PLAN_KEYS_3_6 = ("id", "family", "family_index", "position", "seed", "ratio", "bin", "u", "file")
# index.csv's columns; a run's seconds and peak memory are left out of the comparison
INDEX_COLUMNS_3_6 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points",
                     "polylines", "tips", "junctions", "cycles", "components", "total_length",
                     "diameter_p50", "diameter_p90", "diameter_p99",
                     "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min", "diameter_max",
                     "min_point_diameter", "clearance_min", "clearance_violations", "bridges", "events",
                     "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                     "nodes_sha256", "program_sha256",
                     "frame_kind", "capillary_order", "larger_order", "capillary_polar_order",
                     "capillary_length_share", "capillary_volume_share", "capillary_segment_median",
                     "transverse_spacing_median", "rungs")
UNPINNED_COLUMNS = ("seconds", "peak_rss_mb")
# a library archive's "rungs" is the list of rungs made
METADATA_KEYS_3_6 = ("units", "library_id", "family", "ratio", "bin", "u", "seed", "ratio_law",
                     "ratio_range", "argv", "niter", "generations", "d0", "avoid_collisions",
                     "root_columns", "growth_box", "root_positions", "bridges", "min_point_diameter",
                     "programs", "archive_version", "nodes_sha256", "program_sha256", "d_min", "frame", "rungs")
UNPINNED_MANIFEST_KEYS = ("code_sha256", "library_version")
FLAG_KEYS_3_6 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                 "anastomose", "anastomosis_fraction", "anastomosis_radius", "anastomose_mode",
                 "family", "seed", "iterations", "d0", "d_min")
DESCRIBE_KEYS_3_6 = ("units", "points", "polylines", "vertices", "edges", "components", "cycles",
                     "total_length_mm", "degree_histogram", "length_density_mm_per_mm3", "volume_mm3",
                     "volume_source")
ORIENTATION_KEYS_3_6 = ("eigenvalues", "principal_axis", "fractional_anisotropy", "polar_histogram_deg",
                        "azimuth_histogram_deg")
DESCRIBE_GROUPS_3_6 = {
    "diameter_um": ("p50", "p90", "p99", "min", "max"),
    "arc_chord": ("mean", "median", "p90", "max", "n_segments", "n_closed"),
    "curvature_per_um": ("mean_abs", "n_pairs"),
    "tips": ("count", "per_mm"),
    "orientation": ORIENTATION_KEYS_3_6,
    "bounding_box_um": ("min", "max"),
    "clearance_um": ("min", "min_columns", "min_is_lower_bound", "search_reach_um", "violations",
                     "pairs_checked", "margin"),
    "classes": ("bound", "d_ref", "d_ref_source"),
    "polar_order": ("all", "capillary", "larger"),
    "transverse_spacing": ("planes", "count", "median", "mean", "p10", "p90", "median_d", "mean_d", "p10_d",
                           "p90_d", "crossing_density", "crossing_density_d", "area_source"),
    "junctions": ("count", "degree_3", "degree_4", "degree_5_plus", "mean_degree", "segments_per_junction"),
    "branch_angles_deg": ("count", "skipped", "min", "median", "max"),
}
CLASSES_3_6 = ("all", "capillary", "larger")
# an entry of frame_orientation: about an axis, or about a plane
FRAME_ENTRY_KEYS_3_6 = ("S", "mean_abs_cos", "crossing_ratio", "mean_angle_deg", "within_20_deg", "within_45_deg",
                        "watson_K", "fisher_axial_K", "in_plane_fraction", "S_n", "length", "length_d",
                        "fisher_axial_K_exact")
SEGMENT_KEYS_3_6 = ("count", "median", "p10", "p90", "cv", "median_d", "p10_d", "p90_d")
SHARE_KEYS_3_6 = ("below_1.5", "below_2", "below_3")
PER_TREE_KEYS_3_6 = ("points", "polylines", "tips", "length_mm")
# the shortest loops through sampled segments and junctions; a histogram is kept whole
LOOP_SEGMENT_KEYS_3_6 = ("segments", "every", "sampled", "found", "none_fraction", "median", "mean", "histogram",
                         "cycles", "cycles_per_length", "cycles_per_length_d")
LOOP_NODE_KEYS_3_6 = ("junctions", "every", "sampled", "found", "none_fraction", "median", "mean", "histogram",
                      "length_median", "length_mean", "length_median_d", "length_mean_d")
VARIATION_KEYS_3_6 = ("count", "excluded", "median", "p90")
TISSUE_KEYS_3_6 = ("spacing_requested", "spacing", "raised", "shape", "points", "domain", "capsules",
                   "inside_fraction", "outside_points", "mean", "median", "p90", "p99", "max", "spacing_d", "mean_d",
                   "median_d", "p90_d", "p99_d", "max_d")
TISSUE_DOMAIN_KEYS_3_6 = ("min", "max", "source")
JOIN_EVENT_KEYS_3_6 = ("join_networks", "join_points", "join_isolated", "join_tips",
                       "join_roots", "join_cut_ends", "join_stubs", "join_eligible", "join_selected",
                       "join_not_selected", "join_bridges", "join_bridged_partners", "join_no_partner",
                       "join_collision_failed", "join_over_budget",
                       "join_partner_tips", "join_partner_interior", "join_source_consumed",
                       "join_redraws", "join_box_redraws", "join_kin_skipped", "join_behind_skipped",
                       "join_components_joined",
                       "join_root_eligible", "join_root_selected", "join_root_attached",
                       "join_root_not_selected", "join_root_no_partner", "join_root_collision_failed",
                       "join_root_over_budget")
JOIN_SUMMARY_KEYS_3_6 = ("components", "length", "free_tips", "free_tips_per_length", "bridge_volume",
                         "blunt_roots", "blunt_roots_per_length", "free_ends", "free_ends_per_length")
JOIN_SUMMARY_ENTRY_KEYS_3_6 = ("before", "after")        # of every summary entry but bridge_volume, a number
JOIN_MERGED_KEYS_3_6 = ("nodes", "edges", "node_kind", "network", "tree")
ARCHIVE_ARRAYS_3_6 = ("nodes", "program", "metadata", "edges", "node_kind", "tree")
INDEX_KEYS_3_6 = ("units", "ratio_law", "networks")
JOIN_BRIDGE_FIELDS_3_6 = ("tip", "partner", "source_kind", "partner_kind", "chord", "arc", "diameter",
                          "volume", "redraws")
# the presets as 3.6.1 released them, passed literally so that they stay pinned
MESH_3_6 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
            "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.5, "grow_in_volume": True}
TUMOUR_3_6 = {"tortuosity": "walk", "persistence": 3.0, "avoid_collisions": True, "anastomose": True,
              "anastomose_mode": "any", "anastomosis_fraction": 0.8, "d0": [20.0, 10.0], "aneurysm_prob": 0.1,
              "stenosis_prob": 0.1}
ALIGNED_3_6 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
               "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
               "root_offsets": [[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
               "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                             "polarity": "partner", "length": 4.9, "onset": 2.0},
                            {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0}]}
ALIGNED_TIGHT_3_6 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                     "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                     "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                   "polarity": "partner", "length": 2.0, "onset": 0.0},
                                  {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                   "onset": 0.0}]}
ALIGNED_BED_3_6 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                   "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                   "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                   "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                 "polarity": "partner", "length": 2.0, "onset": 0.0},
                                {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                 "onset": 0.0}],
                   "capillary_generations": 2, "capillary_runs": 4,
                   "cross_connect": True, "rung_below": 2.0, "rung_spacing": 40.0, "rung_radius": 8.0,
                   "rung_lateral_deg": 60.0, "rung_axis": [1, 0, 0], "rung_min_separation": 2}
CAPILLARY_BED_3_6 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 1.0, "grow_in_volume": True,
                     "capillary_generations": 3, "capillary_runs": 2,
                     "cross_connect": True, "rung_below": 2.0, "rung_spacing": 20.0, "rung_radius": 8.0,
                     "rung_lateral_deg": 60.0, "rung_axis": None, "rung_min_separation": 2}


def _whole(keys):
    return {key: None for key in keys}


# How each record is restricted before it is hashed. None keeps a value whole,
# a dict keeps the listed keys and restricts each value in turn, {"*": rule}
# keeps every key (a label or an id, not a field) and restricts each value,
# and {"[]": rule} restricts every entry of a list.
RULE_RULE = {"[]": _whole(RULE_KEYS_3_6)}
RUNG_RULE = {"[]": _whole(RUNG_FIELDS_3_6)}
FRAME_RULE = dict(_whole(FRAME_KEYS_3_6), rules=RULE_RULE)
KWARGS_RULE = dict(_whole(KWARGS_KEYS_3_6), guidance=RULE_RULE)
RULES = {
    "events": _whole(EVENT_KEYS_3_6),
    "grown": dict(_whole(GROWN_KEYS_3_6), frame=FRAME_RULE),
    "rungs": RUNG_RULE,
    "kwargs": KWARGS_RULE,
    "frame": FRAME_RULE,
    "member": dict(_whole(MEMBER_KEYS_3_6), properties=_whole(PROPERTY_KEYS_3_6)),
    "plan": _whole(PLAN_KEYS_3_6),
    "sidecar": dict(_whole(SIDECAR_KEYS_3_6), properties=_whole(PROPERTY_KEYS_3_6),
                    rng_streams=_whole(RNG_STREAMS_3_6), events=_whole(EVENT_KEYS_3_6), frame=FRAME_RULE,
                    guidance=RULE_RULE),
    "index_row": dict(_whole(column for column in INDEX_COLUMNS_3_6 if column not in UNPINNED_COLUMNS),
                      events=_whole(EVENT_KEYS_3_6)),
    "metadata": dict(_whole(METADATA_KEYS_3_6), properties=_whole(PROPERTY_KEYS_3_6),
                     grow_kwargs=KWARGS_RULE, events=_whole(EVENT_KEYS_3_6), frame=FRAME_RULE, rungs=RUNG_RULE),
    "describe": dict(_whole(DESCRIBE_KEYS_3_6), events=_whole(EVENT_KEYS_3_6), flags=_whole(FLAG_KEYS_3_6),
                     per_tree={"*": _whole(PER_TREE_KEYS_3_6)}, frame=FRAME_RULE,
                     frame_orientation=dict({cls: _whole(FRAME_ENTRY_KEYS_3_6) for cls in CLASSES_3_6}, kind=None),
                     orientation_by_class={cls: _whole(ORIENTATION_KEYS_3_6 + ("S_max", "planarity"))
                                           for cls in CLASSES_3_6},
                     calibre_shares={"length": _whole(SHARE_KEYS_3_6), "volume": _whole(SHARE_KEYS_3_6)},
                     segments_by_class={cls: _whole(SEGMENT_KEYS_3_6) for cls in ("capillary", "larger")},
                     loops={"by_segment": _whole(LOOP_SEGMENT_KEYS_3_6), "by_node": _whole(LOOP_NODE_KEYS_3_6)},
                     segment_diameter_variation={cls: _whole(VARIATION_KEYS_3_6) for cls in CLASSES_3_6},
                     tissue_distance=dict(_whole(TISSUE_KEYS_3_6), domain=_whole(TISSUE_DOMAIN_KEYS_3_6)),
                     **{group: _whole(keys) for group, keys in DESCRIBE_GROUPS_3_6.items()}),
    "join_events": _whole(JOIN_EVENT_KEYS_3_6),
    "join_summary": dict({key: _whole(JOIN_SUMMARY_ENTRY_KEYS_3_6) for key in JOIN_SUMMARY_KEYS_3_6},
                         bridge_volume=None),
    "join_bridge": _whole(JOIN_BRIDGE_FIELDS_3_6),
    "index": _whole(INDEX_KEYS_3_6),
}

# Plain lists the driver reads beside the rules.
LISTS = {
    "modules": list(REFERENCE_MODULES),
    "archives": list(ARCHIVES),
    "archive_arrays": list(ARCHIVE_ARRAYS_3_6),
    "join_merged": list(JOIN_MERGED_KEYS_3_6),
    "index_columns": list(INDEX_COLUMNS_3_6),
    "unpinned_columns": list(UNPINNED_COLUMNS),
    "unpinned_manifest": list(UNPINNED_MANIFEST_KEYS),
}
# the presets the driver grows literally; aligned_bed and capillary_bed are grown by grow_member, which reads each
# side's own presets, and are held to the released ones by test_the_released_presets_are_unchanged
SETTINGS = {"mesh": MESH_3_6, "tumour": TUMOUR_3_6, "aligned": ALIGNED_3_6, "aligned_tight": ALIGNED_TIGHT_3_6}

# Runs in a subprocess with the modules of one code directory first on the
# path and that directory as the working directory, and prints, per case,
# the hashes of what those modules produce ("hashes"), a few plain counts of
# the same output ("facts") and whatever the output holds beyond the 3.6 keys
# ("extras"). It uses nothing but what 3.6 offers.
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
    # the 3.6 part of a record; see RULES in the test
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
    # what a record holds beyond its 3.6 keys, one level deep
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
            "rungs": record_sha(restrict(grown["rungs"], rules["rungs"])),
            "events": record_sha(restrict(grown["events"], rules["events"])),
            "record": record_sha(restrict(grown, rules["grown"])),
            "global_generators": global_generators(), "stage_generators": stage_generators()}


def grown_facts(grown):
    # a program's moves are its f(...) tokens, which the capillary fill adds to
    return {"columns": int(grown["nodes"].shape[1]), "bridges": len(grown["bridges"]),
            "trees": len(grown["programs"]), "rungs": len(grown["rungs"]),
            "moves": sum(program.count("f(") for program in grown["programs"]),
            "events": {key: grown["events"][key] for key in rules["events"] if grown["events"].get(key)}}


def grown_extras(grown):
    known = set(rules["grown"]) | {"nodes", "tree", "edges", "node_kind", "programs", "bridges", "rungs", "events"}
    return {"events": beyond(grown["events"], rules["events"]),
            "returned": {key: grown[key] for key in sorted(set(grown) - known)},
            "rungs": sorted({key for rung in grown["rungs"] for key in beyond(rung, rules["rungs"]["[]"])})}


def archive_hashes(path, metadata_rule):
    # the arrays 3.6 stored, as stored (read with numpy, not with the loader under test)
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
    tissue = report["tissue_distance"]
    facts[name] = {"points": report["points"], "components": report["components"], "cycles": report["cycles"],
                   "volume_source": report["volume_source"], "violations": report["clearance_um"]["violations"],
                   "events": len(restrict(report["events"], rules["events"])),
                   "flags": sorted(restrict(report["flags"], rules["describe"]["flags"])),
                   "trees": sorted(report["per_tree"] or {}),
                   "tissue_source": tissue["domain"]["source"] if tissue is not None else None,
                   "junctions": report["junctions"]["count"], "loops_found": report["loops"]["by_segment"]["found"],
                   "loops_found_by_node": report["loops"]["by_node"]["found"],
                   "K_exact": ((report["frame_orientation"] or {}).get("all") or {}).get("fisher_axial_K_exact")}
    extras[name] = {"report": sorted(beyond(report, rules["describe"])),
                    "events": beyond(report["events"], rules["events"])}


hashes, facts, extras = {}, {}, {}

# --- grow_network ----------------------------------------------------------
PROPERTIES = {"k": 3, "epsilon": 7.0, "randmarg": 0.2, "sigma": 5, "stochparams": True}
VOLUME = (48, 48, 24)
WALK = {"tortuosity": "walk", "persistence": 8.0}
# a walked tree with avoidance and anastomosis within the tree, fine enough at d_min 1 to carry rungs
FINE = dict(WALK, persistence=10.0, avoid_collisions=True, anastomose=True, anastomose_mode="any",
            anastomosis_fraction=0.5, d_min=1.0)


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
    # one case of each kind of the 3.5 pin, which that pin and the 3.4 pin
    # cover in full, so that the records 3.6 added (no rungs, the fill and the
    # rungs off) are pinned on each kind of growth
    ("tree_stems", 11, 6, 20.0, PROPERTIES, {}),
    ("tree_walk_avoidance_anastomosis", 14, 6, 20.0, PROPERTIES,
     dict(WALK, avoid_collisions=True, anastomose=True, anastomose_mode="any", anastomosis_min_separation=2)),
    ("mesh", 3, 6, 20.0) + preset("mesh"),
    ("tumour", 1, 6, 20.0) + preset("tumour"),
    ("aligned", 5, 6, 20.0, preset("aligned")[0], dict(preset("aligned")[1], d_min=4.0)),
    # a family 3.6 added, with its released kwargs
    ("aligned_tight", 5, 6, 20.0, preset("aligned_tight")[0], dict(preset("aligned_tight")[1], d_min=2.0)),
    # calibre classes: a fixed polar axis below 2 d_min, an unguided class below 4 and a banked plane above
    ("guided_classes", 19, 6, 20.0, dict(PROPERTIES, roll_angle=70.0),
     dict(WALK, d_min=1.0,
          guidance=[{"below": 2.0, "field": "axis", "axis": [0, 0, 1], "sense": "polar", "polarity": "fixed",
                     "length": 2.0, "onset": 1.0},
                    {"below": 4.0, "field": None},
                    {"below": None, "field": "plane", "normal": [0, 0, 1], "length": 3.0, "bank": True}])),
    # what 3.6 added: a stem tree at d_min without and with capillary trees of
    # two generations whose stems run as three blocks
    ("stems_unfilled", 21, 6, 8.0, PROPERTIES, {"d_min": 4.0}),
    ("fill_stems", 21, 6, 8.0, PROPERTIES, {"d_min": 4.0, "capillary_generations": 2, "capillary_runs": 3}),
    # a walked tree filled with capillary trees of one generation run as two
    # blocks, its capillaries cross-connected by rungs about their tangents
    ("fill_walk_rungs", 22, 8, 4.0, dict(PROPERTIES, epsilon=10.0),
     dict(FINE, capillary_generations=1, capillary_runs=2, cross_connect=True)),
    # rungs kept 45 degrees from a fixed axis, without the fill
    ("rungs_axis", 1, 8, 4.0, dict(PROPERTIES, epsilon=10.0),
     dict(FINE, cross_connect=True, rung_axis=[1, 0, 0], rung_lateral_deg=45.0)),
    # a filled arteriovenous pair grown free, whose second root is still placed by the free extent
    ("fill_free_pair", 23, 6, 8.0, PROPERTIES,
     {"d_min": 4.0, "capillary_generations": 2, "capillary_runs": 2, "anastomose": True,
      "anastomose_mode": "arteriovenous"}),
)
for name, seed, niter, d0, properties, options in GROW_CASES:
    random.seed(seed)
    np.random.seed(seed)
    grown = main.grow_network(niter, d0, dict(properties), VOLUME, seed=seed, **options)
    hashes[name], facts[name], extras[name] = grown_hashes(grown), grown_facts(grown), grown_extras(grown)

# --- library members, their index rows and a plan ---------------------------
# seeds on both sides of 2 ** 31, as a library's own seeds are; the families
# 3.6 added at the smallest ratio of the default range
MEMBERS = (("tree", 3.0, 101), ("mesh", 3.0, 2147483649), ("aligned", 3.0, 107),
           ("aligned_bed", 2.52, 3), ("aligned_tight", 2.52, 3), ("capillary_bed", 2.52, 4))
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
                    "edges": grown["edges"], "frame": grown["frame"], "growth_box_um": grown["growth_box_um"],
                    "family": family})

# a plan of every family 3.6 offers, which leaves a remainder (the 3.5 pin plans the default families)
PLANS = (("plan_3_6", 9, 9, ("tree", "mesh", "tumour", "aligned", "aligned_tight", "aligned_bed", "capillary_bed"),
          (1, 1, 1, 1, 1, 1, 1)),)
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
# what 3.6 added: an aligned_bed member against its frame, with the tissue
# distance over its growth box as a library archive records it, which is read
# before the field of a fixed voxel size given beside it (that field still
# sets the volume), and at another class bound with the tissue distance over a
# box given
bed_member = next(member for member in members if member["family"] == "aligned_bed")
describe_case("describe_aligned_bed_member", describe_module.describe(
    bed_member["nodes"], bed_member["edges"],
    metadata={"growth_box": bed_member["growth_box_um"], "fit": "voxel_size", "volume": [48, 48, 24],
              "voxel_size": 4.0},
    tree=bed_member["tree"], frame=bed_member["frame"], d_ref=1.0, evd_spacing=1.0))
describe_case("describe_aligned_bed_member_bound", describe_module.describe(
    bed_member["nodes"], bed_member["edges"], volume=bed_member["growth_box_um"], tree=bed_member["tree"],
    frame=bed_member["frame"], d_ref=1.0, class_bound=3.0, evd_spacing=2.0))
# a banked plane against its frame, with d_ref from the argument
random.seed(19)
np.random.seed(19)
banked = main.grow_network(6, 20.0, dict(PROPERTIES, roll_angle=70.0), VOLUME, seed=19, d_min=1.0,
                           guidance=[{"below": None, "field": "plane", "normal": [0, 0, 1], "length": 2.0,
                                      "bank": True}], **WALK)
stage_generators()
describe_case("describe_banked_plane", describe_module.describe(banked["nodes"], banked["edges"], tree=banked["tree"],
                                                                frame=banked["frame"], d_ref=1.0))


def straight(degrees, length=100.0, points=11, diameter=1.0):
    # one straight vessel in the x-y plane at `degrees` to x, the aligned_bed member's axis
    t = np.radians(degrees)
    line = np.outer(np.linspace(0.0, length, points), [np.cos(t), np.sin(t), 0.0]).T + 10.0
    return np.vstack([line, np.full((1, points), diameter)])


def ring(n, side=10.0, stub=4.0, diameter=1.0):
    # n junctions on a regular polygon, each with a stub outwards, as polylines separated by NaN columns
    radius = side / (2.0 * np.sin(np.pi / n))
    angles = 2.0 * np.pi * np.arange(n) / n
    corners = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)], axis=1) + 100.0
    columns = []
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        for piece in (np.stack([a, (a + b) / 2.0, b]), np.stack([a, a + stub * (a - 100.0) / radius])):
            columns.extend([np.vstack([piece.T, np.full((1, len(piece)), diameter)]), np.full((4, 1), np.nan)])
    return np.concatenate(columns[:-1], axis=1)


# tangents nearly along and nearly across the frame's axis, where the exact
# Fisher-axial concentration lies far out in its bracket
for name, degrees in (("describe_straight_along", 6.0), ("describe_straight_across", 89.5)):
    describe_case(name, describe_module.describe(straight(degrees), frame=bed_member["frame"], d_ref=1.0))
# a ring of 32 junctions, whose loop through a junction the search finds only at its full depth
describe_case("describe_long_loop", describe_module.describe(ring(32)))

# --- frames: rotations, align and the transforms on an aligned member ----------
aligned_member = next(member for member in members if member["family"] == "aligned")
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
             # what 3.6 added: every capillary and rung option, at a fixed voxel size and not in the volume
             ("cli_fill", ["--count", "1", "--seed", "11", "--volume", "48", "48", "24", "--iterations", "6", "6",
                           "--d0", "8", "0", "--d-min", "4", "--fit", "voxel_size", "--voxel-size", "2",
                           "--tortuosity", "walk", "--persistence", "8", "--avoid-collisions", "--anastomose",
                           "--capillary-generations", "2", "--capillary-runs", "2", "--cross-connect",
                           "--rung-below", "2.5", "--rung-spacing", "15", "--rung-radius", "10",
                           "--rung-lateral-deg", "50", "--rung-axis", "[0, 1, 0]", "--rung-min-separation", "3"]),
             # and a family 3.6 added
             ("cli_aligned_tight", ["--family", "aligned_tight", "--d-min", "2", "--count", "1", "--seed", "9",
                                    "--volume", "48", "48", "24", "--iterations", "6", "6"]))
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
                             "bridges": sidecar["bridges"], "rungs": sidecar["rungs"], "d0": sidecar["d0"],
                             "seed": sidecar["seed"]}
        extras[name][stem] = {"sidecar": beyond(sidecar, rules["sidecar"]),
                              "metadata": beyond(metadata, rules["sidecar"]),
                              "rng_streams": beyond(sidecar["rng_streams"], rules["sidecar"]["rng_streams"]),
                              "events": beyond(sidecar["events"], rules["events"]), "arrays": unknown_arrays}
    hashes[name].update(global_generators=global_generators(), stage_generators=stage_generators())

# --- describe on the archives just written --------------------------------------
# the tissue distance over the bounding box, over the field a fixed voxel size
# sets and over the growth box grow_network records, at spacings that keep each
# grid small
for name, spacing in (("cli_tree", 16.0), ("cli_fill", 4.0), ("cli_aligned_tight", 4.0)):
    stem = next(entry[:-4] for entry in hashes[name]["files"] if entry.endswith(".npz"))
    describe_case("describe_" + name, describe_module.describe_archive(os.path.join(out_dir, name, stem + ".npz"),
                                                                       evd_spacing=spacing))

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
# the rungs of the aligned_bed member are columns of the bridge label: obstacles to the join, never roots
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
# each with the tissue-distance spacing its archives are described at, None for none
LIBRARIES = (("library", None, ["--count", "3", "--seed", "1", "--ratio-range", "2.52", "3.5", "--workers", "1"]),
             # what 3.6 added: capillary_bed grown in a cube of its own constant
             # beside a tree, with rungs, and the tissue distance over a library archive's growth box
             ("library_fill", 2.0, ["--count", "2", "--seed", "1", "--families", "tree", "capillary_bed",
                                    "--box-c", "capillary_bed", "10", "--ratio-range", "2.52", "2.6", "--workers",
                                    "1"]))
for case, spacing, argv in LIBRARIES:
    directory = os.path.join(out_dir, case)
    with contextlib.redirect_stdout(io.StringIO()):
        status = library.main(["--out", directory] + argv)
    with open(os.path.join(directory, "manifest.json")) as handle:
        content = json.load(handle)["content"]
    with open(os.path.join(directory, "index.json")) as handle:
        index = json.load(handle)
    with open(os.path.join(directory, "index.csv"), newline="") as handle:
        table = list(csv.reader(handle))
    # index.csv restricted to the 3.6 columns, which a later version keeps first and in order
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
    metadata_by_file = {}
    for row in index["networks"]:
        entry[row["file"]], extra["arrays"][row["file"]], metadata = archive_hashes(
            os.path.join(directory, row["file"]), "metadata")
        metadata_by_file[row["file"]] = metadata
        extra["metadata"][row["file"]] = sorted(beyond(metadata, rules["metadata"]))
        extra["metadata_values"][row["file"]] = {key: value
                                                 for key, value in beyond(metadata, rules["metadata"]).items()
                                                 if key not in ("seconds", "peak_rss_mb", "events")}
        extra["events"][row["file"]] = beyond(metadata["events"], rules["events"])
        describe_case(f"describe_{case}_" + row["file"], describe_module.describe_archive(
            os.path.join(directory, row["file"]), evd_spacing=spacing))
    hashes[case] = entry
    facts[case] = {"files": [row["file"] for row in index["networks"]], "families": content["families"],
                   "columns": table[0][:len(lists["index_columns"])], "rows": len(table) - 1,
                   "growth": content["growth"], "failures": len(content["failures"]),
                   "weights": library.library_weights(index).tolist(),
                   "rungs": {row["file"]: {"rungs": row["rungs"], "bridges": row["bridges"],
                                           "metadata_rungs": len(metadata_by_file[row["file"]]["rungs"])}
                             for row in index["networks"]}}
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
    What the 3.6 modules and the current ones produce, run side by side:
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
    """Every recording of the 3.6 hashes, by file name: tests/fixtures/reference_hashes_3_6*.json."""
    recordings = {}
    for name in sorted(os.listdir(FIXTURES)):
        if name.startswith(RECORDINGS) and name.endswith(".json"):
            with open(os.path.join(FIXTURES, name)) as handle:
                recordings[name] = json.load(handle)
    return recordings


class PinnedRelease36Tests(unittest.TestCase):
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

    def test_the_current_modules_produce_what_the_3_6_modules_produce_in_this_environment(self):
        self.assertEqual(sorted(self.current["hashes"]), sorted(self.reference["hashes"]))
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                self.assertEqual(self.current["hashes"][case], self.reference["hashes"][case])
                self.assertEqual(self.current["facts"][case], self.reference["facts"][case])

    def test_the_cases_reach_the_stages_they_pin(self):
        facts = self.reference["facts"]
        events = {case: entry["events"] for case, entry in facts.items() if "events" in entry}
        # the cases kept from the 3.5 pin still reach what they reached there
        self.assertGreater(events["mesh"]["walk_bound_redraws"], 0)
        for case in ("tree_walk_avoidance_anastomosis", "tumour"):
            self.assertGreater(events[case]["collision_redraws"], 0, case)
        for case in ("mesh", "aligned", "member_mesh_r3", "member_aligned_r3"):
            self.assertGreater(events[case]["anastomosis_bridges_arteriovenous"], 0, case)
        for case in ("aligned", "guided_classes", "member_aligned_r3"):
            self.assertGreater(events[case]["guided_steps"], 0, case)
            self.assertGreater(events[case]["guidance_onset_steps"], 0, case)
        self.assertGreater(events["guided_classes"]["unguided_steps"], 0)
        self.assertGreater(events["guided_classes"]["guidance_bank_steps"], 0)
        for case in ("tree_stems", "mesh", "tumour", "member_tree_r3", "fill_stems", "fill_walk_rungs"):
            self.assertEqual(events[case].get("guided_steps", 0), 0, case)
        # every pair grows two trees: the presets, the members and the filled pair grown free
        for case in ("mesh", "aligned", "aligned_tight", "fill_free_pair", "member_mesh_r3", "member_aligned_r3",
                     "member_aligned_bed_r2.52", "member_aligned_tight_r2.52", "member_capillary_bed_r2.52"):
            self.assertEqual(facts[case]["trees"], 2, case)
        # aligned_tight and aligned_bed steer their capillaries, and every family 3.6 added joins arterial to venous
        for case in ("aligned_tight", "member_aligned_bed_r2.52", "member_aligned_tight_r2.52"):
            self.assertGreater(events[case]["guided_steps"], 0, case)
        for case in ("aligned_tight", "member_aligned_bed_r2.52", "member_aligned_tight_r2.52",
                     "member_capillary_bed_r2.52"):
            self.assertGreater(events[case]["anastomosis_bridges_arteriovenous"], 0, case)
        # the fill adds moves to the program; rungs are made where they are asked for and nowhere else
        self.assertGreater(facts["fill_stems"]["moves"], facts["stems_unfilled"]["moves"])
        self.assertGreater(facts["fill_free_pair"]["bridges"], 0)
        for case in ("fill_walk_rungs", "rungs_axis", "member_aligned_bed_r2.52", "member_capillary_bed_r2.52"):
            self.assertGreater(events[case]["rung_bridges"], 0, case)
            self.assertEqual(facts[case]["rungs"], events[case]["rung_bridges"], case)
        for case in ("tree_stems", "mesh", "aligned", "aligned_tight", "fill_stems", "fill_free_pair",
                     "member_aligned_tight_r2.52"):
            self.assertEqual(events[case].get("rung_sites", 0), 0, case)
            self.assertEqual(facts[case]["rungs"], 0, case)
        # the plan: every family 3.6 offers
        self.assertEqual(sorted(set(facts["plan_3_6"]["families"])),
                         sorted(["tree", "mesh", "tumour", "aligned", "aligned_tight", "aligned_bed", "capillary_bed"]))
        # the command line: the fill, its rungs and aligned_tight were grown, with voxels
        for case in ("cli_tree", "cli_fill", "cli_aligned_tight"):
            for stem, entry in facts[case].items():
                self.assertGreater(entry["voxels"], 0, stem)
                self.assertIn(f"_s{entry['seed']}", stem)
        (filled,), (tight,) = facts["cli_fill"].values(), facts["cli_aligned_tight"].values()
        self.assertEqual((filled["trees"], tight["trees"]), (1, 2))
        self.assertGreater(filled["rungs"], 0)
        # describe meets every key of 3.6: tree labels, events, flags, the frame-relative keys and loops
        for case in ("describe_cli_tree", "describe_cli_fill"):
            self.assertEqual(facts[case]["events"], len(EVENT_KEYS_3_6), case)
            self.assertEqual(len(facts[case]["flags"]), len(FLAG_KEYS_3_6), case)
        self.assertGreater(facts["describe_margin"]["violations"], 0)
        self.assertGreater(facts["describe_forest"]["components"], 1)
        self.assertGreater(facts["describe_aligned_bed_member"]["junctions"], 0)
        self.assertGreater(facts["describe_aligned_bed_member"]["loops_found"], 0)
        self.assertGreater(facts["describe_forest"]["loops_found"], 0)
        # a loop through every junction of the ring, found only at the full depth of the search
        self.assertEqual(facts["describe_long_loop"]["loops_found_by_node"], 32)
        # the exact Fisher-axial concentration far out in its bracket, on either side
        self.assertGreater(facts["describe_straight_along"]["K_exact"], 100.0)
        self.assertLess(facts["describe_straight_across"]["K_exact"], -100.0)
        self.assertEqual(facts["frames"]["kind"], "axis")
        # the tissue distance is taken from every source it can have
        sources = {"describe_aligned_bed_member": "growth_box", "describe_aligned_bed_member_bound": "argument",
                   "describe_cli_tree": "bounding_box", "describe_cli_fill": "voxel_size",
                   "describe_cli_aligned_tight": "growth_box_um",
                   "describe_library_fill_net_00000_tree.npz": "bounding_box",
                   "describe_library_fill_net_00001_capillary_bed.npz": "growth_box"}
        self.assertEqual({case: facts[case]["tissue_source"] for case in sources}, sources)
        # the growth box comes before a fixed voxel size for the tissue distance, and after it for the volume
        self.assertEqual(facts["describe_aligned_bed_member"]["volume_source"], "voxel_size")
        self.assertEqual(facts["describe_library_fill_net_00001_capillary_bed.npz"]["volume_source"], "bounding_box")
        # joining and the libraries
        self.assertGreater(facts["join_cropped_roots"]["bridges"], 1)
        self.assertGreater(facts["join_cropped_roots"]["root_bridges"], 0)
        self.assertEqual(facts["library"]["files"],
                         ["net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        self.assertEqual(facts["library_fill"]["files"], ["net_00000_tree.npz", "net_00001_capillary_bed.npz"])
        self.assertEqual(facts["library_fill"]["growth"]["box_c"], {"capillary_bed": 10.0})
        # the capillary_bed archive's rungs reach its metadata and the index, as rungs and not as bridges
        rungs = facts["library_fill"]["rungs"]["net_00001_capillary_bed.npz"]
        self.assertGreater(rungs["rungs"], 0)
        self.assertEqual(rungs["rungs"], rungs["metadata_rungs"])
        self.assertNotEqual(rungs["rungs"], rungs["bridges"])
        for case in ("library", "library_fill"):
            self.assertEqual(facts[case]["columns"], list(INDEX_COLUMNS_3_6), case)
            self.assertEqual(facts[case]["failures"], 0, case)

    def test_the_frame_records_reach_every_kind(self):
        # the frame-relative descriptors were computed, about an axis and about a plane
        for case in ("describe_aligned_bed_member", "describe_aligned_bed_member_bound", "describe_banked_plane",
                     "describe_moved_member", "describe_cli_aligned_tight"):
            self.assertIn(case, self.reference["hashes"])
        self.assertEqual(self.reference["facts"]["frames"]["rotations"],
                         sorted(["about", "between", "between_antiparallel", "between_nematic", "of_frames", "random",
                                 "align_axis", "align_normal"]))

    def test_every_counter_appended_after_3_6_is_zero(self):
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

    def test_with_the_later_options_off_what_3_7_added_is_off(self):
        # 3.7 appends the explicit bed: with it off, every record holds the
        # shaping key bed as None, no run records the bed's stream, no bed
        # counter counts and nothing else is returned, stored or recorded
        import bed
        import library
        import numpy as np
        extras = self.current["extras"]
        grown = [case for case in extras if isinstance(extras[case].get("returned"), dict)]
        self.assertGreater(len(grown), 10)
        for case in grown:
            with self.subTest(case=case):
                self.assertNotIn("bed", extras[case]["returned"])
                if "kwargs" in extras[case]:
                    self.assertIn("bed", extras[case]["kwargs"])
                    self.assertIsNone(extras[case]["kwargs"]["bed"])
        for case in ("cli_tree", "cli_fill", "cli_aligned_tight"):
            for stem, entry in extras[case].items():
                with self.subTest(case=case, stem=stem):
                    for record in (entry["sidecar"], entry["metadata"]):
                        self.assertIn("bed", record)
                        self.assertIsNone(record["bed"])
                    self.assertNotIn("bed", entry["rng_streams"])
                    self.assertEqual(entry["arrays"], [])
        for case in ("library", "library_fill"):
            with self.subTest(case=case):
                # an archive's grow_kwargs is pinned to its 3.6 keys, so its bed is read from the archive itself
                for name in self.current["facts"][case]["files"]:
                    with np.load(os.path.join(self.scratch, "current", case, name), allow_pickle=False) as handle:
                        grow_kwargs = json.loads(handle["metadata"].item())["grow_kwargs"]
                    self.assertIn("bed", grow_kwargs, name)
                    self.assertIsNone(grow_kwargs["bed"], name)
                # a library listing no family with a bed records no cap on one
                self.assertNotIn("bed_max_candidates", self.current["facts"][case]["growth"])
                self.assertEqual(extras[case]["library_version"], library.LIBRARY_VERSION)

        # every bed counter is listed wherever the events are recorded, and none counts
        def events_of(value):
            # the "events" records of a case's extras, at any depth; a library's are kept by archive
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "events" and all(not isinstance(count, dict) for count in item.values()):
                        yield item
                    elif key == "events":
                        yield from item.values()
                    else:
                        yield from events_of(item)

        listed = 0
        for case, entry in extras.items():
            with self.subTest(case=case):
                for events in events_of(entry):
                    counted = {key: count for key, count in events.items() if key.startswith("bed_")}
                    if counted:
                        self.assertEqual(counted, dict.fromkeys(bed.EVENT_KEYS, 0))
                        listed += 1
        self.assertGreater(listed, len(grown))
        # a stage that creates a generator while off fails the hash of the generators each case created
        for case, hashes in self.current["hashes"].items():
            if isinstance(hashes, dict) and "stage_generators" in hashes:
                with self.subTest(case=case):
                    self.assertEqual(hashes["stage_generators"], self.reference["hashes"][case]["stage_generators"])

    def test_the_released_presets_are_unchanged(self):
        import main
        for family, released in (("tree", {}), ("mesh", MESH_3_6), ("tumour", TUMOUR_3_6), ("aligned", ALIGNED_3_6),
                                 ("aligned_tight", ALIGNED_TIGHT_3_6), ("aligned_bed", ALIGNED_BED_3_6),
                                 ("capillary_bed", CAPILLARY_BED_3_6)):
            with self.subTest(family=family):
                self.assertEqual(json.loads(json.dumps(main.FAMILIES[family])), released)

    def test_the_recorded_3_6_hashes_are_matched(self):
        # a case passes when it matches a recording; a recording that the 3.6
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
