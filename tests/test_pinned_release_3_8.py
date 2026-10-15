"""
Pins what 3.8 grows and gathers into libraries.

The 3.8 modules (tests/fixtures/reference_code_3_8: main's import closure as
released in 3.8.0, bed included, with describe, join, library and frames)
and the current ones each run one driver in a subprocess, in the same
environment, and the sha256 of everything the driver produces are compared.
The earlier pins (tests/test_pinned_release_3_4.py to _3_7.py) cover what
those releases drew, built and measured in full; this one pins the library
of 3.8 on every family it offers:

    member_argv    the command line of every family, at the defaults and
                   with each library option changed (d_min and margin, the
                   mesh box, a family's box, the iteration cap, collision
                   avoidance off);
    growth         the manifest's growth settings of the default library,
                   of one listing foam and of one listing every family with
                   boxes and a cap given;
    library        grow_member for every family 3.8 offers: tree, mesh,
                   tumour and aligned at R 3, aligned_tight, aligned_bed and
                   capillary_bed at 2.52, foam and fibre at 4, whose beds
                   then attach to the tips of their feeder trees, with argv,
                   kwargs, nodes, program and index row, and a plan of every
                   family;
    library.main   the default three-network library, and one of foam,
                   capillary_bed and aligned_bed, each grown and then
                   resumed with its index removed, so that every row is
                   rebuilt from the archives the plan keeps: every archive's
                   arrays and metadata, the index rows and the manifest
                   content.

Every record (events, kwargs, bed settings, frame records and their rules,
bridge and rung records, feeder trees, index rows, archive metadata, the
manifest) is
first restricted, recursively, to the keys 3.8 wrote, which are listed
literally below (index.csv's events cell to the 3.8 counters), so that what a later version appends never enters the
comparison while everything 3.8 wrote must still be there, unchanged; a list
of dicts, such as a frame's rules or the rungs, is restricted entry by
entry, and a bed's settings, where a record holds None for no bed, keep the
None. A run's seconds and peak resident set (index columns and archive
metadata, UNPINNED_COLUMNS) and the manifest's library_version and
code_sha256 are left out. The global generators start from a fixed state,
and their states after each case and the states of the generators each stage
created are compared too, so a case that seeds nothing must draw nothing and a
stage added later must create no generator while its option is off.

Running both in one environment is the only portable comparison: the
B-spline, the walk and the collision checks go through kernels whose last
bits differ between machines. Hashes recorded on a machine
(tests/fixtures/reference_hashes_3_8*.json, one file per recording: Linux
x86_64 with OpenBLAS's SkylakeX kernels and numpy's AVX-512 loops) are
checked as well: a case passes when it matches a recording, and the check is
skipped only when the reference code itself reproduces none of them here.
The reference directory is an unchanged copy of the release: the sha256 of
each of its files is recorded with the hashes and checked, and the drivers
write nothing but to a temporary directory outside the repository. The two
drivers run side by side, each in its own process and output directory.

Run from the repository root with
    python -m unittest tests.test_pinned_release_3_8
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
REFERENCE = os.path.join(FIXTURES, "reference_code_3_8")
RECORDINGS = "reference_hashes_3_8"           # tests/fixtures/reference_hashes_3_8*.json, one per recording

REFERENCE_MODULES = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance", "connections", "bed",
                     "describe", "join", "library", "frames")

# ---------------------------------------------------------------------------
# The keys 3.8 wrote, listed literally so that a later version's additions
# never enter the comparison.
# ---------------------------------------------------------------------------
# the counters the bed reports, appended to main's; those from bed_proposals_outside on are 3.8's fibre counters
BED_EVENT_KEYS_3_8 = ("bed_proposals", "bed_proposals_on_feeders", "bed_proposals_overlapping", "bed_seeds",
                      "bed_tips", "bed_tips_inside_junction",
                      "bed_candidates", "bed_candidates_tip",
                      "bed_refused_degree", "bed_refused_angle", "bed_refused_girth",
                      "bed_refused_clearance", "bed_refused_clearance_at_end", "bed_edges",
                      "bed_edges_dead_end", "bed_edges_island", "bed_edges_dropped",
                      "bed_seeds_dead_end", "bed_seeds_island",
                      "bed_segments", "bed_segments_walked", "bed_segments_chord", "bed_segments_dropped",
                      "bed_segments_removed", "bed_walk_redraws",
                      "bed_tips_attached", "bed_tip_edges", "bed_components", "bed_components_one_tree",
                      "bed_proposals_outside",
                      "bed_knots", "bed_knots_boundary", "bed_knots_on_feeders",
                      "bed_lane_candidates", "bed_lane_edges_on_feeders", "bed_breaks", "bed_lane_edges",
                      "bed_link_sites", "bed_link_sites_no_partner", "bed_link_sites_angle", "bed_link_sites_on_feeders",
                      "bed_candidates_link", "bed_links", "bed_links_kept",
                      "bed_knots_dead_end", "bed_knots_island")
EVENT_KEYS_3_8 = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
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
                  "rung_kin_skipped", "rung_angle_skipped", "rung_bridge_redraws") + BED_EVENT_KEYS_3_8
SHAPING_KEYS_3_8 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                    "collision_attempts", "collision_index", "anastomose", "anastomosis_radius",
                    "anastomosis_fraction", "anastomose_mode", "anastomosis_min_separation",
                    "guidance", "root_offsets",
                    "capillary_generations", "capillary_runs", "cross_connect", "rung_below", "rung_spacing",
                    "rung_radius", "rung_lateral_deg", "rung_axis", "rung_min_separation", "bed")
KWARGS_KEYS_3_8 = ("tVol", "fit", "clip_axes", "voxel_size", "subdivisions", "d_min",
                   "grow_in_volume", "seed") + SHAPING_KEYS_3_8
PROPERTY_KEYS_3_8 = ("k", "epsilon", "randmarg", "sigma", "stochparams", "roll_angle", "stem_angle",
                     "aneurysm_prob", "stenosis_prob")
# a bed's settings, normalised, wherever a record holds them (None where there is no bed): foam's
# thirteen keys and fibre's twenty-one, of which stretch is foam's alone and the last nine fibre's
BED_SETTING_KEYS_3_8 = ("kind", "spacing", "girth", "min_angle", "reach", "diameter", "persistence",
                        "feeder_stop", "tip_edges", "stretch", "density", "jitter", "step",
                        "axis", "knot", "sway", "link_spacing", "link_reach", "run", "layout", "unit", "sheet_length")
# what grow_network returned beside its arrays, programs, bridges, rungs and events; feeder_trees for a sheets layout
GROWN_KEYS_3_8 = ("program", "growth_box_um", "root_positions_um", "clip_axes", "collision_index", "frame",
                  "feeder_trees")
FEEDER_TREE_KEYS_3_8 = ("label", "role", "sheet", "root")
# an anastomosis bridge, as grow_network returns it and a library archive's metadata stores it
BRIDGE_FIELDS_3_8 = ("tip", "partner", "partner_kind", "tip_tree", "partner_tree", "chord_um", "arc_um", "diameter_um",
                     "points", "redraws")
# a rung, as grow_network returns it and a library archive's metadata stores it
RUNG_FIELDS_3_8 = ("site", "partner", "site_tree", "partner_tree", "chord_um", "arc_um", "diameter_um", "points",
                   "redraws", "angle_deg")
# the frame record and the keys of a rule of any field, as given or normalised
FRAME_KEYS_3_8 = ("frame_version", "kind", "axis", "sense", "tree_senses", "normal", "origin",
                  "grow_direction", "grow_perpendicular", "rules")
RULE_KEYS_3_8 = ("below", "field", "axis", "normal", "sense", "polarity", "length", "onset", "bank")
MEMBER_KEYS_3_8 = ("argv", "niter", "d0")
PLAN_KEYS_3_8 = ("id", "family", "family_index", "position", "seed", "ratio", "bin", "u", "file")
# index.csv's columns; a run's seconds and peak memory are left out of the comparison
INDEX_COLUMNS_3_8 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points",
                     "polylines", "tips", "junctions", "cycles", "components", "total_length",
                     "diameter_p50", "diameter_p90", "diameter_p99",
                     "diameter_v50", "diameter_v90", "diameter_v99", "diameter_min", "diameter_max",
                     "min_point_diameter", "clearance_min", "clearance_violations", "bridges", "events",
                     "avoid_collisions", "ratio_law", "ratio_lo", "ratio_hi", "seconds", "peak_rss_mb",
                     "nodes_sha256", "program_sha256",
                     "frame_kind", "capillary_order", "larger_order", "capillary_polar_order",
                     "capillary_length_share", "capillary_volume_share", "capillary_segment_median",
                     "transverse_spacing_median", "rungs",
                     "capillary_crossing_density", "capillary_crossing_ratio", "cross_connections_per_length")
UNPINNED_COLUMNS = ("seconds", "peak_rss_mb")
# a library archive's metadata; "rungs" is the list of rungs made, "feeder_trees" a sheets layout's, and the
# run's seconds and peak memory are left out as the index's are
METADATA_KEYS_3_8 = ("units", "library_id", "family", "ratio", "bin", "u", "seed", "ratio_law",
                     "ratio_range", "argv", "niter", "generations", "d0", "properties", "grow_kwargs",
                     "avoid_collisions", "events", "root_columns", "growth_box", "root_positions", "bridges",
                     "min_point_diameter", "programs", "archive_version", "seconds", "peak_rss_mb", "frame", "d_min",
                     "rungs", "feeder_trees", "nodes_sha256", "program_sha256")
ARCHIVE_ARRAYS_3_8 = ("nodes", "program", "metadata", "edges", "node_kind", "tree")
# the manifest's content, its growth settings (box_c is absent when no box family but mesh is listed, and
# bed_max_candidates when no family listed has a bed), an entry of its networks and of its failures, whose
# seconds are left out as a run's are
MANIFEST_KEYS_3_8 = ("version", "library_version", "units", "seed", "count", "families", "family_shares",
                     "family_counts", "ratio_law", "ratio_range", "growth", "rng_streams", "code_sha256", "networks",
                     "failures")
UNPINNED_MANIFEST_KEYS = ("code_sha256", "library_version")
GROWTH_KEYS_3_8 = ("collision_margin", "mesh_box_c", "iteration_cap", "avoid_collisions", "d_min", "box_c",
                   "bed_max_candidates")
MANIFEST_NETWORK_KEYS_3_8 = ("family", "seed", "ratio", "bin", "u", "file", "nodes_sha256", "program_sha256")
FAILURE_KEYS_3_8 = ("id", "family", "seed", "ratio", "reason")
LIBRARY_RNG_STREAMS_3_8 = ("ratio",)
MAIN_RNG_STREAMS_3_8 = ("walk", "anastomosis", "rungs")
INDEX_KEYS_3_8 = ("units", "ratio_law", "networks")
# the box families and their constants, and the families a library holds by default
BOX_C_3_8 = {"mesh": 15.0, "aligned": 15.0, "aligned_tight": 20.0, "aligned_bed": 20.0, "capillary_bed": 15.0,
             "foam": 15.0, "fibre": 15.0}
DEFAULT_FAMILIES_3_8 = ("tree", "mesh", "tumour")
# the presets as 3.8.0 released them, held to the current ones by test_the_released_presets_are_unchanged
TREE_3_8 = {}
MESH_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
            "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.5, "grow_in_volume": True}
TUMOUR_3_8 = {"tortuosity": "walk", "persistence": 3.0, "avoid_collisions": True, "anastomose": True,
              "anastomose_mode": "any", "anastomosis_fraction": 0.8, "d0": [20.0, 10.0], "aneurysm_prob": 0.1,
              "stenosis_prob": 0.1}
ALIGNED_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
               "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
               "root_offsets": [[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
               "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                             "polarity": "partner", "length": 4.9, "onset": 2.0},
                            {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0}]}
ALIGNED_TIGHT_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                     "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                     "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                   "polarity": "partner", "length": 2.0, "onset": 0.0},
                                  {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                   "onset": 0.0}]}
ALIGNED_BED_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                   "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                   "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                   "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                 "polarity": "partner", "length": 2.0, "onset": 0.0},
                                {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                 "onset": 0.0}],
                   "capillary_generations": 2, "capillary_runs": 4,
                   "cross_connect": True, "rung_below": 2.0, "rung_spacing": 40.0, "rung_radius": 8.0,
                   "rung_lateral_deg": 60.0, "rung_axis": [1, 0, 0], "rung_min_separation": 2}
CAPILLARY_BED_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 1.0, "grow_in_volume": True,
                     "capillary_generations": 3, "capillary_runs": 2,
                     "cross_connect": True, "rung_below": 2.0, "rung_spacing": 20.0, "rung_radius": 8.0,
                     "rung_lateral_deg": 60.0, "rung_axis": None, "rung_min_separation": 2}
FOAM_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "grow_in_volume": True,
            "bed": {"kind": "foam", "spacing": 7.5, "girth": 8, "min_angle": 60.0, "reach": 2.5,
                    "diameter": 1.0, "persistence": 8.0, "feeder_stop": 2.0, "tip_edges": 2,
                    "stretch": 1.0, "density": 0.5, "jitter": 0.25, "step": 0.5}}
FIBRE_3_8 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "grow_in_volume": True,
             "bed": {"kind": "fibre", "spacing": 5.0, "girth": 8, "min_angle": 60.0, "reach": 2.0, "diameter": 1.0,
                     "persistence": 16.0, "feeder_stop": 2.0, "tip_edges": 2, "density": 0.5, "jitter": 0.25,
                     "step": 0.5, "axis": [1.0, 0.0, 0.0], "knot": 4.0, "sway": 1.0, "link_spacing": 20.0,
                     "link_reach": 1.75, "run": None, "layout": "sheets", "unit": 60.0, "sheet_length": 2.0}}
FAMILIES_3_8 = {"tree": TREE_3_8, "mesh": MESH_3_8, "tumour": TUMOUR_3_8, "aligned": ALIGNED_3_8,
                "aligned_tight": ALIGNED_TIGHT_3_8, "aligned_bed": ALIGNED_BED_3_8,
                "capillary_bed": CAPILLARY_BED_3_8, "foam": FOAM_3_8, "fibre": FIBRE_3_8}


def _whole(keys):
    return {key: None for key in keys}


# How each record is restricted before it is hashed. None keeps a value whole,
# a dict keeps the listed keys and restricts each value in turn (a value that
# is not a dict, such as None for no bed, is kept as it is), {"*": rule} keeps
# every key (a label or an id, not a field) and restricts each value, and
# {"[]": rule} restricts every entry of a list.
RULE_RULE = {"[]": _whole(RULE_KEYS_3_8)}
BRIDGE_RULE = {"[]": _whole(BRIDGE_FIELDS_3_8)}
RUNG_RULE = {"[]": _whole(RUNG_FIELDS_3_8)}
FEEDER_RULE = {"[]": _whole(FEEDER_TREE_KEYS_3_8)}
FRAME_RULE = dict(_whole(FRAME_KEYS_3_8), rules=RULE_RULE)
BED_RULE = _whole(BED_SETTING_KEYS_3_8)
KWARGS_RULE = dict(_whole(KWARGS_KEYS_3_8), guidance=RULE_RULE, bed=BED_RULE)
EVENTS_RULE = _whole(EVENT_KEYS_3_8)
GROWTH_RULE = _whole(GROWTH_KEYS_3_8)
RULES = {
    "events": EVENTS_RULE,
    "grown": dict(_whole(GROWN_KEYS_3_8), frame=FRAME_RULE, feeder_trees=FEEDER_RULE),
    "bridges": BRIDGE_RULE,
    "rungs": RUNG_RULE,
    "kwargs": KWARGS_RULE,
    "member": dict(_whole(MEMBER_KEYS_3_8), properties=_whole(PROPERTY_KEYS_3_8)),
    "plan": _whole(PLAN_KEYS_3_8),
    "growth": GROWTH_RULE,
    "index_row": dict(_whole(column for column in INDEX_COLUMNS_3_8 if column not in UNPINNED_COLUMNS),
                      events=EVENTS_RULE),
    "metadata": dict(_whole(key for key in METADATA_KEYS_3_8 if key not in UNPINNED_COLUMNS),
                     properties=_whole(PROPERTY_KEYS_3_8), grow_kwargs=KWARGS_RULE, events=EVENTS_RULE,
                     frame=FRAME_RULE, bridges=BRIDGE_RULE, rungs=RUNG_RULE, feeder_trees=FEEDER_RULE),
    "manifest": dict(_whole(key for key in MANIFEST_KEYS_3_8 if key not in UNPINNED_MANIFEST_KEYS),
                     growth=GROWTH_RULE, rng_streams=_whole(LIBRARY_RNG_STREAMS_3_8),
                     networks={"*": _whole(MANIFEST_NETWORK_KEYS_3_8)}, failures={"[]": _whole(FAILURE_KEYS_3_8)}),
    "index": _whole(INDEX_KEYS_3_8),
    "main_rng_streams": _whole(MAIN_RNG_STREAMS_3_8),
    "library_rng_streams": _whole(LIBRARY_RNG_STREAMS_3_8),
    "box_c": _whole(BOX_C_3_8),
}

# Plain lists the driver reads beside the rules.
LISTS = {
    "modules": list(REFERENCE_MODULES),
    "archive_arrays": list(ARCHIVE_ARRAYS_3_8),
    "families": list(FAMILIES_3_8),
    "box_families": list(BOX_C_3_8),
    "index_columns": list(INDEX_COLUMNS_3_8),
    "unpinned_columns": list(UNPINNED_COLUMNS),
    "manifest_keys": list(MANIFEST_KEYS_3_8),
    "growth_keys": list(GROWTH_KEYS_3_8),
    "metadata_keys": list(METADATA_KEYS_3_8),
}

# Runs in a subprocess with the modules of one code directory first on the
# path and that directory as the working directory, and prints, per case,
# the hashes of what those modules produce ("hashes"), a few plain counts of
# the same output ("facts") and whatever the output holds beyond the 3.8 keys
# ("extras"). It uses nothing but what 3.8 offers.
DRIVER = r"""
import contextlib, csv, hashlib, importlib, io, json, os, random, sys
code_dir, out_dir = sys.argv[1], sys.argv[2]
rules, lists = json.loads(sys.argv[3]), json.loads(sys.argv[4])
sys.path.insert(0, code_dir)
import numpy as np
import library
import main

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
# the global generators start from a fixed state, so that a case that seeds
# neither (the command lines, the growth settings) is pinned to draw nothing from them
random.seed(20251)
np.random.seed(20251)


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
    # the 3.8 part of a record; see RULES in the test
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
    # what a record holds beyond its 3.8 keys, one level deep
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


hashes, facts, extras = {}, {}, {}

# --- the constants a library is planned and grown with ---------------------
hashes["constants"] = {
    "main_rng_streams": record_sha(restrict(main.RNG_STREAMS, rules["main_rng_streams"])),
    "library_rng_streams": record_sha(restrict(library.RNG_STREAMS, rules["library_rng_streams"])),
    "box_c": record_sha(restrict(library.BOX_C, rules["box_c"])),
    "default_families": record_sha(list(library.DEFAULT_FAMILIES)),
    "index_columns": record_sha(list(library.INDEX_COLUMNS[:len(lists["index_columns"])])),
    "defaults": record_sha([library.DEFAULT_RATIO_RANGE, library.DEFAULT_MESH_BOX_C, library.DEFAULT_ITERATION_CAP,
                            library.DEFAULT_BED_MAX_CANDIDATES, library.DEFAULT_COLLISION_MARGIN,
                            list(library.VOLUME), library.RATIO_LAW, library.UNITS]),
    "global_generators": global_generators(), "stage_generators": stage_generators()}
facts["constants"] = {"families": library.available_families()[:len(lists["families"])],
                      "index_columns": list(library.INDEX_COLUMNS[:len(lists["index_columns"])])}
extras["constants"] = {"index_columns": list(library.INDEX_COLUMNS[len(lists["index_columns"]):]),
                       "main_rng_streams": beyond(main.RNG_STREAMS, rules["main_rng_streams"]),
                       "library_rng_streams": beyond(library.RNG_STREAMS, rules["library_rng_streams"])}

# --- member_argv over every family and each library option -------------------
OPTIONS = (("defaults", {}), ("unit_2", {"d_min": 2.0, "collision_margin": 2.0}), ("mesh_box_12", {"mesh_box_c": 12.0}),
           ("cap_40", {"iteration_cap": 40}), ("no_avoidance", {"avoid_collisions": False}),
           ("margin_0.5", {"collision_margin": 0.5}))
argvs = {}
for family in lists["families"]:
    for label, options in OPTIONS:
        for ratio in (2.52, 7.25):
            argvs[f"{family} {label} {ratio:g}"] = library.member_argv(family, ratio, **options)
    if family in lists["box_families"]:
        argvs[f"{family} box_11"] = library.member_argv(family, 4.0, box_c={family: 11.0})
hashes["command_lines"] = {"argv": record_sha(argvs), "global_generators": global_generators(),
                         "stage_generators": stage_generators()}
facts["command_lines"] = {"argv": argvs}

# --- the manifest's growth settings --------------------------------------------
GROWTH = (("growth_default", [], None),
          ("growth_foam", ["--families", "foam"], None),
          ("growth_all", ["--families"] + lists["families"] + ["--collision-margin", "0.5", "--mesh-box-c", "12",
                                                                "--iteration-cap", "40", "--bed-max-candidates", "3e6",
                                                                "--no-avoid-collisions"],
           {"aligned": 12.0, "foam": 11.0}))
for name, argv, box_c in GROWTH:
    args = library.build_parser().parse_args(["--out", "x", "--count", "1", "--seed", "1"] + argv)
    settings = library.growth_settings(args, box_c)
    hashes[name] = {"growth": record_sha(restrict(settings, rules["growth"])),
                    "global_generators": global_generators(), "stage_generators": stage_generators()}
    facts[name] = {"growth": restrict(settings, rules["growth"])}
    extras[name] = {"growth": beyond(settings, rules["growth"])}

# --- library members of every family, their index rows and a plan ----------
# seeds on both sides of 2 ** 31, as a library's own seeds are; the filled and
# tight families at the smallest ratio of the default range, and the two bed
# families at 4, whose beds then attach to the feeders' tips
MEMBERS = (("tree", 3.0, 101), ("mesh", 3.0, 2147483649), ("tumour", 3.0, 5), ("aligned", 3.0, 107),
           ("aligned_tight", 2.52, 3), ("aligned_bed", 2.52, 3), ("capillary_bed", 2.52, 2), ("foam", 4.0, 5),
           ("fibre", 4.0, 5))
for family, ratio, seed in MEMBERS:
    name = f"member_{family}_r{ratio:g}"
    result = library.grow_member(family, ratio, seed)
    grown = result["grown"]
    row = library.describe_member(grown, 1.0)
    hashes[name] = {"argv": record_sha(result["argv"]),
                    "kwargs": record_sha(restrict(result["kwargs"], rules["kwargs"])),
                    "member": record_sha(restrict(result, rules["member"])),
                    "nodes": array_sha(grown["nodes"]), "tree": array_sha(grown["tree"]),
                    "edges": array_sha(grown["edges"]), "node_kind": array_sha(grown["node_kind"]),
                    "program": record_sha(grown["program"]), "programs": record_sha(grown["programs"]),
                    "bridges": record_sha(restrict(grown["bridges"], rules["bridges"])),
                    "rungs": record_sha(restrict(grown["rungs"], rules["rungs"])),
                    "events": record_sha(restrict(grown["events"], rules["events"])),
                    "record": record_sha(restrict(grown, rules["grown"])),
                    "row": record_sha(restrict(row, rules["index_row"])),
                    "global_generators": global_generators(), "stage_generators": stage_generators()}
    facts[name] = {"argv": result["argv"], "columns": int(grown["nodes"].shape[1]), "trees": len(grown["programs"]),
                   "bridges": len(grown["bridges"]), "rungs": len(grown["rungs"]),
                   "feeder_trees": len(grown.get("feeder_trees") or []),
                   "bed": restrict(result["kwargs"]["bed"], rules["kwargs"]["bed"]),
                   "events": {key: grown["events"][key] for key in rules["events"] if grown["events"].get(key)}}
    known = set(rules["grown"]) | {"nodes", "tree", "edges", "node_kind", "programs", "bridges", "rungs", "events"}
    extras[name] = {"kwargs": beyond(result["kwargs"], rules["kwargs"]),
                    "bridges": sorted({key for bridge in grown["bridges"]
                                       for key in beyond(bridge, rules["bridges"]["[]"])}),
                    "member": sorted(set(result) - set(rules["member"]) - {"grown", "kwargs"}),
                    "returned": sorted(set(grown) - known), "events": beyond(grown["events"], rules["events"]),
                    "row": sorted(beyond(row, rules["index_row"])),
                    "bed": sorted(beyond(result["kwargs"]["bed"] or {}, rules["kwargs"]["bed"]))}

# a plan of every family 3.8 offers, which leaves a remainder
PLANS = (("plan_3_8", 9, 11, tuple(lists["families"]), (1,) * len(lists["families"])),)
for name, seed, count, families, shares in PLANS:
    plan = library.plan_library(seed, count, families, shares, (2.52, 25.0))
    hashes[name] = {"members": record_sha([restrict(member, rules["plan"]) for member in plan]),
                    "global_generators": global_generators(), "stage_generators": stage_generators()}
    facts[name] = {"members": len(plan), "families": [member["family"] for member in plan]}
    extras[name] = {"members": sorted({key for member in plan for key in beyond(member, rules["plan"])})}


# --- small libraries, grown and then resumed with their index removed ---------
def archive_hashes(path):
    # the arrays 3.8 stored, as stored (read with numpy, not with the loader under test)
    with np.load(path, allow_pickle=False) as handle:
        stored = {name: handle[name] for name in handle.files}
    entry = {}
    for name in lists["archive_arrays"]:
        if name == "program":
            entry[name] = record_sha(stored[name].item())
        elif name == "metadata":
            entry[name] = record_sha(restrict(json.loads(stored[name].item()), rules["metadata"]))
        else:
            entry[name] = array_sha(stored[name])
    return entry, sorted(set(stored) - set(lists["archive_arrays"])), json.loads(stored["metadata"].item())


def library_run(directory, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        status = library.main(["--out", directory] + argv)
    with open(os.path.join(directory, "manifest.json")) as handle:
        manifest = json.load(handle)
    with open(os.path.join(directory, "index.json")) as handle:
        index = json.load(handle)
    with open(os.path.join(directory, "index.csv"), newline="") as handle:
        table = list(csv.reader(handle))
    # index.csv restricted to the 3.8 columns, which a later version keeps first and in order
    pinned = [k for k, column in enumerate(table[0])
              if column in lists["index_columns"] and column not in lists["unpinned_columns"]]
    events = table[0].index("events")
    for line in table[1:]:
        # the events cell, "key=value" for every counter that is not zero, kept to the 3.8 counters
        line[events] = " ".join(item for item in line[events].split() if item.split("=")[0] in rules["events"])
    content = manifest["content"]
    entry = {"status": status, "files": sorted(os.listdir(directory)),
             "content": record_sha(restrict(content, rules["manifest"])),
             "index": record_sha(dict(restrict(index, rules["index"]),
                                      networks=[restrict(row, rules["index_row"]) for row in index["networks"]])),
             "index_csv": record_sha([[line[k] for k in pinned] for line in table])}
    last = manifest["run"]["runs"][-1]
    return entry, content, index, table, {key: last[key] for key in ("grown", "kept", "failed")}


LIBRARIES = (("library", ["--count", "3", "--seed", "1", "--ratio-range", "2.52", "3.5", "--workers", "1"]),
             # the bed family beside the two filled families, each in a cube of its own constant
             ("library_bed", ["--count", "3", "--seed", "3", "--families", "foam", "capillary_bed", "aligned_bed",
                              "--ratio-range", "2.52", "3.0", "--workers", "1"]))
for case, argv in LIBRARIES:
    directory = os.path.join(out_dir, case)
    entry, content, index, table, run = library_run(directory, argv)
    entry.update(global_generators=global_generators(), stage_generators=stage_generators())
    extra = {"rows": sorted({key for row in index["networks"] for key in beyond(row, rules["index_row"])}),
             "index": sorted(beyond(index, rules["index"])),
             "columns": table[0][len(lists["index_columns"]):],
             "manifest": sorted(set(content) - set(lists["manifest_keys"])),
             "growth": sorted(set(content["growth"]) - set(lists["growth_keys"])),
             "row_values": [{key: value for key, value in beyond(row, rules["index_row"]).items()
                             if key not in lists["unpinned_columns"]} for row in index["networks"]],
             "library_version": content["library_version"], "metadata": {}, "metadata_values": {}, "arrays": {},
             "events": {}, "beds": {}}
    metadata_by_file = {}
    for row in index["networks"]:
        entry[row["file"]], extra["arrays"][row["file"]], metadata = archive_hashes(os.path.join(directory, row["file"]))
        metadata_by_file[row["file"]] = metadata
        extra["metadata"][row["file"]] = sorted(beyond(metadata, rules["metadata"]))
        extra["metadata_values"][row["file"]] = {key: value
                                                 for key, value in beyond(metadata, rules["metadata"]).items()
                                                 if key not in lists["unpinned_columns"]}
        extra["events"][row["file"]] = beyond(metadata["events"], rules["events"])
        extra["beds"][row["file"]] = sorted(beyond(metadata["grow_kwargs"]["bed"] or {},
                                                   rules["metadata"]["grow_kwargs"]["bed"]))
    hashes[case] = entry
    facts[case] = {"files": [row["file"] for row in index["networks"]], "families": content["families"],
                   "columns": table[0][:len(lists["index_columns"])], "rows": len(table) - 1, "run": run,
                   "growth": restrict(content["growth"], rules["manifest"]["growth"]),
                   "failures": len(content["failures"]),
                   "metadata_keys": {name: [key for key in lists["metadata_keys"] if key in metadata]
                                     for name, metadata in metadata_by_file.items()},
                   "argv": {name: metadata["argv"] for name, metadata in metadata_by_file.items()},
                   "beds": {name: restrict(metadata["grow_kwargs"]["bed"], rules["metadata"]["grow_kwargs"]["bed"])
                            for name, metadata in metadata_by_file.items()},
                   "rungs": {row["file"]: row["rungs"] for row in index["networks"]},
                   "events": {name: {key: value for key, value in restrict(metadata["events"], rules["events"]).items()
                                     if value} for name, metadata in metadata_by_file.items()}}
    extras[case] = extra

    # resumed with the index removed: every archive is kept, its row rebuilt from it, and nothing is grown
    os.remove(os.path.join(directory, "index.json"))
    resumed, content, index, table, run = library_run(directory, argv)
    resumed.update(global_generators=global_generators(), stage_generators=stage_generators())
    hashes[case + "_resumed"] = resumed
    facts[case + "_resumed"] = {"run": run, "content_unchanged": resumed["content"] == entry["content"]}
    extras[case + "_resumed"] = {"rows": sorted({key for row in index["networks"]
                                                 for key in beyond(row, rules["index_row"])})}

print(json.dumps({"hashes": hashes, "facts": facts, "extras": extras}, default=plain))
"""


def start_driver(code_dir, out_dir):
    """Starts the driver on the modules in `code_dir`, writing its files under `out_dir`."""
    os.makedirs(out_dir)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")      # nothing is written beside the modules
    return subprocess.Popen([sys.executable, "-c", DRIVER, code_dir, out_dir, json.dumps(RULES), json.dumps(LISTS)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=code_dir,
                            env=environment)


def finish_driver(process, code_dir):
    stdout, stderr = process.communicate()
    if process.returncode:
        raise AssertionError(f"{code_dir}: {stderr}")
    return json.loads(stdout)


def run_drivers(scratch):
    """
    What the 3.8 modules and the current ones produce, run side by side:
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
    """Every recording of the 3.8 hashes, by file name: tests/fixtures/reference_hashes_3_8*.json."""
    recordings = {}
    for name in sorted(os.listdir(FIXTURES)):
        if name.startswith(RECORDINGS) and name.endswith(".json"):
            with open(os.path.join(FIXTURES, name)) as handle:
                recordings[name] = json.load(handle)
    return recordings


class PinnedRelease38Tests(unittest.TestCase):
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

    def test_the_current_modules_produce_what_the_3_8_modules_produce_in_this_environment(self):
        self.assertEqual(sorted(self.current["hashes"]), sorted(self.reference["hashes"]))
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                self.assertEqual(self.current["hashes"][case], self.reference["hashes"][case])
                self.assertEqual(self.current["facts"][case], self.reference["facts"][case])

    def test_the_cases_reach_the_stages_they_pin(self):
        facts = self.reference["facts"]
        self.assertEqual(facts["constants"]["families"], list(FAMILIES_3_8))
        self.assertEqual(facts["constants"]["index_columns"], list(INDEX_COLUMNS_3_8))
        # member_argv: one per family, option and ratio, and a box of its own for each box family
        self.assertEqual(len(facts["command_lines"]["argv"]), len(FAMILIES_3_8) * 6 * 2 + len(BOX_C_3_8))
        # growth: no box_c nor cap by default, foam's box and cap when listed, and every given value recorded
        self.assertEqual(facts["growth_default"]["growth"],
                         {"collision_margin": 1.0, "mesh_box_c": 15.0, "iteration_cap": 64, "avoid_collisions": True,
                          "d_min": 1.0})
        self.assertEqual(facts["growth_foam"]["growth"]["box_c"], {"foam": 15.0})
        self.assertEqual(facts["growth_foam"]["growth"]["bed_max_candidates"], 2000000)
        self.assertEqual(facts["growth_all"]["growth"],
                         {"collision_margin": 0.5, "mesh_box_c": 12.0, "iteration_cap": 40, "avoid_collisions": False,
                          "d_min": 1.0, "bed_max_candidates": 3000000,
                          "box_c": {"aligned": 12.0, "aligned_tight": 20.0, "aligned_bed": 20.0,
                                    "capillary_bed": 15.0, "foam": 11.0, "fibre": 15.0}})
        # a member of every family 3.8 offers, the bed families' with the released bed and every other with none
        grown = {case[len("member_"):].rsplit("_r", 1)[0]: case for case in facts if case.startswith("member_")}
        self.assertEqual(sorted(grown), sorted(FAMILIES_3_8))
        events = {family: facts[case]["events"] for family, case in grown.items()}
        for family, case in grown.items():
            with self.subTest(family=family):
                self.assertEqual(facts[case]["bed"], FAMILIES_3_8[family].get("bed"))
                self.assertEqual(facts[case]["trees"], 1 if family in ("tree", "tumour") else 2)
                self.assertEqual(facts[case]["argv"][:2], ["--family", family])
                if family not in ("foam", "fibre"):
                    self.assertEqual([key for key in events[family] if key.startswith("bed_")], [])
        # the pairs joined arterial to venous, the aligned families steered and the filled ones cross-connected
        for family in ("mesh", "aligned", "aligned_tight", "aligned_bed", "capillary_bed"):
            self.assertGreater(events[family]["anastomosis_bridges_arteriovenous"], 0, family)
        for family in ("aligned", "aligned_tight", "aligned_bed"):
            self.assertGreater(events[family]["guided_steps"], 0, family)
        for family in ("aligned_bed", "capillary_bed"):
            self.assertGreater(events[family]["rung_bridges"], 0, family)
            self.assertEqual(facts[grown[family]]["rungs"], events[family]["rung_bridges"], family)
        # foam and fibre grow a bed between their feeder trees, attached to their tips, and nothing else joins them
        for family in ("foam", "fibre"):
            for key in ("bed_seeds", "bed_edges", "bed_segments_walked", "bed_tips_attached", "bed_components"):
                self.assertGreater(events[family][key], 0, (family, key))
            self.assertEqual((facts[grown[family]]["bridges"], facts[grown[family]]["rungs"]), (0, 0), family)
            self.assertNotIn("anastomosis_tips", events[family])
        for key in ("bed_knots", "bed_lane_edges", "bed_links"):
            self.assertGreater(events["fibre"][key], 0, key)
        self.assertEqual(facts[grown["fibre"]]["feeder_trees"], 2)
        # the plan: every family 3.8 offers
        self.assertEqual(sorted(set(facts["plan_3_8"]["families"])), sorted(FAMILIES_3_8))
        # the libraries: their files and columns, every 3.8 metadata key, no failure, and their growth settings
        self.assertEqual(facts["library"]["files"],
                         ["net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        self.assertEqual(facts["library_bed"]["files"],
                         ["net_00000_foam.npz", "net_00001_capillary_bed.npz", "net_00002_aligned_bed.npz"])
        for case in ("library", "library_bed"):
            self.assertEqual(facts[case]["columns"], list(INDEX_COLUMNS_3_8), case)
            self.assertEqual(facts[case]["failures"], 0, case)
            self.assertEqual(facts[case]["run"], {"grown": 3, "kept": 0, "failed": 0}, case)
            for name, keys in facts[case]["metadata_keys"].items():
                self.assertEqual(keys, [key for key in METADATA_KEYS_3_8 if key != "feeder_trees"], name)
            # resumed with the index removed: every archive kept, nothing grown and the content unchanged
            self.assertEqual(facts[case + "_resumed"], {"run": {"grown": 0, "kept": 3, "failed": 0},
                                                        "content_unchanged": True}, case)
        self.assertEqual(facts["library"]["growth"], facts["growth_default"]["growth"])
        self.assertEqual(facts["library_bed"]["growth"]["box_c"], {"foam": 15.0, "capillary_bed": 15.0,
                                                                   "aligned_bed": 20.0})
        self.assertEqual(facts["library_bed"]["growth"]["bed_max_candidates"], 2000000)
        # the foam archive records the released bed and its bed attached; the filled archives carry rungs
        self.assertEqual(facts["library_bed"]["beds"], {"net_00000_foam.npz": FOAM_3_8["bed"],
                                                        "net_00001_capillary_bed.npz": None,
                                                        "net_00002_aligned_bed.npz": None})
        self.assertGreater(facts["library_bed"]["events"]["net_00000_foam.npz"]["bed_tips_attached"], 0)
        for name in ("net_00001_capillary_bed.npz", "net_00002_aligned_bed.npz"):
            self.assertGreater(facts["library_bed"]["rungs"][name], 0, name)

    def test_the_released_presets_are_unchanged(self):
        import main
        for family, released in FAMILIES_3_8.items():
            with self.subTest(family=family):
                self.assertEqual(json.loads(json.dumps(main.FAMILIES[family])), released)

    def test_the_recorded_3_8_hashes_are_matched(self):
        # a case passes when it matches a recording; a recording that the 3.8
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
