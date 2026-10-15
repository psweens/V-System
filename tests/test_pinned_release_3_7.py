"""
Pins what 3.7 grows, builds, measures, writes and gathers into libraries.

The 3.7 modules (tests/fixtures/reference_code_3_7: main's import closure as
released in 3.7.0, bed included, with describe, join, library and frames)
and the current ones each run one driver in a subprocess, in the same
environment, and the sha256 of everything the driver produces are compared.
The 3.4, 3.5 and 3.6 pins (tests/test_pinned_release_3_4.py, _3_5.py and
_3_6.py) cover what those releases drew in full; this one grows a member of
every family 3.7 offers, so that what 3.7 added is pinned on each kind of
growth, and adds the cases 3.7 opened:

    library        grow_member for every family 3.7 offers: tree, mesh,
                   tumour and aligned at R 3, aligned_tight, aligned_bed and
                   capillary_bed at 2.52, and foam at 4, whose bed then
                   attaches to the tips of its two feeder trees, with argv,
                   kwargs and index row, and a plan of every family;
    bed.build      a foam bed alone in a cube, the foam member's bed built
                   again from the feeders grow_network gave it, and a bed
                   alone at stretch 2 against an unguided frame: the
                   network, the seeds, the edges the greedy pass accepted,
                   the seeds kept, the tips and their edges, the kept edges,
                   the chains and the bed's counters;
    describe       the three fixture archives, one at a margin, the foam
                   member with the tissue distance over its growth box, the
                   bed at stretch 2 against its axis frame, and the archives
                   below;
    main.main      a foam run at a fixed voxel size whose bed attaches to the
                   feeders' tips: archive, sidecar, TIFF pixels and file;
    library.main   the default three-network library, and one of foam,
                   capillary_bed and aligned_bed.

Every record (events, the bed's counters, sidecars, metadata, kwargs, bed
settings, frame records and their rules, rung records, chain records,
describe output, index rows, the manifest) is first restricted, recursively,
to the keys 3.7 wrote, which are listed literally below, so that what a
later version appends never enters the comparison while everything 3.7 wrote
must still be there, unchanged; a list of dicts, such as a frame's rules, the
rungs or the chains, is restricted entry by entry, and a bed's settings,
where a record holds None for no bed, keep the None. The global generators'
states after each case and the states of the generators each stage created
are compared too, so a stage added later must create no generator while its
option is off, and every counter appended to the events must be zero.

Running both in one environment is the only portable comparison: the
B-spline, the walk and the collision checks go through kernels whose last
bits differ between machines. Hashes recorded on a machine
(tests/fixtures/reference_hashes_3_7*.json, one file per recording: Linux
x86_64 with OpenBLAS's SkylakeX kernels and numpy's AVX-512 loops) are
checked as well: a case passes when it matches a recording, and the check is
skipped only when the reference code itself reproduces none of them here.
The reference directory is an unchanged copy of the release: the sha256 of
each of its files is recorded with the hashes and checked, and the drivers
write nothing but to a temporary directory outside the repository. The two
drivers run side by side, each in its own process and output directory.

Run from the repository root with
    python -m unittest tests.test_pinned_release_3_7
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
REFERENCE = os.path.join(FIXTURES, "reference_code_3_7")
RECORDINGS = "reference_hashes_3_7"           # tests/fixtures/reference_hashes_3_7*.json, one per recording

REFERENCE_MODULES = ("main", "vSystem", "libGenerator", "analyseGrammar", "utils", "computeVoxel",
                     "tortuosity", "collisions", "anastomosis", "graph", "spatial", "guidance", "connections", "bed",
                     "describe", "join", "library", "frames")
ARCHIVES = ("Lnet_i4_s2.npz", "Lnet_i5_s1.npz", "Lnet_i9_s3.npz")

# ---------------------------------------------------------------------------
# The keys 3.7 wrote, listed literally so that a later version's additions
# never enter the comparison.
# ---------------------------------------------------------------------------
# the counters the bed reports, appended to main's
BED_EVENT_KEYS_3_7 = ("bed_proposals", "bed_proposals_on_feeders", "bed_proposals_overlapping", "bed_seeds",
                      "bed_tips", "bed_tips_inside_junction",
                      "bed_candidates", "bed_candidates_tip",
                      "bed_refused_degree", "bed_refused_angle", "bed_refused_girth",
                      "bed_refused_clearance", "bed_refused_clearance_at_end", "bed_edges",
                      "bed_edges_dead_end", "bed_edges_island", "bed_edges_dropped",
                      "bed_seeds_dead_end", "bed_seeds_island",
                      "bed_segments", "bed_segments_walked", "bed_segments_chord", "bed_segments_dropped",
                      "bed_segments_removed", "bed_walk_redraws",
                      "bed_tips_attached", "bed_tip_edges", "bed_components", "bed_components_one_tree")
EVENT_KEYS_3_7 = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
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
                  "rung_kin_skipped", "rung_angle_skipped", "rung_bridge_redraws") + BED_EVENT_KEYS_3_7
SHAPING_KEYS_3_7 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                    "collision_attempts", "collision_index", "anastomose", "anastomosis_radius",
                    "anastomosis_fraction", "anastomose_mode", "anastomosis_min_separation",
                    "guidance", "root_offsets",
                    "capillary_generations", "capillary_runs", "cross_connect", "rung_below", "rung_spacing",
                    "rung_radius", "rung_lateral_deg", "rung_axis", "rung_min_separation", "bed")
KWARGS_KEYS_3_7 = ("tVol", "fit", "clip_axes", "voxel_size", "subdivisions", "d_min",
                   "grow_in_volume", "seed") + SHAPING_KEYS_3_7
PROPERTY_KEYS_3_7 = ("k", "epsilon", "randmarg", "sigma", "stochparams", "roll_angle", "stem_angle",
                     "aneurysm_prob", "stenosis_prob")
# a bed's settings, normalised, wherever a record holds them (None where there is no bed)
BED_SETTING_KEYS_3_7 = ("kind", "spacing", "girth", "min_angle", "reach", "diameter", "persistence",
                        "feeder_stop", "tip_edges", "stretch", "density", "jitter", "step")
# what grow_network returned beside its arrays, programs, bridges, rungs and events
GROWN_KEYS_3_7 = ("program", "growth_box_um", "root_positions_um", "clip_axes", "collision_index", "frame")
# a rung, as grow_network returns it and a library archive's metadata stores it
RUNG_FIELDS_3_7 = ("site", "partner", "site_tree", "partner_tree", "chord_um", "arc_um", "diameter_um", "points",
                   "redraws", "angle_deg")
# what bed.build returns, and a chain of its "chains"
BUILD_KEYS_3_7 = ("nodes", "tree", "frame", "seeds", "greedy_edges", "kept", "tips", "tip_edges", "edges", "chains")
CHAIN_FIELDS_3_7 = ("vertices", "outcome", "first", "last", "closed")
# the frame record and the keys of a rule of any field, as given or normalised
FRAME_KEYS_3_7 = ("frame_version", "kind", "axis", "sense", "tree_senses", "normal", "origin",
                  "grow_direction", "grow_perpendicular", "rules")
RULE_KEYS_3_7 = ("below", "field", "axis", "normal", "sense", "polarity", "length", "onset", "bank")
# a sidecar's "rungs" is the number of rungs made
SIDECAR_KEYS_3_7 = ("seed", "iterations", "d0", "d_min", "volume", "axis_order", "units", "fit",
                    "clip_axes", "connect", "grow_in_volume", "voxel_size", "subdivisions",
                    "archive_version", "family", "trees", "growth_box_um", "root_positions_um",
                    "bridges", "bridge_persistence", "frame", "rungs") + SHAPING_KEYS_3_7
# a bed run's sidecar records the bed's stream beside these
RNG_STREAMS_3_7 = ("walk", "anastomosis", "rungs", "bed")
MEMBER_KEYS_3_7 = ("argv", "niter", "d0")
PLAN_KEYS_3_7 = ("id", "family", "family_index", "position", "seed", "ratio", "bin", "u", "file")
# index.csv's columns; a run's seconds and peak memory are left out of the comparison
INDEX_COLUMNS_3_7 = ("id", "family", "ratio", "bin", "u", "seed", "file", "generations", "points",
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
METADATA_KEYS_3_7 = ("units", "library_id", "family", "ratio", "bin", "u", "seed", "ratio_law",
                     "ratio_range", "argv", "niter", "generations", "d0", "avoid_collisions",
                     "root_columns", "growth_box", "root_positions", "bridges", "min_point_diameter",
                     "programs", "archive_version", "nodes_sha256", "program_sha256", "d_min", "frame", "rungs")
# the manifest's content, its growth settings (box_c is absent when no box family but mesh is listed, and
# bed_max_candidates when no family listed has a bed), an entry of its networks and of its failures, whose
# seconds are left out as a run's are
MANIFEST_KEYS_3_7 = ("version", "library_version", "units", "seed", "count", "families", "family_shares",
                     "family_counts", "ratio_law", "ratio_range", "growth", "rng_streams", "code_sha256", "networks",
                     "failures")
UNPINNED_MANIFEST_KEYS = ("code_sha256", "library_version")
GROWTH_KEYS_3_7 = ("collision_margin", "mesh_box_c", "iteration_cap", "avoid_collisions", "d_min", "box_c",
                   "bed_max_candidates")
MANIFEST_NETWORK_KEYS_3_7 = ("family", "seed", "ratio", "bin", "u", "file", "nodes_sha256", "program_sha256")
FAILURE_KEYS_3_7 = ("id", "family", "seed", "ratio", "reason")
LIBRARY_RNG_STREAMS_3_7 = ("ratio",)
FLAG_KEYS_3_7 = ("tortuosity", "persistence", "avoid_collisions", "collision_margin",
                 "anastomose", "anastomosis_fraction", "anastomosis_radius", "anastomose_mode",
                 "family", "seed", "iterations", "d0", "d_min")
DESCRIBE_KEYS_3_7 = ("units", "points", "polylines", "vertices", "edges", "components", "cycles",
                     "total_length_mm", "degree_histogram", "length_density_mm_per_mm3", "volume_mm3",
                     "volume_source")
ORIENTATION_KEYS_3_7 = ("eigenvalues", "principal_axis", "fractional_anisotropy", "polar_histogram_deg",
                        "azimuth_histogram_deg")
DESCRIBE_GROUPS_3_7 = {
    "diameter_um": ("p50", "p90", "p99", "min", "max"),
    "arc_chord": ("mean", "median", "p90", "max", "n_segments", "n_closed"),
    "curvature_per_um": ("mean_abs", "n_pairs"),
    "tips": ("count", "per_mm"),
    "orientation": ORIENTATION_KEYS_3_7,
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
CLASSES_3_7 = ("all", "capillary", "larger")
# an entry of frame_orientation: about an axis, or about a plane
FRAME_ENTRY_KEYS_3_7 = ("S", "mean_abs_cos", "crossing_ratio", "mean_angle_deg", "within_20_deg", "within_45_deg",
                        "watson_K", "fisher_axial_K", "in_plane_fraction", "S_n", "length", "length_d",
                        "fisher_axial_K_exact")
SEGMENT_KEYS_3_7 = ("count", "median", "p10", "p90", "cv", "median_d", "p10_d", "p90_d")
SHARE_KEYS_3_7 = ("below_1.5", "below_2", "below_3")
PER_TREE_KEYS_3_7 = ("points", "polylines", "tips", "length_mm")
# the shortest loops through sampled segments and junctions; a histogram is kept whole
LOOP_SEGMENT_KEYS_3_7 = ("segments", "every", "sampled", "found", "none_fraction", "median", "mean", "histogram",
                         "cycles", "cycles_per_length", "cycles_per_length_d")
LOOP_NODE_KEYS_3_7 = ("junctions", "every", "sampled", "found", "none_fraction", "median", "mean", "histogram",
                      "length_median", "length_mean", "length_median_d", "length_mean_d")
VARIATION_KEYS_3_7 = ("count", "excluded", "median", "p90")
TISSUE_KEYS_3_7 = ("spacing_requested", "spacing", "raised", "shape", "points", "domain", "capsules",
                   "inside_fraction", "outside_points", "mean", "median", "p90", "p99", "max", "spacing_d", "mean_d",
                   "median_d", "p90_d", "p99_d", "max_d")
TISSUE_DOMAIN_KEYS_3_7 = ("min", "max", "source")
ARCHIVE_ARRAYS_3_7 = ("nodes", "program", "metadata", "edges", "node_kind", "tree")
INDEX_KEYS_3_7 = ("units", "ratio_law", "networks")
# the presets as 3.7.0 released them, held to the current ones by test_the_released_presets_are_unchanged
TREE_3_7 = {}
MESH_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
            "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.5, "grow_in_volume": True}
TUMOUR_3_7 = {"tortuosity": "walk", "persistence": 3.0, "avoid_collisions": True, "anastomose": True,
              "anastomose_mode": "any", "anastomosis_fraction": 0.8, "d0": [20.0, 10.0], "aneurysm_prob": 0.1,
              "stenosis_prob": 0.1}
ALIGNED_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
               "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
               "root_offsets": [[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
               "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                             "polarity": "partner", "length": 4.9, "onset": 2.0},
                            {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0}]}
ALIGNED_TIGHT_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                     "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                     "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                   "polarity": "partner", "length": 2.0, "onset": 0.0},
                                  {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                   "onset": 0.0}]}
ALIGNED_BED_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                   "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.8, "grow_in_volume": True,
                   "root_offsets": [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                   "guidance": [{"below": 2.0, "field": "axis", "axis": [1, 0, 0], "sense": "polar",
                                 "polarity": "partner", "length": 2.0, "onset": 0.0},
                                {"below": None, "field": "plane", "normal": [1, 0, 0], "length": 5.0,
                                 "onset": 0.0}],
                   "capillary_generations": 2, "capillary_runs": 4,
                   "cross_connect": True, "rung_below": 2.0, "rung_spacing": 40.0, "rung_radius": 8.0,
                   "rung_lateral_deg": 60.0, "rung_axis": [1, 0, 0], "rung_min_separation": 2}
CAPILLARY_BED_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "anastomose": True,
                     "anastomose_mode": "arteriovenous", "anastomosis_fraction": 1.0, "grow_in_volume": True,
                     "capillary_generations": 3, "capillary_runs": 2,
                     "cross_connect": True, "rung_below": 2.0, "rung_spacing": 20.0, "rung_radius": 8.0,
                     "rung_lateral_deg": 60.0, "rung_axis": None, "rung_min_separation": 2}
FOAM_3_7 = {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True, "grow_in_volume": True,
            "bed": {"kind": "foam", "spacing": 7.5, "girth": 8, "min_angle": 60.0, "reach": 2.5,
                    "diameter": 1.0, "persistence": 8.0, "feeder_stop": 2.0, "tip_edges": 2,
                    "stretch": 1.0, "density": 0.5, "jitter": 0.25, "step": 0.5}}
FAMILIES_3_7 = {"tree": TREE_3_7, "mesh": MESH_3_7, "tumour": TUMOUR_3_7, "aligned": ALIGNED_3_7,
                "aligned_tight": ALIGNED_TIGHT_3_7, "aligned_bed": ALIGNED_BED_3_7,
                "capillary_bed": CAPILLARY_BED_3_7, "foam": FOAM_3_7}


def _whole(keys):
    return {key: None for key in keys}


# How each record is restricted before it is hashed. None keeps a value whole,
# a dict keeps the listed keys and restricts each value in turn (a value that
# is not a dict, such as None for no bed, is kept as it is), {"*": rule} keeps
# every key (a label or an id, not a field) and restricts each value, and
# {"[]": rule} restricts every entry of a list.
RULE_RULE = {"[]": _whole(RULE_KEYS_3_7)}
RUNG_RULE = {"[]": _whole(RUNG_FIELDS_3_7)}
CHAIN_RULE = {"[]": _whole(CHAIN_FIELDS_3_7)}
FRAME_RULE = dict(_whole(FRAME_KEYS_3_7), rules=RULE_RULE)
BED_RULE = _whole(BED_SETTING_KEYS_3_7)
KWARGS_RULE = dict(_whole(KWARGS_KEYS_3_7), guidance=RULE_RULE, bed=BED_RULE)
RULES = {
    "events": _whole(EVENT_KEYS_3_7),
    "bed_events": _whole(BED_EVENT_KEYS_3_7),
    "grown": dict(_whole(GROWN_KEYS_3_7), frame=FRAME_RULE),
    "rungs": RUNG_RULE,
    "chains": CHAIN_RULE,
    "kwargs": KWARGS_RULE,
    "frame": FRAME_RULE,
    "member": dict(_whole(MEMBER_KEYS_3_7), properties=_whole(PROPERTY_KEYS_3_7)),
    "plan": _whole(PLAN_KEYS_3_7),
    "sidecar": dict(_whole(SIDECAR_KEYS_3_7), properties=_whole(PROPERTY_KEYS_3_7),
                    rng_streams=_whole(RNG_STREAMS_3_7), events=_whole(EVENT_KEYS_3_7), frame=FRAME_RULE,
                    guidance=RULE_RULE, bed=BED_RULE),
    "index_row": dict(_whole(column for column in INDEX_COLUMNS_3_7 if column not in UNPINNED_COLUMNS),
                      events=_whole(EVENT_KEYS_3_7)),
    "metadata": dict(_whole(METADATA_KEYS_3_7), properties=_whole(PROPERTY_KEYS_3_7),
                     grow_kwargs=KWARGS_RULE, events=_whole(EVENT_KEYS_3_7), frame=FRAME_RULE, rungs=RUNG_RULE),
    "describe": dict(_whole(DESCRIBE_KEYS_3_7), events=_whole(EVENT_KEYS_3_7), flags=_whole(FLAG_KEYS_3_7),
                     per_tree={"*": _whole(PER_TREE_KEYS_3_7)}, frame=FRAME_RULE,
                     frame_orientation=dict({cls: _whole(FRAME_ENTRY_KEYS_3_7) for cls in CLASSES_3_7}, kind=None),
                     orientation_by_class={cls: _whole(ORIENTATION_KEYS_3_7 + ("S_max", "planarity"))
                                           for cls in CLASSES_3_7},
                     calibre_shares={"length": _whole(SHARE_KEYS_3_7), "volume": _whole(SHARE_KEYS_3_7)},
                     segments_by_class={cls: _whole(SEGMENT_KEYS_3_7) for cls in ("capillary", "larger")},
                     loops={"by_segment": _whole(LOOP_SEGMENT_KEYS_3_7), "by_node": _whole(LOOP_NODE_KEYS_3_7)},
                     segment_diameter_variation={cls: _whole(VARIATION_KEYS_3_7) for cls in CLASSES_3_7},
                     tissue_distance=dict(_whole(TISSUE_KEYS_3_7), domain=_whole(TISSUE_DOMAIN_KEYS_3_7)),
                     **{group: _whole(keys) for group, keys in DESCRIBE_GROUPS_3_7.items()}),
    "manifest": dict(_whole(key for key in MANIFEST_KEYS_3_7 if key not in UNPINNED_MANIFEST_KEYS),
                     growth=_whole(GROWTH_KEYS_3_7), rng_streams=_whole(LIBRARY_RNG_STREAMS_3_7),
                     networks={"*": _whole(MANIFEST_NETWORK_KEYS_3_7)}, failures={"[]": _whole(FAILURE_KEYS_3_7)}),
    "index": _whole(INDEX_KEYS_3_7),
}

# Plain lists the driver reads beside the rules.
LISTS = {
    "modules": list(REFERENCE_MODULES),
    "archives": list(ARCHIVES),
    "archive_arrays": list(ARCHIVE_ARRAYS_3_7),
    "build_keys": list(BUILD_KEYS_3_7),
    "families": list(FAMILIES_3_7),
    "index_columns": list(INDEX_COLUMNS_3_7),
    "unpinned_columns": list(UNPINNED_COLUMNS),
    "manifest_keys": list(MANIFEST_KEYS_3_7),
    "growth_keys": list(GROWTH_KEYS_3_7),
}
# the bed settings the driver builds with literally, foam's as released; the members are grown by grow_member,
# which reads each side's own presets, and are held to the released ones by test_the_released_presets_are_unchanged
SETTINGS = {"bed": FOAM_3_7["bed"]}

# Runs in a subprocess with the modules of one code directory first on the
# path and that directory as the working directory, and prints, per case,
# the hashes of what those modules produce ("hashes"), a few plain counts of
# the same output ("facts") and whatever the output holds beyond the 3.7 keys
# ("extras"). It uses nothing but what 3.7 offers.
DRIVER = r"""
import contextlib, copy, csv, hashlib, importlib, io, json, os, random, sys
code_dir, out_dir, fixtures = sys.argv[1], sys.argv[2], sys.argv[3]
rules, lists, settings = json.loads(sys.argv[4]), json.loads(sys.argv[5]), json.loads(sys.argv[6])
sys.path.insert(0, code_dir)
import numpy as np
import tifffile
import bed
import describe as describe_module
import guidance
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
    # the 3.7 part of a record; see RULES in the test
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
    # what a record holds beyond its 3.7 keys, one level deep
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
    # the arrays 3.7 stored, as stored (read with numpy, not with the loader under test)
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
                   "frame_kind": (report["frame"] or {}).get("kind"),
                   "tissue_source": tissue["domain"]["source"] if tissue is not None else None,
                   "junctions": report["junctions"]["count"], "loops_found": report["loops"]["by_segment"]["found"],
                   "loops_found_by_node": report["loops"]["by_node"]["found"],
                   "K_exact": ((report["frame_orientation"] or {}).get("all") or {}).get("fisher_axial_K_exact")}
    extras[name] = {"report": sorted(beyond(report, rules["describe"])),
                    "events": beyond(report["events"], rules["events"])}


hashes, facts, extras = {}, {}, {}

# --- library members of every family, their index rows and a plan ----------
# seeds on both sides of 2 ** 31, as a library's own seeds are; the families
# 3.6 added at the smallest ratio of the default range, and foam at 4, whose
# bed then attaches to the feeders' tips
MEMBERS = (("tree", 3.0, 101), ("mesh", 3.0, 2147483649), ("tumour", 3.0, 5), ("aligned", 3.0, 107),
           ("aligned_tight", 2.52, 3), ("aligned_bed", 2.52, 3), ("capillary_bed", 2.52, 2), ("foam", 4.0, 5))
# the arguments grow_network gives the bed, kept so that the bed can be built
# again from them below; it is reached through main's name for it or through bed's own
build_bed = bed.build
captured = []


def capturing_build(*args, **kwargs):
    captured.append(copy.deepcopy((args, {key: value for key, value in kwargs.items() if key != "events"})))
    return build_bed(*args, **kwargs)


patched = [(module, attribute) for module, attribute in ((main, "build_bed"), (bed, "build"))
           if getattr(module, attribute, None) is build_bed]
for module, attribute in patched:
    setattr(module, attribute, capturing_build)
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
    facts[name] = dict(grown_facts(grown), argv=result["argv"], avoid_collisions=result["kwargs"]["avoid_collisions"],
                       bed=restrict(result["kwargs"]["bed"], rules["kwargs"]["bed"]))
    extras[name] = dict(grown_extras(grown), kwargs=beyond(result["kwargs"], rules["kwargs"]),
                        row=sorted(beyond(row, rules["index_row"])),
                        bed=sorted(beyond(result["kwargs"]["bed"] or {}, rules["kwargs"]["bed"])))
    members.append({"nodes": grown["nodes"], "tree": grown["tree"], "edges": grown["edges"], "frame": grown["frame"],
                    "growth_box_um": grown["growth_box_um"], "family": family})
for module, attribute in patched:
    setattr(module, attribute, build_bed)
if len(captured) != 1:
    raise SystemExit(f"the members built {len(captured)} beds, not the foam member's one")

# a plan of every family 3.7 offers, which leaves a remainder
PLANS = (("plan_3_7", 9, 9, tuple(lists["families"]), (1,) * len(lists["families"])),)
for name, seed, count, families, shares in PLANS:
    plan = library.plan_library(seed, count, families, shares, (2.52, 25.0))
    hashes[name] = {"members": record_sha([restrict(member, rules["plan"]) for member in plan]),
                    "stage_generators": stage_generators()}
    facts[name] = {"members": len(plan), "families": [member["family"] for member in plan]}
    extras[name] = {"members": sorted({key for member in plan for key in beyond(member, rules["plan"])})}

# --- bed.build ---------------------------------------------------------------
FOAM_BED = settings["bed"]
built_beds = {}


def bed_case(name, domain, bed_settings, **options):
    events = {}
    built = bed.build(domain, bed_settings, events=events, **options)
    built_beds[name] = built
    hashes[name] = dict({key: array_sha(built[key]) for key in ("nodes", "tree", "seeds", "greedy_edges", "kept",
                                                                 "tips", "tip_edges", "edges")},
                        frame=record_sha(restrict(built["frame"], rules["frame"])),
                        chains=record_sha(restrict(built["chains"], rules["chains"])),
                        events=record_sha(restrict(events, rules["bed_events"])),
                        global_generators=global_generators(), stage_generators=stage_generators())
    outcomes = [chain["outcome"] for chain in built["chains"]]
    facts[name] = {"columns": int(built["nodes"].shape[1]), "seeds": int(len(built["seeds"])),
                   "kept": int(np.count_nonzero(built["kept"])), "greedy_edges": int(len(built["greedy_edges"])),
                   "edges": int(len(built["edges"])), "tips": int(built["tips"].size),
                   "tips_attached": int(np.count_nonzero(built["tip_edges"])),
                   "tip_edges": int(np.sum(built["tip_edges"])), "chains": len(built["chains"]),
                   "outcomes": {outcome: outcomes.count(outcome) for outcome in sorted(set(outcomes))},
                   "frame_kind": None if built["frame"] is None else built["frame"]["kind"],
                   "events": {key: events[key] for key in rules["bed_events"] if events.get(key)}}
    extras[name] = {"returned": sorted(set(built) - set(lists["build_keys"])),
                    "events": beyond(events, rules["bed_events"]),
                    "chains": sorted({key for chain in built["chains"]
                                      for key in beyond(chain, rules["chains"]["[]"])}),
                    "frame": beyond(built["frame"] or {}, rules["frame"])}


# a bed alone in a cube: no tips, so no island pass, and no frame at stretch 1
bed_case("bed_alone", (np.zeros(3), np.full(3, 40.0)), FOAM_BED, d_min=1.0, seed=11, collision_margin=1.0)
# the foam member's bed built again from its feeders, exactly as grow_network asked for it
(domain, bed_settings), options = captured[0]
bed_case("bed_feeders", domain, bed_settings, **options)
# a bed alone at stretch 2, its seeds and candidates compressed along the
# unguided frame's grow direction, whose frame record becomes an axis
bed_case("bed_stretch", (np.zeros(3), np.array([40.0, 60.0, 40.0])), dict(FOAM_BED, stretch=2.0), d_min=1.0, seed=13,
         collision_margin=1.0, frame=guidance.unguided_frame(main.DEFAULT_DIRECTION, main.DEFAULT_PERPENDICULAR,
                                                             [20.0, 30.0, 20.0]))

# --- describe -----------------------------------------------------------------
# (the fixtures are version-1 archives: no tree labels, no events, few flags,
# no growth box and no frame; the archives the command line and the libraries write below carry all of them)
# the three fixtures, the last at a margin it was not grown to keep, so that violations are counted
for archive in lists["archives"][:-1]:
    describe_case("describe_" + archive[:-4], describe_module.describe_archive(os.path.join(fixtures, archive)))
describe_case("describe_" + lists["archives"][-1][:-4] + "_margin",
              describe_module.describe_archive(os.path.join(fixtures, lists["archives"][-1]), margin=1.0))
# the foam member, its bed's columns labelled as bridges, with the tissue
# distance over its growth box as a library archive records it
foam_member = next(member for member in members if member["family"] == "foam")
describe_case("describe_foam_member", describe_module.describe(
    foam_member["nodes"], foam_member["edges"], metadata={"growth_box": foam_member["growth_box_um"]},
    tree=foam_member["tree"], frame=foam_member["frame"], d_ref=1.0, evd_spacing=2.0))
# the bed at stretch 2 against the axis its frame record names, a nematic one
stretched = built_beds["bed_stretch"]
describe_case("describe_bed_stretch", describe_module.describe(stretched["nodes"], tree=stretched["tree"],
                                                               frame=stretched["frame"], d_ref=1.0))

# --- the command line -------------------------------------------------------
# foam at a fixed voxel size, its bed attached to the feeders' tips
CLI_CASES = (("cli_foam", ["--family", "foam", "--d-min", "1", "--count", "1", "--seed", "1", "--volume", "48", "48",
                           "48", "--fit", "voxel_size", "--voxel-size", "1", "--iterations", "6", "6", "--d0", "8",
                           "0"]),)
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
                             "seed": sidecar["seed"],
                             "rng_streams": restrict(sidecar["rng_streams"], rules["sidecar"]["rng_streams"]),
                             "bed": restrict(sidecar["bed"], rules["sidecar"]["bed"]),
                             "events": {key: sidecar["events"][key] for key in rules["events"]
                                        if sidecar["events"].get(key)}}
        extras[name][stem] = {"sidecar": beyond(sidecar, rules["sidecar"]),
                              "metadata": beyond(metadata, rules["sidecar"]),
                              "rng_streams": beyond(sidecar["rng_streams"], rules["sidecar"]["rng_streams"]),
                              "bed": beyond(sidecar["bed"] or {}, rules["sidecar"]["bed"]),
                              "events": beyond(sidecar["events"], rules["events"]), "arrays": unknown_arrays}
    hashes[name].update(global_generators=global_generators(), stage_generators=stage_generators())

# --- describe on the archive just written, with the tissue distance over its growth box ---
for name, spacing in (("cli_foam", 4.0),):
    stem = next(entry[:-4] for entry in hashes[name]["files"] if entry.endswith(".npz"))
    describe_case("describe_" + name, describe_module.describe_archive(os.path.join(out_dir, name, stem + ".npz"),
                                                                       evd_spacing=spacing))

# --- small libraries ----------------------------------------------------------
# each with the tissue-distance spacing its archives are described at, None for none
LIBRARIES = (("library", None, ["--count", "3", "--seed", "1", "--ratio-range", "2.52", "3.5", "--workers", "1"]),
             # what 3.7 added: foam, beside the two filled families, each in a cube of its own constant
             ("library_bed", 4.0, ["--count", "3", "--seed", "3", "--families", "foam", "capillary_bed", "aligned_bed",
                                   "--ratio-range", "2.52", "3.0", "--workers", "1"]))
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
    # index.csv restricted to the 3.7 columns, which a later version keeps first and in order
    pinned = [k for k, column in enumerate(table[0])
              if column in lists["index_columns"] and column not in lists["unpinned_columns"]]
    entry = {"status": status, "files": sorted(os.listdir(directory)),
             "content": record_sha(restrict(content, rules["manifest"])),
             "index": record_sha(dict(restrict(index, rules["index"]),
                                      networks=[restrict(row, rules["index_row"]) for row in index["networks"]])),
             "index_csv": record_sha([[line[k] for k in pinned] for line in table]),
             "stage_generators": stage_generators()}
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
        entry[row["file"]], extra["arrays"][row["file"]], metadata = archive_hashes(
            os.path.join(directory, row["file"]), "metadata")
        metadata_by_file[row["file"]] = metadata
        extra["metadata"][row["file"]] = sorted(beyond(metadata, rules["metadata"]))
        extra["metadata_values"][row["file"]] = {key: value
                                                 for key, value in beyond(metadata, rules["metadata"]).items()
                                                 if key not in ("seconds", "peak_rss_mb", "events")}
        extra["events"][row["file"]] = beyond(metadata["events"], rules["events"])
        extra["beds"][row["file"]] = sorted(beyond(metadata["grow_kwargs"]["bed"] or {},
                                                   rules["metadata"]["grow_kwargs"]["bed"]))
        describe_case(f"describe_{case}_" + row["file"], describe_module.describe_archive(
            os.path.join(directory, row["file"]), evd_spacing=spacing))
    hashes[case] = entry
    facts[case] = {"files": [row["file"] for row in index["networks"]], "families": content["families"],
                   "columns": table[0][:len(lists["index_columns"])], "rows": len(table) - 1,
                   "growth": restrict(content["growth"], rules["manifest"]["growth"]),
                   "failures": len(content["failures"]),
                   "weights": library.library_weights(index).tolist(),
                   "rungs": {row["file"]: {"rungs": row["rungs"], "bridges": row["bridges"],
                                           "metadata_rungs": len(metadata_by_file[row["file"]]["rungs"])}
                             for row in index["networks"]},
                   "beds": {row["file"]: restrict(metadata_by_file[row["file"]]["grow_kwargs"]["bed"],
                                                  rules["metadata"]["grow_kwargs"]["bed"])
                            for row in index["networks"]},
                   "events": {row["file"]: {key: value for key, value
                                            in restrict(metadata_by_file[row["file"]]["events"],
                                                        rules["events"]).items() if value}
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
    What the 3.7 modules and the current ones produce, run side by side:
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
    """Every recording of the 3.7 hashes, by file name: tests/fixtures/reference_hashes_3_7*.json."""
    recordings = {}
    for name in sorted(os.listdir(FIXTURES)):
        if name.startswith(RECORDINGS) and name.endswith(".json"):
            with open(os.path.join(FIXTURES, name)) as handle:
                recordings[name] = json.load(handle)
    return recordings


class PinnedRelease37Tests(unittest.TestCase):
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

    def test_the_current_modules_produce_what_the_3_7_modules_produce_in_this_environment(self):
        self.assertEqual(sorted(self.current["hashes"]), sorted(self.reference["hashes"]))
        for case in self.reference["hashes"]:
            with self.subTest(case=case):
                self.assertEqual(self.current["hashes"][case], self.reference["hashes"][case])
                self.assertEqual(self.current["facts"][case], self.reference["facts"][case])

    def test_the_cases_reach_the_stages_they_pin(self):
        facts = self.reference["facts"]
        events = {case: entry["events"] for case, entry in facts.items()
                  if isinstance(entry.get("events"), dict)}
        # a member of every family 3.7 offers, foam's with the released bed and every other with none
        grown = {case[len("member_"):].rsplit("_r", 1)[0]: case for case in facts if case.startswith("member_")}
        self.assertEqual(sorted(grown), sorted(FAMILIES_3_7))
        for family, case in grown.items():
            with self.subTest(family=family):
                self.assertEqual(facts[case]["bed"], FAMILIES_3_7[family].get("bed"))
                self.assertEqual(facts[case]["trees"], 1 if family in ("tree", "tumour") else 2)
                if family != "foam":
                    self.assertEqual([key for key in events[case] if key.startswith("bed_")], [])
        # the members still reach what their families reach: the pairs joined
        # arterial to venous, the aligned families steered and the filled ones cross-connected
        for family in ("mesh", "aligned", "aligned_tight", "aligned_bed", "capillary_bed"):
            self.assertGreater(events[grown[family]]["anastomosis_bridges_arteriovenous"], 0, family)
        for family in ("aligned", "aligned_tight", "aligned_bed"):
            self.assertGreater(events[grown[family]]["guided_steps"], 0, family)
        for family in ("aligned_bed", "capillary_bed"):
            self.assertGreater(events[grown[family]]["rung_bridges"], 0, family)
            self.assertEqual(facts[grown[family]]["rungs"], events[grown[family]]["rung_bridges"], family)
        # foam grows a bed between its two feeder trees, attached to their tips, and nothing else joins them
        foam = events[grown["foam"]]
        for key in ("bed_seeds", "bed_candidates", "bed_edges", "bed_segments_walked", "bed_tips_attached",
                    "bed_tip_edges", "bed_components"):
            self.assertGreater(foam[key], 0, key)
        self.assertEqual((facts[grown["foam"]]["bridges"], facts[grown["foam"]]["rungs"]), (0, 0))
        self.assertNotIn("anastomosis_tips", foam)
        # the plan: every family 3.7 offers
        self.assertEqual(sorted(set(facts["plan_3_7"]["families"])), sorted(FAMILIES_3_7))
        # the beds built directly: seeds, accepted and kept edges and chains in each
        for case in ("bed_alone", "bed_feeders", "bed_stretch"):
            with self.subTest(case=case):
                bed = facts[case]
                self.assertGreater(bed["seeds"], 0)
                self.assertEqual(bed["seeds"], bed["events"]["bed_seeds"])
                self.assertGreater(bed["kept"], 0)
                self.assertGreater(bed["greedy_edges"], 0)
                self.assertGreater(bed["edges"], 0)
                self.assertEqual(bed["greedy_edges"], bed["events"]["bed_edges"])
                self.assertGreater(bed["chains"], 0)
                self.assertGreater(bed["outcomes"]["walked"], 0)
        # with feeders the bed attaches to their tips; alone it has none
        self.assertGreater(facts["bed_feeders"]["tips_attached"], 0)
        self.assertEqual(facts["bed_feeders"]["tip_edges"], facts["bed_feeders"]["events"]["bed_tip_edges"])
        for case in ("bed_alone", "bed_stretch"):
            self.assertEqual((facts[case]["tips"], facts[case]["tips_attached"]), (0, 0), case)
        # the foam member's bed, built again from its feeders, is the bed grow_network made
        self.assertEqual(facts["bed_feeders"]["events"], {key: value for key, value in foam.items()
                                                          if key.startswith("bed_")})
        self.assertEqual(facts["bed_feeders"]["columns"], facts[grown["foam"]]["columns"])
        # the frame record: none alone at stretch 1, the feeders' unguided one, and an axis at stretch 2
        self.assertEqual([facts[case]["frame_kind"] for case in ("bed_alone", "bed_feeders", "bed_stretch")],
                         [None, "none", "axis"])
        self.assertEqual(facts["describe_bed_stretch"]["frame_kind"], "axis")
        # the command line: a foam run with voxels, its bed attached and its stream recorded
        (stem, run), = facts["cli_foam"].items()
        self.assertGreater(run["voxels"], 0)
        self.assertIn(f"_s{run['seed']}", stem)
        self.assertEqual(run["trees"], 2)
        self.assertEqual(run["bed"], FOAM_3_7["bed"])
        self.assertEqual(run["rng_streams"]["bed"], [run["seed"], 7])
        self.assertEqual(sorted(run["rng_streams"]), sorted(RNG_STREAMS_3_7))
        for key in ("bed_seeds", "bed_edges", "bed_segments_walked", "bed_tips_attached"):
            self.assertGreater(run["events"][key], 0, key)
        # describe meets every key of 3.7: tree labels, events, flags, the frame-relative keys and loops
        self.assertEqual(facts["describe_cli_foam"]["events"], len(EVENT_KEYS_3_7))
        self.assertEqual(len(facts["describe_cli_foam"]["flags"]), len(FLAG_KEYS_3_7))
        self.assertGreater(facts["describe_Lnet_i9_s3_margin"]["violations"], 0)
        for case in ("describe_foam_member", "describe_cli_foam"):
            self.assertGreater(facts[case]["junctions"], 0, case)
            self.assertGreater(facts[case]["loops_found"], 0, case)
        # the tissue distance over the growth box, given and recorded
        sources = {"describe_foam_member": "growth_box", "describe_cli_foam": "growth_box_um",
                   "describe_library_bed_net_00000_foam.npz": "growth_box"}
        self.assertEqual({case: facts[case]["tissue_source"] for case in sources}, sources)
        # the libraries: their files and columns, no failure, and the bed library's growth settings
        self.assertEqual(facts["library"]["files"],
                         ["net_00000_tree.npz", "net_00001_mesh.npz", "net_00002_tumour.npz"])
        self.assertEqual(facts["library_bed"]["files"],
                         ["net_00000_foam.npz", "net_00001_capillary_bed.npz", "net_00002_aligned_bed.npz"])
        for case in ("library", "library_bed"):
            self.assertEqual(facts[case]["columns"], list(INDEX_COLUMNS_3_7), case)
            self.assertEqual(facts[case]["failures"], 0, case)
        self.assertNotIn("bed_max_candidates", facts["library"]["growth"])
        self.assertEqual(facts["library_bed"]["growth"]["box_c"], {"foam": 15.0, "capillary_bed": 15.0,
                                                                   "aligned_bed": 20.0})
        self.assertEqual(facts["library_bed"]["growth"]["bed_max_candidates"], 2000000)
        # the foam archive records the released bed and its bed attached; the filled archives carry rungs
        beds, archived = facts["library_bed"]["beds"], facts["library_bed"]["events"]
        self.assertEqual(beds, {"net_00000_foam.npz": FOAM_3_7["bed"], "net_00001_capillary_bed.npz": None,
                                "net_00002_aligned_bed.npz": None})
        self.assertGreater(archived["net_00000_foam.npz"]["bed_tips_attached"], 0)
        for name in ("net_00001_capillary_bed.npz", "net_00002_aligned_bed.npz"):
            rungs = facts["library_bed"]["rungs"][name]
            self.assertGreater(rungs["rungs"], 0, name)
            self.assertEqual(rungs["rungs"], rungs["metadata_rungs"], name)

    def test_every_counter_appended_after_3_7_is_zero(self):
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

    def test_the_released_presets_are_unchanged(self):
        import main
        for family, released in FAMILIES_3_7.items():
            with self.subTest(family=family):
                self.assertEqual(json.loads(json.dumps(main.FAMILIES[family])), released)

    def test_the_recorded_3_7_hashes_are_matched(self):
        # a case passes when it matches a recording; a recording that the 3.7
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
