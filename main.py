"""
Command-line entry point: generates synthetic vascular networks and writes each
one as a binary TIFF volume, a compressed centreline archive and a JSON sidecar
recording how it was made.

    python main.py --count 5 --out ./output --seed 1

Every network is reproducible from its seed and the recorded parameters.
The pipeline is: grammar string (vSystem.F) -> turtle interpretation
(analyseGrammar) -> B-spline interpolation of stems (utils) -> mapping and
capsule rasterisation into the volume (computeVoxel). Three optional stages
shape the geometry without changing what the grammar decides: a persistent
random walk that bends each stem smoothly (--tortuosity walk), collision
avoidance between branches (--avoid-collisions), and anastomosis, which joins
a fraction of the tips into loops (--anastomose). --family bundles them.

The centreline archive is the source of truth for the geometry and the TIFF one
rasterisation of it. Because computeVoxel maps grammar units to voxels at
rasterisation time, the same saved centreline can be rendered at any voxel size
-- once per imaging modality, say -- without regenerating the network and so
without depending on generation being reproducible across versions:

    from computeVoxel import process_network
    from main import load_network

    network = load_network("output/Lnet_i8_s1.npz")
    volume = process_network(network["nodes"], (512, 512, 140),
                             fit="voxel_size", voxel_size=2.0)

Lengths and diameters are in grammar units, which the pipeline takes to be
micrometres; see the Units section of the README. Each JSON sidecar records the
convention it was written under in its "units" field.
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import tifffile

import graph
import libGenerator
from analyseGrammar import WalkSettings, branching_turtle_to_coords
from anastomosis import EVENT_KEYS as ANASTOMOSIS_EVENTS, anastomose as bridge_tips
from collisions import CollisionAvoider
from computeVoxel import AXES, FITS, process_network
from spatial import make_index
from utils import interpolate_segments
from vSystem import F

# The unit one grammar unit is taken to stand for. Diameters, segment lengths
# and --voxel-size are all in this unit, so that --voxel-size is a modality's
# physical voxel size. Recorded in every sidecar so a rescaled dataset declares
# itself rather than being mistaken for the default.
DEFAULT_UNITS = "um"

# Version of the archive layout. 1 held nodes, program and metadata; 2 adds the
# edges, node_kind and tree arrays. A reader of version 1 ignores the additions.
ARCHIVE_VERSION = 2

TORTUOSITIES = ("stems", "walk")
ANASTOMOSE_MODES = ("any", "arteriovenous")

# Every counter a run can report, so that a sidecar lists the zero ones too.
EVENT_KEYS = ("bound_terminations", "walk_bound_redraws", "walk_bound_terminations",
              "collision_redraws", "collision_terminations", "collision_truncated_stems",
              "root_relocations", "root_collisions") + ANASTOMOSIS_EVENTS

# Independent random streams for the stages after the grammar, each seeded
# from the run seed together with a fixed tag, so that enabling a stage never
# changes a draw the grammar makes and every stage is reproducible on its own.
RNG_STREAMS = {"walk": 1, "anastomosis": 2}

# Bundles of settings for a family of networks. `tree` is the plain grammar;
# `mesh` is a capillary-bed-like network: two trees grown into the volume from
# opposite faces, bent by the walk, kept apart by collision avoidance and
# joined arterial-to-venous; `tumour` is tortuous, heavily looped and irregular
# in calibre. The persistence values are provisional calibrations, chosen from
# the arc/chord ratios they produce (see docs/geometry). `aligned` -- growth
# along a preferred direction, as in muscle -- needs a directional bias in the
# turtle, which parameter bundling cannot express, and is not offered yet.
FAMILIES = {
    "tree": {},
    "mesh": {"tortuosity": "walk", "persistence": 10.0, "avoid_collisions": True,
             "anastomose": True, "anastomose_mode": "arteriovenous", "anastomosis_fraction": 0.5,
             "anastomosis_radius": 20.0, "grow_in_volume": True},
    "tumour": {"tortuosity": "walk", "persistence": 3.0, "avoid_collisions": True,
               "anastomose": True, "anastomose_mode": "any", "anastomosis_fraction": 0.8,
               "d0": (20.0, 10.0), "aneurysm_prob": 0.1, "stenosis_prob": 0.1},
    "aligned": None,
}


def _walk_settings(tortuosity, persistence, seed):
    if tortuosity == "stems":
        return None
    if tortuosity != "walk":
        raise ValueError(f"tortuosity must be one of {TORTUOSITIES}, got {tortuosity!r}")
    if persistence is None or not persistence > 0.0:
        raise ValueError("the walk needs a positive persistence, in multiples of the vessel diameter")
    if seed is None:
        raise ValueError("the walk needs a seed so that the network is reproducible")
    return WalkSettings(persistence, np.random.default_rng([int(seed), RNG_STREAMS["walk"]]))


def _free_extent(program, d0, direction, perpendicular, walk):
    """Bounding extent of a tree grown without bounds, and its farthest reach along `direction`."""
    rows = branching_turtle_to_coords(program, d0, direction=direction, perpendicular=perpendicular,
                                      walk=walk)
    free = np.array([row[:3] for row in rows if not math.isnan(row[0])])
    extent = free.max(axis=0) - free.min(axis=0)
    along = float(np.max(free @ np.asarray(direction, dtype=float)))
    return extent, along


def _clear_root(avoider, position, direction, box, radius, events):
    """
    Moves a tree's root across its face of the box until the root point clears
    the network already placed, in rings of one root diameter about the
    face's centre. A second tree's root can otherwise land inside a branch of
    the first, which no later check would catch since the root is the first
    point of its stem.

    Returns:
        ndarray: the root position, the original when it was clear or no clear
        spot exists (then counted in events["root_collisions"]).
    """
    position = np.asarray(position, dtype=float)
    if avoider is None or avoider.n_points == 0:
        return position
    axes = [a for a in range(3) if abs(direction[a]) < 0.5]
    step = 2.0 * radius
    span = int(np.floor(min(box[a] for a in axes) / (2.0 * step)))
    for ring in range(span + 1):
        offsets = [(i, j) for i in range(-ring, ring + 1) for j in range(-ring, ring + 1)
                   if max(abs(i), abs(j)) == ring]
        for i, j in sorted(offsets, key=lambda o: (o[0] ** 2 + o[1] ** 2, o)):
            candidate = position.copy()
            candidate[axes[0]] += i * step
            candidate[axes[1]] += j * step
            inside = [radius <= candidate[a] <= box[a] - radius for a in axes]
            if not all(inside):
                continue
            qi, _ = avoider.index.query(candidate[None, :], [radius], avoider.margin)
            if len(qi) == 0:
                if ring:
                    events["root_relocations"] += 1
                return candidate
    events["root_collisions"] += 1
    return position


def grow_network(niter, d0, properties, tVol, fit="isotropic", clip_axes=(2,), voxel_size=None,
                 subdivisions=3, direction=(0.0, 1.0, 0.0), perpendicular=(0.0, 0.0, 1.0),
                 d_min=None, grow_in_volume=False, tortuosity="stems", persistence=None,
                 avoid_collisions=False, collision_margin=1.0, collision_attempts=10,
                 collision_index="auto", anastomose=False, anastomosis_radius=10.0,
                 anastomosis_fraction=0.5, anastomose_mode="any", seed=None):
    """
    Grows one network and returns its geometry and graph, without rendering it.

    Args:
        niter, d0, properties, tVol, fit, clip_axes, voxel_size, subdivisions,
        direction, perpendicular, d_min, grow_in_volume: as for generate_network.
        tortuosity (str): "stems" draws each stem as the grammar's five
            sub-segments smoothed by a B-spline; "walk" replaces the path of
            each stem by a persistent random walk of the same arc length.
        persistence (float or None): persistence length of the walk in
            multiples of the local diameter; required by "walk".
        avoid_collisions (bool): reject points closer than the sum of the radii
            plus `collision_margin` to any branch other than their own or their
            junction's; a walk step is redrawn up to `collision_attempts` times,
            a spline stem is shortened, and a branch that cannot be placed is
            terminated and counted.
        collision_margin (float): required clearance between vessel surfaces,
            grammar units.
        collision_index (str): "grid", "kdtree" or "auto", which is the grid
            hash: it was the faster of the two at 10^5 and 10^6 points.
        anastomose (bool): after growth, bridge a fraction of the tips to
            partners within `anastomosis_radius` tip diameters.
        anastomose_mode (str): "any" joins tips within one tree;
            "arteriovenous" grows a second tree from the opposite face of the
            volume and joins arterial tips to venous partners first.
        seed (int or None): the run seed; required whenever the walk or
            anastomosis is on, since their random streams derive from it.

    Returns:
        dict: "nodes" the (4, N) centreline; "program" the grammar string (of
        the first tree); "programs" one string per tree; "edges" the (2, E)
        graph; "node_kind" and "tree" per-column labels; "events" the counters
        of everything terminated, redrawn or skipped; "growth_box_um" the
        confining box or None; "clip_axes" the axes the rasteriser should clip;
        "bridges" the joins made; "collision_index" the index kind used.
    """
    libGenerator.setProperties(properties)
    if anastomose_mode not in ANASTOMOSE_MODES:
        raise ValueError(f"anastomose_mode must be one of {ANASTOMOSE_MODES}, got {anastomose_mode!r}")
    if anastomose and seed is None:
        raise ValueError("anastomosis needs a seed so that the network is reproducible")
    if anastomose and not 0.0 <= anastomosis_fraction <= 1.0:
        raise ValueError("the anastomosis fraction must lie in [0, 1]")
    trees = 2 if anastomose and anastomose_mode == "arteriovenous" else 1
    programs = [F(niter, d0, d_min) for _ in range(trees)]
    events = {key: 0 for key in EVENT_KEYS}
    direction = np.asarray(direction, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)

    bounds = None
    box = None
    positions = [np.zeros(3)]
    directions = [direction]
    clip = tuple(clip_axes)
    need_extent = grow_in_volume and fit != "voxel_size"
    extent = along = None
    if need_extent or trees == 2:
        # Interpreting the grammar consumes no randomness -- every token F emits
        # carries its operands -- and the walk's stream is re-seeded below, so
        # measuring the free extent first changes nothing about the drawn tree.
        extent, along = _free_extent(programs[0], d0, direction, perpendicular,
                                     _walk_settings(tortuosity, persistence, seed))
    if grow_in_volume:
        shape = np.asarray(tVol, dtype=float)
        if fit == "voxel_size":
            if voxel_size is None or voxel_size <= 0:
                raise ValueError("growing in the volume at a fixed voxel size needs a positive "
                                 "voxel_size, since it sets the field of view in grammar units")
            box = shape * float(voxel_size)
        else:
            # The box takes the volume's proportions at the largest size the tree
            # still overflows, so growth is confined rather than merely contained.
            box = shape * float(np.min(extent / shape))
        bounds = (np.zeros(3), box)
        positions = [np.array([box[0] / 2.0, 0.0, box[2] / 2.0])]
        clip = ()               # the tree already fits, so clipping has nothing to do
        if trees == 2:
            positions.append(np.array([box[0] / 2.0, box[1], box[2] / 2.0]))
            directions.append(-direction)
    elif trees == 2:
        # the second tree starts where the first one reaches and grows back towards it
        positions.append(along * direction)
        directions.append(-direction)

    # the grid hash is the faster index at every size measured (docs/geometry), so
    # "auto" means grid; the k-d tree stays available for comparison
    resolved_index = "grid" if collision_index == "auto" else collision_index
    avoider = None
    index_kind = None
    if avoid_collisions:
        index_kind = resolved_index
        cell = d0 * libGenerator.aneurysm_factor + collision_margin
        avoider = CollisionAvoider(make_index(index_kind, cell_size=cell), margin=collision_margin,
                                   attempts=collision_attempts)
    walk = _walk_settings(tortuosity, persistence, seed)

    pieces = []
    labels = []
    for index, (program, position, heading) in enumerate(zip(programs, positions, directions)):
        if index and box is not None:
            position = _clear_root(avoider, position, heading, box, d0 / 2.0, events)
            positions[index] = position
        rows = branching_turtle_to_coords(program, d0, position=position, direction=heading,
                                          perpendicular=perpendicular, bounds=bounds, walk=walk,
                                          avoid=avoider, events=events, subdivisions=subdivisions)
        part = interpolate_segments(rows, subdivisions=subdivisions)
        if pieces:
            pieces.append(np.full((4, 1), np.nan))
            labels.append(np.array([-1], dtype=np.int8))
        pieces.append(part)
        labels.append(np.where(np.isnan(part[0]), -1, index).astype(np.int8))
    nodes = np.concatenate(pieces, axis=1)
    tree = np.concatenate(labels)

    bridges = []
    if anastomose:
        rng = np.random.default_rng([int(seed), RNG_STREAMS["anastomosis"]])
        nodes, tree, bridges = bridge_tips(nodes, rng, anastomosis_fraction, anastomosis_radius,
                                           mode=anastomose_mode, tree=tree, persistence=persistence,
                                           collision_margin=collision_margin if avoid_collisions else None,
                                           events=events, attempts=collision_attempts,
                                           index_kind=resolved_index)
    built = graph.build(nodes)
    return {
        "nodes": nodes, "program": programs[0], "programs": programs,
        "edges": built["edges"], "node_kind": built["node_kind"], "tree": tree,
        "events": events, "growth_box_um": None if box is None else [float(v) for v in box],
        "root_positions_um": [[float(v) for v in p] for p in positions],
        "clip_axes": clip, "bridges": bridges, "collision_index": index_kind,
    }


def generate_network(niter, d0, properties, tVol, fit="isotropic", clip_axes=(2,),
                     voxel_size=None, subdivisions=3,
                     direction=(0.0, 1.0, 0.0), perpendicular=(0.0, 0.0, 1.0),
                     d_min=None, connect=True, grow_in_volume=False, **shaping):
    """
    Generates one network and renders it into a volume.

    Args:
        niter (int): number of drawn generations.
        d0 (float): root diameter, in grammar units (micrometres by convention).
        properties (dict): libGenerator parameters; missing keys take defaults.
        tVol (sequence): volume shape (nx, ny, nz).
        fit, clip_axes, voxel_size: see computeVoxel.fit_to_volume. `clip_axes`
            defaults to the depth axis here because the default volume is an
            imaging slab, where computeVoxel defaults to clipping nothing; only
            the isotropic fit reads it.
        subdivisions (int): B-spline sampling depth for braced stems.
        direction, perpendicular: initial turtle frame.
        d_min (float or None): smallest drawn vessel diameter, in the same units
            as d0; a branch terminates once it falls below it, so the tree stops
            on whichever of `niter` and `d_min` comes first. None stops on
            `niter` alone. It bounds the diameter a branch is drawn at, not the
            diameter after a local anomaly: a stenosis still narrows a drawn
            sub-segment by stenosis_factor.
        connect (bool): see computeVoxel.rasterise_segments. Left true, a vessel
            too thin to contain a voxel centre still renders as an unbroken
            one-voxel path instead of a dotted line.
        grow_in_volume (bool): confine growth to a box with the volume's
            proportions, terminating any branch that would leave it, so the tree
            takes the volume's shape and no vessel is cut part way along. The
            box is `tVol` times `voxel_size` under the voxel_size fit, and
            otherwise the largest box of those proportions the tree still
            overflows. `clip_axes` is then ignored, since nothing is clipped.
        **shaping: the geometry options of grow_network (tortuosity,
            persistence, avoid_collisions, collision_margin, collision_attempts,
            collision_index, anastomose, anastomosis_radius,
            anastomosis_fraction, anastomose_mode, seed). Left out, the network
            is the plain grammar's.

    Returns:
        tuple: (volume, program, nodes) with volume a uint8 array of 0 and 1,
        program the grammar string and nodes the (4, N) interpolated centreline
        of x, y, z and diameter with NaN column separators.

    `nodes` is the geometry in grammar units and independent of `tVol`, `fit`
    and `voxel_size`: passing it back to computeVoxel.process_network with
    different settings re-renders the same network at another resolution.
    """
    grown = grow_network(niter, d0, properties, tVol, fit=fit, clip_axes=clip_axes,
                         voxel_size=voxel_size, subdivisions=subdivisions, direction=direction,
                         perpendicular=perpendicular, d_min=d_min, grow_in_volume=grow_in_volume,
                         **shaping)
    volume = process_network(grown["nodes"], tVol, fit=fit, voxel_size=voxel_size,
                             clip_axes=grown["clip_axes"], connect=connect)
    return volume, grown["program"], grown["nodes"]


def write_volume(path, volume):
    """
    Writes a 0/1 volume indexed (x, y, z) as a uint8 TIFF of 0 and 255 whose
    pages are z, rows y and columns x.
    """
    stack = np.transpose(volume.astype(np.uint8) * 255, (2, 1, 0))
    tifffile.imwrite(path, stack, photometric="minisblack")


def save_network(path, nodes, program=None, metadata=None, edges=None, node_kind=None, tree=None):
    """
    Writes a centreline to a compressed .npz archive.

    The archive holds the geometry in grammar units, so a reader can rasterise
    it at any voxel size with computeVoxel.process_network instead of
    regenerating the network or resampling a finished volume. It grows with the
    number of drawn generations rather than with the volume: tens of kilobytes
    at four generations and about seven megabytes at twelve, against 37 MB for
    a 512 x 512 x 140 volume whatever the network inside it.

    Args:
        path (str): destination; numpy appends '.npz' if it is missing, and
            load_network accepts the path either way.
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN column
            separators, as returned by generate_network.
        program (str or None): the grammar string the network was drawn from.
        metadata (dict or None): a JSON-serialisable record of how it was made,
            including the unit its coordinates are in.
        edges (ndarray or None): (2, E) int column indices into `nodes`, the
            network's graph (see graph.py); stored as int64.
        node_kind (ndarray or None): (N,) per-column label from graph.node_kind;
            stored as int8.
        tree (ndarray or None): (N,) per-column tree label (0 and 1 for the
            trees, 2 for anastomosis bridges, -1 at separators); stored as int8.

    The stored arrays are exact, but the archive is a zip and its entries carry
    a modification time, so two runs of the same seed produce equal arrays in
    files that differ byte for byte.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError(
            f"nodes must be a (4, N) array of x, y, z, diameter, got {nodes.shape}")
    arrays = {"nodes": nodes}
    if program is not None:
        arrays["program"] = np.array(str(program))
    if metadata is not None:
        arrays["metadata"] = np.array(json.dumps(metadata))
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError(f"edges must be a (2, E) array of column indices, got {edges.shape}")
        if edges.size and (edges.min() < 0 or edges.max() >= nodes.shape[1]):
            raise ValueError("edges index columns outside nodes")
        arrays["edges"] = edges
    for name, array in (("node_kind", node_kind), ("tree", tree)):
        if array is not None:
            array = np.asarray(array, dtype=np.int8)
            if array.shape != (nodes.shape[1],):
                raise ValueError(f"{name} must hold one label per column of nodes")
            arrays[name] = array
    np.savez_compressed(path, **arrays)


def load_network(path, upgrade=True):
    """
    Reads a centreline written by save_network.

    Args:
        path (str): the .npz archive, with or without its '.npz' suffix, since
            save_network appends one to a path that lacks it.
        upgrade (bool): when the archive predates the graph arrays, derive
            "edges" and "node_kind" from the coordinates (graph.build) so that
            every archive reads back with a graph. False leaves them None.

    Returns:
        dict: "nodes" the (4, N) float array of x, y, z and diameter, "program"
        the grammar string or None, "metadata" the decoded record or None,
        "edges" the (2, E) int64 graph, "node_kind" the (N,) int8 labels and
        "tree" the (N,) int8 tree labels or None when the archive has none.
    """
    if not os.path.exists(path) and not path.endswith(".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=False) as handle:
        nodes = np.asarray(handle["nodes"], dtype=float)
        program = handle["program"].item() if "program" in handle.files else None
        record = json.loads(handle["metadata"].item()) if "metadata" in handle.files else None
        edges = np.asarray(handle["edges"], dtype=np.int64) if "edges" in handle.files else None
        node_kind = np.asarray(handle["node_kind"], dtype=np.int8) if "node_kind" in handle.files else None
        tree = np.asarray(handle["tree"], dtype=np.int8) if "tree" in handle.files else None
    if upgrade and (edges is None or node_kind is None):
        built = graph.build(nodes)
        edges = built["edges"] if edges is None else edges
        node_kind = built["node_kind"] if node_kind is None else node_kind
    return {"nodes": nodes, "program": program, "metadata": record,
            "edges": edges, "node_kind": node_kind, "tree": tree}


def sample_parameters(args):
    """Draws the per-network parameters from the ranges given on the command line."""
    properties = {
        "k": args.k,
        "epsilon": random.uniform(*args.epsilon),
        "randmarg": random.uniform(*args.randmarg),
        "sigma": args.sigma,
        "stochparams": not args.deterministic,
        "roll_angle": args.roll_angle,
        "stem_angle": args.stem_angle,
        "aneurysm_prob": args.aneurysm_prob,
        "stenosis_prob": args.stenosis_prob,
    }
    mean, std = args.d0
    d0 = np.random.normal(mean, std)
    while d0 < args.d0_min:  # truncated draw: a non-positive diameter has no bifurcation solution
        d0 = np.random.normal(mean, std)
    niter = random.randint(*args.iterations)
    return properties, float(d0), niter


def parse_axes(text):
    if text.lower() in ("", "none"):
        return ()
    axes = []
    for token in text.lower().replace(",", " ").split():
        if token in AXES:
            axes.append(AXES[token])
        elif token in ("0", "1", "2"):
            axes.append(int(token))
        else:
            raise argparse.ArgumentTypeError(f"unknown axis {token!r}; use x, y, z or none")
    return tuple(sorted(set(axes)))


def shaping_options(args):
    """The geometry options of grow_network, read from parsed arguments."""
    return {
        "tortuosity": args.tortuosity, "persistence": args.persistence,
        "avoid_collisions": args.avoid_collisions, "collision_margin": args.collision_margin,
        "collision_attempts": args.collision_attempts, "collision_index": args.collision_index,
        "anastomose": args.anastomose, "anastomosis_radius": args.anastomosis_radius,
        "anastomosis_fraction": args.anastomosis_fraction, "anastomose_mode": args.anastomose_mode,
    }


def build_parser(family="tree"):
    """
    Builds the command-line parser. `family` names the preset whose values
    become the defaults, so that an option given explicitly still overrides it.
    """
    d = libGenerator.default
    parser = argparse.ArgumentParser(
        description="Generate synthetic vascular networks as binary TIFF volumes.")
    parser.add_argument("--count", type=int, default=5, help="number of networks (default 5)")
    parser.add_argument("--out", default="output", help="output directory (default ./output)")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed; network i uses seed + i (default: random)")
    parser.add_argument("--volume", type=int, nargs=3, default=(512, 512, 140), metavar=("NX", "NY", "NZ"),
                        help="volume shape in voxels (default 512 512 140)")
    parser.add_argument("--iterations", type=int, nargs=2, default=(4, 12), metavar=("MIN", "MAX"),
                        help="range of drawn generations (default 4 12)")
    parser.add_argument("--d0", type=float, nargs=2, default=(20.0, 5.0), metavar=("MEAN", "STD"),
                        help="root diameter distribution in grammar units (default 20 5)")
    parser.add_argument("--d0-min", type=float, default=1.0, help="smallest accepted root diameter (default 1)")
    parser.add_argument("--d-min", type=float, default=None, dest="d_min",
                        help="smallest drawn vessel diameter in grammar units; a branch stops "
                             "bifurcating once it falls below it (default: none, stop on "
                             "--iterations alone). A stenosis still narrows a drawn sub-segment below it")
    parser.add_argument("--epsilon", type=float, nargs=2, default=(4.0, 10.0), metavar=("MIN", "MAX"),
                        help="length-to-diameter ratio range (default 4 10)")
    parser.add_argument("--randmarg", type=float, nargs=2, default=(0.1, 0.3), metavar=("MIN", "MAX"),
                        help="relative half-width of the segment-length distribution (default 0.1 0.3)")
    parser.add_argument("--sigma", type=float, default=d["sigma"],
                        help="d_opt / sigma is the spread of the first daughter diameter (default 5)")
    parser.add_argument("--k", type=float, default=d["k"], help="Murray's law exponent (default 3)")
    parser.add_argument("--roll-angle", type=float, default=d["roll_angle"],
                        help="roll after each daughter turn, degrees (default 70)")
    parser.add_argument("--stem-angle", type=float, default=d["stem_angle"],
                        help="turn between stem sub-segments, degrees (default 25)")
    parser.add_argument("--aneurysm-prob", type=float, default=d["aneurysm_prob"],
                        help="per sub-segment probability of a local dilation (default 0.02)")
    parser.add_argument("--stenosis-prob", type=float, default=d["stenosis_prob"],
                        help="per sub-segment probability of a local constriction (default 0.02)")
    parser.add_argument("--deterministic", action="store_true",
                        help="use the symmetric optimum instead of drawing daughter diameters")
    parser.add_argument("--fit", choices=FITS, default="isotropic",
                        help="mapping into the volume (default isotropic)")
    parser.add_argument("--clip-axes", type=parse_axes, default=(2,),
                        help="axes left out of the isotropic fit and clipped, e.g. 'z' or 'none' (default z)")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="grammar units per voxel, for --fit voxel_size; under the default "
                             "unit convention this is the modality's voxel size in micrometres")
    parser.add_argument("--units", default=DEFAULT_UNITS,
                        help="physical unit one grammar unit stands for, recorded in the sidecar "
                             f"(default {DEFAULT_UNITS}); it labels --d0, --d-min and --voxel-size "
                             "and does not rescale anything")
    parser.add_argument("--grow-in-volume", action="store_true",
                        help="confine growth to the volume's proportions so the tree takes its "
                             "shape and no vessel is cut part way along; --clip-axes is then unused")
    parser.add_argument("--no-connect", action="store_false", dest="connect",
                        help="rasterise bare capsules, so that a vessel thinner than a voxel "
                             "renders as a dotted line rather than a one-voxel path")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="B-spline sampling depth for stems, 2^i points per span (default 3); "
                             "also the number of walk steps per stem sub-segment")

    shape = parser.add_argument_group(
        "geometry", "opt-in stages that shape the geometry the grammar decides; the default "
                    "command line is unaffected by them")
    shape.add_argument("--family", choices=sorted(FAMILIES), default="tree",
                       help="preset bundle of the options below (default tree, the plain grammar): "
                            "mesh = walk + collision avoidance + arteriovenous anastomosis grown in "
                            "the volume; tumour = low persistence, dense anastomosis, wide --d0 and "
                            "raised anomaly probabilities; aligned is not available yet. Options "
                            "given explicitly override the preset")
    shape.add_argument("--tortuosity", choices=TORTUOSITIES, default="stems",
                       help="stems: each stem is the grammar's five sub-segments smoothed by a "
                            "B-spline (default); walk: each stem is a persistent random walk of the "
                            "same arc length, bending continuously")
    shape.add_argument("--persistence", type=float, default=None,
                       help="persistence length of the walk in multiples of the local vessel "
                            "diameter; required by --tortuosity walk (lower is more tortuous)")
    shape.add_argument("--avoid-collisions", action="store_true",
                       help="keep branches apart: a point closer than the sum of the radii plus "
                            "--collision-margin to another branch is redrawn (walk) or the stem is "
                            "shortened (stems); a branch that cannot be placed terminates and is counted")
    shape.add_argument("--collision-margin", type=float, default=1.0,
                       help="clearance required between vessel surfaces, grammar units (default 1)")
    shape.add_argument("--collision-attempts", type=int, default=10,
                       help="redraws of a colliding walk step or bridge before giving up (default 10)")
    shape.add_argument("--collision-index", choices=("auto", "grid", "kdtree"), default="auto",
                       help="spatial index for the collision check (default auto, which is the "
                            "grid hash; kdtree needs scipy and was slower at every size measured)")
    shape.add_argument("--anastomose", action="store_true",
                       help="after growth, bridge a random fraction of the tips to nearby "
                            "partners, closing the tree into a network with loops")
    shape.add_argument("--anastomosis-radius", type=float, default=10.0,
                       help="partner search radius in multiples of the tip diameter (default 10: "
                            "at the default segment lengths the nearest eligible partner of a tip "
                            "lies about 6.5 diameters away and the nearest other tip about 8)")
    shape.add_argument("--anastomosis-fraction", type=float, default=0.5,
                       help="fraction of tips that seek a partner (default 0.5)")
    shape.add_argument("--anastomose-mode", choices=ANASTOMOSE_MODES, default="any",
                       help="any: partners anywhere in the network (default); arteriovenous: grow "
                            "a second tree from the opposite face and join arterial tips to "
                            "venous partners first")
    preset = FAMILIES.get(family)
    if preset:
        parser.set_defaults(**preset)
    return parser


def validate_shaping(args):
    """Refuses option combinations that cannot be honoured, before anything is written."""
    if args.tortuosity == "walk" and args.persistence is None:
        raise SystemExit("--tortuosity walk needs --persistence (a positive number of vessel diameters)")
    if args.persistence is not None and not args.persistence > 0.0:
        raise SystemExit("--persistence must be positive")
    if args.collision_margin < 0.0:
        raise SystemExit("--collision-margin cannot be negative")
    if args.collision_attempts < 0:
        raise SystemExit("--collision-attempts cannot be negative")
    if not 0.0 <= args.anastomosis_fraction <= 1.0:
        raise SystemExit("--anastomosis-fraction must lie between 0 and 1")
    if not args.anastomosis_radius > 0.0:
        raise SystemExit("--anastomosis-radius must be positive")


def main(argv=None):
    peel = argparse.ArgumentParser(add_help=False)
    peel.add_argument("--family", choices=sorted(FAMILIES), default="tree")
    chosen, _ = peel.parse_known_args(argv)
    if FAMILIES[chosen.family] is None:
        raise SystemExit(
            f"--family {chosen.family} is not available: growth along a preferred direction needs "
            "a directional bias in the turtle, which no combination of the existing options "
            "expresses; see the README")
    args = build_parser(chosen.family).parse_args(argv)
    if args.fit == "voxel_size" and args.voxel_size is None:
        raise SystemExit("--fit voxel_size requires --voxel-size")
    if args.d_min is not None and args.d_min <= 0:
        raise SystemExit("--d-min must be a positive diameter")
    validate_shaping(args)
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    os.makedirs(args.out, exist_ok=True)
    tVol = tuple(args.volume)
    shaping = shaping_options(args)

    for index in range(args.count):
        seed = base_seed + index
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        properties, d0, niter = sample_parameters(args)
        if args.d_min is not None and d0 < args.d_min:
            raise SystemExit(
                f"--d-min {args.d_min:g} exceeds the root diameter {d0:.3g} sampled for seed "
                f"{seed}, so the network would be empty; lower --d-min or raise --d0")
        grown = grow_network(niter, d0, properties, tVol, fit=args.fit, clip_axes=args.clip_axes,
                             voxel_size=args.voxel_size, subdivisions=args.subdivisions,
                             d_min=args.d_min, grow_in_volume=args.grow_in_volume, seed=seed,
                             **shaping)
        nodes = grown["nodes"]
        volume = process_network(nodes, tVol, fit=args.fit, voxel_size=args.voxel_size,
                                 clip_axes=grown["clip_axes"], connect=args.connect)
        stem = f"Lnet_i{niter}_s{seed}"
        write_volume(os.path.join(args.out, stem + ".tiff"), volume)
        record = {
            "seed": seed,
            "iterations": niter,
            "d0": d0,
            "d_min": args.d_min,
            "properties": properties,
            "volume": list(tVol),
            "axis_order": "zyx",
            "units": args.units,
            "fit": args.fit,
            "clip_axes": list(grown["clip_axes"]),
            "connect": args.connect,
            "grow_in_volume": args.grow_in_volume,
            "voxel_size": args.voxel_size,
            "subdivisions": args.subdivisions,
            "archive_version": ARCHIVE_VERSION,
            "family": args.family,
            "trees": len(grown["programs"]),
            "growth_box_um": grown["growth_box_um"],
            "root_positions_um": grown["root_positions_um"],
            "rng_streams": {name: [seed, tag] for name, tag in RNG_STREAMS.items()},
            "collision_index": grown["collision_index"],
            "events": grown["events"],
            "bridges": len(grown["bridges"]),
        }
        record.update(shaping)
        record["collision_index"] = grown["collision_index"]      # the kind used, not "auto"
        with open(os.path.join(args.out, stem + ".json"), "w") as handle:
            json.dump(record, handle, indent=2)
        save_network(os.path.join(args.out, stem + ".npz"), nodes, program=grown["program"],
                     metadata=record, edges=grown["edges"], node_kind=grown["node_kind"],
                     tree=grown["tree"])
        if not volume.any():
            print(f"{stem}.tiff: warning: no vessel voxels; the network is outside the volume or "
                  f"below its resolution", file=sys.stderr)
        calibre = ""
        if args.fit == "voxel_size":
            # the calibre a mis-scaled voxel size shows up in first
            calibre = f", root diameter {d0 / args.voxel_size:.1f} voxels"
        shaped = ""
        counted = {k: v for k, v in grown["events"].items() if v}
        if counted:
            shaped = ", events " + " ".join(f"{k}={v}" for k, v in sorted(counted.items()))
        canonical = graph.canonical_columns(nodes)
        vertices = np.unique(canonical[canonical >= 0])
        tips = int(np.sum(grown["node_kind"][vertices] == graph.TIP))
        cycles = graph.betti(grown["edges"], canonical)["cycles"]
        print(f"{stem}.tiff: {niter} generations, d0 {d0:.1f} {args.units}, "
              f"vessel fraction {volume.mean() * 100:.2f}%{calibre}, {tips} tips, {cycles} cycles{shaped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
