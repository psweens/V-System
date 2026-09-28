"""
Command-line entry point: generates synthetic vascular networks and writes each
one as a binary TIFF volume, a compressed centreline archive and a JSON sidecar
recording how it was made.

    python main.py --count 5 --out ./output --seed 1

Every network is reproducible from its seed and the recorded parameters.
The pipeline is: grammar string (vSystem.F) -> turtle interpretation
(analyseGrammar) -> B-spline interpolation of stems (utils) -> mapping and
capsule rasterisation into the volume (computeVoxel).

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

import libGenerator
from analyseGrammar import branching_turtle_to_coords
from computeVoxel import AXES, FITS, process_network
from utils import interpolate_segments
from vSystem import F

# The unit one grammar unit is taken to stand for. Diameters, segment lengths
# and --voxel-size are all in this unit, so that --voxel-size is a modality's
# physical voxel size. Recorded in every sidecar so a rescaled dataset declares
# itself rather than being mistaken for the default.
DEFAULT_UNITS = "um"


def generate_network(niter, d0, properties, tVol, fit="isotropic", clip_axes=(2,),
                     voxel_size=None, subdivisions=3,
                     direction=(0.0, 1.0, 0.0), perpendicular=(0.0, 0.0, 1.0),
                     d_min=None, connect=True, grow_in_volume=False):
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

    Returns:
        tuple: (volume, program, nodes) with volume a uint8 array of 0 and 1,
        program the grammar string and nodes the (4, N) interpolated centreline
        of x, y, z and diameter with NaN column separators.

    `nodes` is the geometry in grammar units and independent of `tVol`, `fit`
    and `voxel_size`: passing it back to computeVoxel.process_network with
    different settings re-renders the same network at another resolution.
    """
    libGenerator.setProperties(properties)
    program = F(niter, d0, d_min)
    bounds = None
    position = (0.0, 0.0, 0.0)
    if grow_in_volume:
        shape = np.asarray(tVol, dtype=float)
        if fit == "voxel_size":
            if voxel_size is None or voxel_size <= 0:
                raise ValueError("growing in the volume at a fixed voxel size needs a positive "
                                 "voxel_size, since it sets the field of view in grammar units")
            box = shape * float(voxel_size)
        else:
            # Interpreting the grammar consumes no randomness -- every token F emits
            # carries its operands -- so measuring the free extent first is safe. The
            # box takes the volume's proportions at the largest size the tree still
            # overflows, so growth is confined rather than merely contained.
            free = np.array([row[:3] for row in
                             branching_turtle_to_coords(program, d0, direction=direction,
                                                        perpendicular=perpendicular)
                             if not math.isnan(row[0])])
            extent = free.max(axis=0) - free.min(axis=0)
            box = shape * float(np.min(extent / shape))
        bounds = (np.zeros(3), box)
        position = (box[0] / 2.0, 0.0, box[2] / 2.0)
        clip_axes = ()          # the tree already fits, so clipping has nothing to do
    rows = branching_turtle_to_coords(program, d0, position=position, direction=direction,
                                      perpendicular=perpendicular, bounds=bounds)
    nodes = interpolate_segments(rows, subdivisions=subdivisions)
    volume = process_network(nodes, tVol, fit=fit, voxel_size=voxel_size, clip_axes=clip_axes,
                             connect=connect)
    return volume, program, nodes


def write_volume(path, volume):
    """
    Writes a 0/1 volume indexed (x, y, z) as a uint8 TIFF of 0 and 255 whose
    pages are z, rows y and columns x.
    """
    stack = np.transpose(volume.astype(np.uint8) * 255, (2, 1, 0))
    tifffile.imwrite(path, stack, photometric="minisblack")


def save_network(path, nodes, program=None, metadata=None):
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
    np.savez_compressed(path, **arrays)


def load_network(path):
    """
    Reads a centreline written by save_network.

    Args:
        path (str): the .npz archive, with or without its '.npz' suffix, since
            save_network appends one to a path that lacks it.

    Returns:
        dict: "nodes" the (4, N) float array of x, y, z and diameter, "program"
        the grammar string or None, and "metadata" the decoded record or None.
    """
    if not os.path.exists(path) and not path.endswith(".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=False) as handle:
        nodes = np.asarray(handle["nodes"], dtype=float)
        program = handle["program"].item() if "program" in handle.files else None
        record = json.loads(handle["metadata"].item()) if "metadata" in handle.files else None
    return {"nodes": nodes, "program": program, "metadata": record}


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


def build_parser():
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
                        help="B-spline sampling depth for stems, 2^i points per span (default 3)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fit == "voxel_size" and args.voxel_size is None:
        raise SystemExit("--fit voxel_size requires --voxel-size")
    if args.d_min is not None and args.d_min <= 0:
        raise SystemExit("--d-min must be a positive diameter")
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    os.makedirs(args.out, exist_ok=True)
    tVol = tuple(args.volume)

    for index in range(args.count):
        seed = base_seed + index
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        properties, d0, niter = sample_parameters(args)
        if args.d_min is not None and d0 < args.d_min:
            raise SystemExit(
                f"--d-min {args.d_min:g} exceeds the root diameter {d0:.3g} sampled for seed "
                f"{seed}, so the network would be empty; lower --d-min or raise --d0")
        volume, program, nodes = generate_network(niter, d0, properties, tVol, fit=args.fit,
                                                  clip_axes=args.clip_axes, voxel_size=args.voxel_size,
                                                  subdivisions=args.subdivisions, d_min=args.d_min,
                                                  connect=args.connect,
                                                  grow_in_volume=args.grow_in_volume)
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
            "clip_axes": [] if args.grow_in_volume else list(args.clip_axes),
            "connect": args.connect,
            "grow_in_volume": args.grow_in_volume,
            "voxel_size": args.voxel_size,
            "subdivisions": args.subdivisions,
        }
        with open(os.path.join(args.out, stem + ".json"), "w") as handle:
            json.dump(record, handle, indent=2)
        save_network(os.path.join(args.out, stem + ".npz"), nodes, program=program, metadata=record)
        if not volume.any():
            print(f"{stem}.tiff: warning: no vessel voxels; the network is outside the volume or "
                  f"below its resolution", file=sys.stderr)
        calibre = ""
        if args.fit == "voxel_size":
            # the calibre a mis-scaled voxel size shows up in first
            calibre = f", root diameter {d0 / args.voxel_size:.1f} voxels"
        print(f"{stem}.tiff: {niter} generations, d0 {d0:.1f} {args.units}, "
              f"vessel fraction {volume.mean() * 100:.2f}%{calibre}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
