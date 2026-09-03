"""
Command-line entry point: generates synthetic vascular networks and writes
each one as a binary TIFF volume with a JSON sidecar recording how it was made.

    python main.py --count 5 --out ./output --seed 1

Every network is reproducible from its seed and the recorded parameters.
The pipeline is: grammar string (vSystem.F) -> turtle interpretation
(analyseGrammar) -> B-spline interpolation of stems (utils) -> mapping and
capsule rasterisation into the volume (computeVoxel).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import tifffile

import libGenerator
from analyseGrammar import branching_turtle_to_coords
from computeVoxel import FITS, process_network
from utils import interpolate_segments
from vSystem import F

AXES = {"x": 0, "y": 1, "z": 2}


def generate_network(niter, d0, properties, tVol, fit="isotropic", clip_axes=(2,),
                     voxel_size=None, subdivisions=3,
                     direction=(0.0, 1.0, 0.0), perpendicular=(0.0, 0.0, 1.0)):
    """
    Generates one network and renders it into a volume.

    Args:
        niter (int): number of drawn generations.
        d0 (float): root diameter, in grammar units.
        properties (dict): libGenerator parameters; missing keys take defaults.
        tVol (sequence): volume shape (nx, ny, nz).
        fit, clip_axes, voxel_size: see computeVoxel.fit_to_volume.
        subdivisions (int): B-spline sampling depth for braced stems.
        direction, perpendicular: initial turtle frame.

    Returns:
        tuple: (volume, program, nodes) with volume a uint8 array of 0 and 1,
        program the grammar string and nodes the (4, N) interpolated centreline.
    """
    libGenerator.setProperties(properties)
    program = F(niter, d0)
    rows = branching_turtle_to_coords(program, d0, direction=direction, perpendicular=perpendicular)
    nodes = interpolate_segments(rows, subdivisions=subdivisions)
    volume = process_network(nodes, tVol, fit=fit, voxel_size=voxel_size, clip_axes=clip_axes)
    return volume, program, nodes


def write_volume(path, volume):
    """
    Writes a 0/1 volume indexed (x, y, z) as a uint8 TIFF of 0 and 255 whose
    pages are z, rows y and columns x.
    """
    stack = np.transpose(volume.astype(np.uint8) * 255, (2, 1, 0))
    tifffile.imwrite(path, stack, photometric="minisblack")


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
                        help="grammar units per voxel, for --fit voxel_size")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="B-spline sampling depth for stems, 2^i points per span (default 3)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fit == "voxel_size" and args.voxel_size is None:
        raise SystemExit("--fit voxel_size requires --voxel-size")
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    os.makedirs(args.out, exist_ok=True)
    tVol = tuple(args.volume)

    for index in range(args.count):
        seed = base_seed + index
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        properties, d0, niter = sample_parameters(args)
        volume, _, _ = generate_network(niter, d0, properties, tVol, fit=args.fit,
                                        clip_axes=args.clip_axes, voxel_size=args.voxel_size,
                                        subdivisions=args.subdivisions)
        stem = f"Lnet_i{niter}_s{seed}"
        write_volume(os.path.join(args.out, stem + ".tiff"), volume)
        record = {
            "seed": seed,
            "iterations": niter,
            "d0": d0,
            "properties": properties,
            "volume": list(tVol),
            "axis_order": "zyx",
            "fit": args.fit,
            "clip_axes": list(args.clip_axes),
            "voxel_size": args.voxel_size,
            "subdivisions": args.subdivisions,
        }
        with open(os.path.join(args.out, stem + ".json"), "w") as handle:
            json.dump(record, handle, indent=2)
        print(f"{stem}.tiff: {niter} generations, d0 {d0:.1f}, vessel fraction {volume.mean() * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
