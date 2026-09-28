"""
Maximum-intensity projections of one seed under each geometry option.

Renders the same network (same seed, so the same grammar draws) with the
plain grammar and with each geometry stage, at one voxel size into volumes
of one shape, and writes the projections along z (the xy view) and along x
(the yz view) side by side:

    python docs/geometry/mips.py --out docs/geometry/mips --seed 3

Because every configuration of a seed shares its grammar, the panels differ
only by the stage under test: the walk bends the same stems, avoidance
removes the same crossings, anastomosis closes the same tips. matplotlib is
imported on use.
"""
import argparse
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from computeVoxel import process_network  # noqa: E402
from describe import describe  # noqa: E402
from main import build_parser, grow_network, sample_parameters, shaping_options  # noqa: E402

PANELS = [
    ("plain grammar (stems)", ["--iterations", "8", "8"]),
    ("walk, persistence 10", ["--iterations", "8", "8", "--tortuosity", "walk", "--persistence", "10"]),
    ("walk, persistence 3", ["--iterations", "8", "8", "--tortuosity", "walk", "--persistence", "3"]),
    ("anastomosis, fraction 0.5", ["--iterations", "8", "8", "--anastomose", "--anastomosis-fraction", "0.5"]),
    ("family mesh", ["--iterations", "8", "8", "--family", "mesh"]),
    ("family tumour", ["--iterations", "8", "8", "--family", "tumour"]),
]
DEEP = [
    ("plain grammar, 11 generations", ["--iterations", "11", "11", "--epsilon", "4", "4"]),
    ("collision avoidance, 11 generations", ["--iterations", "11", "11", "--epsilon", "4", "4", "--avoid-collisions"]),
    ("walk 10 + avoidance, 11 generations", ["--iterations", "11", "11", "--epsilon", "4", "4",
                                             "--avoid-collisions", "--tortuosity", "walk", "--persistence", "10"]),
]


def grow(options, seed):
    peel = argparse.ArgumentParser(add_help=False)
    peel.add_argument("--family", default="tree")
    family = peel.parse_known_args(options)[0].family
    args = build_parser(family).parse_args(options)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    properties, d0, niter = sample_parameters(args)
    grown = grow_network(niter, d0, properties, tuple(args.volume), fit=args.fit, clip_axes=args.clip_axes,
                         voxel_size=args.voxel_size, subdivisions=args.subdivisions, d_min=args.d_min,
                         grow_in_volume=args.grow_in_volume, seed=seed, **shaping_options(args))
    margin = args.collision_margin if args.avoid_collisions else 0.0
    report = describe(grown["nodes"], grown["edges"], metadata={"events": grown["events"]},
                      tree=grown["tree"], margin=margin)
    return grown, report


def render_all(panels, seed, voxel_size):
    """Grows every panel's network and renders all of them into volumes of one shape."""
    grown = [grow(options, seed) for _, options in panels]
    extents = [np.nanmax(g["nodes"][:3], axis=1) - np.nanmin(g["nodes"][:3], axis=1) for g, _ in grown]
    shape = tuple(int(v) for v in np.ceil(np.max(extents, axis=0) / voxel_size) + 8)
    volumes = [process_network(g["nodes"], shape, fit="voxel_size", voxel_size=voxel_size) for g, _ in grown]
    return grown, volumes, shape


def caption(report, events):
    counted = {k: v for k, v in events.items() if v and ("termin" in k or k in ("collision_truncated_stems",
                                                                                "anastomosis_bridges"))}
    lines = [f"tips {report['tips']['count']}  cycles {report['cycles']}  arc/chord {report['arc_chord']['mean']:.3f}",
             f"length {report['total_length_mm']:.1f} mm  min clearance "
             + (f"{report['clearance_um']['min']:.2f} um" if report['clearance_um']['min'] is not None else "-")
             + f"  violations {report['clearance_um']['violations']}"]
    if counted:
        lines.append(" ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(counted.items())))
    return "\n".join(lines)


def figure(panels, grown, volumes, shape, voxel_size, seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.6), squeeze=False)
    for k, ((title, _), (g, report), volume) in enumerate(zip(panels, grown, volumes)):
        xy = volume.max(axis=2).T          # rows y, columns x
        yz = volume.max(axis=0).T          # rows z, columns y
        axes[0, k].imshow(xy, cmap="gray_r", origin="lower", interpolation="nearest")
        axes[1, k].imshow(yz, cmap="gray_r", origin="lower", interpolation="nearest", aspect="equal")
        axes[0, k].set_title(title, fontsize=10)
        axes[1, k].set_xlabel(caption(report, g["events"]), fontsize=7, loc="left")
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("MIP along z (x-y)", fontsize=9)
    axes[1, 0].set_ylabel("MIP along x (y-z)", fontsize=9)
    fig.suptitle(f"seed {seed}, {voxel_size:g} um voxels, {shape[0]}x{shape[1]}x{shape[2]} voxels", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mips"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--voxel-size", type=float, default=2.0)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for name, panels in (("stages", PANELS), ("deep", DEEP)):
        grown, volumes, shape = render_all(panels, args.seed, args.voxel_size)
        path = os.path.join(args.out, f"mip_{name}_seed{args.seed}.png")
        figure(panels, grown, volumes, shape, args.voxel_size, args.seed, path)
        print(path)
        for (title, _), (g, report) in zip(panels, grown):
            print(f"  {title}: {caption(report, g['events']).replace(chr(10), ' | ')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
