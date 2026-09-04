#!/usr/bin/env python3
"""
Reports whether a generated network is one connected piece, and when it is not,
whether the volume's own boundary accounts for the breaks.

    python check_connectivity.py --generate 10 --seed 1
    python check_connectivity.py --generate 5 --volume 256 256 70 --iterations 8 8
    python check_connectivity.py output/*.tiff
    python check_connectivity.py output/*.npz output/*.tiff

`--generate N` makes N fresh networks and checks them without writing any files.
Every other option is passed through to the generator, so the networks are the
ones `main.py` would write for the same seeds and settings.

A centreline archive (.npz) is checked as geometry: the branches are walked and
joined wherever they share a point, so the answer is independent of any volume.
More than one component there is a defect in generation or interpretation.

A volume (.tiff) is checked as voxels, under both six-connectivity (faces only,
what a flood fill, `scipy.ndimage.label` or a morphological operation assumes by
default) and 26-connectivity (faces, edges and corners). Every piece other than
the largest is then tested against the six faces of the volume: a piece that
touches one is a branch the volume cut, which is expected whenever the network
is larger than the box it is rendered into. A piece touching no face is a break
the boundary does not explain, and is the signature of a real defect.

The exit status is 1 if any file has a break the boundary does not explain, so
this can gate a pipeline; a network merely clipped by its volume exits 0.
"""
import argparse
import itertools
import os
import sys

import numpy as np

AXES = "xyz"


def label(mask, connectivity=6):
    """
    Labels the connected components of a boolean array by union-find.

    Args:
        mask (ndarray): 3-D boolean array.
        connectivity (int): 6 joins voxels sharing a face, 26 also joins those
            sharing only an edge or a corner.

    Returns:
        tuple: (sizes, labels) with sizes the voxel count of each component in
        descending order and labels a 3-D array of 1-based component indices
        matching that order, 0 outside the mask.
    """
    flat = np.flatnonzero(mask.ravel())
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(mask.shape, dtype=np.int64)

    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[flat] = np.arange(flat.size)
    parent = np.arange(flat.size, dtype=np.int64)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:          # path compression, so repeated finds stay cheap
            parent[a], a = root, parent[a]
        return root

    coords = np.stack(np.unravel_index(flat, mask.shape), axis=1)
    shape = np.array(mask.shape)
    if connectivity == 6:
        offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:                                  # the forward half of the 26-neighbourhood
        offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]

    for offset in offsets:
        moved = coords + offset
        within = np.all((moved >= 0) & (moved < shape), axis=1)
        if not within.any():
            continue
        neighbour = lookup[np.ravel_multi_index(moved[within].T, mask.shape)]
        source = np.flatnonzero(within)
        joined = neighbour >= 0
        for a, b in zip(source[joined], neighbour[joined]):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    roots = np.fromiter((find(i) for i in range(flat.size)), dtype=np.int64, count=flat.size)
    _, index, counts = np.unique(roots, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    labels = np.zeros(mask.size, dtype=np.int64)
    labels[flat] = rank[index] + 1
    return counts[order], labels.reshape(mask.shape)


def report_volume(volume, name):
    """
    Reports the components of a rendered volume, indexed (x, y, z), and how many
    of them the volume's own boundary explains. Returns the number it does not.
    """
    volume = np.asarray(volume) > 0
    total = int(volume.sum())
    if total == 0:
        print(f"{name}: empty volume, no vessels")
        return 0

    sizes, labels = label(volume, 6)
    sizes26, _ = label(volume, 26)
    print(f"{name}: {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}, "
          f"{total} vessel voxels")
    print(f"    components: {sizes.size} six-connected, {sizes26.size} 26-connected; "
          f"largest holds {sizes[0] / total * 100:.1f}%")
    if sizes.size == 1:
        print("    one piece: connected")
        return 0

    unexplained = []
    faces = np.zeros(3, dtype=int)
    for k in range(2, sizes.size + 1):                 # every piece but the largest
        where = np.argwhere(labels == k)
        low = where.min(axis=0)
        high = where.max(axis=0)
        touching = [a for a in range(3)
                    if low[a] == 0 or high[a] == volume.shape[a] - 1]
        if touching:
            for a in touching:
                faces[a] += 1
        else:
            unexplained.append((k, int(sizes[k - 1]), low, high))

    cut = sizes.size - 1 - len(unexplained)
    print(f"    of the {sizes.size - 1} smaller pieces, {cut} touch a face of the volume "
          f"(cut by it: " + ", ".join(f"{AXES[a]} {faces[a]}" for a in range(3)) + ")")
    if unexplained:
        print(f"    {len(unexplained)} piece(s) touch NO face -- the volume does not explain these:")
        for k, size, low, high in unexplained[:10]:
            print(f"        piece {k}: {size} voxels, x {low[0]}-{high[0]}, "
                  f"y {low[1]}-{high[1]}, z {low[2]}-{high[2]}")
        if len(unexplained) > 10:
            print(f"        ... and {len(unexplained) - 10} more")
    else:
        print("    every break is the volume cutting the network, not a defect")
    return len(unexplained)


def check_volume(path):
    """Reads a written TIFF and reports it. Returns the unexplained piece count."""
    import tifffile
    stack = tifffile.imread(path)
    # pages are z, rows y and columns x; the generator indexes (x, y, z)
    return report_volume(np.transpose(stack, (2, 1, 0)), os.path.basename(path))


def report_centreline(nodes, name):
    """
    Reports whether a centreline is connected, independently of any volume.
    Returns the number of components beyond the first.
    """
    finite = np.flatnonzero(~np.isnan(nodes[0]))
    if finite.size == 0:
        print(f"{name}: empty centreline")
        return 0
    gaps = np.flatnonzero(np.diff(finite) > 1)
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps + 1, [finite.size]])
    chains = [finite[s:e] for s, e in zip(starts, ends)]

    parent = list(range(len(chains)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner = {}
    for ci, chain in enumerate(chains):
        for j in chain:
            key = nodes[:3, j].tobytes()      # a branch starts exactly on a point of its parent
            if key in owner:
                ra, rb = find(ci), find(owner[key])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            else:
                owner[key] = ci
    components = len({find(i) for i in range(len(chains))})

    diam = nodes[3][~np.isnan(nodes[3])]
    print(f"{name}: {len(chains)} branches, {finite.size} points, "
          f"diameters {diam.min():.3f}-{diam.max():.3f}")
    print(f"    centreline components: {components}"
          + ("  (connected)" if components == 1 else "  <-- the geometry itself is broken"))
    return components - 1


def check_centreline(path):
    """Reads a saved centreline archive and reports it."""
    from main import load_network
    return report_centreline(load_network(path)["nodes"], os.path.basename(path))


def generate_and_check(count, passthrough):
    """
    Generates `count` networks with the generator's own parser and settings and
    reports each one. Nothing is written to disk. Returns the number of networks
    carrying a break the volume boundary does not explain.
    """
    import random
    from main import build_parser, generate_network, sample_parameters

    args = build_parser().parse_args(passthrough)
    if args.fit == "voxel_size" and args.voxel_size is None:
        raise SystemExit("--fit voxel_size requires --voxel-size")
    base = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
    tVol = tuple(args.volume)
    print(f"generating {count} network(s) at {tVol[0]}x{tVol[1]}x{tVol[2]}, "
          f"fit {args.fit}, clip axes {[AXES[a] for a in args.clip_axes] or 'none'}, "
          f"connect {args.connect}, seeds {base}..{base + count - 1}\n")

    failed = 0
    for index in range(count):
        seed = base + index
        random.seed(seed)                      # the seeding main() uses, so these are its networks
        np.random.seed(seed % (2 ** 32))
        properties, d0, niter = sample_parameters(args)
        if args.d_min is not None and d0 < args.d_min:
            print(f"seed {seed}: --d-min exceeds the sampled root diameter, skipped\n")
            continue
        volume, _, nodes = generate_network(niter, d0, properties, tVol, fit=args.fit,
                                            clip_axes=args.clip_axes, voxel_size=args.voxel_size,
                                            subdivisions=args.subdivisions, d_min=args.d_min,
                                            connect=args.connect)
        stem = f"Lnet_i{niter}_s{seed}"
        broken = report_centreline(nodes, stem + " (centreline)")
        broken += report_volume(volume, stem + " (volume)")
        failed += 1 if broken else 0
        print()
    print(f"{count - failed}/{count} network(s) had no break beyond what the volume cut")
    return failed


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # --generate forwards everything else to the generator's own parser, so peel it
    # off first: a positional list here would swallow operands like "--volume 256 256 70"
    peel = argparse.ArgumentParser(add_help=False)
    peel.add_argument("--generate", type=int, metavar="N")
    peeled, passthrough = peel.parse_known_args(argv)
    if peeled.generate is not None:
        if peeled.generate < 1:
            raise SystemExit("--generate needs a positive count")
        return 1 if generate_and_check(peeled.generate, passthrough) else 0

    parser = argparse.ArgumentParser(
        description="Report the connectivity of generated networks and whether the "
                    "volume boundary explains any breaks.",
        epilog="With --generate, every other option is passed to the generator, so "
               "'--generate 5 --seed 1 --iterations 8 8' checks the networks "
               "'main.py --count 5 --seed 1 --iterations 8 8' would write.")
    parser.add_argument("paths", nargs="+", help=".tiff volumes and/or .npz centrelines")
    parser.add_argument("--generate", type=int, metavar="N",
                        help="generate and check N fresh networks instead of reading files")
    args = parser.parse_args(argv)

    problems = 0
    for path in args.paths:
        if not os.path.exists(path):
            print(f"{path}: no such file", file=sys.stderr)
            problems += 1
            continue
        if path.endswith(".npz"):
            problems += check_centreline(path)
        else:
            problems += check_volume(path)
        print()
    if problems:
        print(f"{problems} break(s) not explained by the volume boundary")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
