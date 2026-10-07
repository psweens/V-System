# V-System: Vascular Lindenmayer Systems for Synthetic Vessel Generation

**V-System** generates **synthetic 3D vascular networks** with stochastic, parametric
**Lindenmayer systems (L-systems)** and renders them as binary image volumes for
training and validating vessel segmentation methods.

The grammars, the turtle interpreter and the bifurcation model follow
Galarreta-Valverde's 2012 dissertation and the SPIE 2013 paper by
[Galarreta-Valverde et al.](https://doi.org/10.1117/12.2007532); daughter diameters
obey Murray's law and bifurcation angles follow Zamir's minimum-volume rule.

![Example](https://github.com/psweens/V-System/blob/master/Lnet_Generations.jpg)

---

## Pipeline

1. **Grammar generation** (`vSystem.py`): the tree rule `F` is expanded for a
   chosen number of drawn generations into a string of turtle instructions.
2. **Interpretation** (`analyseGrammar.py`): the string is executed by a 3D turtle
   that keeps a direction vector and a perpendicular vector, giving the centreline
   points and diameters.
3. **Interpolation** (`utils.py`): each stem is smoothed with a cubic B-spline.
4. **Voxelisation** (`computeVoxel.py`): the network is mapped into the volume
   with a single isotropic scale factor and every segment is drawn as a tapered
   capsule, so calibre and bifurcation angles survive into the image, plus the
   face-connected chain of voxels it passes through, so a vessel thinner than a
   voxel stays unbroken.

Steps 1–3 produce the **centreline**, which is saved alongside the volume. The
centreline is the source of truth for the geometry and the TIFF is one
rasterisation of it, so step 4 can be repeated at any resolution without
regenerating the network.

Three optional stages shape the geometry between steps 2 and 4 without changing
what the grammar decides: a persistent random walk that bends each stem
smoothly (`--tortuosity walk`), collision avoidance between branches
(`--avoid-collisions`) and anastomosis, which joins tips into loops
(`--anastomose`). `--family` bundles them, and `describe.py` measures what
they change. See *Geometry* below.

---

## Units

One grammar unit is **one micrometre**. Nothing in the code enforces this — the
generator is scale free — but every output declares it, so that a dataset cannot
be silently mis-scaled:

- `--d0` and `--d-min` are vessel diameters in micrometres;
- `--voxel-size` is *grammar units per voxel*, and therefore a modality's
  physical voxel size in micrometres per voxel;
- `epsilon` (the length-to-diameter ratio) is dimensionless, so segment lengths
  follow the diameters;
- every JSON sidecar records `"units": "um"`.

A vessel of diameter `d` rendered with `--fit voxel_size --voxel-size v` is
`d / v` voxels across, so calibre in voxels scales as `1 / v` and the same
centreline rendered at two voxel sizes gives two correctly scaled datasets.
Rasterising once and resampling the binary volume afterwards does **not**: it
destroys vessels a voxel wide. Render each modality from the centreline instead.

The `1 / v` law holds down to the grid: a vessel below one voxel across is drawn
at the one-voxel connectivity floor rather than scaled, so its rendered calibre
is overstated. That is the honest rendering of an unresolvable vessel, but it
means a network must be chosen against the voxel size it will be rendered at —
see *Rendering one network for several modalities* below.

To work in another unit, keep the convention self-consistent — interpret `--d0`
and `--voxel-size` in the same unit — and declare it with `--units mm`. The flag
labels the data; it rescales nothing.

---

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/psweens/V-System.git
cd V-System
pip install -e .              # numpy and tifffile only
pip install -e ".[viz]"       # plus matplotlib, for plotting
pip install -e ".[preprocess]"  # plus OpenCV, for resize_volume
```

---

## Usage

Generate five networks into `./output`, reproducibly:

```bash
python main.py --count 5 --out ./output --seed 1
```

Each network is written as three files sharing the stem
`Lnet_i<generations>_s<seed>`:

| File | Contents |
| --- | --- |
| `.tiff` | a uint8 volume of 0 and 255 whose pages are z, rows y and columns x |
| `.npz` | the centreline: `nodes`, the (4, N) array of x, y, z and diameter in grammar units with NaN column separators; `edges`, the (2, E) graph of column indices; `node_kind` and `tree`, per-column labels; `program`, the grammar string; and `metadata`, the sidecar record |
| `.json` | the seed, the unit convention and every parameter used |

Network *i* of a run uses `seed + i`, and the same seed and parameters reproduce
the same volume. The default volume is an imaging slab, fitted in x and y and
clipped in z, so vessels leave and re-enter it and a deep network renders as
several pieces; see *Connectivity* below for what that means and how to change
it. The centreline archive grows with the number of drawn
generations rather than with the volume — tens of kilobytes at four generations
and about seven megabytes at twelve, against 37 MB for a `512 × 512 × 140`
volume whatever the network inside it. Its arrays are exact; being a zip, its
entries carry a modification time, so compare the arrays rather than the bytes.

Useful options (`python main.py --help` lists them all):

| Option | Default | Meaning |
| --- | --- | --- |
| `--volume NX NY NZ` | `512 512 140` | volume shape in voxels |
| `--iterations MIN MAX` | `4 12` | drawn generations, drawn uniformly |
| `--d0 MEAN STD` | `20 5` | root diameter, grammar units, truncated at `--d0-min` |
| `--d-min` | none | smallest drawn vessel diameter; a branch stops bifurcating below it |
| `--epsilon MIN MAX` | `4 10` | length-to-diameter ratio of a segment |
| `--randmarg MIN MAX` | `0.1 0.3` | relative half-width of the segment-length distribution |
| `--sigma` | `5` | d_opt / sigma is the spread of the first daughter diameter |
| `--roll-angle` | `70` | roll of the bifurcation plane after each daughter turn, degrees |
| `--stem-angle` | `25` | turn between the five sub-segments of a stem, degrees |
| `--aneurysm-prob`, `--stenosis-prob` | `0.02` | per sub-segment probability of a local ×1.5 dilation or ×0.5 constriction |
| `--fit` | `isotropic` | `isotropic`, `voxel_size` (fixed `--voxel-size`) or `stretch` (legacy per-axis fill) |
| `--voxel-size` | none | grammar units per voxel for `--fit voxel_size`: the modality's voxel size |
| `--clip-axes` | `z` | axes left out of the isotropic fit and clipped, as an imaging slab would |
| `--grow-in-volume` | off | confine growth to the volume's proportions so no vessel is cut |
| `--no-connect` | off | rasterise bare capsules, leaving sub-voxel vessels dotted |
| `--units` | `um` | the unit one grammar unit stands for, recorded in the sidecar |
| `--family` | `tree` | preset bundle of the geometry options: `tree`, `mesh`, `tumour`, `aligned`, `aligned_tight` (the last two need `--d-min`) |
| `--tortuosity` | `stems` | `stems`: five sub-segments smoothed by a B-spline; `walk`: a persistent random walk of the same arc length |
| `--persistence` | none | persistence length of the walk in vessel diameters; required by `walk` |
| `--avoid-collisions` | off | keep branches apart by at least `--collision-margin` (default 1 µm), redrawing or shortening, and count what could not be placed |
| `--anastomose` | off | join `--anastomosis-fraction` (0.5) of the tips to partners within `--anastomosis-radius` (25) tip diameters, never closer kin than `--anastomosis-min-separation` (3) segments, by bridges routed clear of the network at `--collision-margin`; `--anastomose-mode arteriovenous` grows a second tree from the opposite face |
| `--guidance JSON` | none | steer the walk towards an axis or a plane per calibre class: a list of rules, each with a bound `below` in d_min (null for none), a `field` (`axis`, `plane` or null), its `axis` or `normal`, a `length` G in diameters, an `onset` (2) and, for an axis, a `sense` and `polarity`, for a plane `bank`; needs `--tortuosity walk` |
| `--root-offsets JSON` | none | move the roots from their default positions on the faces of the growth box, staying inside it: one `[x, y, z]` per tree in units of d_min; needs `--grow-in-volume` and `--d-min` |

`--d-min` and `--iterations` are both stopping criteria and whichever comes first
wins. `--d-min` is the one a modality states directly, as its smallest resolvable
calibre; with `--deterministic` daughters, diameters fall by exactly `2^(-1/k)` a
generation and it is reached after `log(d0 / d_min) / log(2^(1/k))` of them.
Under the default stochastic daughter diameters that is only an estimate, and
the two daughters of a bifurcation reach it at different depths, so the tree
terminates unevenly. Either way, raise `--iterations` above the estimate to let
`--d-min` decide. It bounds the diameter a branch is *drawn* at, not the
diameter after a local anomaly: a stenosis still narrows a drawn sub-segment by
`--stenosis-prob`'s factor of 0.5, so set `--stenosis-prob 0` for a hard floor.

### Connectivity

A capsule sets only the voxels whose centres it contains. A vessel thinner than
a voxel contains no centre along much of its length, so on its own it rasterises
as a dotted line and a connected tree falls apart into fragments — with the
default parameters, more than a third of the centreline is that thin and a
single tree renders as dozens of pieces. Every segment is therefore also drawn
as the chain of voxels it passes through, walked one face at a time
(Amanatides & Woo, 1987), which renders a sub-voxel vessel one voxel wide
instead of dotted. It is not a dilation: every voxel the chain sets lies within
`sqrt(3)/2` of the centreline, so for a radius of 0.866 voxels or more it is
already inside the capsule and calibre is untouched. `--no-connect` restores the
bare capsule rasterisation.

The chain is **face-connected** (6-connected), not merely 26-connected. The
distinction matters downstream: a chain of voxels touching only at edges or
corners is one piece to a 26-connected label but shatters under the
six-connected flood fill, label or morphological operation most tools default
to — `scipy.ndimage.label` among them — and looks like a string of beads in a
viewer. Face connectivity is the weakest assumption any of them makes, so it is
the one the rasteriser guarantees.

`check_connectivity.py` answers the question for a given network. It reports the
components of a volume under both 6- and 26-connectivity and, for every piece
but the largest, whether that piece touches a face of the volume — a piece that
does is a branch the volume cut, a piece that does not is a real break. It also
checks a saved `.npz` centreline as pure geometry, independently of any volume,
where anything but one component is a defect. `--generate N` makes fresh
networks to check instead of reading files, passing every other option through
to the generator, and the exit status is 1 only for a break the volume boundary
does not explain:

```bash
python check_connectivity.py --generate 10 --seed 1
python check_connectivity.py output/*.npz output/*.tiff
```

Two ways of looking can still suggest breaks that are not there. A thin vessel
running obliquely through a stack shows up in any single slice as isolated
voxels even though it is unbroken in three dimensions, so judge connectivity
with a 3-D component count rather than slice by slice. And a rendered network
does still fall into several genuine pieces under two conditions, neither of
them a defect:

- **Clipping.** A branch that leaves a slab and re-enters it is two vessels in
  the image, exactly as it would be in a real acquisition. `--grow-in-volume`
  removes this at the source: growth is confined to a box with the volume's
  proportions and a branch that would leave it is terminated along with its
  subtree, so the tree takes the volume's shape and no vessel is ever cut part
  way along. On a `512 × 512 × 140` slab at twelve generations that turns 43
  pieces into one while raising the vessel count, because branches that used to
  grow out of the slab and be discarded now stay inside it. `--clip-axes` is
  unused when it is set, and it is off by default so existing output is
  unchanged. A volume with equal axes needs none of this: the grammar grows a
  roughly cubical tree, so `--volume 512 512 512 --clip-axes none` is one
  connected piece at the same calibre. The default volume
  is such a slab — fitted in x and y, clipped in z — so this is what the
  quickstart command produces: its deepest network, at twelve generations,
  renders as 48 pieces, every one of them touching the top or bottom face of
  the slab, with 95% of the vessel voxels in the largest. `--clip-axes` and
  the volume shape control this. With `--clip-axes none` each of the five
  quickstart networks renders as a single connected component, at the cost of
  scaling the whole tree into 140 slices and so shrinking every vessel.
- **Resolution.** A vessel the modality cannot resolve is still drawn, one voxel
  wide. To stop generating them instead, set `--d-min` to the smallest
  resolvable calibre.

`--clip-axes` is read by the isotropic fit only; `voxel_size` and `stretch`
ignore it. The command line clips z by default because the default volume is an
imaging slab, whereas the library functions `computeVoxel.process_network` and
`fit_to_volume` default to clipping nothing, so that a direct call fits the whole
network unless told otherwise. Axes may be named or indexed in either place, and
an axis that is neither raises.

From Python:

```python
from main import generate_network, save_network

volume, program, nodes = generate_network(
    niter=8, d0=20.0, properties={"epsilon": 7.0, "randmarg": 0.2}, tVol=(512, 512, 140))
save_network("Lnet.npz", nodes, program=program)
```

`volume` is a uint8 array of 0 and 1 indexed (x, y, z); `program` is the grammar
string; `nodes` is the (4, N) centreline of x, y, z and diameter with NaN columns
separating branches. `nodes` is in grammar units and independent of `tVol`, `fit`
and `voxel_size`, so it is the artefact worth keeping.

### Rendering one network for several modalities

Because the mapping from grammar units to voxels happens at rasterisation time, a
saved centreline can be rendered at each modality's own voxel size. The field of
view is `shape * voxel_size`, so `shape` has to follow the voxel size to keep it
fixed; deriving it from the network's own extent does that:

```python
import numpy as np
from computeVoxel import process_network
from main import load_network

# written by: python main.py --count 1 --seed 1 --iterations 8 8 --out output
network = load_network("output/Lnet_i8_s1.npz")
nodes = network["nodes"]
extent = np.nanmax(nodes[:3], axis=1) - np.nanmin(nodes[:3], axis=1)  # micrometres

for modality, voxel_size in [("two-photon", 1.0), ("light-sheet", 2.0)]:
    shape = np.ceil(extent / voxel_size).astype(int) + 8   # same field of view, finer grid
    volume = process_network(nodes, shape, fit="voxel_size", voxel_size=voxel_size)
```

Only the sampling changes between the two, so each result carries the vessel
calibre distribution at that modality's resolution. A 20 µm vessel is 20 voxels
across at 1 µm and 10 voxels across at 2 µm.

Holding `shape` fixed instead would render the same network into two different
fields of view. That is the mistake to avoid: at 20 µm/voxel a
`512 × 512 × 140` volume covers 10 240 × 10 240 × 2800 µm, and the network above
occupies 37 × 33 × 21 of its 36.7 million voxels.

**One centreline serves a modality only while that modality can resolve it.**
Every vessel thinner than a voxel is drawn at the one-voxel connectivity floor,
so once most of the network is sub-voxel its calibre distribution collapses to a
spike at 1 voxel instead of matching anything. Check before trusting a render:

```python
d = nodes[3][~np.isnan(nodes[3])] / voxel_size          # diameters in voxels
print(np.percentile(d, [50, 90]), (d < 1.0).mean())     # ... and the sub-voxel fraction
```

For a modality whose voxel size approaches the network's finest vessels,
generate a network for it — a larger `--d0`, or `--d-min` set to its smallest
resolvable calibre — rather than rendering an existing one more coarsely.

To target a fixed acquisition geometry instead, a `512 × 512 × 140` slab at
2 µm say, choose `--d0`, `--epsilon` and `--iterations` (or `--d-min`) so that
the network spans the field of view `shape * voxel_size`: under `voxel_size` the
network is centred and clipped rather than scaled to fit.

---

## Geometry

The grammar fixes the tree's topology, calibres, segment lengths and
bifurcation angles. What a segmentation model trained on these volumes also
learns is the *local* geometry — how smoothly vessels curve, whether they cross,
how often they end — and the plain grammar has three properties real
vasculature lacks: stems are zig-zags of five straight pieces, branches pass
through one another, and every terminal branch ends in a free tip. Three
opt-in stages address them. None is used unless asked for, and the default
command line writes the same `nodes`, `program` and TIFF as before for a given
seed (`tests/fixtures` pins this).

### The graph in the archive

Every archive carries `edges`, a `(2, E)` array of column indices into `nodes`:
one edge between consecutive points of a polyline and, because a daughter
polyline starts on a bitwise copy of its parent's branch point, one shared
vertex wherever polylines meet. Coincident columns (within `graph.DEFAULT_TOL`,
10⁻⁶ µm) are one vertex, represented by the lowest column index. `node_kind`
labels every column interior (degree 2), junction (degree 3 or more) or tip
(degree 1; the root counts as a tip), and `tree` labels the tree a column
belongs to (0, 1) or marks it as an anastomosis bridge (2). `graph.py` rebuilds
all of this from `nodes` alone — `edges_from_nodes`, `node_kind`, `betti`,
`segments` — so an archive written before this version gains a graph when read
with `load_network`. The rasteriser ignores the graph.

### Smooth tortuosity: `--tortuosity walk --persistence P`

A stem is normally the grammar's five straight sub-segments with alternating
turns of `--stem-angle`, smoothed by an approximating B-spline; its arc/chord
ratio is fixed by the stem angle and does not vary. With `walk`, the path of
each stem is a persistent random walk on the unit tangent: at every step of
length h (a sub-segment divided into 2^`--subdivisions` steps) the tangent is
rotated by an angle drawn from N(0, √(h / l_p)) about a random perpendicular
axis, with persistence length l_p = P × the local diameter. Total arc length
equals the grammar's, the daughters leave from where the walk actually ends
and at the grammar's angles relative to the walk's final direction, and the
walk draws from its own random stream seeded from the run seed, so the
grammar's draws are untouched. Lower P is more tortuous; `docs/geometry`
tabulates arc/chord and curvature against P. Under `--grow-in-volume` a step
that would leave the box is redrawn up to a budget, turning more sharply each
time, then the branch terminates and the sidecar counts it.

The rotation rule gives a tangent autocorrelation of exp(−s / 2 l_p) in three
dimensions, so P is a shape parameter calibrated against measured arc/chord
ratios rather than a literal persistence length; `tortuosity.py` documents the
factor of two.

### Collision avoidance: `--avoid-collisions`

Every accepted point is stored in a spatial index (`--collision-index grid` or
`kdtree`; `auto` is the grid hash, which was faster than the k-d tree at both
10⁵ and 10⁶ points: 1.3 s against 1.8 s, and 16 s against 60 s, for the
insert-and-query mix of `python spatial.py --benchmark`). A proposed point
collides when it lies closer than the sum of the two radii plus
`--collision-margin` to a stored point that is not on its own stem within an
arc-length window, nor on its parent or sibling close to their common junction
(the two distances to the junction, measured along the vessels, summing to
less than 1.7 times the collision threshold, which is how far tubes meeting at
Zamir's angles overlap; `collisions.py` derives the bound). A colliding walk step is redrawn up to
`--collision-attempts` times, each redraw turning more sharply than the last
(the rotation's spread grows by one multiple of its base per attempt) so the
walk steers away from what it hit rather than repeating almost the same step;
a spline stem is shortened to its longest clear prefix of control points; a
branch that cannot be placed terminates. The
sidecar counts redraws, shortened stems and terminations. Terminated branches
are new tips, so avoidance raises tip density unless anastomosis absorbs them.
The check is on centreline points, so keep the margin at or above the point
spacing of the finest vessels. `describe.py` verifies the result independently
as the minimum surface clearance over the whole network.

### Anastomosis: `--anastomose`

After growth, a seeded random fraction of the tips (`--anastomosis-fraction`)
each search within `--anastomosis-radius` tip diameters for a partner: another
tip first, otherwise an interior point of a segment that is not their own,
nearest first, never a junction, a root, or a point inside a junction's
overlap zone, and never closer kin than `--anastomosis-min-separation`
segments along the tree (default 3, which rules out the parent stem and the
sister stem, two segments away). Without that rule the commonest bridge joins
a tip to its sister's tip, usually the nearest, closing the two sister stems
into a small triangle. The nearest partner beyond the sister lies about
2 ε tip diameters away (8 at ε = 4, 14 at ε = 7, 20 at ε = 10), which is why
the default radius is 25; a radius in multiples of the segment length would
be self-scaling and is an open question.
A branch that avoidance or the growth box terminated within a step or two of
its junction leaves a stub whose tip sits inside that zone; such stubs are
neither sources nor partners, and are counted. Partners are sought ahead of the tip, within 120° of its
direction, as a sprouting tip fuses with what it grows towards rather than
doubling back. A bridge of diameter min(d_tip, d_partner) is traced by the
pinned form of the walk: it leaves the tip along the tip's direction, arrives
into a partner tip along that vessel's direction (so the two tips become one
continuous vessel) or into the side of a vessel along the chord, and carries
the walk's curvature at `--persistence` diameters, or at 8 when the stems are
not walked, so that it is a vessel rather than a straight strut. Every
bridge is routed clear of the network at `--collision-margin`, whether or not
the tree was grown with `--avoid-collisions`, redrawn on a collision and
abandoned (and counted) when no candidate clears. It is appended as a new
polyline whose end columns copy the two joined points, so the graph gains a
cycle (β₁ = E − V + C rises by one) or joins two components. `--anastomose-mode arteriovenous` grows a second tree from the
opposite face of the volume, heading back towards the first, and ranks partners
in the other tree first — the arteriole → capillary → venule design of a
capillary bed; it works best with `--grow-in-volume`, which the `mesh` preset
turns on. The second root is placed at the nearest spot on its face that is
clear of the first tree, and both root positions are recorded in the sidecar. Tips that found no partner, or whose every candidate bridge collided,
are counted. The rasteriser draws polylines independently, so cycles need no
special handling.

### Directional guidance: `--guidance` and `--root-offsets`

The walk can be steered. Before every step's random turn, the heading h is
turned deterministically towards a target t by

    ω = min(θ, r sin θ cos θ)   for a nematic axis or a plane,
    ω = min(θ, r sin θ)         for a polar axis,

where θ is the angle from h to t and r = min(1, step / (G d)) with G the
guidance length in local diameters d. The drift draws nothing: with guidance
off the walk draws exactly what it drew before, and with guidance on it still
draws two numbers per attempt, each redraw turning about the steered frame.
The target depends on the field of the rule that governs the stem: for a
nematic axis a it is the nearer of a and −a; for a polar axis it is s a with
a sense s fixed per tree, +1 (`fixed`), the sign of the tree's root heading
along a (`root`), or the sign along a of the partner tree's root offset
minus this tree's (`partner`, from the `--root-offsets` entries, so it
points towards the partner root when the axis lies in the roots' faces, as
in `aligned`); for a plane with normal n it is the
projection of h onto the plane, undefined when h lies along n. A plane rule
may also `bank`: after the turn the frame's perpendicular is turned towards
the normal by the same rule, so that the branching turns the grammar writes
about the perpendicular stay in the plane despite the roll of 70° before
every daughter.

The rules are a JSON list, each governing the stems whose diameter at the
first move lies below `below` × d_min and above the bound of the rule before
it, so the rules partition the calibres from the finest upwards; the last
rule may have `below` null and a rule with field null leaves its class
unguided. A stem is not steered while the arc walked along it is below
`onset` × its diameter, so that daughters clear their sisters before being
pulled into line; the rule is chosen once per walked polyline, from the
diameter of its first move, and held, so a local anomaly does not switch
class part way along. Bare moves and spline stems are never steered, so
`--guidance` needs `--tortuosity walk`, and a finite bound needs `--d-min`.
`--root-offsets` adds one vector per tree, in units of d_min, to the roots'
default positions on the faces of the growth box before the second root is
cleared of the first tree; a component along a face's normal moves its root
into the box. It needs `--grow-in-volume` and `--d-min`, and a root moved
outside the box is refused. Every refusal fires before the network it
concerns is written; a malformed rule or offset is refused before anything
is written, but without `--fit voxel_size` the box comes from each grown
tree, so a root outside it is refused when that network grows, after the
earlier networks of a `--count` run have been written.

The law. The walk diffuses the heading with D = 1 / (4 P d) per unit length
and the drift descends Φ = −cos²θ / 2 (nematic), −cos θ (polar) or
sin²β / 2 (plane, β the angle out of the plane), so on a stem much longer
than G d the heading settles into a stationary law: Watson exp(K cos²θ) with
K = 2P / G for a nematic axis, Fisher exp(κ cos θ) with κ = 4P / G for a
polar axis, and the girdle exp(−K sin²β) with K = 2P / G for a plane; the
small-angle rms angles are √(G / 2P) to an axis and √(G / 4P) out of a plane.
G follows from a target K. The steps are finite, so the measured
concentration carries a discretisation bias of order step / (G d):
`tests/test_guidance.py` grows one stem of 2000 sub-segments at P 10 and
step 0.175 d and measures, over seeds 1 to 4 at G 3, ⟨cos²θ⟩ = 0.821,
⟨cos θ⟩ = 0.922 and ⟨sin²β⟩ = 0.078 against 0.830, 0.925 and 0.0747 from
the laws: the laws' moments at 0.96 of their concentrations, at
step / (G d) = 0.058. The nematic law is
Watson, not Fisher-axial: Fisher-axial is exp(K |cos θ|), whose ⟨|cos θ|⟩ is
coth K − 1 / K, the polar law's ⟨cos θ⟩ at κ = K up to the polar law's
backward mass.

Every run counts its steering: `guided_steps`, `guidance_onset_steps`,
`unguided_steps` and `guidance_undefined_steps` partition the walk's steps,
`guidance_sense_flips` counts the changes of the nearer end of a nematic axis
between consecutive guided steps of one polyline, and `guidance_bank_steps`
counts bank rotations. Guidance costs about 90 µs per guided step on an
Intel Xeon Gold 5220 workstation (Linux, Python 3.12, numpy 2.5), and a bank
about 45 µs more (`tests/test_topology_benchmark.py`): that doubles the cost
of a bare walk step (89 µs there), but adds about 7% to a network grown with
avoidance and anastomosis, the `aligned` preset taking 1.38 ms per point at
R 8 against 1.29 ms with guidance None (seeds 1 to 3, 87 thousand points in
all either way).

Every network grown from 3.5 records its **frame** in the sidecar and in the
archive's metadata: the kind of the field rule with the smallest bound
(`none`, `axis` or `plane`), its axis or normal, the sense and, for a polar
rule, the +1 or −1 of each tree, every rule normalised, the growth frame
(`grow_direction`, `grow_perpendicular`) and an origin (the centre of the
growth box, else the first root), all in the coordinates and unit of `nodes`.
With guidance off the kind is `none` and the rules are empty. `frames.py`
places networks by their frame (below), and `describe.py` measures against
it.

### Families: `--family`

`tree` is the plain grammar. `mesh` is walk (P = 10) + collision avoidance +
arteriovenous anastomosis of half the tips, grown in the volume. `tumour` is a
low-persistence walk (P = 3), avoidance, anastomosis of 80% of the tips within
one tree, a root calibre of 20 ± 10 µm and aneurysm and stenosis probabilities
of 0.1. `aligned` is an arteriovenous pair in the mesh layout whose roots are
moved by −15 and +15 d_min along x, 30 d_min apart (`root_offsets`), and whose
vessels below 2 d_min are steered along x towards the other tree's root (a
polar axis with polarity `partner`, G 4.9, so κ = 4P / G ≈ 8.2, after an onset
of 2 diameters) while the larger vessels are kept in planes perpendicular to x
(a plane with normal x, G 5, girdle K = 4): each tree feeds one sheet and the
capillaries run from sheet to sheet, with 80% of the tips seeking an
arteriovenous partner. It imitates capillaries running along muscle fibres
from terminal arterioles to collecting venules with cross-connections (Skalak
and Schmid-Schönbein 1986; Sarelius 1986; Emerson and Segal 1997), whose
orientation has been fitted with a Fisher-axial law (Mathieu et al. 1983;
Mathieu-Costello 1987). It needs `--d-min`, and its guidance values are
provisional. `aligned_tight` is `aligned` with its capillaries steered harder
and from their first step (G 2, so κ = 4P / G = 20, and an onset of 0 for
both rules) and its roots moved by −20 and +20 d_min along x, 40 d_min apart,
in a cube of side 20 R in a library; it also needs `--d-min`, and its values
are provisional too. Released presets are frozen, so a changed value takes a
new family name. Options given explicitly override the preset. The persistence
values of the presets are provisional calibrations from the sweep in
`docs/geometry`. A preset listed as None in `main.FAMILIES` is a family this
version does not offer, and is refused.

### Cost

The default command line is unchanged in what it draws and about 8% slower at
twelve generations (21 s against 20 s for a 512 × 512 × 140 volume on one
core), the difference being the graph built for the archive; shallow trees
take the same 1.5 s as before. The opt-in stages are Python loops over walk
steps: at twelve generations the walk adds a quarter, stems-mode avoidance a
half, walk with avoidance takes 2.5×, with anastomosis 4×, and the two-tree
`mesh` preset 7× (146 s). `docs/geometry/audit.md` has the table. The
`aligned` preset of 3.5 costs about what a mesh costs: grown as the library
grows it on an Intel Xeon Gold 5220 workstation, it takes 8.8, 49 and 185 s
at R 4, 8 and 16 (means over seeds 1 to 3, 6.6, 28 and 87 thousand points)
and 600 s at R 25 (298 thousand points, peak RSS 356 MB), where a mesh takes
6.7, 29, 199 and 818 s (`tests/test_topology_benchmark.py`).

### Measuring: `describe.py`

```bash
python describe.py output/Lnet_i8_s1.npz
python describe.py output/*.npz --margin 1 --volume 1024 1024 280
```

reports, as JSON, length-weighted diameter percentiles and total length,
per-segment arc/chord ratios and mean absolute curvature, tip count and tips
per mm, the junction degree histogram, components β₀ and cycles β₁, the
length-weighted tangent covariance and its anisotropy, length density, the
minimum surface clearance between branches with the number of pairs closer
than the margin, and every counted event from the sidecar. The docstring of
`describe.describe` is the definition of each quantity, so that the same
descriptors can be computed from voxel skeletons of real images and compared.

From 3.5, `describe(nodes, edges, ..., frame=None, d_ref=None,
class_bound=2.0)` also measures against the network's frame and by calibre
class. `frame` defaults to the metadata's frame record and `d_ref` to the
metadata's `d_min` (top level, else `grow_kwargs`); an edge is capillary-class
when its diameter is below `class_bound` × d_ref, and the tangent of an edge
runs from its lower to its higher column, the direction of growth for a
network the generator wrote. The appended keys are `frame` (a copy), `classes`
(the bound, d_ref and where it came from), `frame_orientation` (per class
about an axis: the order parameter S = ⟨(3 (t·a)² − 1) / 2⟩, ⟨|t·a|⟩ and its
reciprocal the crossing ratio, the mean angle, the fractions within 20° and
45°, and the Watson K and `fisher_axial_K` that invert ⟨(t·a)²⟩ and ⟨|t·a|⟩;
about a plane: the in-plane fraction ⟨1 − (t·n)²⟩ and S_n), `polar_order`
(polar axis frames with tree labels: ⟨s_k t·a⟩ with each tree's sense s_k,
bridges excluded), `orientation_by_class` (the tangent covariance per class
with S_max and the planarity 1 − 3λ₃), `calibre_shares` (of length and of
volume below 1.5, 2 and 3 d_ref), `segments_by_class` (segment lengths
between junctions and tips, split by their mean diameter) and
`transverse_spacing` (axis frames: on nine planes perpendicular to the axis
over the central 80% of the capillary extent, the nearest-neighbour distance
between the crossings of capillary edges and the crossings per unit area of
plane inside the growth box, else inside the capillary bounding rectangle).
Lengths come in the archive's unit and in d_ref (suffix `_d`). An archive
without a frame reports `frame` None and every frame-relative key None; one
without a d_ref reports every class-dependent key None. The pre-3.5 keys are
unchanged. `describe_archive` with every new key takes 21 s on the largest
network `tests/test_topology_benchmark.py` grows, a tree of 1.08 million
points at R 25, and 1.9 s on an aligned network of 139 thousand points at
R 16, on an Intel Xeon Gold 5220 workstation.

### Placing networks by their frame: `frames.py`

`frames.py` is a caller-side module, outside the generator's import closure
and importing nothing but numpy, that moves an archive and its frame record
together: `transform_nodes(nodes, Q, t, scale)` gives xyz' = scale Q xyz + t
with diameters scaled, and `transform_frame(frame, Q, t, scale)` rotates
every vector of the record, the rules' included, and scales the origin, so
that a descriptor measured about the frame's axis is the same before and
after the move. `rotation_about(axis, angle)`, `rotation_between(u, w,
nematic=False)` (a proper rotation taking u to w, with w flipped first when
nematic and u·w < 0, in a form that does not lose precision near antiparallel
pairs), `rotation_of_frames(u1, u2, w1, w2)` and `random_rotation(rng)` (Haar)
build the rotations, and `align(frame, axis=A, spin=φ)` (or `normal=N`) takes
the frame's axis (or normal) to the given one and spins about it. Two recipes:
a bank of networks along one shared axis A takes Q_i = align(frame_i, axis=A,
spin=U(0, 2π)); random placement takes Q_i = random_rotation(rng). Rescaling
is a transform with Q the identity and the same scale for nodes and frame.
A missing record (every archive before 3.5), a frame of kind `none`, an axis
asked of a plane frame or a normal of an axis frame, and any Q that is not a
proper rotation to 10⁻⁹ are refused. `join.py` needs nothing of this.
`docs/geometry/sweep.py` regenerates the descriptor tables in `docs/geometry`,
and `docs/geometry/mips.py` renders one seed under each option as
maximum-intensity projections (`docs/geometry/mips`), so the effect of every
stage can be seen as well as measured.

### Joining networks inside a field of view: `join.py`

A volume assembled from many independently grown networks is a forest: every
network is its own connected component and ends in free tips everywhere,
whereas the vasculature inside a real field of view is essentially one
connected network whose only free ends are the vessels the field's edge cuts.
`join.join_networks` applies the anastomosis rules across a forest. The caller
places the networks (rotates and translates them into one frame and unit);
`join_networks` adds bridges and returns them separately from the inputs.

```python
import numpy as np
from join import crop_network, join_networks

lo, hi = np.zeros(3), np.array([512.0, 512.0, 140.0]) * 2.0   # the field of view, caller units
margin = 1.0                                                  # clearance between vessel surfaces
r_max = max(np.nanmax(n["nodes"][3]) for n in placed) / 2.0
cropped = [crop_network(n, lo, hi, r_max + margin) for n in placed]
result = join_networks(cropped, np.random.default_rng([seed, 3]), fraction=0.9,
                       collision_margin=margin, box=(lo, hi), boundary_margin=2.0)
```

Each placed network is a dict with `nodes` ((4, N) x, y, z, diameter with NaN
separators), `node_kind` (the kind each column had in the *uncropped*
network), and optionally `tree` and `roots` (root columns, empty meaning
none). A network's identity is its position in the sequence, an int32, never
the int8 `tree` label, so hundreds of networks can be joined. Edges on input
are ignored: each network's graph is rebuilt from its own coordinates with
`tol`, per network, so networks never merge by coincidence.

`crop_network(network, lo, hi, margin)` crops by whole columns: a column is
kept when either segment it belongs to has a radius-padded bounding box
meeting `[lo - margin, hi + margin]`, kept columns stay in archive order, and
each dropped run becomes one NaN column. Polylines are never split into
per-segment pairs: the tip tangent, the sampling step, the stub test and the
collision excusal are all per polyline, and a tree split into two-column
chains makes every one of its tips a stub. It returns the cropped `nodes`,
`node_kind` and `tree`, the `roots` remapped, and `columns`, the original
column of each kept one (-1 at separators). Make the crop margin at least the
largest vessel radius plus `collision_margin`, so that bridges near a face are
checked against the vessels just outside it. A network with no kept column
comes back empty and `join_networks` accepts it.

Keyword arguments, all required unless a default is shown:

| Argument | Meaning |
| --- | --- |
| `rng` | a `numpy.random.Generator` seeded by the caller; `join.RNG_STREAMS["join"]` is the tag to pair with a run seed |
| `fraction` | share of the eligible tips drawn as sources |
| `collision_margin` | clearance every bridge keeps from every input vessel and every earlier bridge, caller units |
| `box` | `(lo, hi)`, the field of view in caller units |
| `boundary_margin` | a tip within this distance of a face is a cut end, caller units |
| `radius=25.0` | partner search radius in tip diameters |
| `policy="cross"` | `cross`: partners on other networks only; `any`: also the same network at a graph separation of at least `min_separation` segments (another component of the same network counts as far enough) |
| `min_separation=3` | the kin rule of anastomosis under `any`; 3 excludes the parent and sister stems |
| `persistence=8.0` | curvature of the bridge walk, in bridge diameters |
| `attempts=10`, `max_candidates=5` | redraws per candidate, partners tried per tip |
| `tol=graph.DEFAULT_TOL` | coincidence tolerance of each network's graph, caller units |
| `max_bridge_volume=None` | budget for the summed bridge volumes, caller units cubed |
| `merged=False` | also return inputs and bridges as one archive with its graph |
| `events=None` | counters incremented in place (`join.EVENT_KEYS`) |
| `attach_roots=False` | let each eligible root attach, as a side branch, to a vessel of another network at least as thick (below) |
| `root_fraction=1.0` | share of the eligible roots drawn as sources |
| `root_partner_min_ratio=1.0` | a root's partner must be at least this many root diameters across, measured at the vertex; at least 1 |
| `root_radius=None` | partner search radius for a root in root diameters; `None` uses `radius` |

Roots come from `roots` when given; otherwise the first finite column of each
`tree` label other than the bridge label (so both trees of an arteriovenous
mesh are roots), or the first finite column when `tree` is absent, a default
valid only for uncropped networks. A root is never a source or a partner,
unless `attach_roots` makes the roots inside the box sources (below), and
is reported as `root` inside the box, `cut_end` outside it. A degree-one
vertex is a `cut_end` when the crop made it (its input `node_kind` is not a
tip) or when it lies outside the box or within `boundary_margin` of a face;
cut ends are vessels that continue beyond the field and are never sources or
partners. A network's own anastomosis bridges (`tree == 2`) are ordinary
vessels of that network.

The rules kept from `anastomose`: partner search within `radius` tip
diameters; partners within 120° of the tip's direction; candidates ranked tip
before interior point, then by distance, then by (network, column);
exclusions of the tip's own polyline, kin closer than `min_separation` under
`any`, junctions, roots, stubs, junction zones, the zone of the junction the
tip's polyline leaves, and points already touching; bridge diameter
min(d_tip, d_partner); step the median spacing of the tip's polyline (0.2
d_tip when undefined); departure along the tip tangent and arrival along a
partner tip's inward tangent; the walk's curvature at `persistence`
diameters; the collision rule of `collisions.py` (strict r + r + margin
against every input vessel and every earlier bridge, excused only along the
tip's and partner's polylines, plus the bridge against itself); bitwise end
columns, so the graph closes; a tip used as a partner stops being a source
and a bridged interior point becomes a junction. The deviations: bridge
points are collision obstacles only, never partners, so every bridge end is
a (network, column) of an input network; partners must lie inside the box;
a bridge whose centreline leaves the box counts as a redraw
(`join_box_redraws`); the selected tips are processed in a permutation drawn
right after the Bernoulli draws, so the result does not depend on the order
of the networks beyond the draws. The walk takes its angle and axis draws in
bulk, one array of each per bridge, where `tortuosity.bridge_path`
interleaves them step by step, so the same generator state gives a different
(equally distributed) bridge here.

In a forest packed into a field of view, many networks have their root
inside the field, and each such root is a blunt start; real vasculature has
none, since a vessel's upstream end is a branch off a parent vessel. With
`attach_roots`, an *eligible* root (a degree-one root vertex inside the box,
clear of the faces by `boundary_margin`, whose input `node_kind` is a tip;
whether it lies in a junction zone does not matter) becomes the source of a
bridge to a vessel of another network at least as thick, so that the blunt
start becomes a side branch, with the machinery of the tip bridges. The
root's upstream direction is the unit vector from the first distinct point
of its polyline to the root, which points out of the vessel; the bridge
departs along it and the 120° cone is taken about it (a root with no such
point gets no cone and a chord departure, as a tip without a tangent does).
A root's partner is a vertex of degree one or two that the partner rules
above allow, on another network whatever `policy` is, not already touching
the root, inside the cone, within `root_radius` root diameters, and whose
vertex diameter (the largest over the columns it gathers, so an aneurysm
point counts at its bulged diameter) is at least `root_partner_min_ratio`
times the root's; the zone of the junction the polyline leaves (a disc
around the root itself) and the kin rule (no same-network partner is
possible) do not apply. Interior points rank before tips, then distance,
then (network, column), so a root prefers a side branch to a tip-to-tip
junction; as for tips, a tip already bridged has degree two and counts as an
interior partner. The bridge has the root's diameter, not the smaller of
the two, so the vessel continues upstream at its own calibre; its step,
persistence, redraws, collision rule and excusals, bitwise end columns, the
junction at an attached interior point and the consumed attached tip are
those of a tip bridge. An ineligible root (outside the box, within
`boundary_margin` of a face, or not a tip of the uncropped network) is
reported as `cut_end`, and an eligible one as exactly one of
`root_attached`, `root_not_selected`, `root_no_partner`,
`root_collision_failed` and `root_over_budget`; the code `root` never occurs
with the option on. With the option on, a root column given as a NaN
separator is refused.

Draws come from `rng` in a fixed order: one Bernoulli per eligible tip in
(network, column) order, the processing permutation, then the bridge walks in
processing order. With `attach_roots` and at least one eligible root, a root
phase comes first: one Bernoulli per eligible root in (network, column)
order (`root_fraction`), a permutation of the selected roots, then the root
bridges in that order, each an obstacle for every later bridge; the tip
draws follow, and are made even when the budget was spent during the root
phase, in which case the remaining selected roots are `root_over_budget` and
the selected tips `over_budget`. With no eligible root nothing is drawn, so
the bridges, the generator state and the summary equal the option off (the
tips table then differs only where a root inside the box is ineligible,
reported as `cut_end` rather than `root`); with `root_fraction` 0 the root
Bernoulli draws are still made. A tip consumed by a root bridge counts in
`join_source_consumed` when its turn comes in the tip phase, as one
consumed by a tip bridge does. Identical inputs, parameters and
generator state give identical output. The result is also invariant under a
change of unit: scaling the nodes, `collision_margin`, `box`,
`boundary_margin` and `tol` by λ (and `max_bridge_volume` by λ³) gives the
same topology and outcomes and bridges equal to λ times the original
(`root_radius` is in root diameters and is not scaled). To keep a library's
clearance when joining library networks rescaled by λ, pass λ times the
library margin, and work in a frame where one library unit maps to at least
one caller unit, or scale `tol` by λ as well.

With `max_bridge_volume` set (π r² × arc per bridge, caller units cubed), no
new bridge starts once the sum would exceed it; the remaining selected tips
are reported as `over_budget` (and the remaining selected roots as
`root_over_budget`: root and tip bridges share the budget). Each bridge
reports its `volume`, and the first k bridges of any result are themselves a
valid result (each was checked against the inputs and the bridges before it
only), so a caller may truncate after measuring what it rendered: a run with
`max_bridge_volume` equal to the volume of the first k bridges reproduces
exactly those k.

The result holds `bridges` (per bridge: `tip` and `partner` as (network,
column), `source_kind` (`tip` or `root`: a root bridge's `tip` field holds
the root's (network, column), so readers of the record work unchanged),
`partner_kind`, `geometry` as a (4, m) float64 array in the caller's frame
whose first and last columns are bitwise copies of the joined columns,
`chord`, `arc`, `diameter`, `volume` and `redraws`), `bridge_nodes` (the
bridges concatenated with NaN separators), `tips` (a structured array with
one row per degree-one vertex of the input: `network`, `column` and an
`outcome` code into `join.TIP_OUTCOMES`, exactly one of `root`, `cut_end`,
`stub`, `not_selected`, `bridged_source`, `bridged_partner`, `no_partner`,
`collision_failed`, `over_budget`, `root_attached`, `root_not_selected`,
`root_no_partner`, `root_collision_failed`, `root_over_budget`; degree-zero
columns are counted in `events["join_isolated"]` and are not tips), `events`
and `summary`. The counters `join_root_eligible`, `join_root_selected`,
`join_root_attached`, `join_root_not_selected`, `join_root_no_partner`,
`join_root_collision_failed` and `join_root_over_budget` are appended to
`join.EVENT_KEYS` after those of 3.3 (as the root codes are to
`TIP_OUTCOMES`), so existing codes and keys keep their values; the last five
are read from the final table like the other outcome counters, so the
outcome counters still partition `join_tips`. Root bridges also count in
`join_redraws`, `join_box_redraws`, `join_behind_skipped`,
`join_partner_tips` or `join_partner_interior` and `join_components_joined`,
never in `join_eligible`, `join_selected`, `join_bridges` or
`join_source_consumed`, so `len(bridges) == join_bridges +
join_root_attached`. The summary holds, before and after, the components,
the length, `free_tips` (tips that are neither roots, under any root code,
nor cut ends), `blunt_roots` (the eligible roots, less those attached),
`free_ends` (both together) and each of the three per unit length, plus the
bridge volume; with the option off `blunt_roots` is the same before and
after. With `merged=True` it also holds `merged`: `nodes` with the bridges
appended after NaN separators, `edges`, `node_kind`, a `network` id per
column (-2 on bridges) and `tree`.

To render placed geometry, map it to voxels yourself and call the rasteriser
directly; `process_network` re-centres the network and must not be used here.
With voxel i centred at i, the field of view of a volume of n voxels is
`[-0.5, n - 0.5]` per axis, so a box `(lo, hi)` renders into `(hi - lo) / v`
voxels when `lo` maps to -0.5, which an origin half a voxel inside the box
does:

```python
from computeVoxel import rasterise_segments

v = 2.0                                                        # caller units per voxel
shape = tuple(int(round(s)) for s in (hi - lo) / v)
origin = lo + v / 2.0
result = join_networks(cropped, rng, fraction=0.9, collision_margin=margin, box=(lo, hi),
                       boundary_margin=2.0, merged=True)
xyz = result["merged"]["nodes"]                               # inputs and bridges, NaN separated
volume = rasterise_segments((xyz[:3] - origin[:, None]) / v, xyz[3] / (2.0 * v), shape, connect=True)
```

`result["bridge_nodes"]` alone renders the bridges on their own, for
measuring what they add.

Measured cost (one process, this machine): the test forest of 253 networks of
4-generation trees (186 thousand points) goes from 310 components to 31 and
from 0.0141 to 0.0002 free in-box tips per unit length with 721 bridges. The
two benchmark forests of grown trees placed at random rotations and offsets
with a point-clearance rejection and cropped with `crop_network`
(`VSYSTEM_SLOW_TESTS=1 python -m unittest tests.test_join.BenchmarkTests`):

| forest | networks | points | eligible tips | selected | bridges | joining | per selected tip | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| typical | 317 | 256 208 | 3 778 | 3 365 | 1 908 | 8.5 s | 2.5 ms | 160 MB |
| large | 650 | 2 117 568 | 17 118 | 15 408 | 7 293 | 143 s | 9.3 ms | 968 MB |

The large forest is denser (6-generation trees), so a bridge needs 8.5
redraws on average against 3.2 in the typical one, which is where the extra
time per tip goes; the fixed setup (graph, per-column arrays, indexes) is
about 1.3 s per 0.35 million points. The cost per tip splits roughly into the
collision checks (half, through one grid per octave of vessel radius and a
growing grid of the earlier bridges), the bridge walk (a quarter) and the
partner search with its vectorised filtering.

Root attachment, measured on the typical forest grown with root diameters
spread log-uniformly over [2, 8] at d_min = 1 (330 networks, 235 thousand
points, `VSYSTEM_SLOW_TESTS=1 python -m unittest tests.test_join_roots`),
on a machine where the option off takes 15.5 s: the option on takes 26.2 s;
of 266 eligible roots 220 attach, 30 find no partner (nothing as thick
within reach and the cone) and 16 fail every redraw; the root bridges have
a median chord of 14.7 and a median arc of 16.8 root diameters at the
default `root_radius` (4.1 and 4.3 at `root_radius` 5, where 9 roots
attach), all to interior points; a root bridge holds 1 773 units cubed on
average against 106 for a tip bridge, since it is as thick as the root and
long; the free ends per unit length go from 0.0289 to 0.0067 with the option
off and to 0.0053 with it on, the blunt roots per unit length from 0.0021
to 0.0002. The attachment rate is 74 to 93 % in every root-diameter bin.

---

## Network libraries: `vsystem-library`

A library is a fixed, reproducible set of networks grown once and sampled from
afterwards, rendered at whatever scale a use needs. V-System is scale free:
growth at (λ d0, λ d_min, λ margin, same seed) gives λ times the nodes up to
float rounding, so a library grown in **relative units**, with the smallest
drawn vessel diameter d_min = 1 unit, can be rescaled to any resolution. Its
networks are described by the root ratio R = d0 / d_min.

```bash
vsystem-library --out lib --count 2000 --seed 1 --workers 8
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--out DIR` | required | output directory; a run into a directory holding a manifest resumes it |
| `--count N` | required | number of networks |
| `--families` | `tree mesh tumour` | families to grow, in id order (`aligned` and `aligned_tight` when listed); only presets that exist are accepted, so a family without one and unknown names are refused before anything is grown |
| `--family-shares` | `1 1 1` | relative share of each family, by largest remainder with ties to the family listed first (2000 at `1 1 1` gives 667 667 666) |
| `--ratio-range R_LO R_HI` | `2.52 25` | range of R, log-uniform and stratified; 2^(4/3) gives at least about four generations |
| `--seed S` | required | library seed |
| `--workers W` | `1` | worker processes |
| `--collision-margin` | `1.0` | clearance between vessel surfaces, units of d_min |
| `--mesh-box-c` | `15` | a mesh grows in a cube of side this many times its root diameter, which keeps its two trees within reach of each other |
| `--box-c FAMILY C` | `mesh 15`, `aligned 15`, `aligned_tight 20` | the cube side of a box family (mesh, aligned, aligned_tight) in root diameters (repeatable); refused for a family that grows free, and with `--mesh-box-c` for mesh; an aligned member whose root offsets leave a small cube is recorded as a failure |
| `--iteration-cap` | `64` | generations allowed; d_min stops growth first |
| `--avoid-collisions` / `--no-avoid-collisions` | on | collision avoidance for the tree family; the mesh and tumour presets avoid collisions already |

Ids run 0 .. N-1 in contiguous family blocks in `--families` order and the
archives are named `net_{id:05d}_{family}.npz`. Network `id` grows from the
seed `int(np.random.SeedSequence([S, id]).generate_state(1)[0])`. For family
f at index j in `--families` with n_f networks, `default_rng([S, 4, j])`
draws `u = rng.random(n_f)` and `perm = rng.permutation(n_f)`, and the k-th
network of f gets `R = R_LO (R_HI / R_LO) ** ((perm[k] + u[k]) / n_f)`: one
network per bin of equal width in log R, in a random order. The bins depend
on n_f, so a library of a different size or share is a new library, not an
extension of an old one. The law (`log-uniform`), range, bin and u are
recorded per network.

Each network is grown exactly as `vsystem` would grow it from

```
--family f --d0 R 0 --d0-min R --d-min 1.0 --iterations 64 64 --collision-margin 1.0
--volume 3 3 3 --fit voxel_size --voxel-size (c R / 3 for a box family, c from library.BOX_C, 1.0 otherwise) [--avoid-collisions]
```

with the global generators seeded from the network's seed, through
`main.build_parser(f)`, `sample_parameters` and `grow_network`, so a library
network's `nodes` and `program` equal what `vsystem` writes for the same
arguments and seed on the same machine (the tests check this). The tumour
preset's root calibre is replaced by R; its aneurysm and stenosis
probabilities reach the properties through the parser's defaults as on the
command line. Tree and tumour growth never read the volume; a mesh and an
aligned network grow in a cube of side 3 × voxel size = c R, with c from
`library.BOX_C` (15 for mesh and aligned, 20 for aligned_tight, whose roots
are 40 d_min apart; `--mesh-box-c` for mesh, `--box-c FAMILY C` for any box
family, not both for mesh). A library in another unit is the same
library rescaled: growth at (2R, d_min 2, margin 2, box 2 × 15 R) equals
twice the library network to float rounding. The default library holds
`tree`, `mesh` and `tumour` (`library.DEFAULT_FAMILIES`); `aligned` is grown
when listed, its box constant then recorded under `growth.box_c` in the
manifest, and every archive's metadata carries the network's `frame` and its
`d_min`. The index holds the 3.4 columns, then `frame_kind`,
`capillary_order` and `larger_order` (S about the frame axis), `capillary_polar_order`,
`capillary_length_share`, `capillary_volume_share`, `capillary_segment_median`
and `transverse_spacing_median` (in d_min), empty where a network has no frame
axis or no polar sense.

Every archive is written atomically (`save_network` to a temporary file,
then renamed) with metadata recording `"units": "d_min"`, the family, R, bin,
u and seed, every keyword passed to `grow_network`, the counters, the root
column of each tree, the smallest per-point diameter (stenoses and spline
blending dip to about 0.5–0.67 d_min) and the sha256 of `nodes` and
`program`. `index.json` and `index.csv` hold, per network and in units of
d_min, id, family, R, seed, file, generations reached, points, polylines,
tips, junctions, cycles, components, total length, length- and
volume-weighted diameter percentiles (P50, P90, P99), the smallest and
largest diameter, the minimum clearance with the count of pairs below the
margin, the bridges, the counters, and in separate columns the time and the
peak resident set of the growth. `manifest.json` has two parts: `content`
(every parameter, the ratio law, the family counts, a code hash over
`library.py` and `main`'s import closure, per-network nodes and program
hashes, and failures with their reasons), whose canonical JSON (`sort_keys`)
is hashed into `content_sha256`, and `run` (host, Python and numpy versions,
per-network time and peak memory, dates). Archive bytes are not compared:
zip entries carry timestamps and BLAS rounding differs between machines.

A run into a directory that holds a manifest resumes it: a different
parameter set or a different code hash is refused, a different Python or
numpy version is recorded with a warning, every archive that loads and
matches its plan (seed, family, ratio, arguments and recorded hash) is kept,
an unreadable one is grown again, an archive of another network under a
planned name is refused, and failures are listed in the manifest and
reported, never dropped. Networks grow in fresh processes
(`multiprocessing` with a spawned worker per network and the BLAS thread
counts set to one), so the peak memory recorded per network is that
network's own; the grammar's global generators and `libGenerator`'s module
globals rule out threads.

`library.library_weights(index, exponent=3.0, mask=None)` gives the weight to
draw each network with so that the drawn root ratios follow a power law,
p_target(R) ∝ R^(-exponent) per unit log R, instead of the library's law:
w_i ∝ p_target(R_i) / p_library(R_i), with p_library read from the recorded
law, constant per unit log R for `log-uniform`, so w_i ∝ R_i^(-exponent). The
weights are normalised to sum one over `mask` and are zero elsewhere. Writing
x = ln R, uniform on [a, b] = [ln R_LO, ln R_HI], the effective sample size
(Σw)² / Σw² of N draws is N (E[e^(-3x)])² / E[e^(-6x)] =
N (e^(-3a) - e^(-3b))² / (2 (b - a) (e^(-6a) - e^(-6b))) ≈ 0.29 N at the
defaults, so a library of 2000 networks is worth about 580 equally weighted
draws from the R^(-3) law.

Measured cost (one process per network, this machine, seed 777, growth and
the descriptors timed separately; peak RSS is the worker's resident set,
which includes numpy's own 70 MB):

| family | R | generations | points | tips | growth | descriptors | archive | peak RSS | clearance violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tree | 2.52 | 7 | 1 364 | 8 | 0.0 s | 0.2 s | 0.04 MB | 75 MB | 0 |
| tree (no avoidance) | 2.52 | 7 | 1 449 | 8 | 0.0 s | 0.2 s | 0.04 MB | 73 MB | 1 |
| tumour | 2.52 | 7 | 431 | 3 | 0.1 s | 0.2 s | 0.02 MB | 70 MB | 0 |
| mesh | 2.52 | 8 | 2 166 | 9 | 1.3 s | 0.2 s | 0.06 MB | 77 MB | 0 |
| tree | 4 | 12 | 5 678 | 36 | 0.2 s | 0.4 s | 0.17 MB | 95 MB | 0 |
| tree (no avoidance) | 4 | 12 | 6 420 | 37 | 0.1 s | 0.3 s | 0.20 MB | 92 MB | 11 |
| tumour | 4 | 12 | 1 078 | 6 | 0.7 s | 0.2 s | 0.04 MB | 72 MB | 0 |
| mesh | 4 | 12 | 3 511 | 19 | 2.7 s | 0.3 s | 0.10 MB | 82 MB | 0 |
| tree | 6.3 | 17 | 24 280 | 131 | 0.7 s | 0.9 s | 0.74 MB | 127 MB | 0 |
| tree (no avoidance) | 6.3 | 17 | 26 302 | 137 | 0.2 s | 0.6 s | 0.79 MB | 123 MB | 13 |
| tumour | 6.3 | 17 | 1 950 | 6 | 1.2 s | 0.2 s | 0.11 MB | 75 MB | 0 |
| mesh | 6.3 | 17 | 17 272 | 72 | 14.0 s | 0.6 s | 0.48 MB | 104 MB | 0 |
| tree | 10 | 20 | 91 527 | 519 | 3.1 s | 2.9 s | 2.76 MB | 165 MB | 0 |
| tree (no avoidance) | 10 | 20 | 104 846 | 557 | 0.8 s | 2.0 s | 3.13 MB | 153 MB | 1 417 |
| tumour | 10 | 20 | 8 530 | 30 | 7.7 s | 0.4 s | 0.47 MB | 101 MB | 0 |
| mesh | 10 | 20 | 47 565 | 179 | 31.2 s | 1.4 s | 1.39 MB | 124 MB | 0 |
| tree | 16 | 23 | 383 934 | 2 116 | 22.5 s | 12.2 s | 11.50 MB | 314 MB | 0 |
| tree (no avoidance) | 16 | 23 | 432 769 | 2 257 | 3.6 s | 8.2 s | 12.85 MB | 320 MB | 20 337 |
| tumour | 16 | 23 | 83 294 | 264 | 75.4 s | 2.6 s | 3.12 MB | 285 MB | 0 |
| mesh | 16 | 24 | 72 169 | 283 | 67.3 s | 2.0 s | 2.65 MB | 159 MB | 0 |
| tree | 25 | 28 | 547 328 | 3 110 | 62.6 s | 18.2 s | 18.40 MB | 442 MB | 0 |
| tree (no avoidance) | 25 | 28 | 1 651 750 | 8 715 | 15.5 s | 36.8 s | 48.42 MB | 1 039 MB | 186 777 |
| tumour | 25 | 28 | 30 133 | 114 | 43.1 s | 1.1 s | 4.75 MB | 280 MB | 0 |
| mesh | 25 | 28 | 391 572 | 1 331 | 444.9 s | 11.3 s | 12.85 MB | 449 MB | 0 |

Points grow as about 105 R³ for a plain tree (1.65 million at R = 25); with
avoidance a tree keeps a third of that at R = 25, since branches that would
cross are terminated, and the tumour preset, which always avoids collisions
and bends at low persistence, stays an order of magnitude smaller again (and
varies a lot between seeds: one seed gives 83 thousand points at R = 16 and
30 thousand at R = 25). A mesh costs the most: its two trees are grown with
the walk and avoidance and then anastomosed, 7.4 minutes at R = 25 for
392 thousand points, within 0.45 GB. The cost of `--avoid-collisions` for
trees is the collision index: the growth takes 4 to 6 times longer from R =
10 up (3.1 s against 0.8 s at R = 10, 62.6 s against 15.5 s at R = 25), and
it buys a network with no pair of vessels inside each other, where a plain
tree has 1 417 overlapping point pairs at R = 10 and 186 777 at R = 25
(margin 1). The tumour and mesh networks show no clearance violation at any
ratio measured, and neither does the avoided tree.

Projection for N = 2000 at the default shares and range, taking the
log-uniform mean of growth plus descriptors interpolated in (log R, log t)
between the measured points: about 15 s per tree, 56 s per mesh and 19 s per
tumour on this machine, 16.7 CPU hours in all, about 2.1 h of wall time at 8
workers, 5.0 GB of archives, and at most 8 × 0.45 GB = 3.6 GB of memory with
every worker at the largest peak measured.

---

## Grammar

The alphabet is the dissertation's. `f(l, d)` moves by `l` along the direction
vector and records diameter `d`; `+(θ)` and `-(θ)` rotate the direction about the
perpendicular vector; `/(β)` and `*(β)` rotate the perpendicular about the
direction; `[` and `]` push and pop the whole state; `{` and `}` delimit a stem
that is interpolated as one smooth curve. A `[` inside a stem pins the stem at
the branch point — the part before it and the part after it are interpolated as
separate curves that meet exactly there — so a branch may leave a stem midway
without disconnecting the centreline. An operand left out takes its default:
the segment length for the current diameter, the Zamir angle θ1, or the roll angle.

| Rule | Production | Source |
| --- | --- | --- |
| `F(n, d0)` | `{S(d0)} [+(th1) /(roll) F(n-1, d1)] [-(th2) /(roll) F(n-1, d2)]` | §4.3.1 tree grammar |
| `S(d0)` | `D +(a) D -(a) D -(a) D +(a) D` or its mirror, each with probability ½ | §4.3.1 |
| `D(d0)` | `f(co/5, d0)`; with probability `aneurysm_prob` or `stenosis_prob`, `f(co/25, d0) f(3co/25, d0·factor) f(co/25, d0)` | §3.3.1 and grammars (i), (j) |
| `A(n, d0)` | `{S(d0)} [+(th1) A(n-1, d1)] [-(th2) A(n-1, d2)]` | example (b) |
| `B(n, d0)` | `C C C /(90) A(n-1, d0)` | example (b) |
| `R(n, d0)` | `f(co/3) C C C [B(n-1, d1)] f(co/2, d2) B(n-1, d2)` | example (b) |
| `I(n, d0)` | `f(co/3, d0) +(a) [R(n-1, d0)]` | example (b) |

`th1`, `th2`, `d1`, `d2` and `co` come from `libGenerator.calBifurcation`:
`d1` is drawn from a Gaussian about the symmetric optimum `d0 / 2^(1/k)`,
`d2` follows from Murray's law `d0^k = d1^k + d2^k`, the angles from Zamir's
rule in the asymmetry ratio `d2 / d1`, and `co = epsilon · d0` scaled by a
uniform factor in `[1 - randmarg, 1 + randmarg]`.

Departures from the source, all deliberate: stems are always drawn, so an
iteration count is a count of drawn generations; the length margin is relative
rather than absolute, so lengths stay positive at every depth; and anomalies are
drawn per sub-segment with configurable probabilities rather than written into a
grammar by hand.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite covers the turtle semantics, grammar balance, the bifurcation law,
the capsule volume at three orientations, angle preservation under the
isotropic fit, the `d_min` stopping criterion, seed determinism and the
command-line entry point. It also pins the centreline contract: a saved
centreline re-renders the volume it was generated with, calibre in voxels
scales as `1 / voxel_size`, and a sidecar plus its `.npz` reproduce the written
TIFF exactly. Connectivity is covered too: an unclipped network renders as one
component under both 6- and 26-connectivity, the chain drawn for a segment is
face-connected at every orientation and never strays more than `sqrt(3)/2`
from it, bare capsules do break, and connecting changes nothing once a vessel
fills a voxel. `tests/test_geometry.py` covers the geometry stages: the default
command line draws bit for bit what the previous generator (kept under
`tests/fixtures/reference_code`) draws in the same environment, centreline and
TIFF alike, and matches the stored reference centrelines to rounding (the
B-spline's matrix products go through BLAS kernels whose last bits differ
between machines), every new option is reproducible from its seed, `edges` round-trip and the
degree-derived tip count matches the grammar's, avoided networks have no pair
of branches closer than the margin while plain trees do, anastomosed networks
have cycles and a lower tip density than the same seed without, arc/chord
rises as persistence falls, a cycle-containing archive renders through
`process_network(..., connect=True)`, and `computeVoxel` imports under Python
3.9 when an interpreter is available. `tests/test_pinned_geometry.py` pins the
mesh and tumour geometry: the 3.2 modules (`tests/fixtures/reference_code_3_2`)
and the current ones draw the same `nodes`, `tree` and bridges for two seeds
per family in the same environment. `tests/test_join.py` checks that two
facing networks join into one component with no intra-network bridge, that
bridges keep the margin from every vessel and from each other, that cut ends
are never used and no bridge leaves the box, that the tip outcomes partition
the degree-one vertices and agree with the counters, determinism, invariance
under a change of unit, more than 127 networks, a mesh's two roots, a forest
losing components and free tips, that any prefix of the bridges is a valid
result, and, with `VSYSTEM_SLOW_TESTS=1`, the two performance targets.
`tests/test_pinned_join.py` pins the joining of 3.3: the 3.3 `join.py`
(`tests/fixtures/reference_code_3_3`) and the current one draw the same
bridges, tips table, counters, summary and generator state for two small
forests in the same environment, and with root attachment off nothing added
since is live. `tests/test_join_roots.py` checks root attachment: a root near
a thicker vessel attaches as a side branch at the root's own diameter and
makes a junction, a root never attaches to a thinner vessel, root bridges
depart upstream, the cone is taken about the upstream direction and the
search radius limits it, interior points rank before tips and an attached
tip is consumed, a root in a junction zone or without a tangent still
attaches, cut-end roots never attach, root partners lie on other networks
whatever the policy, the outcomes partition the degree-one vertices (a mesh
pair's four roots included) and agree with the counters, determinism and
invariance under a change of unit with root bridges present, the shared
budget with roots first, the prefix property, that in-box roots that are
ineligible change only their own rows and that the option draws nothing
without an eligible root, and, with `VSYSTEM_SLOW_TESTS=1`, the typical
forest with root attachment off and on.
`tests/test_library.py` checks the library's determinism, that the presets
take effect, that no drawn diameter falls below d_min but a stenosis middle,
the scaling with the unit, equality with the command line's output,
independence from the volume, the stratification, resume with the refusal of
a different library or code, the stability of the manifest hash, the weights
against a power law, the refusal of a family without a preset and of unknown
families, a tiny end-to-end run, and from 3.5 the aligned family's box, frame
and index columns, the box constants and their refusals, and a tiny aligned
library described and resumed with every new column.
`tests/test_pinned_release_3_4.py` pins what 3.4 drew, wrote, measured and
joined: the 3.4 modules (`tests/fixtures/reference_code_3_4`) and the
current ones run one driver side by side in the same environment over
growth, library members and plans, description, the command line, cropping
and joining, and a three-network library, every record restricted to the keys
3.4 wrote, so that a later version may append keys but must keep what 3.4
wrote, with every appended counter at zero and the frame of kind `none` while
the new options are off; it takes about 30 s on a laptop (Apple M4) and
95 s on an Intel Xeon Gold 5220 workstation. `tests/test_guidance.py`
checks the guidance rules and their refusals, the turn and the bank of one
step, the draw count, that the walk without guidance is what it was, the
stationary laws on a long stem, the counters, the choice of rule, the polar
senses, the free extent, root offsets, scale invariance, the frame record in
the return, sidecar and archive, and the aligned family end to end, and, with
`VSYSTEM_SLOW_TESTS=1`, the capillary order, polar order and feeder planarity
of `aligned` against the same seeds unguided, the bank's gain in planarity
and the onset's effect on collision redraws over ten seeds, and the fall of
the order with the guidance length over four seeds. `tests/test_frames.py`
checks the rotations, `align` and the transforms, and that a random transform
leaves the order about the carried axis unchanged. `tests/test_describe.py`
checks the frame descriptors on exact constructions (S of 1 and −0.5, the
Watson and Fisher-axial inversions on 10⁴ samples, exact calibre shares and
transverse spacing of a square array, each d_ref source) and their unit
invariance. `tests/test_topology_benchmark.py`, with `VSYSTEM_SLOW_TESTS=1`,
grows every family at three root ratios and the aligned preset over its
guidance length and onset, prints the Phase 1 descriptors and the cost, and
projects a 2000-network library.

---

## Applications

- Generating realistic 3D vascular structures for simulation.
- Training datasets for segmentation algorithms (e.g. VAN-GAN).
- Creating synthetic benchmarks for vascular imaging pipelines.
- Testing robustness of deep learning models under anatomical variability.

---

## Citation

Please cite the following if you use V-System in your research:

**Version 1.0**
> [Quantification of vascular networks in photoacoustic mesoscopy](https://www.sciencedirect.com/science/article/pii/S221359792200026X)
> Emma L. Brown, Thierry L. Lefebvre, Paul W. Sweeney et al.

**Version 2.0**
> [Unsupervised Segmentation of 3D Microvascular Photoacoustic Images Using Deep Generative Learning](https://doi.org/10.1002/advs.202402195)
> Paul W. Sweeney et al., *Advanced Science*, 2024

Version 3.0 changes the interpreter, the grammars and the voxeliser as described
above; volumes generated with it are not identical to those of earlier versions.
Version 3.1 adds the centreline archive, the declared unit convention and the
`d_min` stopping criterion. It also renders sub-voxel vessels as face-connected
one-voxel paths rather than dotted lines, so its volumes contain vessels that
3.0 dropped and hold together under a six-connected label; `--no-connect`
reproduces the 3.0 rasterisation. Version 3.2 adds the graph to the archive,
the walk, collision avoidance, anastomosis, the family presets and
`describe.py`; the default command line still writes the 3.1 geometry.
Version 3.3 adds `join.py`, which joins separately grown networks inside a
field of view, and `vsystem-library`, which grows reproducible network
libraries in relative units; the default command line and the family presets
draw what 3.2 drew. Version 3.4 adds root attachment to `join_networks`, an
option off by default; with it off the joining draws what 3.3 drew.

---

## Acknowledgements

This project builds upon foundational work by Miguel A. Galarreta-Valverde:
*Geração de redes vasculares sintéticas tridimensionais utilizando sistemas de
Lindenmayer estocásticos e parametrizados*, MSc dissertation, University of São
Paulo, 2012 ([DOI 10.11606/D.45.2012.tde-30112012-172822](https://doi.org/10.11606/D.45.2012.tde-30112012-172822)),
and Galarreta-Valverde, Macedo, Mekkaoui and Jackowski, *Three-dimensional synthetic
blood vessel generation using stochastic L-systems*, Proc. SPIE 8669 (2013).

---

## Contact

For bugs, ideas, or contributions, open an issue on the
[GitHub repository](https://github.com/psweens/V-System).
