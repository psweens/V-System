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
   capsule, so calibre and bifurcation angles survive into the image, plus a
   connected digital line, so a vessel thinner than a voxel stays unbroken.

Steps 1–3 produce the **centreline**, which is saved alongside the volume. The
centreline is the source of truth for the geometry and the TIFF is one
rasterisation of it, so step 4 can be repeated at any resolution without
regenerating the network.

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
| `.npz` | the centreline: `nodes`, the (4, N) array of x, y, z and diameter in grammar units with NaN column separators; `program`, the grammar string; and `metadata`, the sidecar record |
| `.json` | the seed, the unit convention and every parameter used |

Network *i* of a run uses `seed + i`, and the same seed and parameters reproduce
the same volume. The centreline archive is a few hundred kilobytes against
hundreds of megabytes for a volume, and its arrays are exact; being a zip, its
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
| `--no-connect` | off | rasterise bare capsules, leaving sub-voxel vessels dotted |
| `--units` | `um` | the unit one grammar unit stands for, recorded in the sidecar |

`--d-min` and `--iterations` are both stopping criteria and whichever comes first
wins. `--d-min` is the one a modality states directly, as its smallest resolvable
calibre; diameters fall by `2^(-1/k)` a generation, so it is reached after about
`log(d0 / d_min) / log(2^(1/k))` generations — raise `--iterations` above that to
let `--d-min` decide. It bounds the diameter a branch is *drawn* at, not the
diameter after a local anomaly: a stenosis still narrows a drawn sub-segment by
`--stenosis-prob`'s factor of 0.5, so set `--stenosis-prob 0` for a hard floor.

### Connectivity

A capsule sets only the voxels whose centres it contains. A vessel thinner than
a voxel contains no centre along much of its length, so on its own it rasterises
as a dotted line and a connected tree falls apart into fragments — with the
default parameters, more than a third of the centreline is that thin and a
single tree renders as dozens of pieces. Every segment is therefore also drawn
as a 26-connected digital line, which renders a sub-voxel vessel one voxel wide
instead of dotted. It is not a dilation: every voxel the line sets lies within
`sqrt(3)/2` of the centreline, so for a radius of 0.866 voxels or more it is
already inside the capsule and calibre is untouched. `--no-connect` restores the
bare capsule rasterisation.

Two things still legitimately split a rendered network into pieces, and neither
is a defect:

- **Clipping.** A branch that leaves a slab and re-enters it is two vessels in
  the image, exactly as it would be in a real acquisition. `--clip-axes` and the
  volume shape control this; with `--clip-axes none` a fitted network renders as
  a single connected component.
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

Only the sampling changes between the two, so the vessel calibre distribution of
each result is that modality's — which is exactly what an unpaired
image-to-image model learns as its output prior. A 20 µm vessel is 20 voxels
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

## Grammar

The alphabet is the dissertation's. `f(l, d)` moves by `l` along the direction
vector and records diameter `d`; `+(θ)` and `-(θ)` rotate the direction about the
perpendicular vector; `/(β)` and `*(β)` rotate the perpendicular about the
direction; `[` and `]` push and pop the whole state; `{` and `}` delimit a stem
that is interpolated as one smooth curve. An operand left out takes its default:
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
26-connected component, bare capsules do not, and connecting changes nothing
once a vessel fills a voxel.

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
`d_min` stopping criterion. It also renders sub-voxel vessels as connected
one-voxel paths rather than dotted lines, so its volumes contain vessels that
3.0 dropped; `--no-connect` reproduces the 3.0 rasterisation.

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
