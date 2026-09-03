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
   capsule, so calibre and bifurcation angles survive into the image.

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

Each network is written as `Lnet_i<generations>_s<seed>.tiff`, a uint8 TIFF of
0 and 255 whose pages are z, rows y and columns x, next to a JSON sidecar with the
seed and every parameter used. Network *i* of a run uses `seed + i`, and the same
seed and parameters reproduce the same file.

Useful options (`python main.py --help` lists them all):

| Option | Default | Meaning |
| --- | --- | --- |
| `--volume NX NY NZ` | `512 512 140` | volume shape in voxels |
| `--iterations MIN MAX` | `4 12` | drawn generations, drawn uniformly |
| `--d0 MEAN STD` | `20 5` | root diameter, grammar units, truncated at `--d0-min` |
| `--epsilon MIN MAX` | `4 10` | length-to-diameter ratio of a segment |
| `--randmarg MIN MAX` | `0.1 0.3` | relative half-width of the segment-length distribution |
| `--sigma` | `5` | d_opt / sigma is the spread of the first daughter diameter |
| `--roll-angle` | `70` | roll of the bifurcation plane after each daughter turn, degrees |
| `--stem-angle` | `25` | turn between the five sub-segments of a stem, degrees |
| `--aneurysm-prob`, `--stenosis-prob` | `0.02` | per sub-segment probability of a local ×1.5 dilation or ×0.5 constriction |
| `--fit` | `isotropic` | `isotropic`, `voxel_size` (fixed `--voxel-size`) or `stretch` (legacy per-axis fill) |
| `--clip-axes` | `z` | axes left out of the isotropic fit and clipped, as an imaging slab would |

From Python:

```python
from main import generate_network

volume, program, nodes = generate_network(
    niter=8, d0=20.0, properties={"epsilon": 7.0, "randmarg": 0.2}, tVol=(512, 512, 140))
```

`volume` is a uint8 array of 0 and 1 indexed (x, y, z); `program` is the grammar
string; `nodes` is the (4, N) centreline of x, y, z and diameter with NaN columns
separating branches.

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
isotropic fit, seed determinism and the command-line entry point.

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
