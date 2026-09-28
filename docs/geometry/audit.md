# Step 0: audit of the generator before the geometry upgrades

Audit of the V-System checkout at commit 30c1a2b (`master`), against the
assumptions behind the geometry upgrades. Every claim below was checked by
reading the code and, where numerical, by running it, and each was then
re-derived independently by a second reader who tried to refute it; the
numbers come from the default parameters unless stated.

## 1. `process_network(..., connect=)` is a rendering flag; nothing joins tips

`computeVoxel.process_network(data, tVol, fit="isotropic", voxel_size=None,
margin=1.0, clip_axes=(), connect=True)` is two calls: `fit_to_volume` maps
grammar units to voxels, `rasterise_segments` draws each pair of consecutive
non-NaN columns as a tapered capsule and, when `connect` is true, also as the
face-connected chain of voxels the segment passes through (Amanatides–Woo
traversal, `rasterise_line`). `connect` therefore fixes the *dotted-line*
problem of sub-voxel vessels; it never looks beyond one polyline and never
creates geometry. Grepping the repository for tip joining, merging, loops or
anastomosis finds nothing: the only place polylines are related to one another
is `check_connectivity.report_centreline`, which *reads* shared coordinates to
count components and writes nothing. vox2vess's reading of `connect` is right.

## 2. How a stem's path is produced

- `vSystem.F(n, d0)` emits `{S(d0)}` then `[+(θ1)/(roll)F(n−1,d1)][−(θ2)/(roll)F(n−1,d2)]`.
  `S` is `D +(a) D −(a) D −(a) D +(a) D` or its mirror, chosen by
  `random.random() < 0.5`, with `a = stem_angle` (25°). A stem is therefore
  **five straight sub-segments with four alternating turns** (net turn zero).
- Each `D` is `f(getLength(d0)/5, d0)`; `getLength` draws
  `d0 · ε · U(1 − randmarg, 1 + randmarg)` **independently for every
  sub-segment**, so the five pieces of one stem have different lengths and a
  stem's total length is the sum of five such draws, not one draw divided by
  five. `ε` and `randmarg` are drawn once per network in
  `main.sample_parameters` (uniform in `--epsilon 4 10` and `--randmarg 0.1
  0.3`) and then fixed for the whole tree.
- With probability `aneurysm_prob` or `stenosis_prob` (0.02 each) a `D` is
  three moves, `f(l/5, d) f(3l/5, d·1.5 or 0.5) f(l/5, d)`, so at the default
  rates a third of all stems are 7 or 9 moves rather than 5 and carry up to
  three diameters (measured on a 6-generation tree: 42 stems of 5 moves, 18
  of 7, 3 of 9).
- Only the second and fourth sub-segments deviate from the entry direction
  (headings h, R(+25°)h, h, R(−25°)h, h), so a stem exits parallel to its
  entry but laterally offset by (L2 − L4) sin 25°, the two lengths being
  independent draws; the stem is planar and its net rotation is exactly zero.
- The interpreter (`analyseGrammar.branching_turtle_to_coords`) yields one row
  per move, `(x, y, z, diameter, segment)`, plus the start point at `{`; it
  consumes **no random numbers** when every operand is written, which `F`
  always does (measured: zero draws during interpretation and interpolation).
- `utils.interpolate_segments` replaces each run of rows sharing a segment
  index by `utils.bspline(control, subdivisions=3)`: a **clamped uniform cubic
  B-spline**, which is approximating, not interpolating. It passes exactly
  through the first and last control points (they are repeated) and *not*
  through the interior ones, and samples 2^3 = 8 points per span with
  `len(control) + 1` spans, so a plain stem of 6 control points becomes 57
  samples. The last sample is a bitwise copy of the last control point; the
  first sample equals the first control point only in about a quarter of
  cases and otherwise differs by up to ~6·10⁻¹⁴ grammar units (rounding in the
  basis weights). The spline's arc length is shorter than the control
  polygon's, so the grammar's "segment length" is the zig-zag's length, not
  the drawn curve's; the walk preserves the former.
- Diameter is the fourth coordinate of the spline, so it is **constant along a
  plain stem and smoothly blended across an anomaly**. Two exceptions: the
  first control point of a daughter stem is the `{` row, which still carries
  the *parent's* diameter, so every daughter tapers from d_parent to d_child
  over its first span; and a stenosis or aneurysm is smeared by the spline
  rather than stepped.
- Bifurcation: daughters start exactly at the stem's end (the `]` restores
  the pushed position bitwise) and turn by the Zamir angles about the
  perpendicular, which has been rolled by 70° after each daughter turn.

## 3. Junction adjacency in the archive

The archive has no edge list. `nodes` is `(4, N)` float64 with all-NaN
columns separating polylines. A polyline is a chain from a branch point along
successive *first* daughters until a `]`; the `]` yields a NaN row and then
the restored position, which begins the next polyline. The rule that joins
polylines is **bitwise equality of coordinates**: a daughter polyline's first
column is the restored `pos` from the stack, a copy of the parent stem's last
control point, which is also the parent spline's last sample. Measured on a
6-generation tree: every one of the 126 polylines after the first starts on a
column that appears earlier, bitwise. `check_connectivity.report_centreline`
relies on exactly this (`nodes[:3, j].tobytes()` as the key). Within a
polyline there are zero- and near-zero-length steps (the `{` row duplicates
the preceding row; the spline's first sample is within 10⁻¹³ of it): 27 exact
and 36 below 10⁻⁹ µm among 3975 steps of that tree. A restored position is a
polyline of its own whenever no move follows it before the next `]`: after
the `]` of every second daughter (which the grammar always writes last inside
its bracket) and after both `]` of a leaf stem, so 3·2^(n−1) − 1 of the
2^(n+1) − 1 polylines of an n-generation tree are single points, every one a
bitwise copy of a stem end or a junction, drawing no capsule of their own.

A graph can therefore be rebuilt with: vertices = groups of columns coincident
within a tolerance (10⁻⁶ µm is far below any geometry and far above the
rounding), edges = consecutive columns of a polyline mapped to their groups
with zero-length edges dropped. `graph.py` does this; the tolerance is the
one thing the exact-bytes rule of `check_connectivity` lacked.

## 4. Volume bounds

Bounds are enforced in one place: the `f` handler of the interpreter, when
`bounds` is given. The test is on the move's **end point** only (a move that
crosses a corner of the box and comes back is not caught). A move that would
leave the box is skipped and the branch is terminated: everything up to the
matching `]` is ignored, so the stem is drawn up to its last in-box control
point (a partial stem is left behind) and the whole subtree is dropped. Under
`--grow-in-volume`, `main.generate_network` derives the box as `tVol ×
voxel_size` for the `voxel_size` fit and otherwise, from a first bound-free
interpretation of the same grammar string (which consumes no randomness), as
the largest box of the volume's proportions the free tree overflows; growth
starts at the middle of the box's y = 0 face heading +y. `--clip-axes` is a
**rasterisation-time** setting: the isotropic fit ignores the clipped axes
when choosing the scale and `rasterise_capsule` / `rasterise_line` drop voxels
outside the volume. None of this was counted or reported before; the sidecar
now carries `events.bound_terminations`.

## 5. Random-number usage and the order of draws

Two generators, both global: Python's `random` and numpy's legacy
`np.random`. `main.main` seeds both per network (`random.seed(seed)`,
`np.random.seed(seed % 2**32)`), and `check_connectivity.generate_and_check`
repeats that so its networks are the CLI's. Order on the default path for one
network:

1. `sample_parameters`: `random.uniform` (ε), `random.uniform` (randmarg),
   `np.random.normal` (d0, repeated while d0 < `--d0-min`), `random.randint`
   (generations).
2. `F(n, d0)` recursively, depth first, first daughter before second. Per stem:
   `calBifurcation` → `np.random.normal` (d1) then `getLength` →
   `np.random.uniform` (the `co` value, which `F` never uses but which is
   drawn all the same); `S` → `random.random` (mirror coin); then for each of
   the five `D`: `np.random.uniform` (length) and `random.random` (anomaly).
   13 draws per stem, 39 for `F(2, ·)`, measured.
3. Interpretation, interpolation, fitting and rasterisation: no draws.

A new feature keeps the default path byte-identical if it (a) makes no draw
from either global generator on the default path, (b) changes no operand `F`
writes, (c) changes neither the rows the default interpreter yields nor
`interpolate_segments`, and (d) adds only new archive arrays and new metadata
keys. Every stochastic addition draws from its own
`numpy.random.Generator(SeedSequence([seed, tag]))`, so it also leaves the
grammar's draws untouched when it *is* enabled: the same seed gives the same
tree under every combination of geometry options, which is what makes the
descriptor comparisons below paired.

## 6. Contracts a downstream reader depends on

`nodes` float64 `(4, N)`, x, y, z, diameter in grammar units (micrometres by
convention), NaN separators. `metadata` JSON with keys seed, iterations, d0,
d_min, properties, volume, axis_order, units, fit, clip_axes, connect,
grow_in_volume, voxel_size, subdivisions. `program` the grammar string. The
`.npz` is a zip whose entries carry timestamps, so files differ byte for byte
between runs while the arrays are exact. `computeVoxel.py` imports only numpy
and uses no syntax beyond Python 3.8; the same now holds for `graph.py`,
`spatial.py`, `describe.py` and the other new modules, and
`tests/test_geometry.py` compiles every module with the 3.9 grammar and runs
the import under a 3.9 interpreter when one is available.

## 7. Spatial index

`python spatial.py --benchmark 100000 1000000` (random points, inserts in
batches of 40 each preceded by a query of the batch, cell size 20 µm):

| points | grid hash | scipy cKDTree (rebuilt lazily) | pairs found |
| --- | --- | --- | --- |
| 10⁵ | 1.26 s | 1.79 s | 45 903 (both) |
| 10⁶ | 16.0 s | 60.5 s | 461 204 (both) |

The grid is the default (`--collision-index auto`); the k-d tree stays
available. The k-d tree's cost is its rebuilds, since the index grows by a few
dozen points between queries.

## 8. Baseline descriptors

`descriptors.md` in this directory (generated by `sweep.py`, means over seeds
1–5) holds the full table; the definitions are those of `describe.describe`.
The baselines:

| configuration | points | length (mm) | d P50 / P90 (µm) | arc/chord mean / P90 | curvature (1/µm) | tips | tips/mm | min clearance (µm) | pairs below 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default CLI (4–12 generations drawn) | 70 547 | 18.5 | 7.8 / 14.5 | 1.02 / 1.03 | 0.054 | 569 | 20.5 | −0.80 | 172 |
| default, 8 generations | 15 936 | 8.6 | 6.5 / 14.1 | 1.02 / 1.03 | 0.042 | 129 | 15.6 | 0.04 | 0 |
| LSM calibration (d0 25 ± 5, d_min 1), drawn depth | 58 129 | 20.9 | 9.5 / 17.3 | 1.02 / 1.03 | 0.041 | 441 | 14.9 | −0.18 | 82 |
| LSM calibration, 8 generations | 15 764 | 10.5 | 7.8 / 17.3 | 1.02 / 1.03 | 0.034 | 127 | 12.6 | 0.07 | 0 |

Three baseline facts the upgrades are measured against: the drawn stem has an
arc/chord ratio of 1.02 (P90 1.03) whatever the parameters, since it is the
spline of a fixed 25° zig-zag; tip density is 15–20 tips per mm of
centreline, every one a free end; and branches cross only in deep trees, at
a rate that the default depth range reaches (172 overlapping point pairs per
network on average, all from the seeds that drew ten or more generations).
The LSM calibration could not be checked against vox2vess's `REBUILD_LOG.md`,
which is not in this checkout; the values used are the ones quoted in the
brief.

## 9. Cost

Wall time of `python main.py --count 1` on one core, twelve generations
(262 660 centreline points, 512 × 512 × 140 volume), seed 5:

| configuration | seconds |
| --- | --- |
| default, code before the upgrades | 19.6 |
| default, this version (graph build and archive arrays added) | 21.2 |
| `--tortuosity walk --persistence 8` | 26.6 |
| `--avoid-collisions` (stems) | 30.6 |
| walk + avoidance | 53.7 |
| walk + avoidance + `--anastomose` | 83.3 |
| `--family tumour` | 74.1 |
| `--family mesh` (two trees) | 145.8 |

At five generations the default takes 1.5 s before and after. The walk is a
Python loop over steps (about 100 µs a step, twice that with the collision
query), which is where the opt-in cost goes; the rasterisation of the default
volume is most of the default time.

## 10. What contradicts the brief's assumptions

- **Stems are not simply "five straight sub-segments"** in the drawn geometry:
  the archive holds the approximating B-spline of that zig-zag, which is
  smooth but wavy. Its arc/chord ratio is fixed by the stem angle and
  independent of any parameter a user would tune; the measured value is in
  the table.
- **Sub-segment lengths are drawn independently** (five draws per stem), and
  `calBifurcation` draws a length it discards, so the draw count per stem is
  13, not 7.
- **Diameter is not strictly constant per segment**: daughters taper from the
  parent's diameter over their first span, and anomalies are smoothed.
- **Coincidence is bitwise for junctions but not within a polyline**: the
  spline's first sample usually misses its control point by ~10⁻¹⁴, so any
  graph builder needs a tolerance, and a curvature estimate must drop the
  near-zero steps or it will be dominated by them.
- **Bounds are tested at move end points only**, so a long move can leave and
  re-enter the box unnoticed; with the walk's short steps this gap closes.
- **Branches do pass through each other, but only in deep trees.** With the
  junction overlaps excused as `collisions.py` defines them, the descriptor
  finds no pair of vessels inside each other in plain trees of up to nine
  generations at ε = 7 (minimum surface clearance 0.01–0.1 µm, at junction
  boundaries). Crossings appear from ten generations, and more of them at
  the shorter segment lengths of the default `--epsilon 4 10` range: at ten
  generations and ε = 4, 47 and 1570 point pairs overlap in two seeds; at
  twelve generations and ε = 4, 2848 and 14 315. The default command line
  draws up to twelve generations, so the claim holds for the bank as
  generated, and the count in the table is the measure of it. The
  measurement also shows that a naive rule — excusing only pairs within one
  parent diameter of a junction — reports hundreds of false collisions at
  junctions whose parent stem carries an aneurysm; the rule that scales with
  the radii (`KIN_REACH`) removes them, which is why the generator and the
  descriptor share it.
