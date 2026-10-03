# Changelog

## 3.3.0

Version 3.3 adds two opt-in tools built around forests of many independently
grown networks: `join.py` joins networks placed in one field of view with the
anastomosis rules, and `vsystem-library` grows a reproducible library of
networks in relative units to sample from and rescale. The default command
line is unchanged: for a given seed it writes the same `nodes`, `program` and
TIFF as 3.1 and 3.2, and the mesh and tumour families draw what 3.2 drew,
which the tests check by running the 3.2 modules, kept under
`tests/fixtures/reference_code_3_2`, alongside the current ones.

### Archive

- A library archive records, in its metadata, `"units": "d_min"`, the
  family, the root ratio with its law, bin and draw, the seed, every keyword
  passed to `grow_network`, the counters, the root column of each tree, the
  smallest per-point diameter and the sha256 of its `nodes` and `program`.
  The layout of `nodes`, `edges`, `node_kind` and `tree` is the one of 3.2.

### Geometry

- `join.join_networks` bridges a seeded fraction of the free tips of placed
  networks to partners on other networks (or, under policy `any`, anywhere
  at least `min_separation` segments away along the graph), inside a box.
  The rules are those of `anastomose`: the search radius in tip diameters,
  the forward cone, the ranking of tips before interior points, the
  exclusions of junctions, roots, stubs, junction zones and kin, the bridge
  diameter, sampling step, departure and arrival tangents, the walk's
  curvature, the collision rule against every vessel and every earlier
  bridge, and the bitwise end columns. It differs where a forest needs it
  to: networks are identified by their position in the input, never by the
  int8 tree label, so hundreds can be joined; roots and cut ends (vertices
  the crop made, or lying outside the box or within `boundary_margin` of a
  face) are never sources or partners; partners lie inside the box and a
  bridge that leaves it is redrawn; bridge points are obstacles, never
  partners; the selected tips are processed in a drawn permutation; and a
  volume budget stops the joining once the bridges placed would exceed it.
  Every degree-one vertex of the input receives exactly one outcome, the
  outcome counts agree with the event counters, and the first k bridges of
  any result are themselves a valid result.
- `join.crop_network` crops a placed network to a box by whole columns,
  keeping a column when either of its segments reaches the padded box, so
  that polylines are never split into per-segment pairs and the tip
  tangents, sampling steps, stub tests and collision excusals stay per
  polyline.
- The joining runs at a few milliseconds per selected tip: about 300
  networks of 0.25 million points with 3.4 thousand selected tips join in
  under 10 s, 650 networks of 2.1 million points with 15 thousand selected
  tips in under 2.5 minutes, both within 1 GB.

### Measurement

- `join_networks` returns a tips table (network, column, outcome) and a
  summary of components and free in-box tips per unit length before and
  after joining, besides the counters.
- The library index records, per network and in units of d_min, points,
  tips, junctions, cycles, components, total length, length- and
  volume-weighted diameter percentiles, the smallest and largest diameter,
  the minimum clearance with the count of pairs below the margin, the
  counters, and the time and peak memory of its growth.

### Tools

- `vsystem-library` grows `--count` networks of the families listed, with
  the smallest drawn diameter d_min = 1 unit and root ratios R = d0 / d_min
  drawn log-uniformly and stratified over `--ratio-range`, one network per
  bin per family; each network grows in a fresh process from a seed derived
  from the library seed and its id, through `main`'s own parser, parameter
  sampling and `grow_network`, so it equals what `vsystem` writes for the
  same settings. Archives are written atomically; `index.json` and
  `index.csv` hold the descriptors; `manifest.json` records every
  parameter, the ratio law, the code hash and the hash of every network's
  nodes and program, and its `content` part is hashed canonically. A run
  into an existing library resumes it, refusing a different parameter set
  or code hash and reporting every failure. `library_weights` gives the
  weights that turn the log-uniform library into a power-law sample.

### Compatibility

- `main.py` and every module it imports are unchanged, so `main.RNG_STREAMS`
  and the sidecar are as in 3.2; `join.py` and `library.py` keep their own
  random-stream tags, which collide with none of `main`'s.
- The new modules parse as Python 3.9 and add no dependency: numpy and
  tifffile remain the only ones, scipy optional.
- `pyproject` version 3.3.0 lists `join` and `library` among the modules and
  adds the `vsystem-library` console script.

## 3.2.0

Version 3.2 adds an explicit graph to the centreline archive, three opt-in
geometry stages, family presets that bundle them, and a descriptor tool that
measures what they change. The default command line is unchanged: for a given
seed it writes the same `nodes` and `program` arrays and the same TIFF as 3.1,
which the tests check by running the 3.1 generator, kept under
`tests/fixtures/reference_code`, alongside the current one.

### Archive

- Every archive now carries `edges`, a `(2, E)` array of column indices into
  `nodes` chaining consecutive points of a polyline and joining polylines where
  they meet, `node_kind`, a per-column label (interior, junction, tip), and
  `tree`, a per-column label of the tree a point belongs to (two trees and
  bridges are distinguished). `nodes` keeps its `(4, N)` layout with NaN
  separators, and the rasteriser ignores the additions. `graph.py` rebuilds
  the graph from coordinates alone, so archives written by earlier versions
  gain a graph when read with `load_network`.
- The sidecar records every geometry option, the seeds of the random streams
  each stage uses, the confining box when growth is confined, and a count of
  every event that terminated, redrew or skipped part of the network.

### Geometry

- `--tortuosity walk` bends each stem continuously. The grammar still decides
  where a stem starts, its length, its diameter and the bifurcation angles of
  its daughters; the path in between is a persistent random walk on the
  tangent whose persistence length is `--persistence` times the local diameter.
  Arc length equals the grammar's, so the tree's size distribution is unchanged
  while its arc/chord ratio and curvature follow `--persistence`. The default,
  `stems`, is the piecewise-linear zig-zag smoothed by a B-spline as before.
- `--avoid-collisions` keeps branches from passing through one another. Every
  point is checked against the network placed so far; a point closer than the
  sum of the radii plus `--collision-margin` to another branch is redrawn
  (walk; each redraw turns more sharply than the last, so the walk steers
  clear) or the stem is shortened to its longest clear prefix (stems). The
  overlap of vessels meeting at a junction is excused by a rule that scales
  with their radii and measures distances along the vessels, so an aneurysm
  at a branch point is not a collision while a stem curling back onto its own
  junction is. A branch that cannot be placed within
  `--collision-attempts` terminates, and the count is reported. A grid hash
  and a k-d tree index are both provided (`--collision-index`).
- `--anastomose` closes the tree into a network. A seeded fraction of tips
  (`--anastomosis-fraction`) each search within `--anastomosis-radius` tip
  diameters for a partner, another tip first and otherwise an interior point
  of a segment that is not their own and not closer kin than
  `--anastomosis-min-separation` segments along the tree (the parent and
  sister stems by default), and lying ahead of the tip within a 120° cone, and are
  joined by a bridge of diameter min(d_tip, d_partner) traced by the walk and
  pinned at both ends, leaving along the tip's direction and arriving along a
  partner tip's vessel, with the walk's curvature at the stems' persistence or
  at 8 diameters when the stems are not walked, and always routed clear of
  the network at `--collision-margin` under the collision rules, whether or
  not the tree was grown with avoidance.
  Bridges are appended as polylines whose end columns copy the joined points,
  so the graph gains a cycle per bridge. Stub tips left inside a junction's
  overlap zone by a terminated branch are excluded and counted. `--anastomose-mode arteriovenous` grows a
  second tree from the opposite face of the volume and joins arterial tips to
  venous partners first, the arteriole-capillary-venule design of a capillary
  bed. The second tree's root is moved across its face of the growth box to
  the nearest spot clear of the first tree, and the sidecar records both root
  positions.
- `--family` bundles the options: `tree` is the plain grammar, `mesh` a
  capillary-bed-like network (walk, collision avoidance, arteriovenous
  anastomosis, grown in the volume) and `tumour` a tortuous, densely looped
  network with a wide root-calibre spread and frequent aneurysms and stenoses.
  `aligned`, growth along a preferred direction, is listed but not available:
  it needs a directional bias in the turtle that no bundle of the existing
  options expresses.

### Measurement

- `describe.py` reports, for any archive with or without `edges`,
  length-weighted diameter percentiles and total length, arc/chord ratios and
  mean absolute curvature per segment, tip count and density, the junction
  degree histogram, connected components and cycles, tangent orientation and
  anisotropy, length density, the minimum surface clearance between branches
  with the count of pairs closer than the collision margin, and every counted
  event from the sidecar. The definitions live in its docstring so that other
  tools can compute the same quantities from voxel skeletons.

### Compatibility

- `computeVoxel.py` and every module it imports remain importable under
  Python 3.9; the new modules follow the same restriction.
- `main.generate_network` keeps its signature and return value and accepts the
  geometry options as keyword arguments; `main.grow_network` returns the
  geometry, graph and counters without rendering.
