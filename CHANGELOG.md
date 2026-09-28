# Changelog

## 3.2.0

Version 3.2 adds an explicit graph to the centreline archive, three opt-in
geometry stages, family presets that bundle them, and a descriptor tool that
measures what they change. The default command line is unchanged: for a given
seed it writes the same `nodes` and `program` arrays and the same TIFF as 3.1,
and the reference fixtures under `tests/fixtures` pin that.

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
  of a segment that is not their own, and are joined by a bridge of diameter
  min(d_tip, d_partner) traced by the walk and pinned at both ends, and
  checked against the network under the collision rules when avoidance is on.
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
