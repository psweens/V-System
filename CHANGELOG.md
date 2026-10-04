# Changelog

## 3.4.0

Version 3.4 adds one opt-in option to `join.join_networks`: with
`attach_roots`, a network's root inside the field of view may attach, as a
side branch, to a nearby vessel of another network that is at least as
thick. With the option off (the default) the joining draws what 3.3 drew,
which the tests check by running the 3.3 `join.py`, kept under
`tests/fixtures/reference_code_3_3`, alongside the current one.

### Geometry

- In a forest packed into a field of view, every network whose root falls
  inside the field starts with a blunt end, which real vasculature has none
  of: a vessel's upstream end is a branch off a parent vessel. With
  `attach_roots`, an eligible root (a degree-one root vertex inside the
  box, clear of the faces by `boundary_margin`, whose input `node_kind` is
  a tip) becomes the source of a bridge with the machinery of the tip
  bridges: its upstream direction is the tip tangent of the root, the
  partner is a vertex of degree one or two on another network, whatever the
  policy, inside the forward cone, within `root_radius` (default `radius`)
  root diameters, not already touching, and at least `root_partner_min_ratio`
  root diameters across at the vertex; interior points rank before tips, so
  a root prefers a side branch; the bridge has the root's diameter, so the
  vessel continues upstream at its own calibre; and the step, persistence,
  redraws, collision rule and excusals, bitwise end columns, the junction at
  an attached interior point and the consumed attached tip are as for tips.
- The root phase comes first: one Bernoulli per eligible root in (network,
  column) order (`root_fraction`), a permutation of the selected roots, then
  the root bridges, each an obstacle for every later bridge; the tip draws
  follow and are made even when the budget was spent during the root phase.
  Root and tip bridges share `max_bridge_volume`, and the first k bridges of
  any result remain a valid result. With no eligible root nothing is drawn,
  so the bridges, the generator state and the summary equal the option
  off; the tips table then differs only where a root inside the box is
  ineligible, reported as `cut_end` rather than `root`.
- With the option on, an ineligible root (outside the box, near a face or
  not a tip of the uncropped network) is reported as `cut_end` and an
  eligible one as exactly one of `root_attached`, `root_not_selected`,
  `root_no_partner`, `root_collision_failed` and `root_over_budget`, codes
  appended to `TIP_OUTCOMES`; the code `root` never occurs. A root column
  given as a NaN separator is refused with the option on.

### Measurement

- Every bridge record carries `source_kind` (`tip` or `root`); a root
  bridge's `tip` field holds the root's (network, column), so existing
  readers work unchanged.
- The counters `join_root_eligible`, `join_root_selected`,
  `join_root_attached`, `join_root_not_selected`, `join_root_no_partner`,
  `join_root_collision_failed` and `join_root_over_budget` are appended to
  `EVENT_KEYS`; the last five are read from the final table, so the outcome
  counters still partition `join_tips`. Root bridges count in the redraw,
  behind-skipped, partner and components counters but never in
  `join_eligible`, `join_selected`, `join_bridges` or
  `join_source_consumed`, so `len(bridges) == join_bridges +
  join_root_attached`.
- The summary always reports `blunt_roots` (the eligible roots, less those
  attached), `free_ends` (free tips plus blunt roots) and both per unit
  length, before and after, beside the entries of 3.3; `free_tips` keeps
  its meaning, tips that are neither roots nor cut ends.
- Measured on the typical benchmark forest (330 networks of five-generation
  trees with root diameters spread log-uniformly over [2, 8] at d_min = 1,
  235 thousand points, boundary margin 2, fraction 0.9, one process): the
  joining takes 15.5 s with the option off and 26.2 s with it on; of 266
  eligible roots 220 attach, 30 find no partner and 16 fail every redraw;
  the root bridges have a median chord of 14.7 and a median arc of 16.8
  root diameters at the default `root_radius`, and 4.1 and 4.3 at
  `root_radius` 5, where 9 roots attach; a root bridge holds 1 773 units
  cubed on average against 106 for a tip bridge; the free ends per unit
  length go from 0.0289 to 0.0067 with the option off and to 0.0053 with it
  on, the blunt roots per unit length from 0.0021 to 0.0002.

### Compatibility

- `main.py`, every module it imports and `library.py` are unchanged;
  library archives keep `"library_version": "3.3.0"`, and `join.RNG_STREAMS`
  is as in 3.3, so no new random stream is needed.
- With `attach_roots` off, `join_networks` returns what 3.3 returned on the
  3.3 fields and draws nothing more; the new keyword arguments are
  validated whether or not the option is on. New outcome codes and counters
  are appended, so existing codes and keys keep their values.
- `pyproject` version 3.4.0. Python 3.9 syntax, numpy and tifffile only.

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
