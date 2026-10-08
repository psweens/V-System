# Changelog

## 3.6.1

Version 3.6.1 makes the walk faster and changes no output.

### Geometry

- `tortuosity` turns the walk's frame without `np.cross` and
  `np.linalg.norm`, whose argument handling costs far more than the
  arithmetic on a 3-vector. `_cross` rounds each product and then each
  difference, as `np.cross` does, and `_norm` takes the square root of the
  vector's dot product with itself, as `np.linalg.norm` does for a real
  vector. Every step of the walk uses them, and so does every anastomosis
  bridge and rung walked with `bridge_path`. On an Intel Xeon Gold 5220
  workstation, walked growth takes 22 to 37% less time (capillary_bed at
  R 4, 81 s against 54 s; mesh at R 16, 78 s against 49 s); growth without
  the walk is unchanged.

### Tools

- `library_version` is 3.6.1, and the code hash changes with
  `tortuosity.py`, so a library grown with 3.6.0 refuses to resume under
  3.6.1, as any change of the generator does.

### Compatibility

- Every network, archive, sidecar, library member and description is the
  same bit for bit as in 3.6.0. `tests/test_geometry.py` checks `_cross` and
  `_norm` against numpy on 4500 vector pairs, signed zeros, subnormals,
  infinities and NaN among them, and the 3.4 and 3.5 pins pass unchanged.
  The growth times quoted under 3.6.0 and in the README were measured
  before this change.
- pyproject version 3.6.1.

## 3.6.0

Version 3.6 fills the finest calibres and joins them side to side.
`--capillary-generations` and `--capillary-runs` extend each branch that
d_min would end into a capillary tree of its own and lengthen the finest
stems; `--cross-connect` adds rungs between neighbouring capillaries after
anastomosis; and the families `aligned_tight`, `aligned_bed` and
`capillary_bed` are offered. `describe.py` measures junctions, branch angles,
shortest loops, calibre variation within segments and, on request, the
distance from tissue to the nearest vessel. With the new options off, every
family 3.5 offered draws, writes and measures what 3.5 did, which the tests
check by running the 3.5 modules, kept under
`tests/fixtures/reference_code_3_5`, alongside the current ones.

### Geometry

- `vSystem.F(n, d0, d_min=None, capillary_generations=0,
  capillary_runs=1)` and `vSystem.capillary_tree(m, d, runs)`. With m > 0
  (which needs d_min) a daughter that d_min would end becomes a capillary
  tree: m generations of symmetric bifurcations drawn at d_min, turned by
  the Zamir angle of equal daughters (acos 2^(−1/3)) and rolled by the roll
  angle, unless the iteration count ends it first; the m generations do not
  count against it, and a root thinner than d_min is still drawn as
  nothing. Every capillary stem, and every stem whose daughters both end, is
  E stem blocks, each drawn afresh. Every drawn diameter stays at or above
  d_min but a stenosis middle. At the defaults F is the released rule,
  string for string and draw for draw; the fill draws only on its own path.
- `connections.py` (new, in main's import closure): `cross_connect` joins
  capillary vessels (tree segments whose median diameter is below
  `rung_below` × d_min) side to side by rungs. Sites follow a Poisson process
  of mean gap `rung_spacing` vessel diameters (at least 1), taken in random
  order; each is joined to the nearest capillary point of another vessel
  within `rung_radius` site diameters that is not touching it, not near a
  junction, at least `rung_min_separation` segments away along the trees and
  whose chord stands at least `rung_lateral_deg` off the vessel tangent or
  off `rung_axis`. The rung has the smaller diameter, is walked along the
  chord like an anastomosis bridge, keeps the collision margin and joins
  both vessels at new junctions with bitwise shared ends; no existing
  column moves. It draws on stream tag 5 (`main.RNG_STREAMS["rungs"]`),
  created only when the rungs are on.
- `main.grow_network(..., capillary_generations=0, capillary_runs=1,
  cross_connect=False, rung_below=2.0, rung_spacing=20.0, rung_radius=8.0,
  rung_lateral_deg=60.0, rung_axis=None, rung_min_separation=2)`, which
  returns `rungs`; the command line takes `--capillary-generations`,
  `--capillary-runs`, `--cross-connect`/`--no-cross-connect` and `--rung-*`,
  and `shaping_options` carries all nine. Every malformed setting is refused
  from `grow_network` and the command line before anything is written. A
  pair filled in the volume no longer runs the free-extent pass, whose
  result a box at a fixed voxel size never uses.
- Counters appended after the guidance ones: `rung_sites`,
  `rung_sites_near_junction`, `rung_sites_consumed`, `rung_no_partner`,
  `rung_collision_failed`, `rung_bridges`, `rung_bridges_cross_tree`,
  `rung_kin_skipped`, `rung_angle_skipped` and `rung_bridge_redraws`, with
  `rung_sites` = near_junction + consumed + no_partner + collision_failed +
  bridges, all zero with the rungs off.
- The `aligned_tight` family: `aligned` with its capillaries steered harder
  and from their first step (G 2 and an onset of 0 for both rules) and its
  roots moved by −20 and +20 d_min along x, 40 d_min apart. `aligned_bed` is
  aligned_tight with capillary generations 2, runs 4 and rungs at spacing
  40, radius 8 and at least 60° off x. `capillary_bed` is an unguided
  arteriovenous pair grown in the volume with anastomosis arteriovenous 1.0,
  capillary generations 3, runs 2 and rungs at spacing 20, radius 8 and at
  least 60° off the vessel tangent. All three need d_min; their values are
  provisional. `aligned` keeps the preset 3.5.0 released.

### Measurement

- `describe(..., evd_spacing=None)` and `describe_archive` likewise. The
  appended keys are `junctions` (fractions of degree 3, 4 and at least 5,
  mean degree, segments per junction), `branch_angles_deg` (medians of the
  smallest, middle and largest angle at degree-3 vertices, between arms
  taken to arc min(2 d_v, half the segment)), `loops` (`by_segment` and
  `by_node`: the shortest cycle through a segment and through a junction on
  every k-th up to 2000, searched on the 2-core within a depth of 16 and 4096
  vertices: median, mean, histogram, the share with none, cycles per unit
  length and, by junction, loop lengths), `segment_diameter_variation`
  ((max − min)/mean of a segment's vertex diameters, junction ends left out,
  per class) and `tissue_distance` (with `evd_spacing` only: distance to the
  nearest vessel wall on a grid of at most 64³ points by the voxeliser's
  capsule rule, exact and draw-free, over describe's `volume`, else the
  growth box (`growth_box_um`, or a library archive's `growth_box`), else
  the voxel_size field, else the bounding box). Every earlier key is
  byte-identical.
- `frame_orientation` gains `fisher_axial_K_exact` per class about an
  axis: the K of the Fisher-axial law exp(K |t·a|) whose ⟨|t·a|⟩,
  1 / (1 − e^−K) − 1 / K, equals the measured one, by bisection on
  [−200, 200]. It reads 0 for isotropic tangents and is negative for
  tangents gathered across the axis. `fisher_axial_K` keeps its 3.5
  definition, coth K − 1 / K = ⟨|t·a|⟩, which reads about 1.8 for isotropic
  tangents and agrees with the exact value only as K grows.
- The library index gains `rungs`, the number of cross-connections,
  appended after the 3.5 columns; archive metadata lists the rungs.
- `tests/test_topology_benchmark.py` (slow) also reports the topology
  keys, the tissue distance, the rung outcomes and the size of each
  member's program, and grows the filled families only in their own
  invocation, named in `VSYSTEM_BENCHMARK_FAMILIES`, at the ratios in
  `VSYSTEM_BENCHMARK_RATIOS` (the last for cost only). On an Intel Xeon
  Gold 5220 workstation (Linux, Python 3.12, numpy 2.5) the fill multiplies
  the grammar's moves by about 13 in aligned_bed and 15 in capillary_bed.
  aligned_bed grows in 13, 54 and 690 s at R 4, 8 and 16 (means over seeds
  1 to 3) and 28 minutes at R 25, and capillary_bed in 34 and 335 s at R 4
  and 8 and 55 minutes for one network at R 16, against 5.6, 31, 170 and
  890 s for aligned_tight. `describe_archive` takes 21 s on the largest
  network grown, a tree of 1.08 million points at R 25, as in 3.5, and the
  tissue distance adds 2 to 6.5 s on the largest networks. A 2000-network
  library of tree, mesh, tumour, aligned and aligned_tight in equal shares
  over the default range projects to 43 CPU hours, 6.7 GB of archives and a
  peak of 0.83 GB per worker.

### Tools

- `vsystem-library` grows the new families when listed, in cubes of side
  20 R (aligned_tight, aligned_bed) and 15 R (capillary_bed) from
  `library.BOX_C`; `library_version` is 3.6.0 and `CODE_MODULES` gains
  `connections`, so libraries grown with 3.5 refuse to resume under 3.6, as
  any change of the generator does.

### Compatibility

- With the new options off, tree, mesh, tumour and aligned reproduce 3.5
  exactly: nodes, programs, tree, edges, node_kind, bridges, frames, the 3.5
  counters, the global and per-stage generator states, the command line's
  archives, sidecars and TIFFs, the library's members, index rows, manifest
  content and archives, description, placement and joining.
  `tests/test_pinned_release_3_5.py` runs the 3.5 modules beside the current
  ones, restricts every record to the keys 3.5 wrote, and checks that every
  setting 3.6 added is off, no rung is made and every appended counter is
  zero; it takes about 55 s on an Intel Xeon Gold 5220 workstation. Its
  hashes are recorded on Linux x86_64 twice, with OpenBLAS's SkylakeX
  kernels and numpy's AVX-512 loops and with OpenBLAS's Zen kernels and
  numpy's AVX2 loops, and the 3.4 pin checks the same two Linux recordings
  beside its macOS one. A recorded hash can match only where rounding
  matches a recording's: on other kernels or another numpy version a pin
  skips those cases, and its comparison of the two sets of modules in one
  environment still runs.
- Additions only: the return key `rungs`; the sidecar and metadata keys of
  the nine settings and `rungs`; the describe keys above; the ten counters;
  the index column `rungs`; and `rng_streams.rungs` in every sidecar.
- pyproject version 3.6.0; `py-modules` gains `connections`. Python 3.9
  syntax, numpy and tifffile only.

## 3.5.0

Version 3.5 lets the walk be steered. `--guidance` turns each step's heading
deterministically towards an axis or a plane under a rule chosen per calibre
class, `--root-offsets` moves the roots of a network grown in the volume,
and the `aligned` family, an arteriovenous pair whose capillaries run from
one feeder sheet to the other, is offered. Every network records its frame,
`frames.py` places archives by it and `describe.py` measures against it.
With the new options off, tree, mesh and tumour draw what 3.4 drew, which
the tests check by running the 3.4 modules, kept under
`tests/fixtures/reference_code_3_4`, alongside the current ones.

### Geometry

- `guidance.py` (new, in main's import closure) steers the walk: before
  each step's draw the heading is turned towards the target by
  ω = min(θ, r sin θ cos θ) for a nematic axis or a plane and
  ω = min(θ, r sin θ) for a polar axis, with r = min(1, step / (G d)). The
  drift draws nothing and the walk still draws two numbers per attempt. On a
  long stem the heading settles into a Watson law with K = 2P / G (nematic),
  a Fisher law with κ = 4P / G (polar) or a girdle law with K = 2P / G
  (plane); at G 3 and P 10 one stem of 2000 sub-segments measures, over
  seeds 1 to 4, ⟨cos²θ⟩ 0.821, ⟨cos θ⟩ 0.922 and ⟨sin²β⟩ 0.078 against
  0.830, 0.925 and 0.0747 from the laws: the laws' moments at 0.96 of
  their concentrations, at step / (G d) = 0.058.
- A rule has a class bound `below` in d_min (None for no bound), a `field`
  (`axis`, `plane` or None for an unguided class), an `axis` or `normal`, a
  `length` G in diameters, an `onset` (2 diameters) before which a stem is
  not steered, and for an axis a `sense` (`nematic`, `polar`) and a
  `polarity` (`root`, `fixed`, `partner`), for a plane a `bank` that turns
  the frame's perpendicular towards the normal so that branching turns stay
  in the plane. `guidance.parse_rules` validates and normalises the rules;
  every malformed specification is refused, from `parse_rules`,
  `grow_network` and the command line, before anything is written (a root
  offset outside a box that comes from the grown tree is refused when that
  network grows). The rule of a walked polyline is chosen once, from the
  diameter of its first move.
- `main.grow_network(..., guidance=None, root_offsets=None)` and
  `analyseGrammar.WalkSettings(..., guidance=None)`; the command line takes
  `--guidance` and `--root-offsets` as inline JSON, and `shaping_options`
  carries both to the presets, the sidecar and the library's `kwargs`. Root
  offsets are one vector per tree in units of d_min, added to the roots'
  default positions on the faces of the growth box before the second root
  is cleared of the first tree; they need `grow_in_volume` and d_min, and a
  root moved outside the box is refused. A `partner` polarity needs a second
  tree and root offsets that differ along its axis.
- The `aligned` preset: the mesh layout with roots offset ±15 d_min along x,
  anastomosis arteriovenous at 0.8, vessels below 2 d_min steered along x
  towards the other root (polar, `partner`, G 4.9, onset 2) and the larger
  vessels kept in planes perpendicular to x (G 5). Over seeds 1 to 10 at
  R 5 its capillary-class order S_x is 0.141 against −0.002 for the same
  seeds unguided, its capillary polar order 0.485 against 0.063, and its
  larger-class S_x −0.165 against −0.043. A preset listed as None is a
  family this version does not offer and is refused with a generic message.
- Every run counts `guided_steps`, `guidance_onset_steps`,
  `unguided_steps`, `guidance_undefined_steps`, `guidance_sense_flips` and
  `guidance_bank_steps`, appended to `main.EVENT_KEYS` after the
  anastomosis counters and at zero with guidance off.
- `grow_network` returns `frame`, which the sidecar and the library metadata
  store: `frame_version` 1, the `kind` (`none`, `axis` or `plane`) of the
  field rule with the smallest bound, its `axis` or `normal`, the `sense`,
  the `tree_senses` of a polar rule, the `origin` (the growth-box centre,
  else the first root), `grow_direction`, `grow_perpendicular` and every
  rule, in the coordinates and unit of `nodes`.
- `frames.py` (new, caller-side, outside main's import closure and
  `library.CODE_MODULES`): `rotation_about`, `rotation_between`,
  `rotation_of_frames`, `random_rotation`, `align`, `transform_nodes` and
  `transform_frame`, which move an archive and its frame together so that a
  descriptor measured about the frame's axis is unchanged by the move.

### Measurement

- `describe(..., frame=None, d_ref=None, class_bound=2.0)` and
  `describe_archive` likewise. `frame` defaults to the metadata's record and
  `d_ref` to the metadata's `d_min` (top level, else `grow_kwargs`); it never
  falls back to the smallest vertex diameter. The appended keys are `frame`,
  `classes`, `frame_orientation` (S, ⟨|t·a|⟩, the crossing ratio, the mean
  angle, the fractions within 20° and 45°, Watson K and `fisher_axial_K`
  per class about an axis; the in-plane fraction and S_n about a plane),
  `polar_order`, `orientation_by_class` (with S_max and the planarity
  1 − 3λ₃), `calibre_shares` (length and volume below 1.5, 2 and 3 d_ref),
  `segments_by_class` and `transverse_spacing` (nearest-neighbour distance
  between the crossings of capillary edges on nine planes across the axis,
  and the crossing density per unit area). Lengths come in the archive's
  unit and in d_ref. Every pre-3.5 key is byte-identical.
- The library index gains `frame_kind`, `capillary_order`, `larger_order`,
  `capillary_polar_order`, `capillary_length_share`,
  `capillary_volume_share`, `capillary_segment_median` and
  `transverse_spacing_median`, appended after the 3.4 columns; archive
  metadata gains `frame` and a top-level `d_min`.
- `tests/test_topology_benchmark.py` (slow) grows every family at R 4, 8
  and 16 over three seeds and the aligned preset over its guidance length
  and onset, prints the descriptors and the cost, and projects a
  2000-network library. On an Intel Xeon Gold 5220 workstation (Linux,
  Python 3.12, numpy 2.5) the aligned preset grows in 8.8, 49 and 185 s at
  R 4, 8 and 16 (means over seeds 1 to 3) and 600 s at R 25, about what a
  mesh takes (6.7, 29, 199 and 818 s); guidance adds about 90 µs per guided
  step and a bank about 45 µs more, 7% more time per point for the preset at
  R 8; `describe_archive` takes 21 s on the largest network grown, a tree
  of 1.08 million points at R 25; and a 2000-network library of tree,
  mesh, tumour and aligned in equal shares over the default range projects
  to 39 CPU hours, 7.0 GB of archives and a peak of 0.8 GB per worker.

### Tools

- `vsystem-library` grows `tree`, `mesh` and `tumour` by default
  (`library.DEFAULT_FAMILIES`); `aligned` is grown when listed, in a cube of
  side `library.BOX_C["aligned"]` = 15 R like a mesh. `--box-c FAMILY C`
  sets the cube side of any box family, is refused for a family that grows
  free and together with `--mesh-box-c` for mesh, and a member whose root
  offset leaves its cube is recorded as a failure; the manifest records
  `growth.box_c` only when a box family other than mesh is listed, so a
  default library's manifest content differs from 3.4's in `code_sha256`
  and `library_version` only.
  `library_version` is now the release version.

### Compatibility

- With the new options off, tree, mesh and tumour reproduce 3.4 exactly:
  nodes, programs, tree, edges, node_kind, bridges, the 3.4 counters, the
  global and per-stage generator states, the command line's archives,
  sidecars and TIFFs, the library's members, index rows, manifest content
  and archives, description and joining. `tests/test_pinned_release_3_4.py`
  runs the 3.4 modules beside the current ones, restricts every record to
  the keys 3.4 wrote, and checks that every appended counter is zero, the
  frame is of kind `none` and the new kwargs are None; it takes about 30 s
  on a laptop (Apple M4) and 95 s on an Intel Xeon Gold 5220 workstation.
- Additions only: the return key `frame`, the sidecar and metadata keys
  `frame`, `guidance` and `root_offsets` (and `d_min` in library metadata),
  the describe keys above, the six counters and the eight index columns.
  `main.RNG_STREAMS` is unchanged; guidance draws nothing. `CODE_MODULES`
  gains `guidance`, so libraries grown with 3.4 refuse to resume under 3.5,
  as any change of the generator does; they load and describe as before,
  with `frame` and the frame-relative keys None and the class keys computed
  with d_ref from `grow_kwargs`.
- pyproject version 3.5.0; `py-modules` gains `guidance` and `frames`.
  Python 3.9 syntax, numpy and tifffile only.

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
