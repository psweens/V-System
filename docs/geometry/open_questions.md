# Open questions for the maintainer

Decisions the geometry upgrades leave to the project owner, with the
measurements that bear on them (tables in `descriptors.md`, definitions in
`describe.py`). Nothing below changes a default of the plain grammar; the
provisional values live in the `--family` presets and in the defaults of the
new options only.

## 1. Persistence default (`--persistence`)

`--tortuosity walk` has no default persistence. The sweep in `descriptors.md`
(8 generations, five seeds, walk alone) gives:

| P (diameters) | 2 | 3 | 5 | 8 | 12 | 20 | 40 | stems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arc/chord, mean | 1.34 | 1.21 | 1.12 | 1.07 | 1.05 | 1.03 | 1.01 | 1.02 |
| arc/chord, P90 | 1.68 | 1.41 | 1.23 | 1.14 | 1.09 | 1.05 | 1.03 | 1.03 |
| curvature (1/µm) | 0.28 | 0.23 | 0.18 | 0.14 | 0.11 | 0.09 | 0.06 | 0.04 |

The 1.05–1.2 band for normal vasculature is P ≈ 3.5–12. Proposal:

- normal vasculature: **P = 8** (mean 1.07, P90 1.14; 1.08 with avoidance
  on). The `mesh` preset carries P = 10, whose network-level ratio is 1.25
  because the bridges are the most tortuous segments; if the whole-network
  figure is what should sit in the band, the preset's P belongs nearer 15.
- tumour-like: **P = 3** (mean 1.21, P90 1.41; the `tumour` preset's
  network-level ratio is 1.46 with its bridges).

Two things to weigh. First, the rotation rule as specified (one angle with
standard deviation sqrt(h / l_p) about one random perpendicular axis) gives a
tangent autocorrelation of exp(−s / 2 l_p) in three dimensions, so "P
diameters" is half the worm-like-chain persistence length; sqrt(2h / l_p)
would make P literal. Since P is being calibrated empirically the labelling is
what matters; the current rule is kept and documented in `tortuosity.py`.
Second, the plain `stems` mode already has arc/chord above 1 (the spline of
the 25° zig-zag; see the "default" rows), so the walk is not the only source
of tortuosity in the bank, and the lowest useful P is the one whose arc/chord
still exceeds the stems value.

## 2. Bridge shape and diameter

Three rules shape a bridge, all chosen by eye against the projections rather
than measured against tissue, and all worth a decision: partners are sought
within 120° of the tip's direction (a tip does not double back into a
hairpin); a bridge into a partner tip arrives along that vessel's direction so
the two tips become one continuous vessel, and into the side of a vessel
along the chord; and a bridge always carries the walk's curvature, at
`--persistence` when the stems are walked and at 8 diameters otherwise. The
last is a default set without a measurement of bridge tortuosity in real
capillary beds; the network-level arc/chord of anastomosed trees (1.2–1.5 in
the table) is dominated by it.

### Diameter

Bridges are min(d_tip, d_partner), which makes every bridge the finest vessel
at its junction and gives capillary-like calibres for tip-to-tip joins. A
Murray-consistent alternative would treat a tip-to-interior join as a new
bifurcation and resize the partner segment downstream of the join, or set the
bridge to the Murray daughter of the partner's diameter. That changes the
calibre distribution of the partner tree and needs a rule for which side is
"downstream" in a network with loops; it is not built. Recommendation: keep
min, since the descriptor comparison with real skeletons will show whether the
fine tail of the diameter distribution is wrong.

## 3. Whether `aligned` is worth building now

`--family aligned` is refused. Parallel capillaries (muscle, myocardium) are
not reachable by bundling parameters: the Zamir angles put daughters 37° off
the parent whatever the stem angle, and the roll of 70° spreads bifurcation
planes. Two ways to build it, both small: (a) a drift term in the walk that
rotates the tangent towards a preferred unit vector by an angle proportional
to the step (a persistent walk in a field), applied only under
`--tortuosity walk`; (b) a bias on the turtle's roll so that bifurcation planes
stay near the plane containing the preferred direction, which also affects
`stems` mode. Option (a) is about forty lines and reuses the walk's own
stream and counters; it would be measured by the tangent covariance
eigenvalues and fractional anisotropy that `describe.py` already reports.
Deferred pending a decision.

## 4. Anastomosis radius and the arteriovenous design

- The radius is in tip diameters, as specified, but partner distance scales
  with segment length (ε × d), not diameter. With the parent and sister stems
  excluded (§5 below), the nearest eligible partner lies at about 2 ε tip
  diameters: median 8.0 at ε = 4, 13.9 at ε = 7, 19.9 at ε = 10 (P90 9, 16,
  23), so the default is 25 and a radius of 10 finds nothing at ε = 7. A radius
  in multiples of the tip's own segment length (about 2.5 of them) would be
  self-scaling and is the better unit if ε is to vary between families.
- Bridges never join a tip to its parent stem or its sister's stem
  (`--anastomosis-min-separation 3`): a sister's tip is usually the nearest
  partner and joining it closes the two short sister stems into a small
  triangle, which looks nothing like an anastomosis. Setting the separation
  to 2 restores sister loops for a family where terminal arcades are wanted.
- In `arteriovenous` mode the two trees grow from opposite faces and each
  fills the box, so an arterial tip's neighbourhood is mostly arterial: with
  every tip seeking a partner, 7.0 of 35.6 bridges cross between the trees
  at radius 10 and 14.0 of 46.2 at radius 20 (the `mesh` preset, at the
  default radius and half the tips, makes 17.0 of 39.4 cross). Same-tree bridges are allowed as a
  fallback and counted separately. If a stricter capillary-bed model is wanted, the
  fallback can be turned off, or the two trees interdigitated by offsetting
  the second root within the face rather than centring it.
- With `--anastomosis-fraction 1` and a large radius, bridges pair sibling
  tips (β₁ = tips / 2) — capillary loops. Whether that is the intended limit
  or whether tips should prefer *distant* partners is a modelling choice.

## 5. Two measured trade-offs that need a decision

- **The walk alone multiplies crossings.** The zig-zag stems have zero net
  turn, so the plain tree keeps a regular layout and its branches only cross
  in deep trees. The walk lets sub-trees drift into each other: at 8
  generations, where the plain tree has no crossing, the walk has 126
  overlapping point pairs at P = 40 and 9394 at P = 2. `--tortuosity walk`
  should therefore be used with `--avoid-collisions` (the presets do), or the
  bank will contain more crossings than the plain grammar, not fewer.
- **Avoidance removes network.** A walk step that cannot be placed after
  `--collision-attempts` redraws terminates its branch and everything below
  it. Each redraw widens the turn so the walk can steer clear, which keeps
  far more of the tree than repeating the same step did (at 11 generations
  one seed keeps 18.0 of 26.4 mm instead of 1.2 mm), but at 8 generations the
  walk with avoidance still retains 68% of the plain tree's points (12.8
  terminations per tree) and the `mesh` preset 84%, with the losses falling
  on whole sub-trees. Alternatives: back up a few steps and redraw them, or
  terminate only the offending stem and reroot its daughters; either is a
  small change to the interpreter. Terminated branches also leave stub tips
  inside their junctions (a third of the tips in a mesh), which anastomosis
  now skips and counts.

## 6. Points from the audit that contradict the brief's assumptions

Listed in `audit.md` §10: the drawn stem is the approximating B-spline of the
five-piece zig-zag, not the zig-zag itself; sub-segment lengths are five
independent draws; daughters taper from the parent's diameter over their first
span and anomalies are smoothed; coincidence within a polyline is only
approximate; bounds are tested at move end points; and the default trees do
contain branch crossings (measured).

## 7. Smaller decisions taken provisionally

- `--collision-margin` defaults to 1 µm. The check is on centreline points, so
  a crossing can be missed by up to half the point spacing; the finest default
  vessels are sampled at about 0.1–0.25 diameters, i.e. 0.5–1 µm at
  d = 4 µm. A margin at or above that spacing is recommended.
- Own-stem points are checked outside an arc-length window of
  2 (r_P + r_Q + margin), which flags a walk whose radius of curvature falls
  below about 1.2 vessel radii (a self-intersecting tube). With P below about
  3 and avoidance on this produces frequent redraws, as the events columns
  show.
- Two trees in `arteriovenous` mode share d0, ε and the generation count;
  venous vessels are typically larger than their arterial counterparts, which
  a `--venous-d0-factor` could express.
- The junction overlap excused from collision checks is radius-scaled: a
  parent-or-sibling pair is excused when its two distances to the junction
  sum to less than 1.7 × (r_P + r_Q + margin). A fixed sphere of one parent
  diameter was tried first and reported hundreds of false collisions wherever
  a parent stem ended in an aneurysm; the factor 1.7 is derived from Zamir's
  angles in `collisions.py` and would need raising for a grammar with
  bifurcations narrower than about 45°. The descriptor applies the same rule
  and the same arc-length window, so the two agree on what a violation is.
- The `tumour` preset's `--d0 20 10` and anomaly probabilities of 0.1 are
  guesses, deliberately outside the plain grammar's defaults and only active
  under `--family tumour`.
