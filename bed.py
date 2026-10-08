"""
An explicit capillary bed: hard-core seeds placed by random sequential
addition, joined greedily shortest-first under a degree cap, a smallest
branch angle, a girth bound and the collision rule, pruned of dead ends and
islands, walked as persistent segments, and attached to the tips of two
feeder trees by bitwise end columns. States its rules, its draws
(sub-streams [seed, 7, k]) and its laws. Every decision is computed
elementwise, never through BLAS, so that seeds, candidates and the greedy
pass do not depend on the kernel. Python 3.9 compatible.

Units. Lengths are in grammar units. With d_min the unit of the settings,
the hard core is h = spacing d_min, the bed vessel diameter is
d = diameter d_min, the clearance between two bed vessels is
T = d + collision_margin, chords and walks are sampled every step d, and a
walk has persistence `persistence` bed diameters.

Seeds. The domain (low, high) is inset by d/2 on every side. With stretch
s > 1 the seed metric is compressed along the axis a of the network's frame
(frame["grow_direction"]): c = x + (1/s - 1)(x . a) a, so that seeds and
candidate edges stand s times further apart along a than across it; s = 1
leaves the metric alone. The target count is N_t = floor(density V_c / h^3),
V_c the volume of the inset domain over s, known before anything is drawn. A
proposal is x = low + (high - low) u with u three doubles of the stream; it
is refused when it lies strictly within r_f + d/2 + margin of a feeder column
(r_f its vertex radius, outcome "on_feeders", tested first) or strictly
within h of an accepted seed in the compressed metric (outcome
"overlapping"), and accepted otherwise, until N_t seeds stand. The
semantics are sequential, one proposal at a time; the proposals are drawn
in blocks of RSA_BLOCK and resolved in chunks of RSA_CHUNK, which changes
neither the seeds nor the outcomes, only how far the generator has run.
When ceil(RSA_PROPOSAL_CAP V_c / h^3) proposals have been considered short
of N_t, SeedsNotPlaced is raised; nothing is shrunk.

Tips. A feeder tip is a vertex of degree one other than a root (the first
column of each tree label other than BRIDGE). A tip inside the overlap zone
of a junction of its own polyline, |x_t - x_J| < KIN_REACH (r_t + r_J +
margin), J the polyline's first vertex or a vertex of degree three or more
on it, is a stub and takes no edge, as in anastomosis. Every other tip is
eligible, whatever its calibre; tip t is bed vertex N_t + t.

Candidates. Every pair of seeds, and every pair of a seed and an eligible
tip, closer than reach h in the compressed metric (strictly) is a candidate
edge; two tips never are. The pairs (i, j), i < j, are listed in
lexicographic order, ranked by the key chord_c (1 + jitter u), chord_c the
compressed length and u one uniform draw per pair, and processed in the order
np.lexsort((j, i, key)).

The greedy pass. A candidate is refused at the first rule it fails, in this
order:
- degree: an end already holds its cap, three bed edges at a seed and
  tip_edges at a tip (whose feeder stem makes 1 + tip_edges);
- angle: at either end, the new edge's unit direction e from that end and an
  existing direction u there have u . e > cos(min_angle); at a tip the
  feeder's inward direction is one of them, the unit vector from the tip
  towards the last distinct point of its polyline before it (the opposite
  of anastomosis._tip_tangent, normalised through _dist);
- girth: the two ends are at most girth - 2 edges apart in the graph of the
  accepted bed edges and the feeder segment graph (one edge per
  graph.segments path of the feeders, between its end vertices), so the new
  edge would close a cycle shorter than girth;
- clearance: at a shared end, u . e > cos(KIN_ANGLE_DEG) with u the
  direction of another bed edge (also at a tip, never against the feeder):
  two straight chords meeting below 2 asin(1 / KIN_REACH) = 72.06 degrees
  overlap beyond the kin excuse of collisions.py; or the chord comes closer
  than T to an accepted chord that shares no end with it (the clamped closed
  form of the distance between two segments); or a point of its stored chord
  samples collides with a feeder column under the rule below.
A refusal at a shared end counts in bed_refused_clearance and in
bed_refused_clearance_at_end. With min_angle 60 the smallest angle between
two bed chords is therefore 72.06 degrees. The chord samples of an edge are
bridge_path(x_lo, x_hi, step d) with no tangents and no generator, from the
lower vertex id to the higher, computed once and stored: they are a straight
chord, which the fallback below replays byte for byte. Seeds are not
obstacles to chords: a chord accepted close to a seed leaves that seed no edge
that would clear it, and the seed is peeled.

Pruning. Every seed with at most one bed edge is removed, with its edge,
until none is left; tips are never removed. With feeders, every remaining
component of the bed graph (kept edges, seeds and tips) that holds no tip is
an island and is removed.

Chains. The pruned graph is cut into chains by graph.segments over the kept
edges plus one pendant edge from every tip with a bed edge to a phantom
vertex, the phantoms then stripped, so that every tip ends its chains and
every chain's interior is seeds of degree two. A chain that returns to its
start is closed: a ring of seeds of degree two, or a lasso [J, ..., J] through
a junction or a tip J.

The walk. The chains are walked in that order. An attempt traces one
bridge_path per edge, from vertex to vertex along the chain, with the
generator of stream 3, persistence `persistence` bed diameters and diameter
d, leaving and reaching an interior seed along the Catmull-Rom tangent
unit(x_{i+1} - x_{i-1}) and the chain's ends along the chord; the pieces are
joined into one polyline, each joint once. The polyline is checked against
the feeders, the chains placed before it, the stored chords of the chains
not placed yet, and itself, and redrawn up to `attempts` times. When every
attempt collides, the chain's stored chord samples (reversed per edge where
the chain runs from a higher vertex id to a lower) are checked the same way
and placed as its "chord"; when they collide too the chain is "dropped": its
edges go, the graph is peeled and pruned of islands again, and every chain
whose edges go with it is retired (a placed one removed, one not yet walked
never attempted, drawing nothing). A lasso is walked from J as graph.segments
returns it, then written starting and ending at its middle seed, so that J
lies inside the polyline: describe measures the arc length from a point to a
shared vertex from that vertex's first column in the point's polyline, and a
polyline starting and ending at J would leave its last points unexcused
against the third vessel at J. A chain's record lists its vertices in the
order written. Walked chains are not confined to the box, as anastomosis
bridges and rungs are not.

The collision rule is describe.clearance's, as connections applies it: two
points P and Q collide when |P - Q| - r_P - r_Q < margin, except two points
of one polyline whose arc-length separation (the shorter way round a closed
one) is at most 2 (r_P + r_Q + margin), and two points of polylines that
share a vertex J when arc(P, J) + arc(Q, J) < KIN_REACH (r_P + r_Q + margin),
arc lengths measured along each polyline. A feeder point decides with its
vertex radius and is excused with its own column's, so that a bed that
passes this rule passes describe.clearance.

Attachment. The feeders' columns are kept unchanged; each placed chain is
appended after one NaN separator, labelled BRIDGE, with diameter d, its end
columns and every seed column written from the stored coordinates, so that a
graph rebuilt from coordinates joins it to the feeder tips and to the other
chains bitwise. A tip with k bed edges ends with degree 1 + k.

Laws (the counters of EVENT_KEYS, appended to main's):
  E1 bed_proposals = bed_proposals_on_feeders + bed_proposals_overlapping
     + bed_seeds;
  E2 bed_seeds = planned_seeds(high - low, settings, d_min), the seeds returned;
  E3 bed_candidates = bed_refused_degree + bed_refused_angle
     + bed_refused_girth + bed_refused_clearance + bed_edges, and
     bed_candidates_tip <= bed_candidates;
  E4 bed_refused_clearance_at_end <= bed_refused_clearance, and bed_edges is
     the number of edges the greedy pass accepted;
  E5 bed_edges = kept edges + bed_edges_dead_end + bed_edges_island
     + bed_edges_dropped;
  E6 bed_seeds = kept seeds + bed_seeds_dead_end + bed_seeds_island;
  E7 bed_segments = bed_segments_walked + bed_segments_chord
     + bed_segments_dropped, and the placed chains number
     bed_segments_walked + bed_segments_chord - bed_segments_removed;
  E8 bed_tip_edges is the sum of the tips' kept bed edges and
     bed_tips_attached the tips with one or more, at most
     bed_tips - bed_tips_inside_junction;
  E9 bed_components_one_tree <= bed_components (kept components of the bed
     graph with an edge; one tree when every tip in it belongs to one feeder
     tree), and bed_walk_redraws <= attempts * bed_segments.

Draws. Nothing is drawn from the global generators. Sub-stream k is
numpy.random.default_rng([seed, RNG_TAG, k]), each created only when it
draws: k = 1 the proposals, rng.random((RSA_BLOCK, 3)) per block; k = 2 one
rng.random(n_pairs) for the sort key, when jitter > 0 and a pair exists;
k = 3 the walks, bridge_path's two draws per step, by chain, then attempt,
then edge. k starts at 1, so that no sub-stream draws what the recorded
[seed, 7] would.

Arithmetic. Every decision goes through _dist (np.linalg.norm(a - b,
axis=1)), _dot3 (the three products summed in order) and _compress, all
elementwise; never through @, np.dot, np.einsum, np.inner, np.matmul or a
one-dimensional np.linalg.norm, whose last bit depends on the BLAS kernel.
The walk and the feeder test stay kernel-bound through bridge_path, as the
anastomosis bridges are.
"""
import copy
import dataclasses
import math
import numbers
import time

import numpy as np

import graph
from anastomosis import BRIDGE
from collisions import KIN_REACH
from spatial import make_index
from tortuosity import bridge_path

KINDS = ("foam",)                                                      # append-only
SETTING_KEYS = ("kind", "spacing", "girth", "min_angle", "reach", "diameter", "persistence",
                "feeder_stop", "tip_edges", "stretch", "density", "jitter", "step")   # append-only

# The counters the bed reports, appended to main.EVENT_KEYS. Append-only.
EVENT_KEYS = ("bed_proposals", "bed_proposals_on_feeders", "bed_proposals_overlapping", "bed_seeds",
              "bed_tips", "bed_tips_inside_junction",
              "bed_candidates", "bed_candidates_tip",
              "bed_refused_degree", "bed_refused_angle", "bed_refused_girth",
              "bed_refused_clearance", "bed_refused_clearance_at_end", "bed_edges",
              "bed_edges_dead_end", "bed_edges_island", "bed_edges_dropped",
              "bed_seeds_dead_end", "bed_seeds_island",
              "bed_segments", "bed_segments_walked", "bed_segments_chord", "bed_segments_dropped",
              "bed_segments_removed", "bed_walk_redraws",
              "bed_tips_attached", "bed_tip_edges", "bed_components", "bed_components_one_tree")

SEED_OUTCOMES = ("seed", "on_feeders", "overlapping")                   # append-only
SEGMENT_OUTCOMES = ("walked", "chord", "dropped")                       # append-only
RNG_TAG = 7                                    # bed.py's own tag, as join's 3 and the library's 4
SUBSTREAMS = {"seeds": 1, "order": 2, "walks": 3}   # k of default_rng([seed, RNG_TAG, k])
RSA_PROPOSAL_CAP = 64.0                        # proposals per V_c / h^3 before refusing; part of the rule
RSA_BLOCK = 4096                               # proposals drawn per call; a performance constant
RSA_CHUNK = 256                                # acceptance resolved in prefix chunks of a block
GIRTH_MAX, STRETCH_MAX, STEP_MIN, DENSITY_MAX = 16, 4.0, 0.1, 0.6     # parse bounds
KIN_ANGLE_DEG = 2.0 * math.degrees(math.asin(1.0 / KIN_REACH))   # 72.06; the shared-end bound

# Relative widening of a search that only gathers candidates for an exact
# test, so that rounding in the index's own comparison cannot drop a pair the
# exact test would keep.
_WIDEN = 1e-9

# Relative safety of the prefilter that spares a chord far from every feeder
# the pointwise test; it only decides which chords are sampled, never a verdict.
_NEAR_SAFETY = 1e-6

# Pairs compared per block in the self-check of a polyline.
_PAIR_BLOCK = 1 << 20


class BedTooLarge(ValueError):
    """A bed whose planned candidate edges exceed the caller's cap (the library's --bed-max-candidates)."""


class SeedsNotPlaced(ValueError):
    """The RSA reached its proposal cap before the target count; nothing is shrunk."""


# --------------------------------------------------------------------------
# Elementwise arithmetic


def _dist(a, b):
    """Row-wise distances of two (m, 3) arrays (or a (1, 3) row against (m, 3)): np.linalg.norm(a - b, axis=1)."""
    return np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), axis=1)


def _dot3(u, v):
    """The dot product over the last axis of length three, summed in order, elementwise."""
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]


def _compress(X, axis, stretch):
    """
    The seed metric's coordinates, c = x + (1/stretch - 1)(x . a) a,
    elementwise; X itself when stretch is 1.
    """
    X = np.asarray(X, dtype=float)
    if stretch == 1.0:
        return X
    if axis is None:
        raise ValueError("a stretch above 1 needs an axis")
    axis = np.asarray(axis, dtype=float)
    return X + (1.0 / stretch - 1.0) * _dot3(X, axis)[..., None] * axis


def _stream(seed, k):
    """Sub-stream k of the bed, numpy.random.default_rng([seed, RNG_TAG, k]), looked up at call time."""
    return np.random.default_rng([int(seed), RNG_TAG, int(k)])


def _unit_rows(delta):
    """Rows of delta divided by their lengths, (m, 3), and the lengths."""
    length = _dist(delta, np.zeros((1, 3)))
    return delta / length[:, None], length


# --------------------------------------------------------------------------
# Settings


def _is_number(value):
    # any real number but a bool, numpy scalars included; an integer too large
    # for a float is not finite
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_whole(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _real(value, name, test, wording):
    if not _is_number(value) or not test(float(value)):
        raise ValueError(f"the bed {name} must be {wording}, got {value!r}")
    return float(value)


def _normalised(settings):
    """The fields of a BedSettings checked and in their JSON types; ValueError on the first bad one."""
    kind = settings.kind
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError(f"the bed kind must be one of {KINDS}, got {kind!r}")
    values = {"kind": str(kind)}
    values["spacing"] = _real(settings.spacing, "spacing", lambda v: v > 0.0, "a positive number (d_min)")
    girth = settings.girth
    if not _is_whole(girth) or not 3 <= girth <= GIRTH_MAX:
        raise ValueError(f"the bed girth must be an integer in [3, {GIRTH_MAX}] (edges), got {girth!r}")
    values["girth"] = int(girth)
    values["min_angle"] = _real(settings.min_angle, "min_angle", lambda v: 0.0 <= v <= 120.0,
                                "a number in [0, 120] (degrees)")
    values["reach"] = _real(settings.reach, "reach", lambda v: v > 1.0, "a number above 1 (spacings)")
    values["diameter"] = _real(settings.diameter, "diameter", lambda v: v > 0.0, "a positive number (d_min)")
    values["persistence"] = _real(settings.persistence, "persistence", lambda v: v > 0.0,
                                  "a positive number (bed diameters)")
    values["feeder_stop"] = _real(settings.feeder_stop, "feeder_stop", lambda v: v >= 1.0,
                                  "a number of at least 1 (d_min)")
    tip_edges = settings.tip_edges
    if not _is_whole(tip_edges) or tip_edges not in (1, 2):
        raise ValueError(f"the bed tip_edges must be the integer 1 or 2, got {tip_edges!r}")
    values["tip_edges"] = int(tip_edges)
    values["stretch"] = _real(settings.stretch, "stretch", lambda v: 1.0 <= v <= STRETCH_MAX,
                              f"a number in [1, {STRETCH_MAX:g}]")
    values["density"] = _real(settings.density, "density", lambda v: 0.0 < v <= DENSITY_MAX,
                              f"a number in (0, {DENSITY_MAX:g}] (rho h^3)")
    values["jitter"] = _real(settings.jitter, "jitter", lambda v: v >= 0.0, "a non-negative number")
    values["step"] = _real(settings.step, "step", lambda v: v >= STEP_MIN,
                           f"a number of at least {STEP_MIN:g} (bed diameters)")
    return values


@dataclasses.dataclass(frozen=True)
class BedSettings:
    """
    The settings of an explicit bed, checked when made: every instance is
    valid, with its numbers as floats and girth and tip_edges as ints.
    """
    kind: str = "foam"
    spacing: float = 7.5        # hard-core distance of the seeds, d_min
    girth: int = 8              # shortest cycle allowed, in bed-graph edges
    min_angle: float = 60.0     # smallest angle between two edges at a vertex, degrees
    reach: float = 2.5          # longest candidate edge, in spacings (compressed metric)
    diameter: float = 1.0       # bed vessel diameter, d_min
    persistence: float = 8.0    # walk persistence, bed diameters
    feeder_stop: float = 2.0    # the feeders' grammar stops below feeder_stop x d_min
    tip_edges: int = 2          # most bed edges one feeder tip takes
    stretch: float = 1.0        # >= 1; seed metric compressed by 1/stretch along the frame axis
    density: float = 0.5        # target rho h^3 of the seeds
    jitter: float = 0.25        # sort key = chord (1 + jitter u), u ~ U[0, 1)
    step: float = 0.5           # walk and chord point spacing, bed diameters

    def __post_init__(self):
        for name, value in _normalised(self).items():
            object.__setattr__(self, name, value)

    @classmethod
    def parse(cls, spec):
        """
        The settings of a dict (keys from SETTING_KEYS, the rest defaulted) or
        of a BedSettings. Draws nothing.

        Raises:
            ValueError: for anything else, an unknown key or a value out of
            its bounds (see _normalised).
        """
        if isinstance(spec, BedSettings):
            return cls(**spec.as_dict())
        if not isinstance(spec, dict):
            raise ValueError(f"the bed settings must be a dict or BedSettings, got {type(spec).__name__}")
        unknown = [key for key in spec if key not in SETTING_KEYS]
        if unknown:
            raise ValueError(f"unknown bed setting {unknown[0]!r}; the settings are {', '.join(SETTING_KEYS)}")
        return cls(**spec)

    def as_dict(self):
        """JSON-ready, keys in SETTING_KEYS order, girth and tip_edges as int."""
        return {key: getattr(self, key) for key in SETTING_KEYS}


def parse_settings(spec):
    """The normalised dict of a spec, BedSettings.parse(spec).as_dict(); normalising twice changes nothing."""
    return BedSettings.parse(spec).as_dict()


def _positive_d_min(d_min):
    if not _is_number(d_min) or not d_min > 0.0:
        raise ValueError(f"the bed needs d_min, a positive number, got {d_min!r}")
    return float(d_min)


def _margin(collision_margin):
    if not _is_number(collision_margin) or not collision_margin >= 0.0:
        raise ValueError(f"the bed needs a collision margin, a non-negative number, got {collision_margin!r}")
    return float(collision_margin)


def check_settings(settings, *, d_min, collision_margin):
    """
    The refusals that need d_min and the margin: seeds a hard core apart must
    not touch, spacing d_min > diameter d_min + collision_margin.

    Returns:
        BedSettings: the parsed settings.

    Raises:
        ValueError: for settings BedSettings.parse refuses, a d_min that is
        not a positive number, a margin that is not a non-negative number,
        and a spacing at or below the diameter plus the margin.
    """
    parsed = BedSettings.parse(settings)
    d_min = _positive_d_min(d_min)
    margin = _margin(collision_margin)
    if not parsed.spacing * d_min > parsed.diameter * d_min + margin:
        raise ValueError(f"the bed spacing {parsed.spacing:g} d_min must exceed the diameter {parsed.diameter:g} "
                         f"d_min plus the collision margin {margin:g}, or seeds would touch")
    return parsed


def _extents(extents):
    extents = np.asarray(extents, dtype=float)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError(f"the bed extents must be three finite numbers, got {extents!r}")
    return [float(v) for v in extents]


def planned_seeds(extents, settings, d_min):
    """
    The target seed count N_t = floor(density V_c / h^3) of a box of the
    given extents, V_c = prod(extent - d) / stretch; 0 when the inset box is
    empty. Draws nothing.
    """
    parsed = BedSettings.parse(settings)
    d_min = _positive_d_min(d_min)
    h = parsed.spacing * d_min
    d = parsed.diameter * d_min
    sides = [side - d for side in _extents(extents)]
    if min(sides) <= 0.0:
        return 0
    volume = sides[0] * sides[1] * sides[2]
    return int(math.floor(parsed.density * (volume / parsed.stretch) / h ** 3))


def planned_candidates(extents, settings, d_min):
    """
    The expected seed-pair candidate edges of a box, the guard's count:
    ceil(N_t density (4 pi / 3) reach^3 / 2). Tips are left out. Draws nothing.
    """
    parsed = BedSettings.parse(settings)
    n = planned_seeds(extents, parsed, d_min)
    return int(math.ceil(n * parsed.density * (4.0 * math.pi / 3.0) * parsed.reach ** 3 / 2.0))


# --------------------------------------------------------------------------
# Seeds


class _Blocked:
    """
    Points with radii a proposal must stay out of: x is blocked when
    _dist(x, p_j) < radii_j for some j. Indexed by octave of radius, so that
    a query reaches only as far as the widest radius of each class.
    """

    def __init__(self, points, radii):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        if len(points) != len(radii) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(radii)) \
                or np.any(radii < 0.0):
            raise ValueError("blocked must be (points (m, 3), radii (m,)) with finite points and radii >= 0")
        self.points, self.radii = points, radii
        self.indexes = []
        for members in _radius_classes(radii):
            r_class = float(radii[members].max())
            index = make_index("grid", cell_size=max(2.0 * r_class, 1e-9), rebuild_every=0)
            index.add(points[members], radii[members], members)
            self.indexes.append((index, r_class))

    def mask(self, X):
        blocked = np.zeros(len(X), dtype=bool)
        for index, r_class in self.indexes:
            qi, si = index.query(X, 0.0, margin=_WIDEN * r_class + 1e-300)
            if qi.size:
                j = index.tags[si]
                hit = _dist(X[qi], self.points[j]) < self.radii[j]
                blocked[qi[hit]] = True
        return blocked


def _radius_classes(radii, levels=4):
    """Member indices by octave of radius below the largest, at most `levels` classes, the widest first."""
    radii = np.asarray(radii, dtype=float)
    if radii.size == 0:
        return []
    r_max = float(radii.max())
    if r_max <= 0.0:
        return [np.arange(radii.size)]
    with np.errstate(divide="ignore"):
        level = np.floor(np.log2(r_max / np.maximum(radii, np.finfo(float).tiny)))
    level = np.clip(level, 0, levels - 1).astype(np.int64)
    return [np.flatnonzero(level == k) for k in range(levels) if np.any(level == k)]


def place_seeds(rng, low, high, *, spacing, count, stretch=1.0, axis=None, blocked=None,
                block=RSA_BLOCK, cap=None):
    """
    Hard-core random sequential addition in the box [low, high).

    Proposal i is low + (high - low) u_i, u_i the doubles 3i .. 3i + 2 of
    `rng`, drawn as rng.random((block, 3)) per block. It is refused when
    blocked (outcome 1, tested first), when its compressed coordinates lie
    strictly within `spacing` of an accepted seed's, _dist < spacing
    (outcome 2), and accepted otherwise (outcome 0), until `count` seeds
    stand; the proposals after the last accepted one are not considered.
    The answer is the one a loop over the proposals one at a time gives,
    whatever `block`; blocks are resolved in chunks of RSA_CHUNK.

    Args:
        rng (numpy.random.Generator): the seed stream.
        low, high (sequence): the corners of the proposal box.
        spacing (float): the hard core h.
        count (int): the target number of seeds; 0 draws nothing.
        stretch (float): >= 1; the metric is compressed by 1/stretch along `axis`.
        axis (sequence or None): unit axis of the compression, needed when stretch > 1.
        blocked (tuple or None): (points (m, 3), radii (m,)).
        block (int): proposals drawn per call.
        cap (int or None): proposals considered before refusing; None is
            ceil(RSA_PROPOSAL_CAP V_c / h^3), V_c the box volume over stretch.

    Returns:
        tuple: (seeds (count, 3) in acceptance order, outcomes (considered,)
        int8 indices into SEED_OUTCOMES).

    Raises:
        SeedsNotPlaced: when `cap` proposals are considered short of `count`.
    """
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))) \
            or not np.all(high > low):
        raise ValueError("place_seeds needs finite corners with high > low")
    h = float(spacing)
    if not h > 0.0 or not math.isfinite(h):
        raise ValueError(f"the spacing must be a positive number, got {spacing!r}")
    if not _is_whole(count) or count < 0:
        raise ValueError(f"the seed count must be a non-negative integer, got {count!r}")
    if not _is_whole(block) or block < 1:
        raise ValueError(f"the block must be a positive integer, got {block!r}")
    stretch = float(stretch)
    if not 1.0 <= stretch < math.inf:
        raise ValueError(f"the stretch must be at least 1, got {stretch!r}")
    if stretch != 1.0 and axis is None:
        raise ValueError("a stretch above 1 needs an axis")
    count, block = int(count), int(block)
    seeds = np.empty((count, 3))
    if count == 0:
        return seeds, np.zeros(0, dtype=np.int8)
    if cap is None:
        size = high - low
        volume = float(size[0]) * float(size[1]) * float(size[2])
        cap = int(math.ceil(RSA_PROPOSAL_CAP * (volume / stretch) / h ** 3))
    cap = int(cap)
    hits = None if blocked is None else _Blocked(*blocked)

    compressed = np.empty((count, 3))
    index = make_index("grid", cell_size=h)         # accepted seeds, compressed; grows
    accepted = 0
    considered = 0
    outcomes = []
    while accepted < count:
        if considered >= cap:
            raise SeedsNotPlaced(f"the bed placed only {accepted} of {count} seeds within {cap} proposals")
        X = low + (high - low) * rng.random((block, 3))
        usable = min(block, cap - considered)
        X = X[:usable]
        C = _compress(X, axis, stretch)
        out = np.full(usable, 2, dtype=np.int8)
        on_feeders = hits.mask(X) if hits is not None else np.zeros(usable, dtype=bool)
        out[on_feeders] = 1
        stop = usable
        for start in range(0, usable, RSA_CHUNK):
            end = min(start + RSA_CHUNK, usable)
            chosen = _resolve_chunk(C[start:end], ~on_feeders[start:end], index, compressed[:accepted], h)
            need = count - accepted
            if chosen.size >= need:
                chosen = chosen[:need]
                stop = start + int(chosen[-1]) + 1
            rows = start + chosen
            out[rows] = 0
            seeds[accepted:accepted + rows.size] = X[rows]
            compressed[accepted:accepted + rows.size] = C[rows]
            if rows.size:
                index.add(C[rows], 0.0, np.arange(accepted, accepted + rows.size))
            accepted += rows.size
            if accepted == count:
                break
        outcomes.append(out[:stop])
        considered += stop
    return seeds, np.concatenate(outcomes)


def _resolve_chunk(C, free, index, accepted, h):
    """
    The chunk positions accepted in sequence: free, not within h of an
    earlier seed, and not within h of an earlier position of the chunk that
    was itself accepted.
    """
    candidates = np.flatnonzero(free)
    if candidates.size and len(index):
        qi, si = index.query(C[candidates], 0.0, margin=h * (1.0 + _WIDEN))
        if qi.size:
            near = _dist(C[candidates[qi]], accepted[index.tags[si]]) < h
            clash = np.zeros(candidates.size, dtype=bool)
            clash[qi[near]] = True
            candidates = candidates[~clash]
    if candidates.size < 2:
        return candidates
    # conflicts inside the chunk, earlier position first, by a box prefilter
    # and the exact distance
    P = C[candidates]
    reach = h * (1.0 + _WIDEN)
    close = np.abs(P[:, None, 0] - P[None, :, 0]) < reach
    a, b = np.nonzero(np.triu(close, 1))
    if a.size:
        keep = (np.abs(P[a, 1] - P[b, 1]) < reach) & (np.abs(P[a, 2] - P[b, 2]) < reach)
        a, b = a[keep], b[keep]
    if a.size:
        keep = _dist(P[b], P[a]) < h
        a, b = a[keep], b[keep]
    if a.size == 0:
        return candidates
    order = np.lexsort((a, b))
    ok = [True] * candidates.size
    for i, j in zip(a[order].tolist(), b[order].tolist()):
        if ok[i]:
            ok[j] = False
    return candidates[np.array(ok, dtype=bool)]


# --------------------------------------------------------------------------
# The feeders


def _feeder_tips(nodes, tree, margin, built=None):
    """
    The eligible tips and the stubs of a feeder archive, as column arrays.

    A tip is a vertex of degree one that is not a root (the first column of
    each tree label other than BRIDGE). It is a stub when it lies inside the
    overlap zone of a junction of its own polyline, |x_t - x_J| < KIN_REACH
    (r_t + r_J + margin), J the polyline's first vertex or a vertex of degree
    three or more on it other than t, radii half the vertex diameters
    (anastomosis's rule, measured through _dist).

    Returns:
        tuple: (tips, stubs), int64 column arrays in increasing order.
    """
    nodes = np.asarray(nodes, dtype=float)
    tree = np.asarray(tree)
    n = nodes.shape[1]
    empty = np.zeros(0, dtype=np.int64)
    if n == 0:
        return empty, empty
    if built is None:
        built = graph.build(nodes)
    canonical, degree = built["canonical"], built["degree"]
    vertices = np.flatnonzero(canonical == np.arange(n))
    roots = set()
    for label in np.unique(tree[tree >= 0]).tolist():
        if label == BRIDGE:
            continue
        first = np.flatnonzero((tree == label) & (canonical >= 0))
        if first.size:
            roots.add(int(canonical[first[0]]))
    candidates = vertices[degree[vertices] == 1]
    candidates = candidates[~np.isin(candidates, np.array(sorted(roots), dtype=np.int64))]
    if candidates.size == 0:
        return empty, empty
    diameter = graph.vertex_diameter(nodes, canonical)
    polyline = graph.polyline_of_column(nodes)

    # the junctions of every polyline: its first vertex and every vertex of
    # degree three or more along it, as sorted codes polyline * n + vertex
    columns = np.flatnonzero(canonical >= 0)
    owner = polyline[columns]
    vertex = canonical[columns]
    first = np.ones(columns.size, dtype=bool)
    first[1:] = owner[1:] != owner[:-1]
    junction = first | (degree[vertex] >= 3)
    codes = np.unique(owner[junction] * n + vertex[junction])
    tip_owner = polyline[candidates]
    lo = np.searchsorted(codes, tip_owner * n, side="left")
    hi = np.searchsorted(codes, (tip_owner + 1) * n, side="left")
    counts = hi - lo
    which = np.repeat(np.arange(candidates.size), counts)
    slots = np.repeat(lo - (np.cumsum(counts) - counts), counts) + np.arange(int(counts.sum()))
    J = codes[slots] % n
    t = candidates[which]
    other = J != t
    which, J, t = which[other], J[other], t[other]
    stub = np.zeros(candidates.size, dtype=bool)
    if which.size:
        distance = _dist(nodes[:3, J].T, nodes[:3, t].T)
        reach = KIN_REACH * (diameter[t] / 2.0 + diameter[J] / 2.0 + margin)
        stub[which[distance < reach]] = True
    return candidates[~stub].astype(np.int64), candidates[stub].astype(np.int64)


def _inward(nodes, columns, tip):
    """
    The feeder's unit inward direction at a tip: from the tip towards the
    last distinct point of its polyline `columns` before it (the first after
    it when the tip starts the polyline), the opposite of
    anastomosis._tip_tangent, normalised through _dist so that the angle
    test does not depend on the BLAS kernel. None when every other point of
    the polyline coincides with the tip.
    """
    columns = [int(c) for c in columns]
    at = columns.index(int(tip))
    behind = columns[at - 1::-1] if at > 0 else columns[at + 1:]
    here = nodes[:3, tip]
    for other in behind:
        there = nodes[:3, other]
        length = float(_dist(here[None, :], there[None, :])[0])
        if length > 0.0:
            return -((here - there) / length)
    return None


def _column_arcs(nodes, polyline):
    """Arc-length position of every finite column along its polyline, as describe and connections compute it."""
    n = nodes.shape[1]
    arc = np.zeros(n)
    columns = np.flatnonzero(np.isfinite(nodes[:3]).all(axis=0))
    if columns.size == 0:
        return arc
    points = nodes[:3, columns].T
    owner = polyline[columns]
    first = np.ones(columns.size, dtype=bool)
    first[1:] = owner[1:] != owner[:-1]
    step = np.zeros(columns.size)
    inside = ~first[1:]
    if inside.any():
        step[1:][inside] = _dist(points[1:][inside], points[:-1][inside])
    cumulative = np.cumsum(step)
    arc[columns] = cumulative - cumulative[np.flatnonzero(first)][owner]
    return arc


def _polyline_arcs(points):
    """Arc-length position of every point of one polyline from its first."""
    arcs = np.zeros(len(points))
    if len(points) > 1:
        arcs[1:] = np.cumsum(_dist(points[1:], points[:-1]))
    return arcs


class _Obstacles:
    """
    What a bed polyline is checked against, under describe's clearance rules.

    - The feeders: their vertices (canonical columns), indexed by octave of
      vertex radius, each class built once (rebuild_every 0, cell
      2 r_class + margin), tagged by column, with every column's own radius,
      polyline and arc position for the kin excuse.
    - The placed chains: a growing grid of their points (cell 2 radius +
      margin), with each point's chain and arc position along its polyline.
    - The chords of the chains not placed yet: indexed once by `set_chords`,
      with each point's chain and arc position.
    A chain's chord is masked once the chain is placed, dropped or retired,
    and its placed polyline once it is retired as placed. Each chain carries
    anchors, {bed vertex: arc position}, of the vertices its polyline may
    share with another; `tip_columns` maps a bed tip vertex to its feeder
    column, the vertex a bed polyline shares with a feeder polyline.

    Args:
        nodes (ndarray): the feeders' (4, N) archive; (4, 0) for none.
        radius (float): the bed radius, d / 2.
        margin (float): the collision margin.
        index_kind (str): spatial.make_index kind of the indexes built once.
        tip_columns (dict or None): bed tip vertex -> feeder column.
    """

    def __init__(self, nodes, radius, margin, index_kind="grid", tip_columns=None):
        nodes = np.asarray(nodes, dtype=float).reshape(4, -1)
        self.radius = float(radius)
        self.margin = float(margin)
        self.kind = index_kind
        self.tip_columns = {int(v): int(c) for v, c in (tip_columns or {}).items()}
        n = nodes.shape[1]
        self.nodes = nodes
        if n:
            canonical = graph.canonical_columns(nodes)
            vertices = np.flatnonzero(canonical == np.arange(n))
            vertex_radius = np.maximum(np.nan_to_num(graph.vertex_diameter(nodes, canonical) / 2.0, nan=0.0), 0.0)
            self.polyline = graph.polyline_of_column(nodes)
            self.arc = _column_arcs(nodes, self.polyline)
        else:
            vertices = np.zeros(0, dtype=np.int64)
            vertex_radius = np.zeros(0)
            self.polyline = np.zeros(0, dtype=np.int64)
            self.arc = np.zeros(0)
        self.own_radius = np.maximum(np.nan_to_num(nodes[3] / 2.0, nan=0.0), 0.0)
        self.feeders = []
        radii = vertex_radius[vertices]
        for members in _radius_classes(radii):
            columns = vertices[members]
            cell = 2.0 * float(radii[members].max()) + self.margin
            index = make_index(self.kind, cell_size=max(cell, 1e-9), rebuild_every=0)
            index.add(nodes[:3, columns].T, radii[members], columns)
            self.feeders.append(index)
        self.bed = None
        self._bed_chain = np.zeros(0, dtype=np.int64)
        self._bed_arc = np.zeros(0)
        self.chords = None
        self.chord_chain = np.zeros(0, dtype=np.int64)
        self.chord_arc = np.zeros(0)
        self.anchors = {}
        self.retired = np.zeros(0, dtype=bool)      # chords masked
        self.removed = np.zeros(0, dtype=bool)      # placed polylines masked

    @property
    def bed_chain(self):
        return self._bed_chain[:0 if self.bed is None else len(self.bed)]

    @property
    def bed_arc(self):
        return self._bed_arc[:0 if self.bed is None else len(self.bed)]

    def _flag(self, name, chain):
        mask = getattr(self, name)
        if chain >= mask.size:
            grown = np.zeros(max(chain + 1, 2 * mask.size), dtype=bool)
            grown[:mask.size] = mask
            mask = grown
            setattr(self, name, mask)
        mask[chain] = True

    def set_chords(self, polylines, anchors):
        """
        Indexes the chord polyline of every chain not placed yet, polylines[c]
        for chain c, with its anchors {bed vertex: position in the polyline}.
        """
        points, chain, arc = [], [], []
        for c, path in enumerate(polylines):
            if path is None or len(path) == 0:
                continue
            arcs = _polyline_arcs(path)
            points.append(path)
            chain.append(np.full(len(path), c, dtype=np.int64))
            arc.append(arcs)
            self.anchors[c] = {int(v): float(arcs[p]) for v, p in anchors[c].items()}
        if not points:
            return
        points = np.concatenate(points)
        self.chord_chain = np.concatenate(chain)
        self.chord_arc = np.concatenate(arc)
        cell = max(2.0 * self.radius + self.margin, 1e-9)
        self.chords = make_index(self.kind, cell_size=cell, rebuild_every=0)
        self.chords.add(points, self.radius, np.arange(len(points)))

    def place(self, chain, points, anchors):
        """
        Adds a placed chain's polyline, with its anchors {bed vertex:
        position}; the chain's chord stops being an obstacle.
        """
        points = np.asarray(points, dtype=float)
        arcs = _polyline_arcs(points)
        if self.bed is None:
            self.bed = make_index("grid", cell_size=max(2.0 * self.radius + self.margin, 1e-9))
        first = len(self.bed)
        needed = first + len(points)
        if needed > self._bed_chain.size:
            capacity = max(needed, 2 * self._bed_chain.size, 1024)
            chain_of = np.empty(capacity, dtype=np.int64)
            arc_of = np.empty(capacity)
            chain_of[:first] = self._bed_chain[:first]
            arc_of[:first] = self._bed_arc[:first]
            self._bed_chain, self._bed_arc = chain_of, arc_of
        self._bed_chain[first:needed] = chain
        self._bed_arc[first:needed] = arcs
        self.bed.add(points, self.radius, np.arange(first, needed))
        self.anchors[chain] = {int(v): float(arcs[p]) for v, p in anchors.items()}
        self._flag("retired", chain)

    def retire(self, chain, placed=False):
        """Masks a chain's chord, and its placed polyline when `placed`."""
        self._flag("retired", chain)
        if placed:
            self._flag("removed", chain)


def _flagged(mask, chains):
    """mask[chains], False beyond the mask and for negative chains."""
    out = np.zeros(chains.size, dtype=bool)
    inside = (chains >= 0) & (chains < mask.size)
    out[inside] = mask[chains[inside]]
    return out


def _excused_on_feeders(obstacles, qi, columns, arcs, anchors):
    """The pairs (point qi, feeder column) that the kin rule excuses at a tip the polyline shares."""
    excused = np.zeros(qi.size, dtype=bool)
    for vertex, at in anchors.items():
        tip = obstacles.tip_columns.get(vertex)
        if tip is None:
            continue
        kin = obstacles.polyline[columns] == obstacles.polyline[tip]
        if not kin.any():
            continue
        k = np.flatnonzero(kin)
        together = np.abs(obstacles.arc[columns[k]] - obstacles.arc[tip]) + np.abs(arcs[qi[k]] - at)
        reach = KIN_REACH * (obstacles.own_radius[columns[k]] + obstacles.radius + obstacles.margin)
        excused[k[together < reach]] = True
    return excused


def _excused_on_bed(obstacles, qi, chains, other_arcs, arcs, anchors):
    """The pairs (point qi, bed point of `chains`) that the kin rule excuses at a vertex both polylines share."""
    excused = np.zeros(qi.size, dtype=bool)
    reach = KIN_REACH * (obstacles.radius + obstacles.radius + obstacles.margin)
    for vertex, at in anchors.items():
        for chain in np.unique(chains).tolist():
            there = obstacles.anchors.get(chain, {}).get(vertex)
            if there is None:
                continue
            k = np.flatnonzero(chains == chain)
            together = np.abs(other_arcs[k] - there) + np.abs(arcs[qi[k]] - at)
            excused[k[together < reach]] = True
    return excused


def _collides(obstacles, points, chain, anchors, closed, self_check=True):
    """
    Whether a bed polyline collides with the feeders, the placed chains, the
    chords of the chains not placed yet, or itself, under describe's rules.

    Args:
        obstacles (_Obstacles): what it is checked against.
        points (ndarray): (m, 3) the polyline as it would be written.
        chain (int): its chain, never an obstacle to itself.
        anchors (dict): bed vertex -> position (point index) of the vertices
            it may share with other polylines.
        closed (bool): the polyline returns to its first point; its own
            separations are measured the shorter way round.
        self_check (bool): False skips the polyline against itself, which a
            straight chord cannot fail.
    """
    points = np.asarray(points, dtype=float)
    radius, margin = obstacles.radius, obstacles.margin
    arcs = _polyline_arcs(points)
    at = {vertex: float(arcs[position]) for vertex, position in anchors.items()}
    widen = margin * (1.0 + _WIDEN) + _WIDEN * radius

    for index in obstacles.feeders:
        qi, si = index.query(points, radius, margin=widen)
        if qi.size == 0:
            continue
        columns = index.tags[si]
        bad = _dist(points[qi], obstacles.nodes[:3, columns].T) - index.radii[si] - radius < margin
        if bad.any():
            qi, columns = qi[bad], columns[bad]
            if not _excused_on_feeders(obstacles, qi, columns, arcs, at).all():
                return True

    for index, chain_of, arc_of, masked in ((obstacles.bed, obstacles.bed_chain, obstacles.bed_arc,
                                             obstacles.removed),
                                            (obstacles.chords, obstacles.chord_chain, obstacles.chord_arc,
                                             obstacles.retired)):
        if index is None or len(index) == 0:
            continue
        qi, si = index.query(points, radius, margin=widen)
        if qi.size == 0:
            continue
        stored = index.tags[si]
        chains = chain_of[stored]
        keep = (chains != chain) & ~_flagged(masked, chains)
        if not keep.any():
            continue
        qi, stored, chains = qi[keep], stored[keep], chains[keep]
        bad = _dist(points[qi], index.points[stored]) - radius - radius < margin
        if not bad.any():
            continue
        qi, stored, chains = qi[bad], stored[bad], chains[bad]
        if not _excused_on_bed(obstacles, qi, chains, arc_of[stored], arcs, at).all():
            return True

    if self_check:
        return _self_collides(points, arcs, closed, radius, margin)
    return False


def _self_collides(points, arcs, closed, radius, margin):
    """Two points of one polyline closer than the margin while further apart along it than 2 (2 r + margin)."""
    n = len(points)
    if n < 3:
        return False
    window = 2.0 * (radius + radius + margin)
    total = float(arcs[-1])
    rows = max(1, _PAIR_BLOCK // n)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        separation = np.abs(arcs[start:stop, None] - arcs[None, :])
        if closed:
            separation = np.minimum(separation, total - separation)
        a, b = np.nonzero(separation > window)
        if a.size == 0:
            continue
        a = a + start
        if np.any(_dist(points[a], points[b]) - radius - radius < margin):
            return True
    return False


# --------------------------------------------------------------------------
# Candidates and the greedy pass


def _candidates(seeds_c, tips_c, reach_h, index_kind="grid"):
    """
    The candidate edges: every pair of seeds, and every pair of a seed and a
    tip, strictly closer than reach_h in the compressed coordinates given.

    Returns:
        tuple: (i, j, chord, n_tip) with i < j int64 in lexicographic order,
        a tip t numbered len(seeds_c) + t, chord the compressed length and
        n_tip the tip pairs among them.
    """
    seeds_c = np.asarray(seeds_c, dtype=float).reshape(-1, 3)
    tips_c = np.asarray(tips_c, dtype=float).reshape(-1, 3)
    n = len(seeds_c)
    empty = np.zeros(0, dtype=np.int64)
    if n == 0:
        return empty, empty, np.zeros(0), 0
    half = reach_h / 2.0
    index = make_index(index_kind, cell_size=max(half, 1e-9), rebuild_every=0)
    index.add(seeds_c, half, np.arange(n))
    qi, si = index.query(seeds_c, half, margin=_WIDEN * reach_h)
    i, j = index.tags[si], qi
    keep = i < j
    i, j = i[keep], j[keep]
    chord = _dist(seeds_c[i], seeds_c[j])
    keep = chord < reach_h
    i, j, chord = i[keep], j[keep], chord[keep]
    n_tip = 0
    if len(tips_c):
        qi, si = index.query(tips_c, half, margin=_WIDEN * reach_h)
        s = index.tags[si]
        tip_chord = _dist(seeds_c[s], tips_c[qi])
        keep = tip_chord < reach_h
        n_tip = int(np.count_nonzero(keep))
        i = np.concatenate([i, s[keep]])
        j = np.concatenate([j, n + qi[keep]])
        chord = np.concatenate([chord, tip_chord[keep]])
    order = np.lexsort((j, i))
    return i[order].astype(np.int64), j[order].astype(np.int64), chord[order], n_tip


def _within(adj, i, j, depth):
    """
    Whether vertices i and j are at most `depth` edges apart in the graph
    whose neighbours of v are adj[v]: a breadth-first search of
    ceil(depth / 2) levels from each end, which meet at a vertex v when
    dist(i, v) + dist(j, v) <= depth.
    """
    if i == j:
        return True
    if depth <= 0:
        return False
    half = (depth + 1) // 2
    from_i = {i: 0}
    frontier = [i]
    for level in range(1, half + 1):
        following = []
        for v in frontier:
            for w in adj[v]:
                if w not in from_i:
                    from_i[w] = level
                    following.append(w)
        frontier = following
        if not frontier:
            break
    if j in from_i:
        return True
    from_j = {j: 0}
    frontier = [j]
    for level in range(1, half + 1):
        following = []
        for v in frontier:
            for w in adj[v]:
                if w not in from_j:
                    from_j[w] = level
                    if w in from_i and from_i[w] + level <= depth:
                        return True
                    following.append(w)
        frontier = following
        if not frontier:
            break
    return False


def _segment_distance(p0, p1, q0, q1):
    """
    The distance between the segments [p0, p1] and [q0, q1], elementwise over
    the leading axes: the clamped closed form of the closest points, every
    dot product through _dot3.
    """
    p0, p1, q0, q1 = (np.asarray(v, dtype=float) for v in (p0, p1, q0, q1))
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = _dot3(d1, d1)
    e = _dot3(d2, d2)
    f = _dot3(d2, r)
    c = _dot3(d1, r)
    b = _dot3(d1, d2)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 0.0, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
        t = np.clip(t, 0.0, 1.0)
        # a point for a segment: its one position
        point_p = ~(a > 0.0)
        point_q = ~(e > 0.0)
        s = np.where(point_p, 0.0, np.where(point_q, np.clip(-c / a, 0.0, 1.0), s))
        t = np.where(point_q, 0.0, np.where(point_p, np.clip(f / e, 0.0, 1.0), t))
    gap = (p0 + d1 * s[..., None]) - (q0 + d2 * t[..., None])
    return np.sqrt(_dot3(gap, gap))


class _ChordGrid:
    """The accepted chords by the cell of their midpoint, for the greedy pass's distance test; grows."""

    def __init__(self, cell):
        self.cell = float(cell)
        self.cells = {}
        self.half_max = 0.0

    def add(self, k, mid, half):
        key = tuple(int(math.floor(v / self.cell)) for v in mid)
        self.cells.setdefault(key, []).append(k)
        self.half_max = max(self.half_max, half)

    def near(self, mid, half, reach):
        """Every chord whose midpoint lies within half + the longest stored half-length + reach of mid."""
        if not self.cells:
            return []
        extent = (half + self.half_max + reach) * (1.0 + _WIDEN)
        cell = self.cell
        lo = [int(math.floor((v - extent) / cell)) for v in mid]
        hi = [int(math.floor((v + extent) / cell)) for v in mid]
        found = []
        cells = self.cells
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    members = cells.get((x, y, z))
                    if members:
                        found.extend(members)
        return found


def _chord_samples(points, a, b, step):
    """The stored chord of edge (a, b), a < b: bridge_path(x_a, x_b, step) with no tangents and no generator."""
    return bridge_path(points[a], points[b], step)


def _greedy(points, i, j, order, *, caps, inward, adjacency, girth, min_angle, clearance, step,
            obstacles=None, near=None, samples=None):
    """
    The greedy pass over the candidates in `order`: each is refused at the
    first rule it fails, degree, angle, girth, clearance, and accepted
    otherwise (see the module docstring).

    Args:
        points (ndarray): (V, 3) true coordinates of the bed vertices, seeds then tips.
        i, j (ndarray): the candidate pairs, i < j.
        order (ndarray): the candidates' processing order.
        caps (sequence): most bed edges per vertex.
        inward (dict): tip vertex -> the feeder's unit inward direction at it.
        adjacency (list): neighbour lists of the girth graph (the feeder
            segment graph, bed vertices first); accepted edges are appended.
        girth (int): shortest cycle allowed, in edges.
        min_angle (float): smallest angle between two edges at a vertex, degrees.
        clearance (float): T, the closest two bed chords may come.
        step (float): spacing of the chord samples.
        obstacles (_Obstacles or None): the feeders, for the pointwise test.
        near (ndarray or None): (n_pairs,) whether a candidate's chord may
            come near a feeder; only those are sampled and tested.
        samples (dict or None): edge (a, b) -> chord samples, filled in.

    Returns:
        dict: "edges" (E, 2) int64 in acceptance order, "counts" of the
        refusals by rule ("degree", "angle", "girth", "clearance", "at_end").
    """
    points = np.asarray(points, dtype=float)
    if samples is None:
        samples = {}
    cos_min = math.cos(math.radians(min_angle))
    cos_kin = math.cos(math.radians(KIN_ANGLE_DEG))
    depth = girth - 2
    degree = [0] * len(points)
    caps = [int(c) for c in caps]
    directions = [[] for _ in range(len(points))]
    feeder_direction = {int(v): [float(x) for x in u] for v, u in inward.items() if u is not None}
    counts = {"degree": 0, "angle": 0, "girth": 0, "clearance": 0, "at_end": 0}
    accepted = []
    grid = None
    order = np.asarray(order, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if order.size == 0:
        return {"edges": np.zeros((0, 2), dtype=np.int64), "counts": counts}
    unit, length = _unit_rows(points[j] - points[i])
    I = i.tolist()
    J = j.tolist()
    chord_ends = np.zeros((0, 2, 3))
    capacity = 0

    for k in order.tolist():
        a, b = I[k], J[k]
        if degree[a] >= caps[a] or degree[b] >= caps[b]:
            counts["degree"] += 1
            continue
        ex, ey, ez = unit[k].tolist()
        refused = False
        # angle: e leaves a; -e leaves b
        for ux, uy, uz in directions[a]:
            if ux * ex + uy * ey + uz * ez > cos_min:
                refused = True
                break
        if not refused and a in feeder_direction:
            ux, uy, uz = feeder_direction[a]
            refused = ux * ex + uy * ey + uz * ez > cos_min
        if not refused:
            fx, fy, fz = -ex, -ey, -ez
            for ux, uy, uz in directions[b]:
                if ux * fx + uy * fy + uz * fz > cos_min:
                    refused = True
                    break
            if not refused and b in feeder_direction:
                ux, uy, uz = feeder_direction[b]
                refused = ux * fx + uy * fy + uz * fz > cos_min
        if refused:
            counts["angle"] += 1
            continue
        if _within(adjacency, a, b, depth):
            counts["girth"] += 1
            continue
        # clearance at a shared end, between bed chords only
        at_end = False
        for ux, uy, uz in directions[a]:
            if ux * ex + uy * ey + uz * ez > cos_kin:
                at_end = True
                break
        if not at_end:
            fx, fy, fz = -ex, -ey, -ez
            for ux, uy, uz in directions[b]:
                if ux * fx + uy * fy + uz * fz > cos_kin:
                    at_end = True
                    break
        if at_end:
            counts["clearance"] += 1
            counts["at_end"] += 1
            continue
        # clearance from the accepted chords that share no end
        pa, pb = points[a], points[b]
        mid = ((pa + pb) / 2.0).tolist()
        half = float(length[k]) / 2.0
        if grid is None:
            grid = _ChordGrid(max(float(length.max()) / 2.0 + clearance, 1e-9))
        found = grid.near(mid, half, clearance)
        if found:
            found = [m for m in found if a not in accepted[m] and b not in accepted[m]]
        if found:
            other = chord_ends[np.array(found, dtype=np.int64)]
            if np.any(_segment_distance(other[:, 0], other[:, 1], pa, pb) < clearance):
                counts["clearance"] += 1
                continue
        # clearance from the feeders, pointwise on the stored chord samples
        if obstacles is not None and (near is None or near[k]):
            chord = samples.get((a, b))
            if chord is None:
                chord = samples[(a, b)] = _chord_samples(points, a, b, step)
            anchors = {a: 0, b: len(chord) - 1}
            if _collides(obstacles, chord, -1, anchors, False, self_check=False):
                counts["clearance"] += 1
                continue

        # accepted
        degree[a] += 1
        degree[b] += 1
        directions[a].append((ex, ey, ez))
        directions[b].append((-ex, -ey, -ez))
        adjacency[a].append(b)
        adjacency[b].append(a)
        e = len(accepted)
        accepted.append((a, b))
        if e >= capacity:
            capacity = max(1024, 2 * capacity)
            grown = np.empty((capacity, 2, 3))
            grown[:e] = chord_ends[:e]
            chord_ends = grown
        chord_ends[e, 0] = pa
        chord_ends[e, 1] = pb
        grid.add(e, mid, half)

    edges = np.array(accepted, dtype=np.int64).reshape(-1, 2)
    return {"edges": edges, "counts": counts}


# --------------------------------------------------------------------------
# Pruning and chains


def _prune(n_vertices, edges, alive, is_tip, kept, islands):
    """
    Peels every kept seed with at most one alive edge, until none is left
    (tips are never peeled), then, when `islands`, removes every component
    of the alive edges that holds no tip. `alive` and `kept` are updated in
    place.

    Args:
        n_vertices (int): bed vertices, seeds then tips.
        edges (ndarray): (E, 2) the bed edges.
        alive (ndarray): (E,) bool, the edges still kept.
        is_tip (ndarray): (n_vertices,) bool.
        kept (ndarray): (n_vertices,) bool, the vertices still kept (tips
            stay True).
        islands (bool): whether islands are removed (with feeders).

    Returns:
        dict: "edges_dead_end", "seeds_dead_end", "edges_island",
        "seeds_island", the edges and seeds this call removed.
    """
    counts = {"edges_dead_end": 0, "seeds_dead_end": 0, "edges_island": 0, "seeds_island": 0}
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    incident = [[] for _ in range(n_vertices)]
    degree = [0] * n_vertices
    for e in np.flatnonzero(alive).tolist():
        a, b = int(edges[e, 0]), int(edges[e, 1])
        incident[a].append(e)
        incident[b].append(e)
        degree[a] += 1
        degree[b] += 1
    tip = is_tip.tolist() if isinstance(is_tip, np.ndarray) else list(is_tip)
    queue = [v for v in np.flatnonzero(kept).tolist() if not tip[v] and degree[v] <= 1]
    queued = set(queue)
    alive_list = alive.tolist()
    while queue:
        v = queue.pop()
        kept[v] = False
        counts["seeds_dead_end"] += 1
        for e in incident[v]:
            if not alive_list[e]:
                continue
            alive_list[e] = False
            counts["edges_dead_end"] += 1
            a, b = int(edges[e, 0]), int(edges[e, 1])
            w = b if a == v else a
            degree[w] -= 1
            degree[v] -= 1
            if not tip[w] and kept[w] and degree[w] <= 1 and w not in queued:
                queued.add(w)
                queue.append(w)
    alive[:] = alive_list

    if islands:
        live = np.flatnonzero(alive)
        if live.size:
            labels, _ = graph.components(edges[live].T, np.arange(n_vertices))
            with_tip = np.zeros(int(labels.max()) + 1, dtype=bool)
            tips = np.flatnonzero(is_tip)
            with_tip[labels[tips]] = True
            lost = ~with_tip[labels[edges[live, 0]]]
            if lost.any():
                gone = live[lost]
                alive[gone] = False
                counts["edges_island"] += int(gone.size)
                seeds = np.unique(edges[gone].ravel())
                seeds = seeds[~is_tip[seeds] & kept[seeds]]
                kept[seeds] = False
                counts["seeds_island"] += int(seeds.size)
    return counts


def _chains(n_vertices, edges, is_tip):
    """
    The chains of the bed graph: graph.segments over the edges plus one
    pendant edge from every tip with an edge to a phantom vertex, the
    phantoms and the [tip, phantom] paths stripped. Every tip ends its
    chains; a closed chain [J, ..., J] (a ring, or a lasso through a junction
    or a tip J) is returned closed.

    Returns:
        list: lists of bed vertex ids, in graph.segments order.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return []
    degree = np.bincount(edges.ravel(), minlength=n_vertices)
    tips = np.flatnonzero(np.asarray(is_tip, dtype=bool) & (degree[:n_vertices] > 0))
    phantoms = n_vertices + np.arange(tips.size, dtype=np.int64)
    pendant = np.stack([tips, phantoms], axis=1)
    plus = np.concatenate([edges, pendant]).T
    paths = graph.segments(np.empty((3, n_vertices + tips.size)), plus)
    chains = []
    for path in paths:
        path = path.tolist()
        if path[0] >= n_vertices:
            path = path[1:]
        if path[-1] >= n_vertices:
            path = path[:-1]
        if len(path) >= 2:
            chains.append(path)
    return chains


def _middle(chain):
    """Where a lasso is written from: its middle vertex position."""
    return len(chain) // 2


def _rotate(path, joints, position):
    """A closed polyline written from its joint `position` round to it again; returns (path, joints)."""
    at = joints[position]
    rotated = np.concatenate([path[at:], path[1:at + 1]])
    last = len(path) - 1
    shifted = [joints[k] - at for k in range(position, len(joints))]
    shifted += [last - at + joints[k] for k in range(1, position + 1)]
    return rotated, shifted


def _joined(pieces):
    """Pieces joined end to end, each joint once; returns (path, joint positions)."""
    joints = [0]
    for piece in pieces:
        joints.append(joints[-1] + len(piece) - 1)
    path = np.concatenate([pieces[0]] + [piece[1:] for piece in pieces[1:]])
    return path, joints


def _assemble(nodes, tree, polylines, diameter):
    """
    The feeders' columns unchanged, then each polyline after one NaN
    separator, rows[:3] = path.T and rows[3] = diameter, with the stored
    coordinates written at its joints; labels BRIDGE on bed columns and -1
    on separators.

    Args:
        nodes (ndarray): (4, N) the feeders; (4, 0) for none.
        tree (ndarray): (N,) their labels.
        polylines (list): (path (m, 3), joints (positions), coordinates
            (len(joints), 3)) per polyline.
        diameter (float): d.

    Returns:
        tuple: (nodes, tree, first, last) with first and last the column of
        each polyline's first and last point.
    """
    blocks = [np.asarray(nodes, dtype=float).reshape(4, -1)]
    labels = [np.asarray(tree).astype(np.int8)]
    column = blocks[0].shape[1]
    first, last = [], []
    separator = np.full((4, 1), np.nan)
    for path, joints, coordinates in polylines:
        rows = np.empty((4, len(path)))
        rows[:3] = path.T
        rows[3] = diameter
        for position, xyz in zip(joints, coordinates):
            rows[:3, position] = xyz          # bitwise copies, so the graph joins the polyline
        blocks.extend([separator, rows])
        labels.extend([np.array([-1], dtype=np.int8), np.full(len(path), BRIDGE, dtype=np.int8)])
        first.append(column + 1)
        last.append(column + len(path))
        column += 1 + len(path)
    return np.concatenate(blocks, axis=1), np.concatenate(labels), first, last


# --------------------------------------------------------------------------
# The frame


def _grow_axis(frame):
    """frame["grow_direction"] as a unit (3,) array, kept as it is when already unit."""
    if not isinstance(frame, dict) or "grow_direction" not in frame:
        raise ValueError("a bed with stretch above 1 needs the network's frame with its grow_direction")
    value = frame["grow_direction"]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_is_number(v) for v in value):
        raise ValueError(f"the frame's grow_direction must be three finite numbers, got {value!r}")
    axis = np.array([float(v) for v in value])
    norm = float(_dist(axis[None, :], np.zeros((1, 3)))[0])
    if not norm > 0.0:
        raise ValueError("the frame's grow_direction must not be a zero vector")
    if abs(norm - 1.0) > 1e-12:
        axis = axis / norm
    return axis


def frame_record(frame, settings):
    """
    The frame record of a network with this bed: the input unchanged at
    stretch 1; at stretch above 1 kind "axis", axis the unit axis the seed
    metric is compressed along (the frame's grow_direction, normalised
    unless already unit), sense "nematic", tree_senses and normal None,
    rules [], the other fields kept.
    """
    parsed = BedSettings.parse(settings)
    if parsed.stretch == 1.0:
        return frame
    axis = _grow_axis(frame)
    record = copy.deepcopy(frame)
    record.update({"kind": "axis", "axis": [float(v) for v in axis], "sense": "nematic",
                   "tree_senses": None, "normal": None, "rules": []})
    return record


# --------------------------------------------------------------------------
# The bed


def _feeders(feeders):
    """
    The feeders as (nodes (4, N) float, tree (N,) int8), refused before
    anything is drawn when malformed: a shape that does not match, a label
    that is not an integer or does not fit the archive's int8, a finite
    column unlabelled or a separator labelled, or a finite column whose
    diameter is not a finite number above 0.
    """
    if not isinstance(feeders, (tuple, list)) or len(feeders) != 2:
        raise ValueError("feeders must be (nodes (4, N), tree (N,)) or None")
    nodes = np.asarray(feeders[0], dtype=float)
    tree = np.asarray(feeders[1])
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError(f"the feeder nodes must be a (4, N) array, got shape {nodes.shape}")
    if tree.shape != (nodes.shape[1],) or (tree.size and not np.issubdtype(tree.dtype, np.integer)):
        raise ValueError("the feeder tree must hold one integer label per column of nodes")
    labels = tree.astype(np.int8)
    if np.any(labels != tree):
        raise ValueError("the feeder tree's labels must fit an int8, as the archive's do")
    finite = np.isfinite(nodes[:3]).all(axis=0)
    if np.any(finite & (tree < 0)) or np.any(~finite & (tree >= 0)):
        raise ValueError("the feeder tree must label every finite column and only those")
    diameter = nodes[3, finite]
    if not np.all(np.isfinite(diameter) & (diameter > 0.0)):
        raise ValueError("every finite feeder column needs a finite diameter above 0")
    return nodes, labels


def build(domain, settings, *, d_min, seed, collision_margin, feeders=None, frame=None,
          attempts=10, index_kind="grid", events=None, timings=None):
    """
    Grows an explicit capillary bed in a box and attaches it to the feeders' tips.

    Args:
        domain (tuple): (low, high), finite 3-vectors, high > low + d componentwise.
        settings (dict or BedSettings): parsed with BedSettings.parse.
        d_min (float): the unit of the settings, > 0.
        seed (int): the run's seed; the sub-streams are [seed, RNG_TAG, k].
        collision_margin (float): finite and >= 0; the clearance rule is one
            of the bed's four, so None is refused.
        feeders (tuple or None): (nodes (4, N), tree (N,)), the two trees the
            bed joins; None grows a bed alone, with no tips and no island pass.
        frame (dict or None): the network's frame record; its grow_direction
            is the stretch axis, needed when stretch > 1.
        attempts (int): walk redraws per chain, >= 0.
        index_kind (str): "grid", "kdtree" or "auto" (the grid) for the
            indexes built once, but for the seeds' blocked set, which
            place_seeds always builds as a grid; every index that grows is
            a grid.
        events (dict or None): counters incremented in place (EVENT_KEYS).
        timings (dict or None): filled with the seconds of each stage
            (seeds, candidates, greedy, prune, walk, assemble).

    Returns:
        dict: "nodes" the feeders' columns unchanged, then one polyline per
        placed chain, each after one NaN separator; "tree" the feeders'
        labels, BRIDGE on bed columns, -1 on separators; "frame"
        (frame_record); "seeds" (N_t, 3) in acceptance order;
        "greedy_edges" (E_g, 2) int64 every edge the greedy pass accepted,
        in acceptance order; "kept" (N_t,) bool the seeds that survive;
        "tips" (T,) int64 the feeder columns of the eligible tips;
        "tip_edges" (T,) int64 their kept bed edges; "edges" (E, 2) int64
        the kept bed edges (vertex i < N_t is seed i, N_t + t is tip t);
        "chains" one dict per placed chain in chain order, "vertices" in the
        order written, "outcome" "walked" or "chord", "first" and "last"
        the columns of its first and last point, "closed".

    Raises:
        ValueError: before any generator is created, for anything the
        settings or check_settings refuse, a malformed argument, stretch > 1
        without a frame, malformed feeders, and a box that holds no seed.
        SeedsNotPlaced: when the proposals run out before N_t seeds stand.
    """
    parsed = BedSettings.parse(settings)
    d_min = _positive_d_min(d_min)
    if not _is_whole(seed) or seed < 0:
        raise ValueError(f"the bed needs a seed, a non-negative integer, got {seed!r}")
    margin = _margin(collision_margin)
    check_settings(parsed, d_min=d_min, collision_margin=margin)
    if not _is_whole(attempts) or attempts < 0:
        raise ValueError(f"the walk attempts must be an integer of at least 0, got {attempts!r}")
    if not isinstance(index_kind, str) or index_kind not in ("grid", "kdtree", "auto"):
        raise ValueError(f"the bed index kind must be 'grid', 'kdtree' or 'auto', got {index_kind!r}")
    if events is not None and not isinstance(events, dict):
        raise ValueError("events must be a dict or None")
    if timings is not None and not isinstance(timings, dict):
        raise ValueError("timings must be a dict or None")
    kind = "grid" if index_kind == "auto" else index_kind
    h = parsed.spacing * d_min
    d = parsed.diameter * d_min
    radius = d / 2.0
    T = d + margin
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise ValueError("the bed domain must be (low, high)")
    low = np.asarray(domain[0], dtype=float)
    high = np.asarray(domain[1], dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("the bed domain corners must be finite 3-vectors")
    if not np.all(high > low + d):
        raise ValueError(f"the bed domain must be wider than the bed diameter {d:g} along every axis")
    stretch = parsed.stretch
    axis = None
    if stretch != 1.0:
        if frame is None:
            raise ValueError("a bed with stretch above 1 needs the network's frame (its grow_direction is the axis)")
        axis = _grow_axis(frame)
    if feeders is not None:
        fnodes, ftree = _feeders(feeders)
    else:
        fnodes, ftree = np.empty((4, 0)), np.zeros(0, dtype=np.int8)
    n_seeds = planned_seeds(high - low, parsed, d_min)
    if n_seeds == 0:
        raise ValueError("the box holds no seed: it is too small for the bed's spacing and density")
    record = frame_record(frame, parsed)

    counts = {key: 0 for key in EVENT_KEYS}
    stages = {}
    clock = time.perf_counter()

    # the feeders: tips, their inward directions, the segment graph and the obstacles
    if feeders is not None and fnodes.shape[1]:
        built = graph.build(fnodes)
        tips, stubs = _feeder_tips(fnodes, ftree, margin, built)
        canonical = built["canonical"]
        vertex_diameter = graph.vertex_diameter(fnodes, canonical)
        polylines = graph.polylines(fnodes)
        polyline_of = graph.polyline_of_column(fnodes)
        finite = np.flatnonzero(canonical >= 0)
        r_f = vertex_diameter[canonical[finite]] / 2.0
        blocked = (fnodes[:3, finite].T, r_f + radius + margin)
        feeder_segments = graph.segments(fnodes, built["edges"])
    else:
        tips = stubs = np.zeros(0, dtype=np.int64)
        polylines, polyline_of = [], None
        blocked = None
        feeder_segments = []
    counts["bed_tips"] = int(tips.size + stubs.size)
    counts["bed_tips_inside_junction"] = int(stubs.size)
    n_tips = int(tips.size)
    tip_points = fnodes[:3, tips].T.copy() if n_tips else np.zeros((0, 3))
    tip_inward = [_inward(fnodes, polylines[polyline_of[column]], column) for column in tips.tolist()]

    # seeds
    inset = (low + radius, high - radius)
    seeds, outcomes = place_seeds(_stream(seed, SUBSTREAMS["seeds"]), inset[0], inset[1], spacing=h,
                                  count=n_seeds, stretch=stretch, axis=axis, blocked=blocked)
    n_seeds = len(seeds)
    n_vertices = n_seeds + n_tips
    inward = {n_seeds + t: direction for t, direction in enumerate(tip_inward)}
    counts["bed_proposals"] = int(outcomes.size)
    counts["bed_proposals_on_feeders"] = int(np.count_nonzero(outcomes == 1))
    counts["bed_proposals_overlapping"] = int(np.count_nonzero(outcomes == 2))
    counts["bed_seeds"] = int(np.count_nonzero(outcomes == 0))
    now = time.perf_counter()
    stages["seeds"], clock = now - clock, now

    # candidates and their order
    points = np.concatenate([seeds, tip_points])
    i, j, chord, n_tip = _candidates(_compress(seeds, axis, stretch), _compress(tip_points, axis, stretch),
                                     parsed.reach * h, kind)
    n_pairs = int(i.size)
    counts["bed_candidates"] = n_pairs
    counts["bed_candidates_tip"] = n_tip
    if parsed.jitter > 0.0 and n_pairs > 0:
        u = _stream(seed, SUBSTREAMS["order"]).random(n_pairs)
        key = chord * (1.0 + parsed.jitter * u)
    else:
        key = chord
    order = np.lexsort((j, i, key))
    now = time.perf_counter()
    stages["candidates"], clock = now - clock, now

    # the greedy pass
    tip_columns = {n_seeds + t: int(c) for t, c in enumerate(tips.tolist())}
    obstacles = _Obstacles(fnodes, radius, margin, kind, tip_columns)
    feeder_ids = {}
    for t, column in enumerate(tips.tolist()):
        feeder_ids[column] = n_seeds + t
    for path in feeder_segments:
        for end in (int(path[0]), int(path[-1])):
            if end not in feeder_ids:
                feeder_ids[end] = n_vertices + len(feeder_ids) - n_tips
    adjacency = [[] for _ in range(n_vertices + len(feeder_ids) - n_tips)]
    for path in feeder_segments:
        a, b = feeder_ids[int(path[0])], feeder_ids[int(path[-1])]
        if a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)
    caps = [3] * n_seeds + [parsed.tip_edges] * n_tips
    near = None
    if obstacles.feeders and n_pairs:
        near = _near_feeders(obstacles, points, i, j)
    samples = {}
    greedy = _greedy(points, i, j, order, caps=caps, inward=inward, adjacency=adjacency, girth=parsed.girth,
                     min_angle=parsed.min_angle, clearance=T, step=parsed.step * d,
                     obstacles=obstacles if obstacles.feeders else None, near=near, samples=samples)
    greedy_edges = greedy["edges"]
    refusals = greedy["counts"]
    counts["bed_refused_degree"] = refusals["degree"]
    counts["bed_refused_angle"] = refusals["angle"]
    counts["bed_refused_girth"] = refusals["girth"]
    counts["bed_refused_clearance"] = refusals["clearance"]
    counts["bed_refused_clearance_at_end"] = refusals["at_end"]
    counts["bed_edges"] = int(len(greedy_edges))
    now = time.perf_counter()
    stages["greedy"], clock = now - clock, now

    # pruning
    is_tip = np.zeros(n_vertices, dtype=bool)
    is_tip[n_seeds:] = True
    kept = np.ones(n_vertices, dtype=bool)
    alive = np.ones(len(greedy_edges), dtype=bool)
    removed = _prune(n_vertices, greedy_edges, alive, is_tip, kept, feeders is not None)
    _count_removed(counts, removed)
    now = time.perf_counter()
    stages["prune"], clock = now - clock, now

    # chains, walked in order
    edge_of = {(int(a), int(b)): e for e, (a, b) in enumerate(greedy_edges.tolist())}
    chains = _chains(n_vertices, greedy_edges[alive], is_tip)
    bed_degree = np.bincount(greedy_edges[alive].ravel(), minlength=n_vertices)
    step = parsed.step * d
    chord_paths, writes = [], []
    for chain in chains:
        pieces = []
        for a, b in zip(chain[:-1], chain[1:]):
            lo, hi = (a, b) if a < b else (b, a)
            piece = samples.get((lo, hi))
            if piece is None:
                piece = samples[(lo, hi)] = _chord_samples(points, lo, hi, step)
            pieces.append(piece if a < b else piece[::-1])
        path, joints = _joined(pieces)
        written = list(chain)
        lasso = chain[0] == chain[-1] and bool(is_tip[chain[0]] or bed_degree[chain[0]] != 2)
        if lasso:
            position = _middle(chain)
            path, joints = _rotate(path, joints, position)
            written = chain[position:] + chain[1:position + 1]
        chord_paths.append(path)
        writes.append((written, joints, lasso))
    del samples                      # the chains now hold every chord they replay
    anchors = [_anchors(written, joints, lasso) for written, joints, lasso in writes]
    obstacles.set_chords(chord_paths, anchors)
    retired = [False] * len(chains)
    placed = {}
    rng = None
    persistence = parsed.persistence
    for c, chain in enumerate(chains):
        if retired[c]:
            continue
        counts["bed_segments"] += 1
        written, joints, lasso = writes[c]
        closed = chain[0] == chain[-1]
        if rng is None:
            rng = _stream(seed, SUBSTREAMS["walks"])
        tangents = _tangents(points, chain)
        outcome = None
        for attempt in range(1 + attempts):
            if attempt:
                counts["bed_walk_redraws"] += 1
            pieces = [bridge_path(points[a], points[b], step, start_tangent=tangents[k], rng=rng,
                                  persistence=persistence, diameter=d, end_tangent=tangents[k + 1])
                      for k, (a, b) in enumerate(zip(chain[:-1], chain[1:]))]
            path, walk_joints = _joined(pieces)
            if lasso:
                path, walk_joints = _rotate(path, walk_joints, _middle(chain))
            path[walk_joints] = points[written]
            if not _collides(obstacles, path, c, _anchors(written, walk_joints, lasso), closed):
                outcome = ("walked", path, walk_joints)
                break
        if outcome is None:
            path = chord_paths[c]
            if not _collides(obstacles, path, c, anchors[c], closed):
                outcome = ("chord", path, joints)
        if outcome is not None:
            name, path, at = outcome
            counts["bed_segments_" + name] += 1
            obstacles.place(c, path, _anchors(written, at, lasso))
            placed[c] = (name, path, at)
            retired[c] = True
            continue
        # dropped: its edges go, and the graph is peeled and pruned again
        counts["bed_segments_dropped"] += 1
        retired[c] = True
        obstacles.retire(c)
        for a, b in zip(chain[:-1], chain[1:]):
            e = edge_of[(a, b) if a < b else (b, a)]
            if alive[e]:
                alive[e] = False
                counts["bed_edges_dropped"] += 1
        removed = _prune(n_vertices, greedy_edges, alive, is_tip, kept, feeders is not None)
        _count_removed(counts, removed)
        for other, vertices in enumerate(chains):
            if other == c:
                continue
            gone = any(not alive[edge_of[(a, b) if a < b else (b, a)]] for a, b in zip(vertices[:-1], vertices[1:]))
            if not gone:
                continue
            if other in placed:
                del placed[other]
                counts["bed_segments_removed"] += 1
                obstacles.retire(other, placed=True)
            elif not retired[other]:
                obstacles.retire(other)
            retired[other] = True
    now = time.perf_counter()
    stages["walk"], clock = now - clock, now

    # assembly
    order_placed = sorted(placed)
    pieces = []
    for c in order_placed:
        name, path, at = placed[c]
        pieces.append((path, at, points[writes[c][0]]))
    nodes, tree, first, last = _assemble(fnodes, ftree, pieces, d)
    records = []
    for c, column_first, column_last in zip(order_placed, first, last):
        name, path, at = placed[c]
        records.append({"vertices": [int(v) for v in writes[c][0]], "outcome": name,
                        "first": int(column_first), "last": int(column_last),
                        "closed": bool(chains[c][0] == chains[c][-1])})
    edges = greedy_edges[alive]
    tip_edges = np.bincount(edges.ravel(), minlength=n_vertices)[n_seeds:].astype(np.int64) \
        if n_tips else np.zeros(0, dtype=np.int64)
    counts["bed_tips_attached"] = int(np.count_nonzero(tip_edges))
    counts["bed_tip_edges"] = int(tip_edges.sum())
    components, one_tree = _components(n_vertices, edges, n_seeds, ftree[tips] if n_tips else None,
                                       feeders is not None)
    counts["bed_components"] = components
    counts["bed_components_one_tree"] = one_tree
    now = time.perf_counter()
    stages["assemble"] = now - clock

    if events is not None:
        for key in EVENT_KEYS:
            events[key] = events.get(key, 0) + counts[key]
    if timings is not None:
        timings.update(stages)
    return {"nodes": nodes, "tree": tree, "frame": record, "seeds": seeds, "greedy_edges": greedy_edges,
            "kept": kept[:n_seeds].copy(), "tips": tips.astype(np.int64), "tip_edges": tip_edges,
            "edges": edges, "chains": records}


def _count_removed(counts, removed):
    counts["bed_edges_dead_end"] += removed["edges_dead_end"]
    counts["bed_seeds_dead_end"] += removed["seeds_dead_end"]
    counts["bed_edges_island"] += removed["edges_island"]
    counts["bed_seeds_island"] += removed["seeds_island"]


def _anchors(written, joints, lasso):
    """
    The vertices a written polyline may share with other polylines, at their
    positions in it: its two ends, a lasso's J, nothing for a ring.
    """
    if lasso:
        # [J, v_1, ..., v_k, J] written from its middle vertex m is
        # [v_m, ..., v_k, J, v_1, ..., v_m]: J sits at k + 1 - m, and the seam
        # v_m is shared with nothing
        at = len(written) - 1 - _middle(written)
        return {written[at]: joints[at]}
    if written[0] == written[-1]:
        return {}
    return {written[0]: joints[0], written[-1]: joints[-1]}


def _tangents(points, chain):
    """None at the chain's ends; the Catmull-Rom tangent unit(x_{i+1} - x_{i-1}) at its interior seeds."""
    tangents = [None] * len(chain)
    for k in range(1, len(chain) - 1):
        delta = points[chain[k + 1]] - points[chain[k - 1]]
        length = float(_dist(delta[None, :], np.zeros((1, 3)))[0])
        if length > 0.0:
            tangents[k] = delta / length
    return tangents


def _near_feeders(obstacles, points, i, j):
    """
    Whether each candidate's chord may come near a feeder column. A chord of
    half-length H cannot reach a feeder column's collision zone when both its
    ends stand clear of every one by H or more, since each of its points lies
    within H of an end; only the other chords are sampled and tested. The
    safety factor only widens the set sampled.
    """
    bound = _dist(points[j], points[i]) / 2.0 * (1.0 + _NEAR_SAFETY)
    reach = float(bound.max()) + obstacles.radius
    clear = np.full(len(points), np.inf)
    for index in obstacles.feeders:
        r_class = float(index.radii.max())
        coarse = make_index(obstacles.kind, cell_size=max(reach + r_class + obstacles.margin, 1e-9), rebuild_every=0)
        coarse.add(index.points, index.radii, np.arange(len(index)))
        qi, si = coarse.query(points, reach, margin=obstacles.margin)
        if qi.size:
            gap = _dist(points[qi], coarse.points[si]) - coarse.radii[si] - obstacles.radius - obstacles.margin
            np.minimum.at(clear, qi, gap)
    return ~((clear[i] >= bound) & (clear[j] >= bound))


def _components(n_vertices, edges, n_seeds, tip_trees, with_feeders):
    """Kept components with an edge, and those whose tips all belong to one feeder tree."""
    if len(edges) == 0:
        return 0, 0
    labels, _ = graph.components(edges.T, np.arange(n_vertices))
    present = np.unique(labels[edges[:, 0]])
    if not with_feeders or tip_trees is None:
        return int(present.size), 0
    trees = {}
    for t, label in enumerate(tip_trees.tolist()):
        trees.setdefault(int(labels[n_seeds + t]), set()).add(int(label))
    one = sum(1 for component in present.tolist() if len(trees.get(component, ())) == 1)
    return int(present.size), int(one)
