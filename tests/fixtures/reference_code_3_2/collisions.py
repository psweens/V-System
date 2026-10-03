"""
Collision avoidance for growing centrelines.

Every accepted centreline point is stored in a spatial index with its radius
and the id of the stem it belongs to. A proposed point Q of radius r_Q
collides with a stored point P of radius r_P when

    |P - Q| < r_P + r_Q + margin,

unless the pair is one that legitimately overlaps:

- P lies on Q's own stem within an arc-length window of 2 (r_P + r_Q + margin)
  behind Q. Consecutive points of one vessel are always within each other's
  radii; the window is long enough that a bend of radius of curvature above
  about 1.3 vessel radii passes, while a stem that curls back onto itself is
  still caught.
- P lies on the parent stem or on a sibling stem (one leaving the same
  junction J) and the two points are close to J together, distances being
  measured along the vessels:

      arc(P, J) + arc(Q, J) < KIN_REACH (r_P + r_Q + margin),

  where arc(P, J) is the arc length from P along its stem to J (the parent's
  remaining length after P, a sibling's or Q's own length before it). Arc
  length is never shorter than the straight distance, so a stem that curls
  back onto its junction from far along its path is not excused; straight
  vessels give the same bound either way. Two tubes meeting at a junction
  overlap out to a distance set by their radii and the angle between them. A daughter leaves its parent at Zamir's
  angle, at least 140 degrees from the parent's backward direction, so the
  distance between a parent point s behind J and a daughter point t beyond it
  is at least 0.95 (s + t) and they can only overlap while s + t is below
  about 1.05 (r_P + r_Q + margin); two daughters diverge by about 75 degrees
  and overlap while s + t is below about 1.65 of it. KIN_REACH = 1.7 covers
  both, and because the bound scales with the actual radii it also covers a
  parent whose last sub-segment carries an aneurysm and a daughter that
  starts at the parent's calibre. Grammars with narrower bifurcations (below
  about 45 degrees between daughters) overlap further than this excuses and
  are reported as collisions, which is what they are geometrically. Nothing
  else is excused: an unrelated branch passing through a junction is a
  collision wherever it passes.

The descriptor tool (describe.py) applies the same two rules to a finished
network, so a network grown with a margin shows no pair below it there, and a
claim of "no collisions" is a measurement rather than an assumption. The check
is on centreline points, not on the segments between them, so a crossing can
be missed by up to half the point spacing; `margin` should be at least the
spacing of the finest vessels.

Python 3.9 compatible; the spatial index comes from `spatial.py`.
"""
import numpy as np

# Multiple of (r_P + r_Q + margin) within which two points of vessels meeting
# at a junction may sit, measured as the sum of their distances to it.
KIN_REACH = 1.7


class CollisionAvoider:
    """
    Bookkeeping around a spatial index for one growing network.

    Args:
        index: a spatial index with `add(points, radii, tags)` and
            `query(points, radii, margin) -> (qi, si)` (see spatial.py).
        margin (float): clearance required between vessel surfaces, in
            grammar units.
        attempts (int): redraws a walk step is allowed before its branch is
            terminated.
    """

    def __init__(self, index, margin=0.0, attempts=10):
        if margin < 0.0:
            raise ValueError("the collision margin cannot be negative")
        if attempts < 0:
            raise ValueError("the attempt budget cannot be negative")
        self.index = index
        self.margin = float(margin)
        self.attempts = int(attempts)
        self._parent = []          # stem id -> parent stem id or -1
        self._junction = []        # stem id -> (3,) start point
        self._length = []          # stem id -> arc position of its last stored point
        self._arc = np.empty(0)    # arc position along its stem of every stored point
        self.events = {"collision_redraws": 0, "collision_terminations": 0,
                       "collision_truncated_stems": 0}

    @property
    def n_points(self):
        return self.index.n

    def open_stem(self, parent, junction, junction_diameter=None):
        """Registers a stem leaving `junction` and returns its id."""
        self._parent.append(-1 if parent is None else int(parent))
        self._junction.append(np.asarray(junction, dtype=float).copy())
        self._length.append(0.0)
        return len(self._parent) - 1

    def conflicts(self, points, radii, stem, arcs):
        """
        Returns a boolean mask over `points` marking those that collide with a
        stored point, after the exclusions described in the module docstring.

        Args:
            points (ndarray): (m, 3) proposed points.
            radii (ndarray): (m,) their radii.
            stem (int): id from open_stem of the stem they extend.
            arcs (ndarray): (m,) arc-length position of each point along its stem.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        radii = np.asarray(radii, dtype=float).reshape(-1)
        arcs = np.asarray(arcs, dtype=float).reshape(-1)
        bad = np.zeros(len(points), dtype=bool)
        if self.index.n == 0 or len(points) == 0:
            return bad
        qi, si = self.index.query(points, radii, self.margin)
        if len(qi) == 0:
            return bad
        tags = self.index.tags[si]
        threshold = self.index.radii[si] + radii[qi] + self.margin

        own = tags == stem
        excused = own & (np.abs(arcs[qi] - self._arc[si]) <= 2.0 * threshold)

        parent = self._parent[stem]
        if parent >= 0:
            parents = np.asarray(self._parent)
            on_parent = tags == parent
            kin = (on_parent | (parents[tags] == parent)) & ~own
            if kin.any():
                # the parent's points are measured from its end, a sibling's from its start
                to_junction = np.where(on_parent, self._length[parent] - self._arc[si], self._arc[si])
                together = to_junction + arcs[qi]
                excused |= kin & (together < KIN_REACH * threshold)

        bad[qi[~excused]] = True
        return bad

    def commit(self, points, radii, stem, arcs):
        """Stores accepted points of `stem` so later proposals are checked against them."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return
        radii = np.asarray(radii, dtype=float).reshape(-1)
        arcs = np.asarray(arcs, dtype=float).reshape(-1)
        self.index.add(points, radii, np.full(len(points), stem, dtype=np.int64))
        self._arc = np.concatenate([self._arc, arcs])
        self._length[stem] = max(self._length[stem], float(arcs.max()))
