"""
Rigid placement of a network by the frame its archive records.

A network grown with the guidance of 3.5 carries a frame record in its
metadata: the directions its growth was steered by, as unit vectors in the
coordinates and unit of `nodes`. A frame of kind "axis" holds the axis the
vessels were drawn along and its sense, "nematic" when the rule cares only
for the line of the axis and "polar" when it also cares which way along it,
in which case "tree_senses" gives each tree a +1 or -1; a frame of kind
"plane" holds the normal of the plane the vessels were kept close to; a frame
of kind "none", which a network grown with the guidance off carries, holds
neither. Every frame also records "origin", the centre of the growth box when
there was one and otherwise the first root, and the turtle frame the first
tree started from: "grow_direction", the heading of its root (the generator's
default is [0, 1, 0]), and "grow_perpendicular" (default [0, 0, 1]). The
growth box, when there is one, is centred on the origin with its sides along
the x, y and z axes of `nodes`; those are (grow_direction x
grow_perpendicular, grow_direction, grow_perpendicular) for a network grown
from the default turtle frame and for one moved from it by transform_frame,
but not for a network grown from another direction. "rules" lists the
guidance rules as normalised dicts, each carrying its own axis or normal
when its field has one. Archives written before 3.5 carry no record at all.

The record exists so that a network can be placed in a scene without being
grown again. A network grows in its own coordinates, where the guidance axis
is whatever its run asked for; to assemble several into one volume the caller
rotates each so that its frame agrees with where the vessels should run, and
this module provides the rotations. They are plain 3 x 3 proper orthonormal
matrices Q, applied to the columns of `nodes` as xyz' = scale Q xyz + t and
to the frame as Q v for every vector in it, so that the moved record still
describes the moved vessels: a descriptor measured about the frame's axis,
the order parameter for one, is the same before and after the move.

    rotation_about(axis, angle)          the rotation by `angle` about `axis` (Rodrigues)
    rotation_between(u, w)               the rotation taking the direction u to w
    rotation_of_frames(u1, u2, w1, w2)   the rotation taking the pair (u1, u2) to (w1, w2)
    random_rotation(rng)                 a rotation drawn uniformly (Haar) over SO(3)
    align(frame, axis=A, spin=phi)       the rotation taking the frame's axis (or normal) to A, then spun about A
    transform_nodes(nodes, Q, t, scale)  the moved archive
    transform_frame(frame, Q, t, scale)  the moved record

Two recipes cover the usual needs. To lay a bank of networks along one shared
axis A, network i takes Q_i = align(frame_i, axis=A, spin=U(0, 2 pi)): its own
axis goes to A, and the spin, drawn uniformly, removes the preferred direction
around A that the generator's turtle frame would otherwise give every network
alike. A polar frame's axis goes to A exactly, so a caller mixing polarities
along a shared axis passes -A for some of the networks. To place networks
with no preferred direction, network i takes Q_i = random_rotation(rng),
uniform over the rotations, and its frame is carried along so that the record
still says where its vessels run. Rescaling, from one unit to another say, is
a transform with Q the identity: the scale given to transform_nodes, which
multiplies the coordinates and the diameters, is given to transform_frame
too, which applies it to the origin and leaves the unit vectors and the rules
as they are.

This module is deliberately outside main's import closure and imports nothing
from the repository, so that a caller assembling archives needs no more than
numpy. Python 3.9 compatible.
"""
import copy
import math
import numbers

import numpy as np

# A matrix passed in as a rotation must have a determinant within this of 1
# and Q^T Q within this of the identity, entry by entry. A reflection, a
# non-orthogonal matrix or one of the wrong shape is refused rather than
# applied to a network it would distort or mirror.
ROTATION_TOL = 1e-9

# Below this |u x w| two directions count as parallel or antiparallel in
# rotation_between, and the Rodrigues form, whose closing term carries
# 1 / |u x w|^2, gives way to the identity or to a half turn.
PARALLEL_TOL = 1e-8

# The vectors of a frame record, and of each of its rules, that a rotation moves.
FRAME_VECTORS = ("axis", "normal", "grow_direction", "grow_perpendicular")
RULE_VECTORS = ("axis", "normal")
FRAME_KINDS = ("none", "axis", "plane")


def _vector(value, name):
    """`value` as a (3,) float array; ValueError when it is not three finite numbers."""
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return vector


def _unit(value, name):
    """`value` as a unit (3,) float array; ValueError for a zero or non-finite vector."""
    vector = _vector(value, name)
    # scaled by its largest component first, so that the norm can neither
    # overflow nor lose precision in subnormal squares
    largest = float(np.max(np.abs(vector)))
    if largest == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    vector = vector / largest
    return vector / float(np.linalg.norm(vector))


def _skew(k):
    """The matrix [k]x, with [k]x v = k x v."""
    return np.array([[0.0, -k[2], k[1]],
                     [k[2], 0.0, -k[0]],
                     [-k[1], k[0], 0.0]])


def _rotation(matrix):
    """`matrix` as a (3, 3) float array, checked to be a proper rotation to ROTATION_TOL."""
    try:
        q = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Q must be a 3 x 3 matrix of numbers")
    if q.shape != (3, 3) or not np.all(np.isfinite(q)):
        raise ValueError(f"Q must be a finite 3 x 3 matrix, got shape {q.shape}")
    determinant = float(np.linalg.det(q))
    if abs(determinant - 1.0) > ROTATION_TOL:
        raise ValueError(f"Q must be a proper rotation (determinant 1), got determinant {determinant!r}")
    if float(np.max(np.abs(q.T @ q - np.eye(3)))) > ROTATION_TOL:
        raise ValueError("Q must be orthonormal: Q^T Q differs from the identity beyond ROTATION_TOL")
    return q


def _placement(t, scale):
    """The translation as a (3,) float array and the scale as a float, both checked."""
    t = _vector(t, "t")
    # a number and nothing else: a string that float() would read is not a scale
    if not isinstance(scale, numbers.Real) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return t, float(scale)


def _kind(frame):
    """The kind of a frame record; ValueError for a missing or malformed one."""
    if frame is None:
        raise ValueError("the archive carries no frame record (every archive before 3.5 has none)")
    if not isinstance(frame, dict):
        raise ValueError(f"a frame record is a dict, got {type(frame).__name__}")
    kind = frame.get("kind")
    if kind not in FRAME_KINDS:
        raise ValueError(f"a frame's kind is one of {FRAME_KINDS}, got {kind!r}")
    return kind


def rotation_about(axis, angle):
    """
    The right-handed rotation by `angle` radians about `axis`.

    Rodrigues' formula, cos a I + sin a [u]x + (1 - cos a) u u^T with u the
    unit axis; `axis` is normalised first.

    Raises:
        ValueError: for a zero or non-finite axis or a non-finite angle.
    """
    u = _unit(axis, "axis")
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError(f"angle must be finite, got {angle!r}")
    c, s = math.cos(angle), math.sin(angle)
    return c * np.eye(3) + s * _skew(u) + (1.0 - c) * np.outer(u, u)


def rotation_between(u, w, nematic=False):
    """
    A proper rotation Q with Q u = w: the one about u x w, which turns by the
    angle between the two and no more. `u` and `w` are normalised first.

    With `nematic` the two are lines rather than arrows: w is replaced by -w
    when u . w < 0, so that Q takes u onto the line of w by the nearer of the
    two ends and turns by at most a right angle.

    Let k = u x w, c = u . w and s^2 = k . k. For s above PARALLEL_TOL,
    Q = I + [k]x + [k]x^2 (1 - c) / s^2, Rodrigues' rotation about k by the
    angle between u and w. The textbook closing factor is 1 / (1 + c), which
    the identity s^2 = (1 - c)(1 + c) turns into this one; near an
    antiparallel pair 1 + c is the difference of two nearly equal numbers and
    loses most of its digits, whereas (1 - c) / s^2 has none to lose, so this
    form stays orthonormal to rounding however close the pair comes to a half
    turn. Below the threshold the pair is parallel, and Q = I, or
    antiparallel, and Q is the half turn 2 e e^T - I about e = unit(u x e_j),
    e_j being the lab axis least aligned with u, so that e is perpendicular to
    u and the half turn takes u to -u.

    Raises:
        ValueError: for a zero or non-finite direction.
    """
    u = _unit(u, "u")
    w = _unit(w, "w")
    c = float(u @ w)
    if nematic and c < 0.0:
        w, c = -w, -c
    k = np.cross(u, w)
    s2 = float(k @ k)
    if math.sqrt(s2) > PARALLEL_TOL:
        skew = _skew(k)
        return np.eye(3) + skew + (skew @ skew) * ((1.0 - c) / s2)
    if c > 0.0:
        return np.eye(3)
    e = _unit(np.cross(u, np.eye(3)[int(np.argmin(np.abs(u)))]), "e")
    return 2.0 * np.outer(e, e) - np.eye(3)


def _basis(first, second, name):
    """The right-handed orthonormal basis [first, second', first x second'] as columns."""
    a = _unit(first, name + "1")
    b = _unit(second, name + "2")
    rejected = b - (b @ a) * a
    norm = float(np.linalg.norm(rejected))
    if norm <= PARALLEL_TOL:
        raise ValueError(f"{name}2 must not be parallel to {name}1, which would leave the spin free")
    b = rejected / norm
    # a second pass removes what rounding left along `a` when the pair is
    # close to parallel, so that the basis stays orthonormal to rounding
    b = b - (b @ a) * a
    b = b / float(np.linalg.norm(b))
    return np.stack([a, b, np.cross(a, b)], axis=1)


def rotation_of_frames(u1, u2, w1, w2):
    """
    The rotation taking the pair of directions (u1, u2) to (w1, w2).

    One direction fixes a rotation only up to the spin about it; a pair fixes
    it completely, which is how a heading and its perpendicular are carried
    from one placement to another. Each pair is first made a right-handed
    orthonormal basis by normalising its first direction and keeping the part
    of its second that is perpendicular to it (Gram-Schmidt on the first:
    u2' = unit(u2 - (u2 . u1) u1)), and Q = W U^T with U = [u1, u2', u1 x u2']
    and W = [w1, w2', w1 x w2'] as columns. A second direction that is not
    perpendicular to its first is therefore mapped as u2' to w2', not as u2
    to w2.

    Raises:
        ValueError: for a zero or non-finite direction, or a second direction
            parallel to its first.
    """
    return _basis(w1, w2, "w") @ _basis(u1, u2, "u").T


def random_rotation(rng):
    """
    A rotation drawn uniformly over the rotations (Haar measure) from `rng`,
    a numpy.random.Generator.

    Four normal draws, normalised, are a unit quaternion uniform over the
    3-sphere, and that quaternion's rotation matrix is then uniform over
    SO(3). Exactly four draws are consumed, in one call, so a sequence of
    placements is reproducible from the generator's seed. With q = (a, b, c,
    d) the matrix is

        [[a a + b b - c c - d d, 2 (b c - a d),         2 (b d + a c)],
         [2 (b c + a d),         a a - b b + c c - d d, 2 (c d - a b)],
         [2 (b d - a c),         2 (c d + a b),         a a - b b - c c + d d]].
    """
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    a, b, c, d = (float(v) for v in q)
    return np.array([[a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
                     [2.0 * (b * c + a * d), a * a - b * b + c * c - d * d, 2.0 * (c * d - a * b)],
                     [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def align(frame, *, axis=None, normal=None, spin=0.0):
    """
    The rotation taking a frame's axis (or normal) to a given one, then spun
    about that given direction by `spin` radians.

    Exactly one of `axis` and `normal` is given, and it must be the kind of
    direction the frame has: an axis for a frame of kind "axis", a normal for
    one of kind "plane". With an axis, Q = rotation_about(axis, spin) @
    rotation_between(frame axis, axis, nematic=(frame sense is "nematic")): a
    nematic frame is turned onto the line of the axis by the nearer end, a
    right angle at most, while a polar frame's axis goes to the given axis
    exactly, whatever the sign of their dot product (to mix polarities along a
    shared axis the caller passes -A for some of the networks). With a
    normal, the same with the frame's normal and nematic=True, since a plane
    has no side. The spin is about the given direction, so it leaves the
    mapped axis or normal where it is and only turns the vessels around it;
    without it every network aligned to the same axis would arrive with the
    generator's turtle frame turned the same way.

    Raises:
        ValueError: for a missing record (None, as every archive before 3.5
            has) or one of kind "none", which holds no direction to align by;
            for a request the frame cannot answer, an axis of a plane frame
            or a normal of an axis frame; for neither or both of `axis` and
            `normal`; and for a zero or non-finite direction or spin.
    """
    kind = _kind(frame)
    if kind == "none":
        raise ValueError("a frame of kind 'none' holds no direction to align by")
    if (axis is None) == (normal is None):
        raise ValueError("give exactly one of axis and normal")
    if axis is not None:
        if kind != "axis":
            raise ValueError(f"a frame of kind {kind!r} has no axis to align; give a normal")
        sense = frame.get("sense")
        if sense not in ("nematic", "polar"):
            raise ValueError(f"an axis frame's sense is 'nematic' or 'polar', got {sense!r}")
        own, target, nematic = frame.get("axis"), axis, sense == "nematic"
    else:
        if kind != "plane":
            raise ValueError(f"a frame of kind {kind!r} has no normal to align; give an axis")
        own, target, nematic = frame.get("normal"), normal, True
    return rotation_about(target, spin) @ rotation_between(own, target, nematic=nematic)


def transform_nodes(nodes, Q, t=(0.0, 0.0, 0.0), scale=1.0):
    """
    The archive moved rigidly and rescaled: xyz' = scale Q xyz + t and d' = scale d.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN
            separators, in any unit.
        Q (ndarray): (3, 3) proper rotation, as the functions above return.
        t (sequence): the translation, three numbers in the unit of the result.
        scale (float): positive factor applied to the coordinates before the
            translation and to the diameters, 1e-3 from micrometres to
            millimetres for instance.

    Returns:
        ndarray: a new (4, N) float array. A NaN column stays NaN where it
        was, so the polylines and the graph of the archive are unchanged.

    Raises:
        ValueError: for a Q that is not a proper rotation to ROTATION_TOL (a
            reflection, a non-orthogonal matrix, a wrong shape), a scale that
            is not a positive finite number, a translation that is not three
            finite numbers, or nodes that are not (4, N).
    """
    q = _rotation(Q)
    t, scale = _placement(t, scale)
    try:
        nodes = np.asarray(nodes, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("nodes must be a (4, N) array of numbers")
    if nodes.size == 0:
        return np.empty((4, 0))
    if nodes.ndim != 2 or nodes.shape[0] != 4:
        raise ValueError(f"nodes must be a (4, N) array of x, y, z and diameter, got {nodes.shape}")
    out = np.empty(nodes.shape)
    out[:3] = scale * (q @ nodes[:3]) + t[:, None]
    out[3] = scale * nodes[3]
    return out


def _rotated(value, q, name):
    """`value` rotated by q, as a list of three floats."""
    return [float(v) for v in q @ _vector(value, name)]


def transform_frame(frame, Q, t=(0.0, 0.0, 0.0), scale=1.0):
    """
    The frame record of a network moved by transform_nodes with the same Q,
    t and scale, so that the record still describes the moved vessels.

    Every vector of the record (the axis or normal, grow_direction,
    grow_perpendicular and each rule's axis or normal) becomes Q v, literally
    and without re-normalising, and the origin, a point, becomes
    scale Q o + t; everything else (the version, the kind, the sense, the
    tree senses, each rule's field, threshold, length, onset and polarity) is
    copied as it is. The vectors are therefore moved by exactly the matrix
    that moves the nodes, a pure rescale (Q the identity) leaves them
    bitwise as they were, and a Q from the functions above, orthonormal to
    rounding, keeps them unit to rounding; a Q that is orthonormal only to
    ROTATION_TOL leaves them unit only to that. The result is a new record
    of plain Python lists and floats, JSON-ready as the original was; the
    input is deep-copied and never touched.

    Raises:
        ValueError: for a missing record (None) or one that is not a dict or
            of an unknown kind, a vector or origin of the record that is not
            three finite numbers, and the Q, t and scale transform_nodes
            refuses.
    """
    _kind(frame)
    q = _rotation(Q)
    t, scale = _placement(t, scale)
    out = copy.deepcopy(frame)
    for key in FRAME_VECTORS:
        if out.get(key) is not None:
            out[key] = _rotated(out[key], q, key)
    rules = out.get("rules")
    if rules is not None:
        if not isinstance(rules, list) or not all(isinstance(rule, dict) for rule in rules):
            raise ValueError("a frame's rules are a list of dicts")
        for rule in rules:
            for key in RULE_VECTORS:
                if rule.get(key) is not None:
                    rule[key] = _rotated(rule[key], q, "rule " + key)
    if out.get("origin") is not None:
        origin = _vector(out["origin"], "origin")
        out["origin"] = [float(v) for v in scale * (q @ origin) + t]
    return out
