"""
Smooth tortuosity for V-System centrelines: a persistent random walk on the
unit tangent, and the pinned form of it that traces an anastomosis bridge.

The grammar decides where a vessel segment starts, how long it is, what
diameter it has and where its daughters turn off; the walk decides only the
path in between. It replaces the five straight sub-segments of a stem, which
interpolate into a piecewise-smooth zig-zag, by a curve whose direction drifts
continuously: at every step of length h the tangent is rotated by an angle
drawn from N(0, sqrt(h / l_p)) about a random axis perpendicular to it, where
the persistence length l_p is a fixed multiple of the local vessel diameter
(the `persistence` setting). The perpendicular vector of the turtle frame is
carried along by the same rotation, so the frame stays orthonormal and the
bifurcation angles the grammar prescribes are applied relative to the actual
direction the vessel arrives with.

For small angles this rule gives a tangent autocorrelation <t(s) . t(0)> of
exp(-s / (2 l_p)): a single rotation about one random perpendicular axis has
mean squared angle h / l_p, whereas the two transverse degrees of freedom of a
three-dimensional worm-like chain with persistence length l_p would give
2 h / l_p. `persistence` is therefore a shape parameter to be calibrated
against measured arc/chord ratios rather than a literal persistence length in
diameters; the factor of two is documented here so that a reader comparing
against polymer-physics conventions is not misled.

A bridge between two existing points is the same walk pinned at both ends: a
cubic Hermite curve carries the bridge from the first point, leaving along that
vessel's own direction, to the second point, arriving along the partner
vessel's direction when the partner is a tip and along the chord when it is
the side of a vessel, and the walk's
deviation from a straight line is added to it after subtracting the linear
trend that would move the end point (a Brownian bridge on the displacement).
The result starts and ends exactly on the two given points, so a graph built
from coincident coordinates joins it to both vessels, and it carries the walk's
wiggles without a kink at either end beyond the angle at which the bridge
meets the second vessel.

Everything here draws from a `numpy.random.Generator` passed in by the caller,
never from the global `random` or `numpy.random` state that the grammar uses,
so enabling the walk changes no draw the grammar makes. Python 3.9 compatible.
"""
import math

import numpy as np


def _rotate(vector, axis, angle):
    """Rodrigues rotation of `vector` about the unit `axis` by `angle` radians."""
    c = math.cos(angle)
    s = math.sin(angle)
    return vector * c + np.cross(axis, vector) * s + axis * (axis @ vector) * (1.0 - c)


def step_std(step, diameter, persistence):
    """
    Standard deviation of the tangent rotation for one step of length `step`
    on a vessel of `diameter`, with persistence length persistence * diameter.
    """
    if persistence <= 0.0 or diameter <= 0.0:
        raise ValueError("persistence and diameter must be positive")
    return math.sqrt(step / (persistence * diameter))


def perpendicular_rotation(rng, heading, perp, std):
    """
    Rotates the turtle frame by an angle drawn from N(0, std) about a random
    axis perpendicular to `heading`, drawn uniformly in the plane spanned by
    `perp` and heading x perp.

    Two draws are consumed in a fixed order, the angle then the axis, so a run
    is reproducible from the generator's seed. The frame is re-orthonormalised
    after the rotation so that rounding cannot accumulate over thousands of
    steps.

    Returns:
        tuple: (heading, perp) unit vectors after the rotation.
    """
    angle = rng.normal(0.0, std)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    binormal = np.cross(heading, perp)
    axis = math.cos(phi) * perp + math.sin(phi) * binormal
    new_heading = _rotate(heading, axis, angle)
    new_heading = new_heading / np.linalg.norm(new_heading)
    new_perp = _rotate(perp, axis, angle)
    new_perp = new_perp - (new_perp @ new_heading) * new_heading
    new_perp = new_perp / np.linalg.norm(new_perp)
    return new_heading, new_perp


def free_walk(rng, start, heading, perp, n_steps, step, std):
    """
    Traces `n_steps` steps of a persistent random walk with constant rotation
    standard deviation `std`, rotating the frame before every step.

    Returns:
        tuple: (points, heading, perp) with points an (n_steps + 1, 3) array
        starting at `start`, and the frame after the last step.
    """
    points = np.empty((n_steps + 1, 3))
    points[0] = start
    pos = np.asarray(start, dtype=float)
    for i in range(n_steps):
        heading, perp = perpendicular_rotation(rng, heading, perp, std)
        pos = pos + step * heading
        points[i + 1] = pos
    return points, heading, perp


def _hermite(p0, t0, p1, t1, n_samples):
    u = np.linspace(0.0, 1.0, n_samples)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def _resample_by_arc_length(points, n_points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= 0.0:
        return np.repeat(points[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, points.shape[1]))
    for k in range(points.shape[1]):
        out[:, k] = np.interp(targets, cumulative, points[:, k])
    out[0] = points[0]
    out[-1] = points[-1]
    return out


def bridge_path(start, end, step, start_tangent=None, rng=None, persistence=None, diameter=None,
                end_tangent=None, tangent_scale=0.6):
    """
    Traces a bridge from `start` to `end` in steps of about `step`.

    Args:
        start, end (sequence): the two points joined, in grammar units. Both
            appear verbatim as the first and last rows of the result.
        step (float): target spacing of the bridge's points, normally the
            spacing of the vessel the bridge leaves.
        start_tangent (sequence or None): unit direction the bridge leaves
            `start` along; None leaves along the chord.
        end_tangent (sequence or None): unit direction the bridge arrives at
            `end` along, for instance into a partner tip's vessel; None
            arrives along the chord, as into the side of a vessel.
        tangent_scale (float): magnitude of the two Hermite tangents as a
            fraction of the chord; smaller values bend towards the chord
            sooner and never overshoot into a hairpin.
        rng, persistence, diameter: when all three are given the bridge carries
            the deviation of a persistent random walk with persistence length
            persistence * diameter, pinned to zero at both ends. Without them
            the bridge is the plain Hermite curve.

    Returns:
        ndarray: (m, 3) points, m >= 2, from `start` to `end` inclusive.

    Raises:
        ValueError: if the two points coincide or `step` is not positive.
    """
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    chord = p1 - p0
    length = float(np.linalg.norm(chord))
    if length <= 0.0:
        raise ValueError("a bridge needs two distinct points")
    if not step > 0.0:
        raise ValueError("step must be positive")
    direction = chord / length
    if start_tangent is None:
        tangent = direction
    else:
        tangent = np.asarray(start_tangent, dtype=float)
        tangent = tangent / np.linalg.norm(tangent)
    if end_tangent is None:
        arrival = direction
    else:
        arrival = np.asarray(end_tangent, dtype=float)
        arrival = arrival / np.linalg.norm(arrival)
    n_steps = max(2, int(math.ceil(1.05 * length / step)))
    dense = _hermite(p0, tangent * tangent_scale * length, p1, arrival * tangent_scale * length,
                     8 * n_steps + 1)
    base = _resample_by_arc_length(dense, n_steps + 1)
    base[0] = p0
    base[-1] = p1
    if rng is None or persistence is None or diameter is None:
        return base

    arc = float(np.sum(np.linalg.norm(np.diff(base, axis=0), axis=1)))
    h = arc / n_steps
    std = step_std(h, diameter, persistence)
    # any vector not parallel to the tangent serves to start the frame
    trial = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = trial - (trial @ tangent) * tangent
    perp = perp / np.linalg.norm(perp)
    walk, _, _ = free_walk(rng, np.zeros(3), tangent, perp, n_steps, h, std)
    straight = np.arange(n_steps + 1)[:, None] * h * tangent
    deviation = walk - straight
    weights = (np.arange(n_steps + 1) / n_steps)[:, None]
    pinned = deviation - weights * deviation[-1]
    out = base + pinned
    out[0] = p0
    out[-1] = p1
    return out
