"""
Vector helpers and the segment interpolation used by the V-System pipeline.
"""
import numpy as np


def unit(vector):
    """Returns the unit vector in the direction of `vector`."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return vector / norm


def rotate_about(vector, axis, angle_degrees):
    """
    Rotates `vector` about `axis` by `angle_degrees` (right-hand rule),
    using Rodrigues' formula.
    """
    v = np.asarray(vector, dtype=float)
    k = unit(axis)
    theta = np.deg2rad(angle_degrees)
    return v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * (k @ v) * (1.0 - np.cos(theta))


# Uniform cubic B-spline basis in matrix form.
_BSPLINE = np.array([[-1.0, 3.0, -3.0, 1.0],
                     [3.0, -6.0, 3.0, 0.0],
                     [-3.0, 0.0, 3.0, 0.0],
                     [1.0, 4.0, 1.0, 0.0]]) / 6.0


def bspline(control, subdivisions=3):
    """
    Samples a clamped uniform cubic B-spline through a sequence of control points.

    The first and last control points are repeated so that the curve starts and
    ends exactly on them. Each span is sampled with 2**subdivisions points, so
    that 2**subdivisions - 1 intermediate points are inserted between
    consecutive control points, as in the discretisation of the source
    (Galarreta-Valverde 2012, section 3.4.3).

    Args:
        control (ndarray): (n, k) array of control points in k dimensions.
        subdivisions (int): interpolation depth; 0 returns the control points.

    Returns:
        ndarray: (m, k) array of samples, ending on the last control point.
    """
    control = np.asarray(control, dtype=float)
    if control.ndim != 2:
        raise ValueError("control must be a 2-D array of points")
    if len(control) < 2 or subdivisions <= 0:
        return control.copy()
    padded = np.vstack([control[0], control[0], control, control[-1], control[-1]])
    t = np.linspace(0.0, 1.0, 2 ** subdivisions, endpoint=False)
    powers = np.stack([t ** 3, t ** 2, t, np.ones_like(t)], axis=1)
    weights = powers @ _BSPLINE
    samples = [weights @ padded[i:i + 4] for i in range(len(padded) - 3)]
    samples.append(control[-1:])
    return np.vstack(samples)


def interpolate_segments(rows, subdivisions=3):
    """
    Converts interpreter rows into a (4, N) array of x, y, z and diameter with
    NaN columns as branch separators.

    Rows are (x, y, z, diameter, segment) as yielded by
    analyseGrammar.branching_turtle_to_coords. Consecutive rows that share a
    non-negative segment index are the control points of one braced segment
    and are replaced by a B-spline sampled with 2**subdivisions points per
    span; the diameter is interpolated along the curve. Rows outside braces
    are kept as polyline vertices.
    """
    rows = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=float)
    if rows.size == 0:
        return np.empty((4, 0))
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError("rows must be (x, y, z, diameter, segment) tuples")

    separator = np.full(4, np.nan)
    out = []
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        if np.isnan(row[0]):
            out.append(separator)
            i += 1
            continue
        segment = row[4]
        if segment < 0:
            out.append(row[:4])
            i += 1
            continue
        j = i
        while j < n and not np.isnan(rows[j, 0]) and rows[j, 4] == segment:
            j += 1
        control = rows[i:j, :4]
        out.extend(bspline(control, subdivisions) if len(control) >= 3 else control)
        i = j
    return np.array(out).T
