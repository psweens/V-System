"""
Plotting helpers for vessel networks. matplotlib is imported on use, so the
core pipeline does not depend on it.
"""
import math

import numpy as np


def plot_coords(nodes, bare_plot=False, linewidth=0.5, show=True):
    """
    Plots a network in 3D.

    Args:
        nodes (ndarray): (4, N) array of x, y, z and diameter with NaN columns
            separating branches, as returned by utils.interpolate_segments.
        bare_plot (bool): hide the axes.
        linewidth (float): line width per unit of diameter.
        show (bool): call matplotlib's show() before returning.

    Returns:
        matplotlib.axes.Axes: the 3D axes.
    """
    import matplotlib.pyplot as plt

    nodes = np.asarray(nodes, dtype=float)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    start = 0
    for i in range(nodes.shape[1] + 1):
        if i == nodes.shape[1] or np.isnan(nodes[0, i]):
            run = nodes[:, start:i]
            for j in range(run.shape[1] - 1):
                ax.plot(run[0, j:j + 2], run[1, j:j + 2], run[2, j:j + 2],
                        linewidth=linewidth * run[3, j], color="tab:blue")
            start = i + 1
    finite = nodes[:3, ~np.isnan(nodes[0])]
    if finite.size:
        extent = np.ptp(finite, axis=1)
        ax.set_box_aspect(np.maximum(extent, 1e-9))
    if bare_plot:
        ax.set_axis_off()
    if show:
        plt.show()
    return ax


def print_coords(rows):
    """
    Prints interpreter rows, with a gap marker for every branch end.

    Args:
        rows (iterable): (x, y, z, diameter, segment) tuples from
            analyseGrammar.branching_turtle_to_coords.
    """
    for x, y, z, diameter, _ in rows:
        if math.isnan(x):
            print("<gap>")
        else:
            print(f"({x:.2f}, {y:.2f}, {z:.2f}) d={diameter:.2f}")
