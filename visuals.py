import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import isnan
plt.style.use('bmh')  # Use some nicer default colors

def plot_coords(coords, array=True, bare_plot=False):
    '''
    Plots a list of coordinates in 3D.

    Parameters:
        coords (list or ndarray): A list of coordinates. If `array` is `True`,
                                  `coords` should be an ndarray with shape (3, N).
                                  If `array` is `False`, `coords` should be a list of
                                  tuples containing the X, Y, Z, alpha, beta, diam,
                                  ect. values for each coordinate.
        array (bool): Determines if `coords` is an ndarray or a list. Default is `True`.
        bare_plot (bool): Determines if the plot should be bare or not. Default is `False`.
                          If `True`, axis markers are turned off.

    Returns:
        None
    '''
    # Create the plot and configure it appropriately.
    ax = plt.figure()
    if bare_plot:
        # Turns off the axis markers.
        ax = plt.axis('off')
    else:
        # Ensures equal aspect ratio.
        ax = ax.gca(projection='3d')

    if array:
        # Loop over each coordinate and plot it.
        for i in range(coords.shape[1]-1):
            ax.plot(coords[0,i:i+2], coords[1,i:i+2], coords[2,i:i+2], color='blue')
    else:
        # Unpack the coordinates into separate arrays.
        X, Y, Z, alpha, beta, diam, _, _, _, _ = zip(*coords)
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        diam = np.array(diam)

        # Loop over each coordinate and plot it.
        for i in range(len(X)-1):
            ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2], linewidth=0.5*diam[i], color='blue')

    # Show the plot.
    plt.show()

def print_coords(coords):
    """
    Prints the coordinates provided, with gaps for any `NaN` values.

    Parameters:
        coords (list): A list of coordinates to be printed. Each coordinate should be a tuple of (x, y, z, alpha, beta, diam, label).

    Returns:
        None
    """

    # Iterate through each coordinate in the list.
    for (x, y, z, _, _, _, _) in coords:
        if isnan(x):
            # If there is a `NaN` value for the x-coordinate, print a gap.
            print('<gap>')
        else:
            # Otherwise, print the x, y, and z-coordinates rounded to 2 decimal places.
            print('({:.2f}, {:.2f}, {:.2f})'.format(x, y, z))
