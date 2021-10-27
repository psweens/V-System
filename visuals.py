import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import isnan
plt.style.use('bmh')  # Use some nicer default colors

def plot_coords(coords, array=True, bare_plot=False):
    '''
    Takes a list of coordinates and coverts to correct format for matplotlib
    '''

    ax = plt.figure()
    if bare_plot:
        # Turns off the axis markers.
        ax = plt.axis('off')
    else:
        # Ensures equal aspect ratio.
        ax = ax.gca(projection='3d')

    if array:
        print(coords.shape)
        for i in range(coords.shape[1]-1):
            # print('%f %f %f' %(coords[0,i], coords[1,i], coords[2,i]))
            ax.plot(coords[0,i:i+2], coords[1,i:i+2], coords[2,i:i+2], color='blue')
    else:
        # Converts a list of coordinates into
        # lists of X and Y values, respectively.
        X, Y, Z, alpha, beta, diam, _, _, _, _ = zip(*coords)
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        diam = np.array(diam)

        for i in range(len(X)-1):
            # print(Z[i:i+2])
            # print('break')
            ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2], linewidth=0.5*diam[i], color='blue')

    plt.show()

def print_coords(coords):
    for (x, y, z, _, _, _, _) in coords:
        if isnan(x):
            print('<gap>')
        else:
            print('({:.2f}, {:.2f}, {:.2f})'.format(x, y, z))