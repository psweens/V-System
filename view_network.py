import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')  # Use some nicer default colors

def plot_coords(coords, bare_plot=False):    
    '''
    Takes a list of coordinates and coverts to correct format for matplotlib
    '''

    ax = plt.figure()
    if bare_plot:
        # Turns off the axis markers.
        ax = plt.axis('off')
    else:
        # Ensures equal aspect ratio.
        ax = plt.axes(projection='3d')
        
    # Converts a list of coordinates into 
    # lists of X and Y values, respectively.
    X, Y, Z, alpha, beta, diam = zip(*coords) 
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    diam = np.array(diam)
    
    # for i in range(len(X)-1):
    #     print('%f %f %f %f' %(X[i], Y[i], Z[i], diam[i]))

    for i in range(len(X)-1):
        ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2], linewidth=0.5*diam[i], color='blue')

    plt.show()
