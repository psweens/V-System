import matplotlib.pyplot as plt
plt.style.use('bmh')  # Use some nicer default colors

def plot_coords(coords, bare_plot=False):
    
    '''
    Takes a list of coordinates and coverts to correct format for matplotlib
    '''
    fig = plt.figure()
    if bare_plot:
        # Turns off the axis markers.
        ax = plt.axis('off')
    else:
        # Ensures equal aspect ratio.
        ax = plt.axes(projection='3d')
    # Converts a list of coordinates into 
    # lists of X and Y values, respectively.
    X, Y, Z, diam = zip(*coords)
    # Draws the plot.
    ax.plot(X, Y, Z, zdir='z');
    plt.show()
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')