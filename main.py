import random
import numpy as np
import cv2
from skimage import io
from preprocessing import resize_stacks, resize_volume
from libGenerator import setProperties
from vSystem import F, I, R, B
from analyseGrammar import branching_turtle_to_coords
from visuals import plot_coords, print_coords
from computeVoxel import process_network
from utils import bezier_interpolation

# Default Parameters
n = 5  # No. of networks to be created
d0_mean = 20.0  # Diameter of base of tree
d0_std = 5.0  # Standard deviation of base diameter
tissue_volume = (512, 512, 140)  # Specify number of pixels in the tissue volume
outpath = "/home/sweene01/Documents/Test/"  # Output path

for file in range(n):
    # Lindenmayer System Parameters
    properties = {
        "k": 3,
        "epsilon": random.uniform(4, 10),  # Proportion between length & diameter
        "randmarg": random.uniform(2, 4),  # Randomness margin between length & diameter
        "sigma": 5,  # Determines type deviation for Gaussian distributions
        "stochparams": True,  # Whether the generated parameters will also be stochastic
    }

    d0 = np.random.normal(d0_mean, d0_std)  # Randomly assign base diameter (no dimension)
    niter = random.randint(6, 14)  # Randomly assign number of V-System recursive loops
    setProperties(properties)  # Setting V-Sytem properties
    print(f"Creating image ... {file} with {niter} iterations")

    # Run V-System grammar for n iterations
    turtle_program = F(niter, d0)

    # Convert grammar into coordinates
    coords = branching_turtle_to_coords(turtle_program, d0)

    # Analyse / sort coordinate data
    update = bezier_interpolation(coords)

    # If you fancy, plot a 2D image of the network!
    # plot_coords(newdata, array=True, bare_plot=False) # bare_plot removes the axis labels

    # Run 3D voxel traversal to generate binary mask of V-System network
    image = process_network(update, tVol=tissue_volume)

    # Convert to 8-bit format
    image = (255 * image).astype("int8")

    # Save image volume
    io.imsave(
        outpath + "Lnet_i{}_{}.tiff".format(niter, file),
        np.transpose(image, (2, 0, 1)),
        bigtiff=False,
    )
