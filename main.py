# This script generates L-System networks and saves them as 3D TIFF images.

import random
import numpy as np
import cv2
from skimage import io

from preprocessing import resize_stacks, resize_volume
from libGenerator import setProperties
from vSystem import F, I
from analyseGrammar import branching_turtle_to_coords
from visuals import plot_coords, print_coords
from computeVoxel import process_network
from utils import bezier_interpolation


# Default Parameters
n=20 # Number of networks to be created
d0mean = 20.0 # Mean diameter of base of tree
d0std = 5.0 # Standard deviation of base diameter
tissueVolume = (512,512,140)    # Specify number of pixels in the tissue volume
outpath = "/media/sweene01/SSD/More/" # Output path

# Lindenmayer System Parameters
properties = {"k": 3,
              "epsilon": 7,#random.uniform(4,10), # Proportion between length & diameter
              "randmarg": 3, # Randomness margin between length & diameter
              "sigma": 5, # Determines type deviation for Gaussian distributions
              "stochparams": True} # Whether the generated parameters will also be stochastic


def main():
    """Main function that generates L-System networks and saves them as 3D TIFF images."""
    for file in range(n):
        # Randomly assign base diameter (no dimension)
        d0 = np.random.normal(d0mean, d0std)    
        # Randomly assign number of L-System recursive loops
        niter = random.randint(6,14) 
        # Setting L-Sytem properties
        setProperties(properties) 
        
        print('Creating image ... %i with %i iterations' %(file, niter))
        
        # Run L-System grammar for n iterations
        turtle_program = F(niter,d0)
        
        # Convert grammar into coordinates
        coords = branching_turtle_to_coords(turtle_program,d0)
        
        # Analyse / sort coordinate data
        update = bezier_interpolation(coords)
        
        # If you fancy, plot a 2D image of the network!
        # plot_coords(newdata, array=True, bare_plot=False) # bare_plot removes the axis labels
        
        # Run 3D voxel traversal to generate binary mask of L-System network
        image = process_network(update, tVol=tissueVolume)
        
        # Convert to 8-bit format
        image = (255*image).astype('int8')
        
        # Save image volume
        io.imsave(outpath+"Lnet_i{}_{}.tiff".format(niter,file), np.transpose(image, (2, 0, 1)), bigtiff=False)
        
        
if __name__ == "__main__":
    main()
