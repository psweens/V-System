'''
Cohen, D. Voxel traversal along a 3D line.
from Heckbert, P.S. Graphics Gems IV. Morgan Kaufmann.

Visits Subroutine visits all voxels along the line
segment (x,y,z) and (x+dx,y+dy,z+dz).
'''

import numpy as np
from utils import generate_points

def check_boundary(image):
    """
    Sets the values of the boundary pixels of a 3D numpy array to zero.

    Args:
    image (numpy.ndarray): 3D numpy array representing the image.

    Returns:
    numpy.ndarray: 3D numpy array with boundary pixels set to zero.
    """
    image[0,] = 0
    image[-1,] = 0

    image[:, 0, :] = 0
    image[:, -1, :] = 0

    image[..., 0] = 0
    image[..., -1] = 0

    return image

def transversal(x, y, z, dx, dy, dz, diam, 
                img_stack=None, tVol=(600,600,700), resolution=(1,1,1)):

    """
    Finds the voxels transversed by a segment between two points.

    Args:
    x (int): X-coordinate of the starting point.
    y (int): Y-coordinate of the starting point.
    z (int): Z-coordinate of the starting point.
    dx (int): Change in X-coordinate between starting and ending points.
    dy (int): Change in Y-coordinate between starting and ending points.
    dz (int): Change in Z-coordinate between starting and ending points.
    diam (float): Diameter of the segment.
    img_stack (np.ndarray, optional): Image stack. Defaults to None.
    tVol (tuple, optional): Tissue volume. Defaults to (600, 600, 700).
    resolution (tuple, optional): Voxel resolution. Defaults to (1, 1, 1).

    Returns:
    np.ndarray: Image stack with voxels transversed by the segment.

    """
    # Calculate change in coordinates
    dx = dx - x
    dy = dy - y
    dz = dz - z

    # Calculate sign and absolute value of the change in coordinates
    sx = np.sign(dx)
    sy = np.sign(dy)
    sz = np.sign(dz)

    ax = abs(dx)
    ay = abs(dy)
    az = abs(dz)

    bx = 2 * ax
    by = 2 * ay
    bz = 2 * az

    exy = ay - ax
    exz = az - ax
    ezy = ay - az

    n = ax + ay + az

    # Calculate radius
    r = int(np.floor(0.5 * diam))
    mx, my, mz = np.minimum((x + r, y + r, z + r), tVol) - (x, y, z)

    # Find voxels within the tube/vessel
    while n > 0:
        if exy < 0:
            if exz < 0:
                x += sx
                exy += by
                exz += bz
                if 0 <= x < tVol[0]:
                    img_stack = diamVoxels(x, y, z, 0, r, img_stack, tVol, res=resolution)
            else:
                z += sz
                exz -= bx
                ezy += by
                if 0 <= z < tVol[2]:
                    img_stack = diamVoxels(x, y, z, 2, r, img_stack, tVol, res=resolution)
        else:
            if ezy < 0:
                z += sz
                exz -= bx
                ezy += by
                if 0 <= z < tVol[2]:
                    img_stack = diamVoxels(x, y, z, 2, r, img_stack, tVol, res=resolution)
            else:
                y += sy
                exy -= bx
                ezy -= bz
                if 0 <= y < tVol[1]:
                    img_stack = diamVoxels(x, y, z, 1, r, img_stack, tVol, res=resolution)

        n -= 1

    return img_stack

def discretisation(p1, dmin, dmax, tVol):
    """
    Converts a 3D point from real coordinates to discrete voxels coordinates.

    Args:
    - p1 (ndarray): 3D point in real coordinates with the format [x, y, z, d].
    - dmin (ndarray): 3D point representing the minimum values for x, y and z.
    - dmax (ndarray): 3D point representing the maximum values for x, y and z.
    - tVol (tuple): tuple with the 3 dimensions of the target 3D image volume.

    Returns:
    - ndarray: the converted 3D point in discrete voxels coordinates with the format [x, y, z, d].
    """
    # Calculate the discrete voxel coordinates
    x = np.floor(tVol[0] * (p1[0] - dmin[0]) / (dmax[0] - dmin[0]))
    y = np.floor(tVol[1] * (p1[1] - dmin[1]) / (dmax[1] - dmin[1]))
    z = np.floor(tVol[2] * (p1[2] - dmin[2]) / (dmax[2] - dmin[2]))

    # Return the converted point as a column vector
    return np.array([x, y, z, p1[3]]).reshape(-1, 1)

def diamVoxels(x, y, z, idx, r, img_stack, tVol, res):
    '''
    Creates a binary image of the voxels in a sphere centered at (x, y, z) with radius r. 
    The size of the pixels is relative to the diameter of the sphere.

    Args:
        x (int): x-coordinate of the center of the sphere.
        y (int): y-coordinate of the center of the sphere.
        z (int): z-coordinate of the center of the sphere.
        idx (int): The axis index of the diameter. 0, 1, or 2.
        r (int): Radius of the sphere.
        img_stack (ndarray): 3D numpy array representing the stack of images.
        tVol (tuple): Tuple of three integers representing the dimensions of the image stack.
        res (tuple): Tuple of three floats representing the resolution of each voxel in x, y, and z directions.

    Returns:
        ndarray: A binary 3D numpy array of the voxels within the sphere.
    '''

    t0, t1, t2 = tVol[0], tVol[1], tVol[2]
    r0, r1, r2 = res[0], res[1], res[2]

    # Calculate the indices of the voxels within the sphere and set them to 1
    if idx == 0:
        for i in range(-r, r+1):
            if (int(y+i) < t1) & (int(y+i) >= 0):
                for j in range(-r, r+1):
                    if (int(z+j) < t2) & (int(z+j) >= 0):
                        if pow(pow(i*r1, 2) + pow(j*r2, 2), 0.5) <= r:
                            img_stack[int(x-1), int(y+i-1), int(z+j-1)] = 1
    elif idx == 1:
        for i in range(-r, r+1):
            if (int(x+i) < t0) & (int(x+i) >= 0):
                for j in range(-r, r+1):
                    if (int(z+j) < t2) & (int(z+j) >= 0):
                        if pow(pow(i*r0, 2) + pow(j*r2, 2), 0.5) <= r:
                            img_stack[int(x+i-1), int(y-1), int(z+j-1)] = 1
    else:
        for i in range(-r, r+1):
            if (int(x+i) < t0) & (int(x+i) >= 0):
                for j in range(-r, r+1):
                    if (int(y+j) < t1) & (int(y+j) >= 0):
                        if pow(pow(i*r0, 2) + pow(j*r1, 2), 0.5) <= r:
                            img_stack[int(x+i-1), int(y+j-1), int(z-1)] = 1

    return img_stack


def findVessel(i, data):
    """
    Returns a vessel (a numpy array) that contains data from column i until it encounters NaN value. 
    It also returns the index of the next column to be read. 
    
    Args:
    i (int): The index of the column to start reading the data from.
    data (numpy.ndarray): The input data, with each column representing a different variable and each row representing a different observation.
    
    Returns:
    Tuple[int, numpy.ndarray]: The index of the next column to be read and the vessel containing the read data.
    """
    
    # Initialize an empty array to hold the read data
    vessel = np.array([])
    # Initialize a flag variable to indicate if the function should stop reading data
    stop = False
    
    # While the function hasn't encountered a NaN value, it keeps reading data from the next column
    while not stop:
        # If the data in the current column isn't NaN, read it into the vessel
        if not np.isnan(data[0,i]):
            # If the vessel is empty, add the data as a column vector
            if vessel.size == 0:
                vessel = data[:,i].reshape(-1,1)
            # If the vessel already contains data, add the new column to the right of the previous columns
            else:
                vessel = np.hstack((vessel, data[:,i]))
            # Move to the next column
            i += 1
        # If the data in the current column is NaN, stop reading data
        else:
            stop = True
    
    # Return the index of the next column to be read and the vessel containing the read data
    return i, vessel


def process_network(data,tVol):
    
    """
    Processes a 3D medical image data, generates point data,
    and finds the voxels transversed by segments between two points.

    Args:
    data (np.ndarray): a 2D array of medical image data where each column represents a vessel
    tVol (tuple): a tuple of integers (x, y, z) representing the size of the tissue volume

    Returns:
    np.ndarray: A 3D array of integers representing the voxels transversed by segments
    between two points after checking for boundary violations
    """

    # Find non-NaN min/max values across data leaving a 10% margin in case of
    # vessel/tissue boundary contact
    dmin = np.nanmin(data, axis=1) * 1.1
    dmax = np.nanmax(data, axis=1) * 1.1

    # Initializing new array
    newarray = np.array([])

    # Creating an empty 3D array with the given shape
    img_stack = np.empty(tVol, dtype=int)

    # Loop over columns of data
    for i in range(data.shape[1]):
        if not np.isnan(data[0, i]):
            # Discretise data into defined tissue volume
            tempvec = discretisation(data[:, i], dmin, dmax, tVol)
        else:
            tempvec = data[:, i].reshape(-1, 1)

        # Generating & organising point data into array
        newarray = generate_points(newarray, tempvec, usenan=False)

    # Loop over rows of newarray
    j = 0
    for i in range(newarray.shape[1] - 1):
        j += 1
        if not np.isnan(newarray[0, i]):
            if not np.isnan(newarray[0, j]):
                # Calculate diameter and pass into function to find voxels transversed by
                # by segment between two points
                diam = 0.5 * (newarray[3, i] + newarray[3, j])
                transversal(newarray[0, i], newarray[1, i], newarray[2, i],
                            newarray[0, j], newarray[1, j], newarray[2, j],
                            diam, img_stack, tVol)

    # Check boundary violations and return img_stack
    return check_boundary(img_stack)
