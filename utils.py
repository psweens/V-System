import itk
import numpy as np
import bezier as bz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import math as npm

def yaw(angle_degrees):
    """
    Computes a 3x3 yaw rotation matrix for a given angle of rotation around the z-axis.

    Args:
    - angle_degrees: float representing the angle of rotation around the z-axis, in degrees.

    Returns:
    - 3x3 numpy array representing the rotation matrix.
    """

    # convert angle to radians
    angle = np.deg2rad(angle_degrees)

    # create rotation matrix
    A = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                   [np.sin(angle), np.cos(angle), 0.0],
                   [0.0, 0.0, 1.0]])

    return A

def pitch(angle):
    """
    Apply a pitch transformation to a 3D object represented as a numpy array.

    Args:
        angle (float): the angle (in radians) by which to pitch the object

    Returns:
        np.ndarray: a 3x3 numpy array representing the pitch transformation matrix
    """
    # Define the pitch transformation matrix
    A = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                  [0.0, 1.0, 0.0],
                  [-np.sin(angle), 0.0, np.cos(angle)]])

    # Return the transformation matrix
    return A

def roll(angle):
    """
    Applies a roll transformation to a 3x3 numpy array.

    Args:
    angle (float): The angle (in radians) of the desired roll transformation.

    Returns:
    A (numpy.ndarray): A 3x3 numpy array representing the roll transformation.
    """
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(angle), -np.sin(angle)],
                  [0.0, np.sin(angle), np.cos(angle)]])

    return A

def rotate(pitch_angle=0.0, roll_angle=0.0, yaw_angle=0.0, vector=None):
    """
    Applies a sequence of yaw, pitch, and roll rotations to a vector.

    Args:
    - yaw_angle (float): Angle in radians to rotate around the z-axis (yaw).
    - pitch_angle (float): Angle in radians to rotate around the y-axis (pitch).
    - roll_angle (float): Angle in radians to rotate around the x-axis (roll).
    - vector (ndarray): A numpy array of shape (3,) representing the vector to rotate.

    Returns:
    - ndarray: A numpy array of shape (3,) representing the rotated vector.

    """
    # Apply the yaw, pitch, and roll rotations in sequence
    rotated_vector = yaw(yaw_angle).dot(pitch(pitch_angle).dot(roll(roll_angle).dot(vector)))

    return rotated_vector

def normalise(x, scale=1.0):
    """
    Normalizes a given numpy array by scaling it to a specified magnitude.

    Args:
    - x (np.ndarray): The input numpy array to be normalized.
    - scale (float): The desired magnitude to scale the array to (default=1.0).

    Returns:
    - np.ndarray: The normalized numpy array.

    Raises:
    - TypeError: If the input array is not a numpy array.
    """

    # Check if input is a numpy array
    if not isinstance(x, np.ndarray):
        raise TypeError("Input x must be a numpy array.")

    # Calculate the magnitude of the input array
    mag = magnitude(x)

    # Check if the magnitude is zero and return the input array if true
    if mag == 0.:
        return x
    else:
        # Scale the array to the desired magnitude and return the result
        return (scale / mag) * x

def magnitude(x):
    """
    Computes the magnitude (or length) of a three-dimensional vector x.
    
    Args:
        x (numpy.ndarray): A three-dimensional vector.
    
    Returns:
        float: The magnitude of the vector x.
    """
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

def bezier_interpolation(coords=None):

    """
    Interpolates a series of Bezier curves defined by input coordinates.

    Args:
    - coords (list): A list of tuples, each containing the 10 variables representing a point in space. 
    The variables are ordered as: x, y, z, curvature, torsion, diameter, length segment, generation, 
    bifurcation angle, and euclidean distance to parent node.

    Returns:
    - newNodes (ndarray): An array containing the new interpolated points.

    Raises:
    - TypeError: If the input is not a list.
    - ValueError: If the input list is empty.
    """

    # Store coordinates
    X = []
    Y = []
    Z = []
    diam = []

    try:
        for c in coords:
            X.append(float((c[0])))
            Y.append(float((c[1])))
            Z.append(float((c[2])))
            diam.append(float((c[5])))
    except:
        pass
    
    nodes = np.vstack((X, Y, Z, diam))

    # Define the Bezier curve
    vessel = np.array([])
    newNodes = np.array([])
    for i in range(0,nodes.shape[1]-1):

        if np.isnan(nodes[0,i]):
            _, idx = np.unique(vessel, axis=1,return_index=True)
            last = vessel[:,-1]
            vessel = vessel[:,np.sort(idx)]

            # if not np.array_equal(vessel[:,-1], last):
            #     vessel = np.hstack((vessel, last.reshape(-1,1)))
            if vessel.shape[1] > 2:

                daughters = np.arange(0, vessel.shape[1]-5, 5)
                for j in daughters:
                    if j == 0:
                        theChosenOne = vessel[:,j:(j+6)]
                    else:
                        theChosenOne = vessel[:,j:(j+6)]
                    points = theChosenOne
                    # points = interpolate_this(theChosenOne)
                    newNodes = generate_points(newNodes,points)

            vessel = np.array([])
        else:
            if vessel.size == 0:
                vessel = nodes[:,i].reshape(-1,1)
            else:
                vessel = np.hstack((vessel, nodes[:,i].reshape(-1,1)))

    return newNodes

def interpolate_this(vessel=None):
    """
    Interpolates a Bezier curve based on a given set of points.

    Args:
    - vessel (numpy.ndarray): Array of shape (n, m) representing the coordinates of the points. Each column
                              of the array should represent a point in n-dimensional space.

    Returns:
    - points (numpy.ndarray): Array of shape (n, k) representing the interpolated points, where k is the number
                              of points generated based on the interpolation.
    """
    
    # Generate a Bezier curve based on the given set of points
    curve = bz.Curve(vessel, degree=(vessel.shape[1]-1))
    
    # Evaluate the curve at evenly spaced points between 0 and 1
    t = np.linspace(0, 1, 10)
    points = curve.evaluate_multi(t)

    return points

def generate_points(newNodes=None, points=None, usenan=True):
    """
    Generate new points by appending them to existing nodes.

    Args:
    - newNodes: numpy.ndarray, the existing nodes to which the new points will be appended
    - points: numpy.ndarray, the new points to be added
    - usenan: bool, optional, whether to insert NaN values between the new points, default is True

    Returns:
    - newNodes: numpy.ndarray, the updated array of nodes with the new points appended

    """
    if newNodes.size == 0:
        # if there are no existing nodes, set the new nodes to be the points
        newNodes = points
    else:
        # otherwise, append the new points to the existing nodes
        newNodes = np.hstack((newNodes, points))

    if usenan:
        # if the user wants to insert NaN values between the new points,
        # create a vector of NaN values and append it to the updated nodes
        nanVec = np.empty((4,1))
        nanVec[:] = np.NaN
        newNodes = np.hstack((newNodes, nanVec))

    return newNodes

def addsalt_pepper(img, SNR):
    '''
    Adds salt and pepper noise to an image.

    Parameters:
    img (numpy.ndarray): A 3D numpy array representing an image.
    SNR (float): Signal to noise ratio. Probability of salt and pepper noise.

    Returns:
    numpy.ndarray: A 3D numpy array representing the input image with salt and pepper noise added.
    '''
    img_ = img.copy() # Create a copy of the input image to avoid modifying the original image
    c, h, w = img_.shape # Get the number of channels, height and width of the input image
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.]) # Generate a random mask to add noise to the image
    mask = np.repeat(mask, c, axis=0) # Copy the mask by channel to have the same shape as the input image
    img_[mask == 1] = 255 # Add salt noise
    img_[mask == 2] = 0 # Add pepper noise
    return img_

def norm_data(data):
    """
    Normalizes the input data.

    Args:
    data (numpy.ndarray): The input data to be normalized.

    Returns:
    numpy.ndarray: The normalized data.
    """
    # Subtract the minimum value of the data and divide by the range to normalize
    # the data between 0 and 1
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_volume(file, size=(600, 600, 700), ext='*.tif', datatype='uint8'):
    """
    Load a volume from a file, resample it to the given size and convert it to the specified data type.
    
    Parameters:
    file (str): path to the volume file
    size (tuple): target size of the volume in (z, y, x) format
    ext (str): file extension of the volume file
    datatype (str): target data type of the volume array
    
    Returns:
    numpy.ndarray: volume array with the specified target size and data type
    """
    # Load the volume and rescale the values to the range [0, 255]
    vol = np.array(itk.imread(file, itk.F))
    vol = (255 * norm_data(vol)).astype(datatype)
    
    return vol

def wrap_vessel(img, thresh=0):
    """
    Wrap a binary vessel segmentation image with a specified thickness.
    Any voxel with intensity value of 1 (representing the vessel) will be wrapped with 
    127 intensity value in a cube around it with radius = thresh.

    Args:
        img (numpy.ndarray): A 3D binary vessel segmentation image.
        thresh (int): The radius of the cube around the vessel to be wrapped. Default is 0.

    Returns:
        numpy.ndarray: A 3D binary vessel segmentation image with wrapped vessels.

    Raises:
        ValueError: If input image is not binary.

    """
    
    if not np.array_equal(np.unique(img), np.array([0,1])):
        raise ValueError("Input image is not binary.")
    
    # Get dimensions of input array
    d1, d2, d3 = img.shape
    
    # Find indices of non-zero voxels in the image
    idx = np.transpose(np.nonzero(img))
    
    # Create a new array to store the wrapped image data
    wrapped_img = np.copy(img)
    
    # Iterate over all non-zero voxels
    for i, j, k in idx:
        skip = False
        # Create a 3D boolean mask to identify neighboring voxels within threshold distance
        mask = np.sqrt(np.square(np.arange(-thresh, thresh+1, dtype=float)[:, np.newaxis, np.newaxis]) +
                       np.square(np.arange(-thresh, thresh+1, dtype=float)[:, np.newaxis]) +
                       np.square(np.arange(-thresh, thresh+1, dtype=float))) < thresh
        # Use boolean mask to identify neighbor voxels to update
        neighbor_indices = np.transpose(np.nonzero(mask))
        neighbor_indices -= np.array([thresh, thresh, thresh])  # Adjust indices to match center voxel
        neighbor_indices += np.array([i, j, k])  # Apply current voxel position
        
        # Iterate over all neighboring voxels
        for n_i, n_j, n_k in neighbor_indices:
            # Skip if the neighboring voxel is out of bounds
            if not (0 <= n_i < d1 and 0 <= n_j < d2 and 0 <= n_k < d3):
                continue
            # Skip if the neighboring voxel is already wrapped
            if wrapped_img[n_i, n_j, n_k] == 127:
                continue
            # Set the voxel to wrapped value and flag for skip
            wrapped_img[n_i, n_j, n_k] = 127
            skip = True
            break  # Exit neighbor loop if a voxel was wrapped
        
        if skip:
            break  # Exit non-zero voxel loop if a voxel was wrapped
    
    return wrapped_img
