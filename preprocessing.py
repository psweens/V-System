import cv2
import numpy as np

def resize_stacks(image, img_size=(700, 600, 600), target_size=None):
    '''
    Resizes a 3D numpy array of images to a target size.
    :param image: 3D numpy array of images.
    :param img_size: Tuple representing the size of the input image in (depth, height, width) format.
    :param target_size: Tuple representing the target size of the output image in (depth, height, width) format.
    :return: A numpy array of images resized to the target size.
    '''

    # Create empty numpy arrays for the input and output images.
    Y = np.empty([img_size[0], target_size[1], target_size[2]])
    X = np.empty([target_size[0], target_size[1], target_size[2]])

    # Interpolation method for resizing.
    imethod = cv2.INTER_NEAREST

    # Loop through the images in the input array and resize them to the target height and width.
    for j in range(0, img_size[0]):
        Y[j,] = cv2.resize(image[j,], (target_size[1], target_size[2]), interpolation=imethod)

    # Loop through the resized images and resize them to the target depth.
    for j in range(0, target_size[2]):
        X[:,:,j] = cv2.resize(Y[:,:,j], (target_size[1], target_size[0]), interpolation=imethod)

        # Apply a binary threshold to the output images.
        cv2.threshold(X[:,:,j], 127, 255, cv2.THRESH_BINARY)

    # Return the output images as a numpy array.
    return np.uint8(X)

def resize_volume(img, target_size=None):
    """
    Resizes the given image volume to the target size.
    
    Parameters:
    img (numpy.ndarray): A 3D numpy array representing the image volume to be resized.
    target_size (tuple of int): A tuple containing the target size of the resized image volume in the format (height, width, depth). 
    If None, the function returns the original image volume without resizing.
    
    Returns:
    numpy.ndarray: A 3D numpy array representing the resized image volume.
    """
    
    # Create empty arrays to store intermediate results
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='float32')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='float32')
    
    # Resize each slice along the z-axis
    for i in range(img.shape[2]):
        arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_AREA)
        
    # Threshold the intermediate array to convert it into a binary mask
    arr1[arr1 > 0.1] = 1.0
        
    # Resize each slice along the y-axis
    for i in range(target_size[0]):
        arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                  interpolation=cv2.INTER_AREA)
    
    # Threshold the final array
    arr2[arr2 > 0.1] = 1.0
    
    return arr2
