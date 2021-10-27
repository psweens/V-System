import cv2
import numpy as np

def resize_stacks(image, img_size=(700,600,600), target_size=None):

    Y = np.empty([img_size[0], target_size[1], target_size[2]])
    X = np.empty([target_size[0], target_size[1], target_size[2]])

    # Interpolation method
    imethod = cv2.INTER_NEAREST

    for j in range(0, img_size[0]):

        Y[j,] = cv2.resize(image[j,], (target_size[1], target_size[2]),
                              interpolation = imethod)


    for j in range(0, target_size[2]):

        X[:,:,j] = cv2.resize(Y[:,:,j], (target_size[1], target_size[0]),
                           interpolation = imethod)

        cv2.threshold(X[:,:,j], 127, 255, cv2.THRESH_BINARY)



    return np.uint8(X)

def resize_volume(img, target_size=None):
    
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='float32')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='float32')
    
    for i in range(img.shape[2]):
        arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_AREA)
        
    arr1[arr1 > 0.1] = 1.0
        
    for i in range(target_size[0]):
        arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                  interpolation=cv2.INTER_AREA)
    
    mval = np.mean(arr1)
    print('Mean value ... %f' %mval)
    # for i in range(arr2.shape[2]):
        # _, arr1[:,:,i] = cv2.threshold(arr2[:,:,i], mval, 127, cv2.THRESH_BINARY)
        
    arr2[arr2 > 0.1] = 1.0
    return arr2