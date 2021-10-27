import numpy as np
import cv2
import os
import itk
import h5py
import math
from skimage import io

def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_volume(file, size=(600,600,700), ext='*.tif', datatype='uint8'):
    vol = np.array(itk.imread(file, itk.F))
    vol = (255*norm_data(vol)).astype(datatype)
        
    return vol

def wrap_vessel(img,thresh=0):
    
    d1 = img.shape[0]
    d2 = img.shape[1]
    d3 = img.shape[2]
    idx = np.nonzero(img)
    for (i,j,k) in zip(*idx):
        skip = False
        for m in range(-1,2):
            for l in range(-1,2):
                for n in range(-1,2):
                    if (i+m-1 < d1 and j+l-1 < d2 and k+n-1 < d3 and i+m-1 > 0 and j+l-1 > 0 and k+n-1 > 0):
                        if img[i+m-1, j+l-1, k+n-1] == 0:
                            for x in range(-thresh,thresh+1):
                                for y in range(-thresh,thresh+1):
                                    for z in range(-thresh,thresh+1):
                                        if math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2) < float(thresh):
                                            if (i+m-1+x < d1 and j+l-1+y < d2 and k+n-1+z < d3 and i+m-1+x > 0 and j+l-1+y > 0 and k+n-1+z > 0):
                                                if img[i+m-1+x, j+l-1+y, k+n-1+z] == 0:
                                                   img[i+m-1+x, j+l-1+y, k+n-1+z] = 127;
                                                   skip = True
                    break
                if skip: break
            if skip: break
    
    return img

path = '/media/sweene01/SSD/VA_Paper_Datasets/iLastik/rmvdBckg_kWaveProcessed_FilteredLnetsTIFF/Testing_Predictions/'
files = os.listdir(path)
files = [file for file in files if file.endswith('.tiff')]

for i in range(len(files)):
    fullfile = os.path.join(path,files[i])
    imgs = load_volume(fullfile)
    imgs = 255*norm_data(imgs)
    imgs = abs(imgs - 255)
    # imgs = wrap_vessel(imgs,thresh=3)
    # for j in range(imgs.shape[0]):
    #     for k in range(imgs.shape[2]):
    #         for n in range(3):
    #             if imgs[j,n,k] == 0:
    #                imgs[j,n,k] += 125 
    #             if imgs[j,256+n,k] == 0:
    #                imgs[j,256+n,k] += 125 
    #             if imgs[j,511-n,k] == 0:
    #                imgs[j,511-n,k] += 125 

    
    io.imsave(os.path.join(path,'test',files[i]), imgs.astype('uint8'), bigtiff=False)
    
    # newfile = os.path.join(path,'test',files[i])
    # newfile = (os.path.splitext(newfile))[0] + '.h5'
    # h5 = h5py.File(newfile, 'w')
    # h5.create_dataset('dataset_1', data=imgs)
    # h5.close()

