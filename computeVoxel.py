'''
Cohen, D. Voxel traversal along a 3D line.
from Heckbert, P.S. Graphics Gems IV. Morgan Kaufmann.

Visits Subroutine visits all voxels along the line
segment (x,y,z) and (x+dx,y+dy,z+dz).
'''

import numpy as np
from utils import generate_points

def process_network(data,tVol):

    #  Find non-NaN min/max values across data leaving a 10% margin in case of
    #  vessel/tissue boundary contact
    dmin = np.nanmin(data,axis=1)*1.1
    dmax = np.nanmax(data,axis=1)*1.1

    newarray = np.array([])
    img_stack = np.empty([tVol[0], tVol[1], tVol[2]], dtype=int)
    for i in range(0,data.shape[1]):
        if not np.isnan(data[0,i]):
            # Discretise data into defined tissue volume
            tempvec = discretisation(data[:,i],dmin,dmax,tVol)
        else:
            tempvec = data[:,i].reshape(-1,1)

        # Generating & organising point data into array
        newarray = generate_points(newarray,tempvec,usenan=False)


    # Calculate diameter and pass into function to find voxels transversed by
    # by segment between two points
    j = 0
    for i in range(0,newarray.shape[1]-1):
        j += 1
        if not np.isnan(newarray[0,i]):
            if not np.isnan(newarray[0,j]):
                diam = 0.5*(newarray[3,i] + newarray[3,j])
                transversal(newarray[0,i],newarray[1,i],newarray[2,i],
                            newarray[0,j],newarray[1,j],newarray[2,j],
                            diam,img_stack,tVol)

    return check_boundary(img_stack)

def check_boundary(image):

    image[0,:,:] = 0
    image[image.shape[0]-1,:,:] = 0

    image[:,0,:] = 0
    image[:,image.shape[1]-1,:] = 0

    image[:,:,0] = 0
    image[:,:,image.shape[2]-1] = 0

    return image

def transversal(x,y,z,dx,dy,dz,diam,img_stack=None,tVol=(600,600,700),
                resolution=(1,1,1)):

    dx = dx - x
    dy = dy - y
    dz = dz - z

    sx = np.sign(dx)
    sy = np.sign(dy)
    sz = np.sign(dz)

    ax = abs(dx)
    ay = abs(dy)
    az = abs(dz)

    bx = 2*ax
    by = 2*ay
    bz = 2*az

    exy = ay - ax
    exz = az - ax
    ezy = ay - az

    n = ax + ay + az
    # print('next')
    r = int(np.floor(0.5*diam))
    if x + r >= tVol[0]:
        mx = tVol[0] - x
    else:
        mx = r
    if y + r >= tVol[1]:
        my = tVol[1] - y
    else:
        my = r
    if z + r >= tVol[2]:
        mz = tVol[2] - z
    else:
        mz = r

    # 'diamVoxels(...)' finds voxels within the tube/vessel
    while (n > 0):

        if exy < 0:
            if exz < 0:
                x += sx
                exy += by
                exz += bz
                if int(x) >= 0 & int(x) < tVol[0]:
                    img_stack = diamVoxels(x,y,z,0,r,img_stack,tVol,res=resolution)
            else:
                z += sz
                exz -= bx
                ezy += by
                if int(z) >= 0 & int(z) < tVol[2]:
                    img_stack = diamVoxels(x,y,z,2,r,img_stack,tVol,res=resolution)
        else:
            if ezy < 0:
                z += sz
                exz -= bx
                ezy += by
                if int(z) >= 0 & int(z) < tVol[2]:
                    img_stack = diamVoxels(x,y,z,2,r,img_stack,tVol,res=resolution)
            else:
                y += sy
                exy -= bx
                ezy -= bz
                if int(y) >= 0 & int(y) < tVol[1]:
                    img_stack = diamVoxels(x,y,z,1,r,img_stack,tVol,res=resolution)

        n -= 1



def discretisation(p1,dmin,dmax,tVol):
    x = np.floor(tVol[0]*(p1[0]-dmin[0])/(dmax[0]-dmin[0]))
    y = np.floor(tVol[1]*(p1[1]-dmin[1])/(dmax[1]-dmin[1]))
    z = np.floor(tVol[2]*(p1[2]-dmin[2])/(dmax[2]-dmin[2]))
    return np.array([x,y,z,p1[3]]).reshape(-1,1)


def diamVoxels(x,y,z,idx,r,img_stack,tVol,res):
    '''
    NEED TO INCLUDE SIZE OF PIXELS RELATIVE TO DIAMETER
    '''
    t0 = tVol[0]
    t1 = tVol[1]
    t2 = tVol[2]

    r0 = res[0]
    r1 = res[1]
    r2 = res[2]
    if idx == 0:
        for i in range(-r,r+1):
            if (int(y+i) < t1) & (int(y+i) >= 0):
                for j in range(-r,r+1):
                    if (int(z+j) < t2) & (int(z+j) >= 0):
                        if pow(pow(i*r1,2) + pow(j*r2,2),0.5) <= r:
                            img_stack[int(x-1),int(y+i-1),int(z+j-1)] = 1
    elif idx == 1:
        for i in range(-r,r+1):
            if (int(x+i) < t0) & (int(x+i) >= 0):
                for j in range(-r,r+1):
                    if (int(z+j) < t2) & (int(z+j) >= 0):
                        if pow(pow(i*r0,2) + pow(j*r2,2),0.5) <= r:
                            img_stack[int(x+i-1),int(y-1),int(z+j-1)] = 1
    else:
        for i in range(-r,r+1):
            if (int(x+i) < t0) & (int(x+i) >= 0):
                for j in range(-r,r+1):
                    if (int(y+j) < t1) & (int(y+j) >= 0):
                        if pow(pow(i*r0,2) + pow(j*r1,2),0.5) <= r:
                            img_stack[int(x+i-1),int(y+j-1),int(z-1)] = 1

    return img_stack

def findVessel(i,data):

    vessel = np.array([])
    stop = False

    while not stop:

        if not np.isnan(data[0,i]):
            if vessel.size == 0:
                vessel = data[:,i].reshape(-1,1)
            else:
                vessel = np.hstack((vessel, data[:,i]))
            i += 1;
        else:
            stop = True

    return i, vessel
