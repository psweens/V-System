import numpy as np
import bezier as bz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import math as npm

def yaw(angle):

    A = np.array([[npm.cos(angle), -npm.sin(angle), 0.0],
                   [npm.sin(angle), npm.cos(angle), 0.0],
                   [0.0, 0.0, 1.0]])

    return A

def pitch(angle):

    A = np.array([[npm.cos(angle), 0.0, npm.sin(angle)],
                   [0.0, 1.0, 0.0],
                   [-npm.sin(angle), 0.0, npm.cos(angle)]])

    return A

def roll(angle):

    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, npm.cos(angle), -npm.sin(angle)],
                  [0.0, npm.sin(angle), npm.cos(angle)]])

    return A

def rotate(y=0.0,p=0.0,r=0.0,v=None):
    return yaw(y).dot(pitch(p).dot(roll(r).dot(v)))

def normalise(x,scale=1.0):
    if magnitude(x) == 0.:
        return x
    else:
        return (scale / magnitude(x)) * x

def magnitude(x):
    return npm.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

def bezier_interpolation(coords=None):

    # Store coordinates
    X, Y, Z, _, _, diam, lseg, _, _, _ = zip(*coords)
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

    curve = bz.Curve(vessel, degree=(vessel.shape[1]-1))
    t = np.linspace(0, 1, 10)
    # curve = curve.elevate()
    points = curve.evaluate_multi(t)

    return points

def generate_points(newNodes=None,points=None, usenan=True):

    if newNodes.size == 0:
        newNodes = points
    else:
        newNodes = np.hstack((newNodes, points))

    if usenan:
        nanVec = np.empty((4,1))
        nanVec[:] = np.NaN
        newNodes = np.hstack((newNodes, nanVec))

    return newNodes

def addsalt_pepper(img, SNR):
    '''
    Taken from https://www.programmersought.com/article/3363136769/#:~:text=Salt%20and%20pepper%20noise%2C%20also,noise%20(based%20on%20python).
    '''
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0) # Copy by channel to have the same shape as img
    img_[mask == 1] = 255 # salt noise
    img_[mask == 2] = 0 # 
    return img_
