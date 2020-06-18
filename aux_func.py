import numpy as np
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
    return (scale / magnitude(x)) * x
    
def magnitude(x):
    return npm.sqrt(x[0]**2 + x[1]**2 + x[2]**2)