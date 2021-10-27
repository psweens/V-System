import numpy as np
import random
from numpy import math as npm

default={"k": 3,
          "epsilon": 10 , # Proportion between length & diameter
          "randmarg": 3 , # Randomness margin between length & diameter
          "sigma": 5, # Determines type deviation for Gaussian distributions
          "stochparams": True} # Whether the generated parameters will also be stochastic

def setProperties(properties):

    '''
    Establishes property values according to a dictionary given as input
    '''
    if properties == None:
        properties = default

    global k, epsilon, randmarg, sigma, stochparams

    k = properties['k']
    epsilon = properties['epsilon']
    randmarg = properties['randmarg']
    sigma = properties['sigma']
    stochparams = properties['stochparams']

def calParam(text, params):

    '''
    Calculates the value within the paratheses before analysing the text.
    For example, f('co / 4'), f requires the value of 'co'
    '''

    txt = text[:]
    for i in params: txt = txt.replace(i, str(params[i]))
    return str(params['co'] / eval(txt))

def calBifurcation(d0):

    resp = {}

    dOpti = d0 / 2 ** (1.0 / k)
    if stochparams: d1 = abs(np.random.normal(dOpti, dOpti / sigma))
    else: d1 = dOpti # Optimal diameter

    if d1 >= d0: d1 = dOpti # Elimate possibility of d1 being greater than d0

    d2 = (d0 ** k - d1 ** k) ** (1.0 / k) # Calculate second diameter
    # alpha = abs(np.random.uniform(1., 0.25)) * (d2 / d1) # Rate of symmetry of daughters (=1 symmetrical ?)
    alpha = d2 / d1

    '''
    Equations which mimic bifurcation angles in the human body
    Liu et al. (2010) and Zamir et al. (1988)
    '''
    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha * alpha * alpha ) ** (2.0 / 3))
    a1 = npm.acos(xtmp / xtmpb)

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha * alpha * alpha) ** (2.0/3))
    a2 = npm.acos(xtmp / xtmpb)

    resp["d1"] = d1
    resp["d2"] = d2
    resp["d0"] = d0
    resp["th1"] = a1 * 180 / npm.pi
    resp["th2"] = a2 * 180 / npm.pi
    resp["co"] = getLength(d0)

    return resp

def getLength(d0):
    c0 = d0 * epsilon
    # abs(np.random.normal(50,10))
    return np.random.uniform(c0 - randmarg, c0 + randmarg)