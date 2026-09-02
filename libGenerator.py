import math
import numpy as np
import random

default={"k": 3,
          "epsilon": 10 , # Proportion between length & diameter
          "randmarg": 0.2 , # Relative half-width of the segment-length distribution (fraction of epsilon*d0)
          "sigma": 5, # Determines type deviation for Gaussian distributions
          "stochparams": True} # Whether the generated parameters will also be stochastic

def setProperties(properties):
    """
    Sets global property values based on the input dictionary.

    Args:
        properties (dict): A dictionary containing the properties and their values.

    Returns:
        None

    """
    if properties is None:
        properties = default

    global k, epsilon, randmarg, sigma, stochparams

    k = properties['k']
    epsilon = properties['epsilon']
    randmarg = properties['randmarg']
    if not 0.0 <= randmarg < 1.0:
        raise ValueError(
            f"randmarg must be a fraction in [0, 1), got {randmarg!r}: it is the relative "
            "half-width of the segment-length distribution, not an absolute margin")
    sigma = properties['sigma']
    stochparams = properties['stochparams']

def calParam(text, params):
    '''
    Calculates the value within the parentheses before analyzing the text. For example, f('co / 4'), f requires 
    the value of 'co'. 
    
    Args:
    text (str): A string containing the mathematical expression to be evaluated
    params (dict): A dictionary containing parameter names (as keys) and their values (as values)
    
    Returns:
    str: A string representation of the evaluated expression 
    '''

    txt = text[:]
    for i in params: txt = txt.replace(i, str(params[i]))
        
    return str(params['co'] / eval(txt))

def calBifurcation(d0):
    '''
    Calculates the diameters and angles of bifurcation given an input diameter
    
    Args:
    d0 (float): input diameter
    
    Returns:
    resp (dict): a dictionary containing the calculated values for d1, d2, d0, th1, th2, and co
    '''

    resp = {}

    if not d0 > 0:
        raise ValueError(f"bifurcation requires a positive parent diameter, got {d0!r}")

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
    a1 = math.acos(xtmp / xtmpb)

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha * alpha * alpha) ** (2.0/3))
    a2 = math.acos(xtmp / xtmpb)

    resp["d1"] = d1
    resp["d2"] = d2
    resp["d0"] = d0
    resp["th1"] = a1 * 180 / math.pi
    resp["th2"] = a2 * 180 / math.pi
    resp["co"] = getLength(d0)

    return resp

def getLength(d0):
    """
    Returns the length of the branch based on the diameter of the parent branch.

    The length is epsilon * d0 scaled by a uniform factor in
    [1 - randmarg, 1 + randmarg]. Because the margin is relative, the length
    stays positive and proportional to the diameter at every generation.

    Parameters:
    d0 (float): The diameter of the parent branch.

    Returns:
    float: The length of the branch.
    """
    c0 = d0 * epsilon
    return c0 * np.random.uniform(1.0 - randmarg, 1.0 + randmarg)
