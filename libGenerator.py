import ast
import random
import numpy as np
from numpy import math as npm

default={"k": 3,
          "epsilon": 10 , # Proportion between length & diameter
          "randmarg": 3 , # Randomness margin between length & diameter
          "sigma": 5, # Determines type deviation for Gaussian distributions
          "stochparams": True} # Whether the generated parameters will also be stochastic

def setProperties(properties):
    """
    Sets global property values based on the input dictionary.

    Args:
        properties (dict): A dictionary containing the properties and their values.

    Returns:
        None

    Raises:
        None
    """

    # Check if properties is None and assign default value
    if properties is None:
        properties = default

    # Define global variables based on the input dictionary
    global k, epsilon, randmarg, sigma, stochparams
    k = properties['k']
    epsilon = properties['epsilon']
    randmarg = properties['randmarg']
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
    for i in params:
        txt = txt.replace(i, str(params[i]))
    try:
        result = ast.literal_eval(txt)
        return str(params['co'] / result)
    except:
        return 'Error: Invalid expression'

def calBifurcation(d0):

'''
    Calculates the diameters and angles of bifurcation given an input diameter
    
    Parameters:
    d0 (float): input diameter
    
    Returns:
    resp (dict): a dictionary containing the calculated values for d1, d2, d0, th1, th2, and co
    '''
    resp = {}
    
    # Calculate optimal diameter and adjust if stochastic parameter is set to True
    dOpti = d0 / 2 ** (1.0 / k)
    if stochparams:
        d1 = abs(np.random.normal(dOpti, dOpti / sigma))
    else:
        d1 = dOpti # Optimal diameter
    
    # Ensure that d1 is not greater than d0
    if d1 >= d0:
        d1 = dOpti
        
    # Calculate second diameter using the first diameter and the rate of symmetry between daughters
    alpha = d2 / d1
    d2 = (d0 ** k - d1 ** k) ** (1.0 / k)
    
    # Equations which mimic bifurcation angles in the human body (Liu et al. (2010) and Zamir et al. (1988))
    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha * alpha * alpha ) ** (2.0 / 3))
    a1 = npm.acos(xtmp / xtmpb)

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha * alpha * alpha) ** (2.0/3))
    a2 = npm.acos(xtmp / xtmpb)
    
    # Store calculated values in a dictionary and return
    resp["d1"] = d1
    resp["d2"] = d2
    resp["d0"] = d0
    resp["th1"] = a1 * 180 / npm.pi
    resp["th2"] = a2 * 180 / npm.pi
    resp["co"] = getLength(d0)
    
    return resp

def getLength(d0):
    """
    Returns the length of the branch based on the diameter of the parent branch.

    Parameters:
    d0 (float): The diameter of the parent branch.

    Returns:
    float: The length of the branch.
    """
    c0 = d0 * epsilon # calculate c0 based on the parent branch diameter and epsilon
          
    return np.random.uniform(c0, randmarg * 2)
