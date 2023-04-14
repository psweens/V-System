import random
import numpy as np

from math import pi, sin, cos, sqrt
from utils import rotate, normalise, magnitude
DEGREES_TO_RADIANS = pi / 180

def branching_turtle_to_coords(turtle_program, d0, theta=20., phi=20.):

    '''
    Working with discontinuous paths i.e. tree formation
    '+' : postive rotation by (deg)
    '-' : negative rotation by (deg)
    '[' : Save state of turtle by pushing to stack (location and angle)
    ']' : Restore the turtle to state of the last ']' by retrieving the state
            from the stack
            
    Args:
    d0 (float): initial diameter of the turtle path
    theta (float): angle in degrees for rotating around the y-axis (default: 20 degrees)
    phi (float): angle in degrees for rotating around the z-axis (default: 20 degrees)
    
    Returns
    tuple: A list of tuples which provide the coordinates of the branches
    '''
    saved_states = list()
    stateSize = 10
    dx = 0
    dy = 0
    dz = 0
    lseg = 1.
    rim = 400
    
    startidx = 3#random.randint(1,3)
    if startidx == 1:
        state = (1., 0.1, 0.1, 0, 0, d0, lseg, dx, dy, dz)
    elif startidx == 2:
        state = (0.1, 1., 0, 0, 0, d0, lseg, dx, dy, dz)
    else:
        state = (0.1, 0.1, 1., 0, 0, d0, lseg, dx, dy, dz)
    
    yield  state

    index = 0
    origin = (0.1, 0.1, 1.)

    for command in turtle_program:
        x, y, z, alpha, beta, diam, lseg, dx, dy, dz = state


        if command.lower() in 'abcdefghijs':        # Move forward (matches a-j and A-J)
            # segment start


            if command.islower():
                lseg, tdiam = eval_brackets(index, turtle_program)
                dx, dy, dz = rotate(pitch_angle=beta*DEGREES_TO_RADIANS,
                                    roll_angle=alpha*DEGREES_TO_RADIANS,
                                    vector=normalise(np.array([x,y,z]),lseg))
                
                if tdiam > 0.0: diam = tdiam

                #dx, dy, dz, alpha = proximity(state,origin,rim)

                x += dx
                y += dy
                z += dz

            state = (x, y, z, alpha, beta, diam, lseg, dx, dy, dz)

            #  segment end
            yield state

        elif command == '+':                       # Turn clockwise
            phi, _ = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha + phi, beta, diam, lseg, dx, dy, dz)

        elif command == '-':                       # Turn counterclockwise
            phi, _ = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha - phi, beta, diam, lseg, dx, dy, dz)

        elif command == '/':
            theta, _ = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha, beta + theta, diam, lseg, dx, dy, dz)

        elif command == '[':                       # Remember current state
            saved_states.append(state)

        elif command == ']':                       # Return to previous state
            state = saved_states.pop()

            nanValues = []
            for i in range(stateSize): nanValues.append(float('nan'))
            yield tuple(nanValues)

            x, y, z, alpha, beta, diam, lseg, dx, dy, dz = state
            yield state

        index += 1

def eval_brackets(index, turtle):

    """
    Extracts values within brackets in the turtle program and evaluates them to return tuple of values.
    
    Args:
    index (int): integer value of the index position within turtle string.
    turtle (str): a string containing the turtle program.
    
    Returns
    tuple: A tuple of two floats containing the values within the brackets. If only one value, second value is 0.0.
    """

    a = ''                          # initialize string variable to hold first value
    b = ''                          # initialize string variable to hold second value
    double = 0                      # initialize integer variable to 0 for checking if there are two values
    neg1 = 0                        # initialize integer variable to 0 for checking if the first value is negative
    neg2 = 0                        # initialize integer variable to 0 for checking if the second value is negative

    for i in range(index+2, len(turtle)):      # loop through turtle string starting from the index position+2
        if turtle[i] == ')':            # stop if end of bracket is reached
            break
        elif turtle[i] == ',':          # set double flag if comma is reached
            double = 1
        elif turtle[i] == '-':          # set neg1 flag if minus sign is reached before the comma
            if double == 0: neg1 = 1
            else: neg2 = 1;
        elif not (turtle[i] == '(' or turtle[i] == '+'):    # add digit to string variables if not brackets or plus sign
            if double == 0:
                a = a + str(turtle[i])
            else:
                b = b + str(turtle[i])
                
    if double == 1:                 # if double flag is set, evaluate both string variables as floats and return tuple
        aa = eval(a)
        bb = eval(b)
        if neg1 == 1: aa *= -1      # if neg1 flag is set, negate first value
        if neg2 == 1: bb *= -1      # if neg2 flag is set, negate second value
        return aa, bb
    else:                           # if double flag is not set, evaluate first string variable as float and return tuple
        aa = eval(a)
        if neg1 == 1: aa *= -1      # if neg1 flag is set, negate first value
        return aa, 0.0

def randomposneg():
    """
    Return either 1 or -1 with a 50/50 probability.

    Returns:
    int: Either 1 or -1 with a 50/50 probability.
    """
    return 1 if random.random() < 0.5 else -1

def raddist(origin, location, shell=80, core=False):
    """
    Calculate the distance between two points and check if it falls within a specified range.

    Args:
    origin (list): A list of x, y, and z coordinates for the origin point.
    location (list): A list of x, y, and z coordinates for the location point.
    shell (float): The distance range from the origin point. Defaults to 80.
    core (bool): A boolean indicating whether to check if the location point is inside or outside of the shell.
        If False (default), checks if the location point is inside the shell.
        If True, checks if the location point is outside the shell.

    Returns:
    bool: A boolean value indicating whether the location point is inside or outside of the specified range.
    """
    distance = sqrt(pow(origin[0]-location[0], 2) + pow(origin[1]-location[1], 2) + pow(origin[2]-location[2], 2))
    
    if not core:
        # check if location is inside the shell
        return distance < shell
    else:
        # check if location is outside the shell
        return distance > shell

def proximity(state: tuple, origin: np.ndarray, rim: float) -> tuple:
    """
    Calculates the proximity of a point to a given origin within a specified rim.

    Args:
    - state (tuple): A tuple containing six floats representing the (x, y, z) coordinates,
                     alpha (in degrees), beta (in degrees), and diam of the point.
    - origin (np.ndarray): A 1D numpy array of three floats representing the (x, y, z)
                           coordinates of the origin point.
    - rim (float): A float representing the maximum distance from the origin point within
                   which the point can be considered "close" to the origin.

    Returns:
    - A tuple containing four floats:
      - dx (float): The x-coordinate difference between the point and the origin.
      - dy (float): The y-coordinate difference between the point and the origin.
      - dz (float): The z-coordinate difference between the point and the origin.
      - alpha (float): The new value of alpha after undergoing random perturbations.
    """
    # Unpack the state tuple to get the x, y, z, alpha, beta, and diam values of the point.
    x, y, z, alpha, beta, diam, lseg, _, _, _ = state

    # Define the number of points and the range of alpha, beta, and yaw angles to use.
    points = 6
    roll = np.linspace(alpha, 0., points) * DEGREES_TO_RADIANS
    pitch = np.linspace(beta, 0., points) * DEGREES_TO_RADIANS
    yaw = np.linspace(70., 0., points) * DEGREES_TO_RADIANS

    # Generate random perturbations for alpha and beta angles and update the roll and pitch arrays.
    orbitsuccess = False
    while not orbitsuccess:
        # Use the pointCycle function to calculate the point's new position.
        orbitsuccess, dx, dy, dz = pointCycle(np.array([x, y, z]), lseg, origin,
                                              points, pitch, roll, rim, yaw, core=False)

        # Perturb the alpha angle and update the roll array.
        alphaOld = alpha
        alpha = alpha + random.uniform(-25, 25)
        roll = np.linspace(alpha, alphaOld, int(points)) * DEGREES_TO_RADIANS

        # Perturb the beta angle and update the pitch array.
        betaOld = beta
        beta = beta + random.uniform(-25, 25)
        pitch = np.linspace(betaOld, beta, int(points)) * DEGREES_TO_RADIANS

    # Return the differences in x, y, and z coordinates, as well as the new value of alpha.
    return dx, dy, dz, alpha

def posneg(value: float) -> float:
    """
    Returns 1 if the input value is greater than or equal to 0, else -1.

    Parameters:
    value (float): A numerical value.

    Returns:
    float: 1 if the input value is greater than or equal to 0, else -1.
    """
    if value >= 0.:
        return 1.
    else:
        return -1.