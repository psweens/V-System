import math
import random
import re
import numpy as np

from math import pi, sin, cos, sqrt
from utils import rotate, normalise, magnitude
DEGREES_TO_RADIANS = pi / 180

# A token is a single-character command, optionally followed by a parenthesised,
# comma-separated operand list: 'f(46.1,20.0)', '+(-37.5)', '[', '{'.
_TOKEN = re.compile(r"(?P<cmd>[A-Za-z\[\]{}+\-/])(?:\((?P<args>[^()]*)\))?")

# Lower-case terminals that move the turtle. Every other letter is a
# non-terminal left over from the recursion and draws nothing.
MOVE_COMMANDS = frozenset("fg")


def tokenise(turtle_program):
    """
    Splits a turtle program into (command, operands) pairs.

    Operands are converted with float(), so signs, decimals and exponent
    notation ('3.9e-05') are read as numbers rather than scanned as commands.

    Args:
    turtle_program (str): the L-system string to interpret.

    Yields:
    tuple: (command, operands) where command is a one-character string and
           operands is a tuple of floats (empty when the command has none).

    Whitespace between tokens is ignored.

    Raises:
    ValueError: on unexpected characters, an empty or non-numeric operand,
                or a non-finite operand.
    """
    pos = 0
    for match in _TOKEN.finditer(turtle_program):
        gap = turtle_program[pos:match.start()]
        if gap and not gap.isspace():
            raise ValueError(
                f"unexpected text {gap!r} at position {pos} of the turtle program")
        pos = match.end()
        args = match.group("args")
        if args is None:
            params = ()
        else:
            try:
                params = tuple(float(a) for a in args.split(","))
            except ValueError as exc:
                raise ValueError(
                    f"could not parse operands {args!r} of command "
                    f"{match.group('cmd')!r} at position {match.start()}") from exc
            if not all(math.isfinite(v) for v in params):
                raise ValueError(
                    f"non-finite operand in {match.group(0)!r} "
                    f"at position {match.start()}")
        yield match.group("cmd"), params
    tail = turtle_program[pos:]
    if tail and not tail.isspace():
        raise ValueError(
            f"unexpected text {tail!r} at position {pos} of the turtle program")


def _angle(command, params):
    if not params:
        raise ValueError(f"turn command {command!r} requires an angle operand")
    return params[0]

def branching_turtle_to_coords(turtle_program, d0, theta=20., phi=20.):

    '''
    Working with discontinuous paths i.e. tree formation.
    The program is read as tokens (see tokenise): a command character with an
    optional '(...)' operand list.
    'f(length[,diameter])' : move forward; a positive diameter replaces the current one
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

    for command, params in tokenise(turtle_program):
        x, y, z, alpha, beta, diam, lseg, dx, dy, dz = state

        if command in MOVE_COMMANDS:               # Move forward
            if not params:
                raise ValueError(
                    f"move command {command!r} requires a length operand")
            lseg = params[0]
            tdiam = params[1] if len(params) > 1 else 0.0
            dx, dy, dz = rotate(pitch_angle=beta*DEGREES_TO_RADIANS,
                                roll_angle=alpha*DEGREES_TO_RADIANS,
                                vector=normalise(np.array([x,y,z]),lseg))

            if tdiam > 0.0: diam = tdiam

            x += dx
            y += dy
            z += dz

            state = (x, y, z, alpha, beta, diam, lseg, dx, dy, dz)

            #  segment end
            yield state

        elif command == '+':                       # Turn clockwise
            state = (x, y, z, alpha + _angle(command, params), beta, diam, lseg, dx, dy, dz)

        elif command == '-':                       # Turn counterclockwise
            state = (x, y, z, alpha - _angle(command, params), beta, diam, lseg, dx, dy, dz)

        elif command == '/':                       # Pitch
            state = (x, y, z, alpha, beta + _angle(command, params), diam, lseg, dx, dy, dz)

        elif command == '[':                       # Remember current state
            saved_states.append(state)

        elif command == ']':                       # Return to previous state
            state = saved_states.pop()

            nanValues = []
            for i in range(stateSize): nanValues.append(float('nan'))
            yield tuple(nanValues)

            x, y, z, alpha, beta, diam, lseg, dx, dy, dz = state
            yield state

        elif command in '{}' or command.isupper():
            pass                                   # Grouping braces and non-terminals carry no geometry

        else:
            raise ValueError(f"unknown turtle command {command!r}")


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