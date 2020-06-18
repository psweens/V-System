import random
import numpy as np

from math import pi, sin, cos, sqrt
from aux_func import rotate, normalise, magnitude
DEGREES_TO_RADIANS = pi / 180

def branching_turtle_to_coords(turtle_program, theta=20., phi=20.):
    
    '''
    Working with discontinuous paths i.e. tree formation
    '+' : postive rotation by (deg)
    '-' : negative rotation by (deg)
    '[' : Save state of turtle by pushing to stack (location and angle)
    ']' : Restore the turtle to state of the last ']' by retrieving the state
            from the stack
    '''
    saved_states = list()
    state = (0, 0, 0, 0, 0, 20.)
    yield  (0,0,0,0,0,0)
    
    index = 0
    lseg = 5.
    origin = (0, 0, 0)
    surface = 80
    tx = 0
    ty = 0
    tz = 0

    for command in turtle_program:
        # print(command)
        x, y, z, alpha, beta, diam = state
    
        
        if command.lower() in 'abcdefghijs':        # Move forward (matches a-j and A-J)
            # segment start
            
            if command.islower():# Add a break in the line if command matches a-j (lowercase)
                lseg, diam = eval_brackets(index, turtle_program)
            
                if not magnitude(np.array([x,y,z])) == 0.0:
                    tx, ty, tz = rotate(p=beta*DEGREES_TO_RADIANS,
                                        r=alpha*DEGREES_TO_RADIANS,
                                        v=normalise(np.array([x,y,z]),lseg))
                    x += tx
                    y += ty
                    z += tz
                    
                else:
                    y = lseg

            state = (x, y, z, alpha, beta, diam)
            
            #  segment end
            yield (state[0], state[1], state[2], state[3], state[4], state[5])

        elif command == '+':                       # Turn clockwise
            phi = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha + phi, beta, diam)

        elif command == '-':                       # Turn counterclockwise
            phi = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha - phi, beta, diam)
            
        elif command == '/':
            theta = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha, beta + theta, diam)

        elif command == '[':                       # Remember current state
            saved_states.append(state)

        elif command == ']':                       # Return to previous state
            state = saved_states.pop()
            yield (float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'))
            x, y, z, alpha, beta, diam = state
            yield (x, y, z, alpha, beta, diam)

        index += 1
        # Note: silently ignore unknown commands
        
def eval_brackets(index, turtle):
    
    a = ''
    b = ''
    double = 0
    for i in range(index+2, len(turtle)):
        if turtle[i] == ')':
            
            if double == 1:
                return eval(a), eval(b)
            else:
                return eval(a)
        elif turtle[i] == ',':
            double = 1
        else:
            if double == 0:
                a = a + str(turtle[i])
            else:
                b = b + str(turtle[i])
            
def posneg():
    return 1 if random.random() < 0.5 else -1

def raddist(origin, location, radius=80):
    return bool(sqrt(pow(origin[0]-location[0],2) + pow(origin[1]-location[1],2) +
                pow(origin[2]-location[2],2)) > radius)

# def proximity(state,p0):
    

