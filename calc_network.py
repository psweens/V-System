import random
import numpy as np
import matplotlib.pyplot as plt

from math import pi, sin, cos, isnan
DEGREES_TO_RADIANS = pi / 180

def branching_turtle_to_coords(turtle_program, theta=22.5, phi=22.5, precision=1):
    
    '''
    Working with discontinuous paths i.e. tree formation
    '+' : postive rotation by (deg)
    '-' : negative rotation by (deg)
    '[' : Save state of turtle by pushing to stack (location and angle)
    ']' : Restore the turtle to state of the last ']' by retrieving the state
            from the stack
    '''
    saved_states = list()
    state = (0, 0, 0, 90, 90, 5)
    yield (0, 0, 0, 0)
    
    index = 0
    vessel = 0
    for command in turtle_program:
        x, y, z, alpha, beta, diam = state
        
        if command == '{':
            vessel == 1
        
        elif command == '}':
            vessel = 0
        
        if command.lower() in 'abcdefghijs':        # Move forward (matches a-j and A-J)
            # segment start
            lseg = 5
            # eval_brackets(index, turtle_program)
            # print(command)
            # print('%f %f %f %f' %(x, y, angle, diam))
            state = (x - lseg*cos(alpha * DEGREES_TO_RADIANS),
                     y + lseg*sin(alpha * DEGREES_TO_RADIANS),
                     z + lseg*cos(beta * DEGREES_TO_RADIANS),                     
                     alpha, 
                     beta,
                     diam + 5)

            #  segment end
            # print(state)
            if command.islower():# Add a break in the line if command matches a-j (lowercase)
                lseg, diam = eval_brackets(index, turtle_program)
                # print('%f %f' %(lseg, diam))
                # yield (float('nan'), float('nan'), float('nan'), float('nan'))

            yield (state[0], state[1], state[2], state[5])

        elif command == '+':                       # Turn clockwise
            theta = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha + theta, beta, diam)

        elif command == '-':                       # Turn counterclockwise
            theta = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha - theta, beta, diam)
            
        elif command == '/':
            phi = eval_brackets(index, turtle_program)
            state = (x, y, z, alpha, beta + posneg()*phi, diam)

        elif command == '[':                       # Remember current state
            saved_states.append(state)

        elif command == ']':                       # Return to previous state
            state = saved_states.pop()
            yield (float('nan'), float('nan'), float('nan'), float('nan'))
            x, y, z, alpha, beta, diam = state
            yield (x, y, z, diam)

        index += 1
        # Note: We silently ignore unknown commands
        
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