import random
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')  # Use some nicer default colors

def plot_coords(coords, bare_plot=False):
    
    '''
    Takes a list of coordinates and coverts to correct format for matplotlib
    '''
    
    if bare_plot:
        # Turns off the axis markers.
        plt.axis('off')
    # Ensures equal aspect ratio.
    plt.axes().set_aspect('equal', 'datalim')
    # Converts a list of coordinates into 
    # lists of X and Y values, respectively.
    X, Y = zip(*coords)
    # Draws the plot.
    plt.plot(X, Y);
    
def print_coords(coords):
    for (x, y) in coords:
        if isnan(x):
            print('<gap>')
        else:
            print('({:.2f}, {:.2f})'.format(x, y))
    
from math import pi, sin, cos, isnan
DEGREES_TO_RADIANS = pi / 180

def transform_sequence(sequence, transformations):
   
    '''
    Represent transformation rule as an entry with the character key and 
    replacement string as a value e.g.
    a -> aba
    c -> bb
    is represented as {'a': 'aba', 'c': 'bb'}
    '''
    return ''.join(transformations.get(c, c) for c in sequence)

def transform_multiple(sequence, transformations, iterations):
    
    '''
    Apply the transform sequence multiple times
    '''
    for _ in range(iterations):
        sequence = transform_sequence(sequence, transformations)
    return sequence

def branching_turtle_to_coords_orig(turtle_program, turn_amount=45):
    
    '''
    Working with discontinuous paths i.e. tree formation
    '+' : postive rotation by (deg)
    '-' : negative rotation by (deg)
    '[' : Save state of turtle by pushing to stack (location and angle)
    ']' : Restore the turtle to state of the last ']' by retrieving the state
            from the stack
    '''
    saved_states = list()
    state = (0, 0, 90, 5)
    yield (0, 0)

    for command in turtle_program:
        # print(command)
        x, y, angle, diam = state
        
        
        if command.lower() in 'abcdefghij':        # Move forward (matches a-j and A-J)
            # segment start
            lseg = np.random.normal(5, 0.2)
            # print('%f %f %f %f' %(x, y, angle, diam))
            state = (x - lseg*cos(angle * DEGREES_TO_RADIANS),
                     y + lseg*sin(angle * DEGREES_TO_RADIANS),
                     angle,
                     diam + 5)
            #  segment end
            # print(state)
            if command.islower():# Add a break in the line if command matches a-j (lowercase)
                yield (float('nan'), float('nan'))

            yield (state[0], state[1])

        elif command == '+':                       # Turn clockwise
            state = (x, y, angle + turn_amount, diam)

        elif command == '-':                       # Turn counterclockwise
            state = (x, y, angle - turn_amount, diam)

        elif command == '[':                       # Remember current state
            saved_states.append(state)

        elif command == ']':                       # Return to previous state
            state = saved_states.pop()
            yield (float('nan'), float('nan'))
            x, y, angle, diam = state
            yield (x, y)

        # Note: We silently ignore unknown commands

def pos_neg():
    return 1 if random.random() < 0.01 else -1
        
def l_plot(axiom, transformations, iterations=0, angle=45):
    turtle_program = transform_multiple(axiom, transformations, iterations)
    print(turtle_program)
    coords = branching_turtle_to_coords_orig(turtle_program, angle)
    plot_coords(coords, bare_plot=False) # bare_plot removes the axis labels
    
    
l_plot('F', {'F': 'F[-F][+F]'}, 2, 22.5)