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
        # print(command)
        x, y, z, alpha, beta, diam, lseg, dx, dy, dz = state


        if command.lower() in 'abcdefghijs':        # Move forward (matches a-j and A-J)
            # segment start


            if command.islower():
                lseg, tdiam = eval_brackets(index, turtle_program)
                dx, dy, dz = rotate(p=beta*DEGREES_TO_RADIANS,
                                    r=alpha*DEGREES_TO_RADIANS,
                                    v=normalise(np.array([x,y,z]),lseg))
                
                if tdiam > 0.0: diam = tdiam

                #dx, dy, dz, alpha = proximity(state,origin,rim)

                x += dx
                y += dy
                z += dz

                #rim= rim + np.amin(np.array([dx, dy, dz])) # growing rim radius

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
        # Note: silently ignore unknown commands

def eval_brackets(index, turtle):

    a = ''
    b = ''
    double = 0
    neg1 = 0
    neg2 = 0
    for i in range(index+2, len(turtle)):
        if turtle[i] == ')':
            break
        elif turtle[i] == ',':
            double = 1
        elif turtle[i] == '-':
            if double == 0: neg1 = 1
            else: neg2 = 1;
        elif not (turtle[i] == '(' or turtle[i] == '+'):
            if double == 0:
                a = a + str(turtle[i])
            else:
                b = b + str(turtle[i])
                
    if double == 1:
        aa = eval(a)
        bb = eval(b)
        if neg1 == 1: aa *= -1
        if neg2 == 1: bb *= -1
        return aa, bb
    else:
        aa = eval(a)
        if neg1 == 1: aa *= -1
        return aa, 0.0

def randomposneg():
    return 1 if random.random() < 0.5 else -1

def raddist(origin, location, shell=80, core=False):
    if not core:
        return bool(sqrt(pow(origin[0]-location[0],2) + pow(origin[1]-location[1],2) +
                    pow(origin[2]-location[2],2)) < shell)
    else:
        return bool(sqrt(pow(origin[0]-location[0],2) + pow(origin[1]-location[1],2) +
                    pow(origin[2]-location[2],2)) > shell)

def proximity(state,origin,rim):

    x, y, z, alpha, beta, diam, lseg, _, _, _ = state

    points = 6
    roll = np.linspace(alpha,0.,points) * DEGREES_TO_RADIANS
    pitch = np.linspace(beta,0.,points) * DEGREES_TO_RADIANS
    
    yaw = np.linspace(70.,0.,points) * DEGREES_TO_RADIANS
       

    orbitsuccess = False
    while not orbitsuccess:
        orbitsuccess, dx, dy, dz = pointCycle(np.array([x,y,z]),lseg,origin,
                                              points,pitch,roll,rim,yaw,core=False)

        alphaOld = alpha
        alpha = alpha + random.uniform(-25,25)
        roll = np.linspace(alpha,alphaOld,int(points)) * DEGREES_TO_RADIANS

        betaOld = beta
        beta = beta + random.uniform(-25,25)
        pitch = np.linspace(betaOld, beta,int(points)) * DEGREES_TO_RADIANS

    return dx, dy, dz, alpha

def pointCycle(p0,lseg,origin,points,pitch,roll,rim,yaw,core=False):
    orbitsuccess = False
    for k in range(points):
        for i in range(points):
            for j in range(points):
                if not orbitsuccess:
                    dx, dy, dz = rotate(y=yaw[k],p=pitch[j],r=roll[i],v=normalise(p0,lseg))
                    if raddist(origin, (p0[0]+dx,p0[1]+dy,p0[2]+dz), rim):
                        if core:
                            if raddist(origin, (p0[0]+dx,p0[1]+dy,p0[2]+dz), 50, core):
                                orbitsuccess = True
                        else:
                            orbitsuccess = True

    return orbitsuccess, dx, dy, dz

def posneg(value):
    if value >= 0.:
        return 1.
    else:
        return -1.
