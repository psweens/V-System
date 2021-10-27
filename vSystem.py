import random
import numpy as np
import random
from libGenerator import setProperties, calBifurcation, calParam
from analyseGrammar import posneg

def I(n, d0, val=3):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        rotate = np.random.uniform(22.5, 27.5)
        return 'f(' + p1 + ',' + str(params['d0']) + ')' + '+(' + str(rotate) + ')' +\
            '[' + R(n-1, params['d0']) + ']'
    else: return 'I'


def R(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(3))), params)
        p2 = calParam(str.join('co/',str(int(2))), params)
        descrip = 'f(' + p1 + ')' + G(n, d0, val=7) +\
            G(n-1, d0, val=7) + G(n-1, d0, val=7) + '[' + B(n-1, params['d1']) +\
                ']' + 'f(' + p2 + ',' + str(params['d2']) + ')' + B(n-1, params['d2'])
        return descrip
    else: return 'R'
                    

def B(n, d0):
    if n > 0:
        return G(n-1, d0, val=7) + G(n-1, d0, val=7) + G(n-1, d0, val=7) +\
            '/(' +str(90.0) + ')' + A(n, d0)
    else: return 'B'


def F(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        theta1 = params['th1'] #+ np.random.uniform(-2.5, 2.5)
        theta2 = params['th2'] #+ np.random.uniform(-2.5, 2.5)
        tilt = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        return S(n-1, d0)+'['+'+('+str(theta1)+')'+'/('+str(tilt)+')'+\
            F(n-1, params['d1'])+']'+'['+'-('+str(theta2)+')'+'/('+str(tilt)+')'+\
                F(n-1, params['d2'])+']'
    else: return 'F'
    
    
def S(n, d0, val=5, margin=0.5):
    r = random.random()
    if r>= 0.0 and r < margin: return '{' + S1(n, d0, val) + '}'
    if r >= margin and r < 1.0: return '{' + S2(n, d0, val) + '}'


def S1(n, d0, val=5):
    if n>0:
        # "Fanning" of trees
        rotate = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        params = calBifurcation(d0)
        descrip = D(n-1, params['d0'], val) + '+(' + str(rotate) + ')' + \
            D(n-1, params['d0'], val) + '-(' + str(rotate) + ')' + \
                D(n-1, params['d0'], val) + '-(' + str(rotate) + ')' + \
                    D(n-1, params['d0'], val) + '+(' + str(rotate)+ ')' + \
                        D(n-1, params['d0'], val)
        return descrip
    else: return 'S'


def S2(n, d0, val=5):
    if n>0:
        # "Fanning" of trees
        rotate = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        params = calBifurcation(d0)
        descrip = D(n-1, params['d0'], val) + '-(' + str(rotate) + ')' + \
            D(n-1, params['d0'], val) + '+(' + str(rotate) + ')' + \
                D(n-1, params['d0'], val) + '+(' + str(rotate) + ')' + \
                    D(n-1, params['d0'], val) + '-(' + str(rotate) + ')' + \
                        D(n-1, params['d0'], val)
        return descrip
    else: return 'S'


def D(n, d0, val=5):
    if n>0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')'
    else: return 'D'
    

def G(n, d0, val=5, shift=18.0):
    if n>0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')' 
    else: return 'G'
    
    
def A(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        return S(n-1, d0) + '[+(' + str(params['th1']) + ')' + A(n-1, params['d1']) + ']' +\
            '[+(' + str(params['th2']) + ')' + A(n-1, params['d2'])
    else: return 'A'


# def A(d0):
#     return 'f(' + str(1) + ',' + str(d0) + ')' + 'f(' + str(1) + ',' + str(2*d0) + ')' + \
#         'f(' + str(1) + ',' + str(d0) + ')'

# def E(d0):
#     return 'f(' + str(1) + ',' + str(d0) + ')' + 'f(' + str(1) + ',' + str(0.5*d0) + ')' + \
#         'f(' + str(1) + ',' + str(d0) + ')'











# 2D grammars - don't currently work with framework
def simplest_gramma(n=None, theta1=20., theta2=20., params=None):
    return 'f' + '[' + '+' + F(n-1, params['d0']) + ']' + '-' + F(n-1, params['d0'])

def simple_grammar(n=None, theta1=20., theta2=20., params=None):
    return 'f'+'['+'+('+str(theta1)+')'+F(n-1,params['d1'])+']'+'['+'-('+str(theta2)+')'+F(n-1,params['d2'])+']'
