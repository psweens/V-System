import random
import numpy as np
from libGenerator import setProperties, calBifurcation, calParam
from calc_network import posneg

# 70 is a default rotation angle
def F(n, d0, properties=None):
    if n > 0:
        setProperties(properties)
        params = calBifurcation(d0)
        theta1 = str(params['th1'])
        theta2 = str(params['th2'])
        # tilt = np.random.normal(70., 2.5)
        tilt = 70.
        # return simple_gramma(n, theta1, theta2, params=params)
        return stoch_gramma(n, theta1, theta2, tilt, params)
    else: return 'F'

def S1(n, d0):
    
    if n>0:
        rotate = 25
        # rotate = np.random.normal(15,7.5)
        params = calBifurcation(d0)
        descrip = D(n-1, params['d0']) + '+(' + str(rotate) + ')' + D(n-1, params['d0']) + '-(' + str(rotate) + ')' + D(n-1, params['d0']) + '-(' + str(rotate) + ')' + D(n-1, params['d0']) + '+(' + str(rotate)+ ')' + D(n-1, params['d0'])
        # slength = 5
        # descrip = D(n-1, params['d0'])
        # for _ in range(slength):
        #     descrip += plusminus() + str(rotate) + ')' + D(n-1, params['d0'])
        return descrip
    else: return 'S'

def S2(n, d0):

    if n>0:
        rotate = 25
        # rotate = np.random.normal(15,7.5)
        params = calBifurcation(d0)
        descrip = D(n-1, params['d0']) + '-(' + str(rotate) + ')' + D(n-1, params['d0']) + '+(' + str(rotate) + ')' + D(n-1, params['d0']) + '+(' + str(rotate) + ')' + D(n-1, params['d0']) + '-(' + str(rotate) + ')' + D(n-1, params['d0'])
        # slength = 10
        # descrip = D(n-1, params['d0'])
        # for _ in range(slength):
        #     descrip += plusminus() + str(rotate) + ')' + D(n-1, params['d0'])
        return descrip
    else: return 'S'

def D(n, d0, slength=50):

    if n>0:
        params = calBifurcation(d0)
        # p1 = calParam(str.join(str(slength),' / '), params)
        p1 = calParam('co/5', params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')'
    else: return 'D'

def S(n, d0, margin=0.5):
    r = random.random()
    if r>= 0.0 and r < margin: return '{' + S1(n, d0) + '}'
    if r >= margin and r < 1.0: return '{' + S2(n, d0) + '}'
    
#  Stenosis
def E(d0):
    return 'f(' + str(1) + ',' + str(param['d0']) + ')' + 'f(' + str(1) + ',' + str(0.5*param['d0']) + ')' + 'f(' + str(1) + ',' + str(param['d0']) + ')'

# Aneurysm
def A(d0):
    return 'f(' + str(1) + ',' + str(param['d0']) + ')' + 'f(' + str(1) + ',' + str(2*param['d0']) + ')' + 'f(' + str(1) + ',' + str(param['d0']) + ')'
    
def plusminus(val=0.5):
    r = random.random()
    if r>= 0.0 and r < val: return '-('
    if r >= val and r < 1.0: return '+('
    
def simplest_gramma(n=None, theta1=20., theta2=20., params=None):
    return 'f' + '[' + '+' + F(n-1, params['d0']) + ']' + '-' + F(n-1, params['d0'])

def simple_gramma(n=None, theta1=20., theta2=20., params=None):
    return 'f'+'['+'+('+str(theta1)+')'+F(n-1,params['d1'])+']'+'['+'-('+str(theta2)+')'+F(n-1,params['d2'])+']'

def stoch_gramma(n=None, theta1=20., theta2=20., tilt=70., params=None):
    return S(n-1, params['d0'])+'['+'+('+str(theta1)+')'+'/('+str(tilt)+')'+F(n-1, params['d1'])+']'+'['+'-('+str(theta2)+')'+'/('+str(tilt)+')'+F(n-1, params['d2'])+']'