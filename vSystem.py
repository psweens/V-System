import random
from libGenerator import setProperties, calBifurcation, calParam

# 70 is a default rotation angle
def F(n, d0, properties=None):
    if n > 0:
        setProperties(properties)
        params = calBifurcation(d0)
        return '{'+S(n-1, params['d0'])+'}'+'['+'+('+str(params['th1'])+')'+'/('+str(15.0)+')'+F(n-1, params['d1'])+']'+'['+'-('+str(params['th2'])+')'+'/('+str(70.0)+')'+F(n-1, params['d2'])+']'
    else: return 'F'

def S1(n, d0):

    if n>0:
        params = calBifurcation(d0)
        return D(n-1, params['d0']) + '+(' + str(25.0) + ')' + D(n-1, params['d0']) + '-(' + str(25.0) + ')' + D(n-1, params['d0']) + '-(' + str(25.0) + ')' + D(n-1, params['d0']) + '+(' + str(25.0)+ ')' + D(n-1, params['d0'])
    else: return 'S'

def S2(n, d0):

    if n>0:
        params = calBifurcation(d0)
        return D(n-1, params['d0']) + '-(' + str(25.0) + ')' + D(n-1, params['d0']) + '+(' + str(25.0) + ')' + D(n-1, params['d0']) + '+(' + str(25.0) + ')' + D(n-1, params['d0']) + '-(' + str(25.0) + ')' + D(n-1, params['d0'])
    else: return 'S'

def D(n, d0):

    if n>0:
        params = calBifurcation(d0)
        p1 = calParam('co/5', params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')'
    else: return 'D'

def S(n, d0):
    r = random.random()
    if r>= 0.0 and r < 0.5: return S1(n, d0)
    if r >= 0.5 and r < 1.0: return S2(n, d0)