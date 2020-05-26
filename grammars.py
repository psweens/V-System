import random
from libGenerator import setProperties, calBifurcation
from vSystem import *

# def F(n, d0, properties=None): 
#     if n > 0:
#         setProperties(properties)
#         params = calBifurcation(d0)
#         return 'F'+'['+'+('+str(params['th1'])+')'+F(n-1,params['d1'])+']'+'-('+str(params['th2'])+')'+F(n-1, params['d2'])
#     else: return 'F'

    

    
def transform_multiple(sequence, transformations=None, iterations=2, d0=2,
                       properties=None):
    
    '''
    Apply the transform sequence multiple times
    '''
    transformations = {'F': 'F[-F][+F]'}
    # for i in range(iterations):
    sequence = F(iterations,d0=2,properties=properties)
        # sequence = transform_sequence(sequence, transformations)
    print(sequence)
    return sequence


def transform_sequence(sequence, transformations):
   
    '''
    Represent transformation rule as an entry with the character key and 
    replacement string as a value e.g.
    a -> aba
    c -> bb
    is represented as {'a': 'aba', 'c': 'bb'}
    '''
    return ''.join(transformations.get(c, c) for c in sequence)