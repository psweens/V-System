from grammars import transform_multiple
from libGenerator import setProperties
from calc_network import branching_turtle_to_coords
from view_network import plot_coords

# Default parameters
properties={"k": 3,
        "epsilon": 10 , # Proportion between length & diameter
        "randmarg": 3 , # Randomness margin between length & diameter
        "sigma": 5, # Determines type deviation for Gaussian distributions
        "stochparams ": True} # Whether the generated parameters will also be stochastic

# for i in range(0,3):
#     print(F(i,2))

turtle_program = transform_multiple('F', iterations=8, properties=properties)
coords = branching_turtle_to_coords(turtle_program)
plot_coords(coords, bare_plot=False) # bare_plot removes the axis labels