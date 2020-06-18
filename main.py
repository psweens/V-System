from vSystem import F
from calc_network import branching_turtle_to_coords
from view_network import plot_coords

# Default parameters
properties = {"k": 3,
              "epsilon": 7, # Proportion between length & diameter
              "randmarg": 2, # Randomness margin between length & diameter
              "sigma": 5, # Determines type deviation for Gaussian distributions
              "stochparams": True} # Whether the generated parameters will also be stochastic

turtle_program = F(n=11,d0=20.,properties=properties)
coords = branching_turtle_to_coords(turtle_program)
print(turtle_program)
plot_coords(coords, bare_plot=False) # bare_plot removes the axis labels

# Output tuple
# import csv
# with open('test.csv', 'w') as f:
#     writer = csv.writer(f , lineterminator='\n')
#     for tup in coords:
#         writer.writerow(tup)
        
