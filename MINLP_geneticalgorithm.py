#Solving a Mixed Integer NonLinear Programming (MINLP)
#Using geneticalgorithm

import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def f(x):
    pen = 0
    if not -x[0]+2*x[1]*x[0]<=8: pen = np.inf
    if not 2*x[0]+x[1]<=14: pen = np.inf
    if not 2*x[0]-x[1]<=10: pen = np.inf
    if x[0]>10: return x[0]+x[1]
    else:
        return -(x[0]+x[1]*x[0]) + pen


varbounds= np.array([[0,10] , [0,10]])
vartypes= np.array([['int'] , ['real']])

algorithm_param = {'max_num_interation': 1000,\
                   'population_size': 1000,\
                   'mutation_probability': 0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type': 'uniform',\
                   'max_interation_witout_improv':None}

model = ga(function=f, 
           dimension=2, 
           variable_type_mixed=vartypes, 
           variable_boundaries=varbounds)
    
model.run()
