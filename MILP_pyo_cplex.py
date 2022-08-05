#Solving a Mixed Integer Linear Programming (MILP)
#Using Pyomo framework and solver CPLEX
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import time

model = pyo.ConcreteModel()

model.x = pyo.Var(range(1,6), within=Integers, bounds=(0,None))
model.y = pyo.Var(bounds=(0, None))

x = model.x
y = model.y

model.C1 = pyo.Constraint(expr=(sum([x[i] for i in range(1,6)]) + y <= 20))
 
model.C2 = pyo.ConstraintList()
for i in range(1,6):
    model.C2.add(expr= x[i] + y >= 15)
    
model.C3 = pyo.Constraint(expr= sum([i*x[i] for i in range(1,6)]) >= 10)
model.C4 = pyo.Constraint(expr=(x[5] + y >= 30))

model.obj = pyo.Objective(expr= sum([x[i] for i in range(1,6)])+y, sense=minimize)

begin = time.time()
opt = SolverFactory('cplex')
result = opt.solve(model)
deltaT = time.time() - begin

model.pprint()
print(result)
print('Tempo =', np.round(deltaT,2))


for i in range(1,6):
    print('x[%i] = %i' % (i, pyo.value(x[i])))
print('y = %.2f' % pyo.value(y))
print('Objetivo = ', pyo.value(model.obj))