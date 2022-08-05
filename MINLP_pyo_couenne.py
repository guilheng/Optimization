#Solving a Mixed Integer NonLinear Programming (MINLP) (simple example)
#Using Pyomo and solver 'couenne'

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

model = pyo.ConcreteModel()

model.x = pyo.Var(within=Integers, bounds=(0,10))
model.y = pyo.Var(bounds=(0,10))

x = model.x
y = model.y

model.C1 = pyo.Constraint(expr= -x+2*x*y<=8)
model.C2 = pyo.Constraint(expr= 2*x+y<=14)
model.C3 = pyo.Constraint(expr= 2*x-y<=10)

model.obj = pyo.Objective(expr= x+y*x , sense=maximize)

opt = SolverFactory('couenne' , executable= 'C:\\couenne\\bin\\couenne.exe')
opt.solve(model)

model.pprint()

x_value = pyo.value(x)
y_value = pyo.value(y)

print('x=', x_value)
print('y=', y_value)

