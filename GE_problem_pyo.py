#Optimizing the power generation problem
#Using Pyomo and solver CPLEX

import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

dados_ger = pd.read_excel('inputs_dados.xlsx', sheet_name= 'geracao')
dados_car = pd.read_excel('inputs_dados.xlsx', sheet_name= 'carga')
dados_dep = pd.read_excel('inputs_dados.xlsx', sheet_name= 'dependencia')

Ng =len(dados_ger)

model = pyo.ConcreteModel()
model.Pg = pyo.Var(range(Ng), bounds=(0,None))
Pg = model.Pg

#Constraints1
model.balanco = pyo.Constraint(expr=(sum(Pg[g] for g in dados_ger.id)==sum(dados_car.valor)))

#Constraints2
model.condicao = pyo.ConstraintList()
for c in dados_dep.carga.unique():
    model.condicao.add(expr= float(dados_car.valor[c]) <= sum([Pg[g] for g in dados_dep.gerador[dados_dep.carga==c]]))

#Constraints3
model.limite = pyo.ConstraintList()
for g in dados_ger.id:
    model.limite.add(expr= Pg[g] <= float(dados_ger.maximo[g]))
   
   
#Objective Function
model.obj = pyo.Objective(expr = sum([Pg[g] * float(dados_ger.custo[g]) for g in dados_ger.id]))

#Defining the solver
opt = SolverFactory('cplex')
result = opt.solve(model)

model.pprint()

#Creating a DataFrame for energy generation results
dados_ger['geracao'] = [pyo.value(Pg[g]) for g in dados_ger.id]

print(result)


