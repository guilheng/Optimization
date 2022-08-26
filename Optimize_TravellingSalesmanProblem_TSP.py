# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:20:31 2022

@author: guilh
"""
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model


node = pd.read_excel('routes_dats.xlsx' , sheet_name='node')
routes = pd.read_excel('routes_dats.xlsx' , sheet_name='routes')
n_node = len(node)
n_routes = len(routes)
model = cp_model.CpModel()
x = np.zeros(n_routes).tolist()
for c in routes.index: 
    x[c] = model.NewIntVar(0,1, 'x[{}]'.format([x]))

model.Minimize(sum([x[c] * routes.distance[c] for c in routes.index]))

no_origin = int(node.no[node.desc=='origin'])
no_destiny = int(node.no[node.desc=='destiny'])
model.Add(sum([x[c] for c in routes.index[routes.node_from==no_origin]])==1)
model.Add(sum([x[c] for c in routes.index[routes.node_to==no_destiny]])==1)

for no in node.no[node.desc=='middle']:
    sum_enter = sum([x[c] for c in routes.index[routes.node_to==no]])
    sum_out = sum([x[c] for c in routes.index[routes.node_from==no]])
    model.Add(sum_out <= 1)
    model.Add(sum_enter <= 1)
    model.Add(sum_enter == sum_out)
    
solver = cp_model.CpSolver()
status = solver.Solve(model)
print('Status =', solver.StatusName(status))
print('FO =', solver.ObjectiveValue())

routes['activated'] = 0 
for c in routes.index:
    routes.activated[c] = solver.Value(x[c])
print(routes)

print('Routes Optimize')
for c in routes.index[routes.activated==1]:
    print('X%i%i - %.2f' % (routes.node_from[c], routes.node_to[c], routes.distance[c]))

