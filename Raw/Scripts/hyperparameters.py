#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:27:22 2024

@author: carolinaierardi
"""

import os                               #directory changing
import numpy as np 
import pandas as pd

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
import itertools


#Change wd
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Modelling")

#set hyperparameters testing
hyp_par_a = {'solver': ['newton-cg','saga']} # no penalty (set manually to None in vectorized_MLpipeline)
hyp_par_b = {'solver': ['newton-cg'], 'penalty': ['l2'], 'C': [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]} #np.logspace(-4,4,9)}
hyp_par_c = {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]} #np.logspace(-4,4,9)}

a = hyp_par_a.values()
b = hyp_par_b.values()
c = hyp_par_c.values()

#Put all elements into list
parameters = list(itertools.product(*a)) + list(itertools.product(*b)) + list(itertools.product(*c))

#generate pickle file
with open("parameters.pkl", "wb") as f:
    pickle.dump(parameters, f)
    
    

