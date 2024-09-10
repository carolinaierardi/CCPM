#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:33:17 2024

@author: carolinaierardi
"""

import numpy as np 
from numpy.random import seed
import os                               #directory changing

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
#Change directory from where to fetch data from
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed")
    
with open("sensitivity_nogsr.pkl", "rb") as f:
    sensitivity_nogsr = pickle.load(f)
    
with open("stringent_nogsr.pkl", "rb") as f:
    stringent_nogsr = pickle.load(f)
    
#Change directory for where to store permutation pickle in
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Modelling")

seed(1)   #make same combination every time
n_perms = 1000     #permutations 

#%% Permutations for sensitivity analysis

sens_permutations = []  #generate permutations (ensure they are the same for GSR on and off)
for i in range(n_perms): #for each permutation

    sens_permutations += [np.random.permutation(len(sensitivity_nogsr["phenotype"]))]
    #mix the indexes 
    
#the first permutation will be for the actual data     
sens_permutations.insert(0, np.array(range(0,len(sensitivity_nogsr["phenotype"]))) )  

with open("sensitivity_permutations.pkl", "wb") as f:
    pickle.dump(sens_permutations, f)
    

#%% Permutations for stringent analysis


strin_permutations = []  #generate permutations (ensure they are the same for GSR on and off)
for i in range(n_perms): #for each permutation

    strin_permutations += [np.random.permutation(len(stringent_nogsr["phenotype"]))]
    #mix the indexes 
    
#the first permutation will be for the actual data     
strin_permutations.insert(0, np.array(range(0,len(stringent_nogsr["phenotype"]))) )  


with open("stringent_permutations.pkl", "wb") as f:
    pickle.dump(strin_permutations, f)
    

