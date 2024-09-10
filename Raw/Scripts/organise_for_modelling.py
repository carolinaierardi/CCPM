#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:32:39 2024

@author: carolinaierardi
"""

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
import os

os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed")

#%% Sensitivity analysis

with open("sensitivity_gsr.pkl", "rb") as f: #load indices for order in permutations
    sensitivity_gsr = pickle.load(f)         #this should have size fo 1001
                                              #and the first array should be 0 - 726  
    
with open("sensitivity_nogsr.pkl", "rb") as f: #load indices for order in permutations
    sensitivity_nogsr = pickle.load(f)         #this should have size fo 1001
                                              #and the first array should be 0 - 726  
                                              
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Modelling")

with open("sensitive_splits.pkl", "rb") as f: #load indices for order in permutations
    sensitivity_splits = pickle.load(f)         #this should have size fo 1001
      
with open("sensitivity_permutations.pkl", "rb") as f: #load indices for order in permutations
    sensitivity_permutations = pickle.load(f)         #this should have size fo 1001
      
with open("parameters.pkl", "rb") as f: #load indices for order in permutations
    parameters = pickle.load(f)         #this should have size fo 1001
                                              #and the first array should be 0 - 726  
    
sensitivity_data = {"No GSR ROIs": sensitivity_nogsr["FC"],
                  "GSR ROIs": sensitivity_gsr["FC"],
                  "phenotype": sensitivity_gsr["phenotype"],
                  "training": sensitivity_splits["training"],
                  "testing" : sensitivity_splits["testing"],
                  "subtraining": sensitivity_splits["subtraining"],
                  "validating": sensitivity_splits["validation"],
                  "permutations": sensitivity_permutations,
                  "parameters": parameters}


#%% Stringent analysis

os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed")

with open("stringent_gsr.pkl", "rb") as f: #load indices for order in permutations
    stringent_gsr = pickle.load(f)         #this should have size fo 1001
                                              #and the first array should be 0 - 726  
    
with open("stringent_nogsr.pkl", "rb") as f: #load indices for order in permutations
    stringent_nogsr = pickle.load(f)         #this should have size fo 1001
                
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Modelling")

with open("stringent_splits.pkl", "rb") as f: #load indices for order in permutations
    stringent_splits = pickle.load(f)         #this should have size fo 1001
      
with open("stringent_permutations.pkl", "rb") as f: #load indices for order in permutations
    stringent_permutations = pickle.load(f)         #this should have size fo 1001
    

print("If the preprocessing of the data has worked the following statement should be true.")
print("The phenotype for both GSR and No GSR should be the same.")
print(all(stringent_gsr["phenotype"] == stringent_nogsr["phenotype"]))

      
stringent_data = {"No GSR ROIs": stringent_nogsr["FC"],
                  "GSR ROIs": stringent_gsr["FC"],
                  "phenotype": stringent_gsr["phenotype"],
                  "training": stringent_splits["training"],
                  "testing" : stringent_splits["testing"],
                  "subtraining": stringent_splits["subtraining"],
                  "validating": stringent_splits["validation"],
                  "permutations": stringent_permutations,
                  "parameters": parameters}


#%% Vectorised analysis

os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed")


with open("vect_gsr.pkl", "rb") as f:        #load vectorised upper triangle (no GSR)
    vect_gsr = pickle.load(f)                #should be 727 x 6670   

with open("vect_nogsr.pkl", "rb") as f:     #load vectorised upper triangle (GSR)
    vect_nogsr = pickle.load(f)             #should be 727 x 6670                                   
      

vectorised_data = {"No GSR ROIs": vect_nogsr["FC"],
                  "GSR ROIs": vect_gsr["FC"],
                  "phenotype": stringent_gsr["phenotype"],
                  "training": stringent_splits["training"],
                  "testing" : stringent_splits["testing"],
                  "subtraining": stringent_splits["subtraining"],
                  "validating": stringent_splits["validation"],
                  "permutations": stringent_permutations,
                  "parameters": parameters}


#%% Save data to run directory

os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Run/Data")

with open("stringent_data.pkl", "wb") as f: #load indices for order in permutations
     pickle.dump(stringent_data, f)         #this should have size fo 1001

with open("sensitivity_data.pkl", "wb") as f: #load indices for order in permutations
     pickle.dump(sensitivity_data, f)         #this should have size fo 1001

with open("vectorised_data.pkl", "wb") as f: #load indices for order in permutations
     pickle.dump(vectorised_data, f)         #this should have size fo 1001




