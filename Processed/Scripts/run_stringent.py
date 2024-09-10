#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:52:02 2024

@author: carolinaierardi
"""

import os                               #directory changing
import numpy as np 

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    

os.chdir("../Scripts") #change wd

from v3_ccpm_functions import *


os.chdir("../Data")

with open("stringent_data.pkl", "rb") as f:        #load vectorised upper triangle (no GSR)
    stringent_data = pickle.load(f)                #should be 727 x 6670   
    

(fn_mats, fn_mats_gsr, 
 training, testing, train_pheno, test_pheno, 
 subtraining, validating, parameters, permutations) = unpack_dict(stringent_data)


significance = 0.05


val_summary = []        #empty list to store validation results
testing_summary = []    #empty list to store testing results
gsrval_summary = []
gsrtesting_summary = [] 

permutation_i = permutations[:5] #select portion of permutations wanted

for i in range(len(permutation_i)): 
    
    print(f"Permutation: {i + 1}")
    
    fn_mats_copy = fn_mats[:,:,permutation_i[i]]   #assign the indexes of the participants 
    
    training_mats = fn_mats_copy[:,:,training] #get the connectivity matrices so the data can be used only on training participants
    testing_mats = fn_mats_copy[:,:,testing]   #connectivity for testing participants  

    validation_dict, testing_dict = MLpipeline(training_mats, train_pheno, 
                                               subtraining, validating, 
                                               parameters, testing_mats, test_pheno)
    
    val_summary.append(validation_dict)
    testing_summary.append(testing_dict)
    
    #NOW FOR GSR
    fn_mats_gsr_copy = fn_mats_gsr[:,:,permutation_i[i]]
   
    gsr_training_mats = fn_mats_gsr_copy[:,:,training] #get the connectivity matrices so the data can be used only on training participants
    gsr_testing_mats = fn_mats_gsr_copy[:,:,testing]     

    validation_dict, testing_dict = MLpipeline(gsr_training_mats, train_pheno, 
                                               subtraining, validating, 
                                               parameters, gsr_testing_mats, test_pheno)
    
    gsrval_summary.append(validation_dict)
    gsrtesting_summary.append(testing_dict)
    

#save results in pickle files
with open("STRIN_valsummary.pkl", "wb") as f:
    pickle.dump(val_summary, f)
    
with open("STRIN_testsummary.pkl", "wb") as f:
    pickle.dump(testing_summary, f)
    
with open("STRIN_GSR_valsummary.pkl", "wb") as f:
    pickle.dump(gsrval_summary, f)
    
with open("STRIN_GSR_testsummary.pkl", "wb") as f:
    pickle.dump(gsrtesting_summary, f)
    
    
    
    
    