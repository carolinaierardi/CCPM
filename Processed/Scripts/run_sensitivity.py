#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:50:31 2024

@author: carolinaierardi
"""

import os                               #directory changing

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


os.chdir("~/Run/Scripts") #change wd

from v3_ccpm_functions import *


os.chdir("~/Run/Data")

with open("sensitivity_data.pkl", "rb") as f:        #load vectorised upper triangle (no GSR)
    sensitivity_data = pickle.load(f)                #should be 727 x 6670   


(fn_mats, fn_mats_gsr, 
 training, testing, train_pheno, test_pheno, 
 subtraining, validating, parameters, permutations) = unpack_dict(sensitivity_data)



sens_val_summary = []        #empty list to store validation results
sens_testing_summary = []    #empty list to store testing results
sens_gsrval_summary = []     # same for GSR data
sens_gsrtesting_summary = [] 

#ALTER NEXT LINE TO OBTAIN DIFFERENT SET OF permutations
permutation_i = permutations[:5] #select portion of permutations wanted


for i in range(len(permutation_i)): 
    
    print(f"Permutation: {i + 1}")
    
    fn_mats_copy = fn_mats[:,:,permutation_i[i]] #assign the indexes of the participants 
    
    training_mats = fn_mats_copy[:,:,training] #get the connectivity matrices so the data can be used only on training participants
    testing_mats = fn_mats_copy[:,:,testing]   #connectivity for testing participants  

    validation_dict, testing_dict = MLpipeline(training_mats, train_pheno, 
                                               subtraining, validating, parameters, 
                                               testing_mats, test_pheno)
    
    sens_val_summary.append(validation_dict)
    sens_testing_summary.append(testing_dict)
    
    #NOW FOR GSR
    print(f"GSR permutation: {i + 1}" )
    fn_mats_gsr_copy = fn_mats_gsr[:,:,permutation_i[i]]
   
    gsr_training_mats = fn_mats_gsr_copy[:,:,training] #get the connectivity matrices so the data can be used only on training participants
    gsr_testing_mats = fn_mats_gsr_copy[:,:,testing]     

    gsr_validation_dict, gsr_testing_dict = MLpipeline(gsr_training_mats, train_pheno, subtraining, 
                                                       validating, parameters, gsr_testing_mats, test_pheno)
    
    sens_gsrval_summary.append(gsr_validation_dict)
    sens_gsrtesting_summary.append(gsr_testing_dict)
    

with open("SENS_valsummary.pkl", "wb") as f:
    pickle.dump(sens_val_summary, f)
    
with open("SENS_testsummary.pkl", "wb") as f:
    pickle.dump(sens_val_summary, f)
    
with open("SENS_GSR_valsummary.pkl", "wb") as f:
    pickle.dump(sens_gsrval_summary, f)
    
with open("SENS_GSR_testsummary.pkl", "wb") as f:
    pickle.dump(sens_gsrtesting_summary, f)
    
    

    
