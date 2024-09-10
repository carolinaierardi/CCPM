#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:43:28 2023

@author: carolinaierardi
"""

import os                               #directory changing
#os.chdir() #ALTER THIS LINE FOR DIRECTORY WHERE FILES ARE 
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Run/Scripts") #change wd


from v3_ccpm_functions import *         #where function is stored

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Run/Data")

with open("vectorised_data.pkl", "rb") as f:        #load vectorised upper triangle (no GSR)
    vectorised_data = pickle.load(f)                #should be 727 x 6670   


(trans_vec, gsrtrans_vec, 
 training, testing, train_pheno, test_pheno, 
 subtraining, validating, parameters, permutations) = unpack_dict(vectorised_data)


    
#initialise for results    
vecval_summary = []         #validation no GSR results
vectesting_summary = []     #testing no GSR results

gsrvecval_summary = []      #validation GSR results
gsrvectesting_summary = []  #testing GSR results

#ALTER NEXT LINE TO OBTAIN DIFFERENT SET OF permutations
my_vec_permutations = permutations[:2] #select portion of permutations wanted

    
for i in range(len(my_vec_permutations)): 
    
    print(f"Permutation: {i + 1}")

    vec_copy = trans_vec[my_vec_permutations[i],:] #rearrange vectors to get order of first permutation
   
    training_mats = vec_copy[training,:] #separate vectors for training participants
    testing_mats = vec_copy[testing,:]   #separate vectors for testing participants    
    
    #run GS and apply best model to whole training dataset and predict testing scores
    vec_validation_dict, vec_testing_dict = vectorized_MLpipeline(training_mats, train_pheno, 
                                                                  subtraining, validating, parameters, 
                                                                  testing_mats, test_pheno)

    #store validation and testing values in dictionaries
    vecval_summary.append(vec_validation_dict)
    vectesting_summary.append(vec_testing_dict)
    
    #NOW GSR
    print(f"GSR - Permutation: {i + 1}")

    vec_copy = gsrtrans_vec[my_vec_permutations[i],:] #rearrange vectors to get order of first permutation
   
    training_mats = vec_copy[training,:] #separate vectors for training participants
    testing_mats = vec_copy[testing,:]   #separate vectors for testing participants    
    
    #run GS and apply best model to whole training dataset and predict testing scores
    gsrvec_validation_dict, gsrvec_testing_dict = vectorized_MLpipeline(training_mats, train_pheno, 
                                                                        subtraining, validating, parameters, 
                                                                        testing_mats, test_pheno)

    #store validation and testing values in dictionaries
    gsrvecval_summary.append(gsrvec_validation_dict)
    gsrvectesting_summary.append(gsrvec_testing_dict)




#save results in pickle files
with open("VEC_valsummary.pkl", "wb") as f:
    pickle.dump(vecval_summary, f)
    
with open("VEC_testsummary.pkl", "wb") as f:
    pickle.dump(vectesting_summary, f)
    
with open("VEC_GSR_valsummary.pkl", "wb") as f:
    pickle.dump(gsrvecval_summary, f)
    
with open("VEC_GSR_testsummary.pkl", "wb") as f:
    pickle.dump(gsrvectesting_summary, f)
    



    
        