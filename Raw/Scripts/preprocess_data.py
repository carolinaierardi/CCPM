#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:53:13 2024

@author: carolinaierardi

"""

# Title: preprocess_data.py
# Date: 07.06.2024
# Version: 1.0
# Author CMI
# Purpose: Script loads in data from ABIDE and preprocesses it, excluding participants and 
# outputing pickle files for GSR/No GSR for different analyses


import os                               #directory changing
import numpy as np 
from nilearn.datasets import fetch_abide_pcp
import pandas as pd
from scipy import stats
import itertools

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


os.chdir("/Users/carolinaierardi/Documents/Academics/Project") #change wd


abide_aal = fetch_abide_pcp(derivatives='rois_aal',             #get data with AAL atlas and rois
                            pipeline = 'cpac',                  #CPAC pipeline
                            band_pass_filtering=True,           #filtering ON
                            global_signal_regression=False,     #GSR OFF
                            quality_checked=True)               #with quality check

abide_aal_gsr = fetch_abide_pcp(derivatives='rois_aal',          #get data with AAL atlas and rois
                            pipeline = 'cpac',                   #CPAC pipeline
                            band_pass_filtering=True,            #filtering ON
                            global_signal_regression=True,       #GSR ON
                            quality_checked=True)                #with quality check


#%% Functions used 

def make_df(abide_data):
    
    fcs = [np.corrcoef(ptcp.T) for ptcp in abide_data['rois_aal']]   #get connectivity matrices for each ptcp
    mean_fc = [np.mean(ptcp)for ptcp in fcs]
    mean_fd = abide_data["phenotypic"]["func_mean_fd"]
    diagnosis = abide_data["phenotypic"]["DX_GROUP"]
    site = abide_data["phenotypic"]["SITE_ID"]
    ind = abide_data["phenotypic"]["i"]
    
    d = {"index":ind, "ROIs": fcs, "Mean FC":mean_fc,
         "Mean FD":mean_fd, "diag_status": diagnosis, "Site": site}
    
    return pd.DataFrame(d)

def findMAD(df):
    
    mean_no_nan = df["Mean FC"][np.where(df.isna().any(axis = 1) == False)[0]]
    median = np.median(mean_no_nan)
    mad = np.median(abs(mean_no_nan - median))                  #find the median absolute deviation

    ub_mad = median + (3*mad)                                       #upper bound of deviations
    lb_mad = median - (3*mad)                                       #lower bound of deviations

    #this means if mean connectivity is above or below these values, we will exclude them, as follows:
    mc_noMAD = np.where((df["Mean FC"] > ub_mad) | (df["Mean FC"] < lb_mad))[0]
    
    return df.index[mc_noMAD]

def mean_fd_diff(reduced_df):
    
    pat = reduced_df.iloc[np.where(reduced_df["diag_status"] == 1)[0]]
    cont = reduced_df.iloc[np.where(reduced_df["diag_status"] == 2)[0]]
    
    print(f"Mann Whitney U:{stats.mannwhitneyu(pat['Mean FD'], cont['Mean FD'])}")
    
    p = stats.mannwhitneyu(pat['Mean FD'], cont['Mean FD'])[1]
    pat_bal = pat.copy()         #make copy of the ASD mean FD array

    while p < 0.05: #while there is a signficant difference between ASD and TC
        
            #exclude ASD participants that have the highest FD 
            pat_bal = pat_bal.drop(pat_bal.index[np.where(pat_bal["Mean FD"] == max(pat_bal["Mean FD"]))[0][0]])

            #conduct Mann Whitney U test to verify the difference                        
            mann_whi = stats.mannwhitneyu(pat_bal["Mean FD"], cont["Mean FD"])
            
            p = mann_whi[1]   #store p-value
            

    #with this, I have eliminated the 49 ASD participants with the highest mean FD value, I will now store their indices
    n = len(pat) - len(pat_bal) #the while loop revealed I need to eliminate the 49 ASD patients with highest mean FD for no significant association
    
    fd_sorted = np.argsort(reduced_df["Mean FD"])
    
    stringent_excluded_ids = []

    for i in fd_sorted[::-1].index:
        
        if reduced_df["diag_status"][i] == 1:
           
           stringent_excluded_ids += [i]
        if len(stringent_excluded_ids) == n:
            break
        
    return stringent_excluded_ids


def make_pickles(preprocessed_df, vectorise = False):
    
    #fcs = [np.corrcoef(ptcp.T) for ptcp in preprocessed_data["rois_aal"]]
    
    fcs = np.array(preprocessed_df["ROIs"])
    
    if vectorise == True:
        
        edges_upptri = [i[np.triu_indices(len(fcs[0]),1)] for i in fcs]
        fn_mats = np.array(edges_upptri).T                                     #make into one matrix with edges x subjects
        
    else:
        fn_mats = np.stack(fcs, axis=2)  
        
        
    phenotype = np.array(preprocessed_df["diag_status"])
    site = np.array(preprocessed_df["Site"])
    final_dict = {"FC": fn_mats, "phenotype": phenotype, "site": site}
    
    return final_dict

#%% Exclusion criteria
    
no_gsr_data = make_df(abide_aal)
gsr_data = make_df(abide_aal_gsr)

# Exclude based on mean FC

excluded_ids = []


        #exclude NA mean FC
excluded_ids += [no_gsr_data.index[np.where(no_gsr_data.isna().any(axis = 1) == True)[0]]]

        #exclude based on +- MAD
excluded_ids += [findMAD(no_gsr_data)]


# Exclude based on mean FD
mean_fd_threshold = 0.4                                                                   #mean FD threshold
mean_fd_over = np.where((no_gsr_data["Mean FD"] > mean_fd_threshold))[0]   #find where the values exceed 0.4, 
excluded_ids += [no_gsr_data.index[mean_fd_over]]

sensitive_excluded_ids = np.unique(list(itertools.chain(*excluded_ids)))

sensitivity_nogsr = no_gsr_data.drop(sensitive_excluded_ids)
sensitivity_gsr = gsr_data.drop(sensitive_excluded_ids)

    
#%% Stringent 

stringent_excluded_ids = mean_fd_diff(sensitivity_nogsr)

stringent_nogsr = sensitivity_nogsr.drop(stringent_excluded_ids)
stringent_gsr = sensitivity_gsr.drop(stringent_excluded_ids)
           


#%% Make pickles


os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed")

sensitivity_nogsr = make_pickles(sensitivity_nogsr)
sensitivity_gsr = make_pickles(sensitivity_gsr)


with open("sensitivity_nogsr.pkl", "wb") as f:
    pickle.dump(sensitivity_nogsr, f)

with open("sensitivity_gsr.pkl", "wb") as f:
    pickle.dump(sensitivity_gsr, f)
    
    

vect_nogsr = make_pickles(stringent_nogsr, vectorise=True)
vect_gsr = make_pickles(stringent_gsr, vectorise=True)

with open("vect_nogsr.pkl", "wb") as f:
    pickle.dump(vect_nogsr, f)

with open("vect_gsr.pkl", "wb") as f:
    pickle.dump(vect_gsr, f)



stringent_nogsr = make_pickles(stringent_nogsr)
stringent_gsr = make_pickles(stringent_gsr)


with open("stringent_nogsr.pkl", "wb") as f:
    pickle.dump(stringent_nogsr, f)

with open("stringent_gsr.pkl", "wb") as f:
    pickle.dump(stringent_gsr, f)



