#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:56:07 2024

@author: carolinaierardi
"""

# Title: split_data_ML.py
# Date: 07/06/2024
# Author: CMI
# Version: 1.0
# Purpose: loads in the preprocessed data and splits it into train/val/test sets

import os                               #directory changing
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


#%% Import data

os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Preprocessed") #change wd

with open("sensitivity_nogsr.pkl", "rb") as f:
    sensitivity_nogsr = pickle.load(f)
    
#Functions for the script
def reorder_dict(original_dict, reorder_df, vectorised = False):
    
    with open(f"{original_dict}.pkl", "rb") as f:
        dictio = pickle.load(f)
        
    if vectorised == True: 
        dictio["FC"] = dictio["FC"][:,reorder_df.index]
    else: 
        dictio["FC"] = dictio["FC"][:,:,reorder_df.index]
        
    dictio["phenotype"] = np.array(reorder_df["index"])
    dictio["site"] = np.array(reorder_df["testing site"]) 
    
    with open(f"{original_dict}.pkl", "wb") as f:
         pickle.dump(dictio,f)
         
        
#%% Sensitivity analysis

# #Reorder data
# fc_testsite_df_sens = pd.DataFrame(data = {'index':sensitivity_nogsr["phenotype"], 
#                                       "testing site" : sensitivity_nogsr["site"]}) #dataframe only with the data we need

# fc_testsite_df_sens = fc_testsite_df_sens.sort_values(by = ['testing site'])

# reorder_dict("sensitivity_nogsr", fc_testsite_df_sens)
# reorder_dict("sensitivity_gsr", fc_testsite_df_sens)

#Now, we can split the data

n_folds = 5
stratify_col = [str(i) for i in sensitivity_nogsr["phenotype"].astype(str) + sensitivity_nogsr["site"]] #now, we concatenate the columns with the DX and testing site

sens_index = np.arange(len(stratify_col))

training, testing = train_test_split(sens_index, test_size=156, 
                                     random_state = 18, 
                                     stratify=stratify_col) #take 20% of the data as testing set 

stratify_train = [stratify_col[i] for i in training]

skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1) #create stratified K-fold
subtraining = []    #empty list to store training indices for each loop
validating = []     #empty list to store validating indices for each loop

for train_index, test_index in skfold.split(training, stratify_train): #get the splits for our input data
    subtraining += [train_index]                          #get the indices for the training set this fold  
    validating += [test_index]                            #get the indices for the validating set this fold                                     

sens_splits = {"training": training,"testing":testing,
               "subtraining":subtraining,"validation":validating}




#Check this is appropriate
#I have stratified for testing site as well as diagnosis


def make_grouped_barplot(dictio, xaxislab):
    
    x = np.arange(len(xaxislab))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in dictio.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Proportion')
    ax.set_title('Proportion per group')
    ax.set_xticks(x + width, xaxislab, rotation = 90)
    ax.legend(loc='upper left', ncols=3)
    plt.show()
    
def check_stratification(input_data, splits):
    
    
    sites_train = Counter(input_data["site"][splits["training"]])
    sites_test = Counter(input_data["site"][splits["testing"]])

    sites = sorted(sites_train.keys())


    site_prop = {
        'train': [sites_train[now_site] for now_site in sites] ,
        'test': [sites_test[now_site]for now_site in sites]
    }

    site_prop = {"train":[i/len(splits["training"]) for i in site_prop["train"]],
                 "test":[i/len(splits["testing"]) for i in site_prop["test"]]
        }

    make_grouped_barplot(site_prop, sites)
    
    
    diag = ("patients","controls")
        
    groups_per = {
        "train":len(np.where(input_data["phenotype"][splits["training"]] == 1)[0])/len(splits["training"]),
        "test":len(np.where(input_data["phenotype"][splits["testing"]] == 1)[0])/len(splits["testing"])
        }

    make_grouped_barplot(groups_per, diag)

    train_sites = input_data["site"][splits["training"]]
    
    sites_subtrain = [Counter(train_sites[splits["subtraining"][i]]) for i in range(5)]
    sites_val = [Counter(train_sites[splits["validation"][i]]) for i in range(5)]


    for i in range(5):
        
        site_prop = {
            'subtrain': [sites_subtrain[i][now_site] for now_site in sites] ,
            'val': [sites_val[i][now_site]for now_site in sites]
        }

        site_prop = {"subtrain":[i/len(splits["subtraining"][0]) for i in site_prop["subtrain"]],
                     "val":[i/len(splits["validation"][0]) for i in site_prop["val"]]
            }

        make_grouped_barplot(site_prop, sites)
    
check_stratification(sensitivity_nogsr, sens_splits)
    

#%% Stringent and Vectorised

#These two analyses work with the same data so indices can be split in the same way

with open("stringent_nogsr.pkl", "rb") as f:
    stringent_nogsr = pickle.load(f)

    
#First, we need to merge the testing sites because there are too few elements in each class
#We must also change the order of these in the original data    

fc_testsite_df = pd.DataFrame(data = {'index':stringent_nogsr["phenotype"], 
                                      "testing site" : stringent_nogsr["site"]}) #dataframe only with the data we need

fc_testsite_df = fc_testsite_df.sort_values(by = ['testing site'])
         
# reorder_dict("stringent_gsr", fc_testsite_df)
# reorder_dict("stringent_nogsr", fc_testsite_df)
# reorder_dict("vect_nogsr", fc_testsite_df, vectorised=True)
# reorder_dict("vect_gsr", fc_testsite_df, vectorised=True)

#only run this to find the site with highest correlation: 
    #THE ANSWER IS OLIN FOR BOTH
#Now, merge the test_sites

# n_per_site = Counter(fc_testsite_df['testing site'])
# n_per_site = list(n_per_site.values())

# n_per_site_ind = np.cumsum(n_per_site)

# new_fcs = [stringent_nogsr["FC"][i] for i in np.array(fc_testsite_df['index'])]       # [:] is key!

# edges_upptri = [i[np.triu_indices(len(stringent_nogsr["FC"][0]))] for i in new_fcs]   #get edges in the upper triangle of matrix for correlation
# edges_upptri = np.array(edges_upptri)                               #make into an array 

# p_corr2 = np.corrcoef(edges_upptri)                                 #correlation matrix of all 

# splits = np.split(p_corr2, n_per_site_ind)                           #split the rows into the different testing sites

# splitsplit = [np.split(i,n_per_site_ind, axis = 1) for i in splits]  #within the split ones, split into the further sections along the columns

# avgs = [np.mean(item) for sublist in splitsplit for item in sublist] #find the average for each array within the nested list
# avgs = [x for x in avgs if str(x) != 'nan']                          #get rid of nan values
# avgs = np.array(avgs).reshape([len(n_per_site),len(n_per_site)])     #reshape into a site x site matrix 


# test_site_col = list(set(fc_testsite_df["testing site"])) #create a copy of the site ids for all participants


# new_testsite_SBL = int(np.where(avgs[:,4] == np.sort(avgs[:,4])[-1])[0])   #get the second highest correlation for the sites (the highest is within the own site)
# new_testsite_CMU = int(np.where(avgs[:,10] == np.sort(avgs[:,10])[-1])[0])   #get the second highest correlation for the sites (the highest is within the own site)


test_site = stringent_nogsr["site"].copy()

for i in range(len(test_site)):                          #for every participant
    if test_site[i] == 'SBL':                            #if they belong to SBL site
        test_site[i] = "OLIN"   #turn into the next highest correlation site
        #test_site[i] = test_site_col[new_testsite_SBL]   #turn into the next highest correlation site
    elif test_site[i] == 'CMU':                          #if they blong to CMU site
        test_site[i] = "OLIN"   #turn into the next highest correlation site
        #test_site[i] = test_site_col[new_testsite_CMU]   #turn intonext highest correlation site

#Now the test sites are merged, we can split between training and testing

s_stratify_col = [str(i) for i in stringent_nogsr["phenotype"].astype(str) + test_site] #now, we concatenate the columns with the DX and testing site
strin_index = np.arange(len(s_stratify_col))

s_training, s_testing = train_test_split(strin_index, test_size=147, 
                                     random_state = 18, 
                                     stratify=s_stratify_col) #take 20% of the data as testing set 


#Split between training and validation
s_stratify_train = [s_stratify_col[i] for i in s_training]

skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1) #create stratified K-fold
s_subtraining = []    #empty list to store training indices for each loop
s_validating = []     #empty list to store validating indices for each loop

for train_index, test_index in skfold.split(s_training, s_stratify_train): #get the splits for our input data
    s_subtraining += [train_index]                          #get the indices for the training set this fold  
    s_validating += [test_index]                            #get the indices for the validating set this fold                                     


stringent_splits = {"training": s_training,"testing":s_testing,
               "subtraining":s_subtraining,"validation":s_validating}

#check stratification

check_stratification(stringent_nogsr, stringent_splits)



# export data splits
os.chdir("/Users/carolinaierardi/Documents/Academics/Project/Data/Modelling")

with open("sensitive_splits.pkl", "wb") as f:
     pickle.dump(sens_splits, f)
     
with open("stringent_splits.pkl", "wb") as f:
     pickle.dump(stringent_splits, f)
     
    


