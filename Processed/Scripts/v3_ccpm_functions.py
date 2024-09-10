#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:20:56 2023

@author: carolinaierardi
"""


#Script Name: my_cpm_functions.py
#Author: CMI
#Date: 15.04.23
#Version: 1.0
#Purpose: these are the functions built to run categorical CPM with feature selection
#Notes: functions for cpm for categorical variables



import os                               #directory changing
import numpy as np 
import scipy as sp
from scipy import stats
from scipy.spatial import distance
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import random
import glob
from nilearn.datasets import fetch_abide_pcp
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score    
import math
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns                                 #create boxplot
import nibabel as nib 
from nilearn import plotting                          # brain network plotting
from scipy.stats import gaussian_kde
import matplotlib.collections as clt


def dis(x1, y1, z1, x2, y2, z2):                           #create a function to calculate the distance between two nodes
      
    d = math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) +
                math.pow(z2 - z1, 2)* 1.0)
    return(d)
    

def raincloud(data_x, feature1, feature2, xlabel, title, ax):

    boxplots_colors = ['lightblue', 'pink'] # Create a list of colors for the boxplots based on the number of features you have
    violin_colors = ['darkblue', 'red'] # Create a list of colors for the violin plots based on the number of features you have
    scatter_colors = ['darkblue', 'darksalmon']
       
# Boxplot data
    bp = ax.boxplot(data_x, patch_artist = True, vert = False)

# Change to the desired color and add transparency
    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        
    for median,color in zip(bp['boxes'], boxplots_colors):
        median.set_color(color)

# Violinplot data
    vp = ax.violinplot(data_x, points=500, 
                showmeans=False, showextrema=False, showmedians=False, vert=False)

    for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
        m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
    # Change to the desired color
        b.set_color(violin_colors[idx])

# Create a list of colors for the scatter plots based on the number of features you have

# Scatterplot data
    for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax.scatter(features, y, s=.3, c=scatter_colors[idx])
       
    plt.sca(ax)    
    plt.yticks(np.arange(1,3,1), [feature1, feature2], size = 80)  # Set text labels.
    plt.xlabel(xlabel)
    plt.title(title)
    
    
def mean_positive(L):
    """Function that calculates mean only for positive numbers"""

    # Get all positive numbers into another list
    pos_only = [x for x in L if x > 0]
    if pos_only:
        return sum(pos_only) /  len(pos_only)
    raise ValueError('No postive numbers in input')
    

def add_subnetwork_lines(hm,roi_nums,*line_args,**line_kwargs):
    hm.hlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_xlim(),*line_args,**line_kwargs); hm.vlines([0]+[i-0.25 for i in np.cumsum(roi_nums)], *hm.get_ylim(),*line_args,**line_kwargs)

    
def calc_inputs(ipmat, pheno, g1, g2, sig_lvl): 
    
    """
    ipmat: matrix with dimensions total n of edges x total number of participants in training
    g1: patients
    g2: controls
    sig_lvl: level of significance to be used in the t-tests
    The function takes the inputs calculates t-tests for difference in connectivity in edges,
    stores the indices of the edges considered significant 
    as well as the summary scores for the participants
    
    """
    
    
    cc = [] #changed this line to perform a t-test at every edge instead
    
    cc += [stats.ttest_ind(ipmat[edge,g1],ipmat[edge,g2]) for edge in range(0, len(ipmat))]
    
    tmat=np.array([c[0] for c in cc])                   #t-values
    pmat=np.array([c[1] for c in cc])                   #p-values
    tmat=np.reshape(tmat,[int(np.sqrt(len(ipmat))),int(np.sqrt(len(ipmat)))])                     #reshape to fc size
    pmat=np.reshape(pmat,[int(np.sqrt(len(ipmat))),int(np.sqrt(len(ipmat)))])                     #reshape to fc size
    posedges=(tmat > 0) & (pmat < sig_lvl)              #only select the ones below 0.05 signficance
    posedges=posedges.astype(int)                       #make as integers
    negedges=(tmat < 0) & (pmat <sig_lvl)               #only select the ones below 0.05 signficance
    negedges=negedges.astype(int)                       #make as integers
    pe=ipmat[posedges.flatten().astype(bool),:]         #values for edges with difference between each other
    ne=ipmat[negedges.flatten().astype(bool),:]         #values for edges with difference between each other
    pe=pe.sum(axis=0)/2                                 #summary statistic for each  (pos)
    ne=ne.sum(axis=0)/2                                 #summary statistic for each participant (neg)
    pe = pe.reshape(-1,1)                               #reshape to fit logistic regression requirements
    ne = ne.reshape(-1,1)                               #reshape to fit logistic regression requirements
    
    neg_indices = negedges.flatten()                    #flatten the matrix
    neg_indices = np.where(neg_indices == True)[0]      #get the significant ones
    
    pos_indices = posedges.flatten()                    #flatten the matrix
    pos_indices = np.where(pos_indices == True)[0]      #get the significant ones
    
    return pe, ne, posedges, negedges, pos_indices, neg_indices


def unpack_dict(dict_data):
    
    """
    Function unpacks the dictionary with all data for an analysis into different 
    variables
    """
            
    permutations = dict_data["permutations"]
    fn_mats = dict_data["No GSR ROIs"]
    fn_mats_gsr = dict_data["GSR ROIs"]
    training = dict_data["training"]
    testing = dict_data["training"]
    subtraining = dict_data["subtraining"]
    validating = dict_data["validating"]
    phenotype = dict_data["phenotype"]
    parameters = dict_data["parameters"]
    
    train_pheno = phenotype[training]
    test_pheno = phenotype[testing]
    
    return (fn_mats, fn_mats_gsr, 
            training, testing, train_pheno, test_pheno, 
            subtraining, validating, parameters, permutations)



def get_ptcp_scores(training_matrices, train_phenotype, signf, testing = None, subtraining_list = None, validating_list = None):
    
    """
    training_matrices: matrix with shape n nodes x n nodes x n ptcps
    train_phenotype: list with phenotype of each participant
    subtraining_list: iterable list with indices of participants in the subtratining set in each fold
    validating_list: iterable list with indices of participants in the validating set in each fold
    signf: level of significance to be used in group t-tests
    

    Returns
    -------
    train_inputs_pos: iterable list with the training inputs for the ML pipeline for each fold for positive edges
    train_inputs_neg: iterable list with the training inputs for the ML pipeline for each fold for negative edges
    edges_pos: list with edges indices deemed to have a positive difference between groups in each fold
    edges_neg: list with edges indices deemed to have a negative difference between groups in each fold
    test_scores_pos: iterable list with the validating inputs for prediction in the ML pipeline for each fold for positive edges
    test_scores_neg: iterable list with the validating inputs for prediction in the ML pipeline for each fold for negative edges

    """
    numsubs = training_matrices.shape[2]                                              #the number of participants if set as the thrid dimnesion of the input
    training_mats_r = np.reshape(training_matrices,[-1,numsubs])                      #reshape the input matrices to get number of edges x number of subjects - aka vectorize the matrix
    
    if testing is None: 
        train_inputs_pos = [] #empty list for positive scores for participants in each fold
        train_inputs_neg = [] #empty list for negative scores for participants in each fold
        edges_pos = []        #empty list for positive edges found significant in each fold
        edges_neg = []        #empty list for negative edges found significant in each fold
        test_scores_pos = []  #empty list for positive scores for participants in each validation fold
        test_scores_neg = []  #empty list for negative scores for participants in each validation fold
        

        for i in range(len(subtraining_list)):                                    #for each fold
            g1 = np.where(train_phenotype[subtraining_list[i]] == 1)[0]               #where the group is for patients
            g2 = np.where(train_phenotype[subtraining_list[i]] == 2)[0]               #where the group is for controls
            
            train_inpp, train_inpn, posedges, negedges, pos_indices, neg_indices = calc_inputs(training_mats_r[:,subtraining_list[i]],train_phenotype[subtraining_list[i]],g1,g2,signf) #perform t-tests to assess which edges are significant 
    
            train_inputs_pos.append(train_inpp) #save the summary scores for each fold
            train_inputs_neg.append(train_inpn) #save the summary scores for each fold
    
            edges_pos.append(pos_indices)       #save the index of the significant positive edges
            edges_neg.append(neg_indices)       #save the index of the significant negative edges
 
            pe=np.sum(training_mats_r[:,validating_list[i]][posedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the positive edges found predictive in the step before  
            ne=np.sum(training_mats_r[:,validating_list[i]][negedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the negative edges found predictive in the step before

            test_inpp = pe.reshape(-1,1)        #reshape to fit logistic regression requirements
            test_inpn = ne.reshape(-1,1)        #reshape to fit logistic regression requirements
    
            test_scores_pos.append(test_inpp)   #list with the positive scores for validating participants in each fold
            test_scores_neg.append(test_inpn)   #list with the negative scores for validating participants in each fold
            
        return train_inputs_pos, train_inputs_neg, edges_pos, edges_neg, test_scores_pos, test_scores_neg

    else: 
          g1 = np.where(train_phenotype == 1)[0]               #where the group is for patients
          g2 = np.where(train_phenotype == 2)[0]               #where the group is for controls
          
          train_inpp, train_inpn, posedges, negedges, pos_indices, neg_indices = calc_inputs(training_mats_r,train_phenotype,g1,g2,signf) #perform t-tests to assess which edges are significant 
         
          numsubs = testing.shape[2]                                              #the number of participants if set as the thrid dimnesion of the input
          testing_mat = np.reshape(testing,[-1,numsubs])                      #reshape the input matrices to get number of edges x number of subjects - aka vectorize the matrix
         
          pe=np.sum(testing_mat[posedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the positive edges found predictive in the step before  
          ne=np.sum(testing_mat[negedges.flatten().astype(bool),:], axis=0)/2  #get the summary scores for the participants in the testing set, based on the negative edges found predictive in the step before

          test_inpp = pe.reshape(-1,1)        #reshape to fit logistic regression requirements
          test_inpn = ne.reshape(-1,1)        #reshape to fit logistic regression requirements
 
    return train_inpp, train_inpn, posedges, negedges, test_inpp, test_inpn


def MLpipeline(training_matrices, train_phenotype, subtraining_, validating_, 
              parameters, testing_matrices, testing_pheno, significance = 0.05, n_folds = 5): 
    """
    
    Parameters
    ----------
    training_mats: connectivity matrices of participants
    train_pheno : list
        Phenotypes of all participants.
    subtraining: nested list 
        Indices of participants to be put in training set in each fold
    validating:     
        Indices of participants to be put in validating set in each fold
    n_folds: integer
        Number of folds in the CV. If None is given, default = 5
    significance: integer
        level of significance to be used in t-tests, default = .05
    scoring: dict
        performance metrics to be calculated in the Grid Search
    param_grid: dict
        Hyperparameter grid to be used    
    
    Returns
    -------
   Average metrics across folds for positive and negative models

    """
    train_inputs_pos, train_inputs_neg, edges_pos, edges_neg, test_scores_pos, test_scores_neg = get_ptcp_scores(training_matrices, train_phenotype, significance, subtraining_list= subtraining_, validating_list = validating_)
 
    fold_accP = []        #store average metric across folds
    fold_accN = []
    
    fold_rocP = []
    fold_rocN = []
    
    fold_precP = []
    fold_precN = []
    
    for p in range(len(parameters)): #for each parameter set in the list 
                
        if len(parameters[p]) == 3:   #if it has length of 3, assign positive and negative models with C
            clfpos = LogisticRegression(solver = parameters[p][0], penalty = parameters[p][1], C = parameters[p][2], random_state = 1, max_iter=100000)
            clfneg = LogisticRegression(solver = parameters[p][0], penalty = parameters[p][1], C = parameters[p][2], random_state = 1, max_iter=100000)
            
        else:         #else, ignore the C parameter 
            clfpos = LogisticRegression(solver = parameters[p][0], penalty = None, random_state = 1, max_iter=100000)
            clfneg = LogisticRegression(solver = parameters[p][0], penalty = None, random_state = 1, max_iter=100000)
            
        bal_acc_p = []        #best bal accuracy score positive on held out test data
        bal_acc_n = []        #best bal accuracy score negative on held out test data

        roc_p = []            #best AUC-ROC score positive on held out test data
        roc_n = []            #best AUC-ROC score negative on held out test data

        prec_p = []           #best AvgPrec score positive on held out test data
        prec_n = []           #best AvgPrec score negative on held out test data
          
    
        for i in range(n_folds): #for each fold
            
        #assign the variables according to the fold
            train_pos = train_inputs_pos[i]
            train_neg = train_inputs_neg[i]
            val_pos = test_scores_pos[i]
            val_neg = test_scores_neg[i]
            pheno_train_fold = train_phenotype[subtraining_[i]]
            pheno_val_fold = train_phenotype[validating_[i]]
            
            pos_model = clfpos.fit(train_pos, pheno_train_fold)              #fit positive model to the data
            neg_model = clfneg.fit(train_neg, pheno_train_fold)              #fit negative model to the data
        
            predp = pos_model.predict(val_pos)                               #predict to validating set 
            predn = neg_model.predict(val_neg)                               #predict to validating set
                
            # calculate metrics for that fold and add to iterative list
            bal_acc_p += [balanced_accuracy_score(pheno_val_fold, predp)] 
            bal_acc_n += [balanced_accuracy_score(pheno_val_fold, predn)]
        
            roc_p += [roc_auc_score(pheno_val_fold, predp, average = "weighted")]
            roc_n += [roc_auc_score(pheno_val_fold, predn, average="weighted")]
        
            prec_p += [average_precision_score(pheno_val_fold, pos_model.predict_proba(val_pos)[:,0], average="weighted")]
            prec_n += [average_precision_score(pheno_val_fold, neg_model.predict_proba(val_neg)[:,0], average="weighted")]
            
        #calculate the average across folds
        fold_accP.append(np.mean(bal_acc_p))  
        fold_accN.append(np.mean(bal_acc_n))
        
        fold_rocP.append(np.mean(roc_p))
        fold_rocN.append(np.mean(roc_n)) 
        
        fold_precP.append(np.mean(prec_p))
        fold_precN.append(np.mean(prec_n))
       
    #find the best model according to the highest average across folds for AUC-ROC
    bmp = np.where(fold_accP == np.max(fold_accP))[0][0]
    bmn = np.where(fold_accN == np.max(fold_accN))[0][0]
    
    #save a dictionary with the summary of the best parameters and metrics for validation
    val_dict = {"Best Positive params": parameters[bmp], 
                "Best Negative params": parameters[bmn],
                "AUC pos": fold_rocP[bmp],
                "AUC neg": fold_rocN[bmn], 
                "Bal acc pos": fold_accP[bmp],
                "Bal acc neg": fold_accN[bmn],
                "Prec pos": fold_precP[bmp],
                "Prec neg": fold_precN[bmn]}
    
    print("Train and validation... done")
    #now apply the best parameters to the whole training dataset
    if len(parameters[bmp]) == 3:
        clfpostot = LogisticRegression(solver = parameters[bmp][0], penalty = parameters[bmp][1], C = parameters[bmp][2], random_state = 1, max_iter=100000)
    else:
        clfpostot = LogisticRegression(solver = parameters[bmp][0],penalty = None, random_state = 1, max_iter=100000)
    if len(parameters[bmn]) == 3: 
        clfnegtot = LogisticRegression(solver = parameters[bmn][0], penalty = parameters[bmn][1], C = parameters[bmn][2], random_state = 1, max_iter=100000)
    else: 
        clfnegtot = LogisticRegression(solver = parameters[bmn][0], penalty = None, random_state = 1, max_iter=100000)#, max_iter=100000
        
        #calculate scores for whole train set and test set 
    
    train_inputs_pos, train_inputs_neg, pos_indices, neg_indices, test_inpp, test_inpn = get_ptcp_scores(training_matrices, train_phenotype, significance, testing=testing_matrices)
    
    ppos_model = clfpostot.fit(train_inputs_pos, train_phenotype)    
    nneg_model = clfnegtot.fit(train_inputs_neg, train_phenotype)
    
    predp = ppos_model.predict(test_inpp)                                         #predict to validating set 
    predn = nneg_model.predict(test_inpn)    

    
    test_dict = {"AUC pos": roc_auc_score(testing_pheno, predp, average="weighted"),
                 "AUC neg": roc_auc_score(testing_pheno, predn, average="weighted"),
                 "Bal acc pos": balanced_accuracy_score(testing_pheno, predp),
                 "Bal acc neg": balanced_accuracy_score(testing_pheno, predn),
                 
                 "avg Prec pos": average_precision_score(testing_pheno, ppos_model.predict_proba(test_inpp)[:,0], average="weighted"),
                 "avg Prec neg": average_precision_score(testing_pheno, nneg_model.predict_proba(test_inpn)[:,0], average="weighted")
                 }
    
    print("Testing... done")
    
    return val_dict, test_dict


def find_networks(allposedges, allnegedges, X, nfolds = None): 
    
    """allposedges: a list with indices of positive edges considered significant in each fold of a cross-validation
    allnegedges: same as above for negative edges
    X: a matrix with dimensions number of edges x n of participants 
    The function will take the inputs to calculate overall predictive networks. 
    This will involve verifying which edges appear in enough folds as being considered signifciant
    Outputs are binary connectivity matrices with 1 where edges are significant and 0 where not significant"""
    
    all_posedges = [item for sublist in allposedges for item in sublist]  #flatten list for significant positive edges
    all_negedges = [item for sublist in allnegedges for item in sublist]   #flatten list for significant negative edges

    all_posedges = Counter(all_posedges)  #count how many occurences the significant edges have (pos)
    all_negedges = Counter(all_negedges)  #count how many occurences the significant edges have (neg)
     
    predictive_pos = {i for i in all_posedges if all_posedges[i] >= nfolds}    #if there are 3 or more occurences, then consider it a predictive edge in the network 
    predictive_neg = {i for i in all_negedges if all_negedges[i] >= nfolds}    #if there are 3 or more occurences, then consider it a predictive edge in the network
     
    predictive_pos_network = np.zeros([1,X.shape[0]])                     #create matrix to input the signficiant edges
    predictive_pos_network[0,list(predictive_pos)] = 1                    #make those predictive into 1
    predictive_pos_network = predictive_pos_network.reshape([int(math.sqrt(X.shape[0])),int(math.sqrt(X.shape[0]))])  #reshape the matrix to the node x node matrix initiall set
     
    predictive_neg_network = np.zeros([1,X.shape[0]])                     #create matrix to input the signficiant edges
    predictive_neg_network[0,list(predictive_neg)] = 1                    #make those predictive into 1
    predictive_neg_network = predictive_neg_network.reshape([int(math.sqrt(X.shape[0])),int(math.sqrt(X.shape[0]))])  #reshape the matrix to the node x node matrix initiall set      
     
    return predictive_pos_network, predictive_neg_network



def vectorized_MLpipeline(training_matrices, train_phenotype, subtraining_, validating_, 
              parameters, testing_matrices, testing_pheno, significance = 0.05, n_folds = 5): 
    """
    
    Parameters
    ----------
    training_mats: connectivity matrices of participants
    train_pheno : list
        Phenotypes of all participants.
    subtraining: nested list 
        Indices of participants to be put in training set in each fold
    validating:     
        Indices of participants to be put in validating set in each fold
    n_folds: integer
        Number of folds in the CV. If None is given, default = 5
    significance: integer
        level of significance to be used in t-tests, default = .05
    scoring: dict
        performance metrics to be calculated in the Grid Search
    param_grid: dict
        Hyperparameter grid to be used    
    
    Returns
    -------
   Average metrics across folds for positive and negative models

    """
    
 
    fold_accP = []        #store average metric across folds    
    fold_rocP = []
    fold_precP = []
    
    for p in range(len(parameters)): #for each parameter set in the list 
                
        if len(parameters[p]) == 3:   #if it has length of 3, assign positive and negative models with C
            clfpos = LogisticRegression(solver = parameters[p][0], 
                                        penalty = parameters[p][1], 
                                        C = parameters[p][2], 
                                        random_state = 1, max_iter=10000)
            
        else:         #else, ignore the C parameter 
            clfpos = LogisticRegression(solver = parameters[p][0], 
                                        penalty = None,
                                        random_state = 1, max_iter=10000)
            
        bal_acc = []        #best bal accuracy score positive on held out test data
        roc = []            #best AUC-ROC score positive on held out test data
        prec = []           #best AvgPrec score positive on held out test data
          
    
        for i in range(n_folds): #for each fold
            
        #assign the variables according to the fold
            train = training_matrices[subtraining_[i],:]
            val = training_matrices[validating_[i],:]
            pheno_train_fold = train_phenotype[subtraining_[i]]
            pheno_val_fold = train_phenotype[validating_[i]]
            
            pos_model = clfpos.fit(train, pheno_train_fold)              #fit positive model to the data
        
            pred = pos_model.predict(val)                               #predict to validating set 
                
            # calculate metrics for that fold and add to iterative list
            roc += [roc_auc_score(pheno_val_fold, pred, average="weighted")]
            bal_acc += [balanced_accuracy_score(pheno_val_fold, pred)] 
            prec += [average_precision_score(pheno_val_fold, pos_model.predict_proba(val)[:,0], average = "weighted")]
            
        #calculate the average across folds
        fold_accP.append(np.mean(bal_acc))  
        fold_rocP.append(np.mean(roc))
        fold_precP.append(np.mean(prec))
       
    #find the best model according to the highest average across folds for AUC-ROC
    bmp = np.where(fold_accP == np.max(fold_accP))[0][0]
    
    #save a dictionary with the summary of the best parameters and metrics for validation
    val_dict = {"Best params": parameters[bmp], 
                "AUC": fold_rocP[bmp],
                "Bal acc": fold_accP[bmp],
                "Prec": fold_precP[bmp]}
    
    print("Train and validation... done")
    #now apply the best parameters to the whole training dataset
    if len(parameters[bmp]) == 3:
        clfpostot = LogisticRegression(solver = parameters[bmp][0], 
                                       penalty = parameters[bmp][1], 
                                       C = parameters[bmp][2], 
                                       random_state = 1, max_iter=10000)
    else:
        clfpostot = LogisticRegression(solver = parameters[bmp], 
                                       penalty = None,
                                       random_state = 1, max_iter=10000)
   
        #calculate scores for whole train set and test set 
    
    
    ppos_model = clfpostot.fit(training_matrices, train_phenotype)    
    pred = ppos_model.predict(testing_matrices)                                         #predict to validating set 

    
    test_dict = {"AUC": roc_auc_score(testing_pheno, pred, average="weighted"),
                 "Bal acc": balanced_accuracy_score(testing_pheno, pred),
                 "avg Prec": average_precision_score(testing_pheno, ppos_model.predict_proba(testing_matrices)[:,0], average = "weighted"),
                 }
    
    print("Testing... done")
    
    return val_dict, test_dict
    
    



