#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:51:08 2024

@author: carolinaierardi
"""

#Title: connectivity.py
#Author: CMI
#Date: 10.09.2024
#Version: 1.0
#Purpose: perform analysis of predictive networks of ASD
#Notes: script depends on v3_ccpm_functions.py to perform analysis

import os                               #directory changing
import numpy as np 
import scipy as sp
from scipy import stats
import math
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns                                 #create boxplot
import nibabel as nib 
from nilearn import plotting                          # brain network plotting
from scipy.stats import gaussian_kde
import matplotlib.collections as clt
from scipy.spatial import distance

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Processed/Scripts")


from v3_ccpm_functions import *

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


#%% Import data

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Processed/Data")

with open("stringent_data.pkl", "rb") as f:        #load vectorised upper triangle (no GSR)
    stringent_data = pickle.load(f)                #should be 727 x 6670   
    

(fn_mats, fn_mats_gsr, 
 training, testing, train_pheno, test_pheno, 
 subtraining, validating, parameters, permutations) = unpack_dict(stringent_data)

significance = 0.05

#%%
train_inputs_pos, train_inputs_neg, edges_pos, edges_neg, test_scores_pos, test_scores_neg = get_ptcp_scores(fn_mats[:,:,training], train_pheno, significance, 
                                                                                                             subtraining_list = subtraining, validating_list = validating)
# Obtain predictive networks

training_mats = fn_mats[:,:,training]
numsubs = training_mats.shape[2]                                              #the number of participants if set as the thrid dimnesion of the input
training_mats_r = np.reshape(training_mats,[-1,numsubs])                      #reshape the input matrices to get number of edges x number of subjects - aka vectorize the matrix

predictive_pos_network, predictive_neg_network = find_networks(edges_pos, edges_neg, training_mats_r, nfolds = 3)  #find the positive and negative networks across folds
totpredictive_pos_network, totpredictive_neg_network = find_networks(edges_pos, edges_neg, training_mats_r, nfolds = 5)  #find the positive and negative networks across folds

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Processed/Data")

np.savetxt("stringent_positive_matrix.txt", predictive_pos_network, fmt='%i', delimiter=",")  #save positive matrix as text
np.savetxt("stringent_negative_matrix.txt", predictive_neg_network, fmt='%i', delimiter=",")  #save negative matrix as text

#%%Now for GSR data

train_inputs_pos_gsr, train_inputs_neg_gsr, edges_pos_gsr, edges_neg_gsr, test_scores_pos_gsr, test_scores_neg_gsr = get_ptcp_scores(fn_mats_gsr[:,:,training], train_pheno, significance, 
                                                                                                             subtraining_list = subtraining, validating_list = validating)
# Obtain predictive networks

training_mats_gsr = fn_mats_gsr[:,:,training]
numsubs = training_mats_gsr.shape[2]                                              #the number of participants if set as the thrid dimnesion of the input
training_mats_r_gsr = np.reshape(training_mats_gsr,[-1,numsubs])                      #reshape the input matrices to get number of edges x number of subjects - aka vectorize the matrix

predictive_pos_network_gsr, predictive_neg_network_gsr = find_networks(edges_pos_gsr, edges_neg_gsr, training_mats_r_gsr, nfolds = 3)  #find the positive and negative networks across folds
totpredictive_pos_network_gsr, totpredictive_neg_network_gsr = find_networks(edges_pos_gsr, edges_neg_gsr, training_mats_r_gsr, nfolds = 5)  #find the positive and negative networks across folds


np.savetxt("stringent_positive_matrix_gsr.txt", predictive_pos_network_gsr, fmt='%i', delimiter=",")  #save positive matrix as text
np.savetxt("stringent_negative_matrix_gsr.txt", predictive_neg_network_gsr, fmt='%i', delimiter=",")  #save negative matrix as text


#%% Dice coefficients


def calculate_dice(predict_net, predict_net_gsr):
    """
    

    Parameters
    ----------
    predict_net : array of shape n edges x n edges
        mask with 1s corresponding to edges in the predictive
        network.
    predict_net_gsr : array of shape n edges x n edges
        mask with 1s corresponding to edges in the predictive
        network (data with GSR.

    Returns
    -------
    float
        Dice coefficient for two networks.

    """
    
    vecnet = predict_net.flatten().astype(bool)           #flatten positive network and make as bool
    vecnet_gsr = predict_net_gsr.flatten().astype(bool)        #flatten positive network and make as bool

    dice_diss = distance.dice(vecnet, vecnet_gsr) #perform dice simmiliarity between the networks with and without GSR
    #dice_coef = 1 - dice_diss_pos                           #dice coeff is given by 1 - dissimilarity
    
    print(f"Dice coefficient: {1 - dice_diss}")
    
    return 1 - dice_diss
      
dice_coef_pos = calculate_dice(predictive_pos_network, predictive_pos_network_gsr)
dice_coef_neg = calculate_dice(predictive_neg_network, predictive_neg_network_gsr)

#%% Make figures

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Analysis/Figures")


fig, axs = plt.subplot_mosaic("AB",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Positive networks 60% folds - Dice = {dice_coef_pos:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}  
axs['A'].imshow(predictive_pos_network, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['A'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['A'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['A'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['B'].imshow(predictive_pos_network_gsr, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['B'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['B'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['B'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('positive_mats.png',bbox_inches='tight')

fig, axs = plt.subplot_mosaic("CD",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Negative networks 60% folds - Dice = {dice_coef_neg:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2) 
axs['C'].imshow(predictive_neg_network, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['C'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['C'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['C'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['D'].imshow(predictive_neg_network_gsr, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['D'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['D'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['D'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('negative_mats.png',bbox_inches='tight')

#%% brain plots

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Processed/Data")

aal_atlas = nib.load('aal_roi_atlas.nii')                       #get nifti file with the atlas
dis_matrix = plotting.find_parcellation_cut_coords(aal_atlas)   #calculate node coordinates


def make_colorbar_brain(network, cmap = "viridis"):
    
    cmap = plt.cm.get_cmap(cmap)                                                   #get colormap
    norm = Normalize(vmin=np.min(3*(np.sum(network, axis = 0))),
                     vmax=np.max(3*(np.sum(network, axis = 0))))         #normalise values
    sm = ScalarMappable(cmap=cmap, norm=norm)
    
    return sm                                                                   #set empty array for scalar mappable


os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Analysis/Figures")

    
sns.set(font_scale = 1.1)                                       #set scale for figure
fig, axs = plt.subplot_mosaic("AB",figsize=(6,3))               #get mosaic plot 
fig.suptitle(f"Positive networks 60% folds",**hfont, fontsize = 20)

fig.tight_layout(h_pad = 4)                                     #tight layout
plt.rcdefaults()

plotting.plot_connectome(np.zeros(predictive_pos_network.shape),                    # network
                          node_coords=dis_matrix,                                   # node coordinates
                          node_color=3*(np.sum(predictive_pos_network, axis = 0)),  # node colors (here, uniform)
                          node_size=(np.sum(predictive_pos_network, axis = 0))*2,   #node sizes
                          display_mode = 'z',
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("No GSR", **hfont,fontsize = 20)                                 #add subtitle
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=12, weight='bold')                                                 #add the letter at the corner of the plot

sm = make_colorbar_brain(predictive_pos_network, cmap = "viridis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])               #set colorbar



plotting.plot_connectome(np.zeros(predictive_pos_network_gsr.shape),                #network
                          node_coords=dis_matrix,                                   #node coordinates
                          node_color=3*np.sum(predictive_pos_network_gsr, axis = 0),#node colors (here, uniform)
                          node_size=(np.sum(predictive_pos_network_gsr, axis = 0))*2,
                          display_mode = 'z',
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("GSR",**hfont, fontsize = 20)                                    #subtitle
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=12, weight='bold')  

sm = make_colorbar_brain(predictive_pos_network_gsr, cmap = "viridis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])

plt.savefig('brain_plots_pos.png',bbox_inches='tight')

#Negative networks 

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("CD",figsize=(6,3))                                   #get mosaic plot 
fig.suptitle(f"Negative networks 60% folds",**hfont, fontsize = 20)
fig.tight_layout(h_pad = 4)
plt.rcParams['image.cmap'] = 'magma'

plotting.plot_connectome(np.zeros(predictive_neg_network.shape),                     # network
                          node_coords=dis_matrix,                                    # node coordinates
                          node_color=2*np.sum(predictive_neg_network, axis = 0),     # node colors (here, uniform)
                          node_size=(np.sum(predictive_neg_network, axis = 0))*2, 
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["C"]) 

axs['C'].set_title("No GSR",**hfont, fontsize = 20)                                  # add text & p-value label
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=12, weight='bold')         

sm = make_colorbar_brain(predictive_neg_network, cmap = "magma")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['C'])


plotting.plot_connectome(np.zeros(predictive_neg_network_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*np.sum(predictive_neg_network_gsr, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(predictive_neg_network_gsr, axis = 0))*2,
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["D"]) 
axs['D'].set_title("GSR", **hfont, fontsize = 20) # add text & p-value label
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=12, weight='bold')                                                  #add the letter at the corner of the plot

sm = make_colorbar_brain(predictive_neg_network_gsr, cmap = "magma")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['D'])

plt.savefig('brain_plots_neg.png',bbox_inches='tight')


#%% Regional dice coefficients


def regional_dice(predict_net, predict_net_gsr):
    
    reg_dice = []                                                        #set up empty list

    for i in range(len(predict_net)):                                    #for all nodes in network
            
        dice_diss = distance.dice(predict_net[i], predict_net_gsr[i])    #dice dissimilarity
        reg_dice += [1 - dice_diss]                                      #dice coeff is given by 1 - dissimilarity
        
    return reg_dice


reg_dice_pos = regional_dice(predictive_pos_network, predictive_pos_network_gsr)
reg_dice_neg = regional_dice(predictive_neg_network, predictive_neg_network_gsr)



os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Processed/Data")

aal_labels = pd.read_csv('./aal_labels.csv')         #download labels file
aal_labels = aal_labels.drop(0).reset_index()        #drop the first unnecessary row 
aal_labels = aal_labels.drop(["index"], axis = 1)    #drop the old index column
aal_labels.columns = ['numeric','labels']            #rename the column names

order_pos_reg = aal_labels.iloc[np.argsort(reg_dice_pos)]   #make a column with the sorted regional dice coefficient labels 
order_pos_reg['DC'] = np.sort(reg_dice_pos)                 #add a label with how much the coefficient actually is 

order_neg_reg = aal_labels.iloc[np.argsort(reg_dice_neg)]   #make a column with the sorted regional dice coefficient labels 
order_neg_reg['DC'] = np.sort(reg_dice_neg)                 #add a label with how much the coefficient actually is 

aal_labels_dice = aal_labels
aal_labels_dice['pos dice'] = reg_dice_pos
aal_labels_dice['neg dice'] = reg_dice_neg

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Analysis/Figures")

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("A;B",figsize=(10,8))                                   #get mosaic plot 

fig.tight_layout(h_pad = 4)
                                                          #tight layout so there is no overlay between plots
plt.rcParams['image.cmap'] = 'cividis'

plotting.plot_connectome(np.zeros(predictive_pos_network.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_pos ,
                          node_size= np.multiply(reg_dice_pos,300),
                          # node sizes (here, uniform)
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("Positive - 60% folds", **hfont,fontsize = 20) # add text & p-value label
axs['A'].text(-0.1, 1.1, 'A',**hfont, transform=axs['A'].transAxes, 
            size=20, weight='bold')                                                  #add the letter at the corner of the plot


sm = make_colorbar_brain(predictive_neg_network, cmap = "cividis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])


plotting.plot_connectome(np.zeros(predictive_pos_network.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_neg ,
                          node_size=np.multiply(reg_dice_neg,300),
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("Negative 60% folds", **hfont,fontsize = 20) # add text & p-value label
axs['B'].text(-0.1, 1.1, 'B', **hfont,transform=axs['B'].transAxes, 
            size=20, weight='bold')  


sm = make_colorbar_brain(predictive_neg_network, cmap = "cividis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])
plt.subplots_adjust(right=1.15)

plt.savefig('regional_dice.png',bbox_inches='tight')

#%% 100% folds

dice_coef_pos_tot = calculate_dice(totpredictive_pos_network, totpredictive_pos_network_gsr)
dice_coef_neg_tot = calculate_dice(totpredictive_neg_network, totpredictive_neg_network_gsr)

      
fig, axs = plt.subplot_mosaic("AB",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Positive networks 100% folds - Dice = {dice_coef_pos_tot:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

hfont = {'fontname':'Arial'}  
axs['A'].imshow(totpredictive_pos_network, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['A'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['A'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['A'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['B'].imshow(totpredictive_pos_network_gsr, cmap = 'GnBu')                                    # line corresponding to empirical data
axs['B'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['B'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['B'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('totpositive_mats.png',bbox_inches='tight')

fig, axs = plt.subplot_mosaic("CD",figsize=(12,6))                                   #get mosaic plot 
fig.suptitle(f"Negative networks 100% folds - Dice = {dice_coef_neg_tot:.3f}", fontsize = 40)

fig.tight_layout(h_pad = 2) 
axs['C'].imshow(totpredictive_neg_network, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['C'].set_title("No GSR",**hfont, fontsize = 40) # add text & p-value label
axs['C'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['C'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

axs['D'].imshow(totpredictive_neg_network_gsr, cmap = 'PuRd')                                    # line corresponding to empirical data
axs['D'].set_title("GSR",**hfont, fontsize = 40) # add text & p-value label
fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots
axs['D'].set_xlabel("ROIs",**hfont, fontsize = 20);
axs['D'].set_ylabel("ROIs",**hfont, fontsize = 20);
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot

plt.savefig('totnegative_mats.png',bbox_inches='tight')

#Brain plots

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("AB",figsize=(6,3))                                   #get mosaic plot 
fig.suptitle(f"Positive networks 100% folds",**hfont, fontsize = 20)

fig.tight_layout(h_pad = 4)
                                                          #tight layout so there is no overlay between plots
plt.rcdefaults()

plotting.plot_connectome(np.zeros(totpredictive_pos_network.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*(np.sum(totpredictive_pos_network, axis = 0)),            # node colors (here, uniform)
                          node_size=(np.sum(totpredictive_pos_network, axis = 0))*2,
                          display_mode = 'z',
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("No GSR", **hfont,fontsize = 20) # add text & p-value label
axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=12, weight='bold')                                                  #add the letter at the corner of the plot

sm = make_colorbar_brain(totpredictive_pos_network, cmap = "viridis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])

plotting.plot_connectome(np.zeros(totpredictive_pos_network_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*np.sum(totpredictive_pos_network_gsr, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(totpredictive_pos_network_gsr, axis = 0))*2,
                          display_mode = 'z',
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("GSR",**hfont, fontsize = 20) # add text & p-value label
axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=12, weight='bold')  

sm = make_colorbar_brain(totpredictive_pos_network_gsr, cmap = "viridis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])

plt.savefig('tot_brain_plots_pos.png',bbox_inches='tight')

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("CD",figsize=(6,3))                                   #get mosaic plot 
fig.suptitle(f"Negative networks 100% folds",**hfont, fontsize = 20)
fig.tight_layout(h_pad = 4)
plt.rcParams['image.cmap'] = 'magma'

plotting.plot_connectome(np.zeros(totpredictive_neg_network.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=2*np.sum(totpredictive_neg_network, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(totpredictive_neg_network, axis = 0))*2, 
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["C"]) 

axs['C'].set_title("No GSR",**hfont, fontsize = 20) # add text & p-value label
axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
            size=12, weight='bold')         

sm = make_colorbar_brain(totpredictive_neg_network, cmap = "magma")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['C'])

plotting.plot_connectome(np.zeros(totpredictive_neg_network_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color=3*np.sum(totpredictive_neg_network_gsr, axis = 0),            # node colors (here, uniform)
                          node_size=(np.sum(totpredictive_neg_network_gsr, axis = 0))*2,
                          display_mode = 'z',
                          
                          figure=fig, axes = axs["D"]) 
axs['D'].set_title("GSR", **hfont, fontsize = 20) # add text & p-value label
axs['D'].text(-0.1, 1.1, 'D', transform=axs['D'].transAxes, 
            size=12, weight='bold')                                                  #add the letter at the corner of the plot

sm = make_colorbar_brain(totpredictive_neg_network_gsr, cmap = "magma")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['D'])

plt.savefig('tot_brain_plots_neg.png',bbox_inches='tight')


#%% Regional dice coefficients


reg_dice_pos_tot = regional_dice(totpredictive_pos_network, totpredictive_pos_network_gsr)
reg_dice_neg_tot = regional_dice(totpredictive_neg_network, totpredictive_neg_network_gsr)

sns.set(font_scale = 1.1)
fig, axs = plt.subplot_mosaic("A;B",figsize=(10,8))                                   #get mosaic plot 

fig.tight_layout(h_pad = 4)
                                                          #tight layout so there is no overlay between plots
plt.rcParams['image.cmap'] = 'cividis'

plotting.plot_connectome(np.zeros(totpredictive_pos_network.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_pos_tot ,
                          node_size= np.multiply(reg_dice_pos_tot,300),
                          # node sizes (here, uniform)
                          figure=fig, axes = axs["A"]) 
axs['A'].set_title("Positive - 100% folds", **hfont,fontsize = 20) # add text & p-value label
axs['A'].text(-0.1, 1.1, 'A',**hfont, transform=axs['A'].transAxes, 
            size=20, weight='bold')                                                  #add the letter at the corner of the plot


sm = make_colorbar_brain(totpredictive_pos_network, cmap = "cividis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['A'])


plotting.plot_connectome(np.zeros(totpredictive_pos_network_gsr.shape),                         # network
                          node_coords=dis_matrix,        # node coordinates
                          node_color =reg_dice_neg_tot ,
                          node_size=np.multiply(reg_dice_neg_tot,300),
                          figure=fig, axes = axs["B"]) 
axs['B'].set_title("Negative 100% folds", **hfont,fontsize = 20) # add text & p-value label
axs['B'].text(-0.1, 1.1, 'B', **hfont,transform=axs['B'].transAxes, 
            size=20, weight='bold')  


sm = make_colorbar_brain(totpredictive_pos_network_gsr, cmap = "cividis")
sm.set_array([])  # Set empty array for scalar mappable
cbar = plt.colorbar(sm, shrink=.8, location = 'right', ax = axs['B'])
plt.subplots_adjust(right=1.15)

plt.savefig('tot_regional_dice.png',bbox_inches='tight')

