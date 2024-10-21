#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 13:45:55 2024

@author: carolinaierardi
"""

#import data and produce figures

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns                                 #create boxplot


def load_data(directory, file_pattern): 
    
    loaded_data = []

    files = [f for f in os.listdir(directory) if f.startswith(file_pattern) and f.endswith('.pkl')]
    files.sort(key=lambda f: int(f[len(file_pattern):-4]))


    # Loop through all files in the directory
    for filename in files:
        # Full path to the file
        file_path = os.path.join(directory, filename)
        
        # Load the .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # Append the data to the merged_data list
            loaded_data.append(data)
            
    return loaded_data


def get_stat(input_array, stat): 
    flat_array = [x for xs in input_array for x in xs]
    return np.array([sub[stat] for sub in flat_array])

def get_pval(input_array):
    
    """
    Parameters
    ----------
    input_array : array with metrics for permutations
        The first metric MUST be the empirical one.

    """

    return (1 + len(np.where(input_array[0] < input_array)[0])) / (len(input_array) + 1)


def dist_diff(nogsr, gsr):
    
    empirical = gsr[0] - nogsr[0]
    null = gsr[1:] - nogsr[1:]
        
    pval = (1 + len(np.where(null > empirical)[0])) / (len(null) + 1)
    return empirical, null, pval
    
    

# Initialize a list to store the merged data


def make_plot(no_gsr_array, gsr_array, stat, positive = True):
    
    metrics_gsr = get_stat(gsr_array, stat) 
    metrics_nogsr = get_stat(no_gsr_array, stat) 
    
    pval_gsr = get_pval(metrics_gsr)
    pval_nogsr = get_pval(metrics_nogsr)
    
    diff_emp, diff_null, pval_diff = dist_diff(metrics_nogsr, metrics_gsr)
    
    if positive == True: 
        c = '#48D1CC'
    else: 
        c = '#DB7093'
    title = f"{stat}"
        
    
    sns.set(font_scale = 2)
    fig, axs = plt.subplot_mosaic("ABC",figsize=(24,10))                                   #get mosaic plot 
    fig.suptitle(title, fontsize = 40)

    fig.tight_layout(h_pad = 1)                                                          #tight layout so there is no overlay between plots

    hfont = {'fontname':'Arial'}  
    N, bins, patches = axs['A'].hist(metrics_nogsr, bins = 20, density=True, 
                                     edgecolor='black', linewidth=1, color = c)
        
    # for j in range(0,len(patches)):
    #     if bins[j] < modelspos[i][0][0]:
    #         patches[j].set_facecolor(('#48D1CC'))
    #     elif bins[j] >= modelspos[i][0][0]:
    #         patches[j].set_facecolor(('#000080'))  
            
    axs['A'].axvline(metrics_nogsr[0], ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
    axs['A'].set_title(f"No GSR: Empirical {stat}: {metrics_nogsr[0]:.2f}\n(p-value = {pval_nogsr:.3f})", fontsize = 30) # add text & p-value label
    fig.tight_layout(h_pad = 1)   
    axs['A'].set_xlabel(f"{stat}",**hfont, fontsize = 30);
    axs['A'].set_ylabel("Probability",**hfont, fontsize = 30);
    axs['A'].text(-0.1, 1.1, 'A', transform=axs['A'].transAxes, 
            size=30, weight='bold')   
    fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

    N, bins, patches = axs['B'].hist(metrics_gsr, bins = 20, density=True, 
                                     edgecolor='black', linewidth=1, color = c)
 

    axs['B'].axvline(metrics_gsr[0], ls="--", color="k", lw = 5)                                    # line corresponding to empirical data
    axs['B'].set_title(f"GSR: Empirical {stat}: {metrics_gsr[0]:.2f}\n(p-value = {pval_gsr:.4f})", fontsize = 30) # add text & p-value label
    axs['B'].set_xlabel(f"{stat}", fontsize = 30);
    axs['B'].set_ylabel("Probability", fontsize = 30);
    axs['B'].text(-0.1, 1.1, 'B', transform=axs['B'].transAxes, 
            size=30, weight='bold')                                                  #add the letter at the corner of the plot
    fig.tight_layout(h_pad = 2)                                                          #tight layout so there is no overlay between plots

    N, bins, patches = axs['C'].hist(diff_null, bins = 20, density=True, 
                                     edgecolor='black', linewidth=1, color = c)
        
    # for j in range(0,len(patches)):
    #     if bins[j] < diff_emp[i][0]:
    #         patches[j].set_facecolor(('#48D1CC'))
    #     elif bins[j] >= diff_emp[i][0]:
    #         patches[j].set_facecolor(('#000080'))


    axs['C'].axvline(diff_emp, ls="--", color="k", lw = 5)                                    # line corresponding to empirical data# histogram of scores on permuted data
    axs['C'].set_xlabel(f"{stat} difference", fontsize = 30);
    axs['C'].set_ylabel("Probability", fontsize = 30);
    axs['C'].set_title(f"Accuracy difference: {diff_emp:.2f}\n (two-tailed p-value = {pval_diff:.3f})", fontsize = 30)
    axs['C'].text(-0.1, 1.1, 'C', transform=axs['C'].transAxes, 
                  size=30, weight='bold')                                                  #add the letter at the corner of the plot

    plt.savefig(f'{stat} models.png',bbox_inches='tight')



#%% Analysis

#%%% Stringent analysis

directory = '/Users/carolinaierardi/Downloads/ccpm_out/ccpm_out_strin'
strin_gsr = load_data(directory, "STRIN_GSR_testsummary_")
string_nogsr = load_data(directory, "STRIN_testsummary_")

os.chdir("/Users/carolinaierardi/Documents/Academics/CCPM/Analysis/Figures")

make_plot(string_nogsr, strin_gsr, "AUC pos", positive=True)
make_plot(string_nogsr, strin_gsr, "AUC neg", positive=False)

make_plot(string_nogsr, strin_gsr, "avg Prec pos", positive=True)
make_plot(string_nogsr, strin_gsr, "avg Prec neg", positive=False)

make_plot(string_nogsr, strin_gsr, "Bal acc pos", positive=True)
make_plot(string_nogsr, strin_gsr, "Bal acc neg", positive=False)

t = get_stat(strin_gsr,"avg Prec neg")



#%% Sensitivity analysis

directory = '/Users/carolinaierardi/Downloads/ccpm_out/ccpm_out_sens'
sens_gsr = load_data(directory, "SENS_GSR_testsummary_")
sens_nogsr = load_data(directory, "SENS_testsummary_")

make_plot(sens_nogsr, sens_gsr, "AUC pos", positive=True)
make_plot(sens_nogsr, sens_gsr, "AUC neg", positive=False)

make_plot(sens_nogsr, sens_gsr, "avg Prec pos", positive=True)
make_plot(sens_nogsr, sens_gsr, "avg Prec neg", positive=False)

make_plot(sens_nogsr, sens_gsr, "Bal acc pos", positive=True)
make_plot(sens_nogsr, sens_gsr, "Bal acc neg", positive=False)







