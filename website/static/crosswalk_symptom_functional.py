# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 02:58:41 2023

@author: vshas
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib as mpl


def link_distributions(A,B, a_val = 1,random_seed = 42):
    
    """
    Generate a 'semi-random distribution linked' value for:
        B, given a value from A
    """ 
    # Never guesses a 0 when given a 0
    
    np.random.seed(random_seed)
    
    factor = 10**3
    
    breakpoints_A = np.round((factor)*np.concatenate(([0], np.cumsum(A)/sum(A))))
    
    breakpoints_B = (factor)*np.concatenate(([0], np.cumsum(B)/sum(B)))
    
    linspace_B = -1*np.ones(shape=factor)
    

    
    for i in range(5): 
        
        # if np.round(breakpoints_B[i + 1]) == np.floor(breakpoints_B[i]):
            
        #     0
            
        # else:
        
        ints = np.arange(np.floor(breakpoints_B[i]), np.round(breakpoints_B[i+1])).astype(int)
        
        linspace_B[ints] = i
            
    # if breakpoints_A[a_val] == breakpoints_A[a_val + 1]:
        
    #     print(f"No Density In Histogram! Score,{a_val} has no ")
        
    #     out = linspace_B[int(breakpoints_A[a_val])]
    # else:
        
    A_cut = np.arange(breakpoints_A[a_val], breakpoints_A[a_val+1]).astype(int)

    out = np.random.permutation(linspace_B[A_cut])[0:9].astype(int)
    
    
    
    output = np.argmax(np.bincount(out))
        
    return output
    
def set_crosswalk_files(score_file = "score_dict.p",
                        text_file = "text_dict.p",
                        hist_file = "hist_dict.p",
                        inv_in = "BSI",
                        inv_out = "RPQ"
                        ):
    
    score_dict = pkl.load( open(score_file, "rb"))
    text_dict  = pkl.load( open(text_file, "rb" ))
    hist_dict  = pkl.load( open(hist_file, "rb" ))
    
    # find data dict key which contains both the input and output inventories
    A = [inv_in in i for i in list(score_dict.keys())]
    B = [inv_out in  i for i in list(score_dict.keys())]
    
    # Both inventory in and out must be present in the dictionary key to load the right one
    key_index = np.where([i[0] * i[1] for i in zip(A, B)])[0][0]
    dict_key = list(score_dict.keys())[key_index] # the right key is found
    
    # get the right data array for this crosswalk
    # Account for the fact that first axis should be the input axis always
    first_is_input = (inv_in == dict_key[0:3])
    if first_is_input:
        simil_arr = score_dict[dict_key]
    else:
        simil_arr = np.transpose(score_dict[dict_key])
        
    return score_dict,text_dict,hist_dict,simil_arr
    
def crosswalk_scores(input_scores,
                     
                          score_dict,
                          text_dict,
                          hist_dict,
                          simil_arr,
                          empirical_shift_down = True,
                          inv_in = "BSI" ,
                          inv_out = "RPQ" ,
                          verbose= True,
                          link_hists=True,
                          random_seed = 42,
                          ):
    
    
    num_items_in_input, num_items_predict = np.shape(simil_arr)    
    inds_identical = {}

    for i in range(num_items_predict):
        vec = simil_arr[:,i] # vector of 1 item's cosine similarities
        # For a RPQ to BSI, inds_identical is 
        # For each BSI (i.e. col in simil_arr), index of max val in col (col wise)
        inds_identical[i] = vec.argmax()  
    
    # Prediction code
    input_scores = np.asarray(input_scores)
    if empirical_shift_down:
        input_scores = input_scores - 1
    predicted_scores = -1*np.ones(num_items_predict)
    # print(simil_arr.shape)
    # print(inds_identical)
    # print(input_scores)
    for i in range(num_items_predict):
        # print(inds_identical[i])
        # inds_identical for RPQ -> BSI ; { BSI[0]: RPQ[closest];BSI[1]: ...}
        
        if link_hists:
            predicted_scores[i] = link_distributions(hist_dict[(inv_in,inds_identical[i])],
                                                     hist_dict[(inv_out,i)],
                                                     a_val = input_scores[inds_identical[i]],
                                                     random_seed = random_seed * i)
        else:
            predicted_scores[i] = input_scores[inds_identical[i]]
    predicted_scores = predicted_scores + 1 if empirical_shift_down else predicted_scores
    
    if verbose:
        print('--------------------------------------------------')
        print('Input scores for', inv_in,':')    
        [print(i,j[0],j[1]) for i,j in enumerate(zip(text_dict[inv_in], input_scores))]
        
        print('--------------------------------------------------')
        print('Predicted scores for', inv_out,':')    
        [print(i,j[0],j[1]) for i,j in enumerate(zip(text_dict[inv_out], predicted_scores))]
    
    return predicted_scores
#%%
inv_in = "RPQ"
inv_out = "BSI"
score_dict,text_dict,hist_dict,simil_arr = set_crosswalk_files(inv_in = inv_in,inv_out = inv_out)
input_scores = [2, 2, 3, 4, 3, 4, 4, 5, 5, 5, 5, 3, 3, 2, 4, 5]
# input_scores = []

predicted_scores = crosswalk_scores(input_scores, score_dict, text_dict, hist_dict, simil_arr,
                                          inv_in = inv_in,inv_out = inv_out,verbose = False,empirical_shift_down=True,
                                          link_hists = True)
print(predicted_scores)

# acc_link_hists = np.ones(shape=(2058,18))*-1
# for i in range(len(df_bsi_rpq_only)):
#     predicted_scores = crosswalk_scores(df_bsi_rpq_only.loc[i,rpq_cols].values.astype(int), 
#                                               score_dict, text_dict, hist_dict, simil_arr,
#                                               inv_in = inv_in,inv_out = inv_out,verbose = False,
#                                                 empirical_shift_down = False,
#                                               link_hists = False,random_seed= i)
    
#     acc = (np.asarray(predicted_scores) == df_bsi_rpq_only.loc[i,bsi_cols].values).astype(int)
    
#     acc_link_hists[i,:] = acc
    



    
    
    
    