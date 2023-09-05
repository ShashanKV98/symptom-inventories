import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import matplotlib as mpl
from collections import namedtuple


def link_distributions(A,B, a_val = 1,random_seed = 42):
    
    """
    Generate a 'semi-random distribution linked' value for:
        B, given a value from A
    """ 
    
    np.random.seed(random_seed)
    
    factor = 10**3
    
    breakpoints_A = np.round((factor)*np.concatenate(([0], np.cumsum(A)/sum(A))))
    
    breakpoints_B = (factor)*np.concatenate(([0], np.cumsum(B)/sum(B)))
    
    linspace_B = -1*np.ones(shape=factor)
    

    
    for i in range(5): 

        ints = np.arange(np.floor(breakpoints_B[i]), np.round(breakpoints_B[i+1])).astype(int)
        
        linspace_B[ints] = i
        
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
    
    A_group = namedtuple("A_group", "score_dict text_dict hist_dict simil_arr")
    A = A_group(score_dict,text_dict,hist_dict,simil_arr)
    return A
    
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
        inds_identical[i] = vec.argmax()  
    
    # Prediction code
    input_scores = np.asarray(input_scores)
    if empirical_shift_down:
        input_scores = input_scores - 1

    predicted_scores = -1*np.ones(num_items_predict)
    # print(simil_arr.shape)
    # print(inds_identical)
    for i in range(num_items_predict):
        print(inds_identical[i])
        print(hist_dict[(inv_in,inds_identical[i])])
        print(hist_dict[(inv_out,i)])
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