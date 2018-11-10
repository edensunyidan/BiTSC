import sys
import os

import pandas as pd
import re
import numpy as np
from biptightkmeans.cluster import BitKmeans
from convert import convertfile

import json
import gc


if __name__ == "__main__":
    
    #input data file, absolute path
    linkage_filepath = sys.argv[1] # .txt
    enhancer_filepath = sys.argv[2] # .mx
    promoter_filepath = sys.argv[3] # .mx
    
    linkage_csvfilepath =  convertfile.txt_to_csv(linkage_filepath)
    enhancer_csvfilepath = convertfile.matrix_to_csv(enhancer_filepath)
    promoter_csvfilepath = convertfile.matrix_to_csv(promoter_filepath)

    linkage_data = pd.read_csv(linkage_csvfilepath)
    enhancer_raw_data = pd.read_csv(enhancer_csvfilepath)
    promoter_raw_data = pd.read_csv(promoter_csvfilepath)

    print("linkage raw data size: ", linkage_data.shape[0])
    print("enhancer raw data size: ", enhancer_raw_data.shape[0])
    print("promoter raw data size: ", promoter_raw_data.shape[0])
    
    samp_p = 0.8

    tao = 1
    weights=True
    drop=True
    
    #kernel=False
    kernel = True
    
    gamma=None
    simulation_type = 'sp_tsc'
    log_transform=False
    standardization=True
    use_pca=False
    
    #####################################################
    process = True
    #####################################################

    bitkmeans = BitKmeans(left_raw_data=enhancer_raw_data, right_raw_data=promoter_raw_data, orthologs_data=linkage_data,
                          samp_p=samp_p, tao=tao, weights=weights, drop=drop, kernel=kernel, gamma=gamma,
                          simulation_type=simulation_type, log_transform=log_transform, standardization=standardization, use_pca=use_pca)
                          
    #bitkmeans.eigenmatrix_sym(K = 500)
    #####################################################
    if process == True:
        
        k_min = int(sys.argv[4]) #default = 30
   
        alpha = float(sys.argv[5])  #default = 0.8
    
        iteration = 10
        resamp_num = 10

        output_hc = bitkmeans.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)

        root_dir_savedata = sys.argv[6]
        try:
            os.mkdir(root_dir_savedata)
        except FileExistsError:
            pass
    
        with open(root_dir_savedata+"cluster.txt", 'w') as file_object:
            for sub_tclust_id in output_hc['tclust_id']:
                file_object.write("cluster_size:"+str(len(sub_tclust_id))+"\t")
                file_object.write(",".join(sub_tclust_id)+"\n")

        with open(root_dir_savedata+"enhancer.txt", 'w') as file_object:
            file_object.write(",".join(output_hc['left_id']))

        with open(root_dir_savedata+"promoter.txt", 'w') as file_object:
            file_object.write(",".join(output_hc['right_id']))
        
        np.savetxt(root_dir_savedata+"consensus_matrix.txt", output_hc['consensus_matrix'])
        np.savetxt(root_dir_savedata+"adjacency_matrix.txt", output_hc['adjacency_matrix'])
    
        del output_hc
        gc.collect()
