import sys
import os

import pandas as pd
import re
import numpy as np
from biptightkmeans.cluster import BitKmeans

import json
import gc


if __name__ == "__main__":
    
    #input data file, absolute path
    linkage_filepath = sys.argv[1] # .txt
    sideonedata_filepath = sys.argv[2] # .mx
    sidetwodata_filepath = sys.argv[3] # .mx
    
    linkage_data = pd.read_csv(linkage_filepath, sep='\t')
    sideone_raw_data = pd.read_csv(sideonedata_filepath, sep='\t')
    sidetwo_raw_data = pd.read_csv(sidetwodata_filepath, sep='\t')

    print("linkage raw data size: ", linkage_data.shape[0])
    print("enhancer raw data size: ", sideone_raw_data.shape[0])
    print("promoter raw data size: ", sidetwo_raw_data.shape[0])
    

    tao = 1
    weights=True
    drop=True

    kernel = True
    
    gamma=None
    simulation_type = 'sp_tsc'
    log_transform=False
    standardization=True
    use_pca=False

    #################################################################################################################
    bitkmeans = BitKmeans(left_raw_data=sideone_raw_data, right_raw_data=sidetwo_raw_data, orthologs_data=linkage_data,
                          samp_p=samp_p, tao=tao, weights=weights, drop=drop, kernel=kernel, gamma=gamma,
                          simulation_type=simulation_type, log_transform=log_transform, standardization=standardization, use_pca=use_pca)
                          
    #bitkmeans.eigenmatrix_sym(K = 500)
    #################################################################################################################
        
    k_min = int(sys.argv[4])

    resamp_num = int(sys.argv[5])
    iteration = int(int(sys.argv[6])/resamp_num)
   
    samp_p = float(sys.argv[7])
    
    one_thre = int(sys.argv[8])
    two_thre = int(sys.argv[9])
    
    root_dir_savedata = sys.argv[10]
    
    arguments = len(sys.argv) - 1
    alpha_vec = []
    for i in range(11, arguments+1):
        alpha = float(sys.argv[i])
        alpha_vec = alpha_vec.append(alpha)

    output_hc = bitkmeans.fit_hierarchical_clustering(k=k_min, alpha_vec=alpha_vec, plot_root_dir=root_dir_savedata,
                                                      thre_min_cluster_left=one_thre, thre_min_cluster_right=two_thre,
                                                      iteration=iteration, resamp_num=resamp_num)

    try:
        os.mkdir(root_dir_savedata)
    except FileExistsError:
        pass
    
    with open(root_dir_savedata+"/cluster.txt", 'w') as file_object:
        for sub_tclust_id in output_hc['tclust_id']:
            file_object.write("cluster_size:"+str(len(sub_tclust_id))+"\t")
            file_object.write(",".join(sub_tclust_id)+"\n")
    
    del output_hc
    gc.collect()
