
# coding: utf-8

# In[ ]:

import sys
from simdata.synthetic_data import SimData
import numpy as np
import json
import math

__Author__ = "Yidan Sun"
#root_dir = 'project_fly_worm/result/figure'
#root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/figure'


# In[ ]:

if __name__ == "__main__":
    
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    
    #M, N = (80, 120)
    M, N = (50, 70)

    #K = 10
    K = 15
    
    #d_1, d_2 = (50, 50)
    d_1, d_2 = (2, 2) #d_1, d_2 = (30, 30)

    #small_sigma = 1 #clusters are too tight
    #small_sigma = 5 #small_sigma = 10
    small_sigma = 15 #small_sigma = 20

    #large_sigma = 1000 ###choice
    #large_sigma = 100
    large_sigma = 10
    #large_sigma = 8
    #large_sigma = 5

    out_in_ratio = 5 #out_in_ratio = 0.2
    #out_in_ratio = 7 #out_in_ratio = 0.14
    
    #q = 0.05                        ###paramters combination
    #q = 0.1
    #p = out_in_ratio*q              ###paramters combination
    
    #noise_ratio = 0.1               ###paramters combination
    #noise_ratio = 0.2 #noise_ratio = 0.3 #noise_ratio = 0.4
    
    #samp_p = 0.7
    samp_p = 0.8

    tao = 1 #tao = 5 #tao = 10
    
    #kernel = True #kernel = False  ###paramters combination

    rank_based = False
    
    value_based = False
    
    gamma = None
    
    #simulation_type = 'tsc'       ###paramters combination
    
    rank_d = 5

    value_t = 0.95
    
    ############################################ set fitting parameters ##########################################
    
    #target = 10
    #target = 15
    ###estimate of k_min through gap_statistics/prediction_strengths/dimension_reduction
    #k_min = 10
    k_min = 15

    #alpha = 0.9
    #alpha = 0.7

    #beta = 0.8
    #beta = 0.9

    #seq_num = 2
    iteration = 5
    resamp_num = 10
    #remain_p = 0.1

    #k_stop = 1
    #k_stop = 2

    #np.arange(0.001, 0.052, 0.005)
    q_list = [0.001, 0.0015, 0.0021, 0.0028, 0.0036, 0.0045, 0.0055, 0.007,
              0.010, 0.013,0.016, 0.019, 0.022, 0.025, 0.030, 0.035]

    alpha_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    #alpha_list = [0.50, 0.60, 0.70, 0.80, 0.90]
    ############################################ 'sp_tsc' with kernel ############################################
    simulation_type = 'sp_tsc'
    kernel=True
    
    ##################### vary average degree #####################
    #noise_ratio = 0.1
    #noise_ratio = 0.3
    #noise_ratio = 0.5
    #noise_ratio = 1.0
    #noise_ratio = 2.0
    final_output = []
    for noise_ratio in [0.1, 0.5, 1.0, 1.5, 2.0]:
        varyalpha_result = []
        
        q = q_list[-2]
        p = out_in_ratio*q
        
        for alpha in alpha_list:
            data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                           tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                           simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
            print("alpha = ", alpha)
            #data.eigenmatrix_sym(K = 100) #l = 8
            #data.eigenmatrix_rw(K=100)
            #data.eigenmatrix_sym(K = 10)
            #output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
                           
            simulation_type = 'tsc'
            output_tsc_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
            simulation_type = 'tscu'
            output_tscu_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
            simulation_type = 'sp_tsc'
            output_sp_tsc_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
            #output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
            #iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
                           
            output = dict(noise=noise_ratio, alpha=alpha, tsc_hc=output_tsc_hc, tscu_hc=output_tscu_hc, sp_tsc_hc=output_sp_tsc_hc)
            varyalpha_result.append(output)
        
        #print("varyalpha_result is getting calcualted ")
        final_output.append(varyalpha_result)

    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/analysis_varyalphanoise_new.json', 'w') as f:
        json.dump(final_output, f)

