
### To compare the performance of two types of spectral clustering
### 1. K-means
### 2. Spectral Clustering (sklearn)

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
    alpha = 0.7

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
              0.010, 0.013, 0.016, 0.019, 0.022, 0.025, 0.030, 0.035]


    ############################################ 'sp_tsc' with kernel ############################################
    simulation_type = 'sp_tsc'
    kernel=True
    
    ##################### vary average degree #####################
    #noise_ratio = 0.1
    #noise_ratio = 0.3
    noise_ratio = 0.5
    #noise_ratio = 1.0
    #noise_ratio = 2.0
    withkernel_degree_list = []
    
    for q in q_list:
        print("q = ", q)
        p = out_in_ratio*q
    
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
    
        #data.eigenmatrix_sym(K = 100) #l = 8
        #data.eigenmatrix_rw(K=100)
        #data.eigenmatrix_sym(K = 10)
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        
        simulation_type = 'sp_tsc'
        output_sp_tsc_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
        #output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
        #iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
    
        output = dict(expect_degree=data.expected_lambda, average_degree = data.average_degree, sp=output_sp, sp_tsc_hc=output_sp_tsc_hc)
        withkernel_degree_list.append(output)
        print(withkernel_degree_list)
    
    print("withkernel_degree_list is getting calcualted ")

    result = dict(withkernel = withkernel_degree_list)
    print(result)
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/analysis_varydegree_2019.json', 'w') as f:
        json.dump(result, f)

    """
    ############################################ 'tsc' with kernel ############################################
    simulation_type = 'tsc'
    kernel=True
    
    ##################### vary average degree #####################
    #noise_ratio = 0.1
    noise_ratio = 0.3
    
    tsc_degree=[]
    
    for q in np.arange(0.001, 0.052, 0.005):
        print("q = ", q)
        p = out_in_ratio*q
        
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
            
        #data.eigenmatrix_sym(K = 100) #l = 8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)
                       
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
                                              iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
                       
        output = dict(degree=data.expected_lambda, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        tsc_degree.append(output)

    print("tsc_degree is getting calculated")
    
    
    ############################################ 'tsc' without kernel ############################################
    simulation_type = 'tsc'
    kernel = False
    
    ##################### vary average degree #####################
    #noise_ratio = 0.1
    noise_ratio = 0.3
    tsc_degree_nokernel=[]
    
    for q in np.arange(0.001, 0.052, 0.005):
        print("q = ", q)
        p = out_in_ratio*q
        
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
            
        #data.eigenmatrix_sym(K = 100) #l = 8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)
                       
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
                                              iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
                       
        output = dict(degree=data.expected_lambda, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        tsc_degree_nokernel.append(output)
    
    print("tsc_degree_nokernel is getting calculated")


    result = dict(sp_tsc_degree = sp_tsc_degree, sp_tsc_degree_nokernel = sp_tsc_degree_nokernel,
                  tsc_degree = tsc_degree, tsc_degree_nokernel = tsc_degree_nokernel)

    #print(result)
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/main_simulation_varydegree.json', 'w') as f:
        json.dump(result, f)     
    """
            



