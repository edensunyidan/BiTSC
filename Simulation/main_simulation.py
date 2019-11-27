
# coding: utf-8

# In[ ]:

import sys
from ..simdata.synthetic_data import SimData
import numpy as np
import json
import math

__Author__ = "Yidan Sun"
#root_dir = 'project_fly_worm/result/figure'
#root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/figure'

"""加不加kernel 都是keep fully connected component ???"""


# In[ ]:

if __name__ == "__main__":
    
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    
    M, N = (80, 120)
    
    K = 10 #K = 15
    
    d_1, d_2 = (50, 50) #d_1, d_2 = (2, 2) #d_1, d_2 = (30, 30)

    small_sigma = 1 #clusters are too tight 
    #small_sigma = 5 #small_sigma = 10 #small_sigma = 15 #small_sigma = 20

    large_sigma = 1000 ###choice
    #large_sigma = 100  #large_sigma = 1000000 #large_sigma = 10000 
    
    out_in_ratio = 7 #out_in_ratio = 0.14 
    
    #q = 0.05                        ###paramters combination
    #q = 0.1
    #p = out_in_ratio*q              ###paramters combination
    
    #noise_ratio = 0.1               ###paramters combination
    #noise_ratio = 0.2 #noise_ratio = 0.3 #noise_ratio = 0.4
    
    samp_p = 0.7

    tao = 1 #tao = 5 #tao = 10
    
    #kernel = True #kernel = False  ###paramters combination

    rank_based = False
    
    value_based = False
    
    gamma = None
    
    #simulation_type = 'tsc'       ###paramters combination
    
    rank_d = 5
    
    value_t = 0.95
    
    ############################################ set fitting parameters ##########################################
    
    target = 10
    ###estimate of k_min through gap_statistics/prediction_strengths/dimension_reduction
    k_min = 10

    alpha = 0.9

    beta = 0.8

    seq_num = 2
    iteration = 4
    resamp_num = 12
    remain_p = 0.1

    k_stop = 1
    
    ############################################ 'tsc' with kernel ############################################
    simulation_type = 'tsc'
    kernel=True
    
    ##################### vary average degree ##################### 
    noise_ratio = 0.1
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
        
    print("tsc_degree: ", tsc_degree)
    
    ##################### vary noise ratio ##################### 
    q = 0.05
    p = out_in_ratio*q
    tsc_noise=[]
    
    for noise_ratio in np.arange(0.05, 0.55, 0.05):
        print("noise_ratio = ", noise_ratio)
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p, 
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma, 
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)

        #data.eigenmatrix_sym(K = 100) #l=8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)

        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num, 
                                              iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)

        output = dict(noise=noise_ratio, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        tsc_noise.append(output)
        
    print("tsc_noise: ", tsc_noise)
        
        
    ############################################ 'tsc' without kernel ############################################    
    simulation_type = 'tsc'
    kernel = False 
    ##################### vary average degree ##################### 
    noise_ratio = 0.1
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
        
    print("tsc_degree_nokernel: ", tsc_degree_nokernel)
        
    
    ##################### vary noise ratio ##################### 
    q = 0.05
    p = out_in_ratio*q
    tsc_noise_nokernel=[]
    
    for noise_ratio in np.arange(0.05, 0.55, 0.05): 
        print("noise_ratio = ", noise_ratio)
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p, 
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma, 
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)

        #data.eigenmatrix_sym(K = 100) #l=8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)

        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num, 
                                              iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)

        output = dict(noise=noise_ratio, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        tsc_noise_nokernel.append(output) 
        
    print("tsc_noise_nokernel: ", tsc_noise_nokernel)        

    
    ############################################ 'sp_tsc' with kernel ############################################
    simulation_type = 'sp_tsc'
    kernel=True
    
    ##################### vary average degree #####################
    noise_ratio = 0.1
    sp_tsc_degree=[]
    
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
        sp_tsc_degree.append(output)
    
    print("sp_tsc_degree: ", sp_tsc_degree)
    
    ##################### vary noise ratio #####################
    q = 0.05
    p = out_in_ratio*q
    sp_tsc_noise=[]
    
    for noise_ratio in np.arange(0.05, 0.55, 0.05):
        print("noise_ratio = ", noise_ratio)
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
    
        #data.eigenmatrix_sym(K = 100) #l=8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)
    
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
                                              iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
    
        output = dict(noise=noise_ratio, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        sp_tsc_noise.append(output)
    
    print("sp_tsc_noise: ", sp_tsc_noise)
    
    ############################################ 'sp_tsc' without kernel ############################################
    simulation_type = 'sp_tsc'
    kernel=False
    
    ##################### vary average degree #####################
    noise_ratio = 0.1
    sp_tsc_degree_nokernel=[]
    
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
        sp_tsc_degree_nokernel.append(output)
    
    print("sp_tsc_degree_nokernel: ", sp_tsc_degree_nokernel)
    
    ##################### vary noise ratio #####################
    q = 0.05
    p = out_in_ratio*q
    sp_tsc_noise_nokernel=[]
    
    for noise_ratio in np.arange(0.05, 0.55, 0.05):
        print("noise_ratio = ", noise_ratio)
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
    
        #data.eigenmatrix_sym(K = 100) #l=8
        #data.eigenmatrix_rw(K=100)
        data.eigenmatrix_sym(K = 10)
    
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        output_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num)
        output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
        iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
    
        output = dict(noise=noise_ratio, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        sp_tsc_noise_nokernel.append(output)
    
    print("sp_tsc_noise_nokernel: ", sp_tsc_noise_nokernel)


    result = dict(tsc_kernel=dict(tsc_degree=tsc_degree, tsc_noise=tsc_noise),
                  tsc_nokernel=dict(tsc_degree_nokernel=tsc_degree_nokernel, tsc_noise_nokernel=tsc_noise_nokernel),
                  sp_tsc_kernel=dict(sp_tsc_degree=sp_tsc_degree, sp_tsc_noise=sp_tsc_noise),
                  sp_tsc_nokernel=dict(sp_tsc_degree_nokernel=sp_tsc_degree_nokernel, sp_tsc_noise_nokernel=sp_tsc_noise_nokernel))

    print(result)
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/result.json', 'w') as f:
        json.dump(result, f)     
                  
            



