
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


"""robustness analysis of \tau in S_{\tau,\alpha} = (K_1 + \tau I) A (K_2 + \tau I) to the hierachical clusteirng (with kernel)"""

"""robustness analysis of \tau in S_{\tau,\alpha} = (K_1 + \tau I) A (K_2 + \tau I) to the spectral clusteirng (with kernel)"""

"""robustness analysis of \tau in S_{\tau,\alpha} = (K_1 + \tau I) A (K_2 + \tau I) to the bitkmeans (tsc) (with kernel)"""

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

    large_sigma = 10
    #large_sigma = 1000 ###choice
    #large_sigma = 100  #large_sigma = 1000000 #large_sigma = 10000 
    
    out_in_ratio = 5 #out_in_ratio = 0.2
    #out_in_ratio = 7 #out_in_ratio = 0.14
    
    #q = 0.05                        ###paramters combination
    #q = 0.1
    #p = out_in_ratio*q              ###paramters combination
    
    #noise_ratio = 0.1               ###paramters combination
    #noise_ratio = 0.2 #noise_ratio = 0.3 #noise_ratio = 0.4
    
    #samp_p = 0.7
    samp_p = 0.8

    #tao = 1 #tao = 5 #tao = 10
    
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

    q_list = [0.001, 0.0015, 0.0021, 0.0028, 0.0036, 0.0045, 0.0055, 0.007,
              0.010, 0.013,0.016, 0.019, 0.022, 0.025, 0.030, 0.035]
    
    ############################################ 'sp_tsc' with kernel ############################################
    simulation_type = 'sp_tsc'
    kernel=True

    q = q_list[-2]
    p = out_in_ratio*q
    noise_ratio = 0.5

    ##################### vary \tau #####################
    tao_set = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    result_list = []
    
    for tao in tao_set:
        print("tao = ", tao)
        
        data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p,
                       tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma,
                       simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
            
        #data.eigenmatrix_sym(K = 100) #l = 8
        #data.eigenmatrix_rw(K=100)
        #data.eigenmatrix_sym(K = 10)
                       
        output_sp = data.fit_spectral_clustering(clusters_nb=k_min)
        simulation_type = 'tsc'
        
        output_tsc_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
        simulation_type = 'tscu'
        output_tscu_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)
        simulation_type = 'sp_tsc'
        output_sp_tsc_hc = data.fit_hierarchical_clustering(k=k_min, alpha=alpha, iteration=iteration, resamp_num=resamp_num, simulation_type=simulation_type)

        #output_bitkmeans = data.fit_bitkmeans(target=target, k_min=k_min, alpha=alpha, beta=beta, seq_num=seq_num,
        #iteration=iteration, resamp_num=resamp_num, remain_p=remain_p, k_stop=k_stop)
                       
        output = dict(tao=tao, sp=output_sp, tsc_hc=output_tsc_hc, tscu_hc=output_tscu_hc, sp_tsc_hc=output_sp_tsc_hc)
        result_list.append(output)
    
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/robustness_tao_new.json', 'w') as f:
        json.dump(result_list, f)
    
    #print("sp_tsc_tao: ", sp_tsc_tao)
    """
    ############################################ 'tsc' with kernel ############################################
    simulation_type = 'tsc'
    kernel=True

    noise_ratio = 0.3
    q = 0.03
    p = out_in_ratio*q

    ##################### vary \tau #####################
    tao_set = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    tsc_tao=[]
    
    for tao in tao_set:
        print("tao = ", tao)

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

        output = dict(tao=tao, sp=output_sp, hc=output_hc, bitkmeans=output_bitkmeans)
        tsc_tao.append(output)
        
    #print("tsc_tao: ", tsc_tao)

    result = dict(sp_tsc_tao=sp_tsc_tao, tsc_tao=tsc_tao)

    #print(result)
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/data'
    with open(root_dir + '/main_simulation_rbs_tau.json', 'w') as f:
        json.dump(result, f)     
    """
            



