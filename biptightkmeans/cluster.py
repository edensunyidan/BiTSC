import pandas as pd
import numpy as np
import math

import collections

from sklearn.preprocessing import scale

from itertools import product
from itertools import combinations
#import itertools
from sklearn.metrics.pairwise import pairwise_kernels 

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list

from sklearn.cluster import spectral_clustering

try:
    from .consensuscluster.consensus_cluster import cdf_area_plot_new
    #SystemError: Parent module '' not loaded, cannot perform relative import
except Exception:
    from consensuscluster.consensus_cluster import cdf_area_plot_new

import multiprocessing as mp
import os
import gc
import sys

import scanpy as sc
import anndata

from collections import OrderedDict
import string
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


__Author__ = "Yidan Sun"


class BitKmeans:
    
    def __init__(self, covariate_data_path, edge_data_path, kernel_type, samp_p, tao):
        
        """
        e.g.
        [1,2,3]
        covariate_data_vec = [covariate_data_1, covariate_data_2, covariate_data_3]
        edge_data_vec = [edge_data_12, edge_data_13, edge_data_23]
        """

        self.node_id_vec = []
        self.node_data_vec = []
        self.node_index_vec = []
        self.node_num_vec = []
        self.samp_num_vec = []

        for covariate_path in covariate_data_path:
            covariate_data = pd.read_csv(covariate_path)
            covariate_data = covariate_data.dropna(axis=0, how='any')
            covariate_data = covariate_data.sort_values(by=[covariate_data.columns[0]], axis=0, ascending=True, inplace=False)
            node_id = covariate_data.loc[:, covariate_data.columns[0]].values
            node_data = covariate_data.drop(labels=covariate_data.columns[0], axis=1).values
            node_data = np.log2(node_data + 1)
            node_data = scale(node_data, axis=1, with_mean=True, with_std=True, copy=True)
            node_index = np.arange(len(node_id))
            samp_num = math.floor(samp_p*len(node_id))

            self.node_id_vec.append(node_id)
            self.node_data_vec.append(node_data)
            self.node_index_vec.append(node_index)
            self.node_num_vec.append(len(node_id))
            self.samp_num_vec.append(samp_num)


        self.edge_data_vec = []
        for edge_path in edge_data_path:
            edge_data = pd.read_csv(edge_path)
            self.edge_data_vec.append(edge_data)

        self.species_num = len(covariate_data_path)
        self.species_comb = list(combinations(range(self.species_num), 2))

        self.kernel_type = kernel_type
        self.tao = tao
        
        self.orthologs_dict()
        self.adjacency()
        

     
                    
    def orthologs_dict(self):  
        self.d_vec = []
        for edge_data in self.edge_data_vec:
            pairs = edge_data.values
            s = list(map(tuple, pairs))
            d = collections.defaultdict(list)
            for k, v, w in s:
                d[k].append((v,w))
            self.d_vec.append(d)
            
                

    def adjacency(self):
        self.A_vec = []
        for i in range(len(self.species_comb)):
            comb = self.species_comb[i]
            row_num = self.node_num_vec[comb[0]]
            col_num = self.node_num_vec[comb[1]]
            A = np.zeros((row_num, col_num), dtype=np.uint16)

            row_id = self.node_id_vec[comb[0]]
            col_id = self.node_id_vec[comb[1]]


            d = self.d_vec[i]
            for key in d.keys():
                for value in d[key]:
                    i1 = np.searchsorted(row_id, key)
                    i2 = np.searchsorted(col_id, value[0])

                    if row_id[i1] == key and col_id[i2] == value[0]:  
                        A[i1, i2] = value[1]

            self.A_vec.append(A)

   
    def sub_add_kernel(self, X, Y, A, tao_vec):
    
        kernel_l = pairwise_kernels(X, Y=None, metric=self.kernel_type, gamma=None)
        kernel_r = pairwise_kernels(Y, Y=None, metric=self.kernel_type, gamma=None)

        K_l = kernel_l + tao_vec[0]*np.identity(A.shape[0])
        K_r = kernel_r + tao_vec[1]*np.identity(A.shape[1])
        
        A = np.dot(np.dot(K_l, A), K_r)
        return(A)
    
    
    def find_candidates(self, k, alpha_vec, plot_root_dir, thre_num_node, iteration, resamp_num, heatmap):
        
        D_bar = 0
              
        for itr in range(iteration):
              
            print("Start iteration: %s" % (itr+1), "with %s" % (resamp_num), "processes")

            output = mp.Queue()

            processes = [mp.Process(target=self.simulation_sp_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]

            for p in processes:
                p.start()
            
            D_bar_sub = 0 
            for p in processes:
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)

            del co_labels
            gc.collect()

            for p in processes:  
                p.join()

            del output
            gc.collect()
            
            D_bar += (D_bar_sub/resamp_num)
            
            del D_bar_sub
            gc.collect()
            
            print("End iteration: ", itr+1)

        D_bar = D_bar/iteration
        
        D_bar_index = np.arange(sum(self.node_num_vec))
        compressed_distance = squareform((1-D_bar), force='tovector', checks=False)
        Z = linkage(compressed_distance, method='complete', optimal_ordering=False)

        name_list = [''.join(p) for p in product(string.ascii_lowercase, string.ascii_lowercase)]

        res = dict()
        for alpha in alpha_vec:
            max_d = 1-alpha
            hclusters = fcluster(Z, t=max_d, criterion='distance')
            leave_order = leaves_list(Z)
            leave_cluster = hclusters[leave_order]
            hclusters_nb = np.amax(hclusters) 

            node_num_cumsum = list(np.cumsum(self.node_num_vec))
            node_num_cumsum.insert(0,0)
            index_bound = list(zip(node_num_cumsum[:-1], node_num_cumsum[1:]))
            
            if heatmap == False:
                res_temp = []

                for hclusters_k_index in range(1, hclusters_nb+1):
                    candidate_cluster = D_bar_index[hclusters == hclusters_k_index]
                    candidate_cluster_count = [sum((l_b <= candidate_cluster) & (candidate_cluster < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_cluster_count) >= 1:
                        res_temp.append(candidate_cluster)
                
                res[alpha] = res_temp

            else:
                res_temp = []
                hmp = []
                hmp_cluster = []

                for hclusters_k_index in range(1, hclusters_nb+1):
                    candidate_cluster = D_bar_index[hclusters == hclusters_k_index]
                    candidate_cluster_count = [sum((l_b <= candidate_cluster) & (candidate_cluster < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_cluster_count) >= 1:
                        res_temp.append(candidate_cluster)

                    candidate_heatmap = D_bar_index[leave_order[leave_cluster == hclusters_k_index]]
                    candidate_heatmap_count = [sum((l_b <= candidate_heatmap) & (candidate_heatmap < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_heatmap_count) >= thre_num_node:
                        hmp.append(leave_order[leave_cluster == hclusters_k_index])
                        hmp_cluster.append(np.repeat(hclusters_k_index, len(candidate_heatmap)))
                
                res[alpha] = res_temp


                #print("Plot the heatmap")
                hmp = np.array(list(itertools.chain(*hmp)))
                hmp_cluster = np.array(list(itertools.chain(*hmp_cluster)))
                hmp_cluster = hmp_cluster.astype(str)
                D_bar_sub = D_bar[hmp[:, np.newaxis], hmp]

            
                hmp_cluster_idx = hmp_cluster.copy()
                hmp_cluster_unique = list(OrderedDict.fromkeys(hmp_cluster_idx))
                ii = 0
                for ele in hmp_cluster_unique:
                    hmp_cluster[hmp_cluster_idx==ele] = name_list[ii]
                    ii += 1
                
                D_bar_sub_obs = pd.DataFrame({'Co-cluster': pd.Series(hmp_cluster, index=list(range(len(hmp))))}, dtype="category")
                D_bar_sub_var = pd.DataFrame({'Columns': pd.Series(hmp_cluster, index=list(range(len(hmp))))}, dtype="category")

                D_bar_sub_obs.index.name = 'index'
                D_bar_sub_var.index.name = 'index'

                D_bar_sub_obs.index = D_bar_sub_obs.index.astype('str', copy=False)
                D_bar_sub_var.index = D_bar_sub_var.index.astype('str', copy=False)

                D_bar_sub_ann = anndata.AnnData(D_bar_sub, obs=D_bar_sub_obs, var=D_bar_sub_var)
                sc.pl.heatmap(D_bar_sub_ann, D_bar_sub_ann.var.index.values.tolist(), cmap=plt.cm.Blues, use_raw=False, show_gene_labels=False, figsize=(5,5), groupby='Co-cluster')
                
                plt.title('Alpha$=%.2f$' % (alpha))
                plt.savefig(plot_root_dir+'/Alpha$=%.2f$.pdf'% (alpha))
            
        return res


    def find_candidates_given_consensus_matrix(self, k, alpha_vec, plot_root_dir, thre_num_node, heatmap):
          
        D_bar_index = np.arange(sum(self.node_num_vec))
        compressed_distance = squareform((1-self.D_bar), force='tovector', checks=False)
        Z = linkage(compressed_distance, method='complete', optimal_ordering=False)

        name_list = [''.join(p) for p in product(string.ascii_lowercase, string.ascii_lowercase)]

        res = dict()
        for alpha in alpha_vec:
            max_d = 1-alpha
            hclusters = fcluster(Z, t=max_d, criterion='distance')
            leave_order = leaves_list(Z)
            leave_cluster = hclusters[leave_order]
            hclusters_nb = np.amax(hclusters)
            
            node_num_cumsum = list(np.cumsum(self.node_num_vec))
            node_num_cumsum.insert(0,0)
            index_bound = list(zip(node_num_cumsum[:-1], node_num_cumsum[1:]))
            
            if heatmap == False:
                res_temp = []

                for hclusters_k_index in range(1, hclusters_nb+1):
                    candidate_cluster = D_bar_index[hclusters == hclusters_k_index]
                    candidate_cluster_count = [sum((l_b <= candidate_cluster) & (candidate_cluster < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_cluster_count) >= 1:
                        res_temp.append(candidate_cluster)
                
                res[alpha] = res_temp

            else:
                res_temp = []
                hmp = []
                hmp_cluster = []

                for hclusters_k_index in range(1, hclusters_nb+1):
                    candidate_cluster = D_bar_index[hclusters == hclusters_k_index]
                    candidate_cluster_count = [sum((l_b <= candidate_cluster) & (candidate_cluster < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_cluster_count) >= 1:
                        res_temp.append(candidate_cluster)

                    candidate_heatmap = D_bar_index[leave_order[leave_cluster == hclusters_k_index]]
                    candidate_heatmap_count = [sum((l_b <= candidate_heatmap) & (candidate_heatmap < u_b)) for (l_b, u_b) in index_bound]
                    if np.amin(candidate_heatmap_count) >= thre_num_node:
                        hmp.append(leave_order[leave_cluster == hclusters_k_index])
                        hmp_cluster.append(np.repeat(hclusters_k_index, len(candidate_heatmap)))
                
                res[alpha] = res_temp

                #print("Plot the heatmap")
                hmp = np.array(list(itertools.chain(*hmp)))
                hmp_cluster = np.array(list(itertools.chain(*hmp_cluster)))
                hmp_cluster = hmp_cluster.astype(str)
                D_bar_sub = self.D_bar[hmp[:, np.newaxis], hmp]

            
                hmp_cluster_idx = hmp_cluster.copy()
                hmp_cluster_unique = list(OrderedDict.fromkeys(hmp_cluster_idx))
                ii = 0
                for ele in hmp_cluster_unique:
                    hmp_cluster[hmp_cluster_idx==ele] = name_list[ii]
                    ii += 1
                
                D_bar_sub_obs = pd.DataFrame({'Co-cluster': pd.Series(hmp_cluster, index=list(range(len(hmp))))}, dtype="category")
                D_bar_sub_var = pd.DataFrame({'Columns': pd.Series(hmp_cluster, index=list(range(len(hmp))))}, dtype="category")

                D_bar_sub_obs.index.name = 'index'
                D_bar_sub_var.index.name = 'index'

                D_bar_sub_obs.index = D_bar_sub_obs.index.astype('str', copy=False)
                D_bar_sub_var.index = D_bar_sub_var.index.astype('str', copy=False)

                D_bar_sub_ann = anndata.AnnData(D_bar_sub, obs=D_bar_sub_obs, var=D_bar_sub_var)
                sc.pl.heatmap(D_bar_sub_ann, D_bar_sub_ann.var.index.values.tolist(), cmap=plt.cm.Blues, use_raw=False, show_gene_labels=False, figsize=(5,5), groupby='Co-cluster')
                
                plt.title('Alpha$=%.2f$' % (alpha))
                plt.savefig(plot_root_dir+'/Alpha$=%.2f$.pdf'% (alpha))
            
        return res



    def get_consensus_matrix(self, k, iteration, resamp_num):

        D_bar = 0
              
        for itr in range(iteration):
              
            print("Start iteration: %s" % (itr+1), "with %s" % (resamp_num), "processes")

            output = mp.Queue()

            processes = [mp.Process(target=self.simulation_sp_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]

            for p in processes:
                p.start()
            
            D_bar_sub = 0 
            for p in processes:
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)

            del co_labels
            gc.collect()

            for p in processes:  
                p.join()

            del output
            gc.collect()
            
            D_bar += (D_bar_sub/resamp_num)
            
            del D_bar_sub
            gc.collect()
            
            print("End iteration: ", itr+1)

        D_bar = D_bar/iteration
                    
        return D_bar
                        


    def simulation_sp_tsc(self, random_seed, clusters_nb, output):

        np.random.seed(seed=random_seed)
        
        samp_index = []
        mask = []
        for i_species in range(self.species_num):
            i_samp_index = np.sort(np.random.choice(self.node_index_vec[i_species], self.samp_num_vec[i_species], replace=False))
            
            samp_index.append(i_samp_index)
            i_mask = np.zeros(self.node_num_vec[i_species], dtype=bool)
            i_mask[i_samp_index] = True
            mask.append(i_mask)


        samp_KA_vec = []
        for i_comb in range(len(self.species_comb)):
            comb = self.species_comb[i_comb]
            A = self.A_vec[i_comb]
            samp_A = A[samp_index[comb[0]]][:, samp_index[comb[1]]]

            X0 = self.node_data_vec[comb[0]][samp_index[comb[0]]]
            X1 = self.node_data_vec[comb[1]][samp_index[comb[1]]]

            tao_vec = self.tao[i_comb]

            samp_KA = self.sub_add_kernel(X0, X1, samp_A, tao_vec)
            samp_KA_vec.append(samp_KA)

        
        samp_W_dict = {key: [] for key in range(self.species_num)}
        
    
        for index_block_row in range(self.species_num):
            for index_block_col in range(self.species_num):
                if index_block_row >= index_block_col:
                    samp_W_dict[index_block_row].append(np.zeros((self.samp_num_vec[index_block_row],self.samp_num_vec[index_block_col]), dtype=np.uint16))
                else:
                    index_samp_KA = self.species_comb.index((index_block_row,index_block_col))
                    samp_W_dict[index_block_row].append(samp_KA_vec[index_samp_KA])
        

        #upper tri to symmetric
        samp_W = []
        for key in range(self.species_num):
            samp_W.append(np.concatenate(samp_W_dict[key], axis=1))
        samp_W = np.concatenate(samp_W, axis=0)
        samp_W = samp_W + samp_W.T
        
        #some clusters may not contain all sides of nodes (co-cluster or multi-cluster)
        samp_label_total = spectral_clustering(affinity=samp_W, n_clusters=clusters_nb, n_components=None, eigen_solver=None,
            random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')

        samp_label_vec = []
        samp_num_cumsum = np.cumsum(self.samp_num_vec)
        samp_label_vec.append(samp_label_total[: samp_num_cumsum[0]])
        for i_species in range(1, self.species_num):
            samp_label_vec.append(samp_label_total[samp_num_cumsum[i_species-1] : samp_num_cumsum[i_species]])


        label_vec = []
        for i_species in range(self.species_num):
            X = self.node_data_vec[i_species][samp_index[i_species]] #samp_node_data

            samp_label = samp_label_vec[i_species]
            label = np.empty(self.node_num_vec[i_species], dtype = np.uint16)  
            label[mask[i_species]] = samp_label
            node_centroid_vec = []
            for i in np.arange(clusters_nb, dtype=np.uint16):
                if sum(samp_label == i) > 0:
                    node_centroid = np.sum(X[samp_label == i], axis=0) / sum(samp_label == i)  
                    node_centroid_vec.append((i, node_centroid))

            node_centroid_name, node_centroid_m = zip(*node_centroid_vec)
            node_centroid_m = np.array(node_centroid_m)
  

            euclidean_X = euclidean_distances(self.node_data_vec[i_species][~mask[i_species]], node_centroid_m)
            label[~mask[i_species]] = np.array([node_centroid_name[i] for i in np.argmin(euclidean_X, axis=1)])
            label_vec.append(label)


        co_labels = np.concatenate(label_vec, axis=0)
        output.put(co_labels)
        del co_labels
        gc.collect()



    def fit_hierarchical_clustering(self, k_vec, alpha_vec, plot_root_dir, thre_num_node, iteration, resamp_num, heatmap):

        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")
        node_id_vec = np.concatenate(self.node_id_vec, axis=0)

        if len(k_vec) == 2:
            print("select K_0 by binary search")

            k_min = k_vec[0]
            k_max = k_vec[1]
            k_mid = math.ceil((k_min + k_max)/2)
            print("K_min: ", k_min)
            print("K_max: ", k_max)
            print("K_mid: ", k_mid)

            #print("get area under CDF at K_min")
            cdf_k_min = self.get_cdf_consensus_matrix(k=k_min, iteration=iteration, resamp_num=resamp_num)
            areaK_k_min = cdf_k_min['area']
            
            #print("get area under CDF at K_max")
            cdf_k_max = self.get_cdf_consensus_matrix(k=k_max, iteration=iteration, resamp_num=resamp_num)
            areaK_k_max = cdf_k_max['area']
            
            enter_loop = False
            while (abs((areaK_k_min - areaK_k_max)/areaK_k_min) > 0.05) & (k_min < k_max):

                enter_loop = True
                k_mid = math.ceil((k_min + k_max)/2)
                #print("get area under CDF at K_mid")
                cdf_k_mid = self.get_cdf_consensus_matrix(k=k_mid, iteration=iteration, resamp_num=resamp_num)
                areaK_k_mid = cdf_k_mid['area']
                
                if abs(areaK_k_min - areaK_k_mid) < abs(areaK_k_mid - areaK_k_max):
                    k_max = k_mid
                    areaK_k_max = areaK_k_mid
                else:
                    k_min = k_mid
                    areaK_k_min = areaK_k_mid

            k = k_mid
            print("automatically find K_0: ", k)
            output = dict()
            if enter_loop == True:
                result_index = self.find_candidates_given_consensus_matrix(k, alpha_vec, plot_root_dir, thre_num_node, heatmap)
            else:
                result_index = self.find_candidates(k, alpha_vec, plot_root_dir, thre_num_node, iteration, resamp_num, heatmap) 

            for alpha in alpha_vec:  
                result_index_alpha = result_index[alpha]

                result_id = []
                for cand_index in result_index_alpha:
                    cand_index = list(map(int, cand_index))
                    cand_clust = node_id_vec[cand_index]
                    result_id.append(cand_clust)

                output[alpha] = result_id
        
        else:
            k = k_vec[0]
            print("user-specified K_0: ", k)
            output = dict()
            result_index = self.find_candidates(k, alpha_vec, plot_root_dir, thre_num_node, iteration, resamp_num, heatmap) 
            
            for alpha in alpha_vec:  
                result_index_alpha = result_index[alpha]

                result_id = []
                for cand_index in result_index_alpha:
                    cand_index = list(map(int, cand_index))
                    cand_clust = node_id_vec[cand_index]
                    result_id.append(cand_clust)

                output[alpha] = result_id

        return(output)


    def get_cdf_consensus_matrix(self, k, iteration, resamp_num):
        self.D_bar = self.get_consensus_matrix(k, iteration, resamp_num)
        cdf = cdf_area_plot_new(self.D_bar)

        return(cdf)
