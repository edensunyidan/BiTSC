from scipy.stats import bernoulli

try:
    from .evaluation.rand_index import weighted_rand_score
    #SystemError: Parent module '' not loaded, cannot perform relative import
except Exception:
    from evaluation.rand_index import weighted_rand_score
    #ImportError

import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_blobs

#from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from itertools import product
from itertools import combinations
#from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import rankdata
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import pairwise_kernels

from scipy.sparse.linalg import eigsh

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding

import multiprocessing as mp
import os
import json
import time
import gc

#"/home/yidan/Dropbox/project_fly_worm/simulation/synthetic_data.py"
__Author__ = "Yidan Sun"


# In[ ]:

class SimData:
    
    def __init__(self, M, N, K, d_1, d_2, small_sigma, large_sigma, p, q, noise_ratio, samp_p, tao, kernel=False, rank_based=False, value_based=False, gamma=None, simulation_type='tsc', rank_d=5, value_t=0.95):
        
        np.random.seed(0)
        self.M = M  #number of nodes in each true clusters on the left side   
        self.N = N  #number of nodes in each true clusters on the right side
        self.K = K  #number of true matched tight clusters
        self.leftindexs = np.arange(M*K)    
        self.rightindexs = np.arange(N*K) 
        self.leftnodes = M*K
        self.rightnodes = N*K

        ###expected average degree without considering removed nodes for keeping a fully connected component
        self.expected_lambda = (2*M*N*K/((M+N)*(1+noise_ratio)))*(q*((1+noise_ratio)**2) + (1/K)*(p-q))
        
        ###Covariate generation
        """
        d_1: dimension of covariates on the left side
        d_2: dimension of covariates on the right side
        
        large_sigma: \Sigma, (v_rk, r=1,2) ~iid N(u, \Sigma), k = 1, ..., K
        small_sigma: \sigma_r, x_ri|(z_ri=k, v_rk) ~ N(v_rk, \sigma_r^2*I_{d_r}), r=1,2
        """
        self.left_dim = d_1
        self.right_dim = d_2

        minAllowableDistance = large_sigma
    
        # Initialize first point. #.shape = (1, d_1)
        mux_keeper = np.random.multivariate_normal(mean=np.zeros(d_1, dtype=np.float64), cov=np.eye(d_1, dtype=np.float64)*(large_sigma**2), size=1)
        
        counter = 1
        while counter < K:
            mux_cand = np.random.multivariate_normal(mean=np.zeros(d_1, dtype=np.float64), cov=np.eye(d_1, dtype=np.float64)*(large_sigma**2), size=1)
            distances = euclidean_distances(mux_cand, mux_keeper)
            minDistance = np.amin(distances)
            if minDistance >= minAllowableDistance:
                mux_keeper = np.concatenate((mux_keeper, mux_cand), axis=0)
                counter += 1

        self.mux = mux_keeper
        
        # Initialize first point. #.shape = (1, d_1)
        muy_keeper = np.random.multivariate_normal(mean=np.zeros(d_2, dtype=np.float64), cov=np.eye(d_2, dtype=np.float64)*(large_sigma**2), size=1)
        
        counter = 1
        while counter < K:
            muy_cand = np.random.multivariate_normal(mean=np.zeros(d_2, dtype=np.float64), cov=np.eye(d_2, dtype=np.float64)*(large_sigma**2), size=1)
            distances = euclidean_distances(muy_cand, muy_keeper)
            minDistance = np.amin(distances)
            if minDistance >= minAllowableDistance:
                muy_keeper = np.concatenate((muy_keeper, muy_cand), axis=0)
                counter += 1

        self.muy = muy_keeper
    
        """
        self.mux = np.random.multivariate_normal(mean=np.zeros(d_1, dtype=np.float64), cov=np.eye(d_1, dtype=np.float64)*large_sigma, size=K) 
        self.muy = np.random.multivariate_normal(mean=np.zeros(d_2, dtype=np.float64), cov=np.eye(d_2, dtype=np.float64)*large_sigma, size=K) 
        """
        """
        # mean vectors are generated along the "diagonal"
        self.mux = np.array([np.repeat((i+1)*delta, d_1) for i in np.arange(K)])
        self.muy = np.array([np.repeat((i+1)*delta, d_2) for i in np.arange(K)])
        """
        
        #small_sigma = i+1
        self.iX = np.array([np.random.multivariate_normal(mean=self.mux[i], cov=((0.1*(i+1))**2)*np.eye(d_1, dtype=np.float64), size=M)
                            for i in np.arange(K)]).flatten().reshape(M*K, d_1)
        self.iY = np.array([np.random.multivariate_normal(mean=self.muy[i], cov=((0.1*(i+1))**2)*np.eye(d_2, dtype=np.float64), size=N)
                            for i in np.arange(K)]).flatten().reshape(N*K, d_2)
        
        ###Network generation
        self.tllabel = np.array([[np.repeat(i, M)] for i in np.arange(K)]).flatten()
        self.trlabel = np.array([[np.repeat(i, N)] for i in np.arange(K)]).flatten()
        
        self.p = p  #p_in
        self.q = q  #p_out
        
        ###Noise nodes (scattered genes) generation
        self.noise_ratio = noise_ratio
        self.add_noise()
        
        ###Adjacency matrix 
        self.adjacency() 
        
        ###the other paramters
        self.samp_p = samp_p
        self.tao = tao
        self.kernel = kernel
        self.rank_based = rank_based
        self.value_based = value_based
        self.gamma = gamma
        self.simulation_type = simulation_type
        #self.use_pca = use_pca
        self.rank_d = rank_d
        self.value_t = value_t
        #self.update_data()
        self.update_sample()
        self.add_edge()
    
        self.U = None

        
    def add_noise(self): 
        
        np.random.seed(0)
        #if self.noise_ratio is not None:
        if self.noise_ratio > 0:
            self.lnoise = math.floor(self.M*self.K*self.noise_ratio)
            self.rnoise = math.floor(self.N*self.K*self.noise_ratio)
            self.leftindexs = np.arange(self.M*self.K + self.lnoise)
            self.rightindexs = np.arange(self.N*self.K + self.rnoise)
            
            self.leftnodes = self.M*self.K + self.lnoise
            self.rightnodes = self.N*self.K + self.rnoise

            self.tllabel = np.concatenate((self.tllabel, np.repeat(self.K, self.lnoise)), axis=0)
            self.trlabel = np.concatenate((self.trlabel, np.repeat(self.K, self.rnoise)), axis=0)
            
            xmin, xmax = self.bounding_box(self.iX)
            ymin, ymax = self.bounding_box(self.iY)
            
            noiseX = []
            noiseY = []
            for _ in np.arange(self.lnoise):
                noiseX.append([np.random.uniform(low=xmin[p], high=xmax[p], size=1)[0] for p in np.arange(self.left_dim)])
            self.noiseX = np.array(noiseX)
            
            for _ in np.arange(self.rnoise):
                noiseY.append([np.random.uniform(low=ymin[p], high=ymax[p], size=1)[0] for p in np.arange(self.right_dim)])
            self.noiseY = np.array(noiseY)

            self.iX = np.concatenate((self.iX, self.noiseX), axis=0)
            self.iY = np.concatenate((self.iY, self.noiseY), axis=0)
            
           
    def adjacency(self):
        np.random.seed(0)
        """Strict bipartite network: no edges within each single side. Elements of the adjacency matrix belong to {0, 1}"""
        self.A = np.zeros((self.leftnodes, self.rightnodes), dtype=np.uint16)
        
        if self.noise_ratio > 0:
            for i in np.arange(self.leftnodes):
                for j in np.arange(self.rightnodes):
                    if (i < self.M*self.K) and (j < self.N*self.K): 
                        if self.tllabel[i] == self.trlabel[j]:
                            #self.A[i,j] = np.random.binomial(n=1, p=self.p, size=1)   
                            self.A[i,j] = bernoulli.rvs(p=self.p, size=1)
                        else:
                            #self.A[i,j] = np.random.binomial(n=1, p=self.q, size=1)
                            self.A[i,j] = bernoulli.rvs(p=self.q, size=1)
                    else:
                        #self.A[i,j] = np.random.binomial(n=1, p=self.q, size=1)
                        self.A[i,j] = bernoulli.rvs(p=self.q, size=1)

            self.average_degree = (2*np.sum(self.A))/(self.leftnodes+self.rightnodes)
            #The bi_adjacency matrix
            self.W = np.concatenate((np.concatenate((np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16), self.A), axis=1), 
                                     np.concatenate((self.A.T, np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)), axis=1)), axis=0)
    
        else:
            for i in np.arange(self.leftnodes):
                for j in np.arange(self.rightnodes):
                    if self.tllabel[i] == self.trlabel[j]:
                        #self.A[i,j] = np.random.binomial(n=1, p=self.p, size=1)   
                        self.A[i,j] = bernoulli.rvs(p=self.p, size=1)
                    else:
                        #self.A[i,j] = np.random.binomial(n=1, p=self.q, size=1)
                        self.A[i,j] = bernoulli.rvs(p=self.q, size=1)

            #self.average_degree = (2*np.sum(self.A))/(self.leftnodes+self.rightnodes) #only useful when no kernel method employed
            #The bi_adjacency matrix
            self.W = np.concatenate((np.concatenate((np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16), self.A), axis=1), 
                                     np.concatenate((self.A.T, np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)), axis=1)), axis=0)

   
    def update_data(self):
    
        self.leftindexs = np.arange(self.iX.shape[0])
        self.rightindexs = np.arange(self.iY.shape[0])
        
        self.leftnodes = self.iX.shape[0]
        self.rightnodes = self.iY.shape[0]
        
        """
        FPKM --> log2 transformation --> standardization_sample
        FPKM --> log2 transformation --> PCA
        FPKM --> log2 transformation --> quantile normalization
        FPKM --> standardization_sample
        scImpute: X_{ij} = log10(X_{ij} + 1.01)
        """
        #self.log_transformation()

        #self.standardization_feature()
        #self.standardization_sample() 
        #self.normalization()
        
        #if self.use_pca == True: #data standardization
            #self.pca()
        
        #self.orthologs_dict()
        #self.adjacency() 
        #self.update_sample()
        
        
    def update_sample(self):
        """this method should be called as long as self.leftindexs/rightindexs has been changed"""
        
        print(".update_sample() get called")
        self.m = math.floor(self.samp_p*len(self.leftindexs))
        self.n = math.floor(self.samp_p*len(self.rightindexs))
        self.X = self.iX[self.leftindexs]
        self.Y = self.iY[self.rightindexs]  
        
    
    def add_edge(self):
        """
        self.kernel == True or self.rank_based == True or self.value_based == True, but not all at the same time,
        since three methods add different types of edges
        """
        if self.kernel == True:
            print("embedding the kernel method")
            self.add_kernel(kernel_type='rbf')
            self.find_connected_components()
            if self.N_components > 1: 
                self.keep_largest_component()  ###self.update_data()
                self.remain_cluster()
                ###considering directly slice the kernel matrix or rank_based correlation matrix (self.W) instead   
                ###reconstruct the kernel matrix /others??? consider change in cluster.py???
                self.add_kernel(kernel_type = 'rbf') 
                self.find_connected_components() #re-check: self.D has to be non-singular (fully-connected)
    
        else:
            print("no embedding the kernel method")
            self.find_connected_components()
        
        if self.rank_based == True:
            print("embedding the rank_based method")
            self.add_rank_based(rank_d=self.rank_d)
            self.find_connected_components()
            if self.N_components > 1: 
                self.keep_largest_component()  ###self.update_data()
                self.remain_cluster()
                self.add_rank_based(rank_d=self.rank_d)
                self.find_connected_components() #re-check: self.D has to be non-singular (fully-connected) 
        
        if self.value_based == True:
            print("embedding the value_based method")
            self.add_value_based(value_t=self.value_t)
            self.find_connected_components()
            if self.N_components > 1: 
                self.keep_largest_component()  ###self.update_data()
                self.remain_cluster()
                self.add_value_based(value_t=self.value_t)
                self.find_connected_components() #re-check: self.D has to be non-singular (fully-connected) 
                
    
    def remain_cluster(self):
        ###how to cut the main program/processes automatically except the printed warning message
        if self.noise_ratio > 0:
            #if (len(np.unique(self.tllabel)) < (self.K+1)) | (len(np.unique(self.trlabel)) < (self.K+1)):
            if (not (len(np.unique(self.tllabel)) == (self.K+1))) | (not (len(np.unique(self.trlabel)) == (self.K+1))):
                print("some original true clusters are totally removed!")
        else:
            #if (len(np.unique(self.tllabel)) < (self.K)) | (len(np.unique(self.trlabel)) < (self.K)):
            if (not (len(np.unique(self.tllabel)) == self.K)) | (not (len(np.unique(self.trlabel)) == self.K)):
                print("some original true clusters are totally removed!")
        
        
    def add_kernel(self, kernel_type='rbf'):  #kernel vs. kernel_type
        
        print(".add_kernel() get called")
        ###gamma : float, default None. If None, defaults to 1.0 / n_features
        #K1 = rbf_kernel(self.iX, Y=None, gamma=self.gamma)
        #K2 = rbf_kernel(self.iY, Y=None, gamma=self.gamma)
        K1 = pairwise_kernels(self.iX, Y=None, metric=kernel_type, gamma=self.gamma)
        K2 = pairwise_kernels(self.iY, Y=None, metric=kernel_type, gamma=self.gamma)
        """
        #exponential kernel
        K1 = laplacian_kernel(self.iX, Y=None, gamma=self.gamma)
        K2 = laplacian_kernel(self.iY, Y=None, gamma=self.gamma)
        """
        lkernel = K1 + self.tao*np.identity(self.leftnodes)
        rkernel = K2 + self.tao*np.identity(self.rightnodes)
        #self.A = np.dot(np.dot(lkernel, self.A), rkernel)
        self.KA = np.dot(np.dot(lkernel, self.A), rkernel)
        ###check the elements in self.A, if it is too large!!!
        #print(np.amax(self.KA))
        
        #The bi_adjacency matrix. self.W == self.W.T
        self.W = np.concatenate((np.concatenate((np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16), self.KA), axis=1), 
                                 np.concatenate((self.KA.T, np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)), axis=1)), axis=0)
        print(".add_kernel() get ended")
    
    
    def sub_add_kernel(self, X, Y, A, kernel_type='rbf'):
    
        ###gamma : float, default None. If None, defaults to 1.0 / n_features
        #K1 = rbf_kernel(self.iX, Y=None, gamma=self.gamma)
        #K2 = rbf_kernel(self.iY, Y=None, gamma=self.gamma)
        K1 = pairwise_kernels(X, Y=None, metric=kernel_type, gamma=self.gamma)
        K2 = pairwise_kernels(Y, Y=None, metric=kernel_type, gamma=self.gamma)
        """
        #exponential kernel
        K1 = laplacian_kernel(self.iX, Y=None, gamma=self.gamma)
        K2 = laplacian_kernel(self.iY, Y=None, gamma=self.gamma)
        """
        lkernel = K1 + self.tao*np.identity(A.shape[0])
        rkernel = K2 + self.tao*np.identity(A.shape[1])
        #self.A = np.dot(np.dot(lkernel, self.A), rkernel)
        A = np.dot(np.dot(lkernel, A), rkernel)
        ###check the elements in self.A, if it is too large!!!
        #print(np.amax(A))
        
        #The bi_adjacency matrix. self.W == self.W.T
        W = np.concatenate((np.concatenate((np.zeros((A.shape[0], A.shape[0]), dtype=np.uint16), A), axis=1),
                            np.concatenate((A.T, np.zeros((A.shape[1], A.shape[1]), dtype=np.uint16)), axis=1)), axis=0)
        return(W)
    

    def add_rank_based(self, rank_d=5):

        ###np.absolute(x, out=x)
        pearsonr_X = np.absolute(1 - (euclidean_distances(self.iX)**2)/(2*self.left_dim))
        pearsonr_Y = np.absolute(1 - (euclidean_distances(self.iY)**2)/(2*self.right_dim))
        
        np.fill_diagonal(pearsonr_X, 0) #no self-connect edge
        np.fill_diagonal(pearsonr_Y, 0) #no self-connect edge
        
        rank_X = np.array(list(map(lambda x: rankdata(x, method='ordinal'), pearsonr_X)), dtype=np.uint16) #Unsigned integer (0 to 65535)
        rank_Y = np.array(list(map(lambda x: rankdata(x, method='ordinal'), pearsonr_Y)), dtype=np.uint16) #Unsigned integer (0 to 65535)
        
        coexpress_X = np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16)
        coexpress_Y = np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)
        
        coexpress_X[rank_X > (self.leftnodes - rank_d)] = 1
        coexpress_Y[rank_Y > (self.rightnodes - rank_d)] = 1
        
        np.add(coexpress_X, coexpress_X.T, out=coexpress_X)
        np.add(coexpress_Y, coexpress_Y.T, out=coexpress_Y) 
        """
        coexpress_X = coexpress_X + coexpress_X.T
        coexpress_Y = coexpress_Y + coexpress_Y.T
        """
        #np.where(co_express == 0, co_express, 1) # Note: broadcasting.
        coexpress_X[coexpress_X > 0] = 1 #numpy.uint16
        coexpress_Y[coexpress_Y > 0] = 1 #numpy.uint16
        
        #The bi_adjacency matrix. self.W == self.W.T
        self.W = np.concatenate((np.concatenate((coexpress_X, self.A), axis=1), 
                                 np.concatenate((self.A.T, coexpress_Y), axis=1)), axis=0)

        
    def add_value_based(self, value_t=0.95):
        
        print(".add_value_based() get called")
        """
        index_x = np.array(list(combinations(np.arange(self.leftnodes), 2)))
        index_y = np.array(list(combinations(np.arange(self.rightnodes), 2)))
        """
        """
        ###RuntimeWarning: invalid value encountered in double_scalars r = r_num / r_den. 
        pearsonr_X = np.absolute(np.array([pearsonr(self.iX[i,], self.iX[j,])[0] for (i, j) in index_x]))
        pearsonr_Y = np.absolute(np.array([pearsonr(self.iY[i,], self.iY[j,])[0] for (i, j) in index_y]))
        """
        """
        ###the equivalence of normalized Euclidean distance and Pearson Coefficient: dE(rnorm, snorm)**2 = 2*d*dP (rnorm, snorm)
        pearsonr_X = np.absolute(np.array([1-(euclidean(self.iX[i,], self.iX[j,])**2)/(2*self.left_dim) for (i, j) in index_x]))
        pearsonr_Y = np.absolute(np.array([1-(euclidean(self.iY[i,], self.iY[j,])**2)/(2*self.right_dim) for (i, j) in index_y]))
        
        pearsonr_X = (pearsonr_X >= value_t).astype(np.uint16)
        pearsonr_Y = (pearsonr_Y >= value_t).astype(np.uint16)
        
        pearsonr_X = squareform(pearsonr_X, force='tomatrix') #the diagonal elements are zero. np.fill_diagonal(pearsonr_X, 1), no edge connect self
        pearsonr_Y = squareform(pearsonr_Y, force='tomatrix')
        """  
        pearsonr_X = np.absolute(1 - (euclidean_distances(self.iX)**2)/(2*self.left_dim))
        pearsonr_Y = np.absolute(1 - (euclidean_distances(self.iY)**2)/(2*self.right_dim))
        
        np.fill_diagonal(pearsonr_X, 0) #no self-connect edge
        np.fill_diagonal(pearsonr_Y, 0) #no self-connect edge
        """
        np.greater_equal(pearsonr_X, value_t, out=pearsonr_X, dtype=np.uint16) #dtype=np.uint16! #check the matrix is symmetric
        np.greater_equal(pearsonr_X, value_t, out=pearsonr_X, dtype=np.uint16) #dtype=np.uint16! #check the matrix is symmetric
        """
        pearsonr_X = (pearsonr_X >= value_t).astype(np.uint16) #check the matrix is symmetric
        pearsonr_Y = (pearsonr_Y >= value_t).astype(np.uint16) #check the matrix is symmetric
        
        #The bi_adjacency matrix. self.W == self.W.T
        self.W = np.concatenate((np.concatenate((pearsonr_X, self.A), axis=1), 
                                 np.concatenate((self.A.T, pearsonr_Y), axis=1)), axis=0)
        
        
    def find_connected_components(self):

        bi_graph = csr_matrix(self.W) #csr_matrix().toarray()
        N_components, component_list = connected_components(bi_graph, directed=False) #dtype=int32
        self.N_components = N_components
        self.component_list = component_list  
        print("Number of connected components: ", self.N_components)
        print("Size of each connected components: ", [np.sum(self.component_list == i) for i in range(self.N_components)])
        
        
    def keep_largest_component(self):
        """
        Delete the other connected components, and only keep the largest one.
        Need to check the # of connected components w/ kernel after calling this method
        ###considering directly slice the kernel matrix or rank_based correlation matrix (self.W) instead   
        ###reconstruct the kernel matrix /others??? consider change in cluster.py???
        """
        big_comp = np.argmax(np.array([np.sum(self.component_list == i) for i in range(self.N_components)])) #.astype(np.int32) #numpy.int64
        
        #You may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index
        left_data_mask = (self.component_list == big_comp)[ : self.leftnodes]
        right_data_mask = (self.component_list == big_comp)[self.leftnodes : ]
        
        if left_data_mask.any() and right_data_mask.any():
            #self.left_data = self.left_data[left_data_mask]
            #self.right_data = self.right_data[right_data_mask]
            self.iX = self.iX[left_data_mask]
            self.iY = self.iY[right_data_mask]
            self.tllabel = self.tllabel[left_data_mask]
            self.trlabel = self.trlabel[right_data_mask]
            self.A = self.A[left_data_mask][:, right_data_mask]
            self.update_data()
            self.update_sample()
        else:
            print("The largest component only contain one side nodes")

    @staticmethod
    def sub_find_connected_components(W):
        bi_graph = csr_matrix(W) #csr_matrix().toarray()
        N_components, component_list = connected_components(bi_graph, directed=False) #dtype=int32
        #print("Number of connected components: ", N_components)
        #print("Size of each connected components after reduction: ", [np.sum(component_list == i) for i in range(N_components)])

    @staticmethod
    def sub_keep_largest_component(W, left_length):
        
        bi_graph = csr_matrix(W) #csr_matrix().toarray()
        N_components, component_list = connected_components(bi_graph, directed=False) #dtype=int32
        #self.N_components = N_components
        #self.component_list = component_list
        
        #print("Number of connected components: ", N_components)
        #print("Size of each connected components: ", [np.sum(component_list == i) for i in range(N_components)])
        
        big_comp = np.argmax(np.array([np.sum(component_list == i) for i in range(N_components)])) #.astype(np.int32) #numpy.int64
        
        #You may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index
        left_data_mask = (component_list == big_comp)[ : left_length]
        right_data_mask = (component_list == big_comp)[left_length : ]
        
        if left_data_mask.any() and right_data_mask.any():
            #self.left_data = self.left_data[left_data_mask]
            #self.right_data = self.right_data[right_data_mask]
            
            #print(sum(left_data_mask))
            #print(sum(right_data_mask))
            return((left_data_mask, right_data_mask, N_components))
        else:
            print("The largest component only contain one side nodes")

    
    def eigenmatrix_sym(self, K):
        self.D = np.diag(np.sum(self.W, axis=1))
        self.L_sym = np.identity(self.leftnodes + self.rightnodes, dtype=np.uint16) - (self.W.T * (1/np.sqrt(np.diag(self.D)))).T * (1/np.sqrt(np.diag(self.D)))

        if np.isreal(self.L_sym).all():
            
            if (self.L_sym == self.L_sym.T).all():
                print("L_sym is symmetric")
            else:
                print("L_sym is asymmetric")
                self.L_sym = (self.L_sym + self.L_sym.T)/2

        print("solve the eigen problem by eigsh")
        eigvalues, eigvectors = eigsh(A=self.L_sym, k=K, which='SM')

        self.U = normalize(eigvectors[:, np.arange(K)], norm="l2", axis=1, copy=True, return_norm=False)


    def eigenmatrix_rw(self, K):
        
        print("start to calculate the unnormalized laplacian matrix")
        self.D = np.diag(np.sum(self.W, axis=1))
        #print((np.diag(self.D) == 0).any())
        self.L_unn = self.D - self.W
        
        if (self.L_unn == self.L_unn.T).all():
            print("L_unn is symmetric") 
        else:
            print("L_unn is asymmetric") #possible when adopt value_based
            self.L_unn = (self.L_unn + self.L_unn.T)/2

        print("solve the generalized eigen problem by eigsh")
        start_time = time.time()
        l = int(np.ceil(np.log2(K))) + 1
        eigvalues, eigvectors = eigsh(A=self.L_unn, k=l, M=self.D, which='SM')
        print(time.time() - start_time)
        
        if np.isreal(eigvalues).all() & np.isreal(eigvectors).all():
            self.U = normalize(eigvectors[:, np.arange(l)], norm="l2", axis=1, copy=True, return_norm=False)
        else:
            print("The generalized eigenvalue problem (L_rw) has complex eigenvalues/eigenvectors!")
      
    
    @staticmethod
    def sub_eigenmatrix_sym(K, W):
    
        D = np.diag(np.sum(W, axis=1))
        #print("possiblity of sub sampling has nodes with no edge connected: ", (np.diag(D) == 0).any())  #self.simulation_type = 'sp_tsc'
        #print("total sum of zero diagnonal elements: ", sum(np.diag(D) == 0))
        
        L_sym = np.identity(W.shape[0], dtype=np.uint16) - (W.T * (1/np.sqrt(np.diag(D)))).T * (1/np.sqrt(np.diag(D)))
        #print(time.time() - start_time)
        
        #print((self.L_sym == self.L_sym.T).all()) #False in the kernel_based case #True in the rank_based case
        if np.isreal(L_sym).all(): #this step can be removed
            
            if (L_sym == L_sym.T).all():
                pass
                #print("L_sym is symmetric")
            else:
                #print("L_sym is asymmetric")
                L_sym = (L_sym + L_sym.T)/2
            
            #print("start calculate the eigenvectors")
            #start_time = time.time()
            #l = int(np.ceil(np.log2(K))) + 1
            eigvalues, eigvectors = eigsh(A=L_sym, k=K, which='SM')
            #print(time.time() - start_time)
            U = normalize(eigvectors[:, np.arange(K)], norm="l2", axis=1, copy=True, return_norm=False)
            #print(eigvalues)
            
        else:
            print('The Laplacian matrix L_sym has complex elements!')
        
        return(U)
        

    @staticmethod
    def sub_eigenmatrix_rw(K, W):

        print("possiblity of sub sampling has nodes with no edge connected: ", (np.diag(D) == 0).any())  #self.simulation_type = 'sp_tsc'
        print("total sum of zero diagnonal elements: ", sum(np.diag(D) == 0))
        
        D = np.diag(np.sum(W, axis=1))
        #print((np.diag(self.D) == 0).any())
        L_unn = D - W
        
        if (L_unn == L_unn.T).all():
            #print("L_unn is symmetric")
            pass
        else:
            print("L_unn is asymmetric") #possible when adopt value_based
            L_unn = (L_unn + L_unn.T)/2
        
        ###Generalized eigenvalue problem
        print("solve the generalized eigen problem by eigsh")
        #start_time = time.time()
        l = int(np.ceil(np.log2(K))) + 1
        eigvalues, eigvectors = eigsh(A=L_unn, k=l, M=D, which='SM')

        
        if np.isreal(eigvalues).all() & np.isreal(eigvectors).all():
            """normalized each row of the eigenvectors [Zahra]"""
            #self.U = eigvectors[:, np.arange(K)]
            U = normalize(eigvectors[:, np.arange(l)], norm="l2", axis=1, copy=True, return_norm=False)
            #print(eigvalues)
        else:
            print("The generalized eigenvalue problem (L_rw) has complex eigenvalues/eigenvectors!")
            
        return(U)
    
            
    @staticmethod
    def bounding_box(X):
        dimension = X.shape[1]
        bmin, bmax = [], []
        for d in np.arange(dimension):
            #the minimum value for the d_th dimension (or d_th column)
            bmin.append(min(X,key=lambda a:a[d])[d])
            #the maximum value for the d_th dimension
            bmax.append(max(X,key=lambda a:a[d])[d])
            #xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
            #ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
        #return (xmin,xmax), (ymin,ymax)

        return bmin, bmax
    

    def fit_bitkmeans(self, target, k_min, alpha, beta, seq_num, iteration, resamp_num, remain_p, k_stop):

        #random.seed(0)
        print(".fit_bitkmeans() method get called")
        #self.simulation_type = "tsc"
        #print(self.simulation_type)
        print(self.simulation_type)
        
        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = k_min) #l = K
        
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = k_min) #l = K
        
        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")

        nfound = 0
        found = True 
        k_max = k_min+10
        #k_max = k_min+15
        k0 = k_min
        #k = k0

        tclust = [] #store the corresponding elements in self.leftindexs/self.rightindexs for each found cluster
        #tclust_id = []
            
        while ((nfound < target) and ((len(self.leftindexs)/self.leftnodes) > remain_p) and ((len(self.rightindexs)/self.rightnodes) > remain_p) and (found or (k <= k_max))):
                
            if found:
                print("Looking for tight cluster " + str(nfound + 1) + "...")
                k = k0
                candidates = []
                for i in np.arange(seq_num): 
                    print("k = " + str(k + i))
                    candidates.append(self.find_candidates(k + i, alpha, iteration, resamp_num))

            else: 
                del candidates[0]  
                candidates.append(self.find_candidates(k+seq_num-1, alpha, iteration, resamp_num)) 

            beta_temp, index_m = self.calc_beta(candidates)

            if np.any(beta_temp >= beta):
                found = True
                nfound = nfound + 1
                print(str(nfound) + " tight cluster found!")
                if k0 > k_stop:
                    k0 = k0 - 1

                found_temp = candidates[seq_num-1][np.amin(index_m[beta_temp>=beta, seq_num-1])]
                found_temp = list(map(int, found_temp))
                #tclust.append(found_temp)
                #sub_tclust_id = []

                for index_i in found_temp:

                    if index_i in self.leftindexs:
                        self.leftindexs = self.leftindexs[self.leftindexs != index_i]
                    else:
                        self.rightindexs = self.rightindexs[self.rightindexs != (index_i - self.leftnodes)]

                tclust.append(found_temp)

                print("Cluster size: "+ str(len(found_temp)))

                self.update_sample()

                print("Remaining number of points on the left side: " + str(len(self.leftindexs)))
                print("Remaining number of points on the right side: " + str(len(self.rightindexs)))

            else:
                found = False
                k = k + 1
                print("Not found!")
                print("k = " + str(k))

        clust_id = np.ones(self.leftnodes + self.rightnodes, dtype=np.uint16)*nfound
        for i, tclust_sub in enumerate(tclust, start=0):
            clust_id[tclust_sub] = i

        comembership_matrix = (clust_id[:self.leftnodes][:, np.newaxis] == clust_id[self.leftnodes:][np.newaxis, :]).astype(np.uint16)
        comembership_matrix[clust_id[:self.leftnodes] == nfound] = 0
        comembership_matrix = comembership_matrix.tolist()

        true_label = dict(indexs=np.concatenate((self.tllabel, self.trlabel), axis=0), noise=(self.noise_ratio>0))
        clust_id = dict(indexs=clust_id, noise=(clust_id==nfound).any())

        rand_index = weighted_rand_score(true_label, clust_id)

        output = dict(comembership_matrix=comembership_matrix, rand_index=rand_index)
        
        return(output)

    
    def find_candidates(self, k, alpha, iteration=3, resamp_num=20):

        D_bar = 0
              
        for itr in range(iteration):
              
            print("Start iteration: %s" % (itr+1), "with %s" % (resamp_num), "processes")

            # Define an output queue
            output = mp.Queue()

            if self.simulation_type == "tsc":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]

            if self.simulation_type == "tscu":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_tscu, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]
     
            if self.simulation_type == "sp_tsc":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_sp_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]
        
            # Run processes
            for p in processes:
                p.start()

            D_bar_sub = 0 #broadcasting
            for p in processes:
                #D_bar_sub += output.get()
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)
            #print("End getting comembership matrixs from the Queue")
            
            del co_labels
            gc.collect()

            for p in processes:  
                p.join()
            #print("End joining the sub-processes")
            
            del output
            gc.collect()
            
            D_bar += (D_bar_sub/resamp_num)
            
            del D_bar_sub
            gc.collect()

            print("End iteration: ", itr+1)

        D_bar = D_bar/iteration

        D_bar_index = np.concatenate((self.leftindexs, self.rightindexs+self.leftnodes), axis=0)
 
        compressed_distance = squareform((1-D_bar), force='tovector', checks=False)
        Z = linkage(compressed_distance, method='complete')
        max_d = 1-alpha
        hclusters = fcluster(Z, t=max_d, criterion='distance') # an array
        hclusters_nb = np.amax(hclusters) #max(hclusters)

        res = []
        for hclusters_k_index in range(1, hclusters_nb+1): 
            candidate = D_bar_index[hclusters == hclusters_k_index] #Boolean indexing
            if (candidate <= np.amax(self.leftindexs)).any() and (candidate >= self.leftnodes).any(): #len>=2
                res.append(candidate)
            #if (len(np.intersect1d(candidate, Init.leftindexs))>0) and (len(np.intersect1d(candidate, Init.rightindexs+Init.leftnodes))>0):

        #order = list(reversed(np.argsort(list(map(len, res))))) ###max(res, key=lambda a:len(a))
        order = np.argsort(list(map(len, res)))[::-1]

        #res = [res[i] for i in order][: top_can]  ###cehck this part???
        res = [res[i] for i in order] ###res[order]??? check

        return res


    def get_consensus_matrix(self, k, iteration=3, resamp_num=20):

        D_bar = 0
                
        for itr in range(iteration):
                    
            print("Start iteration: %s" % (itr+1), "with %s" % (resamp_num), "processes")
                        
            # Define an output queue
            output = mp.Queue()
                            
            if self.simulation_type == "tsc":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]
                                    
            if self.simulation_type == "tscu":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_tscu, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]
                                            
            if self.simulation_type == "sp_tsc":
                # Setup a list of processes that we want to run
                processes = [mp.Process(target=self.simulation_sp_tsc, args=(i + itr*resamp_num, k, output)) for i in range(resamp_num)]
                                                    
            # Run processes
            for p in processes:
                p.start()

            D_bar_sub = 0 #broadcasting
            for p in processes:
                #D_bar_sub += output.get()
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)
            #print("End getting comembership matrixs from the Queue")
            
            del co_labels
            gc.collect()

            for p in processes:
                p.join()
            #print("End joining the sub-processes")
        
            del output
            gc.collect()

            D_bar += (D_bar_sub/resamp_num)
    
            del D_bar_sub
            gc.collect()
            
            print("End iteration: ", itr+1)

        D_bar = D_bar/iteration
    
        return D_bar


    def simulation_tsc(self, random_seed, clusters_nb, output):
        
        np.random.seed(seed=random_seed)

        lindexs_i = np.sort(np.random.choice(np.arange(len(self.leftindexs)), self.m, replace=False)) ###we could just use len(Init.selfindexs)
        rindexs_i = np.sort(np.random.choice(np.arange(len(self.rightindexs)), self.n, replace=False))

        lmask = np.zeros(len(self.leftindexs), dtype=bool)  ###np.repeat(False, 10)  ###np.zeros_like(Init.leftindexs, dtype=bool)
        lmask[lindexs_i] = True

        rmask = np.zeros(len(self.rightindexs), dtype=bool)
        rmask[rindexs_i] = True

        lindexs = self.leftindexs[lindexs_i]  #indexing of the indexs
        rindexs = self.rightindexs[rindexs_i]

        u_indexs = np.concatenate((lindexs, rindexs+self.leftnodes), axis=0)
        sub_U = self.U[u_indexs]  #a copy of the orginal array

        X1 = self.iX[lindexs]
        Y1 = self.iY[rindexs]

        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_ #numpy.array dtype=int32  

        left_labels = labels[ : self.m]  #llabels = labels[np.arange(Init.m)]   
        right_labels = labels[self.m : ] #rlabels = labels[Init.m : (Init.m + Init.n)]

        X_labels = np.empty(len(self.leftindexs), dtype = np.uint16)  #Unsigned integer (0 to 65535)
        Y_labels = np.empty(len(self.rightindexs), dtype = np.uint16)  #Unsigned integer (0 to 65535)
        X_labels[lmask] = left_labels #X_labels.dtype: dtype('uint16')
        Y_labels[rmask] = right_labels #Y_labels.dtype: dtype('uint16')

        X_centroid = []
        for i in np.arange(clusters_nb, dtype=np.uint16): 
            if sum(left_labels == i) > 0:  #(left_labels == i).any()
                x_centroid = np.sum(X1[left_labels == i], axis=0) / sum(left_labels == i)  #sum along the column
                X_centroid.append((i, x_centroid))

        Y_centroid = []
        for j in np.arange(clusters_nb, dtype=np.uint16):
            if sum(right_labels == j) > 0: #(right_labels == j).any()
                y_centroid = np.sum(Y1[right_labels == j], axis=0) / sum(right_labels == j)
                Y_centroid.append((j, y_centroid))

        X_centroid_names, X_centroid_m = zip(*X_centroid)    #X_centroid_m, X_centroid_names are tuples
        #X_centroid_names = np.array(X_centroid_names) 
        X_centroid_m = np.array(X_centroid_m)  #.shape = (variables, observations) #each column represents a observation, with variables in the rows

        Y_centroid_names, Y_centroid_m = zip(*Y_centroid)
        #Y_centroid_names = np.array(Y_centroid_names)
        Y_centroid_m = np.array(Y_centroid_m)   #.shape = (variables, observations)
        
        euclidean_X = euclidean_distances(self.X[~lmask], X_centroid_m)
        euclidean_Y = euclidean_distances(self.Y[~rmask], Y_centroid_m)
        
        #find the minimum index along the columns for each row
        X_labels[~lmask] = np.array([X_centroid_names[i] for i in np.argmin(euclidean_X, axis=1)]) #type(X_centroid_names[0]): numpy.uint16
        Y_labels[~rmask] = np.array([Y_centroid_names[i] for i in np.argmin(euclidean_Y, axis=1)]) #type(Y_centroid_names[0]): numpy.uint16

        co_labels = np.concatenate((X_labels, Y_labels), axis=0)  #len = self.leftindexs + self.rightindexs
        
        #output.put((co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16))
        output.put(co_labels)
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "get the co_labels vector.")
        
        del co_labels
        gc.collect()
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "end.")


    def simulation_tscu(self, random_seed, clusters_nb, output):

        np.random.seed(seed=random_seed)

        lindexs_i = np.sort(np.random.choice(np.arange(len(self.leftindexs)), self.m, replace=False)) ###or we could just use len(Init.leftindexs)
        rindexs_i = np.sort(np.random.choice(np.arange(len(self.rightindexs)), self.n, replace=False))

        lmask = np.zeros(len(self.leftindexs), dtype=bool)  ###np.repeat(False, 10)  ###np.zeros_like(Init.leftindexs, dtype=bool)
        lmask[lindexs_i] = True

        rmask = np.zeros(len(self.rightindexs), dtype=bool)
        rmask[rindexs_i] = True

        lindexs = self.leftindexs[lindexs_i]  #indexing of the indexs
        rindexs = self.rightindexs[rindexs_i]

        u_indexs = np.concatenate((lindexs, rindexs+self.leftnodes), axis=0) 
        sub_U = self.U[u_indexs]  #a copy of the orginal array

        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, 
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_

        U_remain = np.concatenate((self.U[self.leftindexs][~lmask], 
                                   self.U[(self.leftnodes + self.rightindexs)][~rmask]), axis=0)

        labels_remain = kmeans.predict(U_remain)

        left_labels = labels[ : self.m]  #llabels = labels[np.arange(Init.m)]   
        right_labels = labels[self.m : ] #rlabels = labels[Init.m : (Init.m + Init.n)]

        X_labels = np.empty(len(self.leftindexs), dtype = np.uint16) #Unsigned integer (0 to 65535)
        Y_labels = np.empty(len(self.rightindexs), dtype = np.uint16) #Unsigned integer (0 to 65535)
        X_labels[lmask] = left_labels #X_labels.dtype: dtype('uint16')    
        Y_labels[rmask] = right_labels #Y_labels.dtype: dtype('uint16')

        X_labels[~lmask] = labels_remain[ : sum(~lmask)]
        Y_labels[~rmask] = labels_remain[sum(~lmask) : ]

        co_labels = np.concatenate((X_labels, Y_labels), axis=0)  #len = self.leftindexs + self.rightindexs

        output.put(co_labels)
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "get the co_labels vector.")
        
        del co_labels
        gc.collect()
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "end.")
    
     
    def simulation_sp_tsc(self, random_seed, clusters_nb, output):

        print(".simulation_sp_tsc() get called")
        np.random.seed(seed=random_seed)

        ###random.sample(population, k)
        lindexs_i = np.sort(np.random.choice(np.arange(len(self.leftindexs)), self.m, replace=False)) ###we could just use len(Init.selfindexs)
        rindexs_i = np.sort(np.random.choice(np.arange(len(self.rightindexs)), self.n, replace=False))

        lmask = np.zeros(len(self.leftindexs), dtype=bool)  ###np.repeat(False, 10)  ###np.zeros_like(Init.leftindexs, dtype=bool)
        lmask[lindexs_i] = True

        rmask = np.zeros(len(self.rightindexs), dtype=bool)
        rmask[rindexs_i] = True

        lindexs = self.leftindexs[lindexs_i]  #indexing of the indexs
        rindexs = self.rightindexs[rindexs_i]

        #u_indexs = np.concatenate((lindexs, rindexs+self.leftnodes), axis=0)
        #sub_U = self.U[u_indexs]  #a copy of the orginal array
  
        sub_A = self.A[lindexs][:, rindexs]
        X1 = self.iX[lindexs]
        Y1 = self.iY[rindexs]

        if self.kernel == True:
            sub_W = self.sub_add_kernel(X1, Y1, sub_A, kernel_type='rbf')
            left_mask, right_mask, num_comp = self.sub_keep_largest_component(sub_W, left_length=self.m)
            
            if num_comp > 1:
                sub_A = sub_A[left_mask][:, right_mask]
                X1 = X1[left_mask]
                Y1 = Y1[right_mask]
            
                lmask = np.zeros(len(self.leftindexs), dtype=bool)
                lmask[lindexs_i[left_mask]] = True
            
                rmask = np.zeros(len(self.rightindexs), dtype=bool)
                rmask[rindexs_i[right_mask]] = True
            
                sub_W = self.sub_add_kernel(X1, Y1, sub_A, kernel_type='rbf')
                #recheck if the sub-sample construct a fully connected component
                self.sub_find_connected_components(sub_W)
            else:
                pass

        else:
            sub_W = np.concatenate((np.concatenate((np.zeros((sub_A.shape[0], sub_A.shape[0]), dtype=np.uint16), sub_A), axis=1),
                                    np.concatenate((sub_A.T, np.zeros((sub_A.shape[1], sub_A.shape[1]), dtype=np.uint16)), axis=1)), axis=0)
            self.sub_find_connected_components(sub_W)
        
        sub_U = self.sub_eigenmatrix_sym(clusters_nb, sub_W)

        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_ #numpy.array dtype=int32
        
        left_labels = labels[ : sum(lmask)]
        right_labels = labels[sum(lmask) : ]
        
        X_labels = np.empty(len(self.leftindexs), dtype = np.uint16)  #Unsigned integer (0 to 65535)
        Y_labels = np.empty(len(self.rightindexs), dtype = np.uint16)  #Unsigned integer (0 to 65535)
        X_labels[lmask] = left_labels #X_labels.dtype: dtype('uint16')
        Y_labels[rmask] = right_labels #Y_labels.dtype: dtype('uint16')

        X_centroid = []
        for i in np.arange(clusters_nb, dtype=np.uint16): 
            """It is possible that some clusters only contain single side genes after spectral clustering/Kmeans???"""
            if sum(left_labels == i) > 0:  #(left_labels == i).any()
                x_centroid = np.sum(X1[left_labels == i], axis=0) / sum(left_labels == i)  #sum along the column
                X_centroid.append((i, x_centroid))

        Y_centroid = []
        for j in np.arange(clusters_nb, dtype=np.uint16):
            if sum(right_labels == j) > 0: #(right_labels == j).any()
                y_centroid = np.sum(Y1[right_labels == j], axis=0) / sum(right_labels == j)
                Y_centroid.append((j, y_centroid))

        X_centroid_names, X_centroid_m = zip(*X_centroid)    #X_centroid_m, X_centroid_names are tuples
        #X_centroid_names = np.array(X_centroid_names) 
        X_centroid_m = np.array(X_centroid_m)  #.shape = (variables, observations) #each column represents a observation, with variables in the rows

        Y_centroid_names, Y_centroid_m = zip(*Y_centroid)
        #Y_centroid_names = np.array(Y_centroid_names)
        Y_centroid_m = np.array(Y_centroid_m)   #.shape = (variables, observations)

        euclidean_X = euclidean_distances(self.X[~lmask], X_centroid_m)
        euclidean_Y = euclidean_distances(self.Y[~rmask], Y_centroid_m)
        
        #find the minimum index along the columns for each row
        X_labels[~lmask] = np.array([X_centroid_names[i] for i in np.argmin(euclidean_X, axis=1)]) #type(X_centroid_names[0]): numpy.uint16
        Y_labels[~rmask] = np.array([Y_centroid_names[i] for i in np.argmin(euclidean_Y, axis=1)]) #type(Y_centroid_names[0]): numpy.uint16

        co_labels = np.concatenate((X_labels, Y_labels), axis=0)  #len = self.leftindexs + self.rightindexs

        #output.put((co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16))
        output.put(co_labels)
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "get the co_labels vector.")
        
        del co_labels
        gc.collect()
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "end.")
    
    
    @staticmethod
    def calc_beta(cand): #cand is a list of lists of arrays: #len(cand) = seq_num
        #Python.itertools 
        index_m = np.array(list(product(*[np.arange(x) for x in list(map(len, cand))])))
        beta = []

        for y in index_m:
            temp = list(map(lambda x: cand[x][y[x]], np.arange(len(cand)))) 
            #temp = list(map(np.arange(seq.num), function(z) candidates[z][y[z]]))
            i_temp = temp[0]
            u_temp = temp[0]  ###u.temp<-temp[[i]] in the original R code
            for j in range(1, len(cand)): 
                #i_temp = intersect(i_temp, set(temp[j]))
                #u_temp = union(u_temp, set(temp[j]))
                #i_temp = i_temp.intersection(set(temp[j]))
                #u_temp = i_temp.union(set(temp[j]))
                i_temp = np.intersect1d(i_temp, temp[j], assume_unique=True) ###???
                u_temp = np.union1d(u_temp, temp[j])

            similarity = len(i_temp)/len(u_temp)
            beta.append(similarity)

        beta = np.array(beta)   
        return(beta, index_m)

    
    def fit_spectral_clustering(self, clusters_nb):
        print("spectral clustering with the (same kernel embedded) inherited bi-adjacency matrix get called")
        #print("self.W.shape: ", self.W.shape)
        
        self.eigenmatrix_sym(K = clusters_nb)
        print("k = ", clusters_nb)
        
        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(self.U)
        labels = kmeans.labels_ #numpy.array dtype=int32  
        
        true_label = dict(indexs=np.concatenate((self.tllabel, self.trlabel), axis=0), noise=(self.noise_ratio>0))
        pred_label = dict(indexs=labels, noise=False)
        rand_index = weighted_rand_score(true_label, pred_label)
        output = dict(rand_index=rand_index)
        return(output)

 
    def fit_hierarchical_clustering(self, k, alpha, iteration, resamp_num, simulation_type):
        
        print("hierarchical clustering with the (same kernel embedded) inherited bi-adjacency matrix get called")
        self.simulation_type = simulation_type
        print(self.simulation_type)
        
        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = k) #l = K
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = k) #l = K

        print("k = ", k)
        result = self.find_candidates(k, alpha, iteration, resamp_num) #list of arrays (tight clusters candidate)
        
        result_id = np.ones(self.leftnodes + self.rightnodes, dtype=np.uint16)*len(result) 
        for i, result_sub in enumerate(result, start=0):
            result_id[result_sub] = i

        pred_label = dict(indexs=result_id, noise=(result_id==len(result)).any())
        true_label = dict(indexs=np.concatenate((self.tllabel, self.trlabel), axis=0), noise=(self.noise_ratio>0))
        rand_index = weighted_rand_score(true_label, pred_label)

        output = dict(rand_index=rand_index)
        return(output)


    def fit_consensus_clustering(self, k, iteration, resamp_num, simulation_type):
    
        print("consensus clustering with the (same kernel embedded) inherited bi-adjacency matrix get called")
        self.simulation_type = simulation_type
        print(self.simulation_type)
        
        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = k) #l = K
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = k) #l = K
        
        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")
        
        print("k = ", k)
        consensus_matrix = self.get_consensus_matrix(k, iteration, resamp_num) #list of arrays (tight clusters candidate)
        return(consensus_matrix)


if __name__ == "__main__":
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    #from sklearn.decomposition import PCA
    
    ##################### generate the network by SBM #####################
    ##################### plot the heatmaps of covariate matrices, bi-adjacency matrix and comembership matrix############
    
    #M, N = (80, 120)
    M, N = (50, 70)
    
    #K = 10
    K = 15
    
    #d_1, d_2 = (50, 50)
    
    d_1, d_2 = (2, 2)  #d_1, d_2 = (30, 30)

    #small_sigma = 1 #clusters are too tight
    #small_sigma = 5
    #small_sigma = 10
    small_sigma = 15 #useless
    #small_sigma = 30
    #small_sigma = 100

    #large_sigma = 1000 ###choice
    large_sigma = 10
    #large_sigma = 8
    #large_sigma = 15
    
    out_in_ratio = 5 #out_in_ratio = 0.2
    #out_in_ratio = 7 #out_in_ratio = 0.14
    
    #q = 0.05
    #q = 0.1
    #q = 0.03
    
    #q = np.arange(0.001, 0.052, 0.005)[2]
    q_list = [0.001, 0.0015, 0.0021, 0.0028, 0.0036, 0.0045, 0.0055, 0.007,
              0.010, 0.013,0.016, 0.019, 0.022, 0.025, 0.030, 0.035]
    q = q_list[-2]
    p = out_in_ratio*q
    
    #noise_ratio = 0.3 #noise_ratio = 0.2 #noise_ratio = 0.3 #noise_ratio = 0.4
    noise_ratio = 0.5
    
    #samp_p = 0.7
    samp_p = 0.8

    tao = 1 #tao = 5 #tao = 10

    kernel = False #kernel = True

    rank_based = False
    
    value_based = False
    
    gamma = None
    
    #simulation_type = 'tsc'
    simulation_type = 'sp_tsc'

    rank_d = 5
    
    value_t = 0.95

    #lambda_hat without considering the noise nodes.
    #lambda_average = (2*M*N*K/(M+N))*(q + (1/K)*(p-q))
    #print(lambda_average)

    #expected average degree
    #lambda_hat = (2*M*N*K/((M+N)*(1+noise_p)))*(q*((1+noise_p)**2) + (1/K)*(p-q))
    #print(lambda_hat)
    
    #average degree
    #lambda_hat = (2*np.sum(Initial.A))/(Initial.leftnodes+Initial.rightnodes)

    #perturbation = (lambda_hat/((M+N)*K*(1+noise_p)))*0.25 ###the elbow point are not so clear
    #perturbation = (lambda_hat/((M+N)*K*(1+noise_p)))*0.1
    #print(perturbation)
    
    data = SimData(M, N, K, d_1, d_2, small_sigma, large_sigma, p=p, q=q, noise_ratio=noise_ratio, samp_p=samp_p, 
                   tao=tao, kernel=kernel, rank_based=rank_based, value_based=value_based, gamma=gamma, 
                   simulation_type=simulation_type, rank_d=rank_d, value_t=value_t)
    
    #root_dir = 'project_fly_worm/result/figure'
    root_dir = '/home/yidan/Dropbox/project_fly_worm/simulation/result/figure'
    
    print(data.average_degree)
    print(data.expected_lambda)
    plt.figure(1)
    true_member = (data.tllabel[:, np.newaxis] == data.trlabel[np.newaxis, :]).astype(np.uint16)
    true_member[data.M*data.K:] = 0
    plt.matshow(true_member, cmap=plt.cm.Blues)
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    #cax = plt.axes([1.0, 0.1, 0.085, 0.77])
    #plt.colorbar(cax=cax)
    plt.savefig(root_dir + '/True_Member.pdf')

    plt.figure(2)
    plt.matshow(data.A, cmap=plt.cm.Blues)
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    #cax = plt.axes([1.0, 0.1, 0.085, 0.77])
    # plt.colorbar(cax=cax)
    plt.savefig(root_dir + '/Adjacency_Matrix.pdf')
    
    plt.figure(3)
    plt.figure(figsize=(10,10))
    ### the scatter plot of first 2 dimensions of X covariates ###
    #pca = PCA(n_components=2)
    #X_pca = pca.fit_transform(data.iX)
    plt.scatter(data.iX[:data.M*data.K, 0], data.iX[:data.M*data.K, 1], color='#2166ac', alpha=1, label='true nodes')
    plt.scatter(data.iX[data.M*data.K:, 0], data.iX[data.M*data.K:, 1], color='#d6604d', alpha=1, label='noise nodes')
    plt.legend()
    plt.savefig(root_dir + '/X_scatter_plot.pdf')
    
    plt.figure(4)
    plt.figure(figsize=(10,10))
    ### the scatter plot of first 2 dimensions of Y covariates ###
    #pca = PCA(n_components=2)
    #Y_pca = pca.fit_transform(data.iY)
    plt.scatter(data.iY[:data.N*data.K, 0], data.iY[:data.N*data.K, 1], c='#2166ac', alpha=1, label='true nodes')
    plt.scatter(data.iY[data.N*data.K:, 0], data.iY[data.N*data.K:, 1], c='#d6604d', alpha=1, label='noise nodes')
    plt.legend()
    plt.savefig(root_dir + '/Y_scatter_plot.pdf')
    
    plt.figure(5)
    plt.matshow(data.iX, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
    cax = plt.axes([1.1, 0.09, 0.11, 0.77])
    plt.colorbar(cax=cax)
    plt.savefig(root_dir + '/X_covariate.pdf')
    
    plt.figure(6)
    plt.matshow(data.iY, interpolation='nearest', aspect='auto', cmap=plt.cm.Blues)
    cax = plt.axes([1.1, 0.09, 0.11, 0.77])
    plt.colorbar(cax=cax)
    plt.savefig(root_dir + '/Y_covariate.pdf')




