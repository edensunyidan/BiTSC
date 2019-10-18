
# coding: utf-8

# In[8]:

#"/home/yidan/Dropbox/project_fly_worm/biptightkmeans/cluster.py"

### project_fly_worm ###

### main.py/main_eigen_plot.py file has same directory as package eigen_spectral
from eigen_spectral.eigen_decomposition import my_spectral_embedding

import pandas as pd
import numpy as np
import math
#import random
from copy import deepcopy
import collections

from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from itertools import product
from itertools import combinations
import itertools
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import rankdata
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import pairwise_kernels

#from numpy.linalg import inv
#from scipy.linalg import sqrtm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, leaves_list

from sklearn.cluster import spectral_clustering
from sklearn.manifold import spectral_embedding
#from sklearn.metrics import mutual_info_score
#from sklearn.metrics import normalized_mutual_info_score
#from scipy.sparse import csgraph

import multiprocessing as mp
import os
import json
import time
import gc

import scanpy as sc
import anndata

from collections import OrderedDict
import string
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


__Author__ = "Yidan Sun"

"""Consider the log-transformation of the data"""

"""If not keep it as a connected component"""
    
"""If we consider co-expression edges or add kernel, do we still need to remove genes that are not in the orthologs list"""
    
"""combine the timecourse and tissuecell data together"""
    
"""Delete the genes which has zero expression and also no orthologs"""
    
"""How to deal with the zero values if gene expression is zeros"""


class BitKmeans:
    """Bipartite Tight Kmeans clustering"""
    
    def __init__(self, left_raw_data, right_raw_data, orthologs_data, samp_p, tao, drop=False, kernel=False, rank_based=False, value_based=False, gamma=None, simulation_type='sp_tsc', use_pca=False, rank_d=5, value_t=0.95):
        
        self.left_data = left_raw_data.copy()
        self.right_data = right_raw_data.copy()
        self.orthologs_data = orthologs_data.copy()
        self.left_data = self.left_data.dropna(axis=0, how='any')
        self.right_data = self.right_data.dropna(axis=0, how='any')
        #orthologs_data belongs to fly-worm data

        print("fly raw data size: ", self.left_data.shape)
        print("worm raw data size: ", self.right_data.shape)
        print("orthologs raw data size: ", self.orthologs_data.shape)
        
        """
        if drop == True:
            self.left_data = self.left_data[(self.left_data.iloc[:, 1:] != 0).any(axis=1) |
                                            self.left_data.loc[:, 'gene_id'].isin(list(self.orthologs_data.loc[:, 'V4']))]
            self.right_data = self.right_data[(self.right_data.iloc[:, 1:] != 0).any(axis=1) |
                                              self.right_data.loc[:, 'gene_id'].isin(list(self.orthologs_data.loc[:, 'V5']))]
            print("keeping the genes in the original orthologs, and adding more other genes whose expression are non-zeros ")
            print("fly data size with allZeros and no edges connceted rows deleted: ", self.left_data.shape[0])
            print("worm data size with allZeros and no edges connected rows deleted: ", self.right_data.shape[0])
        """
        
        if drop == True:
            self.left_data = self.left_data[(self.left_data.iloc[:, 1:] != 0).any(axis=1)]
            self.right_data = self.right_data[(self.right_data.iloc[:, 1:] != 0).any(axis=1)]
            
            self.left_data = self.left_data[self.left_data.loc[:, 'gene_id'].isin(list(self.orthologs_data.loc[:, 'V4']))]
            self.right_data = self.right_data[self.right_data.loc[:, 'gene_id'].isin(list(self.orthologs_data.loc[:, 'V5']))]
            
            ###the orthologs_data originally belong to the union of left_data and right_data
            self.slice_orthologs()
            
            print("drop genes whose expression values are allZeros ")
            print("expressed fly genes size: ", self.left_data.shape[0])
            print("expressed worm genes size: ", self.right_data.shape[0])
            
        self.left_data = self.left_data.sort_values(by=[self.left_data.columns[0]], axis=0, ascending=True, inplace=False) #return the copy of the original pandas dataframe
        self.left_id = self.left_data.loc[:, self.left_data.columns[0]].values # 2d numpy array
        self.iX = self.left_data.drop(labels=self.left_data.columns[0], axis=1).values # Return a new object
        self.leftindexs = np.arange(len(self.left_id))

        self.right_data = self.right_data.sort_values(by=[self.right_data.columns[0]], axis=0, ascending=True, inplace=False)
        self.right_id = self.right_data.loc[:, self.right_data.columns[0]].values # 2d numpy array
        self.iY = self.right_data.drop(labels=self.right_data.columns[0], axis=1).values
        self.rightindexs = np.arange(len(self.right_id))

        self.leftnodes = len(self.left_id)
        self.rightnodes = len(self.right_id)
        
        self.left_dim = self.iX.shape[1]
        self.right_dim = self.iY.shape[1]

        #FPKM --> log2 transformation --> standardization_sample
        #FPKM --> log2 transformation --> PCA
        #FPKM --> log2 transformation --> quantile normalization
        #FPKM --> standardization_sample
        #scImpute: X_{ij} = log10(X_{ij} + 1.01)

        self.log_transformation()
    
        #self.standardization_feature()
        self.standardization_sample()
        #self.normalization()
        
        if use_pca == True: #data standardization
            self.pca()

        #self.slice_orthologs()
        self.orthologs_dict()
        self.adjacency()

        self.samp_p = samp_p
        self.tao = tao
        self.kernel = kernel
        self.rank_based = rank_based
        self.value_based = value_based
        self.gamma = gamma
        self.simulation_type = simulation_type
        self.use_pca = use_pca
        self.rank_d = rank_d
        self.value_t = value_t
        
        self.estimated_k = None

        #self.update_data()
        self.update_sample()
        self.add_edge()

        self.U = None
     
    """
    def update_data(self):
        ### It will be called 2nd time if we need to reduce the bipartite network into a fully connected component
        
        self.slice_orthologs()
        self.left_id = self.left_data.loc[:, self.left_data.columns[0]].values # 2d numpy array
        self.iX = self.left_data.drop(labels=self.left_data.columns[0], axis=1).values # Return a new object
        self.leftindexs = np.arange(len(self.left_id))
  
        self.right_id = self.right_data.loc[:, self.right_data.columns[0]].values # 2d numpy array
        self.iY = self.right_data.drop(labels=self.right_data.columns[0], axis=1).values 
        self.rightindexs = np.arange(len(self.right_id))
        
        self.leftnodes = len(self.left_id)  
        self.rightnodes = len(self.right_id)
        
        self.left_dim = self.iX.shape[1]
        self.right_dim = self.iY.shape[1]
        
        #FPKM --> log2 transformation --> standardization
        #FPKM --> log2 transformation --> quantile transformation --> PCA
        #FPKM --> standardization
        #FPKM --> standardization_sample
        #scImpute: X_{ij} = log10(X_{ij} + 1.01)

        self.log_transformation()

        #self.standardization_feature()
        self.standardization_sample()
        #self.normalization()
        
        if self.use_pca == True: #data standardization
            self.pca()
        
        self.orthologs_dict()
        self.adjacency() 
        self.update_sample()
    """

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
                self.keep_largest_component() ###self.update_data()
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
                self.keep_largest_component() ###self.update_data()
                self.add_rank_based(rank_d=self.rank_d)
                self.find_connected_components() #re-check: self.D has to be non-singular (fully-connected)
                                
        if self.value_based == True:
            print("embedding the value_based method")
            self.add_value_based(value_t=self.value_t)
            self.find_connected_components()
            if self.N_components > 1:
                self.keep_largest_component() ###self.update_data()
                self.add_value_based(value_t=self.value_t)
                self.find_connected_components() #re-check: self.D has to be non-singular (fully-connected)

           
    def slice_orthologs(self):
        ###I want to keep those expressed genes but no edges connected
        print("make the nameset of orthologs data belongs to the nameset of fly_worm")
        self.orthologs_data = self.orthologs_data[self.orthologs_data.loc[:, 'V4'].isin(list(self.left_data.loc[:, 'gene_id']))]
        self.orthologs_data = self.orthologs_data[self.orthologs_data.loc[:, 'V5'].isin(list(self.right_data.loc[:, 'gene_id']))]
        print("orthologs data size: ", self.orthologs_data.shape[0]) 
      
        
    def orthologs_dict(self):   
        """the fly-worm orthologs_data"""
        pairs = self.orthologs_data.loc[:, ['V4', 'V5']].values
        s = list(map(tuple, pairs))
        self.d = collections.defaultdict(list)
        for k, v in s:
            self.d[k].append(v)
            
                
    #def adjacency(self, add_kernel=False):
    def adjacency(self):
        """Strict bipartite network: no edges within each single side 
        Elements of the adjacency matrix belong to {0, 1}"""
        self.A = np.zeros((self.leftnodes, self.rightnodes), dtype=np.uint16)
        
        for key in self.d.keys():
            for value in self.d[key]:
                #i1 = dm_gene_id.searchsorted(key)
                #i2 = ce_gene_id.searchsorted(value)
                i1 = np.searchsorted(self.left_id, key)
                i2 = np.searchsorted(self.right_id, value)
                
                #if i1 < len(self.left_id) and i2 < len(self.right_id):
                #if (dm_gene_id.iloc[i1] == key).values and (ce_gene_id.iloc[i2] == value).values:
                if self.left_id[i1] == key and self.right_id[i2] == value:  
                    self.A[i1, i2] = 1   
        
        self.average_degree = (2*np.sum(self.A))/(self.leftnodes+self.rightnodes)
        print("average degree before adding kernel: ", str(self.average_degree))
        #The bi_adjacency matrix
        self.W = np.concatenate((np.concatenate((np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16), self.A), axis=1),
                                 np.concatenate((self.A.T, np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)), axis=1)), axis=0)
    
    
    def log_transformation(self):
        """
        Logarithmic transformation. FPKM are typically treated as continuous and modeled with normal distribution after a log transformation
        log2(FPKM + 1)
        log2(fpkm + 0.001)
        log2(FPKM + \epsilon(random noise) \in (0,1))
        """
        self.iX = np.log2(self.iX + 1)
        self.iY = np.log2(self.iY + 1)
    
            
    def standardization_feature(self):
        ###might behave badly if the individual features do not more or less look like standard normally distributed data: 
        ###Gaussian with zero mean and unit variance.
        ###standardize the features/librarys/conditions
        ###Center | Scaled
        #self.iX = scale(self.iX, axis=0, with_mean=True, with_std=False, copy=False) #copy=True
        #self.iY = scale(self.iY, axis=0, with_mean=True, with_std=False, copy=False) #copy=True
        
        self.iX = scale(self.iX, axis=0, with_mean=True, with_std=True, copy=False) #copy=True
        self.iY = scale(self.iY, axis=0, with_mean=True, with_std=True, copy=False) #copy=True
        
    
    def standardization_sample(self):
        ###standardize the samples/observations/genes
        #since: scale(np.zeros(5)) == 0, what would change if we keep those non-expressed genes
        self.iX = scale(self.iX, axis=1, with_mean=True, with_std=True, copy=True) 
        self.iY = scale(self.iY, axis=1, with_mean=True, with_std=True, copy=True)
        
        
    def normalization(self):
        ###scaling individual samples to have unit norm
        self.iX = normalize(self.iX, norm='l2', axis=1, copy=True, return_norm=False)
        self.iY = normalize(self.iY, norm='l2', axis=1, copy=True, return_norm=False)
    
    
    def pca(self):
        """
        dimension reduction on the original covariant matrices
        Principal component plot with quantile normalized log-transformed F/RPKM values, PC2 and PC3.
        attemp a quantile normalization followed by PCA on the log-transformed values
        """
        ##FPKMs. Within sample normalization
        ###feature scale before PCA
        self.standardization_feature()
        pca = PCA(n_components=2)
        #pca = PCA(.95)
        self.iX = pca.fit_transform(self.iX)
        print(pca.explained_variance_ratio_) #check the usage
        self.iY = pca.fit_transform(self.iY)
        print(pca.explained_variance_ratio_)
        #print(pca.n_components_)
        
        
    def spearman_correlation(self, threshold=True): 
        """Need find a way to normalize the matrix"""
        K1, pval1 = spearmanr(self.iX, self.iX, axis=1)
        K1 = K1[:self.leftnodes, :self.leftnodes]
        pval1 = pval1[:self.leftnodes, :self.leftnodes]
        
        K2, pval2 = spearmanr(self.iY, self.iY, axis=1)
        K2 = K2[:self.rightnodes, :self.rightnodes]
        pval2 = pval2[:self.rightnodes, :self.rightnodes]
        
        if threshold == True:
            K1 = squareform(K1, force='tovector', checks=False)
            pval1 = squareform(pval1, force='tovector', checks=False)
            K1[pval1 > 0.05] = 0
            K1 = squareform(K1, force='tomatrix')
            
            K2 = squareform(K2, force='tovector', checks=False)
            pval2 = squareform(pval2, force='tovector', checks=False)
            K2[pval2 > 0.05] = 0
            K2 = squareform(K2, force='tomatrix')
            
        K1 = np.absolute(K1)
        K2 = np.absolute(K2)   
        """Elements of the adjacency matrix in this case are too large! 1000+"""
        """Scale the elements range from zero to one"""
        #lkernel = K1 + self.tao*np.identity(self.leftnodes)
        #rkernel = K2 + self.tao*np.identity(self.rightnodes)
        #self.A = np.dot(np.dot(lkernel, self.A), rkernel)
        self.A = np.dot(np.dot(K1, self.A), K2)
    
        #The bi_adjacency matrix. self.W == self.W.T
        self.W = np.concatenate((np.concatenate((np.zeros((self.leftnodes, self.leftnodes), dtype=np.uint16), self.A), axis=1),
                                 np.concatenate((self.A.T, np.zeros((self.rightnodes, self.rightnodes), dtype=np.uint16)), axis=1)), axis=0)


    def add_rank_based(self, rank_d=5):
        """
        Mutli-layer network: edges within each single sides are allowed
        Genes in one species will be connected if they are co-associated in a co-expression network 
        (Jianhua Ruan, Angela K Dean, Weixiong Zhang. 2010) 1. rank-based; 2. value-based
        
        Pearson’s correlation requires that each dataset be normally distributed;
        p-value of pearson correlation roughly indicates the probability of an uncorrelated system producing 
        datasets that have a Pearson correlation at least as extreme as the one computed from these datasets. 
        The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
        
        rank_d: the smallest value such that all genes form a fully connected component.
        
        the ajacency matrix(self.W)/laplacian matrix(self.L) is symmetric <=> edges are undirected 
        """
        print(".add_rank_based() get called")
        """
        index_x = np.array(list(combinations(np.arange(self.leftnodes), 2)))
        index_y = np.array(list(combinations(np.arange(self.rightnodes), 2)))
        """
        """
        ###RuntimeWarning: invalid value encountered in double_scalars r = r_num / r_den.
        pearsonr_X = np.absolute(np.array([pearsonr(self.iX[i,], self.iX[j,])[0] for (i, j) in index_x]))
        pearsonr_Y = np.absolute(np.array([pearsonr(self.iY[i,], self.iY[j,])[0] for (i, j) in index_y]))
        """
        """Find a faster way (e.g, a function;) to get the Pairwise distances between observations in n-dimensional space
        pearsonr_X = np.absolute(np.array([1-(euclidean(self.iX[i,], self.iX[j,])**2)/(2*self.left_dim) for (i, j) in index_x]))
        pearsonr_Y = np.absolute(np.array([1-(euclidean(self.iY[i,], self.iY[j,])**2)/(2*self.right_dim) for (i, j) in index_y]))
        """
        """
        pearsonr_X = squareform(pearsonr_X, force='tomatrix') #the diagonal elements are zero. np.fill_diagonal(pearsonr_X, 1), no edge connect self
        pearsonr_Y = squareform(pearsonr_Y, force='tomatrix')
        """
        ###np.absolute(x, out=x)
        pearsonr_X = np.absolute(1 - (euclidean_distances(self.iX)**2)/(2*self.left_dim))
        pearsonr_Y = np.absolute(1 - (euclidean_distances(self.iY)**2)/(2*self.right_dim))
        
        np.fill_diagonal(pearsonr_X, 0) #no self-connect edge
        np.fill_diagonal(pearsonr_Y, 0)
        
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
        print(".add_rank_based() get ended")
    
    
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
        print(".add_value_based() get ended")
        

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
        
        self.average_degree = (2*np.sum(self.KA))/(self.leftnodes+self.rightnodes)
        print("average degree after adding kernel: ", str(self.average_degree))
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
        """
        ###considering directly slice the kernel matrix or rank_based correlation matrix (self.W) instead
        ###reconstruct the kernel matrix /others??? consider change in cluster.py???
        big_comp = np.argmax(np.array([np.sum(self.component_list == i) for i in range(self.N_components)])) #.astype(np.int32) #numpy.int64
        
        #You may select rows from a DataFrame using a boolean vector the same length as the DataFrame’s index
        ###data_mask = (self.component_list == big_comp)
        left_data_mask = (self.component_list == big_comp)[ : self.leftnodes]
        right_data_mask = (self.component_list == big_comp)[self.leftnodes : ]
        
        if left_data_mask.any() and right_data_mask.any():
            #self.left_data = self.left_data[left_data_mask]
            #self.right_data = self.right_data[right_data_mask]
            self.left_id = self.left_id[left_data_mask]
            self.right_id = self.right_id[right_data_mask]
            
            # we don't modify self.A in the add_kernel() step, just creat a new instance, self.KA
            ###self.W = self.W[data_mask][:, data_mask]
            
            ###if self.use_pca == True, do we have recalculate the pca.fit ???
            self.iX = self.iX[left_data_mask]
            self.iY = self.iY[right_data_mask]
            self.A = self.A[left_data_mask][:, right_data_mask]
            
            self.leftindexs = np.arange(self.iX.shape[0])
            self.rightindexs = np.arange(self.iY.shape[0])
            self.leftnodes = self.iX.shape[0]
            self.rightnodes = self.iY.shape[0]
            
            #self.update_data()
            self.update_sample()
        
        else:
            print("The largest component only contain one side nodes")


    @staticmethod
    def sub_find_connected_components(W):
        bi_graph = csr_matrix(W) #csr_matrix().toarray()
        N_components, component_list = connected_components(bi_graph, directed=False) #dtype=int32
        #print("Number of connected components after reduction: ", N_components)
        print("Size of each connected components after reduction: ", [np.sum(component_list == i) for i in range(N_components)])
    
    
    @staticmethod
    def sub_keep_largest_component(W, left_length):
        
        bi_graph = csr_matrix(W) #csr_matrix().toarray()
        N_components, component_list = connected_components(bi_graph, directed=False) #dtype=int32
        #self.N_components = N_components
        #self.component_list = component_list
        
        print("Number of connected components: ", N_components)
        print("Size of each connected components: ", [np.sum(component_list == i) for i in range(N_components)])
        
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


    def update_sample(self):
        """this method should be called as long as self.leftindexs/rightindexs has been changed"""

        print(".update_sample() get called")
        self.m = math.floor(self.samp_p*len(self.leftindexs))
        self.n = math.floor(self.samp_p*len(self.rightindexs))
        self.X = self.iX[self.leftindexs]
        self.Y = self.iY[self.rightindexs]  

    
    def eigenmatrix_sym(self, K):
        """l = ceiling(log2K), K is # of true clusters [Inderjit S. Dhillon, 2001]"""
        """Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)"""
        """L_sym is symmetric real matrix, positive semi-definite"""
        """select the first K eigenvectors of the Laplacian matrix"""
        ###inv(sqrtm(D)) <=> np.diag(np.reciprocal(np.sqrt(np.diag(D)))) #test the speed
        
        #D1 = np.diag(np.sum(self.A, axis=1))
        #D2 = np.diag(np.sum(self.A, axis=0))

        #self.D = np.vstack((np.hstack((D1, np.zeros((self.leftnodes, self.rightnodes), dtype=np.uint16))),
                            #np.hstack((np.zeros((self.rightnodes, self.leftnodes), dtype=np.uint16), D2))))
 
        print("start to run .spectral_embedding() to construct the laplacian matrix")
        start_time = time.time()
         
        if (self.W == self.W.T).all():
            #print("W is symmetric")
            pass
        else:
            #print("W is asymmetric")
            self.W = (self.W + self.W.T)/2
                         
        _, self.U = my_spectral_embedding(adjacency=self.W, n_components=K, eigen_solver=None,
                                          random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=False)
                             
        #self.U = normalize(self.U, norm="l2", axis=1, copy=True, return_norm=False)
        print(time.time() - start_time)
        
    
    def eigenmatrix_rw(self, K):
        """Normalized spectral clustering according to Shi and Malik (2000)"""

        print("start to calculate the unnormalized laplacian matrix")
        self.D = np.diag(np.sum(self.W, axis=1))
        #print((np.diag(self.D) == 0).any())
        self.L_unn = self.D - self.W
        
        if (self.L_unn == self.L_unn.T).all():
            print("L_unn is symmetric")
        else:
            print("L_unn is asymmetric") #possible when adopt value_based
            self.L_unn = (self.L_unn + self.L_unn.T)/2
    
        ###Generalized eigenvalue problem
        print("solve the generalized eigen problem by eigh")
        start_time = time.time()
        eigvalues, eigvectors = eigh(self.L_unn, self.D, lower=True, eigvals_only=False, overwrite_a=False,
                                     overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True)
        print(time.time() - start_time)
        print(eigvalues)
        print(eigvectors)
        
        print("solve the generalized eigen problem by eigsh")
        start_time = time.time()
        #l = int(np.ceil(np.log2(K))) + 1
        l = K
        eigvalues, eigvectors = eigsh(A=self.L_unn, k=l, M=self.D, which='SM')
        print(time.time() - start_time)
        print(eigvalues)
        print(eigvectors)
        
        if np.isreal(eigvalues).all() & np.isreal(eigvectors).all():
            ###normalized each row of the eigenvectors [Zahra]
            #self.U = eigvectors[:, np.arange(K)]
            self.U = normalize(eigvectors[:, np.arange(l)], norm="l2", axis=1, copy=True, return_norm=False)
        else:
            print("The generalized eigenvalue problem (L_rw) has complex eigenvalues/eigenvectors!")


    @staticmethod
    def sub_eigenmatrix_sym(K, W):
        """useless if in the self.simulation_sp_tsc() step we apply the .spectral_clustering()"""
        #print("start to calculate the symmetric laplacian matrix with broadcasting")
        #start_time = time.time()
        
        #print("self.W is symmetric: %s" % ((W == W.T).all()))
        #print("the diagonals of self.W are zeros: %s" % (np.sum(np.diag(W))))
        
        D = np.diag(np.sum(W, axis=1))
        #print((np.diag(self.D) == 0).any())
        L_sym = np.identity(W.shape[0], dtype=np.uint16) - (W.T * (1/np.sqrt(np.diag(D)))).T * (1/np.sqrt(np.diag(D)))
        #print(time.time() - start_time)
        
        #print((self.L_sym == self.L_sym.T).all()) #False in the kernel_based case #True in the rank_based case
        if np.isreal(L_sym).all(): #this step can be removed
            
            if (L_sym == L_sym.T).all():
                #print("L_sym is symmetric")
                pass
            else:
                #print("L_sym is asymmetric")
                L_sym = (L_sym + L_sym.T)/2
            
            #print("solve the eigen problem by eigsh")
            #start_time = time.time()
            #l = int(np.ceil(np.log2(K))) + 1
            l = K
            eigvalues, eigvectors = eigsh(A=L_sym, k=l, which='SM')
            #print(time.time() - start_time)
            U = normalize(eigvectors[:, np.arange(l)], norm="l2", axis=1, copy=True, return_norm=False)
            #print(eigvalues)
        else:
            print('The Laplacian matrix L_sym has complex elements!')
        
        return(U)
    
    
    @staticmethod
    def sub_eigenmatrix_rw(K, W):
        """l = ceiling(log2K), K is # of true clusters [Inderjit S. Dhillon, 2001]"""
        """L_rw := inv(D)L_unn = I- inv(D)W, is real symmetric square matrix;
            positive semi-definite and have n non-negative real-valued eigenvalues"""
        """select the first l eigenvectors of the Laplacian matrix"""
        """Normalized spectral clustering [Shi and Malik, 2000]"""
        
        #print("start to calculate the unnormalized laplacian matrix")
        #print("self.W is symmetric: %s" % ((W == W.T).all()))
        #print("the diagonals of self.W are zeros: %s" % (np.sum(np.diag(W))))
        
        D = np.diag(np.sum(W, axis=1))
        #print((np.diag(self.D) == 0).any())
        L_unn = D - W
        
        if (L_unn == L_unn.T).all():
            print("L_unn is symmetric")
        else:
            print("L_unn is asymmetric") #possible when adopt value_based
            L_unn = (L_unn + L_unn.T)/2
        
        ###Generalized eigenvalue problem
        print("solve the generalized eigen problem by eigsh")
        #start_time = time.time()
        #l = int(np.ceil(np.log2(K))) + 1
        l = K
        eigvalues, eigvectors = eigsh(A=L_unn, k=l, M=D, which='SM')
        
        #print(time.time() - start_time)
        #print(eigvalues)
        #print(eigvectors)
        
        if np.isreal(eigvalues).all() & np.isreal(eigvectors).all():
            """normalized each row of the eigenvectors [Zahra]"""
            #self.U = eigvectors[:, np.arange(K)]
            U = normalize(eigvectors[:, np.arange(l)], norm="l2", axis=1, copy=True, return_norm=False)
            #print(eigvalues)
        else:
            print("The generalized eigenvalue problem (L_rw) has complex eigenvalues/eigenvectors!")
        
        return(U)


    def fit(self, target, k_min, alpha, beta, seq_num, iteration, resamp_num, remain_p, k_stop):
        """
        Algorithm 2: Sequential identification of tight and stable clusters

        target The total number of clusters that the user aims to find
        k_min The starting point of k0
        top_can The number of top (size) candidate clusters for a specific k0
        seq_num The number of subsequent k0 that finds the tight cluster
        resamp_num Total number of resampling to obtain comembership matrix
        remain_p Stop searching when the percentage of remaining points <= remain_p
        k_stop Stop decreasing k0 when k0 <= k_stop

        Choose the largest candidate from the tight cluster candidates (>=beta), 
        so that no need to specify q(top_can)
        """
        """Open question: how to specify/identify the number of clusters: K; Try multiple values"""
        
        print(".fit() method for sequentially tsc get called")
        #self.simulation_type = "tsc"
        #print(self.simulation_type)
        print(self.simulation_type)
        
        """
        sequentially identify tight and stable co-clusters:
        self.fit()
           sp_tsc
           tsc
           tscu
        one-step hierarchical clustering to identify tight co-clusters candidate:
        self.fit_hierarchical_clustering()
           sp_tsc_hc
           tsc_hc
           tscu_hc
        """

        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = 30) #l = K
        
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = 30) #l = K
        
        
        #random.seed(0)
        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")
        #print("Total number of points on the left side: " + str(self.leftnodes))
        #print("Total number of points on the right side: " + str(self.rightnodes))

        """
        if self.estimated_k != None:
            k_start = self.estimated_k
        else:
            k_start = k_min
        """
        
        nfound = 0
        found = True 
        k_max = k_min+10
        #k_max = k_min+15
        k0 = k_min
        #k = k0

        tclust = []
        tclust_id = []
            
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

                #found_temp = candidates[seq_num-1][index_m[np.argmax(beta_temp>=beta), seq_num-1]]   
                #found_temp = candidates[seq_num-1][index_m[np.nonzero(beta_temp>=beta)[0][0], seq_num-1]]
                #found_temp_index = np.concatenate((INIT.leftindexs, INIT.rightindexs), axis=0)[found_temp]
                found_temp = candidates[seq_num-1][np.amin(index_m[beta_temp>=beta, seq_num-1])]
                found_temp = list(map(int, found_temp))
                #tclust.append(found_temp)
                sub_tclust_id = []

                for index_i in found_temp:

                    if index_i in self.leftindexs:
                        self.leftindexs = self.leftindexs[self.leftindexs != index_i]
                        found_id = self.left_id[index_i]  
                    else:
                        self.rightindexs = self.rightindexs[self.rightindexs != (index_i - self.leftnodes)]
                        found_id = self.right_id[(index_i - self.leftnodes)]

                    sub_tclust_id.append(found_id)

                tclust.append(found_temp)
                tclust_id.append(sub_tclust_id)
                print("Cluster size: "+ str(len(found_temp)))
                #print("Cluster index: ", found_temp)
                print("Cluster id: ", sub_tclust_id)

                self.update_sample()

                print("Remaining number of points on the left side: " + str(len(self.leftindexs)))
                print("Remaining number of points on the right side: " + str(len(self.rightindexs)))

            else:
                found = False
                k = k + 1
                print("Not found!")
                print("k = " + str(k))

        #end while  

        left_id = (self.left_id).tolist()
        right_id = (self.right_id).tolist()
        output = dict(tclust=tclust, tclust_id = tclust_id, left_id=left_id, right_id=right_id)
        #output = dict(tclust=tclust, tclust_id = tclust_id)
        #print('output: ' + str(output))
        #with open(root + 'results.json', 'w') as f:
            #json.dump(output, f)

        ###separate fly genes with worm genes in a co-cluster
        #tclust_id[i][tclust[i] < len(left_id)]
        #tclust_id[i][tclust[i] > len(left_id)]

        return(output)


    def find_candidates(self, k, alpha_vec, plot_root_dir, thre_min_cluster_left=10, thre_min_cluster_right=10, iteration=5, resamp_num=10):
        
        #D_bar = []
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
            
            # Get process results from the output queue
            #print("Start getting comembership matrixs from the Queue")
            D_bar_sub = 0 #broadcasting
            for p in processes:
                #D_bar_sub += output.get()
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)
            #print("End getting comembership matrixs from the Queue")

            del co_labels
            gc.collect()

            # Exit the completed processes
            #print("Start joining the sub-processes")
            for p in processes:  
                p.join()
            #print("End joining the sub-processes")

            del output
            gc.collect()
            
            D_bar += (D_bar_sub/resamp_num)
            
            del D_bar_sub
            gc.collect()
            
            print("End iteration: ", itr+1)
    
        #D_bar = np.sum(D_bar, axis=0)/iteration
        #D_bar = sum(D_bar)/iteration  ### Membership matrix
        D_bar = D_bar/iteration
        
        #D_membership = deepcopy(D_bar)
        D_bar_index = np.concatenate((self.leftindexs, self.rightindexs+self.leftnodes), axis=0)
        #D_com_index = np.concatenate((self.leftindexs, self.rightindexs+self.leftnodes), axis=0)
        #D_com = D_bar[D_com_index][:, D_com_index] #copy from the original 2d array  
        compressed_distance = squareform((1-D_bar), force='tovector', checks=False)
        print("Start to get the linkage matrix")
        #Z = linkage(compressed_distance, method='complete', optimal_ordering=True) #slow down the speed
        Z = linkage(compressed_distance, method='complete', optimal_ordering=False)
        print("Get the linkage matrix")
        
        name_list = [''.join(p) for p in itertools.product(string.ascii_lowercase, string.ascii_lowercase)] #676
    
        res = dict()
        for alpha in alpha_vec:
            print("Alpha: ", alpha)
            max_d = 1-alpha
            #Clutering result: An array of length n. T[i] is the flat cluster number to which original observation i belongs.
            hclusters = fcluster(Z, t=max_d, criterion='distance')
            #The list of leaf node indexes as it appears in the tree from left to right.
            leave_order = leaves_list(Z)
            #The clustering results ordered(sorted) by the leave_order
            leave_cluster = hclusters[leave_order]
            hclusters_nb = np.amax(hclusters) #max(hclusters)
            
            print("Get the hierarchical clustering result")
            #Store the output tight co-clusters
            res_temp = []
            hmp = []
            hmp_cluster = []
            for hclusters_k_index in range(1, hclusters_nb+1):
                candidate_cluster = D_bar_index[hclusters == hclusters_k_index]
                if (sum(candidate_cluster <= np.amax(self.leftindexs)) >= thre_min_cluster_left) and (sum(candidate_cluster >= self.leftnodes) >= thre_min_cluster_right):
                    res_temp.append(candidate_cluster)
                candidate_heatmap = D_bar_index[leave_order[leave_cluster == hclusters_k_index]]
                if (sum(candidate_heatmap <= np.amax(self.leftindexs)) >= thre_min_cluster_left) and (sum(candidate_heatmap >= self.leftnodes) >= thre_min_cluster_right):
                    hmp.append(leave_order[leave_cluster == hclusters_k_index])
                    hmp_cluster.append(np.repeat(hclusters_k_index, len(candidate_heatmap)))

            ##order = list(reversed(np.argsort(list(map(len, res))))) ###max(res, key=lambda a:len(a))
            #order = np.argsort(list(map(len, res_temp)))[::-1]
    
            ##res = [res[i] for i in order][: top_can]
            #res_temp = [res_temp[i] for i in order]
            res[alpha] = res_temp

            #Construct the sub-consensus matrix
            hmp = np.array(list(itertools.chain(*hmp)))
            hmp_cluster = np.array(list(itertools.chain(*hmp_cluster)))
            hmp_cluster = hmp_cluster.astype(str)
            D_bar_sub = D_bar[hmp[:, np.newaxis], hmp]
            
            #Plot the heatmap of the sub-consensus matrix
            print("Plot the heatmap")
            
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
    

    def get_consensus_matrix(self, k, iteration=3, resamp_num=20):
        
        """basically, it is same as self.find_candidates(), but it only return the consensus matrix M_bar
        """
        
        #D_bar = []
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

            """
            # Get process results from the output queue
            print("Start getting comembership matrixs from the Queue")
            #D_bar_sub = [output.get() for p in processes]
            D_bar_sub = 0 #broadcasting
            for p in processes:
                D_bar_sub += output.get()
            print("End getting comembership matrixs from the Queue")
            """
    
            # Get process results from the output queue
            #print("Start getting comembership matrixs from the Queue")
            #D_bar_sub = [output.get() for p in processes]
            D_bar_sub = 0 #broadcasting
            for p in processes:
                #D_bar_sub += output.get()
                co_labels = output.get()
                D_bar_sub += (co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16)
            #print("End getting comembership matrixs from the Queue")
            
            del co_labels
            gc.collect()
            #print("length of D_bar_sub: " + str(len(D_bar_sub)))
            #print((D_bar_sub[0] == D_bar_sub[i]).all())
                            
            # Exit the completed processes
            #print("Start joining the sub-processes")
            for p in processes:
                p.join()
            #print("End joining the sub-processes")
                                    
            del output
            gc.collect()
                
            #D_bar_1 = sum(D_bar_sub)/resamp_num
            #D_bar_sub = np.sum(D_bar_sub, axis=0, dtype=np.uint16)/resamp_num
            #D_bar_sub = np.sum(D_bar_sub, axis=0)/resamp_num #this step cause MemoryError
            #D_bar_sub = sum(D_bar_sub)/resamp_num  ###calculate the co-membership matrix after finishing each iteration to save the memory
            #D_bar_sub = D_bar_sub/resamp_num

            #D_bar.append(D_bar_sub)
            #D_bar += D_bar_sub

            D_bar += (D_bar_sub/resamp_num)
    
            del D_bar_sub
            gc.collect()
            
            print("End iteration: ", itr+1)
                
        #D_bar = np.sum(D_bar, axis=0)/iteration
        #D_bar = sum(D_bar)/iteration  ### Membership matrix
        D_bar = D_bar/iteration
                    
        return D_bar
                        

    def simulation_tsc(self, random_seed, clusters_nb, output):
        """TSC + Euclidean distance"""
        """
        Associate each example with closest centroid: assign every node to the cluster centroid
        it is most closest to (smallest distance or largest similarity) 
        
        After normalize/standardize the covariates data, use the Euclidean distance (or Monotonic ensemble)/pearson correlation"""
        
        ###random.seed() ###Fail when use multiprocessing
        #np.random.seed()
        #print("Sub process","PID: %s, Process Name: %s, Random seed: %s" % (os.getpid(), mp.current_process().name, random_seed), "start working")
        #np.random.seed(seed=int(time.time())) ###no sense, since the processes are almost lauched at the same time
        #np.random.seed(seed=mp.current_process().name)
        np.random.seed(seed=random_seed)
        
        ###random.sample(population, k)
        lindexs_i = np.sort(np.random.choice(np.arange(len(self.leftindexs)), self.m, replace=False)) ###we could just use len(self.leftindexs)
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

        #print("Sub process","PID: %s, Process Name: %s, randomly choose left points: %s and right points: %s" % (os.getpid(), mp.current_process().name, sum(lmask), sum(rmask)))
        ###clusters_nb < sub_U.shape[0] !

        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_ #numpy.array dtype=int32
        """
        If we use rank-based/rank-based method to connect the edges within species, then is it possible that
        the difficulty to find the co-cluster is because we produce lots of clusters of single species
        in the spectral clustering step(testify in the simulation study)???
        """
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "finish the KMeans")

        left_labels = labels[ : self.m]  #llabels = labels[np.arange(Init.m)]
        right_labels = labels[self.m : ] #rlabels = labels[Init.m : (Init.m + Init.n)]

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

        #spearmanr() applied on two matrices will be time consuming, since it will calculate the correlation inside the single object 
        #However, I found creating the following Spearman Correlation matrix is much more slower than the first way!

        #index_xcentroid = list(product(np.arange(sum(~lmask)), np.arange(len(X_centroid)))) #or len(X_centroid_m)
        #index_ycentroid = list(product(np.arange(sum(~rmask)), np.arange(len(Y_centroid))))

        #list comprehension: for i, j ; for (i, j)  #check the direction of array for the reshape
        #spearmanr_X = np.array([spearmanr(Init.X[~lmask][i], X_centroid_m[j])[0] for (i, j) in index_xcentroid]).reshape(sum(~lmask), -1)
        #spearmanr_Y = np.array([spearmanr(Init.Y[~rmask][i], Y_centroid_m[j])[0] for (i, j) in index_ycentroid]).reshape(sum(~rmask), -1)

        """
        spearmanr_X = spearmanr(Init.X[~lmask].T, X_centroid_m.T)[0][:sum(~lmask), sum(~lmask):]
        spearmanr_Y = spearmanr(Init.Y[~rmask].T, Y_centroid_m.T)[0][:sum(~rmask), sum(~rmask):]

        ###consider the absolute correlations
        spearmanr_X = np.absolute(spearmanr_X)
        spearmanr_Y = np.absolute(spearmanr_Y)

        #find the maximum index along the columns for each row
        X_labels[~lmask] = np.array([X_centroid_names[i] for i in np.argmax(spearmanr_X, axis=1)])
        Y_labels[~rmask] = np.array([Y_centroid_names[i] for i in np.argmax(spearmanr_Y, axis=1)])
        """
        """
        index_xcentroid = list(product(np.arange(sum(~lmask)), np.arange(len(X_centroid)))) #or len(X_centroid_m)
        index_ycentroid = list(product(np.arange(sum(~rmask)), np.arange(len(Y_centroid))))

        #list comprehension: for i, j ; for (i, j)  #check the direction of array for the reshape
        ###Euclidean distance / pearson correlation
        euclidean_X = np.array([euclidean(self.X[~lmask][i], X_centroid_m[j]) for (i, j) in index_xcentroid]).reshape(sum(~lmask), -1)
        euclidean_Y = np.array([euclidean(self.Y[~rmask][i], Y_centroid_m[j]) for (i, j) in index_ycentroid]).reshape(sum(~rmask), -1)
        """

        euclidean_X = euclidean_distances(self.X[~lmask], X_centroid_m)
        euclidean_Y = euclidean_distances(self.Y[~rmask], Y_centroid_m)
        
        #find the minimum index along the columns for each row
        X_labels[~lmask] = np.array([X_centroid_names[i] for i in np.argmin(euclidean_X, axis=1)]) #type(X_centroid_names[0]): numpy.uint16
        Y_labels[~rmask] = np.array([Y_centroid_names[i] for i in np.argmin(euclidean_Y, axis=1)]) #type(Y_centroid_names[0]): numpy.uint16

        co_labels = np.concatenate((X_labels, Y_labels), axis=0)  #len = self.leftindexs + self.rightindexs
        #print(co_labels)
        #print(co_labels.dtype)
        
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "start calculating the co_labels vector.")
        ###MemoryError. datatype(int) itemsize, test on server using python script.
        ###1.break the comembership_matrix into several blocks; or compress comembership_matrix into scipy.sparse.csr_matrix
        ###2.return the co_labels instead of comembership_matrix, and then calculate the matrix in the find_candidates() step
        ###output.put()
        ###print(co_labels)
        ###print(co_labels.dtype) to check
        #comembership_matrix = np.equal.outer(co_labels, co_labels)*1
        #output.put(comembership_matrix)
        
        #output.put((co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16))
        output.put(co_labels)
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "get the co_labels vector.")
        
        del co_labels
        gc.collect()
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "end.")
        

    def simulation_tscu(self, random_seed, clusters_nb, output):
        """TSCU"""
        """
        Assign the left nodes into K centoids by using the original eigenvector_matrix instead of covariates
        Given a pre-specified positive integer K, apply the K-Means clustering algorithm to sub_U
        """
        ###random.seed() ###Fail when use multiprocessing
        #np.random.seed()
        print("Sub process","PID: %s, Process Name: %s, Random seed: %s" % (os.getpid(), mp.current_process().name, random_seed), "start working")
        #np.random.seed(seed=int(time.time())) ###no sense, since the processes are almost lauched at the same time
        #np.random.seed(seed=mp.current_process().name)
        np.random.seed(seed=random_seed)
                                        
        ###random.sample(population, k)
                                        
        ###sys.getsizeof()
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

        print("Sub process","PID: %s, Process Name: %s, randomly choose left points: %s and right points: %s" % 
              (os.getpid(), mp.current_process().name, sum(lmask), sum(rmask)))
        ###clusters_nb < sub_U.shape[0] !!!
        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, 
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_

        ###When I use multiprocessing, one process got stuck at K-Means step and cannot finish it. Try my own KMeans function
        ###K-Means function would be faster???
        print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "finish the KMeans")

        U_remain = np.concatenate((self.U[self.leftindexs][~lmask], 
                                   self.U[(self.leftnodes + self.rightindexs)][~rmask]), axis=0)

        labels_remain = kmeans.predict(U_remain)

        ###Only assign the remaining nodes to the k centroids with eigenvector matrix
        left_labels = labels[ : self.m]  #llabels = labels[np.arange(Init.m)]   
        right_labels = labels[self.m : ] #rlabels = labels[Init.m : (Init.m + Init.n)]

        X_labels = np.empty(len(self.leftindexs), dtype = np.uint16) #Unsigned integer (0 to 65535)
        Y_labels = np.empty(len(self.rightindexs), dtype = np.uint16) #Unsigned integer (0 to 65535)
        X_labels[lmask] = left_labels #X_labels.dtype: dtype('uint16')
        Y_labels[rmask] = right_labels #Y_labels.dtype: dtype('uint16')

        X_labels[~lmask] = labels_remain[ : sum(~lmask)]
        Y_labels[~rmask] = labels_remain[sum(~lmask) : ]

        co_labels = np.concatenate((X_labels, Y_labels), axis=0)  #len = self.leftindexs + self.rightindexs
        #print(co_labels)
        #print(co_labels.dtype)
        
        print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "start calculating the co_labels vector.")
        ###MemoryError. datatype(int) itemsize, test on server using python script.
        ###1.break the comembership_matrix into several blocks; or compress comembership_matrix into scipy.sparse.csr_matrix
        ###2.return the co_labels instead of comembership_matrix, and then calculate the matrix in the find_candidates() step
        ###output.put()
        ###print(co_labels)
        ###print(co_labels.dtype) to check
        #comembership_matrix = np.equal.outer(co_labels, co_labels)*1
        #output.put(comembership_matrix)
        
        #output.put((co_labels[:, np.newaxis] == co_labels[np.newaxis, :]).astype(np.uint16))
        output.put(co_labels)
        print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "get the co_labels vector.")
        
        del co_labels
        gc.collect()
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "end.")
        

    def simulation_sp_tsc(self, random_seed, clusters_nb, output):
        """
        slice the adjacency matrix A (kernel matrix/rank_based correlation matrix/value_based correlation matrix)
        --> unified similarity matrix S_{\tau,\alpha} --> reconstruct the laplacian matrix
        --> eigen-decomposition --> eigenmatrix --> kmeans
        """
        """
        #sub_W = self.W[u_indexs][:, u_indexs]
        ###after sub-sampling, some node may has no edge connected anymore.
        ###therefore, instead of slice self.W, we need to reconstruct the adjacency matrix (with kernel)
        /home/yidan/Dropbox/project_fly_worm/simulation/simdata/synthetic_data.py:682: RuntimeWarning: divide by zero encountered in true_divide
        L_sym = np.identity(W.shape[0], dtype=np.uint16) - (W.T * (1/np.sqrt(np.diag(D)))).T * (1/np.sqrt(np.diag(D)))
        /home/yidan/Dropbox/project_fly_worm/simulation/simdata/synthetic_data.py:682: RuntimeWarning: invalid value encountered in multiply
        L_sym = np.identity(W.shape[0], dtype=np.uint16) - (W.T * (1/np.sqrt(np.diag(D)))).T * (1/np.sqrt(np.diag(D)))
        """
        print(".simulation_sp_tsc() get called")
        ###random.seed() ###Fail when use multiprocessing
        #np.random.seed()
        #np.random.seed(seed=int(time.time())) ###no sense, since the processes are almost lauched at the same time
        #print("Sub process","PID: %s, Process Name: %s, Random seed: %s" % (os.getpid(), mp.current_process().name, random_seed), "start working")
        #np.random.seed(seed=mp.current_process().name) ###wrong datatype
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
                                                                    
        ###need consider the other add edge method: value_based, rank_based
        ###embed kernel --> keep one connected component
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
            ###no embed kernel --> no keep one connected component
            #the no kernel case may cause there are nodes with no edge connected
        
            sub_W = np.concatenate((np.concatenate((np.zeros((sub_A.shape[0], sub_A.shape[0]), dtype=np.uint16), sub_A), axis=1),
                                    np.concatenate((sub_A.T, np.zeros((sub_A.shape[1], sub_A.shape[1]), dtype=np.uint16)), axis=1)), axis=0)
            self.sub_find_connected_components(sub_W)
        
        """
        #sub_U = self.sub_eigenmatrix_rw(clusters_nb, sub_W)
        sub_U = self.sub_eigenmatrix_sym(30, sub_W) # l = K
        #sub_U = self.sub_eigenmatrix_sym(clusters_nb+1, sub_W) #l = log2(clusters_nb) + 1.
        
        #print("Sub process","PID: %s, Process Name: %s, randomly choose left points: %s and right points: %s" % (os.getpid(), mp.current_process().name, sum(lmask), sum(rmask)))
        ###clusters_nb < sub_U.shape[0] !
        
        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_ #numpy.array dtype=int32
        """
        
        """
        _, sub_U = my_spectral_embedding(adjacency=sub_W, n_components=30, eigen_solver=None,
                                         random_state=None, eigen_tol=0.0, norm_laplacian=True, drop_first=False)

        sub_U = normalize(sub_U, norm="l2", axis=1, copy=True, return_norm=False)

        kmeans = KMeans(n_clusters=clusters_nb, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                        precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1,
                        algorithm='auto').fit(sub_U)
        labels = kmeans.labels_ #numpy.array dtype=int32
        """
  
  
        labels = spectral_clustering(affinity=sub_W, n_clusters=clusters_nb, n_components=30, eigen_solver=None,
                                     random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')
        
        
        #print("Sub process","PID: %s, Process Name: %s" %(os.getpid(), mp.current_process().name), "finish the KMeans")
        """
        left_labels = labels[ : self.m]  #llabels = labels[np.arange(Init.m)]
        right_labels = labels[self.m : ] #rlabels = labels[Init.m : (Init.m + Init.n)]
        """
                                                                                                                                                                    
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
                                                                                                                                                                                                                                                                
        ###Check the method which assign all of nodes into K centroids???
        #euclidean_X = euclidean_distances(self.X, X_centroid_m)
        #euclidean_Y = euclidean_distances(self.Y, Y_centroid_m)
                                                                                                                                                                                                                                                                
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


    def fit_hierarchical_clustering(self, k, alpha_vec, plot_root_dir, thre_min_cluster_left, thre_min_cluster_right, iteration, resamp_num):
        ###we can add the heatmap of the comembership matrix to check???
        
        print("hierarchical clustering with the (same kernel embedded) inherited bi-adjacency matrix get called")
        #self.simulation_type = "tsc"
        #self.simulation_type = "sp_tsc"
        print(self.simulation_type)
        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = 30) #l = K
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = 30) #l = K

        #random.seed(0)
        #np.random.seed(seed=int(time.time()))
        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")

        print("k = ", k)
        output = dict()
        result = self.find_candidates(k, alpha_vec, plot_root_dir, thre_min_cluster_left, thre_min_cluster_right,iteration, resamp_num) #list of arrays (tight clusters candidate)
        #result = result.tolist()
        for alpha in alpha_vec:
            result_alpha = result[alpha]
            tclust = []
            tclust_id = []
            for found_temp in result_alpha:
                found_temp = list(map(int, found_temp))
                #tclust.append(found_temp)
                sub_tclust_id = []
                
                for index_i in found_temp:
                    if index_i in self.leftindexs:
                        #self.leftindexs = self.leftindexs[self.leftindexs != index_i]
                        found_id = self.left_id[index_i]
                    else:
                        #self.rightindexs = self.rightindexs[self.rightindexs != (index_i - self.leftnodes)]
                        found_id = self.right_id[(index_i - self.leftnodes)]
                    
                    sub_tclust_id.append(found_id)
                
                tclust.append(found_temp)
                tclust_id.append(sub_tclust_id)
 
            
            #output_alpha = dict(tclust=tclust, tclust_id = tclust_id, left_id=left_id, right_id=right_id)
            output_alpha = dict(tclust=tclust, tclust_id=tclust_id)
            ###separate fly genes with worm genes in a co-cluster
            #tclust_id[i][tclust[i] < len(left_id)]
            #tclust_id[i][tclust[i] > len(left_id)]
            #self.estimated_k = len(tclust)
            #print("Number of co-clusters found by hierarchical clustering (one-step tsc)" + str(self.estimated_k))
            output[alpha] = output_alpha
        
        #Output gene lists which does not vary with alpha
  
        #output['left_id']=(self.left_id).tolist()
        #output['right_id']=(self.right_id).tolist()

        return(output)
            

    def fit_consensus_clustering(self, k, iteration, resamp_num, simulation_type):
        ###we can add the heatmap of the comembership matrix to check???
        #self.simulation_type = "tsc"
        #self.simulation_type = "sp_tsc"
    
        print("consensus clustering with the (same kernel embedded) inherited bi-adjacency matrix get called")
        self.simulation_type = simulation_type
        print(self.simulation_type)
        
        if self.simulation_type == "tsc":
            self.eigenmatrix_sym(K = k) #l = K
        if self.simulation_type == "tscu":
            self.eigenmatrix_sym(K = k) #l = K
        
        #random.seed(0)
        #np.random.seed(seed=int(time.time()))
        print("CPU_count: %s" % (mp.cpu_count()))
        print("Main process","PID: %s, Process Name: %s" % (os.getpid(), mp.current_process().name), "start working")
        
        print("k = ", k)
        ##the methods depend on the self.simulation_type
        consensus_matrix = self.get_consensus_matrix(k, iteration, resamp_num) #list of arrays (tight clusters candidate)
        print("finish get the consensus matrix")
        return(consensus_matrix)
    
                                        
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

