
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def cdf_area_plot(consensus_list, root_dir):
    """
    Consensus Cumulative Distribution Function (CDF)
    This graphic shows the cumulative distribution functions of the consensus matrix for each k (indicated by colors),
    estimated by a histogram of 100 bins. The figure allows a user to determine at what number of clusters, k, 
    the CDF reaches an approximate maximum, this consensus and cluster confidence is at a maximum at this k.
    
    ###Histogram of elements in the consensus matrix
    
    Delta Area Plot
    This graphic shows the relative change in area under the CDF curve comparing k and k-1. For k=2, there is no k-1,
    so the total area under the curve rather than the relative increase is plotted.
    
    element = consensus_list[0]
       
    element['consensus_matrix']
    element['k_min']
       
    k_min: must be a continuous sequence start from 2!  
    """
    colors_map = cm.rainbow(np.linspace(0, 1, len(consensus_list)))
    #plots = []
    labels = []
    areaK = []
    k_set = []
    
    plt.figure(figsize=(10,10))
    for i, element in enumerate(consensus_list):
        consensus_matrix = element['consensus_matrix']
        k_min = element['k_min']
        k_set.append(k_min)
        dim = consensus_matrix.shape[0] 
        #return the upper triangular part of the consensus matrix
        consensus_values = consensus_matrix[np.triu_indices(dim, k=1)]
        
        #empirical CDF distribution. default number of breaks is 100    
        hist, bin_edges = np.histogram(consensus_values, bins=np.linspace(0, 1, num=101, endpoint=True), density=False)
        counts = np.cumsum(hist)/sum(hist)
        
        mid_points = (bin_edges[1::1] + bin_edges[0:-1:1])/2
        
        #p0, = plt.plot(mid_points, counts, color=colors_map[i], linestyle='solid', linewidth=1.5, alpha=0.5)
        #plots.append(p0)
        plt.plot(mid_points, counts, color=colors_map[i], linestyle='solid', linewidth=1.5, alpha=0.5)
        
        #"k =%5d"%(3) ###python format
        labels.append(r"$k$ = " + str(k_min))
      
        #calculate area under CDF curve, by histogram method.
        thisArea = 0
        for bi in np.arange(len(bin_edges)-2):
            thisArea += counts[bi]*(bin_edges[bi+1]-bin_edges[bi]) #increment by height by width
        areaK.append(thisArea)
        
    plt.legend(labels=labels, loc=4, prop={'size':15}) #handles=plots, 
    plt.ylim(-0.01, 1.01) 
    plt.xlim(-0.01, 1.01)
    #root_dir = '/Users/yidansun/Dropbox (LASSO)/project_fly_worm/spectral_laplacian_result/figure'
    
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
    plt.xlabel('consensus index value', fontsize=15)
    plt.ylabel('CDF', fontsize=15)
    plt.title("consensus CDF", fontsize=15)
    #plt.axis([0, 150, 0, 0.6])
    plt.savefig(root_dir + '/consensus_CDF.pdf')
    #plt.show()
    
    
    #Delta Area Plot: plot area under CDF change.
    deltaK = []
    deltaK.append(areaK[0]) #initial auc at k_min = 2
    for i in range(1, len(areaK)):
        #proportional increase relative to prior K.
        deltaK.append( (areaK[i] - areaK[i-1])/areaK[i-1] )
    plt.figure(figsize=(10,10))
    plt.plot(1+np.arange(len(areaK)), deltaK, color='#000000', marker='o', linestyle='solid', markerfacecolor ='none', linewidth=1.5, markeredgewidth=1.5, markersize=7, alpha=1)
    
    
    plt.xticks(1+np.arange(len(k_set)), k_set, size=15)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=15)
    plt.xlabel('number of clusters k', fontsize=15)
    plt.ylabel(r'$\Delta(k)$', fontsize=15)
    plt.title("Delta area", fontsize=15)
    #plt.axis([0, 150, 0, 0.6])
    plt.savefig(root_dir + '/Delta_area.pdf')
    #plt.show()
    


# In[8]:

if __name__ == "__main__":
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.cluster import k_means
    from sklearn.metrics.pairwise import euclidean_distances
    #import numpy as np
    from scipy.linalg import eigh
    import math
    
    
    ### how to import eigh_AtoL from main_eigen_plot.py
    def eigh_AtoL(W):
        """
        W: the bi-adjacency matrix
        output: get the full eigenvalues of normalized lapalcian """
    
        D = np.diag(np.sum(W, axis=1))
        #print((np.diag(self.D) == 0).any())
        L_sym = np.identity(W.shape[0], dtype=np.uint16) - (W.T * (1/np.sqrt(np.diag(D)))).T * (1/np.sqrt(np.diag(D)))
        #print((self.L_sym == self.L_sym.T).all()) #False in the kernel_based case #True in the rank_based case
        if (L_sym == L_sym.T).all():
            #print("L_sym is symmetric") 
            pass
        else:
            #print("L_sym is asymmetric") 
            L_sym = (L_sym + L_sym.T)/2
            
        eigvals = eigh(a=L_sym, b=None, lower=True, eigvals_only=False, overwrite_a=False, 
                       overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True)[0]
    
        return(eigvals)

    
    """ Compare two sub_sampling methods on simulated data: sub_sampling in Consensus Clustering vs. sub_sampling in TSC
        Both method apply K-Means clustering; 
        
        For the sub_sampling through Consensus Clustering, the expection of the largest elements should be 0.8 """
    
    ### cannot be True at the same time 
    cc_sampling = True
    tsc_sampling = False
    
    X, y = make_blobs(n_samples=500, n_features=2, centers=5, cluster_std=1.0, center_box=(-20.0, 20.0), shuffle=True, random_state=None)
    #X = init_board_gauss(200,3)
    k_set = [2,3,4,5,6,7,8,9,10]
    #k_set = [2,3,4]
    total_index = np.arange(X.shape[0])
    consensus_list = []
    resamp_num = 100
    samp_p = 0.8 
    tolerance = 0.1
    
    for clusters_nb in k_set:
        
        consensus_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
        
        for i in np.arange(resamp_num):
        
            samp_index = np.sort(np.random.choice(np.arange(X.shape[0]), math.floor(samp_p*X.shape[0]), replace=False)) 
            samp_label = k_means(X[samp_index,], clusters_nb, init='k-means++', precompute_distances='auto', 
                                 n_init=10, max_iter=300, verbose=False, tol=0.0001, random_state=None,
                                 copy_x=True, n_jobs=1, algorithm='auto', return_n_iter=False)[1]
            
            if cc_sampling == True:
                
                connectivity_matrix = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
                for k_idx in np.arange(clusters_nb):
                    #samp_mask = np.zeros(X.shape[0], dtype=bool)
                    #samp_mask[samp_index[samp_label == k_idx]] = True
                    sub_index = samp_index[samp_label == k_idx]
                    connectivity_matrix[sub_index[:,np.newaxis], sub_index] = 1
                    
                consensus_matrix += connectivity_matrix
            
            
            if tsc_sampling == True:
                
                samp_mask = np.zeros(X.shape[0], dtype=bool)
                samp_mask[samp_index] = True
                labels = np.empty(X.shape[0], dtype = np.uint16)  
                labels[samp_mask] = samp_label
                
                X1 = X[samp_index]
                X_centroid = []
                for i in np.arange(clusters_nb, dtype=np.uint16): 
                    if sum(samp_labels == i) > 0:  #(left_labels == i).any()
                        x_centroid = np.sum(X1[samp_labels == i], axis=0) / sum(samp_labels == i)  #sum along the column
                        X_centroid.append((i, x_centroid))
                
                X_centroid_names, X_centroid_m = zip(*X_centroid)    #X_centroid_m, X_centroid_names are tuples  
                #X_centroid_names = np.array(X_centroid_names) 
                X_centroid_m = np.array(X_centroid_m)  #.shape = (variables, observations) #each column represents a observation, with variables in the rows
                
                euclidean_X = euclidean_distances(X[~samp_mask], X_centroid_m)
                labels[~samp_mask] = np.array([X_centroid_names[i] for i in np.argmin(euclidean_X, axis=1)]) #type(X_centroid_names[0]): numpy.uint16
            
                connectivity_matrix = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(np.uint16)
                consensus_matrix += connectivity_matrix
            
        
        consensus_matrix_ave = consensus_matrix/resamp_num    
        consensus_list.append(dict(consensus_matrix=consensus_matrix_ave, k_min=clusters_nb, icc=consensus_matrix))
        
    ################### plot the cc_matrix CDF curve and CDF_Area ###################
    root_dir = "/home/yidan/Dropbox/project_fly_worm/consensus_result/blob_data"
    cdf_area_plot(consensus_list, root_dir)
    
    ################### compare the cc_matrix eigvalues and icc_matrix eigvalues distribution ##############
    colors_map = cm.rainbow(np.linspace(0, 1, len(consensus_list)))
    #plots = []
    labels = []
    
    icc = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    plt.figure(figsize=(10,10))
    
    for i, element in enumerate(consensus_list):
        consensus_matrix = element['consensus_matrix']
        icc_matrix = element['icc']
        k_min = element['k_min']
        
        icc += icc_matrix
        cc_eigvals = eigh_AtoL(consensus_matrix)
        print(np.max(cc_eigvals))
        plt.scatter(np.arange(50), cc_eigvals[:50], s=50, marker='o', facecolors=colors_map[i],  edgecolors=colors_map[i], linewidths=1.5, alpha=1)
        
        #"k =%5d"%(3) ###python format
        labels.append(r"$k$ = " + str(k_min))
        

    plt.legend(labels=labels, loc=4, prop={'size':15}) #handles=plots, 
    #plt.ylim(-0.01, 1.01) 
    #plt.xlim(-0.01, 1.01)
    
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('index', fontsize=15)
    plt.ylabel('eigenvalue', fontsize=15)
    plt.title("consensus matrix eigenvalue distribution")
    #plt.axis([0, 150, 0, 0.6])
    plt.savefig(root_dir + '/cc_eigval.pdf')
    #plt.show()
    
    icc_eigvals = eigh_AtoL(icc_matrix)
    print(np.max(icc_eigvals))
    
    plt.figure(figsize=(10,10))
    plt.scatter(np.arange(50), icc_eigvals[:50], s=50, marker='o', facecolors='#000000',  edgecolors='#000000', linewidths=1.5, alpha=1)
    
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('index', fontsize=15)
    plt.ylabel('eigvalue', fontsize=15)
    plt.title("icc matrix eigenvalue distribution")
    #plt.axis([0, 150, 0, 0.6])
    plt.savefig(root_dir + '/icc_eigval.pdf')
    
    




