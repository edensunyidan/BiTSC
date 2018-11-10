
# coding: utf-8

# In[ ]:

#import sys
#print('version is', sys.version)
#print('sys.argv is', sys.argv)
#print('sys.argv is', sys.argv[1])

#sys.path
#dir()
#dir(numpy)
#dir(gap_statistic)

#import imp
#imp.reload(gap_statistics)

#plt.savefig("/Users/yidansun/Desktop/PROJECT/promoter_enhancer/figures/prediction_strength.pdf")
#save figure using the absolute or the relative directory?


# In[14]:

import numpy as np
import pandas as pd
import random
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

__Author__ = "Yidan Sun"


# In[20]:

class GapStats:
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    
    Parameters
    ----------
        data: ndarry of shape (n_samples, n_features)
        
        transform: 
        (a) "uniform": generate each reference feature uniformly over the range of the observed values for that feature;
        (b) "pca":     genrate the reference features from a uniform distribution over a box aligned with principle components 
            of the data. In detail, if X is our n by p data matrix, assume that the columns have mean 0 and compute
            the singular value decomposition X = UDV.T. We transform via X' = XV and then draw uniform features Z'
            over the ranges of the columns of X', as in method (a). Finally we back-transform via Z = Z'V.T to give
            reference data Z.
            
        nrefs: number of sample reference datasets to create
        
        maxClusters: Maximum number of clusters to test for
        
        k_hat = smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        
    Returns
    -------
    (gaps, optimalK)
    
    """
    
    def __init__(self, data, transform='pca', nrefs=10, maxClusters=15):
        self.data = data
        self.transform = transform
        self.nrefs = nrefs
        self.maxClusters = maxClusters
    
    @staticmethod 
    def bounding_box(X):
        dimension = X.shape[1]
        xmin, xmax = [], []
        for d in np.arange(dimension):
            #the minimum value for the d_th dimension
            xmin.append(min(X,key=lambda a:a[d])[d])
            #the maximum value for the d_th dimension
            xmax.append(max(X,key=lambda a:a[d])[d])

        return xmin, xmax
    
    def calculate(self):
        
        #Wks = np.zeros((len(range(1, maxClusters)),))
        #Wkbs = np.zeros((len(range(1, maxClusters)),))
        sks = np.zeros((len(range(1, self.maxClusters+1)),))
        gaps = np.zeros((len(range(1, self.maxClusters+1)),))
    
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[], 'sk':[]})
        
        for gap_index, k in enumerate(range(1, self.maxClusters+1)):
            
            # Fit cluster to original data and create dispersion
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', 
                        verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
            km.fit(self.data)
            # Sum of squared distances of samples to their closest cluster center
            origDisp = km.inertia_
            #Wks[gap_index] = np.log(origDisp)

            # Holder for reference dispersion results
            refDisps = np.zeros(self.nrefs)
            
            if self.transform == "uniform":
                bmin, bmax = self.bounding_box(self.data)  
                obs = self.data.shape[0]
                dim = self.data.shape[1] #len(bmin) = len(bmax) = dim
    
                # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                for i in range(self.nrefs):
                    # Create new random reference set
                    randomReference = np.array([np.random.uniform(low=bmin[p], high=bmax[p], size=obs) for p in np.arange(dim)]).T
        
                    # Create new random reference set
                    #randomReference = np.random.random_sample(size=data.shape)
            
                    # Fit to it
                    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', 
                                verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
                    km.fit(randomReference)
                    #km = KMeans(k).fit(randomReference)
            
                    refDisp = km.inertia_
                    refDisps[i] = refDisp
            
            if self.transform == "pca":
                scaler = StandardScaler(with_mean = True, with_std = False)
                X_scale = scaler.fit_transform(self.data)
                mean = scaler.mean_
                u, s, vh = svd(X_scale)
                X_prime = np.dot(X_scale, vh.T)
                
                bmin, bmax = self.bounding_box(X_prime)
                obs = X_prime.shape[0]
                dim = X_prime.shape[1]
                           
                # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                for i in range(self.nrefs):
                    # Create new random reference set
                    Z_prime = np.array([np.random.uniform(low=bmin[p], high=bmax[p], size=obs) for p in np.arange(dim)]).T
                    Z_scale = np.dot(Z_prime, vh)
                    Z = Z_scale + mean #randomReference
                    
                    # Create new random reference set
                    #randomReference = np.random.random_sample(size=data.shape)
            
                    # Fit to it
                    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', 
                                verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
                    km.fit(Z)
                    #km = KMeans(k).fit(randomReference)
            
                    refDisp = km.inertia_
                    refDisps[i] = refDisp
            
            #Wkbs[gap_index] = np.mean(np.log(refDisps))
            
            # Calculate gap statistic
            gap = np.mean(np.log(refDisps)) - np.log(origDisp)
            sk = np.sqrt(1 + 1/self.nrefs) * np.std(np.log(refDisps))
        
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            sks[gap_index] = sk
        
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap, 'sk':sk}, ignore_index=True)
            #For DataFrames which don’t have a meaningful index, you may wish to append them and 
            #ignore the fact that they may have overlapping indexes: use the ignore_index argument.
            #If ignore_index=True, do not use the index labels.

        # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal 
        #return (gaps.argmax() + 1, resultsdf, gaps, sks) 
        return (gaps.argmax() + 1, resultsdf) #optimalK


# In[12]:

def init_board_gauss(N, k):
    
    """called in the "__main__" file"""
    
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X




# In[22]:

if __name__ == "__main__":
    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    X, y = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.0, center_box=(-100.0, 100.0), shuffle=True, random_state=None)
    #X = init_board_gauss(200,3)
    test_object = GapStats(X, transform='pca', nrefs=10, maxClusters=15)
    k, gapdf = test_object.calculate() 
    
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("subdirectory/package/figures/test_data.pdf")
    #plt.show()
    
    plt.figure(2)
    plt.subplot(221)
    #plt.grid(True)
    plt.plot(gapdf.clusterCount, gapdf.gap, "go-", linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='g')
    
    plt.subplot(222)
    plt.plot(gapdf.clusterCount.values[1:], (gapdf.gap.values[:-1] - gapdf.gap.values[1:] + gapdf.sk.values[1:]), "ro")
    
    plt.subplot(223)
    plt.bar(gapdf.clusterCount.values[1:], (gapdf.gap.values[:-1] - gapdf.gap.values[1:] + gapdf.sk.values[1:]))
    
    plt.subplot(224)
    plt.errorbar(x=gapdf.clusterCount, y=gapdf.gap, yerr=gapdf.sk)

    plt.subplots_adjust(top=1.80, bottom=0.01, left=0.10, right=1.70, hspace=0.25, wspace=0.35)
    
    plt.savefig("subdirectory/package/figures/gap_statistics.pdf")
    #plt.show()

