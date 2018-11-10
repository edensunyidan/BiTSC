
# coding: utf-8

# In[57]:

#import sys
#print('version is', sys.version)
#print('sys.argv is', sys.argv)
#print('sys.argv is', sys.argv[1])
#print(sys.getrecursionlimit())

#sys.path
#dir()
#dir(numpy)
#dir(prediction_strength)

#import imp
#imp.reload(prediction_strength)

#globals()
#locals()

#save figure using the absolute or the relative directory
#plt.savefig("/Users/yidansun/Desktop/PROJECT/promoter_enhancer/figures/prediction_strength.pdf")


# In[1]:

import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from itertools import combinations

__Author__ = "Yidan Sun"


# In[9]:

class PredStr:
    """
    Calculates KMeans optimal K using Prediction Strength from Tibshirani, Walther, Botstein and Brown
    
    Parameters
    ----------
        data: ndarry of shape (n_samples, n_features)
        
        maxclusters: Maximum number of clusters to test for
        
        kfold: k-fold cross validation to estimate the prediction strength
        
    Returns
    -------
      (prediction strength, optimal K)
      
    """
    #def __call__(self):
    def __init__(self, data, maxk=100, kfold=5):
        self.data = data
        self.maxk = maxk
        self.kfold = kfold
        #self.cv_index = KFold(n_splits=self.kfold, shuffle=True, random_state=None).split(np.arange(self.data.shape[0]))
        self.predstr_mat = np.zeros((self.kfold, self.maxk))
        self.ps = None
        self.se = None
    
    def calculate(self):
        """
        Cluster the test data into k clusters
        Cluster the traning data into k clusters
        For each test cluster, we compute the proportion of observation pairs in that cluster
        that are also assigned to the same cluster by the training set centroids
        The prediction strength ps(k) is the (cv-ave) minimum of this quantity over the k test clusters.
        
        we select the number of clusters to be the largest k such that the ps(k) + se(k) >= .80, 
        where se(k) is the standard error of the prediction strength over the k cross-validation folds
        """
        self.predstr_mat[:,0] = 1 #ps(1) = 1
        cv_index = KFold(n_splits=self.kfold, shuffle=True, random_state=None).split(np.arange(self.data.shape[0]))
        
        for (fold_index, (train_index, test_index)) in enumerate(cv_index):
            train_set, test_set = self.data[train_index], self.data[test_index]
            for k in range(2, self.maxk + 1):
                test_km = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.0001).fit(test_set)
                train_km = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.0001).fit(train_set)
                self.predstr_mat[fold_index, k - 1] = self.find_minpsprop(test_set, test_km.labels_, train_km.cluster_centers_, k)
                
        self.ps = np.mean(self.predstr_mat, axis=0)
        self.se = np.std(self.predstr_mat, axis=0)
        #self.ps + self.se
    
    
    def find_minpsprop(self, test_set, test_labels, train_centers, k):
        predstr_ver = [self.calculate_psprop(test_set, test_labels, train_centers, j) for j in range(k)] #k>=2
        return min(predstr_ver)
    

    def calculate_psprop(self, test_set, test_labels, train_centers, j):
        """
        For a candidate number of clusters k. Let A_k1, A_k2, ...A_kk be the indices of the test observations 
        in test clusters 1,2,...k.
        Let n_k1, n_k2,...n_kk be the number of observations in these clusters.
        In python code, j: 0, 1, 2,...k-1
        n_k0, n_k1, n_k2, ... n_k(k-1)
        """ 
        n_kj = test_labels.tolist().count(j)
        if n_kj <= 1:
            return float('inf')
        else:
            count = 0.
            for x, y in combinations(range(len(test_labels)), 2):
                if test_labels[x] == test_labels[y] == j:                       
                    if (self.closest_center(test_set[x], train_centers) == self.closest_center(test_set[y], train_centers)):
                        count += 1
            # Return the proportion of pairs that stayed in the same cluster.
            prop = count / (n_kj * (n_kj - 1) / 2.)
            return prop

    @staticmethod                
    def closest_center(point, centroids):
        center_idx = min([(i[0], np.linalg.norm(point-i[1])) for i in enumerate(centroids)], key=lambda t:t[1])[0]  
        return(center_idx)


# In[30]:

if __name__ == "__main__":
    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    X, y = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.0, center_box=(-100.0, 100.0), shuffle=True, random_state=None)
    test_object = PredStr(X, maxk=10, kfold=5)
    test_object.calculate()
    
    plt.scatter(X[:,0], X[:,1])
    plt.savefig("subdirectory/package/figures/test_data.pdf")
    #plt.show()
    
    plt.errorbar(np.arange(test_object.maxk)+1, y=test_object.ps, yerr=test_object.se)
    plt.savefig("subdirectory/package/figures/test_data_prediction_strength.pdf")
    #plt.show()

