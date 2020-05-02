import numpy as np
from scipy.special import comb
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score 


"""
References
----------
.. [Hubert1985] L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985
    http://link.springer.com/article/10.1007%2FBF01908075

.. [wk] https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

.. [George C. Tseng, 2006] Evaluation and comparison of gene clustering methods in microarray analysis

.. from sklearn.metrics import adjusted_rand_score

.. adjustedRandIndex {mclust}

Formula
----------
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

Rand (Weighted Rand index): \lambda*Rand_1 + (1-\lambda)*Rand_2

Rand_1: View the sets of scattered genes as a regular cluster #adjusted_rand_score() 

        treat scattered genes with equal importance as the clustered genes in concordance evaluation and 
        results in bias against methods without scattered genes especially when n_{(R+1).} is large
        
        
Rand_2: Completely ignore the sets of scattered genes #adjusted_rand_score_nonoise()

        only based on intersection of clustered genes of the two partitions, 
        which is baised against methods with a scattered gene set
        
\lambda: (n_{(R+1).} + n_{(C+1).} - n_{(R+1)(C+1)})/N

Note: Rand_1 and Rand_2 and thus Rand take maximum value 1 when P_R(X, C_R) and P_C(X, C_C) 
      are perfectly identical and have expected value 0 when P_C(X, C_C) is random partition
      
Parameters
----------

the largest index indicate noise points 
V_1, V_2, ..., V_C, V_{C+1}(V_noise)
U_1, U_2, ..., U_C, U_{R+1}(U_noise)

Contingency table
+--------+--------------------------------+------------++-----------+
|        | V_1       V_2   ...     V_C    |   V_{C+1}  ||           |   
+--------+--------------------------------+------------++-----------+
| U_1    |n_l1      n_12   ...    n_1C    |  n_1(C+1)  ||  n_1.     |   
| U_2    |n_21      n_22   ...    n_2C    |  n_1(C+1)  ||  n_2.     |                  
| .      |                                |            ||           |
| .      |                                |            ||           |
| .      |                                |            ||           |      
| U_R    |n_R1      n_R2   ...    n_RC    |  n_R(C+1)  ||  n_R.     |
+--------+--------------------------------+------------++-----------|
| U_{R+1}|n_(R+1)1  n_(R+1)...    n_(R+1)C|n_(R+1)(C+1)||  n_(R+1). |                        
+========+================================+============++-----------+
|        |n_.1      n_.2   ...    n_.C    |  n_.(C+1)  ||  n_..(=n) |
+--------+--------------------------------+------------++------------+


input type: labels = dict(indexs=[], noise=boolean)

labels_true[indexs] : int array, shape = [n_samples]  
    Ground truth class labels to be used as a reference
    
labels_pred[indexs] : int array, shape = [n_samples]
    Cluster labels to evaluate
    
labels_true[noise] : boolean, indicate whether labels_true_index has scattered genes 

labels_pred[noise] : boolean, indicate whether labels_pred_index generates scattered genes 

Returns
-------
ari : float
   Similarity score between -1.0 and 1.0. Random labelings have an ARI
   close to 0.0. 1.0 stands for perfect match.

"""


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    #print("check_clustering() get called")
    labels_true = np.asarray(labels_true) #dtype=np.uint16
    labels_pred = np.asarray(labels_pred) #dtype=np.uint16

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


def adjusted_rand_score_nonoise(labels_true, labels_pred):
    """
    Rand index adjusted for chance without noise
    At least one of the partitions generate scattered genes sets (U_{R+1} or V_{C+1} are not empty) 
    the indexs of rows and columns in the contigency table are sorted
    """
    #print("adjusted_rand_score_nonoise() get called")
    indexs_true, indexs_pred = check_clusterings(labels_true['indexs'], labels_pred['indexs'])
    
    n_samples = indexs_true.shape[0]
    n_clusters_true = np.unique(indexs_true).shape[0] #the largest index indicate noise points, if noise == True
    n_clusters_pred = np.unique(indexs_pred).shape[0] #the largest index indicate noise points, if noise == True
    contingency = contingency_matrix(indexs_true, indexs_pred, eps=None, sparse=False)
    
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    #if (n_clusters_true == n_clusters_pred == 1 or #test the following code
        #n_clusters_true == n_clusters_pred == 0 or  
        #n_clusters_true == n_clusters_pred == n_samples): #need check this condition!
        #return 1.0
    if (n_clusters_true == 0) | (n_clusters_pred == 0):
        print("no cluster(s)!")
    

    if (labels_true['noise'] == False) & (labels_pred['noise'] == False):
        #print("At least one of the partitions generate scattered genes sets, \n got both |U_{R+1}| = %d and |V_{C+1}| = %d" % (0, 0), "reduce to the original ARI")
        # Compute the ARI using the contingency data
        #contingency = contingency_matrix(indexs_true, indexs_pred, eps=None, sparse=False)
        sum_comb_r = sum(comb(n_r, 2) for n_r  in contingency.sum(axis=1)) #sum along the row #np.ravel(contingency.sum(axis=1)
        sum_comb_c = sum(comb(n_c, 2) for n_c  in contingency.sum(axis=0))
        #sum_comb  = sum(comb(n_ij,2) for n_ij in contingency) #comb() needs be applied element-by-element
        sum_comb   = sum(comb(n_ij,2) for n_ij in np.ravel(contingency))
        
        lambda_ = 0
        prod_comb = (sum_comb_r * sum_comb_c) / comb(n_samples, 2)
        mean_comb = (sum_comb_r + sum_comb_c) / 2.
        
        return lambda_, (sum_comb - prod_comb) / (mean_comb - prod_comb)

    
    if (labels_true['noise'] == True) & (labels_pred['noise'] == False):
        # Compute the ARI using the contingency data
        #contingency = contingency_matrix(indexs_true, indexs_pred, eps=None, sparse=False)
        sum_comb_r = sum(comb(n_r, 2) for n_r  in contingency[:-1, :].sum(axis=1)) #sum along the row #np.ravel(contingency.sum(axis=1)
        sum_comb_c = sum(comb(n_c, 2) for n_c  in contingency[:-1, :].sum(axis=0)) #sum along the column
        sum_comb   = sum(comb(n_ij,2) for n_ij in np.ravel(contingency[:-1, :]))
        
        lambda_ = sum(contingency[-1, :]) / n_samples
        n_samples_tilde = n_samples - sum(contingency[-1, :])
        prod_comb = (sum_comb_r * sum_comb_c) / comb(n_samples_tilde, 2)
        mean_comb = (sum_comb_r + sum_comb_c) / 2.
        
        return lambda_, (sum_comb - prod_comb) / (mean_comb - prod_comb)
    

    if (labels_true['noise'] == False) & (labels_pred['noise'] == True):
        # Compute the ARI using the contingency data
        #contingency = contingency_matrix(indexs_true, indexs_pred, eps=None, sparse=False)
        sum_comb_r = sum(comb(n_r, 2) for n_r  in contingency[:, :-1].sum(axis=1))
        sum_comb_c = sum(comb(n_c, 2) for n_c  in contingency[:, :-1].sum(axis=0))
        sum_comb   = sum(comb(n_ij,2) for n_ij in np.ravel(contingency[:, :-1]))
        
        lambda_ = sum(contingency[:, -1]) / n_samples
        n_samples_tilde = n_samples - sum(contingency[:, -1])
        prod_comb = (sum_comb_r * sum_comb_c) / comb(n_samples_tilde, 2)
        mean_comb = (sum_comb_r + sum_comb_c) / 2.
        
        return lambda_, (sum_comb - prod_comb) / (mean_comb - prod_comb)
    

    if (labels_true['noise'] == True) & (labels_pred['noise'] == True):
        # Compute the ARI using the contingency data
        #contingency = contingency_matrix(indexs_true, indexs_pred, eps=None, sparse=False)
        sum_comb_r = sum(comb(n_r, 2) for n_r  in contingency[:-1, :-1].sum(axis=1))
        sum_comb_c = sum(comb(n_c, 2) for n_c  in contingency[:-1, :-1].sum(axis=0))
        sum_comb   = sum(comb(n_ij,2) for n_ij in np.ravel(contingency[:-1, :-1])) 
        
        lambda_ = (sum(contingency[-1, :]) + sum(contingency[:-1, -1])) / n_samples
        n_samples_tilde = n_samples - sum(contingency[-1, :]) - sum(contingency[:-1, -1])
        prod_comb = (sum_comb_r * sum_comb_c) / comb(n_samples_tilde, 2)
        mean_comb = (sum_comb_r + sum_comb_c) / 2.
        
        return lambda_, (sum_comb - prod_comb) / (mean_comb - prod_comb)


def weighted_rand_score(labels_true, labels_pred):
    """
    Rand*, weighted average of the two measures
    when V_{C+1}, U_{R+1} are empty, Rand*, Rand*_1, Rand*_2 all reduce to the original modified Rand
    """
    lambda_, rand_two = adjusted_rand_score_nonoise(labels_true, labels_pred)
    rand_one = adjusted_rand_score(labels_true['indexs'], labels_pred['indexs'])
    
    return (lambda_*rand_one + (1-lambda_)*rand_two)


