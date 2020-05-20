import argparse
import sys
import os

from scipy.special import comb
import numpy as np
from biptightkmeans.cluster import BitKmeans

import gc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parse the command-line arguments as the input for BiTSC.')
    #group = parser.add_mutually_exclusive_group()
    
    parser.add_argument('--covariate', nargs='*', required=True, type=str)
    parser.add_argument('--edge', nargs='*', required=True, type=str)
    parser.add_argument('--k', nargs='*', type=int, default=[5, 50], dest='k_vec')

    
    parser.add_argument('--kernel', default='rbf', type=str, help="Metric used when calculating kernel. Valid values for metric are: {'additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'}")
    parser.add_argument('--tau', nargs='*', type=float, default=None, action='append', dest='tao_vec') 
    parser.add_argument('--rho', type=float, default=0.8, dest='rho')
    parser.add_argument('--ncore', type=int, default=10, dest='num_core')
    parser.add_argument('--niter', type=int, default=100, dest='num_iteration')
    

    parser.add_argument('--alpha', nargs='*', type=float, default=[0.90, 0.95, 1.00], dest='alpha_vec')
    parser.add_argument('--heatmap', default=False)
    parser.add_argument('--threshold', type=int, default=10, dest='thre_num_node')
    parser.add_argument('--output_dir', nargs=1, type=str, default='./', dest='root_dir_savedata')
    
    parser.print_help()
    args = parser.parse_args()


    if len(args.edge) != comb(len(args.covariate), 2):
        print("edge does not have the correct length: ", comb(len(args.covariate), 2))
        sys.exit()

    if args.tao_vec is None:
        tao_vec = len(args.edge)*[[1,1]]
    elif len(args.tao_vec) == comb(len(args.covariate), 2)*2:
        tao_vec = list(zip(tao_vec[::2], tao_vec[1::2]))
    else:
        print("tao does not have the correct length: ", comb(len(args.covariate), 2)*2)
        sys.exit()


    if len(args.k_vec) not in [1,2]:
        print("k does not have the correct length")
        sys.exit()
       
    
    bitkmeans = BitKmeans(covariate_data_path=args.covariate, edge_data_path=args.edge, 
        kernel_type=args.kernel, samp_p=args.rho, tao=tao_vec)
    
    try:
        os.mkdir(args.root_dir_savedata)
    except FileExistsError:
        pass

    output_hc = bitkmeans.fit_hierarchical_clustering(k_vec=args.k_vec, alpha_vec=args.alpha_vec, 
        iteration=int(args.num_iteration/args.num_core), resamp_num=args.num_core, thre_num_node=args.thre_num_node, 
        plot_root_dir=args.root_dir_savedata, heatmap=args.heatmap)

    for alpha in args.alpha_vec:
        with open(args.root_dir_savedata+ "/result_alpha_"+str(alpha)+".txt", 'w') as file_object:
            for sub_tclust_id in output_hc[alpha]:
                file_object.write("cluster_size:"+str(len(sub_tclust_id))+"\t")
                file_object.write(",".join(sub_tclust_id)+"\n")
    
    del output_hc
    gc.collect()

