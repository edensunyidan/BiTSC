# BTSC


0. Run python script on linux:
   yidan@compbio1:~/Dropbox$ python3 /home/yidan/Dropbox/main.py network_filepath fly_filepath worm_filepath k_min alpha root_dir_savedata & disown

1. Four output .txt file, path is ./result/data
   1) cluster.txt
      eg:
      
      cluster_size:123\t"fly_gene_id",…"worm_gene_id"\ncluster_size:123\t"fly_gene_id",…"worm_gene_id"\n
   
   2) fly.txt, store all fly gene id

   3) worm.txt, store all worm gene id

   4) consensus_matrix.txt, store consensus matrix (m+n) \times (m+n)

   5) adjacency_matrix.txt, store bi-adjacency matrix (m \times n)

   Note: the order of rows and columns in consensus matrix and bi-adjacency matrix is follower by fly.txt and worm.txt


2. Input arguments followed by main.py (User-specified parameters)
    
   1). Absolute path of linkage file:  /home/yidan/Dropbox/data/network.txt  
   2). Absolute path of enhancer file: /home/yidan/Dropbox/data/fly_data.txt 
   3). Absolute path of promoter file: /home/yidan/Dropbox/data/worm_data.txt
       Note: 3 .csv file will be generated for those 3 file in the corresponding directory

   4). k_min: default = 30. Input number of clusters of the algorithm
       Note: if we increase k_min, we will get less (tight) clusters.

   5). alpha: default to be 0.8. Tuning parameter for tightness
       Note: if we increase alpha, we will get less (tight) clusters.
   
   6). Absolute path of folder for saving results: /home/yidan/Dropbox/result
       Note: this folder couldn’t exist

3. Other parameters

   1). iteration, default=10, number of iterations for which the user want to run, each iteration will launch ‘resamp_num’ parallel processes
   2). resamp_num, default=10, iteration \times resamp_num is total number of subsampling procedures/processes in the algorithm, suggestion is iteration \times resamp_num >= 100
       Note: (1) if iteration is large, the algorithm may raise MemoryError; if iteration is small, the algorithm is slow
             (2) the values of iteration and resamp_num should be fixed across all running.

   


