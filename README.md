# Bipartite Tight Spectral Clustering (BiTSC) Algorithm for Identifying Conserved Gene Co-clusters in Two Species
Yidan Eden Sun, Heather J. Zhou and Jingyi Jessica Li

## Requirements
* Python 3.6.8
* numpy
* scipy
* pandas
* json
* sklearn
* multiprocessing
* scanpy
* anndata
* Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-157-generic x86_64)

## Implementation details

### How to Run:
1. Download the package from github, which is named by ```BiTSC-master```
2. In Linux terminal, navigate to the folder ```BiTSC-master```, for example:
```shell
$ cd Downloads/BiTSC-master
```
3. In current directory, run:
```console
$ python3 BiTSC.py ortho_dir one_dir two_dir Kcluster out_dir ncores niters subprop one_thre two_thre alphas & disown
```
### Input data files and parameters:

1. ```ortho_dir```: path to orthology data .csv file
2. ```one_dir```: path to node covariate data .csv file on side 1
3. ```two_dir```: path to node covariate data .csv file on side 2
4. ```K_0_min```: lower bound of K_0's
5. ```K_0_max```: upper bound of K_0's
6. ```ncores```: number of cores used in parallel computation
7. ```niters```: number of iterations to run 
8. ```subprop```: subsampling proportion in each iteration 
9. ```one_thre```: minimum number of side 1 nodes in each output co-cluster 
10. ```two_thre```: minimum number of side 2 nodes in each output co-cluster
11. ```out_dir```: path to output folder directory
12. ```alphas```: a series of tightness tuning parameters

### Output files:
1. ```cluster.txt```, clustering results

2. ```Alpha=a_value.pdf```, heatmaps of sub-consensus matrices corresponding to ```alphas```

### An example on the fly-worm data:
In the directory of ```BiTSC-master```, run:
```console
$ python3 BiTSC.py ./data/orthologs_data_uniq.csv ./data/dm_timecourse_FPKMs.csv ./data/ce_timecourse_FPKMs.csv 10 50 10 100 0.8 10 10 ./data/result 0.90 0.95 1.00
```
