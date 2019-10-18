# Bipartite Tight Spectral Clustering (BiTSC) Algorithm for Identifying Conserved GeneCo-clusters in Two Species
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
* Linux

## Implementation details

### How to Run
1. Download the package from github, which is named by ```BiTSC-master```
2. In Linux terminal, navigate to the folder ```BiTSC-master```, for example:
```shell
$ cd Downloads/BiTSC-master
```
3. In current directory, run:
```console
$ python3 BiTSC.py ortho_dir one_dir two_dir Kcluster alpha out_dir ncores niters subprop & disown
```
### Documentations of input data files and parameters:

1. **ortho_dir**: full path to orthology data .csv file
2. **one_dir**: full path to node covariate matrix .csv file on side 1
3. **two_dir**: full path to node covariate matrix .csv file on side 2
4. **Kcluster**: input number of co-clusters
5. **alpha**: tuning parameter for tightness
6. **out_dir**: full path to output folder directory
7. **ncores**: number of cores used in parallel computation
8. **niters**: number of iterations to run 
9. **subprop**: subsampling proportion in each iteration 

### An example on the fly-worm data
```console
$ python3 BiTSC.py /home/yidan/Downloads/BiTSC-master/data/orthologs_data_uniq.csv /home/yidan/Downloads/BiTSC-master/data/dm_timecourse_FPKMs.csv /home/yidan/Downloads/BiTSC-master/data/ce_timecourse_FPKMs.csv 30 0.8 /home/yidan/Downloads/BiTSC-master/data/result
```

### Documentations of output text files, which are all saved in the folder named by root_dir_savedata:
1. **cluster.txt**, file for storing the clustering results, for example:
```
cluster_size:37 FBgn0004244,FBgn0004573,FBgn0010329,FBgn0010414,FBgn0011582,FBgn0013334,FBgn0015519,FBgn0016975,FBgn0029846,FBgn0032151,FBgn0033876,FBgn0033958,FBgn0034136,FBgn0035170,FBgn0035364,FBgn0035756,FBgn0036934,FBgn0037698,FBgn003829,FBgn0038880,FBgn0039536,FBgn0051191,FBgn0052683,FBgn0053516,FBgn0053517,FBgn0053543,FBgn0085385,FBgn0259222,FBgn0259927,FBgn0260657,FBgn0261090,FBgn0261262,B0212.5,D2021.2,F47D12.1,ZC155.5,ZC196.7
cluster_size:35 FBgn0005655,FBgn0011704,FBgn0011762,FBgn0014861,FBgn0015618,FBgn0015925,FBgn0015929,FBgn0017577,FBgn0024332,FBgn0028700,FBgn0031078,FBgn0031252,FBgn0032698,FBgn0033089,FBgn0033846,FBgn0034908,FBgn0035194,FBgn0037569,FBgn0051054,C14B9.4,F10G7.4,F29B9.6,F32D1.1,F58B3.6,F58F6.4,K01G5.4,K08F9.2,K09H9.2,M03C11.4,R10E4.4,R53.6,W02D9.1,Y41C4A.14,Y53F4B.9,Y59A8A.1
...
```
2. **sideone.txt**, file for storing the fly gene IDs
3. **sidetwo.txt**, file for storing the worm gene IDs
4. **consensus_matrix.txt**, file for storing the consensus matrix 
5. **adjacency_matrix.txt**, file for storing the bi-adjacency matrix 
