# A Bipartite Tight Spectral Clustering Algorithm for Identifying Conserved Gene Co-clusters
Yidan Eden Sun and Jingyi Jessica Li

## Requirements
* Python 3.6
* numpy
* pandas
* json
* sklearn
* multiprocessing
* Linux

## Implementation details

### How to Run
1. Download the package from github, which is named by **BiTSC-master**
2. In linux terminal, navigate to the folder **BiTSC-master**, for example:
```
cd Downloads/BiTSC-master
```
3. Under the current directory, run:
```
python3 BiTSC.py linkage_filepath sideonedata_filepath sidetwodata_filepath k_min alpha root_dir_savedata & disown
```
### Documentations the input files and data:

1. **linkage_filepath**: path directory of orthology data
2. **sideonedata_filepath**: path directory of node covariate matrix on side 1
3. **sidetwodata_filepath**: path directory of node covariate matrix on side 2
4. **k_min**: the of number of clusters K_0
5. **alpha**: tuning parameter for tightness
6. **root_dir_savedata**: path directory of the folder for saving the output data

### An example on the fly-worm data
```
python3 BiTSC.py /home/yidan/Downloads/BiTSC-master/data/orthologs_data_uniq.csv /home/yidan/Downloads/BiTSC-master/data/dm_timecourse_FPKMs.csv /home/yidan/Downloads/BiTSC-master/data/ce_timecourse_FPKMs.csv 30 0.8 /home/yidan/Downloads/BiTSC-master/data/result
```

### Documentations for the output text files, which are all saved in the folder named by root_dir_savedata:
1. **cluster.txt**, file for storing the clustering results, for example:
```
cluster_size:5  FBgn0005655,FBgn0011704,FBgn0011762,FBgn0014861,FBgn0015618,C14B9.4,F10G7.4,F29B9.6
cluster_size:7  FBgn0003425,FBgn0015558,FBgn0015774,FBgn0024189,C07A12.7,C10F3.4,C25F6.4
...
```
2. **sideone.txt**, file for storing the fly gene IDs
3. **sidetwo.txt**, file for storing the worm gene IDs
4. **consensus_matrix.txt**, file for storing the consensus matrix 
5. **adjacency_matrix.txt**, file for storing the bi-adjacency matrix 
