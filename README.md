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
2. In terminal, navigate to the folder **BiTSC-master**, for example:
```
cd Downloads/BiTSC-master
```
3. Under the current directory, run:
```
python3 BiTSC.py linkage_filepath sideonedata_filepath sidetwodata_filepath k_min alpha root_dir_savedata & disown
```
### Documentations the input files are:

1. **linkage_filepath**: path directory of orthology data
2. **sideonedata_filepath**: path directory of node covariate matrix on side 1
3. **sidetwodata_filepath**: path directory of node covariate matrix on side 2
4. **k_min**: the of number of clusters K_0
5. **alpha**: tuning parameter for tightness
6. **root_dir_savedata**: path directory of the folder for saving the output data

### An example of fly and worm data
```
python3 BiTSC.py /home/yidan/Downloads/BiTSC-master/data/orthologs_data_uniq.csv /home/yidan/Downloads/BiTSC-master/data/dm_timecourse_FPKMs.csv /home/yidan/Downloads/BiTSC-master/data/ce_timecourse_FPKMs.csv 30 0.8 /home/yidan/Downloads/BiTSC-master/data/result
```

### Documentations for the output text files saved in root_dir_savedata:
1. **cluster.txt**, store the clustering results
2. **sideone.txt**, stores the fly gene IDs
3. **sidetwo.txt**, stores the worm gene IDs
4. **consensus_matrix.txt**, stores the consensus matrix 
5. **adjacency_matrix.txt**, stores the bi-adjacency matrix 
