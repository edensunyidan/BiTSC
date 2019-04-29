# A Bipartite Tight Spectral Clustering Algorithm for Identifying Conserved Gene Co-clusters
Yidan Eden Sun and Jingyi Jessica Li

## Implementation details

### Requirements
* Python 3.6
* numpy
* pandas
* json
* sklearn
* multiprocessing
* Linux

### How to Run
1. Download the folder, which is named by **BiTSC-master**
2. In terminal, navigate to the folder, for example:
```
cd Downloads/BiTSC-master
```
3. Under the current directory, run:
```
python3 BiTSC.py linkage_filepath sideonedata_filepath sidetwodata_filepath k_min alpha root_dir_savedata & disown
```
  where the input files are: 
1. linkage_filepath: path directory of orthology data
2. sideonedata_filepath: path directory of node covariate matrix on side 1
3. sidetwodata_filepath: path directory of node covariate matrix on side 2
4. k_min: the of number of clusters K_0
5. alpha: tuning parameter for tightness
6. root_dir_savedata: path directory of the folder for saving the output data

### A toy example
```
python3 BiTSC.py linkage_filepath sideonedata_filepath sidetwodata_filepath k_min alpha root_dir_savedata & disown
```

## Output .txt file in root_dir_savedata:
1. cluster.txt
2. sideone.txt
3. sidetwo.txt
4. consensus_matrix.txt
5. adjacency_matrix.txt
