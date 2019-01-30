# A Bipartite Tight Spectral Clustering Algorithm for Identifying Conserved Gene Co-clusters
Yidan Eden Sun and Jingyi Jessica Li

## Run BiTSC.py on linux, under the current directory:
```
python3 BiTSC.py linkage_filepath sideonedata_filepath sidetwodata_filepath k_min alpha root_dir_savedata & disown
```
1. linkage_filepath: path directory of orthology data
2. sideonedata_filepath: path directory of node covariate matrix on side 1
3. sidetwodata_filepath: path directory of node covariate matrix on side 2
4. k_min: the of number of clusters K_0
5. alpha: tuning parameter for tightness
6. root_dir_savedata: path directory of the folder for saving the output data

## Output .txt file in root_dir_savedata:
1. cluster.txt
2. sideone.txt
3. sidetwo.txt
4. consensus_matrix.txt
5. adjacency_matrix.txt
