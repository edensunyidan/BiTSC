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
$ python3 BiTSC.py '--covariate' './data/node_covariate_one.csv' './data/node_covariate_two.csv' '--edge' './data/edge_one_two.csv' & disown
```
### Input data files and parameters:

1. ```-covariate```: path to orthology data .csv file
2. ```--edge```: path to node covariate data .csv file on side 1

### Output files:
1. ```cluster.txt```, clustering results
2. ```Alpha=a_value.pdf```, heatmaps of sub-consensus matrices corresponding to ```alphas```

