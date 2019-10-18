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
2. **one_dir**: full path to node covariate data .csv file on side 1
3. **two_dir**: full path to node covariate data .csv file on side 2
4. **Kcluster**: input number of co-clusters
5. **alpha**: tuning parameter for tightness
6. **out_dir**: full path to output folder directory
7. **ncores**: number of cores used in parallel computation
8. **niters**: number of iterations to run 
9. **subprop**: subsampling proportion in each iteration 

### Documentations of output text files, which are all saved in the folder named by root_dir_savedata:
1. **cluster.txt**, file for storing the clustering results, for example:
```
cluster_size:1
oneID1, oneID2, oneID3, twoID1, twoID2, twoID3, twoID4    
cluster_size:2 
oneID4, oneID5, oneID6, oneID7, twoID5, twoID6
...
```
2. ```sideone.txt```, file for storing the fly gene IDs
3. ```sidetwo.txt```, file for storing the worm gene IDs
