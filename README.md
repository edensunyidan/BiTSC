# BiTSC

## Requirements
* Python 3.6.8
* Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-157-generic x86_64)

## Input data files:
| data file          | format     | description   |
| :---               | :---       | :---          |
| ```covariate```   | csv       | node covariate dataset. length  equals the number of species. length>=2. N arguments from the command line will be gathered together into a list|
| ```edge```       | csv        | edge dataset. length equals equals the combinatorial of number of species. ${N \choose 2}$ N arguments from the command line will be gathered together into a list|

## Input parameters:
| parameters       | type       | description |
| :---             | :---       | :---         |
| ```k```          | int        | \[K_0\] or \[K_min, K_max\]. Default=\[5, 50\]    |
| ```kernel```     | string     | metric used when calculating kernel. Default = 'rbk'   |
| ```tau```        | float      | vector containing values of $\tau$'s. Default = 1. otherwise, length equals equals the combinatorial of number of species. |
| ```ncore```      | int        | number of cores used in parallel computation     |
| ```niter```      | int        | number of sub-sampling procedures     |
| ```alpha```      | float      | vector containing values of $\alpha$'s   |
| ```heatmap```    | boolean    | heatmap of sub-consensus matrix. Default = False.      |
| ```threshold```  | int        | minimum number of nodes on each side to appear in the sub-consensus matrix.      |
| ```dir```        | string     | path to output folder directory  |


## Output files:
| output                 | format     | description |
| :---                   | :---       | :---          |
| ```result_alpha.txt```      | csv        | clustering result |
|```alpha.pdf``` | pdf        | heatmap of the sub-consensus matrix corresponding to ```alpha``` |


## Examples

**Scenario 1 : 2 sides, unspecified parameters**
```
$ python BiTSC.py \
'--covariate' 'cov_1.csv' 'cov_2.csv'  \ 
'--edge' 'edge_12.csv'    
```
**Scenario 2 : 2 sides, specified parameters**
```
$ python BiTSC.py \
'--covariate' 'cov_1.csv' 'cov_2.csv' \
'--edge' 'edge_12.csv' \
'--k' '10' '50' \
'--kernel' 'rbk' \
'--tau' '1' '1' \
'--rho' '0.8' \
'--ncore' '10' \
'--niter' '100' \
'--alpha' '0.9' \
'--heatmap' 'True' \
'--threshold' ' 10'
```

**Scenario 3 : >=3 sides, unspecified parameters**
```
$ python BiTSC.py \
'--covariate' 'cov_1.csv' 'cov_2.csv' 'cov_3.csv' \
'--edge' 'edge_12.csv' 'edge_13.csv' 'edge_23.csv'   
```
**Scenario 4 : >=3 sides, specified parameters**
```
$ python BiTSC.py 
'--covariate' 'cov_1.csv' 'cov_2.csv' 'cov_3.csv' \
'--edge' 'edge_12.csv' 'edge_13.csv' 'edge_23.csv' \
'--k' '10' '50' \
'--kernel' 'rbk' \
'--tau' '1' '1' '1' '1' '1' '1' \
'--rho' '0.8' \
'--ncore' '10' \
'--niter' '100' \
'--alpha' '0.9' \
'--heatmap' 'True' \
'--threshold' ' 10'      
```
