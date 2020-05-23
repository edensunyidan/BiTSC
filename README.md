# BiTSC

## Requirements
* Python 3.6.8
* Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-157-generic x86_64)

## Input data files
| File            | Format   | Description   |
| :---            | :---     | :---          |
| ```covariate``` | csv      | Node covariate dataset. Denote N as the length, N>=2| 
| ```edge```      | csv      | Edge dataset. Length=N(N-1)/2 |

## Input parameters
| Parameter       | Type       | Description |
| :---             | :---       | :---         |
| ```k```          | int        | \[K_0\] or \[K_min, K_max\]. Default=\[5, 50\]      |
| ```kernel```     | string     | Metric used when calculating kernel. Default='rbk'   |
| ```tau```        | float      | $\tau$'s. Length=N(N-1). Default=1's. For example, if N=3, $\tau = \[(\tau_1, \tau_2), (\tau_1, \tau_3), (\tau_2, \tau_3)\]$ |
| ```rho```        | float      | Sub-sampling proportion. Default=0.8 |
| ```ncore```      | int        | Number of cores used in parallel computation. Default=10 |
| ```niter```      | int        | Number of sub-sampling iterations. Default=100 |
| ```alpha```      | float      | $\alpha$'s. Default=\[0.90, 0.95, 1.00\] |
| ```heatmap```    | bool    | If True, will return the heatmap of sub-consensus matrix corresponding to each $\alpha$. Default=False|
| ```threshold```  | int        | Minimum number of nodes on each side to appear in the sub-consensus matrix. Default=10|
| ```dir```        | string     | Path to output folder. Default='./'|


## Output files
| File                 | Format     | Description |
| :---                   | :---       | :---          |
| ```result_alpha.txt```      | txt       | Clustering result |
|```alpha.pdf``` | pdf        | Heatmap of the sub-consensus matrix corresponding to $\alpha$ |


## Examples

**Scenario 1 : 2 sides, unspecified parameters**
```
$ python BiTSC.py \
'--covariate' 'cov_1.csv' 'cov_2.csv' \ 
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
'--alpha' '0.90' '0.95' '1.00' \
'--heatmap' 'True' \
'--threshold' '10' \
'--dir' './'
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
'--alpha' '0.90' '0.95' '1.00' \
'--heatmap' 'True' \
'--threshold' '10' \  
'--dir' './'
```
