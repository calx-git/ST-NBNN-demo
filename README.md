# Demo Code -- ST-NBNN for Skeleton-based Action Recognition

## Requirements
* [ANN](http://www.cs.umd.edu/~mount/ANN/) ( Approximate Nearest Neighbor )  
* [Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) ( Matlab )
* [CVX](http://cvxr.com/cvx/) Toolbox ( Matlab )


## Steps
1. In **NBNN** folder, run `make` to compile `nbnn.cpp` file.
2. Run `nbnn.exe` to generate spatial temporal matrix ( stored in **NBNN / MHAD** folder ) .
3. Run `dataTrans.m` in **. / NBNN / MHAD** folder to transfer ST-matrix to mat format.
4. Copy generated mat file `data_X.mat` to **. / ST-NBNN / MHAD**.
5. Run `demo.m` in **ST-NBNN** folder.


## Attention
* You may need to re-complie the ANN, Liblinear and CVX toolboxes depending on what OS you use.
* The MHAD dataset provided is down-sampled by picking one frame of each 20 frames due to the large size. The expected results would be _89.1%_ for NBNN and _100%_ for ST-NBNN.


## Citation
Please cite the following paper if you use this source code in your research.


```
@InProceedings{Weng_2017_CVPR,  
author    = {Weng, Junwu and Weng, Chaoqun and Yuan, Junsong},  
title     = {Spatio-Temporal Naive-Bayes Nearest-Neighbor (ST-NBNN) for Skeleton-Based Action Recognition},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
month     = {July},  
year      = {2017}  
}
```