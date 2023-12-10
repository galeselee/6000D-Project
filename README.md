# 6000D-Project

This is the Project for DSAA 6000D(Graph Processing and Analytics, Prof. Luo) project. The main purpose of this repo is to reproduce the paper "Heuristic Adaptability to Input Dynamics for SpMM on GPUs" and profile the whole SparseSuite to find some insight for optimization general spmm.

Some code files are borrowed from [Ge-spmm](https://github.com/hgyhungry/ge-spmm) and [dgSPARSE](https://github.com/dgSPARSE)

## Dependency
1. cuda
2. cusparse

## Files description

```
├── data_example
│   └── ash219.mtx  # SparseSuite data example
├── download #Please kindly download the dataset, the whole dataset is over 300GB
│   ├── download.py # script to download sparsesuite
│   └── download.sh # script to launch the download.py
├── README.md
├── scripts
│   ├── get_matrix_feature.cpp # collect the matrix feature, such as nrow, ncol, nnz 
│   ├── preprocess_data.py 
│   └── spmm.cu # launch the test
├── experiment_result # experiments result for sparsesuite
│   ├── 1024.csv
│   ├── 128.csv
│   ├── 16.csv
│   ├── 20.csv
│   ├── 256.csv
│   └── 64.csv
└── src
    ├── kernel
    │   ├── csrspmm_non_transpose.cu
    │   ├── csrspmm_parreduce.cu
    │   ├── csrspmm_rowcaching.cu
    │   ├── csrspmm_seqreduce.cu
    │   ├── gespmm.cc
    │   ├── gespmm_csrcoo_v2.cu
    │   ├── gespmm.h
    │   ├── gespmm_v2.cu
    │   ├── gespmm_v2.h
    │   └── Makefile
    └── util
        ├── cuda_util.cuh
        ├── mmio.hpp
        └── sp_util.hpp
```

## Compile

```
cd ./src/kernel
make 
cd ../../scripts
nvcc spmm.cu -I../src/kernel -L../src/kerenl -lgespmm -Lcuda_path/lib64 -lcusparse -o run
```
## Run

```
./run ../data_example ./out
[INFO] Start 1-th/1 file, filename is ash219.mtx
[INFO] Successful 1-th/1 file, filename is ash219.mtx
cat ./out
filename,nrow,ncol,nnz,sparsity,degree,std_node_dgree,N,cusparse,SEQREDUCE_ROWBALANCE,SR_RB_speed,PARREDUCE_ROWBALANCE,PR_RB_speed,SEQREDUCE_NNZBALANCE,SR_NB_speed,PARREDUCE_NNZBALANCE,PR_NB_speed,ROWCACHING_ROWBALANCE,RC_RB_speed,ROWCACHING_NNZBALANCE,RC_NB_speed,THE_BEST_ALG_INDEX,THE_BEST_ALG_NAME,THE_BEST_ALG_TIME,THE_BEST_ALG_SPEED
ash219.mtx,219,85,438,0.0235294,2,0,128,0.0142976,0.003312,4.31691,0.0077632,1.84171,0.008592,1.66406,0.0070624,2.02447,0.0041056,3.48246,0.01024,1.39625,1,SEQREDUCE_ROWBALANCE,0.003312,4.31691
```

