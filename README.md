# InpGCL
Codes, datasets for paper "Interpretable Graph Contrastive Learning with Attention-aware for Recommendation"

![AppVeyor](https://img.shields.io/badge/python==3.9.13)
![AppVeyor](https://img.shields.io/badge/numpy==1.26.1)
![AppVeyor](https://img.shields.io/badge/torch==1.10.1+cu111)
![AppVeyor](https://img.shields.io/badge/scipy==1.11.3)
![AppVeyor](https://img.shields.io/badge/torch-sparse==0.6.13)

## Run Code

1 When using InpGCL for the first time, if there is no attention matrix in the Attention folder, please use the following command for similarity matrix calculation (due to the large size of the matrix, the calculation may take some time).:
```
Yelp:    python Main.py --data yelp --temp 0.3 --epsilon 0.5 --M 300 --R 3
Amazon:  python Main.py --data amazon --temp  0.3 --epsilon 1.0 --M 500 --R 6
Tmall:   python Main.py --data tmall  --M 800 --R 6


```

2 After all attention matrices for the datasets are computed, for the second round of model training, you can use the previously calculated attention matrices for training. Therefore, you can use the following command for quick execution.
```
Yelp:    python Main.py --data yelp --temp 0.3 --epsilon 0.5 --M 300 --R 3  --no_cal_mtx_flag
Amazon:  python Main.py --data amazon --temp  0.3 --epsilon 1.0 --M 500 --R 6  --no_cal_mtx_flag
Tmall:   python Main.py --data tmall  --M 800 --R 6 --no_cal_mtx_flag
```





## Datasets
/data/dataset.txt
