# KG-Mixup

Official Implementation of the paper ["Toward Degree Bias in Embedding-Based Knowledge Graph Completion"]() (WWW 2023). 

## Abstract

A fundamental task for knowledge graphs (KGs) is knowledge graph completion (KGC). It aims to predict unseen edges by learning representations for all the entities and relations in a KG. A common concern when learning representations on traditional graphs is degree bias. It can affect graph algorithms by learning poor representations for lower-degree nodes, often leading to low performance on such nodes. However, there has been limited research on whether there exists degree bias for embedding-based KGC and how such bias affects the performance of KGC. In this paper, we validate the existence of degree bias in embedding-based KGC and identify the key factor to degree bias. We then introduce a novel data augmentation method, KG-Mixup, to generate synthetic triples to mitigate such bias. Extensive experiments have demonstrated that our method can improve various embedding-based KGC methods and outperform other methods tackling the bias problem on multiple benchmark datasets.

## Requirements

All experiments were conducted using python 3.9.12. 

For the required python packages, please see `requirements.txt`.

```
tqdm>=4.0
torch>=1.10
torch_geometric>=2.0
numpy>=1.21
tensorboard>=2.5
optuna>=2.10.0
matplotlib>=3.6.2
pandas>=1.4.4
```

## Reproduce Results

First clone our repository and install the required python packages.
```
git clone https://github.com/HarryShomer/KG-Mixup.git
cd KG-Mixup
pip install -r requirements.txt
```

### Install kgpy

The code relies on the [kgpy](https://github.com/HarryShomer/kgpy) library. A version is found here in the `kgpy` directory.

It is necessary to install it as a python package. To do so `cd` into the directory and run:
```
cd kgpy
pip install -e .
```

### Pre-Train

For KG-Mixup, the embeddings are extracted from a pre-trained model. The scripts for pre-training each model are in the `scripts/pretrain` folder. For example, to pretrain ConvE on FB15K-237:
```
cd scripts/pretrain
bash conve_fb15k237.sh
```
Once training is done, the model will be saved in the `checkpoints/DATASET` folder. For ConvE on FB15K-237 it will saved as `checkpoints/FB15K-237/conve_fb15k237_pretrain.tar`. 


### KG-Mixup

The scripts for replicating KG-Mixup can be found in the `scripts/kg_mixup` folder. As a reminder, you must have already pre-trained the model. Below is how to replicate the results for ConvE on FB15K-237:
```
cd scripts/kg_mixup
bash conve_fb15k237.sh
```

By default all scripts will attempt to run on cuda. If a cuda GPU isn't available

## Cite
```
@inproceedings{
    shomer23degree,
    title={Toward Degree Bias in Embedding-Based Knowledge Graph Completion},
    author={Harry Shomer and Wei Jin and Wentao Wang and Jiliang Tang},
    booktitle={Proceedings of the ACM Web Conference 2023},    
    year={2023},
}
```