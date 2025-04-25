# FraudGraph AI Assistant

> Detecting Financial Fraud Using Graph Neural Networks

This repository contains the code, data processing scripts, and models to support our project on financial fraud detection using graph neural networks (GNNs). The project compares several GNN architectures across public dataset **Elliptic**.

## Project Structure

```bash
fraudgraph/
├── checkpoints/
│   ├── gcn_elliptic.pt
│   ├── gat_elliptic.pt
│   └── graphsage_elliptic.pt
├── data/
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│   ├── elliptic_txs_features.csv
│   └── processed/
├── models/
│   ├── train_gcn.py
│   ├── train_gat.py
│   ├── train_graphsage.py
│   └── train_hetgnn.py
├── evaluation/
│   ├── eval_metrics.py
│   └── plots/
├── utils/
│   ├── graph_utils.py
│   ├── data_loader.py
│   └── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/IsmailHatim/FraudGraph.git
cd FraudGraph
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download data
You can download the original data on the following kaggle challenge : [Elliptic Dataset (2019)](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

### 5. Preprocess data
```bash
python utils/data_loader.py
```
This will preprocesse data and return a Dataframe.

### 6. Train a model
You can train any of the GNN models using the provided scripts. For example, to train a GCN and save a checkpoint in the `/checkpoints` folder.
```bash
python models/train_gcn.py --epochs 200 --lr 0.005 --save
```
Other options:
```bash
python models/train_gat.py --epochs 200 --heads 8 --lr 0.005 --save
python models/train_graphsage.py --epochs 200 --lr 0.005 --save
python models/train_hetgnn.py 
```

### 7. Evaluate and visualize
To run evaluation and generate visual insights:
```bash
python evaluation/eval_metrics.py --model gcn
```
Visualizations and graphs will be saved in the `evaluation/plots/` folder. You will first need to train the model to be able to load the last checkpoint.

## Features
- Graph construction from tabular financial datasets
- Node classification, edge classification, and link prediction
- GCN, GAT, GraphSAGE, and HetGNN implementations
- ROC-AUC, PR-AUC, F1, Recall, and precision-based evaluation

## References
- [Elliptic Dataset (2019)](https://www.elliptic.co)

## Authors
- Ismail Hatim  
- Ethan Bitan  
- Antonin Soulier  

All contributions were made equally as part of a project at CentraleSupélec, 2025.