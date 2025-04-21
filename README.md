# FraudGraph AI Assistant

> Detecting Financial Fraud Using Graph Neural Networks

This repository contains the code, data processing scripts, and models to support our project on financial fraud detection using graph neural networks (GNNs). The project compares several GNN architectures across public datasets such as **Elliptic** and **IEEE-CIS Fraud Detection**.

## Project Structure

```bash
fraudgraph/
├── data/
│   ├── elliptic_txs_classes.csv
│   ├── elliptic_txs_edgelist.csv
│   ├── elliptic_txs_features.csv
│   └── processed/
├── graphs/
│   ├── elliptic_graph.pkl
│   └── ieee_graph.pkl
├── models/
│   ├── train_gcn.py
│   ├── train_gat.py
│   ├── train_graphsage.py
│   └── train_hetgnn.py
├── evaluation/
│   ├── eval_metrics.py
│   ├── explainability.py
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
python utils/data_loader.py --dataset elliptic
python utils/data_loader.py --dataset ieee
```
This will save preprocessed graphs in the `graphs/` folder.

### 6. Train a model
You can train any of the GNN models using the provided scripts. For example, to train a GCN on the Elliptic dataset:
```bash
python models/train_gcn.py --dataset elliptic --epochs 100 --lr 0.01
```
Other options:
```bash
python models/train_gat.py --dataset ieee --epochs 50 --lr 0.005
python models/train_graphsage.py --dataset elliptic
python models/train_hetgnn.py --dataset ieee
```

### 7. Evaluate and visualize
To run evaluation and generate visual insights:
```bash
python evaluation/eval_metrics.py --model gcn --dataset elliptic
python evaluation/explainability.py --model gat --dataset ieee
```
Visualizations and graphs will be saved in the `evaluation/plots/` folder.

## Features
- Graph construction from tabular financial datasets
- Node classification, edge classification, and link prediction
- GCN, GAT, GraphSAGE, and HetGNN implementations
- ROC-AUC, PR-AUC, F1, Recall, and precision-based evaluation
- Graph explainability using attention weights and subgraph tracing

## References
- [Elliptic Dataset (2019)](https://www.elliptic.co)
- [IEEE-CIS Fraud Dataset (Kaggle)](https://kaggle.com/competitions/ieee-fraud-detection)

## Authors
- Ismail Hatim  
- Ethan Bitan  
- Antonin Soulier  

All contributions were made equally as part of a project at CentraleSupélec, 2025.