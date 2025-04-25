"""Load and preprocess raw CSVs into a torch_geometric.data.Data object."""
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from utils.config import DATA_DIR, SEED
from utils.graph_utils import build_edge_index

__all__ = ["load_elliptic_data"]


def load_elliptic_data(device: str = "cpu") -> Data:
    """Returns a *PyG* Data object for the Elliptic Bitcoin dataset."""
    # ---- Features ---------------------------------------------------------
    features_df = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    tx_ids = features_df.iloc[:, 0].values
    x = torch.tensor(features_df.iloc[:, 2:].values, dtype=torch.float)

    # ---- Mapping ----------------------------------------------------------
    tx_id_to_idx = {str(tx_id): idx for idx, tx_id in enumerate(tx_ids)}

    # ---- Labels -----------------------------------------------------------
    classes_df = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv",
                             names=["txId", "class"])

    label_map = {"1": 1,      # fraud
                 "2": 0,      # legit
                 "unknown": -1}

    classes_df["class"] = classes_df["class"].map(label_map)

    # fill the label tensor (-1 for unknown)
    labels = np.full(len(tx_ids), -1, dtype=np.int64)
    for _, row in classes_df.iterrows():
        idx = tx_id_to_idx.get(str(row["txId"]))
        if idx is not None:
            labels[idx] = row["class"]

    y = torch.tensor(labels, dtype=torch.long)

    # ---- Edges ------------------------------------------------------------
    edges_df = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")
    edge_index = build_edge_index(edges_df, tx_id_to_idx)

    # ---- Assemble Data object --------------------------------------------
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = data.y != -1  # labelled nodes only

    return data.to(device)