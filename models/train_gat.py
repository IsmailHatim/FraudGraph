import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

import torch.nn.functional as F

from sklearn.metrics import classification_report

# Load node features
features_df = pd.read_csv('./data/elliptic_txs_features.csv', header=None)

# First column: transaction id, second: timestamp, remaining: features
tx_ids = features_df.iloc[:, 0].values
timestamps = features_df.iloc[:, 1].values
features = features_df.iloc[:, 2:].values

# Map tx id to index
tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

# Node feature matrix
x = torch.tensor(features, dtype=torch.float)

# Load labels and map classes to 0/1
classes_df = pd.read_csv('./data/elliptic_txs_classes.csv')
label_map = {1: 1, 2: 0}  # 1 = fraud, 2 = legit -> map fraud to 1, legit to 0
labels = np.full(len(tx_ids), -1)  # initialize with -1 for unknown
for _, row in classes_df.iterrows():
    cls = int(row['class'])
    if cls in label_map:
        idx = tx_id_to_idx.get(row['txId'])
        if idx is not None:
            labels[idx] = label_map[cls]

y = torch.tensor(labels, dtype=torch.long)

# Load edge list
edges_df = pd.read_csv('./data/elliptic_txs_edgelist.csv')
edge_index = []
for _, row in edges_df.iterrows():
    src = tx_id_to_idx.get(row['txId1'])
    dst = tx_id_to_idx.get(row['txId2'])
    if src is not None and dst is not None:
        edge_index.append([src, dst])
# Transpose to shape [2, num_edges]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        # First GAT layer: multi-head attention
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )
        # Output layer: single head, concat=False to average
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training settings
n_epochs = 700
patience = 20
best_loss = float('inf')
epochs_no_improve = 0

# Initialize model
model = GAT(
    in_channels=data.x.size(1),
    hidden_channels=64,
    out_channels=2,
    heads=8,
    dropout=0.6
).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# Mask for labeled nodes
train_mask = data.y != -1

# Training loop
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    correct = pred[train_mask] == data.y[train_mask]
    acc = int(correct.sum()) / int(train_mask.sum())

    if epoch % 20 == 0 or epoch == 1:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    # Early stopping
    if loss.item() < best_loss - 1e-4:
        best_loss = loss.item()
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered at epoch {epoch:03d}')
            break

# Load best model
model.load_state_dict(best_model_state)

# Evaluation
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

mask = data.y != -1
y_true = data.y[mask].cpu().numpy()
    
y_pred = pred[mask].cpu().numpy()

print(classification_report(y_true, y_pred, target_names=['Legit', 'Fraud']))