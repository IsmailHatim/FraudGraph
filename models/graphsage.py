"""GraphSAGE model (mean aggregator)."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """Two‑layer GraphSAGE using mean aggregation and dropout."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.dropout = dropout

    def forward(self, x, edge_index):  # noqa: D401, N802
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x