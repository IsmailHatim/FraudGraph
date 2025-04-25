"""Graph Attention Network."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 2,
                 heads: int = 8,
                 dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,
                              heads=heads, dropout=dropout)
        # second layer uses a single head, concat=False averages heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                              heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):  # noqa: D401, N802
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x