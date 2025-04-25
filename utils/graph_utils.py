"""Utility helpers for graph construction."""
import torch

__all__ = ["build_edge_index"]

def build_edge_index(edges_df, id_map):
    """Convert an edgelist DataFrame to a PyG edge_index tensor."""
    edges = []
    for _, row in edges_df.iterrows():
        src = id_map.get(str(row["txId1"]))
        dst = id_map.get(str(row["txId2"]))
        if src is not None and dst is not None:
            edges.append([src, dst])
    if not edges:
        raise ValueError("No valid edges found while building edge_index.")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()