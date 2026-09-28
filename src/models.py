"""
Graph Neural Network models and baselines for PPI prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class DotProductDecoder(nn.Module):
    """Simple dot product decoder for link prediction."""
    def forward(self, z, edge_index):
        # z: node embeddings (num_nodes, hidden_dim)
        # edge_index: (2, num_edges)
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)

class GCNModel(nn.Module):
    """2-layer GCN model with dot product decoder."""
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.decoder = DotProductDecoder()
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decoder(z, edge_label_index)

class GATModel(nn.Module):
    """2-layer GAT model with dot product decoder."""
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.3):
        super().__init__()
        # First layer outputs `heads * hidden_channels`
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second layer collapses heads back to `hidden_channels`
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.dropout = dropout
        self.decoder = DotProductDecoder()
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decoder(z, edge_label_index)

class MLPModel(nn.Module):
    """
    MLP baseline that predicts interaction strictly from sequence embeddings.
    Does NOT use the graph structure (edge_index) for message passing.
    Concatenates source and destination features.
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        # We concatenate the two node features, so input is 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x, edge_index, edge_label_index):
        # We ignore edge_index entirely for message passing.
        src = x[edge_label_index[0]]
        dst = x[edge_label_index[1]]
        # Concatenate src and dst features
        combined = torch.cat([src, dst], dim=-1)
        # Returns raw logits (squeeze to match dot product shape)
        return self.mlp(combined).squeeze(-1)
