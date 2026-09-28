"""
Dataset construction and splitting logic (Random Edge Split & Cold-Protein Split).
"""
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from utils import setup_logger

logger = setup_logger("dataset")

def load_ppi_graph(edges_path, embeddings_path=None):
    """
    Loads PPI edges and node features (ESM-2 embeddings if provided, else random).
    Returns a PyG Data object and node mappings.
    """
    logger.info("Loading PPI graph...")
    edges_df = pd.read_csv(edges_path)
    
    # Get all unique proteins
    unique_proteins = set(edges_df["uniprot1"]).union(set(edges_df["uniprot2"]))
    unique_proteins = sorted(list(unique_proteins))
    num_nodes = len(unique_proteins)
    
    node_to_idx = {prot: i for i, prot in enumerate(unique_proteins)}
    
    # Create edge_index
    src = [node_to_idx[p] for p in edges_df["uniprot1"]]
    dst = [node_to_idx[p] for p in edges_df["uniprot2"]]
    
    # Make undirected
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    # Node features
    if embeddings_path and os.path.exists(embeddings_path):
        logger.info(f"Loading ESM-2 embeddings from {embeddings_path}")
        esm2_dict = torch.load(embeddings_path)
        x = torch.zeros((num_nodes, list(esm2_dict.values())[0].shape[0]))
        missing = 0
        for prot, idx in node_to_idx.items():
            if prot in esm2_dict:
                x[idx] = esm2_dict[prot]
            else:
                missing += 1
        if missing > 0:
            logger.warning(f"Missing embeddings for {missing} proteins (initialized to zero).")
    else:
        logger.info("No embeddings provided. Using random 64-dim embeddings.")
        x = torch.randn((num_nodes, 64))
        
    data = Data(x=x, edge_index=edge_index)
    logger.info(f"Graph constructed: {data.num_nodes} nodes, {data.num_edges//2} undirected edges.")
    return data, node_to_idx, unique_proteins

def create_random_split(data):
    """
    Random edge split (70/15/15).
    Removes val/test edges from the message-passing graph.
    Automatically generates 1:1 negative samples.
    """
    logger.info("Creating Random Edge Split (70/15/15)...")
    transform = RandomLinkSplit(
        num_val=0.15,
        num_test=0.15,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data

def create_cold_protein_split(data, unique_proteins, node_to_idx):
    """
    Cold-protein split: 
    - 70% of proteins for training
    - 15% for validation
    - 15% for testing
    Edges between train proteins -> train_data
    Edges touching val proteins -> val_data
    Edges touching test proteins -> test_data
    """
    logger.info("Creating Cold-Protein Split...")
    
    num_nodes = len(unique_proteins)
    indices = np.random.permutation(num_nodes)
    
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)
    
    train_nodes = set(indices[:train_end])
    val_nodes = set(indices[train_end:val_end])
    test_nodes = set(indices[val_end:])
    
    # Convert edge_index to set of pairs for easy filtering
    edges = data.edge_index.t().numpy()
    
    train_edges, val_edges, test_edges = [], [], []
    
    # We only process canonical edges (src < dst) to avoid double counting during assignment
    for src, dst in edges:
        if src >= dst:
            continue
            
        if src in test_nodes or dst in test_nodes:
            test_edges.append([src, dst])
        elif src in val_nodes or dst in val_nodes:
            val_edges.append([src, dst])
        elif src in train_nodes and dst in train_nodes:
            train_edges.append([src, dst])
            
    def build_split_data(pos_edges_list, is_train):
        pos_edges = torch.tensor(pos_edges_list, dtype=torch.long).t()
        
        # Negative sampling (1:1 ratio)
        num_neg = pos_edges.size(1)
        neg_edges = []
        while len(neg_edges) < num_neg:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v: # simplification: might accidentally sample a true edge, but rare in sparse graph
                neg_edges.append([u, v])
        neg_edges = torch.tensor(neg_edges, dtype=torch.long).t()
        
        # Combine labels
        edge_label_index = torch.cat([pos_edges, neg_edges], dim=1)
        edge_label = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])
        
        # For training, message passing graph = positive edges
        # For val/test, message passing graph = train positive edges (no leakage)
        if is_train:
            msg_edges = torch.cat([pos_edges, pos_edges.flip(0)], dim=1)
        else:
            train_pos = torch.tensor(train_edges, dtype=torch.long).t()
            msg_edges = torch.cat([train_pos, train_pos.flip(0)], dim=1)
            
        return Data(x=data.x, edge_index=msg_edges, edge_label_index=edge_label_index, edge_label=edge_label)
        
    train_data = build_split_data(train_edges, is_train=True)
    val_data = build_split_data(val_edges, is_train=False)
    test_data = build_split_data(test_edges, is_train=False)
    
    logger.info(f"Cold Split - Train pos edges: {train_data.edge_label.sum().item()}")
    logger.info(f"Cold Split - Val pos edges: {val_data.edge_label.sum().item()}")
    logger.info(f"Cold Split - Test pos edges: {test_data.edge_label.sum().item()}")
    
    return train_data, val_data, test_data
