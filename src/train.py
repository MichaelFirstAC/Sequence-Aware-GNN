"""
Training and evaluation loop for Sequence-Aware GNN ablation study.
"""
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from utils import load_config, setup_logger, set_seed
from dataset import load_ppi_graph, create_random_split, create_cold_protein_split
from models import GCNModel, GATModel, MLPModel
from evaluate import compute_metrics

logger = setup_logger("train")

def get_device(config):
    dev_str = config["project"]["device"]
    if dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float())
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def eval_model(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, data.edge_label.float()).item()
    probs = torch.sigmoid(out).cpu().numpy()
    labels = data.edge_label.cpu().numpy()
    metrics = compute_metrics(labels, probs)
    metrics["Loss"] = loss
    return metrics, probs, labels

def run_experiment(model, train_data, val_data, test_data, config, device, name):
    logger.info(f"--- Starting experiment: {name} ---")
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    
    best_val_auc = 0
    patience_counter = 0
    patience = config["training"]["patience"]
    
    start_time = time.time()
    for epoch in range(1, config["training"]["epochs"] + 1):
        loss = train_one_epoch(model, optimizer, train_data)
        val_metrics, _, _ = eval_model(model, val_data)
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val AUC: {val_metrics['AUC']:.4f}")
            
        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            patience_counter = 0
            # Save best weights
            torch.save(model.state_dict(), f"best_model_{name.replace(' ', '_')}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
    end_time = time.time()
    
    # Evaluate on test set
    model.load_state_dict(torch.load(f"best_model_{name.replace(' ', '_')}.pt", weights_only=True))
    test_metrics, probs, labels = eval_model(model, test_data)
    test_metrics["Train_Time"] = end_time - start_time
    
    logger.info(f"Test Results [{name}]: AUC={test_metrics['AUC']:.4f}, F1={test_metrics['F1']:.4f}")
    return test_metrics

def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["random_seed"])
    device = get_device(config)
    
    edges_path = os.path.join(config["data"]["processed_dir"], "ppi_edges_uniprot.csv")
    emb_path = os.path.join(config["data"]["processed_dir"], "esm2_embeddings.pt")
    
    # 1. Structure Only (random node features)
    graph_structure, node_to_idx, unique_proteins = load_ppi_graph(edges_path, embeddings_path=None)
    
    # 2. Sequence + Structure (ESM-2 node features)
    graph_seq, _, _ = load_ppi_graph(edges_path, embeddings_path=emb_path)
    
    results = []
    
    for split_name in ["Random Split", "Cold-Protein Split"]:
        logger.info(f"========== SPLIT: {split_name} ==========")
        
        if split_name == "Random Split":
            train_struct, val_struct, test_struct = create_random_split(graph_structure)
            train_seq, val_seq, test_seq = create_random_split(graph_seq)
        else:
            train_struct, val_struct, test_struct = create_cold_protein_split(graph_structure, unique_proteins, node_to_idx)
            train_seq, val_seq, test_seq = create_cold_protein_split(graph_seq, unique_proteins, node_to_idx)
            
        in_dim = train_struct.x.size(1)
        seq_dim = train_seq.x.size(1)
        h_dim = config["training"]["hidden_dim"]
        do = config["training"]["dropout"]
        
        # Experiment 1: Structure Only (GCN)
        m1 = GCNModel(in_dim, h_dim, do)
        res1 = run_experiment(m1, train_struct, val_struct, test_struct, config, device, f"GCN_Struct_{split_name}")
        res1["Model"] = "GCN"
        res1["Features"] = "Structure Only"
        res1["Split"] = split_name
        results.append(res1)
        
        # Experiment 2: Sequence Only (MLP)
        m2 = MLPModel(seq_dim, h_dim, do)
        res2 = run_experiment(m2, train_seq, val_seq, test_seq, config, device, f"MLP_Seq_{split_name}")
        res2["Model"] = "MLP"
        res2["Features"] = "Sequence Only"
        res2["Split"] = split_name
        results.append(res2)
        
        # Experiment 3: Structure + Sequence (GCN)
        m3 = GCNModel(seq_dim, h_dim, do)
        res3 = run_experiment(m3, train_seq, val_seq, test_seq, config, device, f"GCN_Both_{split_name}")
        res3["Model"] = "GCN"
        res3["Features"] = "Sequence + Structure"
        res3["Split"] = split_name
        results.append(res3)

    # Save all results
    os.makedirs(config["data"]["results_dir"], exist_ok=True)
    results_df = pd.DataFrame(results)
    res_path = os.path.join(config["data"]["results_dir"], "metrics.csv")
    results_df.to_csv(res_path, index=False)
    logger.info(f"All experiments finished. Results saved to {res_path}")
    print(results_df.to_string())

if __name__ == "__main__":
    main()
