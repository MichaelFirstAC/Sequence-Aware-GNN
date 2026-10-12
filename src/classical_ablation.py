"""
Phase 6: Classical sequence feature ablation experiments.
Evaluates:
1. Classical Sequence + Structure -> GCN
2. Classical Sequence -> Random Forest
3. Classical Sequence -> Logistic Regression
Across both Random Split and Cold-Protein Split.
Merges results into results/metrics.csv.
"""
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import load_config, setup_logger, set_seed
from dataset import load_ppi_graph, create_random_split, create_cold_protein_split
from models import GCNModel
from evaluate import compute_metrics

logger = setup_logger("classical_ablation")

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

def run_gnn_experiment(model, train_data, val_data, test_data, config, device, name):
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
            torch.save(model.state_dict(), f"best_model_{name.replace(' ', '_')}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
    end_time = time.time()
    
    model.load_state_dict(torch.load(f"best_model_{name.replace(' ', '_')}.pt", weights_only=True))
    test_metrics, _, _ = eval_model(model, test_data)
    test_metrics["Train_Time"] = end_time - start_time
    
    logger.info(f"Test Results [{name}]: AUC={test_metrics['AUC']:.4f}, F1={test_metrics['F1']:.4f}")
    return test_metrics

def run_sklearn_experiment(model, train_data, test_data, name, is_lr=False):
    logger.info(f"--- Starting sklearn experiment: {name} ---")
    start_time = time.time()
    
    x_np = train_data.x.cpu().numpy()
    src_tr = train_data.edge_label_index[0].cpu().numpy()
    dst_tr = train_data.edge_label_index[1].cpu().numpy()
    X_train = np.hstack([x_np[src_tr], x_np[dst_tr]])
    y_train = train_data.edge_label.cpu().numpy().astype(int)
    
    x_test_np = test_data.x.cpu().numpy()
    src_te = test_data.edge_label_index[0].cpu().numpy()
    dst_te = test_data.edge_label_index[1].cpu().numpy()
    X_test = np.hstack([x_test_np[src_te], x_test_np[dst_te]])
    y_test = test_data.edge_label.cpu().numpy().astype(int)
    
    if is_lr:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    model.fit(X_train, y_train)
    end_time = time.time()
    
    test_probs = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_probs)
    metrics["Loss"] = 0.0
    metrics["Train_Time"] = end_time - start_time
    
    logger.info(f"Test Results [{name}]: AUC={metrics['AUC']:.4f}, F1={metrics['F1']:.4f}")
    return metrics

def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["random_seed"])
    device = get_device(config)
    
    edges_path = os.path.join(config["data"]["processed_dir"], "ppi_edges_uniprot.csv")
    classical_emb_path = os.path.join(config["data"]["processed_dir"], "classical_features.pt")
    
    logger.info("Loading PPI graph with Classical (AA comp + 3-mer PCA) features...")
    graph_classical, node_to_idx, unique_proteins = load_ppi_graph(edges_path, embeddings_path=classical_emb_path)
    
    new_results = []
    
    for split_name in ["Random Split", "Cold-Protein Split"]:
        logger.info(f"========== SPLIT: {split_name} ==========")
        
        if split_name == "Random Split":
            train_data, val_data, test_data = create_random_split(graph_classical)
        else:
            train_data, val_data, test_data = create_cold_protein_split(graph_classical, unique_proteins, node_to_idx)
            
        in_dim = train_data.x.size(1)
        h_dim = config["training"]["hidden_dim"]
        do = config["training"]["dropout"]
        
        # 1. Classical Sequence + Structure -> GCN
        gcn_model = GCNModel(in_dim, h_dim, do)
        res_gcn = run_gnn_experiment(gcn_model, train_data, val_data, test_data, config, device, f"GCN_Classical_{split_name}")
        res_gcn["Model"] = "GCN"
        res_gcn["Features"] = "Classical + Structure"
        res_gcn["Split"] = split_name
        new_results.append(res_gcn)
        
        # 2. Classical Sequence -> Random Forest baseline
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        res_rf = run_sklearn_experiment(rf_model, train_data, test_data, f"RF_Classical_{split_name}", is_lr=False)
        res_rf["Model"] = "RandomForest"
        res_rf["Features"] = "Classical Sequence"
        res_rf["Split"] = split_name
        new_results.append(res_rf)
        
        # 3. Classical Sequence -> Logistic Regression baseline
        lr_model = LogisticRegression(max_iter=300, random_state=42, solver="lbfgs")
        res_lr = run_sklearn_experiment(lr_model, train_data, test_data, f"LogReg_Classical_{split_name}", is_lr=True)
        res_lr["Model"] = "LogisticRegression"
        res_lr["Features"] = "Classical Sequence"
        res_lr["Split"] = split_name
        new_results.append(res_lr)
        
    new_df = pd.DataFrame(new_results)
    
    # Merge with existing Phase 5 results
    res_path = os.path.join(config["data"]["results_dir"], "metrics.csv")
    if os.path.exists(res_path):
        old_df = pd.read_csv(res_path)
        # Avoid duplicate rows if re-run
        combined_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["Model", "Features", "Split"], keep="last")
    else:
        combined_df = new_df
        
    combined_df.to_csv(res_path, index=False)
    logger.info(f"Updated all results in {res_path}")
    print("\n" + "="*80)
    print("ALL ABLATION RESULTS (Including Phase 6 Classical Sequence Features):")
    print("="*80)
    print(combined_df[["Model", "Features", "Split", "AUC", "F1", "Precision", "Recall", "Precision@10", "Train_Time"]].to_string(index=False))

if __name__ == "__main__":
    main()
