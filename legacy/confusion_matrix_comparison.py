"""
Confusion Matrix Comparison
Evaluates all models (GCN, GAT, Random Forest, Logistic Regression) on the test set
and generates confusion matrices to identify the best performing model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Import model architectures from model_battle
from model_battle import GCNWithDropout, GATWithDropout, prepare_features_for_sklearn


def load_data():
    """Load and prepare the cleaned knowledge graph data."""
    print("Loading cleaned KG data...")
    df = pd.read_csv('kg_clean.csv')
    
    # Create node mappings
    all_nodes = pd.concat([df['x_name'], df['y_name']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    # Build edge index
    edge_index = torch.tensor([
        [node_to_idx[x] for x in df['x_name']],
        [node_to_idx[y] for y in df['y_name']]
    ], dtype=torch.long)
    
    # Create simple node features (one-hot or degree-based)
    num_nodes = len(all_nodes)
    x = torch.eye(num_nodes)  # Identity matrix as features
    
    data = Data(x=x, edge_index=edge_index)
    
    # Split data
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0
    )
    
    train_data, val_data, test_data = transform(data)
    
    return train_data, val_data, test_data, node_to_idx


def load_gnn_model(model_path, model_type, num_features, hidden_channels):
    """Load a trained GNN model."""
    if model_type == 'GCN':
        model = GCNWithDropout(num_features, hidden_channels, dropout=0.5)
    elif model_type == 'GAT':
        model = GATWithDropout(num_features, hidden_channels, heads=4, dropout=0.5)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded {model_type} model from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using untrained model.")
    
    return model


def predict_gnn(model, data, threshold=0.5):
    """Generate predictions and probabilities for GNN models."""
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        
        # Positive edges
        pos_edge_index = data.pos_edge_label_index
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        pos_scores = torch.sigmoid(pos_scores)
        
        # Negative edges
        neg_edge_index = data.neg_edge_label_index
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_scores = torch.sigmoid(neg_scores)
        
        # Combine
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ])
        
        predictions = (scores >= threshold).long()
        
    return labels.numpy(), predictions.numpy(), scores.numpy()


def predict_sklearn(model, data, node_to_idx):
    """Generate predictions for sklearn models."""
    # Prepare test edges
    pos_edges = data.pos_edge_label_index.t().numpy()
    neg_edges = data.neg_edge_label_index.t().numpy()
    all_edges = np.vstack([pos_edges, neg_edges])
    labels = np.concatenate([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
    
    # Prepare features
    features = prepare_features_for_sklearn(data, all_edges)
    
    # Predict
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else predictions
    
    return labels, predictions, probabilities


def plot_confusion_matrix(cm, title, ax):
    """Plot a single confusion matrix."""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['No Link', 'Link'], 
                yticklabels=['No Link', 'Link'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')


def compute_metrics(labels, predictions):
    """Compute classification metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }


def main():
    """Main function to compare all models."""
    print("=" * 60)
    print("CONFUSION MATRIX COMPARISON FOR MODEL SELECTION")
    print("=" * 60)
    
    # Load data
    train_data, val_data, test_data, node_to_idx = load_data()
    num_features = train_data.x.size(1)
    hidden_channels = 64
    
    models_results = {}
    
    # 1. GCN Model
    print("\n[1/4] Evaluating GCN...")
    gcn_path = 'gcn_model.pth'
    if os.path.exists(gcn_path):
        gcn_model = load_gnn_model(gcn_path, 'GCN', num_features, hidden_channels)
        labels, preds, probs = predict_gnn(gcn_model, test_data)
        cm_gcn = confusion_matrix(labels, preds)
        metrics_gcn = compute_metrics(labels, preds)
        models_results['GCN'] = {'cm': cm_gcn, 'metrics': metrics_gcn}
        print(f"GCN Metrics: {metrics_gcn}")
    else:
        print(f"GCN model not found at {gcn_path}")
    
    # 2. GAT Model
    print("\n[2/4] Evaluating GAT...")
    gat_path = 'gat_model.pth'
    if os.path.exists(gat_path):
        gat_model = load_gnn_model(gat_path, 'GAT', num_features, hidden_channels)
        labels, preds, probs = predict_gnn(gat_model, test_data)
        cm_gat = confusion_matrix(labels, preds)
        metrics_gat = compute_metrics(labels, preds)
        models_results['GAT'] = {'cm': cm_gat, 'metrics': metrics_gat}
        print(f"GAT Metrics: {metrics_gat}")
    else:
        print(f"GAT model not found at {gat_path}")
    
    # 3. Random Forest
    print("\n[3/4] Evaluating Random Forest...")
    rf_path = 'rf_model.pkl'
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        labels, preds, probs = predict_sklearn(rf_model, test_data, node_to_idx)
        cm_rf = confusion_matrix(labels, preds)
        metrics_rf = compute_metrics(labels, preds)
        models_results['Random Forest'] = {'cm': cm_rf, 'metrics': metrics_rf}
        print(f"Random Forest Metrics: {metrics_rf}")
    else:
        print(f"Random Forest model not found at {rf_path}")
    
    # 4. Logistic Regression
    print("\n[4/4] Evaluating Logistic Regression...")
    lr_path = 'lr_model.pkl'
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            lr_model = pickle.load(f)
        labels, preds, probs = predict_sklearn(lr_model, test_data, node_to_idx)
        cm_lr = confusion_matrix(labels, preds)
        metrics_lr = compute_metrics(labels, preds)
        models_results['Logistic Regression'] = {'cm': cm_lr, 'metrics': metrics_lr}
        print(f"Logistic Regression Metrics: {metrics_lr}")
    else:
        print(f"Logistic Regression model not found at {lr_path}")
    
    # Plot all confusion matrices
    if models_results:
        num_models = len(models_results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            if idx < 4:
                plot_confusion_matrix(results['cm'], model_name, axes[idx])
        
        # Hide unused subplots
        for idx in range(len(models_results), 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices saved to 'confusion_matrices_comparison.png'")
        plt.show()
        
        # Summary table
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        summary_df = pd.DataFrame({
            model: results['metrics'] 
            for model, results in models_results.items()
        }).T
        
        print(summary_df.to_string())
        summary_df.to_csv('model_comparison_summary.csv')
        print("\nSummary saved to 'model_comparison_summary.csv'")
        
        # Identify best model
        best_model = summary_df['F1-Score'].idxmax()
        best_f1 = summary_df.loc[best_model, 'F1-Score']
        
        print("\n" + "=" * 60)
        print(f"BEST MODEL: {best_model}")
        print(f"F1-Score: {best_f1:.4f}")
        print("=" * 60)
        
    else:
        print("\nNo models found. Please train models first using run_project.py or model_battle.py")


if __name__ == "__main__":
    main()
