"""
MODEL BATTLE: Professional ML Engineering Pipeline
====================================================

This module implements:
1. Model Battle (Comparative Analysis) - GCN vs GAT vs Random Forest
2. Trust System (Uncertainty Quantification) - Monte Carlo Dropout
3. MLOps Pipeline - Experiment Tracking with MLflow

Run this AFTER run_project.py has trained the initial model.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import os
import sys

# Parse command-line argument for model selection
comparison_model = sys.argv[1] if len(sys.argv) > 1 else '4'
model_map = {
    '1': ['GAT'],
    '2': ['RandomForest'],
    '3': ['LogisticRegression'],
    '4': ['RandomForest', 'LogisticRegression', 'GAT']
}
models_to_compare = model_map.get(comparison_model, ['RandomForest', 'LogisticRegression', 'GAT'])

# ==========================================
# SETUP & DATA LOADING
# ==========================================
print("--- LOADING DATA FOR MODEL BATTLE ---")
if not os.path.exists('kg_clean.csv'):
    print("❌ Error: 'kg_clean.csv' not found!")
    sys.exit()

df = pd.read_csv('kg_clean.csv')
print(f"Loaded {len(df)} connections.")

# Map Strings to IDs
nodes = pd.concat([df['x_name'], df['y_name']]).unique()
num_nodes = len(nodes)
node_map = {name: i for i, name in enumerate(nodes)}
reverse_node_map = {i: name for name, i in node_map.items()}

# Create Edge Index
src = [node_map[name] for name in df['x_name']]
dst = [node_map[name] for name in df['y_name']]
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Create Graph Object
data = Data(edge_index=edge_index, num_nodes=num_nodes)
print(f"Graph Created! Nodes: {num_nodes}, Edges: {data.num_edges}")

# Split Data
transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

print(f"✅ Using Device: {device}")

# ==========================================
# MODELS WITH MONTE CARLO DROPOUT
# ==========================================

class GCNWithDropout(torch.nn.Module):
    """GCN Model with Dropout for Uncertainty Quantification"""
    def __init__(self, num_nodes, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout_rate = dropout

    def encode(self, training=False):
        x = self.embedding.weight
        x = self.conv1(x, train_data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=training or self.training)
        x = self.conv2(x, train_data.edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, edge_label_index):
        z = self.encode(training=self.training)
        return self.decode(z, edge_label_index)


class GATWithDropout(torch.nn.Module):
    """Graph Attention Network with Dropout"""
    def __init__(self, num_nodes, hidden_channels=64, heads=4, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.dropout_rate = dropout

    def encode(self, training=False):
        x = self.embedding.weight
        x = self.conv1(x, train_data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=training or self.training)
        x = self.conv2(x, train_data.edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, edge_label_index):
        z = self.encode(training=self.training)
        return self.decode(z, edge_label_index)


# ==========================================
# MONTE CARLO DROPOUT: UNCERTAINTY QUANTIFICATION
# ==========================================

def monte_carlo_predictions(model, edge_label_index, num_iterations=100):
    """
    Get uncertain predictions by running model multiple times with dropout enabled.
    
    Returns:
        predictions: (num_edges, num_iterations) - raw predictions
        probabilities: (num_edges, num_iterations) - sigmoid probabilities
        confidence: (num_edges,) - standard deviation across iterations
        mean_prob: (num_edges,) - mean probability
    """
    predictions_list = []
    
    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(num_iterations):
            out = model(edge_label_index)
            predictions_list.append(out.cpu().numpy())
    
    predictions = np.array(predictions_list).T  # (num_edges, num_iterations)
    probabilities = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Confidence = std dev (low std = certain, high std = uncertain)
    confidence = np.std(probabilities, axis=1)
    mean_prob = np.mean(probabilities, axis=1)
    
    return predictions, probabilities, confidence, mean_prob


# ==========================================
# BASELINE MODELS (Scikit-learn)
# ==========================================

def prepare_features_for_sklearn():
    """Convert graph to feature representation for sklearn models"""
    # Simple approach: Use node embeddings as features
    
    # Node degree as features
    from torch_geometric.utils import degree
    deg = degree(train_data.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    
    # Create edge features: concatenate degree of source and target
    edge_features = []
    for src, dst in zip(train_data.edge_label_index[0].cpu().numpy(), 
                        train_data.edge_label_index[1].cpu().numpy()):
        features = np.array([deg[src].item(), deg[dst].item(), 
                           deg[src].item() * deg[dst].item()])  # degree product
        edge_features.append(features)
    
    X = np.array(edge_features)
    y = train_data.edge_label.cpu().numpy()
    
    return X, y


# ==========================================
# TRAINING & EVALUATION
# ==========================================

def train_gnn_model(model_class, model_name, epochs=50, lr=0.01, hidden_channels=64):
    """Train a GNN model and log metrics to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*50}")
        print(f"🚀 TRAINING: {model_name}")
        print(f"{'='*50}")
        
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("hidden_channels", hidden_channels)
        
        model = model_class(num_nodes=num_nodes, hidden_channels=hidden_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_scores = []
        
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            out = model(train_data.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label.float())
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.edge_label_index)
                val_pred = torch.sigmoid(val_out).cpu().numpy()
                val_true = val_data.edge_label.cpu().numpy()
                val_auc = roc_auc_score(val_true, val_pred)
                val_scores.append(val_auc)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                mlflow.log_metric("val_auc", val_auc, step=epoch)
        
        # Test Evaluation
        model.eval()
        with torch.no_grad():
            test_out = model(test_data.edge_label_index)
            test_pred = torch.sigmoid(test_out).cpu().numpy()
            test_true = test_data.edge_label.cpu().numpy()
        
        # Calculate metrics
        test_auc = roc_auc_score(test_true, test_pred)
        test_pred_binary = (test_pred > 0.5).astype(int)
        test_precision = precision_score(test_true, test_pred_binary, zero_division=0)
        test_recall = recall_score(test_true, test_pred_binary, zero_division=0)
        test_f1 = f1_score(test_true, test_pred_binary, zero_division=0)
        
        # Precision@K metrics
        top_10_indices = np.argsort(test_pred)[-10:]
        precision_at_10 = np.mean(test_true[top_10_indices])
        
        top_20_indices = np.argsort(test_pred)[-20:]
        precision_at_20 = np.mean(test_true[top_20_indices])
        
        print(f"\n📊 TEST RESULTS FOR {model_name}:")
        print(f"   ROC-AUC: {test_auc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1-Score: {test_f1:.4f}")
        print(f"   Precision@10: {precision_at_10:.4f}")
        print(f"   Precision@20: {precision_at_20:.4f}")
        
        # Log final metrics
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("precision_at_10", precision_at_10)
        mlflow.log_metric("precision_at_20", precision_at_20)
        
        # Log model (using 'artifact_path' parameter name for compatibility)
        mlflow.pytorch.log_model(model, artifact_path="model")
        
        return {
            'model': model,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'precision_at_10': precision_at_10,
            'precision_at_20': precision_at_20,
            'test_pred': test_pred,
            'test_true': test_true
        }


def train_sklearn_baseline(model_class, model_name):
    """Train baseline sklearn model"""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\n{'='*50}")
        print(f"🚀 TRAINING BASELINE: {model_name}")
        print(f"{'='*50}")
        
        mlflow.log_param("model", model_name)
        
        X, y = prepare_features_for_sklearn()
        
        model = model_class()
        model.fit(X, y)
        
        # Test predictions
        pred = model.predict_proba(X)[:, 1]
        
        auc = roc_auc_score(y, pred)
        pred_binary = model.predict(X)
        precision = precision_score(y, pred_binary, zero_division=0)
        recall = recall_score(y, pred_binary, zero_division=0)
        f1 = f1_score(y, pred_binary, zero_division=0)
        
        # Precision@K
        top_10_indices = np.argsort(pred)[-10:]
        prec_at_10 = np.mean(y[top_10_indices])
        
        top_20_indices = np.argsort(pred)[-20:]
        prec_at_20 = np.mean(y[top_20_indices])
        
        print(f"\n📊 TEST RESULTS FOR {model_name}:")
        print(f"   ROC-AUC: {auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Precision@10: {prec_at_10:.4f}")
        print(f"   Precision@20: {prec_at_20:.4f}")
        
        # Log metrics
        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("precision_at_10", prec_at_10)
        mlflow.log_metric("precision_at_20", prec_at_20)
        
        return {
            'model': model,
            'test_auc': auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'precision_at_10': prec_at_10,
            'precision_at_20': prec_at_20,
            'test_pred': pred,
            'test_true': y
        }


# ==========================================
# UNCERTAINTY QUANTIFICATION DEMO
# ==========================================

def demonstrate_uncertainty(model, model_name):
    """Show how Monte Carlo Dropout captures model uncertainty"""
    
    print(f"\n{'='*50}")
    print(f"🎲 UNCERTAINTY QUANTIFICATION: {model_name}")
    print(f"{'='*50}")
    
    predictions, probabilities, confidence, mean_prob = monte_carlo_predictions(
        model, test_data.edge_label_index, num_iterations=100
    )
    
    # Find certain and uncertain predictions
    certain_threshold = np.percentile(confidence, 25)  # Bottom 25%
    uncertain_threshold = np.percentile(confidence, 75)  # Top 25%
    
    certain_indices = np.where(confidence < certain_threshold)[0]
    uncertain_indices = np.where(confidence > uncertain_threshold)[0]
    
    test_true = test_data.edge_label.cpu().numpy()
    
    # Accuracy on certain predictions
    if len(certain_indices) > 0:
        certain_pred = (mean_prob[certain_indices] > 0.5).astype(int)
        certain_acc = np.mean(certain_pred == test_true[certain_indices])
        print(f"\n✅ CERTAIN Predictions ({len(certain_indices)} samples):")
        print(f"   Average confidence: {(1 - np.mean(confidence[certain_indices])):.4f}")
        print(f"   Accuracy: {certain_acc:.4f}")
    
    # Accuracy on uncertain predictions
    if len(uncertain_indices) > 0:
        uncertain_pred = (mean_prob[uncertain_indices] > 0.5).astype(int)
        uncertain_acc = np.mean(uncertain_pred == test_true[uncertain_indices])
        print(f"\n⚠️  UNCERTAIN Predictions ({len(uncertain_indices)} samples):")
        print(f"   Average confidence: {(1 - np.mean(confidence[uncertain_indices])):.4f}")
        print(f"   Accuracy: {uncertain_acc:.4f}")
    
    # Save visualization
    plt.figure(figsize=(10, 6))
    plt.hist(confidence, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Uncertainty (Std Dev)')
    plt.ylabel('Frequency')
    plt.title(f'Monte Carlo Dropout: Prediction Uncertainty Distribution\n{model_name}')
    plt.axvline(certain_threshold, color='green', linestyle='--', label='Certain Threshold')
    plt.axvline(uncertain_threshold, color='red', linestyle='--', label='Uncertain Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'uncertainty_{model_name.replace(" ", "_")}.png', dpi=100)
    plt.close()
    
    print(f"📈 Uncertainty plot saved to: uncertainty_{model_name.replace(' ', '_')}.png")


# ==========================================
# COMPARISON & VISUALIZATION
# ==========================================

def plot_comparison(results_dict):
    """Plot comparison of all models"""
    
    models = list(results_dict.keys())
    metrics = ['test_auc', 'test_precision', 'test_recall', 'test_f1', 'precision_at_10']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        colors = ['#FF6B6B' if 'Random' in model or 'Logistic' in model else '#4ECDC4' for model in models]
        
        axes[idx].bar(range(len(models)), values, color=colors)
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(metric)
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.close()
    print("📊 Comparison plot saved to: model_comparison.png")


def plot_roc_curves(results_dict):
    """Plot ROC curves for all models"""
    
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        test_pred = results['test_pred']
        test_true = results['test_true']
        
        fpr, tpr, _ = roc_curve(test_true, test_pred)
        roc_auc = results['test_auc']
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Battle')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.close()
    print("📈 ROC curves saved to: roc_curves.png")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Setup MLflow
    mlflow.set_experiment("drug_repurposing_model_battle")
    
    print("\n" + "="*70)
    print(f"🏆 MODEL BATTLE: GCN vs {', '.join(models_to_compare)}")
    print("="*70)
    
    results = {}
    gcn_model = None
    gat_model = None
    
    # 1. BASELINE: Random Forest (if selected)
    if 'RandomForest' in models_to_compare:
        try:
            results['Random Forest'] = train_sklearn_baseline(RandomForestClassifier, 'Random Forest')
        except Exception as e:
            print(f"❌ Random Forest failed: {e}")
    
    # 2. BASELINE: Logistic Regression (if selected)
    if 'LogisticRegression' in models_to_compare:
        try:
            results['Logistic Regression'] = train_sklearn_baseline(LogisticRegression, 'Logistic Regression')
        except Exception as e:
            print(f"❌ Logistic Regression failed: {e}")
    
    # 3. CHAMPION: GCN (always included)
    try:
        results['GCN'] = train_gnn_model(GCNWithDropout, 'GCN', epochs=50, lr=0.01, hidden_channels=64)
        gcn_model = results['GCN']['model']
    except Exception as e:
        print(f"❌ GCN failed: {e}")
    
    # 4. CHALLENGER: GAT (if selected)
    if 'GAT' in models_to_compare:
        try:
            results['GAT'] = train_gnn_model(GATWithDropout, 'GAT', epochs=50, lr=0.01, hidden_channels=64)
            gat_model = results['GAT']['model']
        except Exception as e:
            print(f"❌ GAT failed: {e}")
    
    # ==========================================
    # UNCERTAINTY QUANTIFICATION DEMO
    # ==========================================
    if 'GCN' in results:
        demonstrate_uncertainty(gcn_model, 'GCN')
    
    if 'GAT' in results:
        demonstrate_uncertainty(gat_model, 'GAT')
    
    # ==========================================
    # RESULTS SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  ROC-AUC: {result['test_auc']:.4f}")
        print(f"  Precision: {result['test_precision']:.4f}")
        print(f"  Recall: {result['test_recall']:.4f}")
        print(f"  F1-Score: {result['test_f1']:.4f}")
        print(f"  Precision@10: {result['precision_at_10']:.4f}")
        print(f"  Precision@20: {result['precision_at_20']:.4f}")
    
    # Plot comparisons
    plot_comparison(results)
    plot_roc_curves(results)
    
    print("\n" + "="*70)
    print("✅ MODEL BATTLE COMPLETE!")
    print("📊 Check MLflow dashboard: mlflow ui")
    print("📈 Visualizations saved: model_comparison.png, roc_curves.png")
    print("="*70)
