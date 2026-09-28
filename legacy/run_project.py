import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import pubchempy as pcp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("--- STEP 1: LOADING DATA ---")
if not os.path.exists('kg_clean.csv'):
    print("Error: 'kg_clean.csv' not found. Please run the cleaning script first!")
    sys.exit()

# Load Data (disable chunked type inference to avoid dtype warnings)
df = pd.read_csv('kg_clean.csv', low_memory=False)

# OPTIONAL: Uncomment this if you have < 8GB RAM to prevent crashes
# df = df.head(100000) 

print(f"Loaded {len(df)} connections.")

# Map Strings (Names) to Integers (IDs)
nodes = pd.concat([df['x_name'], df['y_name']]).unique()
num_nodes = len(nodes)
node_map = {name: i for i, name in enumerate(nodes)}
reverse_node_map = {i: name for name, i in node_map.items()}

# Build a type lookup for nice labeling in search results
type_map = {}
if 'x_type' in df.columns:
    for name, t in zip(df['x_name'], df['x_type']):
        if name not in type_map and pd.notna(t):
            type_map[name] = str(t)
if 'y_type' in df.columns:
    for name, t in zip(df['y_name'], df['y_type']):
        if name not in type_map and pd.notna(t):
            type_map[name] = str(t)

# Create Edge Index for PyTorch
src = [node_map[name] for name in df['x_name']]
dst = [node_map[name] for name in df['y_name']]
edge_index = torch.tensor([src, dst], dtype=torch.long)

# Create Graph Object
data = Data(edge_index=edge_index, num_nodes=num_nodes)
print(f"Graph Created! Nodes: {num_nodes}, Edges: {data.num_edges}")

# Split Data (Train 80% / Val 10% / Test 10%)
transform = RandomLinkSplit(is_undirected=False, add_negative_train_samples=True)
train_data, val_data, test_data = transform(data)

# ==========================================
# HELPER FUNCTIONS (from drug_info.py & explain.py)
# ==========================================

def get_drug_details(drug_name):
    """
    Fetches real-world details for a drug name using PubChem API.
    """
    print(f"   ...fetching details for {drug_name}...")
    try:
        # Search PubChem for the drug name
        compounds = pcp.get_compounds(drug_name, 'name')
        
        if not compounds:
            return "No details found in PubChem."
            
        # Get the first result (best match)
        c = compounds[0]
        
        # 1. Get Synonyms (Commercial Names like 'Tylenol' or 'Advil')
        # We take the top 5 synonyms because the list can be huge
        synonyms = c.synonyms[:5]
        synonyms_str = ", ".join(synonyms) if synonyms else "None"
        
        # 2. Get Molecular Formula
        formula = c.molecular_formula
        
        # 3. Format the output
        info = f"\n   [Formula]: {formula}"
        info += f"\n   [Brands/Synonyms]: {synonyms_str}"
        
        return info

    except Exception as e:
        return f"Error connecting to PubChem: {str(e)}"

# Load graph for visualization
print("   ...Loading graph for visualization (this happens once)...")
graph_df = pd.read_csv('kg_clean.csv', low_memory=False)
G = nx.from_pandas_edgelist(graph_df, 'x_name', 'y_name')

def visualize_connection(drug, disease):
    """Generate explanation graph for Drug -> Disease connections"""
    print(f"   ...Generating explanation for {drug} -> {disease}...")
    
    # Find paths (The "Why")
    try:
        paths = list(nx.all_shortest_paths(G, source=drug, target=disease))
        paths = paths[:5]  # Limit to first 5 paths
    except nx.NetworkXNoPath:
        print("No direct path found in the training data.")
        return
    except Exception as e:
        print(f"Error finding path: {e}")
        return

    # Build subgraph
    node_set = set()
    for path in paths:
        node_set.update(path)
    
    subgraph = G.subgraph(node_set)
    
    # Draw the graph
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Color code nodes
    color_map = []
    for node in subgraph:
        if node == drug:
            color_map.append('green')
        elif node == disease:
            color_map.append('red')
        else:
            color_map.append('skyblue')
            
    nx.draw(subgraph, pos, 
            with_labels=True, 
            node_color=color_map, 
            node_size=2000, 
            font_size=9, 
            font_weight='bold', 
            edge_color='gray', 
            alpha=0.8)
    
    plt.title(f"Mechanism of Action: {drug} vs {disease}")
    
    filename = f"explanation_{drug}_{disease}.png".replace(" ", "_")
    plt.savefig(filename)
    print(f"  Graph image saved to: {filename}")
    plt.close()

def get_connection_pathways(drug, disease, num_paths=3):
    """Get top N connecting pathways between drug and disease"""
    try:
        all_paths = list(nx.all_shortest_paths(G, source=drug, target=disease))
        paths = all_paths[:num_paths]
        
        if not paths:
            return None
        
        explanations = []
        for i, path in enumerate(paths, 1):
            # Format: Drug -> Node1 -> Node2 -> ... -> Disease
            path_str = " → ".join(path)
            path_types = []
            for node in path:
                node_type = type_map.get(node, 'unknown')
                path_types.append(f"{node} [{node_type}]")
            
            explanations.append({
                'path_num': i,
                'length': len(path),
                'simple': path_str,
                'detailed': " → ".join(path_types)
            })
        
        return explanations
    except:
        return None

# ==========================================
# 2. DEFINE GNN MODEL (with Monte Carlo Dropout)
# ==========================================
class DrugRepurposingNet(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels=64, dropout=0.3):
        super().__init__()
        # Embedding: Learn a vector for every node
        self.embedding = torch.nn.Embedding(num_nodes, hidden_channels)
        
        # GCN Layers: Message passing
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout_rate = dropout

    def encode(self, enable_dropout=False):
        x = self.embedding.weight
        x = self.conv1(x, train_data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=enable_dropout or self.training)
        x = self.conv2(x, train_data.edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Predict link probability via Dot Product
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("\n--- STEP 2: TRAINING (50 Epochs) ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

model = DrugRepurposingNet(num_nodes=num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Move data to GPU/CPU
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    
    z = model.encode()
    out = model.decode(z, train_data.edge_label_index)
    
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

print("Training Complete!")

# ==========================================
# 4. PREDICTION TOOLS
# ==========================================

def search_name(query):
    """Helper to find exact spelling in the dataset"""
    query = query.lower()
    matches = []
    for name in nodes:
        if query in str(name).lower():
            label = type_map.get(name, 'unknown')
            matches.append(f"{name} [{label}]")
        if len(matches) >= 10:
            break
    return matches

def interactive_search():
    """Search with suggestions showing top matches"""
    print("\n🔍 INTERACTIVE SEARCH")
    print("Enter partial name and get suggestions:\n")
    
    query = input("Search query: ").strip()
    
    if not query:
        print(" Empty search")
        return None
    
    matches = search_name(query)
    
    if not matches:
        print(f"❌ No matches for '{query}'")
        return None
    
    print(f"\n Found {len(matches)} match(es):\n")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. {match}")
    
    if len(matches) == 1:
        return matches[0].split(" [")[0]  # Return just the name
    
    # Let user select which one
    try:
        choice = input(f"\nSelect entry (1-{len(matches)}) or press Enter to cancel: ").strip()
        if choice and choice.isdigit() and 1 <= int(choice) <= len(matches):
            selected = matches[int(choice)-1]
            return selected.split(" [")[0]  # Return just the name
    except:
        pass
    
    return None

def predict_single_link(drug, disease):
    """Predicts probability of a link between two specific nodes"""
    if drug not in node_map:
        print(f"'{drug}' not found. Did you mean: {search_name(drug)}?")
        return
    if disease not in node_map:
        print(f"'{disease}' not found. Did you mean: {search_name(disease)}?")
        return
        
    src_id = node_map[drug]
    dst_id = node_map[disease]
    
    edge = torch.tensor([[src_id], [dst_id]], device=device)
    
    model.eval()
    with torch.no_grad():
        z = model.encode()
        score = torch.sigmoid(model.decode(z, edge)).item()
    
    print(f"Probability {drug} <-> {disease}: {score*100:.2f}%")

def find_top_drugs_for_disease(disease_name, top_k=5):
    """Scans ALL drugs and fetches details for the winners"""
    
    if disease_name not in node_map:
        print(f" '{disease_name}' not found. Did you mean: {search_name(disease_name)}?")
        return

    print(f"\n Scanning database for {disease_name} treatments...")
    
    drug_names = df[df['x_type'] == 'drug']['x_name'].unique()
    disease_id = node_map[disease_name]
    candidates = []
    
    model.eval()
    with torch.no_grad():
        z = model.encode()
        for drug in drug_names:
            if drug not in node_map: continue
            drug_id = node_map[drug]
            edge = torch.tensor([[drug_id], [disease_id]], device=device)
            score = torch.sigmoid(model.decode(z, edge)).item()
            candidates.append((drug, score))
    
    # Sort by highest score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- TOP {top_k} PREDICTED DRUGS FOR {disease_name.upper()} ---\n")
    
    # LOOP THROUGH THE WINNERS
    for i, (name, score) in enumerate(candidates[:top_k]):
        print(f"{i+1}. {name} (Confidence: {score*100:.1f}%)")
        
        # Get connecting pathways
        pathways = get_connection_pathways(name, disease_name, num_paths=3)
        if pathways:
            print(f"Top Connecting Pathways:")
            for p in pathways:
                print(f"      Path {p['path_num']} (length {p['length']}):")
                print(f"        {p['detailed']}")
        else:
            print(f"No known pathways between {name} and {disease_name}")
        
        # Fetch PubChem details
        details = get_drug_details(name)
        print(details) 
        print("-" * 70)

def predict_with_uncertainty(drug, disease, num_iterations=100):
    """
    MONTE CARLO DROPOUT: Predict with uncertainty quantification
    Run prediction multiple times with dropout enabled to get confidence intervals
    """
    if drug not in node_map:
        print(f"'{drug}' not found. Did you mean: {search_name(drug)}?")
        return
    if disease not in node_map:
        print(f"'{disease}' not found. Did you mean: {search_name(disease)}?")
        return
    
    src_id = node_map[drug]
    dst_id = node_map[disease]
    edge = torch.tensor([[src_id], [dst_id]], device=device)
    
    predictions = []
    model.train()  # Enable dropout
    
    with torch.no_grad():
        for _ in range(num_iterations):
            z = model.encode(enable_dropout=True)
            out = torch.sigmoid(model.decode(z, edge)).item()
            predictions.append(out)
    
    predictions = np.array(predictions)
    mean_prob = np.mean(predictions)
    std_prob = np.std(predictions)
    
    print(f"\n UNCERTAINTY-AWARE PREDICTION: {drug} <-> {disease}")
    print(f"   Mean Probability: {mean_prob*100:.2f}%")
    print(f"   Std Deviation: {std_prob*100:.2f}%")
    print(f"   95% Confidence Interval: [{(mean_prob - 1.96*std_prob)*100:.2f}%, {(mean_prob + 1.96*std_prob)*100:.2f}%]")
    
    if std_prob < 0.1:
        print(f"   HIGH CONFIDENCE - Model is certain")
    elif std_prob < 0.2:
        print(f"   MODERATE CONFIDENCE - Some uncertainty")
    else:
        print(f"   LOW CONFIDENCE - Model is guessing")

def plot_confidence_calibration():
    """Plot histogram of prediction confidence across the entire dataset"""
    print("\n Computing confidence calibration across all edges...")
    print("This may take a moment...")
    
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        z = model.encode()
        
        # Get all unique drug-disease pairs
        drug_diseases = df[(df['x_type'] == 'drug') & (df['y_type'] == 'disease')].copy()
        
        total = len(drug_diseases)
        for idx, (drug, disease) in enumerate(zip(drug_diseases['x_name'], drug_diseases['y_name'])):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{total}", end='\r')
            
            if drug not in node_map or disease not in node_map:
                continue
            
            drug_id = node_map[drug]
            disease_id = node_map[disease]
            edge = torch.tensor([[drug_id], [disease_id]], device=device)
            
            score = torch.sigmoid(model.decode(z, edge)).item()
            all_scores.append(score)
    
    all_scores = np.array(all_scores)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.3f}')
    plt.axvline(np.median(all_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_scores):.3f}')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution - All Predictions')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_scores = np.sort(all_scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.plot(sorted_scores, cumulative, linewidth=2, color='steelblue')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_calibration.png', dpi=150)
    plt.close()
    
    print(f"\n Confidence Calibration Complete!")
    print(f"   Total predictions: {len(all_scores):,}")
    print(f"   Mean confidence: {np.mean(all_scores):.3f}")
    print(f"   Median confidence: {np.median(all_scores):.3f}")
    print(f"   Std deviation: {np.std(all_scores):.3f}")
    print(f"   Min confidence: {np.min(all_scores):.3f}")
    print(f"   Max confidence: {np.max(all_scores):.3f}")
    print(f"   Plot saved: confidence_calibration.png")
    
    return all_scores

# ==========================================
# 5. INTERACTIVE MENU
# ==========================================
while True:
    print("\n==============================")
    print("1. Check specific Drug <-> Disease")
    print("2. Find BEST Drugs for a Disease (Discovery Mode)")
    print("3. Check with Uncertainty (Monte Carlo Dropout)")
    print("4. Search for a name")
    print("5. Run Model Battle (GCN vs GAT vs Random Forest)")
    print("6. Plot Confidence Calibration")
    print("q. Quit")
    choice = input("Select option: ")
    
    if choice == '1':
        d = input("Enter Drug Name: ")
        dis = input("Enter Disease Name: ")
        
        # 1. Get the Probability Score
        predict_single_link(d, dis)
        
        # 2. (NEW) Explain it visually
        visualize_connection(d, dis)
        
    elif choice == '2':
        dis = input("Enter Disease Name (e.g. Alzheimer disease): ")
        find_top_drugs_for_disease(dis)
        
    elif choice == '3':
        d = input("Enter Drug Name: ")
        dis = input("Enter Disease Name: ")
        predict_with_uncertainty(d, dis, num_iterations=100)
        
    elif choice == '4':
        result = interactive_search()
        if result:
            print(f"\n Selected: {result}")
        
    elif choice == '5':
        print("\n  Model Comparison Mode")
        print("GCN will be compared against:")
        print("1. GAT (Graph Attention Network)")
        print("2. Random Forest")
        print("3. Logistic Regression")
        print("4. All of the above")
        model_choice = input("Select comparison model (1-4): ")
        
        if model_choice in ['1', '2', '3', '4']:
            import subprocess
            subprocess.run([sys.executable, 'model_battle.py', model_choice], cwd=os.getcwd())
        else:
            print("❌ Invalid choice")
    
    elif choice == '6':
        plot_confidence_calibration()
        
    elif choice == 'q':
        break