import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. LOAD DATA ONCE (Global Cache)
print("   ...Loading graph for visualization (this happens once)...")
df = pd.read_csv('kg_clean.csv')

# Create a NetworkX graph (easier for finding paths than PyTorch)
G = nx.from_pandas_edgelist(df, 'x_name', 'y_name')

def visualize_connection(drug, disease):
    print(f"   ...Generating explanation for {drug} -> {disease}...")
    
    # 2. FIND PATHS (The "Why")
    # We look for simple connections: Drug -> Protein -> Disease (2 hops)
    try:
        paths = list(nx.all_shortest_paths(G, source=drug, target=disease))
        # Limit to first 5 paths to keep the image clean
        paths = paths[:5]
    except nx.NetworkXNoPath:
        print("   ❌ No direct path found in the training data.")
        return
    except Exception as e:
        print(f"   ❌ Error finding path: {e}")
        return

    # 3. BUILD SUBGRAPH
    # We collect all nodes involved in these paths
    node_set = set()
    for path in paths:
        node_set.update(path)
    
    subgraph = G.subgraph(node_set)
    
    # 4. DRAW THE GRAPH
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(subgraph, seed=42) # Layout algorithm
    
    # Color Code the Nodes
    color_map = []
    for node in subgraph:
        if node == drug:
            color_map.append('green')  # Drug = Green
        elif node == disease:
            color_map.append('red')    # Disease = Red
        else:
            color_map.append('skyblue') # Intermediate (Proteins) = Blue
            
    nx.draw(subgraph, pos, 
            with_labels=True, 
            node_color=color_map, 
            node_size=2000, 
            font_size=9, 
            font_weight='bold', 
            edge_color='gray', 
            alpha=0.8)
    
    plt.title(f"Mechanism of Action: {drug} vs {disease}")
    
    # Save or Show
    filename = f"explanation_{drug}_{disease}.png".replace(" ", "_")
    plt.savefig(filename)
    print(f"   📸 Graph image saved to: {filename}")
    plt.close() # Close to free memory

# Quick Test
if __name__ == "__main__":
    visualize_connection("Metformin", "type 2 diabetes mellitus")