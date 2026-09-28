# Sequence-Aware GNNs for PPI & DTI Prediction

This repository investigates whether adding protein amino acid sequence information (via ESM-2 language model embeddings) improves Graph Convolutional Network (GCN) link prediction compared to using graph structure alone.

## Project Structure

```
├── data/
│   ├── raw/                  # Downloaded FASTA and PrimeKG kg.csv
│   └── processed/            # Cached ESM-2 embeddings and deduplicated edge lists
├── src/
│   ├── utils.py              # Logging, configuration, and reproducibility (seeding)
│   ├── data_prep.py          # Phase 1: KG edge extraction, UniProt ID mapping, FASTA fetching
│   ├── embeddings.py         # Phase 2: ESM-2 inference and feature extraction
│   ├── dataset.py            # Phase 3: Graph construction and Link Split strategies
│   ├── models.py             # Phase 4: Model definitions (GCN, GAT, MLP)
│   ├── train.py              # Phase 5: Training loops and Ablation Studies
│   ├── evaluate.py           # Evaluation metrics (AUC, F1, Recall, Precision@K)
│   └── dti_task.py           # Phase 7: DTI extension (Davis/KIBA dataset loading and RDKit logic)
├── notebooks/                # Exploratory Data Analysis and visualizations
├── results/                  # Saved metrics.csv and plots
├── legacy/                   # Archived older draft scripts and images
├── config.yaml               # Hyperparameter and model configuration
└── requirements.txt          # Python dependencies
```

## Setup & Installation

1. Create a virtual environment (Python 3.10+):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the PrimeKG dataset (`kg.csv`) is downloaded and placed in the directory specified in `config.yaml` (default: `dataset/kg.csv`).

## Running the Pipeline

### Phase 1: Data Preparation
Extracts PPI edges, maps NCBI Gene IDs to UniProt accessions via the REST API, and downloads FASTA sequences.
```bash
python src/data_prep.py
```

### Phase 2: Sequence Embeddings
Generates 480-dimensional embeddings for all downloaded protein sequences using `facebook/esm2_t12_35M_UR50D`. 
*(Note: This process caches the embeddings to `data/processed/esm2_embeddings.pt`. It may take several hours on CPU, but is very fast on a GPU like Google Colab).*
```bash
python src/embeddings.py
```

### Phase 5: Training and Ablation Studies
Trains and evaluates all feature settings (Structure Only, Sequence Only, Structure + Sequence) across multiple models and graph splitting strategies (Random Link Split and Inductive Cold-Protein Split).
```bash
python src/train.py
```
Results will be saved as a CSV to the `results/` directory.

### Phase 7: DTI Prediction (Extension)
To run the Drug-Target Interaction extension, which utilizes RDKit Morgan fingerprints and TDC (Therapeutics Data Commons) datasets (Davis/KIBA), use the DTI script:
```bash
python src/dti_task.py
```
*(Requires `PyTDC` to be installed).*

## Ablation Study Design

To answer the core research question, we compare three paradigms for link prediction:
1. **Structure Only**: PyG GCN passing messages using randomly initialized or topological node features.
2. **Sequence Only**: A standard MLP evaluating pairwise ESM-2 embeddings without using message passing.
3. **Sequence + Structure**: PyG GCN passing messages across the graph, initialized with ESM-2 embeddings as node features.

Generalization is tested using two splitting methods:
- **Random Split**: Edges are held out randomly across the graph (transductive).
- **Cold-Protein Split**: A set of specific proteins is held out. Edges touching these proteins are strictly evaluated during validation/testing to measure inductive generalization to unseen proteins.
