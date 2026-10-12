# Sequence-Aware Graph Neural Networks for Protein-Protein Interaction Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository investigates the central question: **Does incorporating protein amino acid sequence information (via deep ESM-2 language model embeddings and classical biophysical features) improve Graph Convolutional Network (GCN) link prediction compared to using topological graph structure alone, particularly under out-of-distribution generalization to unseen ("cold") target proteins?**

For an overview of all course requirement milestones, see [**`roadmap.md`**](roadmap.md).

---

## 🔬 Core Findings & Ablation Results

We evaluate link prediction across two rigorous regimes:
1. **Transductive Random Split (70/15/15)**: Edges are randomly held out; all proteins are visible in the training graph.
2. **Inductive Cold-Protein Split**: 15% of proteins are completely withheld from the training graph. Validation and test sets strictly measure generalization to unseen proteins.

### Empirical Results (from `results/metrics.csv`):

| Model | Feature Representation | Split Strategy | ROC-AUC | F1-Score | Precision | Recall | Prec@10 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **GCN** | Structure Only (Random Initialized) | Random Split | 0.8782 | 0.7104 | 0.5584 | 0.9761 | 1.0 |
| **MLP** | Sequence Only (ESM-2, No Graph) | Random Split | 0.7737 | 0.6974 | 0.7093 | 0.6858 | 1.0 |
| **GCN** | **ESM-2 Sequence + Structure** | **Random Split** | **0.8871** | **0.6667** | 0.5000 | 1.0000 | 1.0 |
| **GCN** | **Classical Sequence + Structure** | **Random Split** | **0.8896** | **0.6667** | 0.5000 | 1.0000 | 1.0 |
| **RandomForest** | Classical Sequence (Pairwise) | Random Split | 0.8204 | 0.7325 | 0.7579 | 0.7086 | 1.0 |
| **LogisticRegression** | Classical Sequence (Pairwise) | Random Split | 0.6791 | 0.6380 | 0.6305 | 0.6456 | 1.0 |
| | | | | | | | |
| **GCN** | Structure Only (Random Initialized) | Cold-Protein Split | 0.7628 | 0.6539 | 0.4995 | 0.9466 | 0.6 |
| **MLP** | Sequence Only (ESM-2, No Graph) | Cold-Protein Split | 0.7589 | 0.6757 | 0.7013 | 0.6518 | 1.0 |
| **GCN** | **ESM-2 Sequence + Structure** | **Cold-Protein Split** | **0.7857** | **0.6667** | 0.5000 | 1.0000 | 0.8 |
| **GCN** | **Classical Sequence + Structure** | **Cold-Protein Split** | **0.8290** | **0.6931** | 0.5337 | 0.9883 | 0.9 |
| **RandomForest** | Classical Sequence (Pairwise) | Cold-Protein Split | 0.6691 | 0.5089 | 0.6988 | 0.4001 | 1.0 |
| **LogisticRegression** | Classical Sequence (Pairwise) | Cold-Protein Split | 0.6551 | 0.6091 | 0.6209 | 0.5978 | 1.0 |

### Key Scientific Insights:
* **Inductive Generalization Boost**: When testing unseen proteins (Cold Split), Structure-Only GCN degrades to `0.7628` AUC. Injecting sequence information restores inductive capability, boosting AUC to **`0.7857` (ESM-2)** and **`0.8290` (Classical Features)**.
* **Cold-Split Homology Verification**: Using Smith-Waterman local alignment (`BLOSUM62`), we audited 125,000 pairwise alignments between held-out cold proteins and training proteins. The mean maximum identity is **23.37% ± 6.88%**, with **91.2% of test proteins sharing < 30% sequence identity** with any training protein (deep within the non-homologous twilight zone), confirming strict out-of-distribution evaluation.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                  # Downloaded FASTA sequences (17,328 human proteins)
│   └── processed/            # Cached ESM-2 (480-dim) & Classical (84-dim) feature tensors
├── dataset/                  # Primary interactome (PrimeKG kg.csv)
├── src/
│   ├── utils.py              # Configuration loading, logging, and seed setting (seed=42)
│   ├── data_prep.py          # Phase 1: PrimeKG PPI extraction, NCBI to UniProt mapping
│   ├── embeddings.py         # Phase 2: ESM-2 deep transformer inference
│   ├── classical_features.py # Phase 6: 20-dim AA composition + 64-dim 3-mer PCA
│   ├── cold_split_identity.py# Phase 6: Smith-Waterman BLOSUM62 homology audit
│   ├── dataset.py            # Phase 3: PyG graph construction, Random & Cold splits
│   ├── models.py             # Phase 4: GCN, GAT, and MLP baseline architectures
│   ├── train.py              # Phase 5: Initial GNN training and ablation loop
│   ├── classical_ablation.py # Phase 6: Classical GCN, RF, and Logistic Regression ablation
│   └── evaluate.py           # Metric computations (ROC-AUC, F1, Precision@10)
├── results/
│   ├── metrics.csv           # Full experimental ablation log
│   ├── cold_split_identities.csv # Raw maximum pairwise sequence identities
│   └── figures/              # Publication figures (300 DPI)
├── legacy/                   # Archived prototype scripts
├── roadmap.md                # 10-Phase course milestone roadmap
├── config.yaml               # Central hyperparameter and path configuration
└── requirements.txt          # Python dependency specifications
```

---

## 🚀 Setup & Installation

### 1. Environment Setup
Create and activate a virtual environment (Python 3.10+):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Placement
Ensure `kg.csv` (PrimeKG) is located in `dataset/kg.csv` as specified in `config.yaml`.

---

## 🏃 Execution Pipeline

### Phase 1: Data Preparation & ID Mapping
Extracts deduplicated PPI edges (316,034 edges across 17,328 proteins), maps NCBI Gene IDs to UniProt accessions, and downloads FASTA sequences:
```bash
python src/data_prep.py
```

### Phase 2: ESM-2 Sequence Embeddings
Generates 480-dimensional mean-pooled embeddings using `facebook/esm2_t12_35M_UR50D`:
```bash
python src/embeddings.py
```
*(Precomputed and cached to `data/processed/esm2_embeddings.pt`).*

### Phase 5: Baseline GNN Training & Ablation
Runs the initial Structure vs. ESM-2 vs. Combined comparison:
```bash
python src/train.py
```

### Phase 6: Classical Sequence Features & Homology Audit
1. **Extract Classical Features** (20-dim AA composition + 64-dim 3-mer PCA):
   ```bash
   python src/classical_features.py
   ```
2. **Run Smith-Waterman Homology Audit** (multiprocessed BLOSUM62 alignments):
   ```bash
   python src/cold_split_identity.py
   ```
3. **Run Classical Feature Ablation** (GCN, Random Forest, Logistic Regression):
   ```bash
   python src/classical_ablation.py
   ```

---

## ⚠️ Biological & Computational Limitations

1. **Unlabeled Negative Interactions**: Negative links are sampled randomly from non-interacting pairs (1:1 ratio). In biological interactomes, some sampled "negatives" may represent true, uncharacterized interactions that have not yet been experimentally assayed.
2. **Static Interactome Representation**: The PPI network is modeled as a static graph, omitting tissue-specific expression, subcellular localization, temporal dynamics, and post-translational modifications (PTMs).
3. **In-Silico Evaluation**: While validated on rigorous cold-split partitions and cross-referenced with Smith-Waterman alignment scores, computational predictions require downstream biophysical assays (e.g., surface plasmon resonance or co-immunoprecipitation) for wet-lab confirmation.
