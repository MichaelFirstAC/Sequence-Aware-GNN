# Drug Repurposing with Graph Neural Networks

A professional machine learning system for discovering new drug-disease relationships using Graph Neural Networks (GCN/GAT), uncertainty quantification via Monte Carlo Dropout, and comprehensive model comparison with MLflow experiment tracking.

## Overview

This project uses **Graph Neural Networks** to predict drug-disease connections by analyzing knowledge graphs. It combines:
- **GCN & GAT models** for link prediction
- **Monte Carlo Dropout** for uncertainty quantification  
- **Model Battle framework** comparing 4 different ML approaches
- **Interactive CLI** with search, visualization, and drug discovery modes
- **Confidence calibration** and pathway explanations

## Features

### 1. **Single Drug-Disease Prediction** (Option 1)
Check the predicted probability between a specific drug and disease, with visual pathway explanations.

```
Enter Drug Name: Ibuprofen
Enter Disease Name: Arthritis
→ Generates explanation graph showing mechanism of action
```

### 2. **Drug Discovery Mode** (Option 2)
Scan all drugs for a disease and get top-K predictions with:
- Model confidence scores
- Top 3 connecting pathways with node types
- PubChem chemical properties (formula, brands, synonyms)

```
Enter Disease: Alzheimer disease
→ Returns ranked list of predicted drugs with explanations
```

### 3. **Uncertainty Quantification** (Option 3)
Monte Carlo Dropout predictions with confidence intervals:
- Mean probability
- Standard deviation
- 95% confidence interval
- Confidence classification (HIGH/MODERATE/LOW)

```
Enter Drug: Aspirin
Enter Disease: Heart disease
→ 100 forward passes with dropout enabled
→ Statistical analysis of predictions
```

### 4. **Interactive Search** (Option 4)
Auto-complete search with suggestions:
- Type partial name (e.g., "ibupro")
- See matching drugs/diseases with types
- Select from filtered results

### 5. **Model Battle** (Option 5)
Compare GCN against multiple baselines:
- **GCN** (Graph Convolutional Network) - Primary model
- **GAT** (Graph Attention Network) - Attention-based variant
- **Random Forest** - Tree-based baseline
- **Logistic Regression** - Linear baseline

Metrics: ROC-AUC, Precision@K, F1-Score, logged to MLflow

### 6. **Confidence Calibration** (Option 6)
Visualize prediction confidence across entire dataset:
- Histogram of confidence distribution
- Cumulative distribution function
- Statistical summary (mean, median, std dev)
- Saved as `confidence_calibration.png`

## Quick Start

### Installation

**Prerequisites:** Python 3.8+, pip, kg dataset

Download the kg.csv file from this link: https://dataverse.harvard.edu/file.xhtml?fileId=6180620&version=2.1

```bash
# Clone repository
git clone <repo-url>
cd REPOSITORY_NAME

# Install dependencies
pip install pandas torch torch-geometric scikit-learn matplotlib networkx mlflow pubchempy
```

### Data Preparation

The system uses `kg_clean.csv` (cleaned knowledge graph):
- ~1.5M drug-disease-protein relationships
- Entities: drugs, diseases, genes/proteins
- Preprocessed and deduplicated

To generate cleaned data from raw `kg.csv`:
```bash
python kg_clean.py
```

### Running the Application

```bash
python run_project.py
```

You'll see the interactive menu:
```
==============================
1. Check specific Drug <-> Disease
2. Find BEST Drugs for a Disease (Discovery Mode)
3. Check with Uncertainty (Monte Carlo Dropout)
4. Search for a name
5. Run Model Battle (GCN vs GAT vs Random Forest)
6. Plot Confidence Calibration
q. Quit
```

## Architecture

### Models

**GCN (Graph Convolutional Network):**
- 2 convolutional layers (64 hidden channels)
- Node embedding → message passing → dot product decoder
- 50 epochs training with Adam optimizer
- Dropout: 0.3 for Monte Carlo uncertainty

**GAT (Graph Attention Network):**
- Multi-head attention mechanism (8 heads)
- Learns which neighbors to focus on
- Same training pipeline as GCN

**Baselines:**
- Random Forest: 100 trees, trained on node embeddings
- Logistic Regression: Linear classifier on embeddings

### Data Pipeline

```
kg_clean.csv
    ↓
Load & Map (Names → IDs)
    ↓
Graph Construction (PyTorch Geometric)
    ↓
80/10/10 Split (Train/Val/Test)
    ↓
Model Training & Evaluation
    ↓
Predictions & Visualizations
```

### Metrics

- **ROC-AUC**: Ranking quality across all thresholds
- **Precision@K**: Top-K drug discovery accuracy
- **Precision, Recall, F1**: Classification at 0.5 threshold
- **Confidence Intervals**: 95% CI from Monte Carlo iterations

## Output Files

Generated during execution:

| File | Description |
|------|-------------|
| `explanation_drug_disease.png` | Pathway visualization between entities |
| `confidence_calibration.png` | Distribution of model confidence |
| `model_comparison.png` | Side-by-side metric comparison (Model Battle) |
| `roc_curves.png` | ROC curves for all models (Model Battle) |
| `uncertainty_*.png` | Monte Carlo prediction distributions |

MLflow tracking:
```bash
mlflow ui  # View at http://localhost:5000
```

## Configuration

Edit in `run_project.py`:

```python
# Training hyperparameters
epochs = 50
hidden_channels = 64
dropout = 0.3
learning_rate = 0.01

# Prediction settings
top_k_drugs = 5  # Number of drugs to recommend
mc_iterations = 100  # Monte Carlo forward passes

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Example Workflow

```python
# 1. Search for a drug
Option 4: Search → "ibuprofen" → Select from matches

# 2. Check against specific disease
Option 1: Ibuprofen + Arthritis → See probability & pathway

# 3. Find top drugs for disease
Option 2: Enter "Arthritis" → Get top 5 predicted drugs with explanations

# 4. Quantify uncertainty
Option 3: Ibuprofen + Arthritis → 100 MC iterations → Confidence interval

# 5. Compare models
Option 5: Run Model Battle (GCN vs GAT vs RF vs LogReg)

# 6. Analyze confidence
Option 6: Plot histogram of all predictions
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Graph Neural Networks | PyTorch Geometric |
| Deep Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, NetworkX |
| Experiment Tracking | MLflow |
| Classical ML | Scikit-learn |
| Chemistry API | PubChem (for drug details) |

## Performance

- **Training time**: ~5-10 minutes (GPU), ~30-60 minutes (CPU)
- **Single prediction**: <100ms
- **Top-K discovery**: ~5-15 seconds (for ~5000 drugs)
- **Full calibration**: ~2-5 minutes (for ~45k drug-disease pairs)

## Troubleshooting

**❌ "kg_clean.csv not found"**
→ Run `python kg_clean.py` first

**❌ PubChem timeout errors**
→ Some drug names may not resolve; gracefully handled with warnings

**❌ CUDA out of memory**
→ Uncomment `df = df.head(100000)` in run_project.py

**❌ Slow performance**
→ Use GPU with CUDA installed; CPU-only mode is 5-10x slower

## Project Structure

```
drug-repurposing/
├── run_project.py           # Main interactive application
├── model_battle.py          # Model comparison framework
├── kg_clean.py              # Data cleaning script
└── README.md                # This file
```

## Dependencies

- pandas, numpy (data processing)
- torch, torch-geometric (GNNs)
- scikit-learn (baselines)
- matplotlib (visualization)
- networkx (graph analysis)
- mlflow (experiment tracking)
- pubchempy (drug properties)

## Metrics Explanation

### Why ROC-AUC instead of Accuracy?
Link prediction datasets are **imbalanced** (mostly negative links). Accuracy would be misleading; ROC-AUC measures ranking quality regardless of threshold.

### Why Precision@K?
For drug discovery, we care about **top predictions**. Precision@5 tells us: "Of the top 5 drugs recommended, how many are actually relevant?"

### What about uncertainty?
Monte Carlo Dropout runs 100 forward passes with dropout enabled at test time. The spread of predictions indicates model confidence—useful for clinical decision-making.

---

**Last Updated:** December 2025  
**Status:** Inactive Development




