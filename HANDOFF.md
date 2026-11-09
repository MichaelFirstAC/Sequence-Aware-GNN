# Project Handoff Guide: Sequence-Aware GNN for PPI Link Prediction

Welcome to the project! This document outlines the current state of the codebase, essential constraints, how the data is handled, and what needs to be done for Phases 7 through 10.

---

## 1. Core Research Question & Hypothesis
> **Question:** Does incorporating protein amino acid sequence information (deep ESM-2 transformer embeddings and classical physicochemical features) improve Graph Convolutional Network (GCN) link prediction compared to using topological graph structure alone, particularly under out-of-distribution generalization to unseen ("cold") target proteins?

---

## 2. Project Status Overview

| Phase | Description | Status | Notes |
|:---|:---|:---:|:---|
| **Phase 0–4** | Project scaffolding, ID mapping, ESM-2 inference, models | `COMPLETED` | Architecture: GCN, GAT, sequence-only MLP baseline |
| **Phase 5** | Baseline GNN training & ablation loop | `COMPLETED (PROVISIONAL)` | Addressed split mismatch bug in code; needs 5-seed rerun in Phase 7 |
| **Phase 6** | Classical features (AA comp + 3-mer PCA) & Smith-Waterman audit | `COMPLETED` | 91.2% cold proteins in twilight zone (<30% id). Classical GCN cold AUC = 0.829 |
| **Phase 7** | Network analysis, statistics & 5-seed Wilcoxon tests | `READY TO START` | Next immediate task! |
| **Phase 8** | Cross-dataset validation on STRING v12.0 | `PENDING` | Requires downloading external physical PPI dataset |
| **Phase 9** | Publication figures (300 DPI) & IEEE final tables | `PENDING` | Output to `results/figures/` and `results/final_tables.md` |
| **Phase 10**| Packaging (`run_all.ipynb`), reproducible audit | `PENDING` | Master notebook must run end-to-end in <10 mins on CPU |

---

## 3. Strict Rules & Constraints (Course Requirements)
1. **DO NOT recompute ESM-2 embeddings**: The 480-dim embeddings for all 17,328 proteins took ~4 hours on CPU and are cached in `data/processed/esm2_embeddings.pt`. Always load from this file.
2. **Deterministic Seed (42)**: Global seed must remain 42 across PyTorch, NumPy, and Python for all single-seed operations.
3. **Honest Results (No Fabrication)**: Print real metrics, handle failures transparently.
4. **Sanity Checks**: Always perform a quick sanity check after completing each phase before proceeding.
5. **No Broken Live Runs**: Course rule: If `run_all.ipynb` crashes during live demo, score = 0 points. Test thoroughly with pre-cached tensors!

---

## 4. Known Technical Nuance: The Phase 5 Shared-Split Fix
In the initial exploratory run of Phase 5, the random edge split and cold-protein split were sampled independently for Structure-Only vs ESM-2 models, causing slight edge discrepancies. 

**This has already been fixed in `src/dataset.py`:**
- Use `build_splits(edges_path, seed)`: Generates the exact splits **once**.
- Use `with_features(base_split, x)`: Swaps in the feature matrix (`x_struct`, `x_esm`, `x_classical`) without altering graph topology, held-out edges, or negative pairs.
- In **Phase 7**, when evaluating over 5 seeds (`seeds: [42, 43, 44, 45, 46]`), ensure `build_splits()` is called once per seed to yield matched pairs for the paired Wilcoxon signed-rank test.

---

## 5. Next Steps Checklist

### 🔹 Phase 7: Network Statistics & Significance Testing
- **Script to create**: `src/network_analysis.py`
- Compute graph statistics for PrimeKG PPI:
  - Node & edge count
  - Degree distribution (plot and save to `results/figures/degree_distribution.png` on log-log scale)
  - Mean, median, std of degree
  - Average clustering coefficient (`nx.average_clustering`)
  - Connected components & size of largest connected component
  - Top-10 hub proteins (lookup gene symbols via FASTA headers in `data/raw/protein_sequences.fasta`)
- Multi-seed benchmark:
  - Evaluate main models (Structure-only GCN, ESM-2 GCN, Classical GCN, RF) across `seeds: [42, 43, 44, 45, 46]`.
  - Save `mean ± std` to `results/multiseed_metrics.csv`.
  - Run paired Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) comparing Structure-only vs Structure+ESM-2 AUC; report exact p-values.

### 🔹 Phase 8: Cross-Dataset Validation (STRING Physical Interactome)
- Download STRING v12.0 human physical links (`taxon 9606`, combined score $\ge 700$) and sequences.
- Map STRING Ensembl IDs (`9606.ENSP...`) to UniProt accessions.
- **Decision point**: Filter to proteins shared with PrimeKG to reuse cached ESM-2 embeddings instantly without hours of re-computation.
- Run the identical ablation pipeline on STRING to test generalizability across independent interactomes.

### 🔹 Phase 9: Publication Figures & Reporting
- Generate 300 DPI PNG figures in `results/figures/`:
  - `roc_curves.png`
  - `ablation_barchart.png`
  - `degree_distribution.png`
  - `cold_split_identity_hist.png`
- Generate `results/final_tables.md` formatted for IEEE double-column format.

### 🔹 Phase 10: Master Packaging & Audit
- Create `run_all.ipynb` executing top-to-bottom in <10 mins on CPU using cached files.
- Freeze `requirements.txt` with exact pinned versions.
- Run file size audit to ensure no repository files exceed GitHub's 100 MB limit.
