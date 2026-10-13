# Project Roadmap: Sequence-Aware GNN for PPI & DTI Prediction

> **Research Question:** Does incorporating protein amino acid sequence information (ESM-2 embeddings & classical sequence features) improve Graph Neural Network link prediction compared to using graph structure alone, especially under out-of-distribution generalization (unseen proteins)?

---

## Progress Overview

| Phase | Description | Focus / Course Requirement | Status |
|:---|:---|:---|:---:|
| **Phase 0** | Project Scaffolding & Environment Setup | Reproducibility, Config, Seed 42 | `COMPLETED` |
| **Phase 1** | PPI Graph Extraction & ID Mapping | PrimeKG extraction, NCBI to UniProt mapping | `COMPLETED` |
| **Phase 2** | ESM-2 Sequence Embeddings | Deep protein language model features (480-dim) | `COMPLETED` |
| **Phase 3** | Graph Splitting & Leakage Prevention | Random Link Split (70/15/15) & Cold-Protein Split | `COMPLETED` |
| **Phase 4** | Model Architectures | GCN, GAT, and Sequence-only MLP baseline | `COMPLETED` |
| **Phase 5** | Baseline Training & Initial Ablation | Graph-only vs ESM-2 vs Combined | `COMPLETED` |
| **Phase 6** | Classical Sequence Features & Alignment | Sequence analysis (Biopython AA comp, 3-mer PCA, Smith-Waterman) | `COMPLETED` |
| **Phase 7** | Network Statistics & Significance Testing | Graph theory (degree dist, hubs, clustering) & 5-seed Wilcoxon tests | `READY FOR APPROVAL` |
| **Phase 8** | Cross-Dataset Validation (STRING / HuRI) | Inductive evaluation on secondary physical PPI dataset | `PENDING` |
| **Phase 9** | Publication Figures & Report Tables | 300 DPI figures & IEEE summary tables | `PENDING` |
| **Phase 10**| Packaging, Reproducibility & Auditing | `run_all.ipynb` (<10 min CPU), file size audit, clean README | `PENDING` |

---

## Detailed Phase Breakdown

### [x] Phase 0: Scaffolding & Configuration
- Set up directory layout (`src/`, `data/raw/`, `data/processed/`, `results/`).
- Centralize hyperparameter configuration in `config.yaml`.
- Enforce deterministic global seed (`seed = 42`) across PyTorch, NumPy, and Python.

### [x] Phase 1: Data Preparation & UniProt Mapping
- Filtered PrimeKG (`dataset/kg.csv`) for human `protein_protein` interactions.
- Extracted 316,034 deduplicated, undirected PPI edges across 17,328 unique proteins.
- Resolved NCBI Gene IDs to UniProt accessions and retrieved FASTA amino acid sequences (`data/raw/protein_sequences.fasta`).

### [x] Phase 2: ESM-2 Deep Language Model Embeddings
- Computed 480-dimensional mean-pooled representations from `facebook/esm2_t12_35M_UR50D`.
- Safely cached all 17,328 protein tensors to `data/processed/esm2_embeddings.pt`.
- *Rule: Cached embeddings must never be recomputed.*

### [x] Phase 3: Graph Construction & Split Partitions
- Created strict message-passing separation preventing test edge leakage.
- **Random Split**: Transductive random link holdout (70% train, 15% val, 15% test).
- **Cold-Protein Split**: Inductive holdout of 15% of entire proteins to simulate unseen target proteins.

### [x] Phase 4: Model Implementations
- 2-layer `GCNModel` with dot-product decoder.
- 2-layer `GATModel` with multi-head attention.
- Sequence-only `MLPModel` operating purely on pairwise embeddings without message passing.

### [x] Phase 5: Initial Ablation Study
- Benchmarked Structure-Only (GCN), Sequence-Only (MLP), and Combined (GCN + ESM-2).
- Key finding: Sequence features improve Random Split AUC from 0.878 to 0.887, and Cold-Protein Split AUC from 0.762 to 0.785.

---

### [x] Phase 6: Classical Sequence Features & Homology Check
- [x] **Biopython Feature Extraction**:
  - 20-dimensional standard amino acid composition.
  - 8,000-dimensional 3-mer (tripeptide) frequency distribution, reduced to 64 dimensions via PCA (15.4% explained variance).
  - Total 84-dimensional representation cached to `data/processed/classical_features.pt`.
- [x] **Cold-Split Homology & Identity Audit**:
  - Performed 125,000 Smith-Waterman local alignments (`BLOSUM62`) comparing 500 cold test proteins against 250 training proteins.
  - Results: Mean max identity = **23.37% ± 6.88%**; **91.2% of test proteins have < 30% sequence identity** (confirming strict out-of-distribution evaluation in the twilight zone).
  - Saved raw distribution to `results/cold_split_identities.csv`.
- [x] **Ablation Expansion**:
  - GCN (Classical + Structure): Random Split AUC **0.8896**, Cold Split AUC **0.8290**.
  - Random Forest baseline (Classical pairwise concatenation): Random Split AUC **0.8204**, Cold Split AUC **0.6691**.
  - Logistic Regression baseline (Classical pairwise concatenation): Random Split AUC **0.6791**, Cold Split AUC **0.6551**.
  - Evaluated under both Random Split and Cold-Protein Split; results recorded in `results/metrics.csv`.

---

### [ ] Phase 7: Network Analysis & Statistical Significance Testing
- **Topological & Network Characterization**:
  - Compute node and edge counts, mean/median/std degree.
  - Generate degree distribution plot on a log-log scale.
  - Compute average clustering coefficient.
  - Determine connected components and size of the giant component.
  - Identify and name the top-10 hub proteins.
- **Multi-Seed Robustness & Significance**:
  - Run main models across 5 distinct random seeds (`42, 43, 44, 45, 46`).
  - Report metrics as `Mean ± Std`.
  - Conduct Wilcoxon signed-rank paired significance tests comparing Structure-Only vs Structure + Sequence on test AUC.
  - Report exact p-values to validate scientific claim.

---

### [ ] Phase 8: Cross-Dataset Validation (STRING / HuRI)
- Implement loader for STRING human physical PPIs (`taxon: 9606`, `combined_score >= 700`).
- Fallback support for HuRI (Human Reference Interactome) if STRING download fails.
- Configurable switch in `config.yaml` (`dataset.dataset_name: "primekg"` vs `"string"`).
- Run full ablation pipeline on STRING to test whether findings replicate across independent biological datasets.

---

### [ ] Phase 9: Publication Figures & Reporting Tables
- **High-Resolution Figures (300 DPI, `results/figures/`)**:
  - Combined ROC Curves for Random and Cold-Protein Splits.
  - Grouped Ablation Bar Chart (Structure-Only vs Classical vs ESM-2 vs Combined).
  - Network Degree Distribution Plot (Log-Log).
  - Cold-Split Sequence Identity Distribution Histogram.
- **Reporting Artifacts (`results/final_tables.md`)**:
  - Format IEEE-ready LaTeX/Markdown tables comparing models, feature sets, split strategies, runtime, and p-values.

---

### [ ] Phase 10: Packaging, Verification & Reproducibility Audit
- **`run_all.ipynb`**:
  - Single executable Jupyter notebook executing top-to-bottom.
  - Loads cached data/embeddings and reproduces all figures/tables in **under 10 minutes on CPU**.
- **Documentation & Scripts**:
  - Automated download scripts for PrimeKG, UniProt, and STRING.
  - Fully pinned `requirements.txt`.
  - Comprehensive `README.md` detailing methodology, runtime expectations, and biological limitations (random negative sampling, lack of wet-lab validation, static interactome).
- **Workspace Audit**:
  - Clean temporary cache files.
  - Print full file inventory with byte sizes, verifying GitHub 100 MB file limit compliance.
