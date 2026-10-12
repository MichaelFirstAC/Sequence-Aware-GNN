"""
Classical sequence feature extraction for proteins.
Computes 20-dim Amino Acid Composition and 3-mer (tripeptide) frequencies reduced to 64 dims via PCA.
Total feature dimension: 84.
Caches features to data/processed/classical_features.pt.
"""
import os
import time
import numpy as np
import torch
from Bio import SeqIO
from sklearn.decomposition import PCA
from utils import load_config, setup_logger, set_seed

logger = setup_logger("classical_features")

# Standard 20 amino acids
STANDARD_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

# Map 3-mers (8000 possibilities)
TRIMER_TO_IDX = {}
idx = 0
for a1 in STANDARD_AA:
    for a2 in STANDARD_AA:
        for a3 in STANDARD_AA:
            TRIMER_TO_IDX[a1 + a2 + a3] = idx
            idx += 1

def compute_aa_composition(seq: str) -> np.ndarray:
    """Computes 20-dimensional amino acid composition vector."""
    counts = np.zeros(20, dtype=np.float32)
    valid_len = 0
    for char in seq:
        if char in AA_TO_IDX:
            counts[AA_TO_IDX[char]] += 1
            valid_len += 1
    if valid_len > 0:
        counts /= valid_len
    return counts

def compute_trimer_frequencies(seq: str) -> np.ndarray:
    """Computes 8000-dimensional 3-mer frequency vector."""
    counts = np.zeros(8000, dtype=np.float32)
    n = len(seq)
    if n < 3:
        return counts
    
    valid_trimers = 0
    for i in range(n - 2):
        trimer = seq[i:i+3]
        if trimer in TRIMER_TO_IDX:
            counts[TRIMER_TO_IDX[trimer]] += 1
            valid_trimers += 1
            
    if valid_trimers > 0:
        counts /= valid_trimers
    return counts

def extract_classical_features(fasta_path: str, output_path: str, pca_dim: int = 64, random_seed: int = 42):
    """
    Extracts AA composition (20-dim) and PCA-reduced 3-mer frequencies (64-dim)
    for all proteins in fasta_path and saves to output_path.
    """
    set_seed(random_seed)
    logger.info(f"Loading sequences from {fasta_path}...")
    
    uniprot_ids = []
    aa_comp_list = []
    trimer_list = []
    
    t0 = time.time()
    for record in SeqIO.parse(fasta_path, "fasta"):
        parts = record.id.split("|")
        uniprot_id = parts[1] if len(parts) >= 2 else record.id
        seq = str(record.seq).upper()
        
        uniprot_ids.append(uniprot_id)
        aa_comp_list.append(compute_aa_composition(seq))
        trimer_list.append(compute_trimer_frequencies(seq))
        
    num_proteins = len(uniprot_ids)
    logger.info(f"Loaded and computed raw frequencies for {num_proteins} proteins in {time.time() - t0:.2f}s.")
    
    aa_comp_matrix = np.array(aa_comp_list, dtype=np.float32)
    trimer_matrix = np.array(trimer_list, dtype=np.float32)
    
    logger.info(f"Fitting PCA(n_components={pca_dim}) on 3-mer frequencies matrix {trimer_matrix.shape}...")
    pca = PCA(n_components=pca_dim, random_state=random_seed)
    trimer_pca = pca.fit_transform(trimer_matrix)
    
    var_exp = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA explained variance ratio for {pca_dim} components: {var_exp * 100:.2f}%")
    
    # Concatenate 20-dim AA composition + 64-dim 3-mer PCA -> 84-dim feature vector
    combined_features = np.hstack([aa_comp_matrix, trimer_pca]).astype(np.float32)
    logger.info(f"Combined feature shape: {combined_features.shape}")
    
    # Verify no NaN or Inf
    assert not np.isnan(combined_features).any(), "NaN found in classical features!"
    assert not np.isinf(combined_features).any(), "Inf found in classical features!"
    
    # Build dictionary
    classical_dict = {}
    for i, u_id in enumerate(uniprot_ids):
        classical_dict[u_id] = torch.from_numpy(combined_features[i])
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(classical_dict, output_path)
    logger.info(f"Successfully cached classical features for {len(classical_dict)} proteins to {output_path}")
    
    return classical_dict

if __name__ == "__main__":
    config = load_config("config.yaml")
    fasta_path = os.path.join(config["data"]["raw_dir"], "protein_sequences.fasta")
    output_path = os.path.join(config["data"]["processed_dir"], "classical_features.pt")
    extract_classical_features(fasta_path, output_path, pca_dim=64, random_seed=config["project"]["random_seed"])
