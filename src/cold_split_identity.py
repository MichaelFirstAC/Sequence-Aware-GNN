"""
Cold-Protein Split Sequence Identity Check.
Evaluates sequence similarity between unseen (cold test) proteins and training proteins
using Bio.Align.PairwiseAligner (Smith-Waterman with BLOSUM62).
Reports distribution of maximum sequence identity per test protein.
"""
import os
import time
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from concurrent.futures import ProcessPoolExecutor
from utils import load_config, setup_logger, set_seed

logger = setup_logger("cold_split_identity")

def load_fasta_dict(fasta_path):
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        parts = record.id.split("|")
        uniprot_id = parts[1] if len(parts) >= 2 else record.id
        # Replace non-standard characters with 'X'
        s = str(record.seq).upper()
        clean_s = "".join([c if c in "ARNDCQEGHILKMFPSTWYVBZX*" else "X" for c in s])
        seqs[uniprot_id] = clean_s
    return seqs

def align_single_test_protein(test_item, train_seqs_list):
    """
    Computes local alignment (Smith-Waterman) against training sequence sample.
    Returns (test_id, max_score, max_identity, best_train_id).
    """
    test_id, test_seq = test_item
    
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    best_score = -1e9
    best_train_id = None
    best_train_seq = None
    
    for train_id, train_seq in train_seqs_list:
        try:
            score = aligner.score(test_seq, train_seq)
            if score > best_score:
                best_score = score
                best_train_id = train_id
                best_train_seq = train_seq
        except Exception:
            continue
            
    if best_train_seq is not None:
        try:
            alns = aligner.align(test_seq, best_train_seq)
            aln = alns[0]
            identities = aln.counts().identities
            aln_len = aln.shape[1]
            identity = (identities / aln_len) if aln_len > 0 else 0.0
        except Exception:
            identity = 0.0
    else:
        identity = 0.0
        
    return test_id, best_score, identity, best_train_id

def run_cold_split_check():
    config = load_config("config.yaml")
    seed = config["project"]["random_seed"]
    set_seed(seed)
    
    fasta_path = os.path.join(config["data"]["raw_dir"], "protein_sequences.fasta")
    edges_path = os.path.join(config["data"]["processed_dir"], "ppi_edges_uniprot.csv")
    
    logger.info("Loading protein sequences and cold split partition...")
    seq_dict = load_fasta_dict(fasta_path)
    
    edges_df = pd.read_csv(edges_path)
    unique_proteins = sorted(list(set(edges_df["uniprot1"]).union(set(edges_df["uniprot2"]))))
    num_nodes = len(unique_proteins)
    
    # Exact cold split partition as defined in dataset.py
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)
    
    train_indices = indices[:train_end]
    test_indices = indices[val_end:]
    
    train_proteins = [unique_proteins[i] for i in train_indices if unique_proteins[i] in seq_dict]
    test_proteins = [unique_proteins[i] for i in test_indices if unique_proteins[i] in seq_dict]
    
    logger.info(f"Available with sequences: {len(train_proteins)} train proteins, {len(test_proteins)} test proteins.")
    
    # Random sample of ~500 test proteins (cold)
    np.random.seed(seed)
    n_sample_test = min(500, len(test_proteins))
    sample_test_ids = np.random.choice(test_proteins, size=n_sample_test, replace=False)
    
    # Random sample of training proteins for comparison (250 proteins)
    n_sample_train = min(250, len(train_proteins))
    sample_train_ids = np.random.choice(train_proteins, size=n_sample_train, replace=False)
    
    test_items = [(p_id, seq_dict[p_id]) for p_id in sample_test_ids]
    train_items = [(p_id, seq_dict[p_id]) for p_id in sample_train_ids]
    
    logger.info(f"Comparing {n_sample_test} cold test proteins against {n_sample_train} train proteins using Smith-Waterman (BLOSUM62)...")
    logger.info(f"Total alignment evaluations: {n_sample_test * n_sample_train}")
    
    t0 = time.time()
    results = []
    
    max_workers = min(7, os.cpu_count() or 4)
    logger.info(f"Using {max_workers} parallel workers...")
    
    # Batch processing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(align_single_test_protein, item, train_items) for item in test_items]
        for f in futures:
            results.append(f.result())
            
    elapsed = time.time() - t0
    logger.info(f"All alignments completed in {elapsed:.2f}s ({elapsed/n_sample_test:.3f}s per test protein).")
    
    df_results = pd.DataFrame(results, columns=["test_uniprot_id", "max_score", "max_sequence_identity", "best_train_uniprot_id"])
    
    # Distribution statistics
    identities = df_results["max_sequence_identity"].values
    mean_id = float(np.mean(identities))
    std_id = float(np.std(identities))
    median_id = float(np.median(identities))
    p25 = float(np.percentile(identities, 25))
    p75 = float(np.percentile(identities, 75))
    min_id = float(np.min(identities))
    max_id = float(np.max(identities))
    
    pct_under_30 = float(np.mean(identities < 0.30) * 100)
    pct_under_40 = float(np.mean(identities < 0.40) * 100)
    pct_over_70 = float(np.mean(identities >= 0.70) * 100)
    
    logger.info("=" * 60)
    logger.info("COLD-SPLIT SEQUENCE IDENTITY DISTRIBUTION:")
    logger.info(f"  Sampled Test Proteins: {n_sample_test}")
    logger.info(f"  Mean Maximum Identity:   {mean_id*100:.2f}% ± {std_id*100:.2f}%")
    logger.info(f"  Median Maximum Identity: {median_id*100:.2f}%")
    logger.info(f"  25th - 75th Percentile:  [{p25*100:.2f}%, {p75*100:.2f}%]")
    logger.info(f"  Min / Max Identity:      {min_id*100:.2f}% / {max_id*100:.2f}%")
    logger.info(f"  Proteins < 30% identity: {pct_under_30:.1f}% (Twilight zone / non-homologous)")
    logger.info(f"  Proteins < 40% identity: {pct_under_40:.1f}%")
    logger.info(f"  Proteins >= 70% identity (close homologs): {pct_over_70:.1f}%")
    logger.info("=" * 60)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    out_csv = "results/cold_split_identities.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Saved identity distribution to {out_csv}")
    
    stats_dict = {
        "Sample_Size": n_sample_test,
        "Mean_Identity": mean_id,
        "Std_Identity": std_id,
        "Median_Identity": median_id,
        "P25": p25,
        "P75": p75,
        "Min": min_id,
        "Max": max_id,
        "Pct_Under_30": pct_under_30,
        "Pct_Under_40": pct_under_40,
        "Pct_Over_70": pct_over_70,
    }
    pd.DataFrame([stats_dict]).to_csv("results/cold_split_identity_summary.csv", index=False)
    return stats_dict, df_results

if __name__ == "__main__":
    run_cold_split_check()
