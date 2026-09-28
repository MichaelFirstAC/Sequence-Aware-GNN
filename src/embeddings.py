"""
Generate ESM-2 embeddings for protein sequences.
Reads FASTA sequences, passes them through ESM-2, and caches embeddings to disk.
"""
import os
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel
from utils import load_config, setup_logger, set_seed
from tqdm import tqdm

logger = setup_logger("embeddings")

def get_device(config_device):
    if config_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Add MPS support for Mac, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_sequences(fasta_path):
    """Load sequences from a FASTA file into a dict of {uniprot_id: sequence}."""
    sequences = {}
    # UniProt FASTA headers look like: >sp|P12345|NAME_HUMAN Full Name...
    # Or >tr|A0A024RBG1|...
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract UniProt ID from the header
        parts = record.id.split("|")
        if len(parts) >= 2:
            uniprot_id = parts[1]
            sequences[uniprot_id] = str(record.seq)
        else:
            sequences[record.id] = str(record.seq)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences

def generate_embeddings(sequences, config):
    """
    Generate ESM-2 embeddings. 
    Truncates to config max_length (e.g. 1022) to avoid OOM/length errors.
    """
    model_name = config["esm2"]["model_name"]
    max_length = config["esm2"]["max_length"]
    batch_size = config["esm2"]["batch_size"]
    device = get_device(config["project"]["device"])
    
    logger.info(f"Loading ESM-2 model ({model_name}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    embeddings = {}
    uniprot_ids = list(sequences.keys())
    
    logger.info(f"Generating embeddings for {len(uniprot_ids)} proteins...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="ESM-2 Embeddings"):
            batch_ids = uniprot_ids[i:i+batch_size]
            batch_seqs = [sequences[uid] for uid in batch_ids]
            
            # Truncate sequences to max_length (excluding special tokens)
            batch_seqs = [seq[:max_length] for seq in batch_seqs]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length+2)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # We want the mean over the sequence length, ignoring padding
            attention_mask = inputs["attention_mask"]
            # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
            last_hidden = outputs.last_hidden_state
            
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = mask_expanded.sum(1)
            # Avoid division by zero
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            mean_embeddings = mean_embeddings.cpu()
            
            for j, uid in enumerate(batch_ids):
                embeddings[uid] = mean_embeddings[j].clone()
                
    return embeddings

def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["random_seed"])
    
    fasta_path = os.path.join(config["data"]["raw_dir"], "protein_sequences.fasta")
    out_path = os.path.join(config["data"]["processed_dir"], "esm2_embeddings.pt")
    
    if os.path.exists(out_path):
        logger.info(f"Embeddings already exist at {out_path}. Skipping.")
        return
        
    if not os.path.exists(fasta_path):
        logger.error(f"FASTA file not found at {fasta_path}. Run data_prep.py first.")
        return
        
    sequences = load_sequences(fasta_path)
    embeddings = generate_embeddings(sequences, config)
    
    torch.save(embeddings, out_path)
    logger.info(f"Saved embeddings to {out_path} (dim: {list(embeddings.values())[0].shape})")
    
if __name__ == "__main__":
    main()
