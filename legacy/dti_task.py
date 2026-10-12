"""
Task 2: DTI Prediction using Morgan Fingerprints and ESM-2 embeddings.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import load_config, setup_logger, set_seed
from evaluate import compute_metrics
import urllib.request
import json
import time

logger = setup_logger("dti_task")

def smiles_to_morgan(smiles, radius=2, n_bits=1024):
    """Convert SMILES to Morgan fingerprint (numpy array)."""
    if not isinstance(smiles, str) or not smiles:
        return np.zeros((n_bits,))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def fetch_smiles_from_pubchem(drug_name):
    """
    Fetch SMILES string from PubChem given a drug name.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(drug_name)}/property/CanonicalSMILES/JSON"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception as e:
        return None

class DTIDecoder(nn.Module):
    """Decodes DTI edge by concatenating drug and protein embeddings -> MLP."""
    def __init__(self, drug_dim, prot_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(drug_dim + prot_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, drug_z, prot_z):
        combined = torch.cat([drug_z, prot_z], dim=-1)
        return self.mlp(combined).squeeze(-1)

def extract_primekg_dti(config):
    """
    Extracts DTI edges from PrimeKG and attempts to map DrugBank IDs to SMILES.
    """
    kg_path = config["data"]["raw_kg_path"]
    logger.info(f"Loading raw KG from {kg_path} to extract DTI edges...")
    df = pd.read_csv(kg_path, low_memory=False)
    
    # Filter drug_protein edges
    dti_edges = df[(df["x_type"] == "drug") & (df["y_type"] == "gene/protein")].copy()
    
    # Map drug_id to SMILES (this is a simplified example; for 6000 drugs, batching is needed)
    # We will just save the raw edges for now and mock the mapping if it takes too long.
    unique_drugs = dti_edges[["x_id", "x_name"]].drop_duplicates()
    logger.info(f"Found {len(unique_drugs)} unique drugs in PrimeKG DTI.")
    
    out_path = os.path.join(config["data"]["processed_dir"], "dti_edges.csv")
    dti_edges.to_csv(out_path, index=False)
    logger.info(f"Saved {len(dti_edges)} DTI edges to {out_path}")
    
    return dti_edges

def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["random_seed"])
    
    logger.info("--- Starting Task 2: DTI Preparation ---")
    
    # 1. Extract PrimeKG DTI edges
    extract_primekg_dti(config)
    
    # 2. To fetch TDC datasets:
    logger.info("To train on Davis and KIBA, run this script with PyTDC installed.")
    try:
        from tdc.multi_pred import DTI
        logger.info("PyTDC is installed. Fetching Davis dataset...")
        davis = DTI(name='Davis')
        df_davis = davis.get_data()
        logger.info(f"Davis dataset loaded: {len(df_davis)} interactions.")
        
        # Binarize Davis (pKd >= 7.0 is typically considered active)
        df_davis['Y_bin'] = (df_davis['Y'] >= 7.0).astype(int)
        
        # Example to generate Morgan fingerprints for Davis drugs
        logger.info("Generating Morgan Fingerprints for Davis drugs...")
        unique_smiles = df_davis['Drug'].unique()
        fp_dict = {sm: smiles_to_morgan(sm) for sm in unique_smiles}
        
    except ImportError:
        logger.warning("PyTDC not found. Skipping TDC datasets (Davis/KIBA).")
        logger.warning("Install it using: pip install PyTDC")

if __name__ == "__main__":
    main()
