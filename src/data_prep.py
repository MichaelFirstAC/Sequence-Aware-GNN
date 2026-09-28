"""
Data Preparation for Sequence-Aware GNN
1. Extracts PPI edges from PrimeKG
2. Removes duplicates and self-loops
3. Maps NCBI Gene IDs to UniProt accessions
4. Downloads FASTA sequences
"""
import os
import time
import requests
import pandas as pd
from utils import load_config, setup_logger, set_seed

logger = setup_logger("data_prep")

def fetch_uniprot_mapping(gene_ids):
    """
    Maps NCBI Gene IDs to UniProt accessions using UniProt's ID Mapping API.
    Returns a dictionary of {gene_id: uniprot_id}.
    """
    url_run = "https://rest.uniprot.org/idmapping/run"
    url_status = "https://rest.uniprot.org/idmapping/status"
    url_results = "https://rest.uniprot.org/idmapping/results"

    # 1. Submit job
    payload = {
        "from": "GeneID",
        "to": "UniProtKB",
        "ids": ",".join(gene_ids)
    }
    logger.info(f"Submitting ID mapping job for {len(gene_ids)} IDs...")
    response = requests.post(url_run, data=payload)
    response.raise_for_status()
    job_id = response.json()["jobId"]
    logger.info(f"Job ID: {job_id}")

    # 2. Wait for completion
    while True:
        status_resp = requests.get(f"{url_status}/{job_id}")
        status_resp.raise_for_status()
        status_data = status_resp.json()
        if "jobStatus" in status_data:
            if status_data["jobStatus"] == "FINISHED":
                break
            elif status_data["jobStatus"] in ["FAILED", "ERROR"]:
                raise Exception(f"Job failed: {status_data}")
        elif "results" in status_data:
            break
        logger.info("Waiting for ID mapping to complete...")
        time.sleep(5)

    # 3. Get results (stream format for large results)
    logger.info("Fetching mapping results...")
    result_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}?format=tsv"
    result_resp = requests.get(result_url)
    result_resp.raise_for_status()
    
    lines = result_resp.text.strip().split("\n")
    header = lines[0].split("\t")
    
    mapping = {}
    unmapped = set(gene_ids)
    
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            gene_id = parts[0]
            uniprot_id = parts[1]
            if gene_id not in mapping:
                mapping[gene_id] = uniprot_id
                if gene_id in unmapped:
                    unmapped.remove(gene_id)
                    
    logger.info(f"Mapped {len(mapping)} out of {len(gene_ids)} genes.")
    if unmapped:
        logger.warning(f"{len(unmapped)} genes could not be mapped.")
        
    return mapping

def download_fasta(uniprot_ids, output_path):
    """
    Downloads FASTA sequences for a list of UniProt IDs in chunks.
    """
    logger.info(f"Downloading FASTA sequences for {len(uniprot_ids)} proteins...")
    base_url = "https://rest.uniprot.org/uniprotkb/accessions"
    
    # Process in chunks of 500 to avoid URI too long errors
    chunk_size = 500
    ids_list = list(uniprot_ids)
    
    with open(output_path, "w") as f_out:
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i:i+chunk_size]
            accessions = ",".join(chunk)
            
            resp = requests.get(f"{base_url}?accessions={accessions}&format=fasta")
            if resp.status_code == 200:
                f_out.write(resp.text)
                f_out.write("\n")
            else:
                logger.error(f"Failed to fetch FASTA for chunk {i//chunk_size}: {resp.status_code}")
                
            time.sleep(0.5)  # Be nice to the API
            if (i + chunk_size) % 5000 == 0:
                logger.info(f"Downloaded {min(i + chunk_size, len(ids_list))}/{len(ids_list)} sequences")
                
    logger.info(f"Saved FASTA sequences to {output_path}")

def main():
    config = load_config("config.yaml")
    set_seed(config["project"]["random_seed"])
    
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Loading raw KG from {config['data']['raw_kg_path']}")
    df = pd.read_csv(config['data']['raw_kg_path'], low_memory=False)
    
    # 1. Extract PPI edges
    ppi = df[(df["x_type"] == "gene/protein") & (df["y_type"] == "gene/protein")].copy()
    initial_count = len(ppi)
    
    # 2. Remove self loops
    ppi = ppi[ppi["x_id"] != ppi["y_id"]]
    no_self_loop_count = len(ppi)
    
    # 3. Remove duplicates (undirected)
    # Create canonical edges where source < target to easily drop duplicates
    edges = pd.DataFrame({
        "node1": ppi[["x_id", "y_id"]].min(axis=1),
        "node2": ppi[["x_id", "y_id"]].max(axis=1)
    }).drop_duplicates()
    final_count = len(edges)
    
    logger.info(f"PPI Extraction Summary:")
    logger.info(f"  Initial edges: {initial_count}")
    logger.info(f"  After removing self-loops: {no_self_loop_count}")
    logger.info(f"  After deduplication (undirected): {final_count}")
    
    # Save processed edges
    edges.to_csv(os.path.join(processed_dir, "ppi_edges.csv"), index=False)
    
    # Extract unique proteins
    unique_genes = set(edges["node1"]).union(set(edges["node2"]))
    unique_genes = [str(g) for g in unique_genes]
    logger.info(f"Unique proteins in PPI network: {len(unique_genes)}")
    
    # Map NCBI Gene IDs to UniProt
    mapping_path = os.path.join(processed_dir, "ncbi_to_uniprot.csv")
    if os.path.exists(mapping_path):
        logger.info(f"Mapping already exists at {mapping_path}, loading...")
        mapping_df = pd.read_csv(mapping_path)
        mapping = dict(zip(mapping_df["ncbi_id"].astype(str), mapping_df["uniprot_id"]))
    else:
        mapping = fetch_uniprot_mapping(unique_genes)
        mapping_df = pd.DataFrame(list(mapping.items()), columns=["ncbi_id", "uniprot_id"])
        mapping_df.to_csv(mapping_path, index=False)
        
    # We only keep edges where both nodes could be mapped to UniProt
    mapped_genes = set(mapping.keys())
    valid_edges = edges[edges["node1"].astype(str).isin(mapped_genes) & edges["node2"].astype(str).isin(mapped_genes)].copy()
    logger.info(f"Edges remaining after filtering unmapped proteins: {len(valid_edges)}")
    
    # Save the mapped graph
    valid_edges["uniprot1"] = valid_edges["node1"].astype(str).map(mapping)
    valid_edges["uniprot2"] = valid_edges["node2"].astype(str).map(mapping)
    valid_edges.to_csv(os.path.join(processed_dir, "ppi_edges_uniprot.csv"), index=False)
    
    # Download FASTA sequences
    fasta_path = os.path.join(raw_dir, "protein_sequences.fasta")
    if not os.path.exists(fasta_path):
        unique_uniprot = set(valid_edges["uniprot1"]).union(set(valid_edges["uniprot2"]))
        download_fasta(unique_uniprot, fasta_path)
    else:
        logger.info(f"FASTA sequences already exist at {fasta_path}")
        
    logger.info("Phase 1: Data Preparation Complete.")

if __name__ == "__main__":
    main()
