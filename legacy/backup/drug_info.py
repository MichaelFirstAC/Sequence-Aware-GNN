import pubchempy as pcp

def get_drug_details(drug_name):
    """
    Fetches real-world details for a drug name using PubChem API.
    """
    print(f"   ...fetching details for {drug_name}...")
    try:
        # Search PubChem for the drug name
        compounds = pcp.get_compounds(drug_name, 'name')
        
        if not compounds:
            return "No details found in PubChem."
            
        # Get the first result (best match)
        c = compounds[0]
        
        # 1. Get Synonyms (Commercial Names like 'Tylenol' or 'Advil')
        # We take the top 5 synonyms because the list can be huge
        synonyms = c.synonyms[:5]
        synonyms_str = ", ".join(synonyms) if synonyms else "None"
        
        # 2. Get Molecular Formula
        formula = c.molecular_formula
        
        # 3. Format the output
        info = f"\n   [🔬 Formula]: {formula}"
        info += f"\n   [💊 Brands/Synonyms]: {synonyms_str}"
        
        return info

    except Exception as e:
        return f"Error connecting to PubChem: {str(e)}"

# Quick Test (Run this file directly to check connection)
if __name__ == "__main__":
    print(get_drug_details("Aspirin"))