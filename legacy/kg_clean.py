import pandas as pd

print("Loading original data...")
df = pd.read_csv('kg.csv', low_memory=False)

# 1. DROP DUPLICATES (Just in case)
df = df.drop_duplicates()

# 2. FILTER: Keep only "Drug", "Disease", and "Protein"
# We remove "Anatomy", "Pathway", etc. to make it lightweight for your laptop.
valid_types = ['drug', 'disease', 'gene/protein']
df_clean = df[
    (df['x_type'].isin(valid_types)) & 
    (df['y_type'].isin(valid_types))
]

# 3. REMOVE SELF-LOOPS (Drug A -> Drug A)
df_clean = df_clean[df_clean['x_name'] != df_clean['y_name']]

# 4. SAVE
print(f"Original size: {len(df)}")
print(f"Cleaned size:  {len(df_clean)}")
df_clean.to_csv('kg_clean.csv', index=False)
print("Saved to 'kg_clean.csv'. Use this file for the AI!")