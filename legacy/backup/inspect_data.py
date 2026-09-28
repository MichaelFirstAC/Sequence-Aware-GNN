import pandas as pd

# 1. LOAD RAW DATA
print("Loading kg.csv... (This might take 10-20 seconds)")
df = pd.read_csv('kg.csv', low_memory=False)

# 2. BASIC HEALTH CHECK
print(f"\n--- DATASET VITALS ---")
print(f"Total Rows (Edges): {len(df):,}")
print(f"Columns: {list(df.columns)}")

# 3. CHECK FOR MISSING VALUES (The "NaN" Check)
print(f"\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0]) 
# If this prints nothing, the data is perfectly clean!

# 4. CHECK NODE TYPES (What are we dealing with?)
print(f"\n--- NODE TYPES ---")
# PrimeKG mixes everything (Anatomy, Pathways, Drugs). Let's see the breakdown.
print(df['x_type'].value_counts().head(5))

# 5. CHECK FOR "SELF-LOOPS" & DUPLICATES
# AI models hate it when a node points to itself (A -> A) or duplicates exist.
duplicates = df.duplicated().sum()
self_loops = len(df[df['x_name'] == df['y_name']])

print(f"\n--- DIRTY DATA CHECK ---")
print(f"Duplicate Rows: {duplicates} (We should drop these)")
print(f"Self-Loops (Node connecting to itself): {self_loops}")

# 6. MEMORY USAGE
print(f"\n--- MEMORY USAGE ---")
print(f"{df.memory_usage(deep=True).sum() / 1024**2 :.2f} MB")