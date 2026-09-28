import pandas as pd

# Load the clean data to check names
print("Loading data for search...")
df = pd.read_csv('kg_clean.csv')

# Get all unique names
all_nodes = pd.concat([df['x_name'], df['y_name']]).unique()

def search(query):
    query = query.lower()
    matches = [name for name in all_nodes if query in str(name).lower()]
    return matches[:10]  # Show top 10 matches

while True:
    user_input = input("\nType a partial name (or 'q' to quit): ")
    if user_input == 'q': break
    
    results = search(user_input)
    print(f"Found {len(results)} matches:")
    for r in results:
        print(f" - {r}")