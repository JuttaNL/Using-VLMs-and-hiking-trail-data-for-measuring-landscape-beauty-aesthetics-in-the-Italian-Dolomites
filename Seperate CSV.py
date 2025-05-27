import pandas as pd
import os

# File paths
input_csv = '/Users/just/Documents/ERM/Thesis stuff/Hiking trail data images.xlsx'
output_dir = '/Users/just/Documents/ERM/Thesis stuff/Hiking trail CSV files seperate'

# Load the Excel file
df = pd.read_excel(input_csv, sheet_name='Hiking trail data images')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create separate CSVs for each trail
for trail_name, group in df.groupby('Hiking trail name'):
    # Sanitize the trail name for a valid filename
    safe_name = trail_name.replace(" ", "_").replace("/", "-")
    output_path = os.path.join(output_dir, f"{safe_name}.csv")
    group.to_csv(output_path, index=False)

print(f"✅ CSV files saved in: {output_dir}")
