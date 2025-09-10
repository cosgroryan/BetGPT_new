import pandas as pd

# Path to your parquet file
file_path = "gallops_2025-08-18_to_2025-08-18.parquet"

# Load the dataset
df = pd.read_parquet(file_path)

# Display column names and their datatypes
print("\n=== Column Names and Data Types ===")
print(df.dtypes)

# If you want to see the first few rows as a sample
print("\n=== Sample Data ===")
print(df.head())

# If you also want basic stats for numeric columns
print("\n=== Numeric Column Summary ===")
print(df.describe())

# Save the first 100 rows to CSV
output_path = "first_1000_rows.csv"
df.head(1000).to_csv(output_path, index=False)
print(f"\n✅ First 100 rows saved to {output_path}")
