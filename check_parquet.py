import pandas as pd

# Path to your parquet file
file_path = "five_year_dataset.parquet"

# Load only the metadata to avoid loading the full dataset into memory
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