import pandas as pd

# Paths to CSVs in root directory
runners_path = "runners_flat.csv"
schedule_path = "shedule_flat.csv"

# Load just the top of each file
runners_df = pd.read_csv(runners_path, nrows=5, low_memory=False)
schedule_df = pd.read_csv(schedule_path, nrows=5, low_memory=False)

# Print column names
print("\n🏇 Runners CSV columns:")
print(list(runners_df.columns))

print("\n📅 Schedule CSV columns:")
print(list(schedule_df.columns))
