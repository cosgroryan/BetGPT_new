# merge_parquet.py
import duckdb
from pathlib import Path

PARTS_DIR = r"F:\db_hold\merged_parquet"   # folder with your 9 parts
OUT_FILE  = r"F:\db_hold\five_year_dataset.parquet"

con = duckdb.connect()
con.execute("PRAGMA enable_progress_bar;")

# UNION all part files and write a single Parquet
con.execute(f"""
COPY (
  SELECT * FROM parquet_scan('{PARTS_DIR}\\*.parquet')
) TO '{OUT_FILE}' (FORMAT PARQUET);
""")

print(f"✅ wrote → {OUT_FILE}")
