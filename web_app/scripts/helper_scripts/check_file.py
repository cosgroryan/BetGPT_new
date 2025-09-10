import duckdb

RUNNERS = r'F:\db_hold\runners_flat.csv'
SCHED   = r'F:\db_hold\schedule_flat.csv'
PQ      = r'F:\db_hold\five_year_dataset.parquet'

con = duckdb.connect()
con.execute("PRAGMA enable_progress_bar;")

# Counts
print("Runners CSV rows:", con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{RUNNERS}', SAMPLE_SIZE=-1)").fetchone()[0])
print("Parquet rows:",     con.execute(f"SELECT COUNT(*) FROM parquet_scan('{PQ}')").fetchone()[0])

# Null checks on keys
print("Parquet null runner_number:", con.execute(f"SELECT COUNT(*) FROM parquet_scan('{PQ}') WHERE runner_number IS NULL").fetchone()[0])

# Mismatch checks between runner and schedule fields
print("Race name mismatches:",
      con.execute(f"""
        SELECT COUNT(*) FROM parquet_scan('{PQ}')
        WHERE race_name IS NOT NULL 
          AND race_name_sched IS NOT NULL 
          AND race_name <> race_name_sched
      """).fetchone()[0])

# Pick 5 random keys from Parquet and compare to CSV sources directly in DuckDB
keys = con.execute(f"""
  SELECT race_id, runner_number 
  FROM parquet_scan('{PQ}') 
  USING SAMPLE 5
""").fetchall()

for race_id, rn in keys:
    print(f"\n=== spot: race_id={race_id}, runner_number={rn} ===")
    print("Parquet:",
          con.execute(f"SELECT race_id, runner_number, runner_name, finish_rank, entrant_jockey, entrant_barrier FROM parquet_scan('{PQ}') WHERE race_id=? AND runner_number=?", [race_id, rn]).fetchdf())
    print("Runners CSV:",
          con.execute(f"SELECT race_id, runner_number, runner_name, finish_rank FROM read_csv_auto('{RUNNERS}', SAMPLE_SIZE=-1) WHERE CAST(race_id AS VARCHAR)=? AND TRY_CAST(runner_number AS INTEGER)=?", [race_id, rn]).fetchdf())
    print("Schedule CSV:",
          con.execute(f"SELECT race_id, runner_number, entrant_jockey, entrant_barrier FROM read_csv_auto('{SCHED}', SAMPLE_SIZE=-1) WHERE CAST(race_id AS VARCHAR)=? AND TRY_CAST(runner_number AS INTEGER)=?", [race_id, rn]).fetchdf())
