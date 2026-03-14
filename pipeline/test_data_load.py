import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

# DB connection - URL encoded password
password = urllib.parse.quote_plus("Preet@3753")
engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/insurance_db")

print("\n✅ Checking RAW TABLE\n")

raw_df = pd.read_sql("SELECT * FROM raw_policy_data ORDER BY id DESC LIMIT 5", engine)
print(raw_df)

print("\nTotal RAW rows:", len(pd.read_sql("SELECT * FROM raw_policy_data", engine)))

print("\n✅ Checking PROCESSED TABLE\n")

proc_df = pd.read_sql("SELECT * FROM processed_policy_features ORDER BY id DESC LIMIT 5", engine)
print(proc_df)

print("\nTotal PROCESSED rows:", len(pd.read_sql("SELECT * FROM processed_policy_features", engine)))

print("\n🎯 Data pipeline storage verification completed.")
