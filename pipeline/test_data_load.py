import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

# DB connection - URL encoded password
password = urllib.parse.quote_plus(DB_PASS)
engine = create_engine(f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print("\nChecking RAW TABLE\n")

raw_df = pd.read_sql("SELECT * FROM raw_policy_data ORDER BY id DESC LIMIT 5", engine)
print(raw_df)

print("\nTotal RAW rows:", len(pd.read_sql("SELECT * FROM raw_policy_data", engine)))

print("\nChecking PROCESSED TABLE\n")

proc_df = pd.read_sql("SELECT * FROM processed_policy_features ORDER BY id DESC LIMIT 5", engine)
print(proc_df)

print("\nTotal PROCESSED rows:", len(pd.read_sql("SELECT * FROM processed_policy_features", engine)))

print("\nData pipeline storage verification completed.")
