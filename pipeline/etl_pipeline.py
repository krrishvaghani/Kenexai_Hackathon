import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

# DB connection - password has @ so we must URL encode it
password = urllib.parse.quote_plus("Preet@3753")
engine = create_engine(f"postgresql://postgres:{password}@localhost:5432/insurance_db")

# define columns based on data
columns = [
    "timestamp", "age", "gender", "education", "income", "credit_score", 
    "vehicle_type", "vehicle_age", "annual_mileage", "usage", 
    "speeding_violations", "duis", "past_accidents", "premium", "claim_outcome"
]

# read streaming csv
df = pd.read_csv("../data/synthetic_stream.csv", names=columns)

# Ensure column names match the postgres table `raw_policy_data` definition exactly
df_raw = df.copy()

df_raw.columns = df_raw.columns.str.lower()
# id and timestamp columns might not be in the csv, that's okay, id is SERIAL, timestamp we can let Postgres handle or add it if missing
if 'timestamp' not in df_raw.columns:
    df_raw['timestamp'] = pd.Timestamp.now()

# ---------- RAW LOAD ----------
# We can drop the id column if we want the DB to auto-generate it (if it doesn't exist in CSV)
if 'id' in df_raw.columns:
    df_raw = df_raw.drop(columns=['id'])

df_raw.to_sql("raw_policy_data", engine, if_exists="append", index=False)

# ---------- TRANSFORMATION ----------
df["driver_risk_score"] = (
    (df["speeding_violations"] * 0.3) +
    (df["duis"] * 0.5) +
    (df["past_accidents"] * 0.4) +
    ((1 - df["credit_score"]) * 2)
)

df["vehicle_risk_score"] = df["vehicle_age"] * 0.1

df["premium_to_mileage_ratio"] = df["premium"] / (df["annual_mileage"] + 1)

processed = df[[
    "driver_risk_score",
    "vehicle_risk_score",
    "premium_to_mileage_ratio",
    "claim_outcome"
]].copy()

# Rename to match database table 'processed_policy_features'
# already lowercase so we're good

# ---------- GOLD LOAD ----------
processed.to_sql("processed_policy_features", engine,
                 if_exists="append", index=False)

print("ETL Pipeline executed successfully")
