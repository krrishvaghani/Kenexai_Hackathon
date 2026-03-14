import pandas as pd
import numpy as np
import os
import urllib.parse
from sqlalchemy import create_engine
import json
import datetime

def get_db_engine():
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")
    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_uri)

def generate_synthetic_rows(n_rows: int = 100):
    engine = get_db_engine()
    # Read from raw_policy_data instead of processed
    df = pd.read_sql("SELECT * FROM raw_policy_data", engine)
    
    # Drop ID so Postgres can auto-increment it if it's the primary key
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    synth_data = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Add slight numerical noise
            mean = df[col].mean()
            std = df[col].std() * 1.05 # Add 5% noise variance
            min_val = df[col].min()
            max_val = df[col].max()

            new_vals = np.random.normal(loc=mean, scale=std, size=n_rows)
            # Clip
            new_vals = np.clip(new_vals, min_val, max_val)

            # Insert occasional outliers (1% chance)
            outlier_mask = np.random.random(n_rows) < 0.01
            if outlier_mask.any():
                new_vals[outlier_mask] = max_val * 1.5

            if pd.api.types.is_integer_dtype(df[col]):
                new_vals = np.round(new_vals).astype(int)

            # Insert realistic missing values (3% missing)
            missing_mask = np.random.random(n_rows) < 0.03
            
            # Since Pandas integer columns can't hold NaN cleanly without Float casting, we cast to float or use pd.NA
            synth_col = pd.Series(new_vals, dtype='float64')
            synth_col[missing_mask] = np.nan
            
            synth_data[col] = synth_col
        else:
            # Categorical distribution sampling with some missing values
            freq = df[col].value_counts(normalize=True)
            new_vals = np.random.choice(freq.index, p=freq.values, size=n_rows)
            
            missing_mask = np.random.random(n_rows) < 0.02
            synth_col = pd.Series(new_vals, dtype='object')
            synth_col[missing_mask] = None
            
            synth_data[col] = synth_col

    return pd.DataFrame(synth_data)

def insert_synthetic_rows(n_rows: int = 100):
    print(f"Generating {n_rows} synthetic raw rows...")
    new_df = generate_synthetic_rows(n_rows)

    engine = get_db_engine()
    print("Inserting into Bronze Data Layer (raw_policy_data)...")
    new_df.to_sql('raw_policy_data', engine, if_exists='append', index=False)

    # Log it
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rows_added": n_rows,
        "table_name": "raw_policy_data"
    }

    log_file = "data_growth_log.json"
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except:
                pass
    logs.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

    print(f"Successfully generated and inserted {n_rows} noisy records.")
    return n_rows

if __name__ == "__main__":
    insert_synthetic_rows(10)
