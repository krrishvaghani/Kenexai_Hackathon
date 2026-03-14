import pandas as pd
import numpy as np
import urllib.parse
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load Environment Variables
load_dotenv()
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

password = urllib.parse.quote_plus(DB_PASS)
engine = create_engine(f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def run_real_etl():
    print("🚀 Starting Real Data ETL Pipeline...")

    # =================================================================
    # STEP 1 — LOAD DATASETS
    # =================================================================
    print("Loading datasets...")
    df1 = pd.read_csv("../../Car_Insurance_Claim.csv/Car_Insurance_Claim.csv")
    df2 = pd.read_csv("../../archive/train.csv")

    # =================================================================
    # STEP 2 — STANDARDIZE COLUMN NAMES
    # =================================================================
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()
    
    # Standardize string values
    df1['gender'] = df1['gender'].astype(str).str.lower()
    df2['gender'] = df2['gender'].astype(str).str.lower()

    # =================================================================
    # STEP 4 — DATA STITCHING (Done first to align rows safely)
    # =================================================================
    print("Stitching datasets together...")
    # Age bucketing for df2 to match df1 ('16-25', '26-39', '40-64', '65+')
    df2['age_group'] = pd.cut(df2['age'], bins=[15, 25, 39, 64, 120], labels=['16-25', '26-39', '40-64', '65+']).astype(str)
    
    # Shuffle to ensure random distribution
    df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign a row number per group to perform synthetic 1-to-1 join
    df1['row_num'] = df1.groupby(['age', 'gender']).cumcount()
    df2['row_num'] = df2.groupby(['age_group', 'gender']).cumcount()

    # Merge
    merged_df = pd.merge(
        df1, 
        df2, 
        left_on=['age', 'gender', 'row_num'], 
        right_on=['age_group', 'gender', 'row_num'], 
        how='inner'
    )
    print(f"Merged Data Shape: {merged_df.shape}")

    # =================================================================
    # STEP 3 & 6 — ALIGN FEATURE STRUCTURE
    # =================================================================
    # Clean up fields for mathematical operations
    # Vehicle Damage: Yes -> 1, No -> 0
    merged_df['vehicle_damage'] = merged_df['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Vehicle Age Mapping: < 1 Year -> 1, 1-2 Year -> 2, > 2 Years -> 3
    def map_vehicle_age(v_age):
        if v_age == '< 1 Year': return 1
        elif v_age == '1-2 Year': return 2
        else: return 3
    merged_df['vehicle_age_num'] = merged_df['vehicle_age'].apply(map_vehicle_age)

    # Fill NA values
    merged_df['annual_mileage'] = merged_df['annual_mileage'].fillna(merged_df['annual_mileage'].mean())
    merged_df['credit_score'] = merged_df['credit_score'].fillna(merged_df['credit_score'].mean())
    merged_df['outcome'] = merged_df['outcome'].fillna(0)

    # =================================================================
    # STEP 5 — FEATURE ENGINEERING
    # =================================================================
    print("Engineering risk features...")
    merged_df['driver_risk_score'] = (
        (merged_df['speeding_violations'] * 2) + 
        (merged_df['duis'] * 5) + 
        (merged_df['past_accidents'] * 3)
    )

    merged_df['vehicle_risk_score'] = merged_df['vehicle_damage'] + merged_df['vehicle_age_num']

    # Handle division by zero
    mileage_divisor = merged_df['annual_mileage'].replace(0, 1)
    merged_df['premium_to_mileage_ratio'] = merged_df['annual_premium'] / mileage_divisor

    # Determine Claim Outcome (OUTCOME from df1 OR Response from df2)
    merged_df['claim_outcome'] = np.where((merged_df['outcome'] == 1) | (merged_df['response'] == 1), 1, 0)

    # Assemble Final Schema Map
    final_cols = {
        'age_x': 'driver_age',
        'gender': 'gender',
        'driving_experience': 'driving_experience',
        'vehicle_age_num': 'vehicle_age',
        'annual_mileage': 'annual_mileage',
        'credit_score': 'credit_score',
        'vehicle_damage': 'vehicle_damage',
        'annual_premium': 'annual_premium',
        'driver_risk_score': 'driver_risk_score',
        'vehicle_risk_score': 'vehicle_risk_score',
        'premium_to_mileage_ratio': 'premium_to_mileage_ratio',
        'claim_outcome': 'claim_outcome'
    }
    
    final_df = merged_df[final_cols.keys()].rename(columns=final_cols)

    # =================================================================
    # STEP 7 — STORE DATA INTO POSTGRES WAREHOUSE
    # =================================================================
    print("Pushing data to PostgreSQL warehouse...")
    raw_df = merged_df[['id_x', 'age_x', 'gender', 'driving_experience', 'education', 'income', 
                        'credit_score', 'vehicle_year', 'annual_mileage', 'speeding_violations', 
                        'duis', 'past_accidents', 'annual_premium', 'previously_insured', 'claim_outcome']]
    raw_df = raw_df.rename(columns={'id_x': 'id', 'age_x': 'age'})

    # Dropping tables to rebuild them cleanly with new data size and avoid type mismatch
    with engine.connect() as con:
        # Using raw execute properly for sqlalchemy 2.0
        from sqlalchemy import text
        con.execute(text("DROP TABLE IF EXISTS raw_policy_data;"))
        con.execute(text("DROP TABLE IF EXISTS processed_feature_data;"))
        con.execute(text("DROP TABLE IF EXISTS processed_policy_features;")) # old table
        con.commit()

    # Save to PostgreSQL
    # Table 1: Raw
    raw_df.to_sql("raw_policy_data", engine, if_exists="replace", index=False)
    print(f"Loaded {len(raw_df)} rows into 'raw_policy_data'")

    # Table 2: Processed Features
    final_df.to_sql("processed_feature_data", engine, if_exists="replace", index=False)
    print(f"Loaded {len(final_df)} rows into 'processed_feature_data'")
    
    print("✅ ETL successfully updated to standard.")

if __name__ == "__main__":
    
    # We navigate from the module folder or the workspace base
    # Path adjusting to ensure it runs correctly
    curr_dir = os.getcwd()
    if 'pipeline' not in curr_dir:
        os.chdir(os.path.join(curr_dir, 'pipeline' if os.path.exists('pipeline') else 'Kenexai_Hackathon/pipeline'))

    run_real_etl()
