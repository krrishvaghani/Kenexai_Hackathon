import pandas as pd
import numpy as np
import urllib.parse
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import datetime
import json

def get_db_engine():
    load_dotenv()
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")
    password = urllib.parse.quote_plus(DB_PASS)
    return create_engine(f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def run_etl_pipeline():
    print("?? Triggering ETL Pipeline...")
    engine = get_db_engine()
    
    # 1. Extract from Bronze Layer
    print("Extracting from raw_policy_data (Bronze Layer)...")
    df = pd.read_sql("SELECT * FROM raw_policy_data", engine)
    
    initial_count = len(df)

    # 2. Data Cleaning & Validation
    print("Running Data Quality Validations & Cleaning...")
    
    # Drop completely empty rows
    df = df.dropna(how='all')
    
    # Impute missing numeric values with median, categorical with mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Validation constraints
    df = df[(df['annual_mileage'] >= 0) & (df['annual_mileage'] <= 200000)]
    df = df[(df['credit_score'] >= 0) & (df['credit_score'] <= 1)]
    
    # Type conversion explicitly
    for col in ['speeding_violations', 'duis', 'past_accidents', 'claim_outcome']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 3. Feature Engineering
    print("Performing Feature Engineering...")
    
    # Rename age to match schema
    if 'age' in df.columns:
        df['driver_age'] = df['age']
    
    # Map Vehicle Age
    def map_vehicle_age(v_year):
        if str(v_year).strip() == 'before 2015': return 3
        else: return 1
        
    if 'vehicle_year' in df.columns:
        df['vehicle_age'] = df['vehicle_year'].apply(map_vehicle_age)
    else:
        df['vehicle_age'] = 2 # default fallback

    # Generate Vehicle Damage
    if 'past_accidents' in df.columns:
        df['vehicle_damage'] = (df['past_accidents'] > 0).astype(int)
    else:
        df['vehicle_damage'] = 0

    # Risk Scores
    if all(c in df.columns for c in ['speeding_violations', 'duis', 'past_accidents']):
        df['driver_risk_score'] = (df['speeding_violations'] * 2) + (df['duis'] * 5) + (df['past_accidents'] * 3)
    else:
        df['driver_risk_score'] = 0
        
    # Prevent negative risk scores
    df.loc[df['driver_risk_score'] < 0, 'driver_risk_score'] = 0

    df['vehicle_risk_score'] = df['vehicle_damage'] + df['vehicle_age']
    
    mileage_divisor = df['annual_mileage'].replace(0, 1)
    if 'annual_premium' in df.columns:
        df['premium_to_mileage_ratio'] = df['annual_premium'] / mileage_divisor
    else:
        df['premium_to_mileage_ratio'] = 0.0

    # 4. Filter columns to exactly silver layer schema
    final_cols = [
        'driver_age', 'gender', 'driving_experience', 'vehicle_age', 
        'annual_mileage', 'credit_score', 'vehicle_damage', 'annual_premium', 
        'driver_risk_score', 'vehicle_risk_score', 'premium_to_mileage_ratio', 'claim_outcome'
    ]
    
    # Ensure all exist
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan
            
    final_df = df[final_cols].copy()
    
    # Fill any stragglers introduced by missing feature inputs
    final_df = final_df.fillna(0)
    
    valid_count = len(final_df)
    
    # 5. Load to Silver Layer
    print(f"Loading {valid_count} clean rows to processed_feature_data (Silver Layer)...")
    final_df.to_sql("processed_feature_data", engine, if_exists="replace", index=False)
    
    print(f"ETL completed! Dropped {initial_count - valid_count} invalid records.")
    return valid_count

if __name__ == '__main__':
    run_etl_pipeline()
