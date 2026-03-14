import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

# Connect to default database to create the new database
conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()
try:
    cursor.execute("CREATE DATABASE insurance_db;")
    print("Database 'insurance_db' created successfully.")
except psycopg2.errors.DuplicateDatabase:
    print("Database 'insurance_db' already exists.")

cursor.close()
conn.close()

# Connect to the newly created database and create tables
conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

create_raw_table = """
CREATE TABLE IF NOT EXISTS raw_policy_data (
    id SERIAL PRIMARY KEY,
    age VARCHAR,
    gender VARCHAR,
    education VARCHAR,
    income VARCHAR,
    credit_score FLOAT,
    vehicle_type VARCHAR,
    vehicle_age INT,
    annual_mileage INT,
    speeding_violations INT,
    duis INT,
    past_accidents INT,
    premium FLOAT,
    usage VARCHAR,
    claim_outcome INT,
    timestamp TIMESTAMP
);
"""

create_processed_table = """
CREATE TABLE IF NOT EXISTS processed_policy_features (
    id SERIAL PRIMARY KEY,
    driver_risk_score FLOAT,
    vehicle_risk_score FLOAT,
    premium_to_mileage_ratio FLOAT,
    claim_outcome INT
);
"""

cursor.execute(create_raw_table)
cursor.execute(create_processed_table)

conn.commit()
cursor.close()
conn.close()
print("Tables 'raw_policy_data' and 'processed_policy_features' ensured successfully.")
