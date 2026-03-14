import os
import urllib.parse
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')

def connect_db():
    load_dotenv()
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_uri)

def run_advanced_ml_pipelines():
    print("=" * 60)
    print("Expanding ML Module: Regression, Clustering, Association")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    print("Loading data from PostgreSQL...")
    engine = connect_db()
    
    # We will load both processed and raw data.
    # Raw data is needed for apriori logic (exact boolean flags of DUIs, etc)
    df_processed = pd.read_sql("SELECT * FROM processed_feature_data", engine)
    df_raw = pd.read_sql("SELECT * FROM raw_policy_data", engine)
    
    df_processed = df_processed.dropna()
    
    # ---------------------------------------------------------
    # USE CASE 1 — Regression (Predict Claim Cost)
    # ---------------------------------------------------------
    print("\n--- USE CASE 1: Regression (Claim Cost) ---")
    
    # Let's load the classifier we trained earlier to get the claim probabilities
    clf_model_path = os.path.join(os.path.dirname(__file__), "claim_prediction_model.pkl")
    classifier_pipeline = joblib.load(clf_model_path)
    
    # Obtain probability of claim
    X_clf = df_processed[['driver_age', 'gender', 'driving_experience', 'vehicle_age', 
                          'annual_mileage', 'credit_score', 'vehicle_damage', 
                          'annual_premium', 'driver_risk_score', 'vehicle_risk_score', 
                          'premium_to_mileage_ratio']]
    claim_probs = classifier_pipeline.predict_proba(X_clf)[:, 1]
    
    # Define Target: Proxy logic for Expected Claim Cost
    expected_claim_cost = df_processed['annual_premium'] * claim_probs
    
    # Train Random Forest Regressor
    print("Training RandomForestRegressor for Expected Claim Cost...")
    regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    
    # For numeric modeling purely, handle factors
    X_reg = pd.get_dummies(X_clf, drop_first=True)
    regressor.fit(X_reg, expected_claim_cost)
    
    predicted_cost = regressor.predict(X_reg)
    
    # Save results to df
    cost_df = pd.DataFrame({
        'original_row_index': df_processed.index,
        'annual_premium': df_processed['annual_premium'],
        'claim_probability': claim_probs.round(4),
        'predicted_claim_cost': predicted_cost.round(2)
    })
    
    cost_df.to_sql("claim_cost_predictions", engine, if_exists="replace", index=False)
    print(f"✅ Saved {len(cost_df)} claim cost predictions to database table: claim_cost_predictions")
    
    # Save the regression model artifact
    reg_model_path = os.path.join(os.path.dirname(__file__), "claim_severity_regressor.pkl")
    joblib.dump(regressor, reg_model_path)


    # ---------------------------------------------------------
    # USE CASE 2 — Clustering (Segment Drivers)
    # ---------------------------------------------------------
    print("\n--- USE CASE 2: KMeans Clustering (Risk Segments) ---")
    cluster_features = ['driver_risk_score', 'vehicle_risk_score', 'annual_mileage', 'annual_premium']
    
    X_cluster = df_processed[cluster_features]
    
    # Scale Features for KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    print("Training KMeans algorithm (n_clusters=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    
    # Attach to temporary dataframe to figure out risk order based on average driver_risk_score
    temp_df = X_cluster.copy()
    temp_df['cluster'] = clusters
    
    cluster_risk_means = temp_df.groupby('cluster')['driver_risk_score'].mean().sort_values()
    
    # Determine which cluster ID maps to Low, Medium, High risk
    mapping = {
        cluster_risk_means.index[0]: "Low Risk Drivers",
        cluster_risk_means.index[1]: "Medium Risk Drivers",
        cluster_risk_means.index[2]: "High Risk Drivers"
    }
    
    mapped_clusters = [mapping[c] for c in clusters]
    
    cluster_df = pd.DataFrame({
        'original_row_index': df_processed.index,
        'driver_risk_score': df_processed['driver_risk_score'],
        'vehicle_risk_score': df_processed['vehicle_risk_score'],
        'annual_mileage': df_processed['annual_mileage'],
        'annual_premium': df_processed['annual_premium'],
        'cluster_id': clusters,
        'risk_segment': mapped_clusters
    })

    cluster_df.to_sql("driver_risk_clusters", engine, if_exists="replace", index=False)
    print(f"✅ Saved {len(cluster_df)} driver segments to database table: driver_risk_clusters")

    # Save scaler and kmeans model
    joblib.dump({'scaler': scaler, 'kmeans': kmeans}, os.path.join(os.path.dirname(__file__), "risk_clustering_model.pkl"))

    # ---------------------------------------------------------
    # USE CASE 3 — Association Analysis (Apriori via mlxtend)
    # ---------------------------------------------------------
    print("\n--- USE CASE 3: Association Mining (Rules) ---")
    
    # Using Raw data to get behavior counts cleanly
    print("Building transaction matrix...")
    apriori_df = pd.DataFrame()
    apriori_df['Has_Speeding_Violation'] = df_raw['speeding_violations'] > 0
    apriori_df['Has_DUI'] = df_raw['duis'] > 0
    apriori_df['Has_Past_Accidents'] = df_raw['past_accidents'] > 0
    apriori_df['Resulted_In_Claim'] = df_raw['claim_outcome'] == 1
    
    # Run Algorithm
    frequent_itemsets = apriori(apriori_df, min_support=0.01, use_colnames=True)
    
    # Generate Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # Clean up names for Database Storage to strings
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Save meaningful subset rounding out floats
    final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    final_rules['support'] = final_rules['support'].round(4)
    final_rules['confidence'] = final_rules['confidence'].round(4)
    final_rules['lift'] = final_rules['lift'].round(4)
    
    # Filter only rules that end up producing a "Resulted_In_Claim"
    # Or keep all interesting ones if no direct claim consequent rule has lift > 1.0 at min_support=0.01
    claim_driving_rules = final_rules[final_rules['consequents'].str.contains('Resulted_In_Claim')]
    if len(claim_driving_rules) == 0:
        # Fallback: Save all robust association rules found
        claim_driving_rules = final_rules.sort_values(by="lift", ascending=False).head(50)
    
    claim_driving_rules.to_sql("association_rules", engine, if_exists="replace", index=False)
    print(f"✅ Extracted and saved {len(final_rules)} total association rules (with {len(claim_driving_rules)} stored) to table: association_rules")


    print("\n============================================================")
    print("Advanced Machine Learning Subsystem Expanded Successfully!")
    print("============================================================")

if __name__ == "__main__":
    run_advanced_ml_pipelines()
