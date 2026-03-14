import os
import urllib.parse
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

def run_ml_pipeline():
    print("=" * 60)
    print("Training ML Model on Kaggle Dataset Setup")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Load Real Kaggle Data from DB
    # ---------------------------------------------------------
    print("Loading data from 'processed_feature_data' table...")
    engine = connect_db()
    query = "SELECT * FROM processed_feature_data"
    df = pd.read_sql(query, engine)
    
    # Missing value handling (safety net)
    df = df.dropna()
    print(f"Dataset Loaded Setup: {df.shape[0]} rows, {df.shape[1]} columns")

    # ---------------------------------------------------------
    # STEP 2: Prep Variables
    # ---------------------------------------------------------
    target = 'claim_outcome'
    categorical_features = ['driver_age', 'gender', 'driving_experience']
    numeric_features = ['vehicle_age', 'annual_mileage', 'credit_score', 
                        'vehicle_damage', 'annual_premium', 'driver_risk_score', 
                        'vehicle_risk_score', 'premium_to_mileage_ratio']
    
    X = df[categorical_features + numeric_features]
    y = df[target]

    print(f"Target Distribution: \n{y.value_counts(normalize=True) * 100}")

    # ---------------------------------------------------------
    # STEP 3: Train Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # STEP 4: Build Preprocessing Pipeline & Model
    # ---------------------------------------------------------
    print("\nBuilding Pipeline with Random Forest...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=150, 
            max_depth=12,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # ---------------------------------------------------------
    # STEP 5: Train & Evaluate
    # ---------------------------------------------------------
    print("Training Random Forest Classifier on Real Features...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("-" * 50)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    # ---------------------------------------------------------
    # STEP 6: Save Model
    # ---------------------------------------------------------
    model_path = os.path.join(os.path.dirname(__file__), "claim_prediction_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    run_ml_pipeline()
