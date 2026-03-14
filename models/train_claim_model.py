import pandas as pd
import urllib.parse
import joblib
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("Starting Machine Learning Training Pipeline...")

    # 1. Database Connection
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_uri)

    # 2. Load Processed Data
    print("Loading data from 'processed_policy_features' table...")
    query = "SELECT * FROM processed_policy_features"
    df = pd.read_sql(query, engine)

    print(f"Dataset Loaded Setup: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Feature Selection
    features = [
        "driver_risk_score", 
        "vehicle_risk_score", 
        "premium_to_mileage_ratio"
    ]
    target = "claim_outcome"

    # Handle any potentially missing values just in case
    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    # 4. Train/Test Split
    print("Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Training
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 6. Evaluation
    print("Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("-" * 50)
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    # 7. Saving the Model
    model_path = "claim_prediction_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()
