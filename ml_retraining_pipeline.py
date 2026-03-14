import pandas as pd
import numpy as np
import os
import urllib.parse
from sqlalchemy import create_engine
import json
import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

def get_db_engine():
    load_dotenv()
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")
    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(db_uri)

def retrain_model():
    print("Loading data for retraining...")
    engine = get_db_engine()
    df = pd.read_sql("SELECT * FROM processed_feature_data", engine)
    
    # Drop rows with nulls
    df = df.dropna()
    
    # Extract features and target
    if 'claim_outcome' not in df.columns:
        print("Target column 'claim_outcome' not found.")
        return False, None
    
    # Typically dropping the target, but also any label equivalents
    X = df.drop(columns=['claim_outcome'])
    if 'claim_outcome_label' in X.columns:
        X = X.drop(columns=['claim_outcome_label'])
        
    y = df['claim_outcome']
    
    # Keep original dataframe structure for Pipeline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Automatically identify categorical columns (object type)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    print("Training RandomForestClassifier within Pipeline...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0, average='weighted')
    rec = recall_score(y_test, y_pred, zero_division=0, average='weighted')
    f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
    
    new_metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "dataset_size": len(df)
    }
    
        # Check model registry
    registry_file = "model_registry.json"
    registry = []
    if os.path.exists(registry_file):
        with open(registry_file, "r") as f:
            try:
                registry = json.load(f)
            except:
                pass

    best_acc = max([entry.get("accuracy", 0.0) for entry in registry]) if registry else 0.0

    print(f"New accuracy: {acc:.4f} vs Best previous accuracy: {best_acc:.4f}")

    model_updated = False
    version_num = len(registry) + 1
    model_version = f"v{version_num}"

    if acc > best_acc:
        model_updated = True
        print("Saving new improved model!")
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "claim_prediction_model.pkl")      
        joblib.dump(model, model_path)
    else:
        print("New model did not outperform previous model. Keeping previous model.")       

    # Append to model_registry.json
    new_entry = {
        "model_version": model_version,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_size": len(df),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4)
    }
    registry.append(new_entry)
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=4)

    # Append to model_training_log.json
    training_log_file = "model_training_log.json"
    training_log = []
    if os.path.exists(training_log_file):
        with open(training_log_file, "r") as f:
            try:
                training_log = json.load(f)
            except:
                pass
    
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_version": model_version,
        "dataset_size": len(df),
        "accuracy": round(float(acc), 4),
        "model_updated": model_updated
    }
    training_log.append(log_entry)
    with open(training_log_file, "w") as f:
        json.dump(training_log, f, indent=4)

    return model_updated, new_metrics

if __name__ == "__main__":
    retrain_model()

