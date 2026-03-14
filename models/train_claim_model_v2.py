import os
import urllib.parse
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
    print("Starting Advanced Machine Learning Diagnostic & Training Pipeline")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Data Diagnostics
    # ---------------------------------------------------------
    print("\n--- STEP 1: Data Diagnostics ---")
    engine = connect_db()
    query = "SELECT * FROM processed_policy_features"
    df = pd.read_sql(query, engine)

    # Drop any nulls just in case
    df = df.dropna()

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nClass Distribution (claim_outcome):")
    class_dist = df['claim_outcome'].value_counts(normalize=True) * 100
    print(df['claim_outcome'].value_counts())
    print(class_dist.apply(lambda x: f"{x:.2f}%"))
    
    if class_dist.iloc[0] > 60:
         print("-> [Warning] Dataset exhibits class imbalance.")
    else:
         print("-> [Info] Dataset is relatively balanced.")

    print("\nFeature Summary Statistics:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])

    print("\nCorrelation Matrix:")
    print(df.corr(numeric_only=True).round(3))

    # ---------------------------------------------------------
    # STEP 2: Data Improvement (Feature Engineering)
    # ---------------------------------------------------------
    print("\n--- STEP 2: Data Improvement ---")
    print("Engineering new features...")
    df['total_risk_index'] = df['driver_risk_score'] + df['vehicle_risk_score']
    df['risk_premium_interaction'] = df['driver_risk_score'] * df['premium_to_mileage_ratio']
    
    # Feature Selection
    features = [
        "driver_risk_score", 
        "vehicle_risk_score", 
        "premium_to_mileage_ratio",
        "total_risk_index",
        "risk_premium_interaction"
    ]
    target = "claim_outcome"

    X = df[features]
    y = df[target]

    # ---------------------------------------------------------
    # STEP 3: Model Experimentation
    # ---------------------------------------------------------
    print("\n--- STEP 3: Model Experimentation ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = []
    
    for name, clf in models.items():
        # Create pipeline with scaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "Pipeline": pipeline, # keeping pipeline for selection
            "CM": cm
        })
        
        print(f"\n[Model: {name}]")
        print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

    # ---------------------------------------------------------
    # STEP 4: Best Model Selection
    # ---------------------------------------------------------
    print("\n--- STEP 4: Best Model Selection ---")
    # Sort by F1-Score descending
    results.sort(key=lambda x: x["F1-Score"], reverse=True)
    best_result = results[0]
    best_model_name = best_result["Model"]
    
    print(f"Best Model Selected -> {best_model_name} (F1-score: {best_result['F1-Score']:.4f})")
    print("Retraining the best Model on the ENTIRE dataset for production...")
    
    # Retrain on full dataset
    best_pipeline = best_result["Pipeline"]
    best_pipeline.fit(X, y)

    # ---------------------------------------------------------
    # STEP 5: Save Production Pipeline
    # ---------------------------------------------------------
    print("\n--- STEP 5: Save Production Pipeline ---")
    model_path = os.path.join(os.path.dirname(__file__), "claim_prediction_pipeline.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Production pipeline (Scaler + {best_model_name}) saved to: {model_path}")

    # ---------------------------------------------------------
    # STEP 6: Validation Test
    # ---------------------------------------------------------
    print("\n--- STEP 6: Validation Test ---")
    print("Loading pipeline and running a sample prediction...")
    
    loaded_pipeline = joblib.load(model_path)
    
    # Create sample dummy input matching the new feature set
    # Using features: driver_risk, vehicle_risk, premium_ratio, total_index, interaction
    d_risk = 8.5
    v_risk = 2.0
    p_ratio = 1.1
    t_index = d_risk + v_risk
    interaction = d_risk * p_ratio
    
    sample_input = pd.DataFrame([{
        "driver_risk_score": d_risk,
        "vehicle_risk_score": v_risk,
        "premium_to_mileage_ratio": p_ratio,
        "total_risk_index": t_index,
        "risk_premium_interaction": interaction
    }])
    
    pred_class = int(loaded_pipeline.predict(sample_input)[0])
    pred_prob = loaded_pipeline.predict_proba(sample_input)[0][1]
    
    label = "Claim Expected" if pred_class == 1 else "No Claim"
    print(f"Test Input Features:\n{sample_input.iloc[0].to_dict()}")
    print("-" * 30)
    print(f"Predicted Class : {pred_class} ({label})")
    print(f"Probability     : {pred_prob * 100:.2f}%")
    print("\nPipeline execution finished successfully.")

if __name__ == "__main__":
    run_ml_pipeline()
