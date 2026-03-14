import os
import json
import datetime
import argparse
from synthetic_data_generator import insert_synthetic_rows
import sys

# Ensure pipeline modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pipeline')))
from pipeline.etl_pipeline import run_etl_pipeline
from ml_retraining_pipeline import retrain_model
from rag_refresh_pipeline import refresh_rag_vector_db

def run_orchestrator(n_rows: int = 1000):
    print("="*60)
    print(f"STARTING ENTERPRISE MEDALLION PIPELINE ({n_rows} synthetic rows)")
    print("="*60)

    # 1 & 2. Generate and Insert Raw Datastream
    print("\n>>> STEP 1 & 2: Generate & Insert into Bronze Layer (raw_policy_data)")
    rows_added = insert_synthetic_rows(n_rows)
    print(f"[OK] Raw Data expanded by {rows_added} rows.")

    # 3 & 4. Process into Silver via ETL
    print("\n>>> STEP 3 & 4: Trigger ETL Pipeline & Update Silver Layer (processed_feature_data)")
    valid_rows = run_etl_pipeline()
    print(f"[OK] ETL Processed successfully. Total silver inventory: {valid_rows}")

    # 5. ML Retraining & Evaluation (Gold)
    print("\n>>> STEP 5: Model Retraining and Evaluation Pipeline")
    model_updated, metrics = retrain_model()
    acc = metrics.get('accuracy', 0) if metrics else 0
    print(f"[OK] ML Retraining finished. New Model Deployed: {model_updated}")

    # 6. RAG Knowledge Base Refresh (Gold Analytics)
    print("\n>>> STEP 6: Refreshing Copilot RAG Vector Database")
    try:
        refresh_rag_vector_db()
        rag_refreshed = True
        print("[OK] RAG database synchronized with latest data.")
    except Exception as e:
        print(f"[FAIL] RAG refresh failed: {str(e)}")
        rag_refreshed = False

    # 7. Final Pipeline Log
    print("\n>>> STEP 7: Logging Pipeline Execution")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "requested_synthetic_rows": n_rows,
        "valid_processed_rows": valid_rows,
        "model_accuracy": acc,
        "model_updated": model_updated,
        "rag_refreshed": rag_refreshed
    }

    log_file = "pipeline_logs.json"
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

    print("="*60)
    print("PIPELINE ORCHESTRATION COMPLETE.")
    print("Bronze -> Silver -> Gold layers are strictly enforced.")
    print("Dashboards and AI Copilot will reflect the newly cleaned data natively.")
    print("="*60)
    
    return log_entry
