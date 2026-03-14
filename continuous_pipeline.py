import os
import json
import datetime
from synthetic_data_generator import insert_synthetic_rows
from ml_retraining_pipeline import retrain_model
from rag_refresh_pipeline import refresh_rag_vector_db

def run_full_system_update(n_rows: int = 100):
    print("="*60)
    print(f"STARTING SYSTEM ORCHESTRATION UPDATE ({n_rows} rows)")
    print("="*60)
    
    # 1 & 2. Generate and Insert Data
    print(">>> STEP 1 & 2: Data Generation & Insertion")
    rows_added = insert_synthetic_rows(n_rows)
    print(f"\n[OK] Data expanded by {rows_added} rows.\n")
    
    # 3, 4 & 5. ML Retraining & Evaluation
    print(">>> STEP 3, 4 & 5: Model Retraining and Evaluation")
    model_updated, metrics = retrain_model()
    acc = metrics.get('accuracy', 0) if metrics else 0
    print(f"\n[OK] ML Retraining finished. Model replaced: {model_updated}\n")
    
    # 6. RAG Knowledge Base Refresh
    print(">>> STEP 6: Refreshing RAG Vector DB")
    try:
        refresh_rag_vector_db()
        rag_refreshed = True
        print("\n[OK] RAG base refreshed.\n")
    except Exception as e:
        print(f"\n[FAIL] RAG refresh failed: {str(e)}\n")
        rag_refreshed = False
    
    # 7. Final Pipeline Log
    print(">>> STEP 7: Logging Execution Pipeline")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rows_added": rows_added,
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
    print("The system is behaving as a Continuous Learning ML Platform.")
    print("Dashboards will auto-update on next refresh due to st.cache_data ttl.")
    print("="*60)

if __name__ == "__main__":
    run_full_system_update(100)
