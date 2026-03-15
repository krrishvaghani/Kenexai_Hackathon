
import time
import json
import os
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import traceback

from run_full_pipeline_update import run_orchestrator

LOG_FILE = "pipeline_execution_log.json"

def execute_pipeline_job():
    """
    Job that runs the full pipeline and logs the results.
    """
    print(f"[{datetime.datetime.now().isoformat()}] Scheduled Pipeline Run Started...")
    
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "rows_generated": 500,
        "etl_status": "failed",
        "model_accuracy": 0.0,
        "model_updated": False,
        "rag_refreshed": False
    }

    try:
        # Run the full pipeline update orchestration for 500 rows
        result = run_orchestrator(n_rows=500)
        
        # Capture returning values
        log_entry["etl_status"] = "success"
        log_entry["model_accuracy"] = round(result.get("model_accuracy", 0.0), 4)
        log_entry["model_updated"] = result.get("model_updated", False)
        log_entry["rag_refreshed"] = result.get("rag_refreshed", False)

        print(f"[{datetime.datetime.now().isoformat()}] Scheduled Pipeline Run Completed Successfully.")
        
    except Exception as e:
        print(f"[{datetime.datetime.now().isoformat()}] Scheduled Pipeline Run Failed!")
        traceback.print_exc()
        log_entry["etl_status"] = "error: " + str(e)
        
    finally:
        # Save to pipeline_execution_log.json
        logs = []
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                pass
                
        logs.append(log_entry)
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=4)

if __name__ == "__main__":
    print("="*60)
    print("STARTING PIPELINE SCHEDULER")
    print("Interval: Every 2 hours")
    print("Mode: AUTOMATED CONTINUOUS LEARNING")
    print("="*60)
    
    # Initialize Scheduler
    scheduler = BackgroundScheduler()
    
    # Register the job
    scheduler.add_job(
        execute_pipeline_job,
        'interval',
        hours=2,
        id='medallion_pipeline_task'
    )
    
    scheduler.start()
    
    try:
        # Keep the main thread alive to allow BackgroundScheduler to run jobs
        while True:
            time.sleep(60)
            
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopping...")
        scheduler.shutdown()
        print("Scheduler shut down successfully.")
