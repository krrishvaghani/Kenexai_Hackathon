# AI-Driven Vehicle Insurance Risk Intelligence Platform

## Overview
The AI-Driven Vehicle Insurance Risk Intelligence Platform is an enterprise-grade, continuous-learning analytics ecosystem. Designed to solve data latency and predictive modeling challenges in the auto insurance industry, this platform integrates data engineering, machine learning, and generative AI into a fully automated, closed-loop system. 

It ingests raw policy data, processes it through structured ETL pipelines, trains predictive risk models to estimate claim probabilities, and delivers actionable insights via role-based business intelligence dashboards and a secure Retrieval-Augmented Generation (RAG) AI Copilot.

## System Architecture
The platform strictly adheres to the Medallion Architecture to guarantee data integrity:

*   **Bronze Layer (Raw Data):** Continuous ingestion of messy, real-world insurance data (generated via a synthetic data engine) into a PostgreSQL data warehouse.
*   **Silver Layer (Processed Data):** An automated ETL pipeline cleanses the data, handles missing values, resolves data types, and computes proprietary business metrics (e.g., driver risk scores, premium-to-mileage ratios).
*   **Gold Layer (Analytics & AI):** Clean data is consumed by machine learning pipelines, structured vector databases (ChromaDB) for the AI Copilot, and real-time Streamlit dashboards.

## Key Features

### 1. Automated Data Pipeline & Scheduler
*   **Continuous Ingestion:** Simulates real-world data streams using a robust synthetic data generator.
*   **Production Scheduler:** Utilizes `apscheduler` to seamlessly orchestrate the entire Medallion pipeline (ingestion, ETL, ML retraining, and RAG refresh) at a standard 2-hour interval.

### 2. Continuous Machine Learning (MLOps)
*   **Predictive Risk Modeling:** Deploys a Scikit-Learn `RandomForestClassifier` encapsulated in a transformation pipeline to accurately predict claim probability.
*   **Champion vs. Challenger Framework:** Automatically retrains models as new data arrives, deploying the new model only if its performance metrics exceed the currently deployed version.
*   **Model Performance Monitoring:** Maintains a strict JSON-based model registry to track versioning, dataset size, Accuracy, Precision, Recall, and F1-Score over time.

### 3. Business Intelligence Dashboards
*   **Role-Based Access:** Tailored interfaces for Risk Analysts, Underwriting Managers, and Executive Business Leaders.
*   **Data Quality Monitoring:** Live tracking of database health, invalid record drops, and null-value imputation rates.
*   **Exploratory Data Analysis:** Interactive visualizations for feature engineering validation and subset distribution.

### 4. Generative AI Insurance Copilot
*   **Secure RAG Architecture:** Leverages LangChain, ChromaDB, and local LLMs (Ollama) to query the processed Silver Data securely.
*   **Contextual Assistance:** Allows analysts to interrogate the company's quantitative data via natural language prompts, completely localized and protected.

## Technical Stack
*   **Application Framework:** Python 3.x, Streamlit
*   **Data Engineering:** Pandas, SQLAlchemy, PostgreSQL
*   **Machine Learning:** Scikit-Learn, Joblib
*   **Generative AI:** LangChain, ChromaDB, HuggingFace Embeddings, Ollama
*   **Automation:** APScheduler

## Getting Started

### Prerequisites
1.  Python 3.10+
2.  PostgreSQL Server running locally or remotely (configured via `.env`)
3.  Ollama installed locally (running `llama3-chatqa:8b` or preferred model)

### Installation
1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file in the root directory and configure the database connection:
    ```ini
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=your_password
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=insurance_db
    ```

## Usage Instructions

### Starting the Platform
To launch the primary interactive dashboard:
```bash
streamlit run dashboard/app.py
```

### Running the Medallion Pipeline (Manual Mode)
To trigger a manual data refresh and model retraining sequence (e.g., simulating 1,000 new policies):
```bash
python run_full_pipeline_update.py --rows 1000
```

### Starting the Automated Scheduler (Production Mode)
To initialize the continuous learning platform daemon (runs the comprehensive pipeline automatically based on the configured interval):
```bash
cd insurance-risk-platform
python generator/synthetic_generator.py
```

The script will generate one realistic insurance policy record every 3 seconds
and append it to `data/synthetic_stream.csv`. Press **Ctrl+C** to stop.

## License

MIT
