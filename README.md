# AI-Driven Vehicle Insurance Risk Analytics Platform

An end-to-end AI-powered platform for vehicle insurance risk analytics,
featuring real-time synthetic data streaming, ML-based risk scoring,
and generative-AI-assisted insights.

## Project Structure

```
insurance-risk-platform/
├── data/                 # Generated datasets & streaming CSV output
├── generator/            # Synthetic data generation module
│   ├── __init__.py
│   └── synthetic_generator.py
├── pipeline/             # Data processing & feature engineering
├── models/               # ML model training, evaluation & artifacts
├── dashboard/            # Visualisation & interactive dashboards
├── genai/                # GenAI-powered insights & reporting
├── docker/               # Dockerfiles & compose configs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the synthetic data stream

```bash
cd insurance-risk-platform
python generator/synthetic_generator.py
```

The script will generate one realistic insurance policy record every 3 seconds
and append it to `data/synthetic_stream.csv`. Press **Ctrl+C** to stop.

## License

MIT
