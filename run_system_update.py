import argparse
from continuous_pipeline import run_full_system_update

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger full pipeline update with synthetic data.")
    parser.add_argument("--rows", type=int, default=1000, help="Number of synthetic rows to generate.")
    
    args = parser.parse_args()
    
    run_full_system_update(n_rows=args.rows)