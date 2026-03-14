"""
synthetic_generator.py
======================
Real-time synthetic vehicle insurance policy data generator.

Simulates streaming data ingestion by generating one realistic policy record
every 3 seconds and appending it to ``data/synthetic_stream.csv``.

Part of the AI-Driven Vehicle Insurance Risk Analytics Platform.

Usage:
    python -m generator.synthetic_generator      (from project root)
    python generator/synthetic_generator.py       (from project root)
"""

import os
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_stream.csv")
STREAM_INTERVAL_SECONDS = 3

# Driver attribute distributions
AGE_BUCKETS = ["16-25", "26-39", "40-64", "65+"]
AGE_WEIGHTS = [0.15, 0.35, 0.35, 0.15]

GENDERS = ["Male", "Female"]
GENDER_WEIGHTS = [0.52, 0.48]

EDUCATION_LEVELS = ["High School", "Associate", "Bachelor", "Master", "PhD"]
EDUCATION_WEIGHTS = [0.25, 0.15, 0.35, 0.18, 0.07]

INCOME_RANGES = {
    "High School": (18_000, 45_000),
    "Associate":   (28_000, 60_000),
    "Bachelor":    (40_000, 90_000),
    "Master":      (55_000, 130_000),
    "PhD":         (70_000, 180_000),
}

VEHICLE_TYPES = ["Sedan", "SUV", "Truck", "Bus", "Pickup"]
VEHICLE_TYPE_WEIGHTS = [0.35, 0.25, 0.15, 0.05, 0.20]

USAGE_TYPES = ["Private", "Commercial"]
USAGE_WEIGHTS = [0.75, 0.25]

# Schema column order
COLUMNS = [
    "TIMESTAMP",
    "AGE",
    "GENDER",
    "EDUCATION",
    "INCOME",
    "CREDIT_SCORE",
    "VEHICLE_TYPE",
    "VEHICLE_AGE",
    "ANNUAL_MILEAGE",
    "USAGE",
    "SPEEDING_VIOLATIONS",
    "DUIS",
    "PAST_ACCIDENTS",
    "PREMIUM",
    "CLAIM_OUTCOME",
]


# ---------------------------------------------------------------------------
# Helper Generators
# ---------------------------------------------------------------------------

def _generate_driver_attributes() -> dict:
    """Generate realistic driver demographic attributes."""
    age_bucket = random.choices(AGE_BUCKETS, weights=AGE_WEIGHTS, k=1)[0]
    gender = random.choices(GENDERS, weights=GENDER_WEIGHTS, k=1)[0]
    education = random.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS, k=1)[0]

    income_lo, income_hi = INCOME_RANGES[education]
    income = round(random.uniform(income_lo, income_hi), -2)  # round to nearest 100

    # Credit score correlates loosely with income
    base_credit = np.clip(300 + (income / 180_000) * 550 + random.gauss(0, 60), 300, 850)
    credit_score = int(round(base_credit))

    return {
        "AGE": age_bucket,
        "GENDER": gender,
        "EDUCATION": education,
        "INCOME": income,
        "CREDIT_SCORE": credit_score,
    }


def _generate_vehicle_attributes() -> dict:
    """Generate realistic vehicle attributes."""
    vehicle_type = random.choices(VEHICLE_TYPES, weights=VEHICLE_TYPE_WEIGHTS, k=1)[0]
    vehicle_age = random.randint(0, 20)
    annual_mileage = int(np.clip(random.gauss(12_000, 5_000), 3_000, 50_000))
    usage = random.choices(USAGE_TYPES, weights=USAGE_WEIGHTS, k=1)[0]

    return {
        "VEHICLE_TYPE": vehicle_type,
        "VEHICLE_AGE": vehicle_age,
        "ANNUAL_MILEAGE": annual_mileage,
        "USAGE": usage,
    }


def _generate_risk_behaviour(age_bucket: str) -> dict:
    """Generate risk behaviour attributes, influenced by age bucket."""
    # Younger drivers tend to have more violations
    if age_bucket == "16-25":
        speeding = np.random.poisson(2.5)
        duis = 1 if random.random() < 0.12 else 0
        past_accidents = np.random.poisson(1.2)
    elif age_bucket == "26-39":
        speeding = np.random.poisson(1.2)
        duis = 1 if random.random() < 0.06 else 0
        past_accidents = np.random.poisson(0.7)
    elif age_bucket == "40-64":
        speeding = np.random.poisson(0.6)
        duis = 1 if random.random() < 0.03 else 0
        past_accidents = np.random.poisson(0.5)
    else:  # 65+
        speeding = np.random.poisson(0.3)
        duis = 1 if random.random() < 0.02 else 0
        past_accidents = np.random.poisson(0.8)

    return {
        "SPEEDING_VIOLATIONS": int(speeding),
        "DUIS": int(duis),
        "PAST_ACCIDENTS": int(past_accidents),
    }


def _compute_risk_score(record: dict) -> float:
    """
    Compute a normalised risk score in [0, 1] based on driver, vehicle,
    and behavioural features.  Higher score → higher claim probability.
    """
    score = 0.0

    # --- Age risk ---
    age_risk = {"16-25": 0.30, "26-39": 0.10, "40-64": 0.05, "65+": 0.20}
    score += age_risk.get(record["AGE"], 0.10)

    # --- Credit score risk (lower credit → higher risk) ---
    credit = record["CREDIT_SCORE"]
    if credit < 500:
        score += 0.20
    elif credit < 600:
        score += 0.12
    elif credit < 700:
        score += 0.05

    # --- Speeding violations ---
    score += min(record["SPEEDING_VIOLATIONS"] * 0.06, 0.25)

    # --- DUI presence ---
    if record["DUIS"] >= 1:
        score += 0.20

    # --- Past accidents ---
    score += min(record["PAST_ACCIDENTS"] * 0.08, 0.30)

    # --- Vehicle age (older → slightly higher risk) ---
    if record["VEHICLE_AGE"] > 12:
        score += 0.05

    # --- Mileage risk ---
    if record["ANNUAL_MILEAGE"] > 25_000:
        score += 0.05

    # --- Commercial usage ---
    if record["USAGE"] == "Commercial":
        score += 0.05

    return np.clip(score, 0.0, 1.0)


def _compute_premium(record: dict, risk_score: float) -> float:
    """
    Derive a realistic annual premium from risk score and vehicle type.
    Base premium is modulated by vehicle type, usage, and risk.
    """
    base_premiums = {
        "Sedan":  800,
        "SUV":    1_000,
        "Truck":  1_100,
        "Bus":    1_600,
        "Pickup": 950,
    }
    base = base_premiums.get(record["VEHICLE_TYPE"], 900)

    # Commercial usage adds 30 %
    if record["USAGE"] == "Commercial":
        base *= 1.30

    # Risk multiplier: low risk ≈ 1.0×, high risk up to 3.0×
    risk_multiplier = 1.0 + risk_score * 2.0

    premium = base * risk_multiplier
    # Add some noise (± 8 %)
    premium *= random.uniform(0.92, 1.08)

    return round(premium, 2)


def _determine_claim_outcome(risk_score: float) -> int:
    """
    Stochastically determine whether a claim occurs (1) or not (0).
    Probability is a sigmoid-like mapping of the risk score.
    """
    # Sigmoid centred at risk_score ≈ 0.35
    probability = 1 / (1 + np.exp(-12 * (risk_score - 0.35)))
    return 1 if random.random() < probability else 0


# ---------------------------------------------------------------------------
# Record Assembly
# ---------------------------------------------------------------------------

def generate_record() -> dict:
    """
    Assemble a single synthetic insurance policy record with all fields,
    including a computed PREMIUM and a risk-weighted CLAIM_OUTCOME.
    """
    record = {}

    # Timestamp
    record["TIMESTAMP"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Driver
    record.update(_generate_driver_attributes())

    # Vehicle
    record.update(_generate_vehicle_attributes())

    # Risk behaviour (age-aware)
    record.update(_generate_risk_behaviour(record["AGE"]))

    # Derived fields
    risk_score = _compute_risk_score(record)
    record["PREMIUM"] = _compute_premium(record, risk_score)
    record["CLAIM_OUTCOME"] = _determine_claim_outcome(risk_score)

    return record


# ---------------------------------------------------------------------------
# Streaming Loop
# ---------------------------------------------------------------------------

def stream_records(output_path: str = OUTPUT_FILE,
                   interval: float = STREAM_INTERVAL_SECONDS) -> None:
    """
    Continuously generate synthetic records and append them to a CSV file.

    Parameters
    ----------
    output_path : str
        Absolute path for the output CSV.
    interval : float
        Seconds to wait between records.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine whether the CSV header already exists
    write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0

    print("=" * 65)
    print(" [*] AI-Driven Vehicle Insurance Risk Analytics Platform")
    print("     Synthetic Data Streaming Module")
    print("=" * 65)
    print(f" Output  : {output_path}")
    print(f" Interval: {interval}s per record")
    print(" Press Ctrl+C to stop.\n")

    record_count = 0

    try:
        while True:
            record = generate_record()
            df = pd.DataFrame([record], columns=COLUMNS)

            df.to_csv(
                output_path,
                mode="a",
                header=write_header,
                index=False,
            )

            # After the first write, never write header again
            write_header = False

            record_count += 1
            claim_flag = "!! CLAIM" if record["CLAIM_OUTCOME"] == 1 else "-- No claim"
            print(
                f"[{record['TIMESTAMP']}]  Record #{record_count:>5}  |  "
                f"Age: {record['AGE']:<6}  Premium: ${record['PREMIUM']:>8,.2f}  "
                f"{claim_flag}"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStreaming stopped. Total records generated: {record_count}")
        print(f"   Data saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stream_records()
