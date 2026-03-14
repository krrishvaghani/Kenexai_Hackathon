import joblib
import pandas as pd
import os

# Define the path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "claim_prediction_model.pkl")

def load_model():
    """Helper function to load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_claim_model.py first.")
    return joblib.load(MODEL_PATH)

def predict_claim(driver_risk_score: float, vehicle_risk_score: float, premium_ratio: float) -> dict:
    """
    Predicts the probability of an insurance claim based on risk features.
    
    Args:
        driver_risk_score (float): Calculated risk score of the driver
        vehicle_risk_score (float): Calculated risk score of the vehicle
        premium_ratio (float): Premium to annual mileage ratio
        
    Returns:
        dict: Contains 'prediction' (0 or 1) and 'probability' (float percentage)
    """
    model = load_model()
    
    # Format input as a DataFrame to keep feature names consistent with training
    input_data = pd.DataFrame([{
        "driver_risk_score": driver_risk_score,
        "vehicle_risk_score": vehicle_risk_score,
        "premium_to_mileage_ratio": premium_ratio
    }])
    
    # Inference
    prediction_class = int(model.predict(input_data)[0])
    prediction_probs = model.predict_proba(input_data)[0]
    
    # Assuming class 1 is "claim" and class 0 is "no claim"
    claim_probability = prediction_probs[1]
    
    return {
        "predicted_class": prediction_class,
        "target_label": "Claim Expected" if prediction_class == 1 else "No Claim",
        "claim_probability_pct": round(claim_probability * 100, 2)
    }

# ----------------- TEST INFERENCE SCRIPT -----------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI Insurance Risk Copilot - Prediction Test")
    print("="*50)
    
    # Mock Customer 1: High Risk
    print("\nTest Case 1: High Risk Profile")
    res1 = predict_claim(driver_risk_score=9.5, vehicle_risk_score=3.2, premium_ratio=0.85)
    print(f"Risk Inputs -> Driver: 9.5 | Vehicle: 3.2 | Premium Ratio: 0.85")
    print(f"Prediction  -> {res1['target_label']} (Class {res1['predicted_class']})")
    print(f"Probability -> {res1['claim_probability_pct']}% chance of claiming.")

    # Mock Customer 2: Low Risk
    print("\nTest Case 2: Low Risk Profile")
    # Notice we use negative risk scores since the synthetic data computation resulted in negative ones previously
    res2 = predict_claim(driver_risk_score=-500.0, vehicle_risk_score=1.1, premium_ratio=0.05)
    print(f"Risk Inputs -> Driver: -500.0 | Vehicle: 1.1 | Premium Ratio: 0.05")
    print(f"Prediction  -> {res2['target_label']} (Class {res2['predicted_class']})")
    print(f"Probability -> {res2['claim_probability_pct']}% chance of claiming.")
    print("\n" + "="*50 + "\n")
