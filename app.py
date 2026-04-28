from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI(title="Retail LTV Engine API", description="Predict expected customer lifetime using Cox Proportional Hazards.")

# Load the trained survival model
try:
    model = joblib.load("survival_model.pkl")
except Exception as e:
    logging.warning("No model found. Please train and export survival_model.pkl first.")
    model = None

# Define the expected input schema for a customer
class CustomerFeatures(BaseModel):
    monthly_spend: float
    subscription_tier: int
    support_tickets: int
    # Add other covariates your model was trained on here

@app.post("/predict_ltv")
def predict_ltv(customer: CustomerFeatures):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # Convert incoming JSON into a DataFrame
    input_data = pd.DataFrame([customer.dict()])
    
    # Predict the expected median survival time (LTV) using lifelines
    # predict_median returns a pandas Series/DataFrame
    try:
        expected_lifetime = model.predict_median(input_data).iloc[0]
        # If the median is np.inf, it means survival is > max observed time
        if expected_lifetime == float('inf'):
            expected_lifetime = 60.0 # cap it at max observed time for demo purposes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "status": "success",
        "expected_lifetime_months": float(expected_lifetime),
        "risk_score": float(expected_lifetime) # Modify based on exact model output
    }
