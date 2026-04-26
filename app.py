from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI(title="Retail LTV Engine API", description="Predict expected customer lifetime using Cox Proportional Hazards.")

try:
    model = joblib.load("survival_model.pkl")
except Exception as e:
    logging.warning("No model found. Please train and export survival_model.pkl first.")
    model = None

class CustomerFeatures(BaseModel):
    monthly_spend: float
    subscription_tier: int
    support_tickets: int
