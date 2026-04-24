from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI(title="Retail LTV Engine API", description="Predict expected customer lifetime using Cox Proportional Hazards.")

class CustomerFeatures(BaseModel):
    monthly_spend: float
    subscription_tier: int
    support_tickets: int
