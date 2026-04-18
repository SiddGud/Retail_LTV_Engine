import pandas as pd
from lifelines import CoxPHFitter
import joblib

print("Loading dataset...")
df = pd.read_csv('retail_subscription_data.csv')
print(f"Dataset shape: {df.shape}")
