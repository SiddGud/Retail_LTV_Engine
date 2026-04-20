import pandas as pd
from lifelines import CoxPHFitter
import joblib

print("Loading dataset...")
df = pd.read_csv('retail_subscription_data.csv')

print(f"Dataset shape: {df.shape}")
print("Fitting Cox Proportional Hazards model...")
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df, duration_col='tenure_months', event_col='churned')

print("\nModel Summary:")
cph.print_summary()

print("\nSaving model to survival_model.pkl...")
joblib.dump(cph, 'survival_model.pkl')
print("Model saved successfully.")
