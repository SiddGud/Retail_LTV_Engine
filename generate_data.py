import pandas as pd
import numpy as np

np.random.seed(42)
n_customers = 5000

# 1. Base Features
subscription_tier = np.random.choice([1, 2, 3], size=n_customers, p=[0.5, 0.3, 0.2])
monthly_spend = np.random.normal(20, 5, n_customers) + (subscription_tier * 30)
monthly_spend = np.clip(monthly_spend, 5, 150)
support_tickets = np.random.poisson(lam=0.5 + (subscription_tier == 1)*1.5, size=n_customers)

# 2. Survival Time Generation (using Weibull distribution to simulate hazard)
# Higher tier -> lower hazard (lives longer)
# More tickets -> higher hazard (lives shorter)
# Higher spend (relative to tier) -> lower hazard

baseline_hazard = 0.05
hazard_multiplier = np.exp(
    -0.5 * subscription_tier + 
    0.3 * support_tickets - 
    0.01 * monthly_spend
)
lambda_param = baseline_hazard * hazard_multiplier

# Generate true survival times
true_survival_times = np.random.weibull(1.5, n_customers) / lambda_param

# Censoring: simulate that we only observe them for up to 60 months
censor_time = np.random.uniform(1, 60, n_customers)

tenure_months = np.minimum(true_survival_times, censor_time)
churned = (true_survival_times <= censor_time).astype(int)

df = pd.DataFrame({
    'monthly_spend': np.round(monthly_spend, 2),
    'subscription_tier': subscription_tier,
    'support_tickets': support_tickets,
    'tenure_months': np.round(tenure_months, 1),
    'churned': churned
})

df.to_csv('retail_subscription_data.csv', index=False)
print(f"Generated {n_customers} customer records in retail_subscription_data.csv")
print(f"Overall churn rate: {df['churned'].mean():.2%}")
