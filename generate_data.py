import pandas as pd
import numpy as np

np.random.seed(42)
n_customers = 5000

# 1. Base Features
subscription_tier = np.random.choice([1, 2, 3], size=n_customers, p=[0.5, 0.3, 0.2])
monthly_spend = np.random.normal(20, 5, n_customers) + (subscription_tier * 30)
monthly_spend = np.clip(monthly_spend, 5, 150)
support_tickets = np.random.poisson(lam=0.5 + (subscription_tier == 1)*1.5, size=n_customers)
