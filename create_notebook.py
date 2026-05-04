import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# Retail Customer Survival Analysis\n\nThis notebook demonstrates how to load retail subscription data, estimate baseline survival using Kaplan-Meier, and fit a Cox Proportional Hazards model to predict customer Lifetime Value (LTV)."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom lifelines import KaplanMeierFitter, CoxPHFitter\nimport matplotlib.pyplot as plt\nimport joblib\n\nimport warnings\nwarnings.filterwarnings('ignore')"),
    nbf.v4.new_markdown_cell("## 1. Load Data\nWe load the synthetic retail subscription dataset."),
    nbf.v4.new_code_cell("df = pd.read_csv('retail_subscription_data.csv')\ndisplay(df.head())\nprint(f'Total records: {len(df)}')\nprint(f\"Churn rate: {df['churned'].mean():.2%}\")"),
    nbf.v4.new_markdown_cell("## 2. Kaplan-Meier Baseline Survival\nEstimate the overall baseline survival curve of our customer base."),
    nbf.v4.new_code_cell("kmf = KaplanMeierFitter()\nkmf.fit(durations=df['tenure_months'], event_observed=df['churned'])\n\nkmf.plot_survival_function(figsize=(10, 6))\nplt.title('Baseline Customer Survival Curve (Kaplan-Meier)')\nplt.ylabel('Probability of Remaining Subscribed')\nplt.xlabel('Tenure (Months)')\nplt.grid(True, alpha=0.3)\nplt.show()"),
    nbf.v4.new_markdown_cell("## 3. Cox Proportional Hazards Model\nFit a Cox PH model to understand how covariates (spend, tier, tickets) impact churn hazard."),
    nbf.v4.new_code_cell("cph = CoxPHFitter(penalizer=0.01)\ncph.fit(df, duration_col='tenure_months', event_col='churned')\n\ncph.print_summary()\ncph.plot()\nplt.title('Feature Hazard Ratios (Log Hazard)')\nplt.show()"),
    nbf.v4.new_markdown_cell("## 4. Predicting Lifetime Value (LTV)\nWe can predict the expected survival time for a new customer based on their features."),
    nbf.v4.new_code_cell("# Example: Predict median survival time for the first 5 customers\nexpected_lifetimes = cph.predict_median(df.head())\nprint(\"Expected Median Lifetime (Months) for first 5 customers:\")\nprint(expected_lifetimes)"),
    nbf.v4.new_markdown_cell("## 5. Export Model for API Deployment\nSave the trained Cox model as a `.pkl` file so our FastAPI service can load it for real-time predictions."),
    nbf.v4.new_code_cell("joblib.dump(cph, 'survival_model.pkl')\nprint(\"Model successfully saved as survival_model.pkl\")")
]

with open('Customer_Survival_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook Customer_Survival_Analysis.ipynb created successfully.")
