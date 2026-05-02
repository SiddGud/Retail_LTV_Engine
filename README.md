# Retail Subscription & LTV Prediction Engine

> An enterprise-grade Customer Survival Analysis and Lifetime Value (LTV) prediction engine for e-commerce subscription services.

Unlike traditional binary churn classification (will the customer churn: yes/no), this engine uses **Survival Analysis** to predict *when* a customer is likely to churn. This time-to-event framing allows businesses to calculate expected customer lifetime, accurately forecast revenue, and intervene proactively before high-value accounts cancel their subscriptions.

---

## Architecture & Methodology

This engine adapts clinical survival analysis techniques to retail e-commerce behavior:

- **Kaplan-Meier Estimator:** Calculates the baseline survival function, providing a macro-view of overall customer retention probabilities over time (e.g., "What percentage of customers survive past month 12?").
- **Cox Proportional Hazards (CPH):** A semi-parametric regression model that evaluates how specific customer features (covariates like monthly spend, subscription tier, or support tickets) increase or decrease the baseline hazard rate. This generates individualized survival curves and expected LTV calculations for every customer.

---

## Tech Stack

- **Python 3.9+**
- **scikit-survival / lifelines** — Core survival analysis modeling
- **Pandas & NumPy** — Data manipulation and preprocessing
- **FastAPI** — High-performance REST API for model serving
- **Joblib** — Model serialization

---

## Setup & Run Instructions

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Retail-LTV-Prediction-Engine.git
cd Retail-LTV-Prediction-Engine
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Model Training

Run the Jupyter Notebook to process the customer data, train the Kaplan-Meier and Cox models, and export the serialized model:

```bash
jupyter notebook Customer_Survival_Analysis.ipynb
```
*(Ensure you run all cells to generate the `survival_model.pkl` artifact.)*

### 3. Serving the API

Start the FastAPI server to expose the LTV prediction endpoint:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. You can test the endpoint and view interactive API documentation at `http://localhost:8000/docs`.
