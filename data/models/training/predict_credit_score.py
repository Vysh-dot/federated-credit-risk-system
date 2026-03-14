import torch
import torch.nn as nn
import numpy as np
import joblib
import sys
import os

# --------------------------------------------------
# Import model
# --------------------------------------------------
sys.path.append("data/models")
from credit_model import CreditNet

# --------------------------------------------------
# Settings
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/personalized_bank_2_model.pth"
SCALER_PATH = "preprocessing/bank_2/scaler.pkl"

THRESHOLD = 0.55

FEATURE_COLUMNS = [
    "age",
    "credit_capacity",
    "monthly_income",
    "debt_ratio",
    "num_open_accounts",
    "loan_duration",
    "installment_rate",
    "late_payment_score",
    "recent_payment_amount",
    "credit_utilization"
]

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = CreditNet(input_size=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded:", MODEL_PATH)

# --------------------------------------------------
# Load scaler
# --------------------------------------------------
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Example new customer
# --------------------------------------------------
customer = {
    "age": 35,
    "credit_capacity": 50000,
    "monthly_income": 8000,
    "debt_ratio": 0.25,
    "num_open_accounts": 4,
    "loan_duration": 24,
    "installment_rate": 2,
    "late_payment_score": 1,
    "recent_payment_amount": 2000,
    "credit_utilization": 0.30
}

# --------------------------------------------------
# Convert to array
# --------------------------------------------------
features = np.array([[customer[col] for col in FEATURE_COLUMNS]])

# Scale
features_scaled = scaler.transform(features)

# Convert to tensor
tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

# --------------------------------------------------
# Predict
# --------------------------------------------------
with torch.no_grad():
    logits = model(tensor)
    prob = torch.sigmoid(logits).item()

prediction = 1 if prob >= THRESHOLD else 0

# --------------------------------------------------
# Credit Score Mapping
# --------------------------------------------------
credit_score = int((1 - prob) * 550 + 300)

# --------------------------------------------------
# Output
# --------------------------------------------------
print("\nCustomer Credit Evaluation")
print("---------------------------")
print("Default Probability:", round(prob,4))
print("Credit Score:", credit_score)

if prediction == 1:
    print("Prediction: High Risk (Possible Default)")
else:
    print("Prediction: Low Risk (Safe Borrower)")