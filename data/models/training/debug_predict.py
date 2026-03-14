import sys
from pathlib import Path
import joblib
import numpy as np
import torch

PROJECT_ROOT = Path(r"C:\Users\Dell\Desktop\credit_federated_system")
sys.path.insert(0, str(PROJECT_ROOT / "data" / "models"))

from credit_model import CreditNet

DEVICE = torch.device("cpu")

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

bank = "bank_3"
model_path = PROJECT_ROOT / "models" / f"personalized_{bank}_model.pth"
scaler_path = PROJECT_ROOT / "preprocessing" / bank / "scaler.pkl"

customer = {
    "age": 27,
    "credit_capacity": 50000.0,
    "monthly_income": 50500.0,
    "debt_ratio": 0.25,
    "num_open_accounts": 4.0,
    "loan_duration": 24.0,
    "installment_rate": 2.0,
    "late_payment_score": 0.0,
    "recent_payment_amount": 2000.0,
    "credit_utilization": 0.20
}

x = np.array([[customer[col] for col in FEATURE_COLUMNS]], dtype=float)

print("Raw input:")
for k, v in customer.items():
    print(f"{k}: {v}")

scaler = joblib.load(scaler_path)
x_scaled = scaler.transform(x)

print("\nScaled input:")
for name, val in zip(FEATURE_COLUMNS, x_scaled[0]):
    print(f"{name}: {val:.6f}")

model = CreditNet(input_size=10).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

with torch.no_grad():
    tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    logits = model(tensor).item()
    prob = torch.sigmoid(torch.tensor(logits)).item()

print(f"\nLogit: {logits:.6f}")
print(f"Default probability: {prob:.6f}")