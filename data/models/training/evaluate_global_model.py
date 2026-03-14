import sys
import os
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from credit_model import CreditNet

# --------------------------------------------------
# Settings
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

BATCH_SIZE = 64
MODEL_PATH = "models/global_credit_model.pth"

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

TARGET_COLUMN = "target"

TEST_PATHS = {
    "Bank 1": "preprocessing/bank_1/test_processed.csv",
    "Bank 2": "preprocessing/bank_2/test_processed.csv",
    "Bank 3": "preprocessing/bank_3/test_processed.csv"
}

# You can later tune these if needed
THRESHOLD = 0.55

# --------------------------------------------------
# Load one bank test dataset
# --------------------------------------------------
def load_test_data(test_csv):
    df = pd.read_csv(test_csv)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return loader

# --------------------------------------------------
# Evaluate global model on one bank
# --------------------------------------------------
def evaluate_on_bank(model, bank_name, test_loader, threshold=0.5):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    cm = confusion_matrix(all_labels, preds)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    print(f"\n========== Global Model Evaluation on {bank_name} ==========")
    print(f"Threshold  : {threshold:.2f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

# --------------------------------------------------
# Main
# --------------------------------------------------
print("\n========== Phase 6: Global Model Evaluation ==========")

# Load global model
global_model = CreditNet(input_size=10).to(DEVICE)
global_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"Global model loaded from: {MODEL_PATH}")

# Evaluate on each bank test set
for bank_name, test_path in TEST_PATHS.items():
    test_loader = load_test_data(test_path)
    evaluate_on_bank(global_model, bank_name, test_loader, threshold=THRESHOLD)

print("\nThis Phase completed successfully.")