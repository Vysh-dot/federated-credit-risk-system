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

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# --------------------------------------------------
# Preferred model path
# Change this manually if you want a specific model
# --------------------------------------------------
PREFERRED_MODEL_PATH = "models/global_credit_model.pth"

# --------------------------------------------------
# Resolve model path safely
# --------------------------------------------------
def resolve_model_path():
    candidate_paths = [
        PREFERRED_MODEL_PATH,
        "models/global_credit_model_weighted.pth",
        "models/global_credit_model_equal.pth",
        "models/global_credit_model_custom.pth",
        "models/global_credit_model.pth",
    ]

    checked = []
    for path in candidate_paths:
        if path not in checked:
            checked.append(path)

    for path in checked:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No global model file was found. Checked paths:\n" +
        "\n".join(checked)
    )

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
# Get probabilities and labels from model
# --------------------------------------------------
def get_probs_and_labels(model, test_loader):
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

    return np.array(all_probs), np.array(all_labels)

# --------------------------------------------------
# Evaluate one threshold on one bank
# --------------------------------------------------
def evaluate_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

# --------------------------------------------------
# Main
# --------------------------------------------------
print("\n========== Global Threshold Tuning ==========")

MODEL_PATH = resolve_model_path()
print(f"Global model selected: {MODEL_PATH}")

# Load global model
global_model = CreditNet(input_size=10).to(DEVICE)
global_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print("Global model loaded successfully.")

# Load all bank test data once
bank_probs_labels = {}
for bank_name, test_path in TEST_PATHS.items():
    loader = load_test_data(test_path)
    probs, labels = get_probs_and_labels(global_model, loader)
    bank_probs_labels[bank_name] = (probs, labels)

best_threshold = None
best_macro_f1 = -1
all_results = []

for threshold in THRESHOLDS:
    print(f"\n========== Threshold = {threshold:.2f} ==========")

    threshold_results = {}
    f1_scores = []

    for bank_name, (probs, labels) in bank_probs_labels.items():
        metrics = evaluate_threshold(probs, labels, threshold)
        threshold_results[bank_name] = metrics
        f1_scores.append(metrics["f1"])

        print(
            f"{bank_name} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    macro_f1 = np.mean(f1_scores)
    threshold_results["macro_f1"] = macro_f1
    all_results.append((threshold, threshold_results))

    print(f"Macro-average F1 across banks: {macro_f1:.4f}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_threshold = threshold

print("\n========== Best Threshold Result ==========")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Macro-average F1: {best_macro_f1:.4f}")

for threshold, results in all_results:
    if threshold == best_threshold:
        print(f"\nDetailed metrics for threshold {threshold:.2f}:")
        for bank_name in TEST_PATHS.keys():
            metrics = results[bank_name]
            print(
                f"{bank_name} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"Prec={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

print("\nThreshold tuning completed successfully.")