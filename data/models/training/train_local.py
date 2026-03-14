import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from credit_model import CreditNet

# --------------------------------------------------
# Settings
# --------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

os.makedirs("models", exist_ok=True)

# Thresholds to test
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# --------------------------------------------------
# Load processed train/test data
# --------------------------------------------------
def load_bank_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train_full = train_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    # Split training into train + validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, y_train, y_val, y_test

# --------------------------------------------------
# Compute positive class weight
# --------------------------------------------------
def compute_pos_weight(y_train):
    y_train = np.array(y_train)
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)

    if num_positive == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    pos_weight = num_negative / num_positive
    return torch.tensor([pos_weight], dtype=torch.float32)

# --------------------------------------------------
# Collect probabilities and labels
# --------------------------------------------------
def get_probs_and_labels(model, data_loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    return np.array(all_probs), np.array(all_labels)

# --------------------------------------------------
# Evaluate metrics at a given threshold
# --------------------------------------------------
def evaluate_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

# --------------------------------------------------
# Find best threshold using validation set
# --------------------------------------------------
def find_best_threshold(model, val_loader):
    probs, labels = get_probs_and_labels(model, val_loader)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0

    best_result = None
    best_f1 = -1

    print("\nValidation threshold tuning results:")
    for threshold in THRESHOLDS:
        result = evaluate_at_threshold(probs, labels, threshold)
        print(
            f"Threshold={threshold:.2f} | "
            f"Acc={result['accuracy']:.4f} | "
            f"Prec={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f}"
        )

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_result = result

    best_result["roc_auc"] = roc_auc
    return best_result

# --------------------------------------------------
# Final evaluation on test set
# --------------------------------------------------
def evaluate_model(model, test_loader, threshold):
    probs, labels = get_probs_and_labels(model, test_loader)

    result = evaluate_at_threshold(probs, labels, threshold)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0

    result["roc_auc"] = roc_auc
    return result

# --------------------------------------------------
# Train one local bank model
# --------------------------------------------------
def train_one_bank(bank_name, train_csv, test_csv, save_path):
    print(f"\n========== Training {bank_name} ==========")

    train_loader, val_loader, test_loader, y_train, y_val, y_test = load_bank_data(train_csv, test_csv)

    print(f"{bank_name} train samples: {len(y_train)}")
    print(f"{bank_name} validation samples: {len(y_val)}")
    print(f"{bank_name} test samples: {len(y_test)}")

    model = CreditNet(input_size=len(FEATURE_COLUMNS)).to(DEVICE)

    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    print(f"{bank_name} positive class weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1
    best_model_state = None
    best_threshold = 0.5

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation threshold tuning after each epoch
        val_result = find_best_threshold(model, val_loader)

        print(
            f"\n{bank_name} | Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val F1: {val_result['f1']:.4f} | "
            f"Val ROC-AUC: {val_result['roc_auc']:.4f} | "
            f"Best Threshold: {val_result['threshold']:.2f}"
        )

        if val_result["f1"] > best_val_f1:
            best_val_f1 = val_result["f1"]
            best_threshold = val_result["threshold"]
            best_model_state = model.state_dict()

    # Load best model state before final test evaluation
    model.load_state_dict(best_model_state)

    test_result = evaluate_model(model, test_loader, best_threshold)

    print(f"\n{bank_name} Final Test Results")
    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Accuracy       : {test_result['accuracy']:.4f}")
    print(f"Precision      : {test_result['precision']:.4f}")
    print(f"Recall         : {test_result['recall']:.4f}")
    print(f"F1 Score       : {test_result['f1']:.4f}")
    print(f"ROC-AUC        : {test_result['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(test_result["confusion_matrix"])

    torch.save(model.state_dict(), save_path)
    print(f"{bank_name} model saved to: {save_path}")

# --------------------------------------------------
# Train all 3 banks
# --------------------------------------------------
train_one_bank(
    bank_name="Bank 1",
    train_csv="preprocessing/bank_1/train_processed.csv",
    test_csv="preprocessing/bank_1/test_processed.csv",
    save_path="models/bank_1_model.pth"
)

train_one_bank(
    bank_name="Bank 2",
    train_csv="preprocessing/bank_2/train_processed.csv",
    test_csv="preprocessing/bank_2/test_processed.csv",
    save_path="models/bank_2_model.pth"
)

train_one_bank(
    bank_name="Bank 3",
    train_csv="preprocessing/bank_3/train_processed.csv",
    test_csv="preprocessing/bank_3/test_processed.csv",
    save_path="models/bank_3_model.pth"
)

print("\nUpdated Phase 4 completed successfully.")
print("Saved models:")
print("models/bank_1_model.pth")
print("models/bank_2_model.pth")
print("models/bank_3_model.pth")