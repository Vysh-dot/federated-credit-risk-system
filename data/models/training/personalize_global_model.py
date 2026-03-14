import sys
import os
import copy
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
LEARNING_RATE = 0.0005
FINE_TUNE_EPOCHS = 3
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.55

GLOBAL_MODEL_PATH = "models/global_credit_model.pth"

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

BANK_FILES = {
    "Bank 1": {
        "train": "preprocessing/bank_1/train_processed.csv",
        "test": "preprocessing/bank_1/test_processed.csv",
        "save": "models/personalized_bank_1_model.pth"
    },
    "Bank 2": {
        "train": "preprocessing/bank_2/train_processed.csv",
        "test": "preprocessing/bank_2/test_processed.csv",
        "save": "models/personalized_bank_2_model.pth"
    },
    "Bank 3": {
        "train": "preprocessing/bank_3/train_processed.csv",
        "test": "preprocessing/bank_3/test_processed.csv",
        "save": "models/personalized_bank_3_model.pth"
    }
}

# --------------------------------------------------
# Load bank train/test data
# --------------------------------------------------
def load_bank_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train_full = train_df[FEATURE_COLUMNS].values
    y_train_full = train_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, y_train

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
# Get probabilities and labels
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
# Evaluate model
# --------------------------------------------------
def evaluate_model(model, data_loader, threshold=0.55):
    probs, labels = get_probs_and_labels(model, data_loader)
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

# --------------------------------------------------
# Fine-tune global model for one bank
# --------------------------------------------------
def personalize_for_bank(bank_name, train_csv, test_csv, save_path):
    print(f"\n========== Personalizing for {bank_name} ==========")

    train_loader, val_loader, test_loader, y_train = load_bank_data(train_csv, test_csv)

    # Load global model
    model = CreditNet(input_size=10).to(DEVICE)
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE))

    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1

    for epoch in range(FINE_TUNE_EPOCHS):
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

        val_metrics = evaluate_model(model, val_loader, threshold=THRESHOLD)

        print(
            f"{bank_name} | Epoch [{epoch+1}/{FINE_TUNE_EPOCHS}] | "
            f"Loss: {avg_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best personalized version
    model.load_state_dict(best_model_state)

    test_metrics = evaluate_model(model, test_loader, threshold=THRESHOLD)

    print(f"\n{bank_name} Personalized Model Test Results")
    print(f"Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Precision  : {test_metrics['precision']:.4f}")
    print(f"Recall     : {test_metrics['recall']:.4f}")
    print(f"F1 Score   : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    torch.save(model.state_dict(), save_path)
    print(f"{bank_name} personalized model saved to: {save_path}")

# --------------------------------------------------
# Main
# --------------------------------------------------
print("\n========== Phase 7: Personalized Federated Fine-Tuning ==========")
print(f"Global base model: {GLOBAL_MODEL_PATH}")
print(f"Fine-tuning epochs: {FINE_TUNE_EPOCHS}")
print(f"Threshold: {THRESHOLD}")

for bank_name, paths in BANK_FILES.items():
    personalize_for_bank(
        bank_name=bank_name,
        train_csv=paths["train"],
        test_csv=paths["test"],
        save_path=paths["save"]
    )

print("\nPhase 7 completed successfully.")
print("Generated personalized models:")
print("models/personalized_bank_1_model.pth")
print("models/personalized_bank_2_model.pth")
print("models/personalized_bank_3_model.pth")