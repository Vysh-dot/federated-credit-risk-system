import os
import joblib
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------
# Create folders to save preprocessing artifacts
# --------------------------------------------------
os.makedirs("preprocessing/bank_1", exist_ok=True)
os.makedirs("preprocessing/bank_2", exist_ok=True)
os.makedirs("preprocessing/bank_3", exist_ok=True)

# --------------------------------------------------
# Settings
# --------------------------------------------------
BATCH_SIZE = 64
TEST_SIZE = 0.2
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

# --------------------------------------------------
# Function to preprocess one bank dataset
# --------------------------------------------------
def preprocess_bank(bank_name, input_path, output_dir):
    print(f"\nProcessing {bank_name}...")
    
    # Load bank dataset
    df = pd.read_csv(input_path)
    print(f"{bank_name} dataset shape: {df.shape}")

    # Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"{bank_name} feature shape: {X.shape}")
    print(f"{bank_name} target shape: {y.shape}")

    # Train-test split with stratification to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"{bank_name} X_train shape: {X_train.shape}")
    print(f"{bank_name} X_test shape: {X_test.shape}")
    print(f"{bank_name} y_train distribution:\n{y_train.value_counts()}")
    print(f"{bank_name} y_test distribution:\n{y_test.value_counts()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Save processed arrays as CSV for inspection
    train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
    train_df[TARGET_COLUMN] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
    test_df[TARGET_COLUMN] = y_test.values

    train_csv_path = os.path.join(output_dir, "train_processed.csv")
    test_csv_path = os.path.join(output_dir, "test_processed.csv")

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"{bank_name} scaler saved to: {scaler_path}")
    print(f"{bank_name} processed train data saved to: {train_csv_path}")
    print(f"{bank_name} processed test data saved to: {test_csv_path}")
    print(f"{bank_name} train batches: {len(train_loader)}")
    print(f"{bank_name} test batches: {len(test_loader)}")

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "input_size": X_train_tensor.shape[1],
        "train_shape": X_train_tensor.shape,
        "test_shape": X_test_tensor.shape
    }

# --------------------------------------------------
# Run preprocessing for all banks
# --------------------------------------------------
bank1 = preprocess_bank(
    bank_name="Bank 1",
    input_path="datasets/bank_1/data.csv",
    output_dir="preprocessing/bank_1"
)

bank2 = preprocess_bank(
    bank_name="Bank 2",
    input_path="datasets/bank_2/data.csv",
    output_dir="preprocessing/bank_2"
)

bank3 = preprocess_bank(
    bank_name="Bank 3",
    input_path="datasets/bank_3/data.csv",
    output_dir="preprocessing/bank_3"
)

print("\nPhase 3 completed successfully.")
print("\nInput size for model:", bank1["input_size"])
print("Bank 1 train shape:", bank1["train_shape"], "| test shape:", bank1["test_shape"])
print("Bank 2 train shape:", bank2["train_shape"], "| test shape:", bank2["test_shape"])
print("Bank 3 train shape:", bank3["train_shape"], "| test shape:", bank3["test_shape"])