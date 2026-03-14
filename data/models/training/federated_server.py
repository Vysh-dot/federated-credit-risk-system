import sys
import os
import copy
import torch

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

os.makedirs("models", exist_ok=True)

MODEL_PATHS = {
    "bank_1": "models/bank_1_model.pth",
    "bank_2": "models/bank_2_model.pth",
    "bank_3": "models/bank_3_model.pth"
}

# Training sample sizes from Phase 3
BANK_SIZES = {
    "bank_1": 800,
    "bank_2": 24000,
    "bank_3": 120000
}

# --------------------------------------------------
# Choose aggregation mode here:
# "weighted" -> uses dataset sizes
# "equal"    -> all banks get same weight
# "custom"   -> uses CUSTOM_WEIGHTS below
# --------------------------------------------------
AGGREGATION_MODE = "weighted"

CUSTOM_WEIGHTS = {
    "bank_1": 0.15,
    "bank_2": 0.35,
    "bank_3": 0.50
}

GLOBAL_MODEL_PATH = f"models/global_credit_model.pth"

# --------------------------------------------------
# Load local model state dicts
# --------------------------------------------------
def load_local_models():
    local_states = {}

    for bank_name, model_path in MODEL_PATHS.items():
        model = CreditNet(input_size=10).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        local_states[bank_name] = copy.deepcopy(model.state_dict())
        print(f"{bank_name} model loaded from: {model_path}")

    return local_states

# --------------------------------------------------
# Get aggregation weights
# --------------------------------------------------
def get_aggregation_weights(mode, bank_sizes, custom_weights=None):
    if mode == "weighted":
        total_samples = sum(bank_sizes.values())
        weights = {
            bank_name: size / total_samples
            for bank_name, size in bank_sizes.items()
        }

    elif mode == "equal":
        num_banks = len(bank_sizes)
        weights = {
            bank_name: 1.0 / num_banks
            for bank_name in bank_sizes.keys()
        }

    elif mode == "custom":
        if custom_weights is None:
            raise ValueError("CUSTOM_WEIGHTS must be provided for custom mode.")

        total_weight = sum(custom_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of custom weights cannot be zero.")

        weights = {
            bank_name: custom_weights[bank_name] / total_weight
            for bank_name in bank_sizes.keys()
        }

    else:
        raise ValueError("Invalid aggregation mode. Use 'weighted', 'equal', or 'custom'.")

    return weights

# --------------------------------------------------
# Federated averaging
# --------------------------------------------------
def federated_average(local_states, aggregation_weights):
    global_state = copy.deepcopy(next(iter(local_states.values())))

    for key in global_state.keys():
        weighted_sum = 0

        for bank_name, state in local_states.items():
            weight = aggregation_weights[bank_name]
            weighted_sum += state[key] * weight

        global_state[key] = weighted_sum

    return global_state

# --------------------------------------------------
# Save global model
# --------------------------------------------------
def save_global_model(global_state, save_path):
    global_model = CreditNet(input_size=10).to(DEVICE)
    global_model.load_state_dict(global_state)

    torch.save(global_model.state_dict(), save_path)
    print(f"\nGlobal model saved to: {save_path}")

# --------------------------------------------------
# Main
# --------------------------------------------------
print("\n========== Phase 5: Improved Federated Server ==========")
print(f"Aggregation mode: {AGGREGATION_MODE}")

local_states = load_local_models()

aggregation_weights = get_aggregation_weights(
    mode=AGGREGATION_MODE,
    bank_sizes=BANK_SIZES,
    custom_weights=CUSTOM_WEIGHTS
)

print("\nAggregation weights being used:")
for bank_name, weight in aggregation_weights.items():
    print(f"{bank_name}: {weight * 100:.2f}%")

global_state = federated_average(local_states, aggregation_weights)
save_global_model(global_state, GLOBAL_MODEL_PATH)

print("\nPhase 5 completed successfully.")
print("Generated global model:")
print(GLOBAL_MODEL_PATH)