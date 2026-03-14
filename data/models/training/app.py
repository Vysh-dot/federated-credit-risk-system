import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

# --------------------------------------------------
# Path handling
# --------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
CURRENT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = CURRENT_DIR.parents[2]  # credit_federated_system

MODELS_DIR = PROJECT_ROOT / "data" / "models"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
PERSONALIZED_MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(MODELS_DIR))
from credit_model import CreditNet  # noqa: E402

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Personalized Federated Credit Score System",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Settings
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

BANK_CONFIG = {
    "Bank 1": {
        "model_path": PERSONALIZED_MODELS_DIR / "personalized_bank_1_model.pth",
        "scaler_path": PREPROCESSING_DIR / "bank_1" / "scaler.pkl",
        "inputs": {
            "age": {"label": "Age", "min": 18, "max": 100, "value": 35, "step": 1},
            "credit_capacity": {"label": "Credit Amount", "min": 0.0, "max": 100000.0, "value": 5000.0, "step": 100.0},
            "monthly_income": {"label": "Estimated Monthly Income Proxy", "min": 0.0, "max": 10000.0, "value": 4000.0, "step": 100.0},
            "debt_ratio": {"label": "Debt Ratio Proxy", "min": 0.0, "max": 2.0, "value": 0.30, "step": 0.01},
            "num_open_accounts": {"label": "Existing Credit Accounts", "min": 0.0, "max": 10.0, "value": 2.0, "step": 1.0},
            "loan_duration": {"label": "Loan Duration (months)", "min": 1.0, "max": 72.0, "value": 24.0, "step": 1.0},
            "installment_rate": {"label": "Installment Rate", "min": 0.0, "max": 5.0, "value": 2.0, "step": 0.1},
            "late_payment_score": {"label": "Repayment Behavior Proxy", "min": 0.0, "max": 5.0, "value": 1.0, "step": 1.0},
            "recent_payment_amount": {"label": "Recent Payment Amount Proxy", "min": 0.0, "max": 10000.0, "value": 500.0, "step": 50.0},
            "credit_utilization": {"label": "Credit Utilization", "min": 0.0, "max": 1.5, "value": 0.30, "step": 0.01},
        },
    },
    "Bank 2": {
        "model_path": PERSONALIZED_MODELS_DIR / "personalized_bank_2_model.pth",
        "scaler_path": PREPROCESSING_DIR / "bank_2" / "scaler.pkl",
        "inputs": {
            "age": {"label": "Age", "min": 18, "max": 100, "value": 35, "step": 1},
            "credit_capacity": {"label": "Credit Limit", "min": 0.0, "max": 500000.0, "value": 50000.0, "step": 1000.0},
            "monthly_income": {"label": "Income Proxy", "min": 0.0, "max": 100000.0, "value": 10000.0, "step": 500.0},
            "debt_ratio": {"label": "Debt Ratio Proxy", "min": -1.0, "max": 5.0, "value": 0.25, "step": 0.01},
            "num_open_accounts": {"label": "Open Billing / Credit Accounts Proxy", "min": 0.0, "max": 6.0, "value": 4.0, "step": 1.0},
            "loan_duration": {"label": "Billing Window Proxy", "min": 1.0, "max": 12.0, "value": 6.0, "step": 1.0},
            "installment_rate": {"label": "Installment Rate Proxy", "min": 0.0, "max": 2.0, "value": 0.10, "step": 0.01},
            "late_payment_score": {"label": "Average Delay Score", "min": -2.0, "max": 10.0, "value": 0.0, "step": 1.0},
            "recent_payment_amount": {"label": "Recent Payment Amount", "min": 0.0, "max": 100000.0, "value": 2000.0, "step": 100.0},
            "credit_utilization": {"label": "Credit Utilization", "min": -1.0, "max": 5.0, "value": 0.30, "step": 0.01},
        },
    },
    "Bank 3": {
        "model_path": PERSONALIZED_MODELS_DIR / "personalized_bank_3_model.pth",
        "scaler_path": PREPROCESSING_DIR / "bank_3" / "scaler.pkl",
        "inputs": {
            "age": {"label": "Age", "min": 18, "max": 100, "value": 35, "step": 1},
            "credit_capacity": {"label": "Credit Capacity Proxy", "min": 0.0, "max": 500000.0, "value": 50000.0, "step": 1000.0},
            "monthly_income": {"label": "Monthly Income", "min": 0.0, "max": 100000.0, "value": 10000.0, "step": 500.0},
            "debt_ratio": {"label": "Debt Ratio", "min": 0.0, "max": 5.0, "value": 0.25, "step": 0.01},
            "num_open_accounts": {"label": "Number of Open Credit Lines / Loans", "min": 0.0, "max": 25.0, "value": 4.0, "step": 1.0},
            "loan_duration": {"label": "Real Estate Loans / Lines Proxy", "min": 1.0, "max": 6.0, "value": 2.0, "step": 1.0},
            "installment_rate": {"label": "Installment Pressure Proxy", "min": 0.0, "max": 5.0, "value": 0.20, "step": 0.01},
            "late_payment_score": {"label": "Past-Due Frequency Score", "min": 0.0, "max": 20.0, "value": 0.0, "step": 1.0},
            "recent_payment_amount": {"label": "Repayment Capacity Proxy", "min": 0.0, "max": 100000.0, "value": 2000.0, "step": 100.0},
            "credit_utilization": {"label": "Revolving Utilization", "min": 0.0, "max": 2.0, "value": 0.20, "step": 0.01},
        },
    },
}

# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    model = CreditNet(input_size=10).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def probability_to_credit_score(default_probability: float) -> int:
    score = int((1 - default_probability) * 550 + 300)
    return max(300, min(850, score))


def get_score_band(score: int) -> str:
    if score >= 750:
        return "Excellent"
    if score >= 650:
        return "Good"
    if score >= 550:
        return "Fair"
    return "Poor"


def get_risk_label(default_probability: float) -> str:
    return "High Risk" if default_probability >= THRESHOLD else "Low Risk"


def detect_out_of_distribution(feature_names, scaled_values):
    warnings = []
    for name, z in zip(feature_names, scaled_values):
        if abs(z) > 4:
            warnings.append(f"{name} is far outside the training range (scaled value {z:.2f}).")
        elif abs(z) > 3:
            warnings.append(f"{name} is somewhat outside the normal training range (scaled value {z:.2f}).")
    return warnings


def explain_credit_score(customer: dict, bank_name: str):
    reasons = []
    suggestions = []
    positives = []

    if customer["debt_ratio"] > 0.60:
        reasons.append("Debt ratio is high relative to safer borrowing behavior.")
        suggestions.append("Reduce overall debt burden.")
    elif customer["debt_ratio"] <= 0.30:
        positives.append("Debt ratio is within a healthy range.")

    if customer["credit_utilization"] > 0.70:
        reasons.append("Credit utilization is high and may signal overuse of available credit.")
        suggestions.append("Keep utilization below 30% if possible.")
    elif customer["credit_utilization"] <= 0.30:
        positives.append("Credit utilization is well controlled.")

    if customer["late_payment_score"] >= 3:
        reasons.append("Past-due or delayed payment behavior increases default risk.")
        suggestions.append("Maintain on-time payments consistently.")
    elif customer["late_payment_score"] == 0:
        positives.append("No major late-payment signal is present.")

    if customer["num_open_accounts"] >= 8:
        reasons.append("A high number of active accounts may weaken the profile.")
        suggestions.append("Avoid opening unnecessary new credit accounts.")

    if customer["monthly_income"] < 4000:
        reasons.append("Income level appears low relative to the credit profile.")
        suggestions.append("Increase stable income or reduce requested exposure.")
    elif customer["monthly_income"] >= 10000:
        positives.append("Income level supports stronger repayment capacity.")

    if customer["recent_payment_amount"] < 500:
        reasons.append("Recent repayment amount appears low.")
        suggestions.append("Increase regular repayment amount if affordable.")

    if bank_name == "Bank 3" and customer["loan_duration"] >= 5:
        reasons.append("Real-estate-loan proxy is high for this bank profile.")
        suggestions.append("Reduce additional secured borrowing where possible.")

    if not reasons:
        reasons.append("The profile looks relatively healthy across the main risk factors.")
        suggestions.append("Maintain current repayment discipline and low credit utilization.")

    return reasons[:5], list(dict.fromkeys(suggestions))[:5], positives[:4]


def feature_status(customer: dict):
    rows = []

    def add_row(name, value, status):
        rows.append({"Feature": name, "Value": value, "Status": status})

    add_row("Age", customer["age"], "Normal")

    def status_from_thresholds(value, good_max=None, moderate_max=None, reverse=False):
        if not reverse:
            if good_max is not None and value <= good_max:
                return "Good"
            if moderate_max is not None and value <= moderate_max:
                return "Moderate"
            return "Risky"
        else:
            if good_max is not None and value >= good_max:
                return "Good"
            if moderate_max is not None and value >= moderate_max:
                return "Moderate"
            return "Risky"

    add_row("Debt Ratio", customer["debt_ratio"], status_from_thresholds(customer["debt_ratio"], 0.30, 0.60))
    add_row("Credit Utilization", customer["credit_utilization"], status_from_thresholds(customer["credit_utilization"], 0.30, 0.70))
    add_row("Late Payment Score", customer["late_payment_score"], status_from_thresholds(customer["late_payment_score"], 0, 2))
    add_row("Open Accounts", customer["num_open_accounts"], status_from_thresholds(customer["num_open_accounts"], 5, 8))
    add_row("Monthly Income", customer["monthly_income"], status_from_thresholds(customer["monthly_income"], 10000, 5000, reverse=True))
    add_row("Recent Payment Amount", customer["recent_payment_amount"], status_from_thresholds(customer["recent_payment_amount"], 1500, 500, reverse=True))

    return pd.DataFrame(rows)


def build_risk_chart_df(customer: dict):
    return pd.DataFrame(
        {
            "Feature": [
                "Debt Ratio",
                "Credit Utilization",
                "Late Payment Score",
                "Open Accounts",
                "Installment Rate",
            ],
            "Value": [
                customer["debt_ratio"],
                customer["credit_utilization"],
                customer["late_payment_score"],
                customer["num_open_accounts"],
                customer["installment_rate"],
            ],
        }
    ).set_index("Feature")


def build_probability_df(default_probability: float):
    return pd.DataFrame(
        {
            "Category": ["Safe Probability", "Default Probability"],
            "Probability": [1 - default_probability, default_probability],
        }
    ).set_index("Category")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("💳 Personalized Federated Credit Score System")
st.caption("Privacy-preserving credit risk prediction with bank-specific personalized federated models.")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Model Settings")
    selected_bank = st.selectbox("Select Bank", list(BANK_CONFIG.keys()))
    st.write(f"**Decision Threshold:** {THRESHOLD}")
    st.write("**Deployment Type:** Personalized Federated Model")

    st.markdown("---")
    st.subheader("Important Note")
    st.write("Inputs are bank-specific proxy features aligned to the training data.")
    st.write("Use realistic values for the selected bank.")

# --------------------------------------------------
# Load bank assets
# --------------------------------------------------
bank_info = BANK_CONFIG[selected_bank]
model = load_model(str(bank_info["model_path"]))
scaler = load_scaler(str(bank_info["scaler_path"]))
input_cfg = bank_info["inputs"]

# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(input_cfg["age"]["label"], min_value=input_cfg["age"]["min"], max_value=input_cfg["age"]["max"], value=input_cfg["age"]["value"], step=input_cfg["age"]["step"])
    credit_capacity = st.number_input(input_cfg["credit_capacity"]["label"], min_value=input_cfg["credit_capacity"]["min"], max_value=input_cfg["credit_capacity"]["max"], value=input_cfg["credit_capacity"]["value"], step=input_cfg["credit_capacity"]["step"])
    monthly_income = st.number_input(input_cfg["monthly_income"]["label"], min_value=input_cfg["monthly_income"]["min"], max_value=input_cfg["monthly_income"]["max"], value=input_cfg["monthly_income"]["value"], step=input_cfg["monthly_income"]["step"])
    debt_ratio = st.number_input(input_cfg["debt_ratio"]["label"], min_value=input_cfg["debt_ratio"]["min"], max_value=input_cfg["debt_ratio"]["max"], value=input_cfg["debt_ratio"]["value"], step=input_cfg["debt_ratio"]["step"])
    num_open_accounts = st.number_input(input_cfg["num_open_accounts"]["label"], min_value=input_cfg["num_open_accounts"]["min"], max_value=input_cfg["num_open_accounts"]["max"], value=input_cfg["num_open_accounts"]["value"], step=input_cfg["num_open_accounts"]["step"])

with col2:
    loan_duration = st.number_input(input_cfg["loan_duration"]["label"], min_value=input_cfg["loan_duration"]["min"], max_value=input_cfg["loan_duration"]["max"], value=input_cfg["loan_duration"]["value"], step=input_cfg["loan_duration"]["step"])
    installment_rate = st.number_input(input_cfg["installment_rate"]["label"], min_value=input_cfg["installment_rate"]["min"], max_value=input_cfg["installment_rate"]["max"], value=input_cfg["installment_rate"]["value"], step=input_cfg["installment_rate"]["step"])
    late_payment_score = st.number_input(input_cfg["late_payment_score"]["label"], min_value=input_cfg["late_payment_score"]["min"], max_value=input_cfg["late_payment_score"]["max"], value=input_cfg["late_payment_score"]["value"], step=input_cfg["late_payment_score"]["step"])
    recent_payment_amount = st.number_input(input_cfg["recent_payment_amount"]["label"], min_value=input_cfg["recent_payment_amount"]["min"], max_value=input_cfg["recent_payment_amount"]["max"], value=input_cfg["recent_payment_amount"]["value"], step=input_cfg["recent_payment_amount"]["step"])
    credit_utilization = st.number_input(input_cfg["credit_utilization"]["label"], min_value=input_cfg["credit_utilization"]["min"], max_value=input_cfg["credit_utilization"]["max"], value=input_cfg["credit_utilization"]["value"], step=input_cfg["credit_utilization"]["step"])

predict = st.button("Predict Credit Score", use_container_width=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predict:
    customer = {
        "age": age,
        "credit_capacity": credit_capacity,
        "monthly_income": monthly_income,
        "debt_ratio": debt_ratio,
        "num_open_accounts": num_open_accounts,
        "loan_duration": loan_duration,
        "installment_rate": installment_rate,
        "late_payment_score": late_payment_score,
        "recent_payment_amount": recent_payment_amount,
        "credit_utilization": credit_utilization,
    }

    features = np.array([[customer[col] for col in FEATURE_COLUMNS]], dtype=float)
    features_scaled = scaler.transform(features)
    scaled_row = features_scaled[0]
    ood_warnings = detect_out_of_distribution(FEATURE_COLUMNS, scaled_row)

    tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        default_probability = torch.sigmoid(logits).item()

    credit_score = probability_to_credit_score(default_probability)
    score_band = get_score_band(credit_score)
    risk_label = get_risk_label(default_probability)
    reasons, suggestions, positives = explain_credit_score(customer, selected_bank)

    st.markdown("---")
    st.subheader("Prediction Result")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Bank", selected_bank)
    m2.metric("Credit Score", credit_score)
    m3.metric("Default Probability", f"{default_probability:.6%}")
    m4.metric("Score Band", score_band)
    st.write(f"Raw probability: {default_probability:.10f}")

    if risk_label == "High Risk":
        st.error(f"Prediction: {risk_label} / Possible Default")
    else:
        st.success(f"Prediction: {risk_label} / Safer Borrower")

    if ood_warnings:
        st.warning("Prediction reliability warning: some entered values are outside the normal training range for this bank.")
        for w in ood_warnings[:5]:
            st.write(f"- {w}")

    left, right = st.columns(2)

    with left:
        st.subheader("Probability Overview")
        st.bar_chart(build_probability_df(default_probability))
        st.progress(float(1 - default_probability), text=f"Credit Strength: {(1 - default_probability):.2%}")

    with right:
        st.subheader("Risk Factor Snapshot")
        st.bar_chart(build_risk_chart_df(customer))

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Why this score?")
        for r in reasons:
            st.write(f"- {r}")

        if positives:
            st.subheader("Positive signals")
            for p in positives:
                st.write(f"- {p}")

    with c2:
        st.subheader("How to improve this score")
        for s in suggestions:
            st.write(f"- {s}")

    st.subheader("Feature Health Summary")
    status_df = feature_status(customer)

    def style_status(val):
        if val == "Good":
            return "background-color: #d4edda; color: #155724;"
        if val == "Moderate":
            return "background-color: #fff3cd; color: #856404;"
        if val == "Risky":
            return "background-color: #f8d7da; color: #721c24;"
        return ""

    st.dataframe(
        status_df.style.map(style_status, subset=["Status"]),
        use_container_width=True
    )

    with st.expander("Show entered customer values"):
        st.json(customer)

    with st.expander("Show scaled feature values (debug)"):
        scaled_df = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Scaled Value": scaled_row}
        )
        st.dataframe(scaled_df, use_container_width=True)

else:
    st.info("Enter customer details and click **Predict Credit Score**.")