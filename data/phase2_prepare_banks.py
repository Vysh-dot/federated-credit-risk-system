import os
import pandas as pd
import numpy as np

# -----------------------------
# Create output folders
# -----------------------------
os.makedirs("datasets/bank_1", exist_ok=True)
os.makedirs("datasets/bank_2", exist_ok=True)
os.makedirs("datasets/bank_3", exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
german_df = pd.read_csv("data/german_credit_data.csv")
uci_df = pd.read_csv("data/UCI_Credit_Card.csv")
credit_df = pd.read_csv("data/cs-training.csv")

print("Datasets loaded successfully.")
print("German Credit shape:", german_df.shape)
print("UCI Credit Card shape:", uci_df.shape)
print("Give Me Some Credit shape:", credit_df.shape)

# =========================================================
# BANK 1: German Credit -> Common Schema
# =========================================================
bank1 = pd.DataFrame()

# Direct mappings / proxies
bank1["age"] = german_df["alter"]
bank1["credit_capacity"] = german_df["hoehe"]  # loan / credit amount
bank1["monthly_income"] = german_df["hoehe"] / german_df["laufzeit"].replace(0, 1)  # proxy
bank1["debt_ratio"] = german_df["rate"] / 10.0  # normalized proxy
bank1["num_open_accounts"] = german_df["bishkred"]
bank1["loan_duration"] = german_df["laufzeit"]
bank1["installment_rate"] = german_df["rate"]
bank1["late_payment_score"] = german_df["moral"]  # repayment-behavior proxy
bank1["recent_payment_amount"] = german_df["hoehe"] / german_df["laufzeit"].replace(0, 1)  # proxy
bank1["credit_utilization"] = german_df["hoehe"] / (german_df["hoehe"].max() + 1e-6)

# Target standardization
# Here we assume:
# kredit = 1 -> good credit
# kredit = 0 or 2 -> risky/default
# Adjust if needed after checking value_counts()
if set(german_df["kredit"].unique()) == {1, 2}:
    bank1["target"] = german_df["kredit"].map({1: 0, 2: 1})
else:
    bank1["target"] = german_df["kredit"]

# =========================================================
# BANK 2: UCI Credit Card -> Common Schema
# =========================================================
bank2 = pd.DataFrame()

bank2["age"] = uci_df["AGE"]
bank2["credit_capacity"] = uci_df["LIMIT_BAL"]

# No direct income column -> use credit limit proxy
bank2["monthly_income"] = uci_df["LIMIT_BAL"] / 5.0

# Debt ratio proxy using latest bill / credit limit
bank2["debt_ratio"] = uci_df["BILL_AMT1"] / (uci_df["LIMIT_BAL"] + 1e-6)

# Open accounts proxy
bank2["num_open_accounts"] = (
    (uci_df["BILL_AMT1"] > 0).astype(int) +
    (uci_df["BILL_AMT2"] > 0).astype(int) +
    (uci_df["BILL_AMT3"] > 0).astype(int) +
    (uci_df["BILL_AMT4"] > 0).astype(int) +
    (uci_df["BILL_AMT5"] > 0).astype(int) +
    (uci_df["BILL_AMT6"] > 0).astype(int)
)

# No direct duration -> use fixed/average proxy
bank2["loan_duration"] = 6

# Installment rate proxy using recent payment / limit
bank2["installment_rate"] = uci_df["PAY_AMT1"] / (uci_df["LIMIT_BAL"] + 1e-6)

# Average repayment delay score
bank2["late_payment_score"] = uci_df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)

bank2["recent_payment_amount"] = uci_df["PAY_AMT1"]
bank2["credit_utilization"] = uci_df["BILL_AMT1"] / (uci_df["LIMIT_BAL"] + 1e-6)

# Target already means default next month:
# 0 -> non-default / low risk
# 1 -> default / high risk
bank2["target"] = uci_df["default.payment.next.month"]

# =========================================================
# BANK 3: Give Me Some Credit -> Common Schema
# =========================================================
bank3 = pd.DataFrame()

# Fill missing values first
credit_df["MonthlyIncome"] = credit_df["MonthlyIncome"].fillna(credit_df["MonthlyIncome"].median())
credit_df["NumberOfDependents"] = credit_df["NumberOfDependents"].fillna(credit_df["NumberOfDependents"].median())

bank3["age"] = credit_df["age"]

# No direct credit amount -> estimate from income and debt ratio
bank3["credit_capacity"] = credit_df["MonthlyIncome"] * (1 + credit_df["DebtRatio"])

bank3["monthly_income"] = credit_df["MonthlyIncome"]
bank3["debt_ratio"] = credit_df["DebtRatio"]
bank3["num_open_accounts"] = credit_df["NumberOfOpenCreditLinesAndLoans"]

# No direct duration -> use number of real estate loans / lines as proxy
bank3["loan_duration"] = credit_df["NumberRealEstateLoansOrLines"] + 1

# Installment rate proxy
bank3["installment_rate"] = credit_df["DebtRatio"] / (credit_df["NumberOfOpenCreditLinesAndLoans"] + 1e-6)

# Late payment score from delinquency columns
bank3["late_payment_score"] = (
    credit_df["NumberOfTime30-59DaysPastDueNotWorse"] +
    credit_df["NumberOfTime60-89DaysPastDueNotWorse"] +
    credit_df["NumberOfTimes90DaysLate"]
)

# Recent payment amount proxy
bank3["recent_payment_amount"] = credit_df["MonthlyIncome"] / (credit_df["DebtRatio"] + 1)

bank3["credit_utilization"] = credit_df["RevolvingUtilizationOfUnsecuredLines"]

# Target already means serious delinquency in 2 years
bank3["target"] = credit_df["SeriousDlqin2yrs"]

# =========================================================
# Clean all outputs
# =========================================================
def clean_dataframe(df, name):
    df = df.copy()

    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill any remaining missing numeric values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Make sure target is integer
    df["target"] = df["target"].astype(int)

    print(f"\n{name} summary:")
    print(df.head())
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    return df

bank1 = clean_dataframe(bank1, "Bank 1")
bank2 = clean_dataframe(bank2, "Bank 2")
bank3 = clean_dataframe(bank3, "Bank 3")

# =========================================================
# Save datasets
# =========================================================
bank1.to_csv("datasets/bank_1/data.csv", index=False)
bank2.to_csv("datasets/bank_2/data.csv", index=False)
bank3.to_csv("datasets/bank_3/data.csv", index=False)

print("\nPhase 2 completed successfully.")
print("Saved files:")
print("datasets/bank_1/data.csv")
print("datasets/bank_2/data.csv")
print("datasets/bank_3/data.csv")