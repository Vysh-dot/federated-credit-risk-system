# Personalized Federated Credit Risk Prediction System

A **privacy-preserving credit risk prediction platform** built using **Federated Learning and personalized bank models**.
The system enables multiple banks to collaboratively train a credit risk model **without sharing raw customer data**, improving prediction accuracy while maintaining privacy.

---

## Overview

Traditional credit scoring systems require centralizing sensitive financial data from multiple institutions. This introduces major **privacy, regulatory, and security risks**.

This project implements a **Federated Learning pipeline** where:

1. Each bank trains a **local credit risk model**
2. A **federated server aggregates the models**
3. A **global model is created**
4. Each bank performs **personalized fine-tuning**
5. A **Streamlit dashboard predicts credit scores and explains risk factors**

The result is a **privacy-preserving and interpretable credit scoring system**.

---

## Key Features

### Federated Learning

Multiple banks collaboratively train a global model **without sharing their raw data**.

### Personalized Models

Each bank fine-tunes the global model on its own data to improve local performance.

### Explainable Credit Scoring

The system explains **why a credit score is high or low** and provides suggestions for improvement.

### Interactive Deployment

A **Streamlit web app** allows users to input customer details and obtain:

* Credit score
* Default probability
* Risk band
* Explanation of risk factors
* Improvement suggestions
* Visual charts

---

## System Architecture

Local Bank Training → Federated Aggregation → Global Model → Personalized Fine-Tuning → Streamlit Deployment

**Pipeline**

1. Data preprocessing and feature alignment
2. Local model training at each bank
3. Federated averaging to produce a global model
4. Personalized fine-tuning for each bank
5. Global threshold tuning for optimal classification
6. Interactive deployment with explanations

---

## Project Structure

```
credit_federated_system/
│
├── data/
│   └── datasets used for training
│
├── preprocessing/
│   └── data preprocessing and feature scaling
│
├── models/
│   ├── local bank models
│   ├── global federated model
│   └── personalized bank models
│
├── training/
│   ├── train_local.py
│   ├── federated_server.py
│   ├── evaluate_global_model.py
│   ├── personalize_global_model.py
│   └── tune_global_threshold.py
│
├── deployment/
│   └── Streamlit app for prediction
│
└── README.md
```

---

## Machine Learning Pipeline

### Phase 1 — Data Preparation

* Dataset cleaning
* Feature normalization
* Train/test splitting

### Phase 2 — Local Model Training

Each bank trains its own **credit risk neural network model**.

### Phase 3 — Federated Aggregation

A **federated server aggregates model parameters** using weighted or equal averaging.

### Phase 4 — Global Model Evaluation

The global model is evaluated across all bank datasets.

### Phase 5 — Threshold Optimization

Optimal classification threshold selected using **macro-average F1 score**.

### Phase 6 — Personalized Fine-Tuning

Each bank fine-tunes the global model to create **bank-specific personalized models**.

### Phase 7 — Deployment

A Streamlit app performs **real-time credit score prediction and explanation**.

---

## Model Architecture

The system uses a **feedforward neural network implemented in PyTorch**.

Input features include:

* Age
* Credit capacity
* Monthly income
* Debt ratio
* Number of open accounts
* Loan duration proxy
* Installment rate
* Late payment score
* Recent payment amount
* Credit utilization

Output:

* Default probability
* Credit score (300–850 scale)

---

## Example Prediction Output

The deployed system provides:

Credit Score: 381
Default Probability: 85.16%
Risk Band: Poor

The app also shows:

* Risk factor charts
* Explanation of negative factors
* Positive signals
* Suggestions for improving credit score

---

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* Federated Learning
* Streamlit
* NumPy
* Pandas
* Matplotlib

---

## Privacy Advantages

This system demonstrates how financial institutions can:

* Collaboratively improve models
* Preserve customer privacy
* Avoid centralized data storage
* Comply with data protection regulations

---

## Possible Future Improvements

* Differential Privacy integration
* Secure aggregation protocols
* Explainable AI using SHAP or LIME
* Multi-round federated training
* Real-time API deployment



