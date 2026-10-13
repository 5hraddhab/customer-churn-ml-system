# AI-Based Customer Churn Prediction and Customer Retention System

An end-to-end, generalized, configurable Machine Learning application designed to predict customer churn and recommend targeted retention strategies across multiple business verticals including Banking, Telecommunications, E-Commerce, Retail, and Subscription/SaaS services.

---

## 📌 Project Overview

Customer churn occurs when customers terminate their relationship with a business. Traditional churn prediction systems are frequently rigid—tailored only to specific SaaS subscription models with hardcoded columns and static metrics.

This project delivers a **generalized, schema-agnostic machine learning framework** that:
1. **Accepts any customer churn dataset** (CSV/Excel) with arbitrary feature names.
2. **Automatically audits data quality**, missing values, cardinality, and class balance.
3. **Builds a leak-free preprocessing pipeline** (fitting transforms strictly on training splits).
4. **Benchmarks three classical ML algorithms:** Logistic Regression, Decision Tree, and Random Forest.
5. **Provides real-time and batch churn prediction** with calibrated risk categories (Low, Medium, High).
6. **Attributes prediction drivers** to explain *why* a customer is at risk of churning.
7. **Generates configurable retention recommendations** tailored to specific industry business profiles.

---

## 🏗️ Project Architecture & Structure

```
Aiml.Project/
│
├── Data/
│   └── train.csv                   # Raw original dataset (untouched)
│
├── models/
│   ├── .gitkeep
│   └── best_model.joblib           # Serialized best model & fitted preprocessor
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # File loading, type detection & quality inspection
│   ├── data_preprocessing.py       # Generalized ColumnTransformer pipeline (leak-free)
│   ├── train_model.py              # Tri-model training, stratified split & evaluation
│   ├── predict.py                  # Single & batch inference, risk tiering & factor attribution
│   ├── analytics.py                # Adaptive Plotly interactive visualizations
│   └── recommendations.py          # Industry-specific retention playbooks & simulator
│
├── app.py                          # Multi-page interactive Streamlit dashboard
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation & presentation guide
```

---

## ⚙️ Core Technical Capabilities

### 1. Generalized Dataset Upload & Target Mapping
- **Schema-Agnostic Loading:** Supports both CSV (with UTF-8 and Latin-1 fallback) and Excel (`.xlsx`, `.xls`).
- **Flexible Target Selection:** Enables the user to designate any binary target column and map custom values (e.g. `1` vs `0`, `Yes` vs `No`, `Churned` vs `Retained`).
- **Identifier Contamination Guard:** Automatically detects ID columns (e.g., `Customer_ID`, UUIDs) and excludes them from model training.

### 2. Zero Data Leakage Guarantee
- Preprocessing steps (imputation, numerical scaling, and one-hot encoding) are **fitted strictly on the training partition (`X_train`)**.
- The validation/test partition (`X_test`) and future inference records are only transformed using the pre-fitted parameters.

### 3. Model Training & Comparison
The system trains and benchmarks three complementary classifiers:
- **Logistic Regression:** Linear decision boundary, serves as a high-interpretability baseline with balanced class weights.
- **Decision Tree Classifier:** Non-linear rule-based partitioning with regularized depth to prevent overfitting.
- **Random Forest Classifier:** Ensemble bagging of decision trees reducing variance and capturing feature interactions.

**Evaluation Metrics Computed:**
- **Accuracy:** Overall correctness.
- **Precision (Churn Class):** Proportion of predicted churners who actually churned.
- **Recall (Churn Class):** Proportion of actual churners successfully identified (vital for churn mitigation).
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve.
- **Confusion Matrix:** True Negatives, False Positives, False Negatives, True Positives.

### 4. Empirical Evaluation on the Default Dataset (`Data/train.csv`)
On the provided 8,000-record customer dataset (31.31% churn prevalence, 20% stratified test split):
- **Decision Tree** achieves the highest **Recall (~74.65%)** and **F1-Score (~0.4400)**, capturing the majority of churners.
- **Random Forest** achieves the highest overall **Accuracy (~57.31%)**.
- **Logistic Regression** achieves **~52.00% Accuracy** and **~48.50% Recall**.

*(All metrics are computed on the held-out test split; no values are fabricated).*

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 1. Virtual Environment Setup
```bash
# Using Python venv
python -m venv .venv

# Activate venv:
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```
Or with explicit virtual environment path:
```powershell
.venv\Scripts\streamlit.exe run app.py
```

Open your browser to `http://localhost:8501`.




