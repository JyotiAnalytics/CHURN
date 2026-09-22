# CHURN
Telecom Customer Churn Prediction using Machine Learning and Deep Learning


# 📉 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-f7931e?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end churn prediction project on the IBM Telco Customer Churn dataset — data cleaning, EDA, feature engineering, and model building using both classical ML (Logistic Regression, Random Forest, Gradient Boosting) and Deep Learning (Keras/TensorFlow), with model explainability and a customer-level churn risk scoring system.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#️-dataset)
- [Workflow](#-workflow)
- [Visuals](#️-visuals)
- [Results](#-results)
- [Tech Stack](#️-tech-stack)
- [Repository Structure](#-repository-structure)
- [How to Run](#️-how-to-run)
- [Future Improvements](#-future-improvements)

## 📌 Project Overview

Customer churn — when a customer stops using a company's service — is one of the most costly problems for subscription-based businesses. This project builds and compares multiple models to predict which telecom customers are likely to churn, and turns those predictions into an actionable **risk score** for each customer.

## 🗂️ Dataset

- **Source:** [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- ~7,000 customer records with demographic info, account info, and service usage details
- Target variable: `Churn` (Yes/No)

## 🔧 Workflow

1. **Data Cleaning** — handled missing values, fixed data types, removed duplicates
2. **Data Manipulation** — filtered/segmented customers on multiple business conditions (e.g. gender, internet service, payment method, tenure)
3. **Exploratory Data Analysis** — churn distribution, churn by internet service/contract/payment method
4. **Feature Engineering** — derived features such as average monthly charge, new-customer flag, high-monthly-charge flag
5. **Model Building**
   - Deep Learning (Keras Sequential models — with and without Dropout, single vs multi-feature)
   - Logistic Regression, Random Forest, Gradient Boosting
6. **Model Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
7. **Explainability** — Random Forest feature importance, permutation importance, SHAP
8. **Churn Risk Scoring** — every customer scored and bucketed into Low / Medium / High risk
9. **Model & Result Saving** — trained models (`.pkl` / `.keras`) and result CSVs saved for reuse

## 🖼️ Visuals

| Churn Distribution | Churn by Internet Service |
|---|---|
| ![Churn Distribution](Customer%20Churn%20Distribution.png) | ![Churn by Internet Service](Churn%20by%20Internet%20Service.png) |

| Churn by Payment Method | Feature Importance |
|---|---|
| ![Churn by Payment Method](Churn%20by%20Payment%20Method.png) | ![Feature Importance](feature_importance.png) |

> Replace/add the `.png` filenames above to match whatever your script actually saves — all charts are auto-saved when you run `Churn.py`.

## 📊 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.50% |
| Random Forest | 76.08% |
| Gradient Boosting | 78.64% |
| Deep Learning (3 features) | 78.78% |

**138 high-risk customers** identified and exported to `high_risk_customers.csv` for potential retention action.

> Note: Accuracy alone doesn't tell the full story here — the dataset is imbalanced, so recall on the churn class is also tracked (see `complete_model_results.csv`) and is an area flagged for further tuning (e.g. class rebalancing, threshold adjustment).

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `TensorFlow / Keras` · `SHAP` · `Joblib`

## 📁 Repository Structure

```
├── Churn.py                          # Main script — full pipeline
├── README.md
├── *.png                             # Saved EDA & evaluation charts
├── high_risk_customers.csv           # Exported high-risk customer list
├── customer_churn_risk_results.csv   # Full risk-scored customer results
├── complete_model_results.csv        # Accuracy/precision/recall/F1 for all models
└── *.pkl / *.keras                   # Saved trained models & scalers
```

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap joblib
python Churn.py
```

Place the dataset CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the same folder as the script before running.

## 🚀 Future Improvements

- Address class imbalance (SMOTE / class-weight tuning) to improve churn-class recall
- Hyperparameter tuning via GridSearchCV / Optuna
- Deploy the best model as a simple API or dashboard (Streamlit/Flask)

## 👤 Author

*Jyoti Ranjan Bhanja*
