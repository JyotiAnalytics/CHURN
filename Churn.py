# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION - COMPLETE PROJECT
# Data Science + Machine Learning + Deep Learning
# Feature Engineering + SHAP + Churn Risk + AI Explanation
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import joblib


# ============================================================
# 2. LOAD CSV FILE
# ============================================================

import pandas as pd
import os

file_path = "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"

print("Looking for file:", file_path)

if not os.path.exists(file_path):
    print("Current folder:", os.getcwd())
else:
    df = pd.read_csv(file_path)

    print("\n CSV file loaded successfully!")
    print("Dataset Shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nCleaned Columns:")
print(df.columns.tolist())


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "gender",
    "SeniorCitizen",
    "InternetService",
    "PaymentMethod",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nMissing Columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing from CSV file."
    )

print("\nAll required columns are available!")


# ============================================================
# 6. CLEAN NUMERIC COLUMNS
# ============================================================

df["tenure"] = pd.to_numeric(
    df["tenure"],
    errors="coerce"
)

df["MonthlyCharges"] = pd.to_numeric(
    df["MonthlyCharges"],
    errors="coerce"
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(df.isnull().sum())

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for col in numeric_columns:

    df[col] = df[col].fillna(
        df[col].median()
    )

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 8. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "\nDuplicates removed:",
        duplicate_count
    )

else:

    print("\nNo duplicate rows found.")


# ============================================================
# A) DATA MANIPULATION
# ============================================================

print("\n" + "=" * 70)
print("A) DATA MANIPULATION")
print("=" * 70)


# ------------------------------------------------------------
# A(a) TOTAL NUMBER OF MALE CUSTOMERS
# ------------------------------------------------------------

male_customers = df[
    df["gender"] == "Male"
]

print(
    "\nA(a) Total number of male customers:",
    len(male_customers)
)


# ------------------------------------------------------------
# A(b) TOTAL CUSTOMERS WHOSE INTERNET SERVICE IS DSL
# ------------------------------------------------------------

dsl_customers = df[
    df["InternetService"] == "DSL"
]

print(
    "\nA(b) Total customers with DSL:",
    len(dsl_customers)
)


# ------------------------------------------------------------
# A(c) FEMALE SENIOR CITIZENS + MAILED CHECK
# ------------------------------------------------------------

new_customer = df[
    (df["gender"] == "Female") &
    (df["SeniorCitizen"] == 1) &
    (df["PaymentMethod"] == "Mailed check")
]

print(
    "\nA(c) Female senior citizens with Mailed check:"
)

print(new_customer)

print(
    "\nTotal:",
    len(new_customer)
)


# ------------------------------------------------------------
# A(d) TENURE < 10 OR TOTAL CHARGES < 500
# ------------------------------------------------------------

new_customer = df[
    (df["tenure"] < 10) |
    (df["TotalCharges"] < 500)
]

print(
    "\nA(d) Customers whose tenure < 10 OR TotalCharges < 500:"
)

print(new_customer)

print(
    "\nTotal:",
    len(new_customer)
)


# ============================================================
# B) DATA VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("B) DATA VISUALIZATION")
print("=" * 70)


# ------------------------------------------------------------
# B(a) PIE CHART - CUSTOMER CHURN
# ------------------------------------------------------------

churn_count = df["Churn"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    churn_count.values,
    labels=churn_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Customer Churn Distribution"
)

plt.savefig("Customer Churn Distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# B(b) BAR PLOT - INTERNET SERVICE
# ------------------------------------------------------------

internet_count = df[
    "InternetService"
].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    internet_count.index,
    internet_count.values
)

plt.xlabel(
    "Internet Service"
)

plt.ylabel(
    "Number of Customers"
)

plt.title(
    "Distribution of Internet Service"
)


plt.savefig("Distribution of Internet Service.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# ADDITIONAL DATA SCIENCE EDA
# ============================================================


# ------------------------------------------------------------
# CHURN VS INTERNET SERVICE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title(
    "Churn by Internet Service"
)

plt.savefig("Churn by Internet Service.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# CHURN VS CONTRACT
# ------------------------------------------------------------

if "Contract" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x="Contract",
        hue="Churn"
    )

    plt.title(
        "Churn by Contract"
    )

    plt.xticks(rotation=20)

    plt.show()


# ------------------------------------------------------------
# CHURN VS PAYMENT METHOD
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title(
    "Churn by Payment Method"
)

plt.xticks(rotation=30)

plt.savefig("Churn by Payment Method.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# Average monthly charge

df["AvgMonthlyCharge"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1)
)


# Charge per tenure

df["ChargePerTenure"] = (
    df["TotalCharges"] /
    (df["tenure"] + 1)
)


# New customer

df["IsNewCustomer"] = (
    df["tenure"] < 12
).astype(int)


# High monthly charge

df["HighMonthlyCharge"] = (
    df["MonthlyCharges"] >
    df["MonthlyCharges"].median()
).astype(int)


print("\nNew Features:")
print(
    df[
        [
            "AvgMonthlyCharge",
            "ChargePerTenure",
            "IsNewCustomer",
            "HighMonthlyCharge"
        ]
    ].head()
)


# ============================================================
# 10. CONVERT CHURN INTO 0 AND 1
# ============================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn Values:")
print(df["Churn"].value_counts())


# ============================================================
# C) MODEL BUILDING
# ============================================================

print("\n" + "=" * 70)
print("C) MODEL BUILDING")
print("=" * 70)


# ============================================================
# MODEL 1
# Feature = TENURE
# Target = CHURN
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1 - TENURE")
print("=" * 70)


X = df[
    ["tenure"]
]

y = df[
    "Churn"
]


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ------------------------------------------------------------
# SCALING
# ------------------------------------------------------------

scaler1 = StandardScaler()

X_train_scaled = scaler1.fit_transform(
    X_train
)

X_test_scaled = scaler1.transform(
    X_test
)


# ------------------------------------------------------------
# MODEL 1
# ------------------------------------------------------------

model1 = Sequential()

model1.add(
    Dense(
        12,
        activation="relu",
        input_shape=(1,)
    )
)

model1.add(
    Dense(
        8,
        activation="relu"
    )
)

model1.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


model1.compile(

    optimizer=Adam(),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


print("\nModel 1 Summary:")

model1.summary()


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

history1 = model1.fit(

    X_train_scaled,

    y_train,

    epochs=150,

    batch_size=32,

    validation_split=0.20,

    verbose=1
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

y_pred_probability1 = model1.predict(
    X_test_scaled
)

y_pred1 = (
    y_pred_probability1 >= 0.5
).astype(int).ravel()


accuracy1 = accuracy_score(
    y_test,
    y_pred1
)

print(
    "Model 1 Accuracy:",
    accuracy1
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm1 = confusion_matrix(
    y_test,
    y_pred1
)

print("\nModel 1 Confusion Matrix:")
print(cm1)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nModel 1 Classification Report:")

print(
    classification_report(
        y_test,
        y_pred1
    )
)


# ------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm1,
    annot=True,
    fmt="d"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Confusion Matrix - Model 1"
)


plt.savefig("Confusion Matrix - Model 1.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history1.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history1.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Accuracy vs Epochs - Model 1"
)

plt.legend()

plt.savefig("Accuracy vs Epochs - Model 1.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# MODEL 2
# TENURE + DROPOUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2 - TENURE + DROPOUT")
print("=" * 70)


model2 = Sequential()


# Input layer

model2.add(
    Dense(
        12,
        activation="relu",
        input_shape=(1,)
    )
)


# Dropout after input layer

model2.add(
    Dropout(0.3)
)


# Hidden layer

model2.add(
    Dense(
        8,
        activation="relu"
    )
)


# Dropout after hidden layer

model2.add(
    Dropout(0.2)
)


# Output layer

model2.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


model2.compile(

    optimizer=Adam(),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


print("\nModel 2 Summary:")

model2.summary()


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

history2 = model2.fit(

    X_train_scaled,

    y_train,

    epochs=150,

    batch_size=32,

    validation_split=0.20,

    verbose=1
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

y_pred_probability2 = model2.predict(
    X_test_scaled
)

y_pred2 = (
    y_pred_probability2 >= 0.5
).astype(int).ravel()


accuracy2 = accuracy_score(
    y_test,
    y_pred2
)

print(
    "Model 2 Accuracy:",
    accuracy2
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm2 = confusion_matrix(
    y_test,
    y_pred2
)

print("\nModel 2 Confusion Matrix:")
print(cm2)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nModel 2 Classification Report:")

print(
    classification_report(
        y_test,
        y_pred2
    )
)


# ------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm2,
    annot=True,
    fmt="d"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Confusion Matrix - Model 2"
)

plt.savefig("Confusion Matrix - Model 2.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history2.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history2.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Accuracy vs Epochs - Model 2"
)

plt.legend()

plt.savefig("Accuracy vs Epochs - Model 2.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# MODEL 3
# FEATURES:
# TENURE + MONTHLY CHARGES + TOTAL CHARGES
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3 - THREE FEATURES")
print("=" * 70)


features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X3 = df[
    features
]

y3 = df[
    "Churn"
]


print("\nFeatures:")
print(X3.head())


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X3_train, X3_test, y3_train, y3_test = train_test_split(

    X3,
    y3,

    test_size=0.20,

    random_state=42,

    stratify=y3
)


# ------------------------------------------------------------
# SCALING
# ------------------------------------------------------------

scaler3 = StandardScaler()

X3_train_scaled = scaler3.fit_transform(
    X3_train
)

X3_test_scaled = scaler3.transform(
    X3_test
)


# ------------------------------------------------------------
# BUILD MODEL 3
# ------------------------------------------------------------

model3 = Sequential()


model3.add(
    Dense(
        12,
        activation="relu",
        input_shape=(3,)
    )
)


model3.add(
    Dense(
        8,
        activation="relu"
    )
)


model3.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# ------------------------------------------------------------
# COMPILE
# ------------------------------------------------------------

model3.compile(

    optimizer=Adam(),

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


print("\nModel 3 Summary:")

model3.summary()


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

history3 = model3.fit(

    X3_train_scaled,

    y3_train,

    epochs=150,

    batch_size=32,

    validation_split=0.20,

    verbose=1
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

y_pred_probability3 = model3.predict(
    X3_test_scaled
)

y_pred3 = (
    y_pred_probability3 >= 0.5
).astype(int).ravel()

accuracy3 = accuracy_score(
    y3_test,
    y_pred3
)

print(
    "Model 3 Accuracy:",
    accuracy3
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

cm3 = confusion_matrix(
    y3_test,
    y_pred3
)

print("\nModel 3 Confusion Matrix:")
print(cm3)


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nModel 3 Classification Report:")

print(
    classification_report(
        y3_test,
        y_pred3
    )
)


# ------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm3,
    annot=True,
    fmt="d"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Confusion Matrix - Model 3"
)

plt.savefig("Confusion Matrix - Model 3.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# ACCURACY VS EPOCHS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history3.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history3.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epochs"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Accuracy vs Epochs - Model 3"
)

plt.legend()

plt.savefig("Accuracy vs Epochs - Model 3.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 11. ADVANCED MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("ADVANCED MACHINE LEARNING")
print("=" * 70)


# ============================================================
# PREPARE ML DATA
# ============================================================

ml_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharge",
    "ChargePerTenure",
    "IsNewCustomer",
    "HighMonthlyCharge"
]

X_ml = df[
    ml_features
]

y_ml = df[
    "Churn"
]


X_ml_train, X_ml_test, y_ml_train, y_ml_test = train_test_split(

    X_ml,
    y_ml,

    test_size=0.20,

    random_state=42,

    stratify=y_ml
)


# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(

    n_estimators=300,

    random_state=42,

    class_weight="balanced"
)


rf_model.fit(
    X_ml_train,
    y_ml_train
)


rf_pred = rf_model.predict(
    X_ml_test
)

rf_probability = rf_model.predict_proba(
    X_ml_test
)[:, 1]


print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_ml_test,
        rf_pred
    )
)


print(
    "Random Forest ROC-AUC:",
    round(
        roc_auc_score(
            y_ml_test,
            rf_probability
        ),
        4
    )
)


# ============================================================
# GRADIENT BOOSTING
# ============================================================

gb_model = GradientBoostingClassifier(

    n_estimators=200,

    learning_rate=0.05,

    random_state=42
)


gb_model.fit(
    X_ml_train,
    y_ml_train
)


gb_pred = gb_model.predict(
    X_ml_test
)

gb_probability = gb_model.predict_proba(
    X_ml_test
)[:, 1]


print("\nGradient Boosting Classification Report:")

print(
    classification_report(
        y_ml_test,
        gb_pred
    )
)


print(
    "Gradient Boosting ROC-AUC:",
    round(
        roc_auc_score(
            y_ml_test,
            gb_probability
        ),
        4
    )
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000
)


logistic_model.fit(
    X_ml_train,
    y_ml_train
)


lr_pred = logistic_model.predict(
    X_ml_test
)

lr_probability = logistic_model.predict_proba(
    X_ml_test
)[:, 1]


print("\nLogistic Regression Classification Report:")

print(
    classification_report(
        y_ml_test,
        lr_pred
    )
)


print(
    "Logistic Regression ROC-AUC:",
    round(
        roc_auc_score(
            y_ml_test,
            lr_probability
        ),
        4
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

ml_results = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Random Forest",

        "Gradient Boosting",

        "Deep Learning Model 1",

        "Deep Learning Model 2",

        "Deep Learning Model 3"
    ],

    "Accuracy": [

        accuracy_score(
            y_ml_test,
            lr_pred
        ),

        accuracy_score(
            y_ml_test,
            rf_pred
        ),

        accuracy_score(
            y_ml_test,
            gb_pred
        ),

        accuracy1,

        accuracy2,

        accuracy3
    ]
})


ml_results["Accuracy (%)"] = (

    ml_results["Accuracy"] * 100

).round(2)


print("\nMODEL COMPARISON:")

print(
    ml_results
)


# ============================================================
# ROC CURVE
# ============================================================

lr_fpr, lr_tpr, _ = roc_curve(
    y_ml_test,
    lr_probability
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_ml_test,
    rf_probability
)

gb_fpr, gb_tpr, _ = roc_curve(
    y_ml_test,
    gb_probability
)


plt.figure(figsize=(9, 6))

plt.plot(
    lr_fpr,
    lr_tpr,
    label="Logistic Regression"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label="Random Forest"
)

plt.plot(
    gb_fpr,
    gb_tpr,
    label="Gradient Boosting"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.savefig("ROC Curve Comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": ml_features,

    "Importance":
        rf_model.feature_importances_

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False
)


print("\nRandom Forest Feature Importance:")

print(
    feature_importance
)


plt.figure(figsize=(9, 6))

sns.barplot(

    data=feature_importance,

    x="Importance",

    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.savefig("Random Forest Feature Importance .png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 13. PERMUTATION IMPORTANCE
# ============================================================

from sklearn.inspection import permutation_importance


permutation = permutation_importance(

    rf_model,

    X_ml_test,

    y_ml_test,

    n_repeats=10,

    random_state=42
)


permutation_df = pd.DataFrame({

    "Feature": ml_features,

    "Importance":
        permutation.importances_mean

})


permutation_df = permutation_df.sort_values(

    by="Importance",

    ascending=False
)


print("\nPermutation Importance:")

print(
    permutation_df
)


plt.figure(figsize=(9, 6))

sns.barplot(

    data=permutation_df,

    x="Importance",

    y="Feature"
)

plt.title(
    "Permutation Feature Importance"
)

plt.savefig("Permutation Feature Importance.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 14. CHURN RISK SCORE
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER CHURN RISK ANALYSIS")
print("=" * 70)


risk_df = X_ml_test.copy()

risk_df = risk_df.reset_index(
    drop=True
)

risk_df["Churn_Probability"] = (
    rf_probability
)

risk_df["Churn_Risk_Percentage"] = (

    rf_probability * 100

)


def risk_category(probability):

    if probability < 0.30:

        return "Low Risk"

    elif probability < 0.70:

        return "Medium Risk"

    else:

        return "High Risk"


risk_df["Risk_Level"] = [

    risk_category(probability)

    for probability
    in rf_probability

]


print("\nCustomer Risk Results:")

print(
    risk_df.head(20)
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(

    data=risk_df,

    x="Risk_Level",

    order=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

plt.title(
    "Customer Churn Risk Distribution"
)

plt.savefig("Customer Churn Risk Distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 15. SHAP EXPLAINABLE AI
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINABLE AI - SHAP")
print("=" * 70)


try:

    import shap

    print("SHAP imported successfully.")

    # Use TreeExplainer

    explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values = explainer.shap_values(
        X_ml_test
    )

    # Handle different SHAP versions

    if isinstance(
        shap_values,
        list
    ):

        shap.summary_plot(

            shap_values[1],

            X_ml_test,

            show=True
        )

    else:

        shap.summary_plot(

            shap_values,

            X_ml_test,

            show=True
        )

except ImportError:

    print(
        "\nSHAP is not installed."
    )

    print(
        "Install it using:"
    )

    print(
        "pip install shap"
    )

except Exception as e:

    print(
        "\nSHAP could not be generated:"
    )

    print(e)


# ============================================================
# 16. AI-STYLE CHURN EXPLANATION
# ============================================================

print("\n" + "=" * 70)
print("AI-STYLE CUSTOMER CHURN EXPLANATION")
print("=" * 70)


def generate_churn_explanation(

    customer,

    probability

):

    percentage = probability * 100


    if percentage >= 70:

        risk = "High"

    elif percentage >= 30:

        risk = "Medium"

    else:

        risk = "Low"


    reasons = []


    if (

        "tenure" in customer

        and customer["tenure"] < 12

    ):

        reasons.append(
            "short customer tenure"
        )


    if (

        "MonthlyCharges" in customer

        and customer["MonthlyCharges"]
        > df["MonthlyCharges"].median()

    ):

        reasons.append(
            "relatively high monthly charges"
        )


    if (

        "Contract" in customer

        and customer["Contract"]
        == "Month-to-month"

    ):

        reasons.append(
            "month-to-month contract"
        )


    if len(reasons) == 0:

        reason_text = (
            "No major rule-based risk "
            "factors were identified."
        )

    else:

        reason_text = (

            "Possible contributing factors: "

            + ", ".join(reasons)

            + "."
        )


    explanation = f"""

CUSTOMER CHURN AI ANALYSIS
--------------------------

Predicted Churn Probability:
{percentage:.2f}%

Risk Level:
{risk}

{reason_text}

Note:
This is a model-based prediction and
not a guaranteed future outcome.
"""


    return explanation


# ============================================================
# EXAMPLE CUSTOMER EXPLANATION
# ============================================================

customer_index = 0

customer_data = X_ml_test.iloc[
    customer_index
].to_dict()

customer_probability = rf_probability[
    customer_index
]


print(
    generate_churn_explanation(

        customer_data,

        customer_probability

    )
)


# ============================================================
# 17. HIGH RISK CUSTOMER LIST
# ============================================================

high_risk_customers = risk_df[

    risk_df["Risk_Level"]
    == "High Risk"

]


print("\n" + "=" * 70)

print("HIGH RISK CUSTOMERS")

print("=" * 70)

print(
    high_risk_customers.head(20)
)

print(
    "\nTotal High Risk Customers:",
    len(high_risk_customers)
)


# ============================================================
# 18. SAVE HIGH RISK CUSTOMERS
# ============================================================

high_risk_customers.to_csv(

    "high_risk_customers.csv",

    index=False

)


print(
    "\nHigh risk customers saved as:"
)

print(
    "high_risk_customers.csv"
)


# ============================================================
# 19. SAVE RISK RESULTS
# ============================================================

risk_df.to_csv(

    "customer_churn_risk_results.csv",

    index=False

)


print(
    "\nCustomer risk results saved as:"
)

print(
    "customer_churn_risk_results.csv"
)


# ============================================================
# 20. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({

    "Model": [

        "Model 1 - Tenure",

        "Model 2 - Tenure + Dropout",

        "Model 3 - Tenure + MonthlyCharges + TotalCharges",

        "Logistic Regression",

        "Random Forest",

        "Gradient Boosting"
    ],

    "Accuracy": [

        accuracy1,

        accuracy2,

        accuracy3,

        accuracy_score(
            y_ml_test,
            lr_pred
        ),

        accuracy_score(
            y_ml_test,
            rf_pred
        ),

        accuracy_score(
            y_ml_test,
            gb_pred
        )
    ],

    "Precision": [

        precision_score(
            y_test,
            y_pred1
        ),

        precision_score(
            y_test,
            y_pred2
        ),

        precision_score(
            y3_test,
            y_pred3
        ),

        precision_score(
            y_ml_test,
            lr_pred
        ),

        precision_score(
            y_ml_test,
            rf_pred
        ),

        precision_score(
            y_ml_test,
            gb_pred
        )
    ],

    "Recall": [

        recall_score(
            y_test,
            y_pred1
        ),

        recall_score(
            y_test,
            y_pred2
        ),

        recall_score(
            y3_test,
            y_pred3
        ),

        recall_score(
            y_ml_test,
            lr_pred
        ),

        recall_score(
            y_ml_test,
            rf_pred
        ),

        recall_score(
            y_ml_test,
            gb_pred
        )
    ],

    "F1_Score": [

        f1_score(
            y_test,
            y_pred1
        ),

        f1_score(
            y_test,
            y_pred2
        ),

        f1_score(
            y3_test,
            y_pred3
        ),

        f1_score(
            y_ml_test,
            lr_pred
        ),

        f1_score(
            y_ml_test,
            rf_pred
        ),

        f1_score(
            y_ml_test,
            gb_pred
        )
    ]
})


results["Accuracy (%)"] = (

    results["Accuracy"] * 100

).round(2)


results.to_csv(

    "complete_model_results.csv",

    index=False

)


print(
    "\nComplete model results saved as:"
)

print(
    "complete_model_results.csv"
)


# ============================================================
# 21. SAVE MODELS
# ============================================================


# Save Random Forest

joblib.dump(

    rf_model,

    "telecom_churn_random_forest.pkl"

)


# Save Logistic Regression

joblib.dump(

    logistic_model,

    "telecom_churn_logistic.pkl"

)


# Save Gradient Boosting

joblib.dump(

    gb_model,

    "telecom_churn_gradient_boosting.pkl"

)


# Save Scalers

joblib.dump(

    scaler1,

    "telecom_scaler_tenure.pkl"

)

joblib.dump(

    scaler3,

    "telecom_scaler_three_features.pkl"

)


# Save Deep Learning models

model1.save(
    "telecom_churn_model1.keras"
)

model2.save(
    "telecom_churn_model2_dropout.keras"
)

model3.save(
    "telecom_churn_model3.keras"
)


print(
    "\nAll models saved successfully!"
)


# ============================================================
# 22. FINAL MODEL ACCURACY
# ============================================================

print("\n" + "=" * 70)

print("FINAL MODEL ACCURACY")

print("=" * 70)


print(

    "Deep Learning Model 1:",

    round(
        accuracy1 * 100,
        2
    ),

    "%"

)


print(

    "Deep Learning Model 2:",

    round(
        accuracy2 * 100,
        2
    ),

    "%"

)


print(

    "Deep Learning Model 3:",

    round(
        accuracy3 * 100,
        2
    ),

    "%"

)


print(

    "Logistic Regression:",

    round(

        accuracy_score(
            y_ml_test,
            lr_pred
        ) * 100,

        2

    ),

    "%"

)


print(

    "Random Forest:",

    round(

        accuracy_score(
            y_ml_test,
            rf_pred
        ) * 100,

        2

    ),

    "%"

)


print(

    "Gradient Boosting:",

    round(

        accuracy_score(
            y_ml_test,
            gb_pred
        ) * 100,

        2

    ),

    "%"

)


# ============================================================
# 23. FINAL PROJECT OUTPUT
# ============================================================

print("\n")

print("=" * 70)

print(
    "TELECOM CUSTOMER CHURN AI PROJECT COMPLETED"
)

print("=" * 70)

print(
    """
Skills Covered:

1. Python
2. Pandas
3. NumPy
4. Data Cleaning
5. Data Manipulation
6. Exploratory Data Analysis
7. Data Visualization
8. Feature Engineering
9. Machine Learning
10. Logistic Regression
11. Random Forest
12. Gradient Boosting
13. Deep Learning
14. Artificial Neural Network
15. Dropout
16. Model Evaluation
17. Confusion Matrix
18. ROC-AUC
19. Feature Importance
20. Permutation Importance
21. SHAP Explainable AI
22. Customer Churn Risk Scoring
23. AI-style Churn Explanation
24. Model Saving
25. CSV Result Generation
"""
)

print("=" * 70)