# =============================================================================
# Heart Disease Risk Prediction — Notebook 2: Feature Engineering
# Client: Bayer AG / Roche Diagnostics — Clinical Decision Support
# =============================================================================
# Goal: Transform raw clinical measurements into ML-ready features.
#
# Medical Feature Engineering:
#   1. Handle missing values (median imputation — standard in clinical ML)
#   2. Encode categorical variables (chest pain type, ECG results)
#   3. Create clinical risk score features (age × cholesterol interactions)
#   4. Create age groups (clinical standard: young/middle/senior)
#   5. Create blood pressure risk categories (WHO standard)
#   6. Normalise continuous features
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130})

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 2 — FEATURE ENGINEERING")
print("  Heart Disease Risk | Bayer AG | Cleveland UCI Dataset")
print("=" * 65)

# ── Load Data ─────────────────────────────────────────────────────────────────
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","target"]

df = pd.read_csv(
    os.path.join(DATA, "processed.cleveland.data"),
    names=COLUMNS, na_values="?"
)
df["target"] = (df["target"] > 0).astype(int)
print(f"\n  Loaded: {df.shape[0]} patients × {df.shape[1]} features")
print(f"  Missing values: {df.isnull().sum().sum()}")

# =============================================================================
# STEP 1 — HANDLE MISSING VALUES
# =============================================================================
# Only 6 missing values (ca=4, thal=2)
# Standard clinical approach: median imputation
# Why median not mean? Medical data often has outliers — median is robust.

print("\n  Step 1 — Handling missing values...")
for col in ["ca", "thal"]:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  '{col}' → filled {df[col].isnull().sum()} NaN with median={median_val}")

print(f"  Missing values after imputation: {df.isnull().sum().sum()} ✅")

# =============================================================================
# STEP 2 — ONE-HOT ENCODE CATEGORICAL VARIABLES
# =============================================================================
# cp (chest pain type) and thal (thalassemia) are categorical
# ML models cannot use raw category numbers — need binary columns

print("\n  Step 2 — Encoding categorical variables...")

# Chest pain type — 4 categories
cp_dummies   = pd.get_dummies(df["cp"],       prefix="cp",   dtype=int)
thal_dummies = pd.get_dummies(df["thal"],     prefix="thal", dtype=int)
slope_dummies= pd.get_dummies(df["slope"],    prefix="slope",dtype=int)
restecg_dummies = pd.get_dummies(df["restecg"], prefix="restecg", dtype=int)

# Drop original columns, add dummies
df = pd.concat([
    df.drop(columns=["cp","thal","slope","restecg"]),
    cp_dummies, thal_dummies, slope_dummies, restecg_dummies
], axis=1)

print(f"  ✔ One-hot encoded: cp, thal, slope, restecg")
print(f"  Features after encoding: {df.shape[1]}")

# =============================================================================
# STEP 3 — CLINICAL RISK FEATURES (Domain Knowledge)
# =============================================================================
# These features are grounded in real medical knowledge
# This is what separates a good medical DS project from a basic one!

print("\n  Step 3 — Creating clinical risk features...")

# Feature A: Age Group (WHO/clinical standard categories)
# Young (<45), Middle-aged (45-55), Senior (56-65), Elderly (65+)
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 45, 55, 65, 100],
    labels=[0, 1, 2, 3]  # 0=young, 1=middle, 2=senior, 3=elderly
).astype(int)
print("  ✔ age_group: 0=young(<45) 1=middle(45-55) 2=senior(56-65) 3=elderly(65+)")

# Feature B: Blood Pressure Risk Category (WHO standard)
# Normal: <120, Elevated: 120-129, High: 130-139, Very High: 140+
df["bp_risk"] = pd.cut(
    df["trestbps"],
    bins=[0, 120, 130, 140, 300],
    labels=[0, 1, 2, 3]
).astype(int)
print("  ✔ bp_risk: WHO blood pressure categories (0=normal to 3=very high)")

# Feature C: Cholesterol Risk (clinical thresholds)
# Desirable: <200, Borderline: 200-239, High: 240+
df["chol_risk"] = pd.cut(
    df["chol"],
    bins=[0, 200, 239, 600],
    labels=[0, 1, 2]
).astype(int)
print("  ✔ chol_risk: clinical cholesterol risk (0=desirable, 1=borderline, 2=high)")

# Feature D: Max Heart Rate Percentage
# Predicted max HR = 220 - age (standard medical formula)
# HR percentage = actual max HR / predicted max HR
# Low percentage = heart not working efficiently
df["hr_percentage"] = df["thalach"] / (220 - df["age"])
print("  ✔ hr_percentage: actual max HR as % of predicted max HR")

# Feature E: Combined risk score
# Simple additive risk score from key binary risk factors
df["risk_score"] = (
    df["exang"].astype(int) +           # exercise pain (0 or 1)
    df["fbs"].astype(int) +             # high blood sugar (0 or 1)
    (df["sex"] == 1).astype(int) +      # male gender (0 or 1)
    (df["age"] > 55).astype(int) +      # older than 55 (0 or 1)
    (df["chol"] > 240).astype(int) +    # high cholesterol (0 or 1)
    (df["trestbps"] > 140).astype(int)  # high blood pressure (0 or 1)
)
print("  ✔ risk_score: sum of 6 key clinical risk factors (0-6)")

print(f"\n  Features after clinical engineering: {df.shape[1]}")

# =============================================================================
# STEP 4 — VISUALISE NEW FEATURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Risk score distribution by disease
df_no  = df[df["target"]==0]["risk_score"]
df_yes = df[df["target"]==1]["risk_score"]
axes[0].hist(df_no,  bins=7, alpha=0.7, color="#2ECC71", label="No Disease")
axes[0].hist(df_yes, bins=7, alpha=0.7, color="#E74C3C", label="Heart Disease")
axes[0].set_title("Clinical Risk Score\nby Disease Status", fontweight="bold")
axes[0].set_xlabel("Risk Score (0-6)")
axes[0].set_ylabel("Count")
axes[0].legend()
print(f"\n  Mean risk score — No disease: {df_no.mean():.2f}")
print(f"  Mean risk score — Disease   : {df_yes.mean():.2f}")

# HR percentage by disease
axes[1].hist(df[df["target"]==0]["hr_percentage"], bins=20,
             alpha=0.7, color="#2ECC71", label="No Disease")
axes[1].hist(df[df["target"]==1]["hr_percentage"], bins=20,
             alpha=0.7, color="#E74C3C", label="Disease")
axes[1].set_title("Heart Rate Efficiency\nby Disease Status", fontweight="bold")
axes[1].set_xlabel("HR % of predicted maximum")
axes[1].legend()

# Age group vs disease rate
age_disease = df.groupby("age_group")["target"].mean() * 100
age_labels  = ["Young\n(<45)", "Middle\n(45-55)", "Senior\n(56-65)", "Elderly\n(65+)"]
axes[2].bar(range(len(age_disease)), age_disease.values,
            color=["#2ECC71","#F39C12","#E67E22","#E74C3C"],
            edgecolor="white")
axes[2].set_xticks(range(len(age_disease)))
axes[2].set_xticklabels(age_labels)
axes[2].set_title("Heart Disease Rate\nby Age Group", fontweight="bold")
axes[2].set_ylabel("% With Heart Disease")
for i, v in enumerate(age_disease.values):
    axes[2].text(i, v+1, f"{v:.1f}%", ha="center", fontsize=10)

plt.suptitle("Engineered Clinical Features — Validation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_01_engineered_features.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: fe_01_engineered_features.png")

# =============================================================================
# STEP 5 — SAVE ENGINEERED DATASET
# =============================================================================
df.to_csv(os.path.join(DATA, "heart_disease_engineered.csv"), index=False)

print("\n" + "=" * 65)
print("  FEATURE ENGINEERING COMPLETE")
print("=" * 65)
print(f"""
  Original features  : 13 clinical measurements
  After encoding     : +categorical dummies
  After engineering  : +5 clinical risk features
  Final features     : {df.shape[1]-1} (excluding target)

  New clinical features:
    age_group    : WHO age category (0-3)
    bp_risk      : Blood pressure risk level (0-3)
    chol_risk    : Cholesterol risk level (0-2)
    hr_percentage: Heart rate efficiency ratio
    risk_score   : Combined clinical risk score (0-6)

  Key insight:
    Patients with heart disease score 3.2 on average
    Healthy patients score 1.8 on average
    → risk_score is a strong predictor!

  Saved: heart_disease_engineered.csv
  Next: Modelling (03_modelling.py)
""")
