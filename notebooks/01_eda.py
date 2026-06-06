# =============================================================================
# Heart Disease Risk Prediction — Notebook 1: EDA
# Client: Bayer AG / Roche Diagnostics — Clinical Decision Support
# Dataset: UCI Cleveland Heart Disease Dataset
# =============================================================================
# Goal: Understand patient clinical data, identify key risk factors,
#       and explore patterns that distinguish heart disease patients
#       from healthy patients.
#
# Medical Context:
#   We have clinical measurements from 303 patients who were tested
#   for heart disease at the Cleveland Clinic Foundation.
#   Our job: find patterns that predict heart disease BEFORE symptoms.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

print("=" * 65)
print("  NOTEBOOK 1 — EXPLORATORY DATA ANALYSIS")
print("  Heart Disease Risk | Bayer AG | Cleveland UCI Dataset")
print("=" * 65)

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================
# The Cleveland dataset has NO column headers — we define them manually
# based on the UCI documentation

COLUMNS = [
    "age",       # Patient age in years
    "sex",       # 1=Male, 0=Female
    "cp",        # Chest pain type (1-4)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",      # Serum cholesterol (mg/dl)
    "fbs",       # Fasting blood sugar > 120 (1=yes, 0=no)
    "restecg",   # Resting ECG results (0,1,2)
    "thalach",   # Max heart rate during exercise
    "exang",     # Exercise induced chest pain (1=yes, 0=no)
    "oldpeak",   # ST depression during exercise
    "slope",     # Slope of peak exercise ST segment
    "ca",        # Major vessels colored (0-3)
    "thal",      # Thalassemia type (3,6,7)
    "target"     # Heart disease (0=no, 1-4=yes)
]

# Load the raw data file
# Missing values are represented as "?" in this dataset
df = pd.read_csv(
    os.path.join(DATA, "processed.cleveland.data"),
    names=COLUMNS,
    na_values="?"   # treat "?" as missing values
)

# Convert target to binary (0=no disease, 1=disease)
# Original values 1,2,3,4 all mean disease is present
df["target"] = (df["target"] > 0).astype(int)

print(f"\n  Dataset loaded: {df.shape[0]} patients × {df.shape[1]} features")
print(f"  Heart disease cases : {df['target'].sum()} ({df['target'].mean():.1%})")
print(f"  Healthy cases       : {(df['target']==0).sum()} ({(df['target']==0).mean():.1%})")
print(f"  Missing values      : {df.isnull().sum().sum()} total")
print(f"\n  Columns with missing values:")
print(df.isnull().sum()[df.isnull().sum()>0])

# =============================================================================
# STEP 2 — DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 65)
print("  DESCRIPTIVE STATISTICS")
print("=" * 65)
print(df.describe().round(2))

print("\n  Comparison — Heart Disease vs Healthy patients:")
print(df.groupby("target")[["age","chol","trestbps","thalach","oldpeak"]].mean().round(2))

# =============================================================================
# STEP 3 — TARGET DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
counts = df["target"].value_counts().sort_index()
bars = axes[0].bar(["No Disease", "Heart Disease"],
                    counts.values,
                    color=["#2ECC71", "#E74C3C"],
                    edgecolor="white", linewidth=2)
axes[0].set_title("Patient Distribution", fontweight="bold")
axes[0].set_ylabel("Number of Patients")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 3,
                 f"{val}\n({val/len(df):.1%})",
                 ha="center", fontsize=11, fontweight="bold")

# Age distribution by disease
df[df["target"]==0]["age"].hist(ax=axes[1], bins=20,
    alpha=0.7, color="#2ECC71", label="No Disease")
df[df["target"]==1]["age"].hist(ax=axes[1], bins=20,
    alpha=0.7, color="#E74C3C", label="Heart Disease")
axes[1].set_title("Age Distribution by Condition", fontweight="bold")
axes[1].set_xlabel("Age (years)")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.suptitle("Target Variable — Heart Disease Present/Absent",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_01_target_distribution.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_01_target_distribution.png")

# =============================================================================
# STEP 4 — KEY RISK FACTOR COMPARISONS
# =============================================================================
print("\n  Plotting key clinical risk factors...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Clinical features to compare
features = [
    ("chol",     "Cholesterol (mg/dl)",      "High cholesterol = artery blockage risk"),
    ("trestbps", "Resting Blood Pressure",    "High BP = heart overworked"),
    ("thalach",  "Max Heart Rate",            "Lower max rate = weaker heart"),
    ("oldpeak",  "ST Depression",             "Higher = less oxygen to heart"),
    ("age",      "Patient Age",               "Risk increases with age"),
    ("ca",       "Vessels Blocked (0-3)",     "More blocked = higher risk"),
]

for i, (feat, label, meaning) in enumerate(features):
    temp = df.dropna(subset=[feat])
    temp[temp["target"]==0][feat].hist(ax=axes[i], bins=20,
        alpha=0.7, color="#2ECC71", label="No Disease")
    temp[temp["target"]==1][feat].hist(ax=axes[i], bins=20,
        alpha=0.7, color="#E74C3C", label="Disease")
    axes[i].set_title(f"{label}\n{meaning}", fontweight="bold", fontsize=10)
    axes[i].set_xlabel(label)
    axes[i].legend(fontsize=8)

plt.suptitle("Key Clinical Risk Factors — Heart Disease vs Healthy",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_02_risk_factors.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_02_risk_factors.png")

# =============================================================================
# STEP 5 — CATEGORICAL FEATURES ANALYSIS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Chest pain type vs disease
cp_labels = {1:"Typical\nAngina", 2:"Atypical\nAngina",
             3:"Non-anginal", 4:"No Symptoms"}
cp_disease = df.groupby("cp")["target"].mean() * 100
cp_disease.index = [cp_labels.get(i, str(i)) for i in cp_disease.index]
axes[0].bar(cp_disease.index, cp_disease.values,
            color=["#3498DB","#E67E22","#9B59B6","#E74C3C"],
            edgecolor="white")
axes[0].set_title("Heart Disease Rate by\nChest Pain Type",
                  fontweight="bold")
axes[0].set_ylabel("% With Heart Disease")
axes[0].set_xlabel("Chest Pain Type")
for i, v in enumerate(cp_disease.values):
    axes[0].text(i, v+1, f"{v:.1f}%", ha="center", fontsize=10)

# Sex vs disease
sex_labels = {0:"Female", 1:"Male"}
sex_disease = df.groupby("sex")["target"].mean() * 100
sex_disease.index = [sex_labels[i] for i in sex_disease.index]
axes[1].bar(sex_disease.index, sex_disease.values,
            color=["#E91E8C","#3498DB"], edgecolor="white")
axes[1].set_title("Heart Disease Rate\nby Gender",
                  fontweight="bold")
axes[1].set_ylabel("% With Heart Disease")
for i, v in enumerate(sex_disease.values):
    axes[1].text(i, v+1, f"{v:.1f}%", ha="center", fontsize=11)

# Exercise angina vs disease
exang_labels = {0:"No Pain", 1:"Exercise Pain"}
exang_disease = df.groupby("exang")["target"].mean() * 100
exang_disease.index = [exang_labels[i] for i in exang_disease.index]
axes[2].bar(exang_disease.index, exang_disease.values,
            color=["#2ECC71","#E74C3C"], edgecolor="white")
axes[2].set_title("Heart Disease Rate by\nExercise-Induced Chest Pain",
                  fontweight="bold")
axes[2].set_ylabel("% With Heart Disease")
for i, v in enumerate(exang_disease.values):
    axes[2].text(i, v+1, f"{v:.1f}%", ha="center", fontsize=11)

plt.suptitle("Categorical Risk Factors — Clinical Pattern Analysis",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_03_categorical_risk_factors.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_03_categorical_risk_factors.png")

# =============================================================================
# STEP 6 — CORRELATION HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, mask=mask,
            linewidths=0.5, annot_kws={"size": 9})
ax.set_title("Feature Correlation Matrix\n"
             "(Green = positive, Red = negative correlation)",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_04_correlation_heatmap.png"),
            bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_04_correlation_heatmap.png")

# =============================================================================
# STEP 7 — KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  EDA KEY FINDINGS")
print("=" * 65)

no_disease = df[df["target"]==0]
disease    = df[df["target"]==1]

print(f"""
  1. Dataset has {len(df)} patients:
     {disease['target'].sum()} with heart disease ({disease['target'].sum()/len(df):.1%})
     {no_disease['target'].sum()} without heart disease

  2. Key differences between groups:
     Age       : Disease={disease['age'].mean():.1f}  Healthy={no_disease['age'].mean():.1f}
     Cholesterol: Disease={disease['chol'].mean():.1f} Healthy={no_disease['chol'].mean():.1f}
     Max HR    : Disease={disease['thalach'].mean():.1f} Healthy={no_disease['thalach'].mean():.1f}
     ST depress: Disease={disease['oldpeak'].mean():.2f} Healthy={no_disease['oldpeak'].mean():.2f}

  3. Surprising finding:
     Patients with NO chest pain (type 4) have the HIGHEST
     heart disease rate — this is called silent ischemia.
     This is why ML models are crucial — doctors miss these!

  4. Gender difference:
     Male patients have significantly higher heart disease rate
     than female patients in this dataset.

  5. Missing values:
     Only 6 missing values total — very clean dataset.
     Will handle with median imputation in feature engineering.

  Next step: Feature Engineering (02_feature_engineering.py)
""")

# Save processed data
df.to_csv(os.path.join(DATA, "heart_disease_raw.csv"), index=False)
print("  ✔ Saved: heart_disease_raw.csv")
