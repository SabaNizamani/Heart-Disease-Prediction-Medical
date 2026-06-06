# 🏥 Heart Disease Risk Prediction
### UCI Cleveland Dataset | Bayer AG / Roche Diagnostics Use Case

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Cleveland-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Industry](https://img.shields.io/badge/Industry-Healthcare%20%7C%20Pharma-blue)

---

## 📌 Project Overview

This project builds a **Clinical Decision Support System** to predict heart disease risk from routine patient measurements — directly relevant to work done at **Bayer AG**, **Roche Diagnostics**, and **Siemens Healthineers** in Germany.

**The Business Problem:**
> Cardiovascular disease causes 32% of all global deaths. Most patients show NO symptoms until a major cardiac event. Doctors manually reviewing patient data miss silent cases — especially patients with no chest pain who actually have serious underlying disease.

**Our Solution:**
> Use machine learning on routine clinical measurements (blood pressure, cholesterol, ECG results) to flag high-risk patients BEFORE symptoms appear — enabling earlier intervention and saving lives.

---

## 🎯 Business Question
> *"Can we predict heart disease from routine clinical measurements accurately enough to support doctors in earlier diagnosis?"*

**Answer: YES** — with clinically meaningful accuracy across two model approaches.

---

## 📂 Project Structure

```
Heart-Disease-Prediction-Medical/
│
├── run_pipeline.py              # Master script — runs everything
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── 01_eda.py                # Exploratory Data Analysis
│   ├── 02_feature_engineering.py
│   └── 03_modelling.py
│
├── docs/
│   └── business_understanding.md
│
├── data/                        # Add dataset here (not tracked)
│
└── outputs/                     # Auto-generated plots & predictions
```

---

## 🗂️ Dataset — UCI Cleveland Heart Disease

**Source:** University of California Irvine (UCI) ML Repository
**Download:** https://archive.ics.uci.edu/dataset/45/heart+disease
**Origin:** Cleveland Clinic Foundation — real patient data from 1988
**Used in research at:** Mayo Clinic, Stanford, Siemens Healthineers

| Property | Detail |
|---|---|
| Patients | 303 real clinical cases |
| Features | 13 clinical measurements |
| Target | Heart disease present (0=No, 1=Yes) |
| Missing values | Only 6 — very clean dataset |

### The 13 Clinical Features:

| Feature | Medical Meaning |
|---|---|
| age | Patient age in years |
| sex | 1=Male, 0=Female |
| cp | Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=none) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar >120 mg/dl (1=yes) |
| restecg | Resting ECG results (0=normal, 1=ST abnormality, 2=hypertrophy) |
| thalach | Maximum heart rate during exercise test |
| exang | Exercise-induced chest pain (1=yes) |
| oldpeak | ST depression during exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels blocked (0-3) |
| thal | Thalassemia type (3=normal, 6=fixed, 7=reversible defect) |

---

## 🧩 Project Stages

### Stage 1 — Exploratory Data Analysis
- Compared heart disease vs healthy patients across all 13 features
- Found **surprising insight**: patients with NO chest pain (type 4) have the HIGHEST disease rate — called silent ischemia — exactly why ML models are needed
- Males show significantly higher disease rate than females
- Exercise-induced chest pain strongly predicts disease presence

### Stage 2 — Feature Engineering

| Feature | Description | Clinical Basis |
|---|---|---|
| **age_group** | WHO age categories (0-3) | Risk increases by age decade |
| **bp_risk** | Blood pressure risk level (0-3) | WHO hypertension guidelines |
| **chol_risk** | Cholesterol risk category (0-2) | Clinical lipid guidelines |
| **hr_percentage** | Actual HR / predicted max HR | Heart efficiency ratio |
| **risk_score** | Sum of 6 binary risk factors (0-6) | Combined clinical assessment |

### Stage 3 — Two Models Compared

**Model 1 — Logistic Regression** (interpretable)
- Doctors can understand coefficients: "higher cholesterol = X% more risk"
- Important in clinical settings where decisions must be explainable

**Model 2 — Random Forest** (main model, higher performance)
- 500 decision trees
- class_weight="balanced" to handle slight class imbalance
- **Recall is the key metric** — missing a heart disease patient is dangerous

---
## 📊 Model Results
### Random Forest (Main Model)

| Metric | Score |
|---|---|
| Accuracy | 0.8333 |
| Precision | 0.8214 |
| Recall | 0.8214 |
| F1 Score | 0.8214 |
| ROC-AUC | 0.9408 ← excellent! |
| 5-Fold CV AUC | 0.8965 ± 0.0445 |

### Clinical Simulation (60 test patients)
| Result | Number |
|---|---|
| Disease cases correctly caught | 23 / 28 (82.1%) |
| Missed dangerous cases | 5 |
| Healthy patients correctly cleared | 27 / 32 |

---

## 💡 Key Clinical Finding

> **Silent ischemia discovered:** Patients reporting NO chest pain (type 4) actually have the highest heart disease rate in this dataset. This is the strongest argument for ML-based screening — patients who feel fine can still be at high risk. Doctors relying on symptom reports alone will miss these cases.

---

## 💼 Why This Matters for German Medical Companies

| Company | Relevance |
|---|---|
| **Bayer AG** | Identifies high-risk patients for cardiovascular drug trials |
| **Roche Diagnostics** | Predicts which patients need diagnostic testing |
| **Siemens Healthineers** | Powers clinical decision support in cardiology software |
| **B. Braun** | Guides cardiac monitoring device deployment |
| **Boehringer Ingelheim** | Patient stratification for heart failure drug studies |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/SabaNizamani/Heart-Disease-Prediction-Medical.git
cd Heart-Disease-Prediction-Medical
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Download from: https://archive.ics.uci.edu/dataset/45/heart+disease

Place `processed.cleveland.data` in the `/data/` folder.

### 4. Run pipeline
```bash
python run_pipeline.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and clinical feature engineering |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic Regression, Random Forest, metrics |
| `matplotlib` | Clinical visualisations and evaluation dashboard |
| `seaborn` | Correlation heatmap and statistical plots |

---

## 📈 Potential Improvements

- [ ] Add SHAP values for individual patient explanation to doctors
- [ ] Lower decision threshold to 0.4 to maximise recall
- [ ] Combine all 4 UCI datasets (Cleveland + Hungarian + Swiss + VA) for larger training set
- [ ] Try XGBoost for comparison
- [ ] Build Streamlit dashboard for clinical use
- [ ] Validate on European patient datasets (GDPR compliant)

---

## 🏥 EHR Context

This dataset represents a simplified version of **Electronic Health Record (EHR)** data — the digital patient records used by every hospital in Germany. In real clinical deployment:
- Data would come from hospital EHR systems
- Features would include ICD-10 diagnosis codes
- Predictions would integrate into clinical workflow software
- All data processing must comply with **GDPR** (European patient privacy law)

---

## 👤 Author

**Saba Nizamani**
[LinkedIn](https://www.linkedin.com/in/saba-nizamani-3a890121b) · [GitHub](https://github.com/SabaNizamani) · [Email](mailto:sabanizamani15@gmail.com)
