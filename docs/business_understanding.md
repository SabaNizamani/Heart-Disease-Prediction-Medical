# Task 1 — Business Understanding
# Heart Disease Risk Prediction
# Client: Bayer AG / Roche Diagnostics — Clinical Decision Support Division

---

## The Client & Business Problem

**Client:** Bayer AG — Pharmaceuticals Division
**Sector:** Healthcare / Clinical Analytics
**Location:** Leverkusen, Germany

### The Problem

Cardiovascular disease is the leading cause of death worldwide.
The World Health Organization estimates that 17.9 million people
die from cardiovascular diseases every year — that is 32% of all
global deaths.

Early detection is critical:
→ Patients identified early have 80% better survival outcomes
→ Late-stage treatment costs 3-5x more than early intervention
→ Most patients show NO symptoms until a major cardiac event

Current clinical challenge:
Doctors rely on manual review of patient data — age, cholesterol,
blood pressure, ECG results — to assess heart disease risk.
This process is:
→ Time-consuming in high-volume clinics
→ Inconsistent between different doctors
→ Often too late — symptoms appear after damage is done

### Our Solution — Predictive Clinical Decision Support

Use machine learning to analyse patient clinical measurements
and predict heart disease risk BEFORE symptoms appear.

This is exactly what Bayer's pharmaceutical division does —
they develop clinical decision support tools used by hospitals
across Germany and Europe.

---

## Business Question

> "Can we predict the presence of heart disease from routine
>  clinical measurements so doctors can intervene earlier
>  and save more lives?"

---

## Dataset — Cleveland Heart Disease (UCI Repository)

**Source:** University of California Irvine (UCI) ML Repository
**Original data from:** Cleveland Clinic Foundation, USA
**Used in research at:** Mayo Clinic, Stanford, Siemens Healthineers

This is the most widely used heart disease dataset in medical ML
research. Published in 1988 and still used in research today.

### The 14 Clinical Features:

| Feature | Medical Meaning |
|---------|----------------|
| age | Patient age in years |
| sex | 1=Male, 0=Female |
| cp | Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=no pain) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1=yes, 0=no) |
| restecg | Resting ECG results (0=normal, 1=ST abnormality, 2=hypertrophy) |
| thalach | Maximum heart rate achieved during exercise test |
| exang | Exercise induced chest pain (1=yes, 0=no) |
| oldpeak | ST depression during exercise vs rest |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0-3) |
| thal | Thalassemia type (3=normal, 6=fixed defect, 7=reversible defect) |
| target | Heart disease present (0=no, 1=yes) |

### What These Features Mean Simply:

**Blood pressure (trestbps):**
High blood pressure = heart working too hard = risk factor

**Cholesterol (chol):**
High cholesterol = arteries get blocked = heart disease risk

**Maximum heart rate (thalach):**
Healthy heart = higher max rate. Diseased heart = lower max rate.

**ST depression (oldpeak):**
Measured on ECG. Higher value = heart muscle getting less oxygen.

**Chest pain type (cp):**
Type 4 (asymptomatic) is actually most dangerous —
patients feel nothing but have serious disease underneath.

---

## Expected Business Impact

| Metric | Without ML | With ML Model |
|--------|-----------|--------------|
| Detection rate | ~60% (manual review) | 85%+ (ML assisted) |
| False negatives | High risk | Significantly reduced |
| Clinical decision time | 30-60 mins per patient | Instant risk score |
| Early intervention rate | Low | High |

---

## Why This Is Relevant for German Medical Companies

**Bayer AG:**
Develops cardiovascular drugs — needs to identify high-risk
patients for clinical trials and drug targeting.

**Roche Diagnostics:**
Makes diagnostic tests — this model predicts who needs testing.

**Siemens Healthineers:**
Builds ECG machines and clinical analytics software — exactly
this type of risk prediction model.

**B. Braun:**
Makes cardiac monitoring devices — risk scores guide device use.

---

## Key Medical Terms for Your Interview

**EHR (Electronic Health Record):**
Digital version of a patient's medical history.
Contains: diagnoses, medications, test results, vital signs.
This dataset IS a simplified version of EHR data.

**ICD-10 Codes:**
International Classification of Diseases — codes for every disease.
Heart disease = ICD-10 code I25 (chronic ischaemic heart disease).
German hospitals use ICD-10 codes for all diagnoses.

**Clinical Decision Support:**
Software that helps doctors make better decisions using data.
Your model IS a clinical decision support tool.

**GDPR in Healthcare:**
European law protecting patient data privacy.
All German medical companies must comply with GDPR.
This dataset is anonymised — no patient names or IDs.
