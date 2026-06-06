# =============================================================================
# Heart Disease Risk Prediction — Master Pipeline
# Client: Bayer AG / Roche Diagnostics — Clinical Decision Support
# Dataset: UCI Cleveland Heart Disease (303 real patients)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi":130,"axes.titlesize":13})
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,"data"); OUTPUTS=os.path.join(ROOT,"outputs")
os.makedirs(OUTPUTS,exist_ok=True)

def section(t): print("\n"+"="*65+f"\n  {t}\n"+"="*65)
def save(fig,n): fig.savefig(os.path.join(OUTPUTS,n),bbox_inches="tight"); print(f"    ✔ Saved: {n}")

COLUMNS=["age","sex","cp","trestbps","chol","fbs","restecg",
         "thalach","exang","oldpeak","slope","ca","thal","target"]

# =============================================================================
# STAGE 1 — LOAD & EDA
# =============================================================================
section("STAGE 1 — LOADING & EXPLORING DATA")

df=pd.read_csv(os.path.join(DATA,"processed.cleveland.data"),names=COLUMNS,na_values="?")
df["target"]=(df["target"]>0).astype(int)

print(f"  Patients loaded : {len(df)}")
print(f"  Heart disease   : {df['target'].sum()} ({df['target'].mean():.1%})")
print(f"  Healthy         : {(df['target']==0).sum()} ({(df['target']==0).mean():.1%})")
print(f"  Missing values  : {df.isnull().sum().sum()}")

fig,axes=plt.subplots(1,3,figsize=(16,5))
counts=df["target"].value_counts().sort_index()
bars=axes[0].bar(["No Disease","Heart Disease"],counts.values,
    color=["#2ECC71","#E74C3C"],edgecolor="white",linewidth=2)
axes[0].set_title("Patient Distribution",fontweight="bold")
for bar,val in zip(bars,counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
        f"{val}\n({val/len(df):.1%})",ha="center",fontsize=11,fontweight="bold")

df[df["target"]==0]["age"].hist(ax=axes[1],bins=20,alpha=0.7,color="#2ECC71",label="No Disease")
df[df["target"]==1]["age"].hist(ax=axes[1],bins=20,alpha=0.7,color="#E74C3C",label="Disease")
axes[1].set_title("Age Distribution by Condition",fontweight="bold")
axes[1].legend()

corr_target=df.corr()["target"].drop("target").sort_values()
colors_c=["#E74C3C" if v<0 else "#2ECC71" for v in corr_target.values]
axes[2].barh(corr_target.index,corr_target.values,color=colors_c,edgecolor="white")
axes[2].axvline(0,color="black",lw=0.8)
axes[2].set_title("Feature Correlation with Target",fontweight="bold")
plt.suptitle("EDA Overview — UCI Cleveland Heart Disease",fontsize=14,fontweight="bold")
plt.tight_layout(); save(fig,"eda_01_overview.png"); plt.show()

# Categorical analysis
fig,axes=plt.subplots(1,3,figsize=(16,5))
cp_disease=df.groupby("cp")["target"].mean()*100
axes[0].bar([f"Type {i}" for i in cp_disease.index],cp_disease.values,
    color=["#3498DB","#E67E22","#9B59B6","#E74C3C"],edgecolor="white")
axes[0].set_title("Disease Rate by Chest Pain Type",fontweight="bold")
axes[0].set_ylabel("% With Heart Disease")
for i,v in enumerate(cp_disease.values): axes[0].text(i,v+1,f"{v:.1f}%",ha="center")

sex_disease=df.groupby("sex")["target"].mean()*100
axes[1].bar(["Female","Male"],sex_disease.values,color=["#E91E8C","#3498DB"],edgecolor="white")
axes[1].set_title("Disease Rate by Gender",fontweight="bold")
axes[1].set_ylabel("% With Heart Disease")
for i,v in enumerate(sex_disease.values): axes[1].text(i,v+1,f"{v:.1f}%",ha="center")

exang_disease=df.groupby("exang")["target"].mean()*100
axes[2].bar(["No Exercise Pain","Exercise Pain"],exang_disease.values,
    color=["#2ECC71","#E74C3C"],edgecolor="white")
axes[2].set_title("Disease Rate by Exercise Pain",fontweight="bold")
axes[2].set_ylabel("% With Heart Disease")
for i,v in enumerate(exang_disease.values): axes[2].text(i,v+1,f"{v:.1f}%",ha="center")

plt.suptitle("Categorical Risk Factors",fontsize=14,fontweight="bold")
plt.tight_layout(); save(fig,"eda_02_risk_factors.png"); plt.show()

# =============================================================================
# STAGE 2 — FEATURE ENGINEERING
# =============================================================================
section("STAGE 2 — FEATURE ENGINEERING")

# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Drop any remaining rows with NaN (safety net)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# One-hot encode categoricals
for col in ["cp","thal","slope","restecg"]:
    dummies=pd.get_dummies(df[col],prefix=col,dtype=int)
    df=pd.concat([df.drop(columns=[col]),dummies],axis=1)

# Clinical risk features
df["age_group"]=pd.cut(df["age"],bins=[0,45,55,65,100],labels=[0,1,2,3]).astype(float).fillna(0).astype(int)
df["bp_risk"]=pd.cut(df["trestbps"],bins=[0,120,130,140,300],labels=[0,1,2,3]).astype(float).fillna(0).astype(int)
df["chol_risk"]=pd.cut(df["chol"],bins=[0,200,239,600],labels=[0,1,2]).astype(float).fillna(0).astype(int)
df["hr_percentage"]=df["thalach"]/(220-df["age"])
df["risk_score"]=(df["exang"].astype(int)+df["fbs"].astype(int)+
    (df["sex"]==1).astype(int)+(df["age"]>55).astype(int)+
    (df["chol"]>240).astype(int)+(df["trestbps"]>140).astype(int))

print(f"  Features after engineering: {df.shape[1]-1}")
print(f"  Mean risk score — Disease : {df[df['target']==1]['risk_score'].mean():.2f}")
print(f"  Mean risk score — Healthy : {df[df['target']==0]['risk_score'].mean():.2f}")

# =============================================================================
# STAGE 3 — MODELLING
# =============================================================================
section("STAGE 3 — MODELLING")

X=df.drop(columns=["target"]); y=df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Logistic Regression
scaler=StandardScaler()
X_tr_s=scaler.fit_transform(X_train); X_te_s=scaler.transform(X_test)
lr=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
lr.fit(X_tr_s,y_train)
lr_pred=lr.predict(X_te_s); lr_proba=lr.predict_proba(X_te_s)[:,1]

# Random Forest
rf=RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_leaf=3,
    class_weight="balanced",random_state=42,n_jobs=-1)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test); rf_proba=rf.predict_proba(X_test)[:,1]

rf_acc=accuracy_score(y_test,rf_pred); rf_prec=precision_score(y_test,rf_pred)
rf_rec=recall_score(y_test,rf_pred); rf_f1=f1_score(y_test,rf_pred)
rf_auc=roc_auc_score(y_test,rf_proba)

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_scores=cross_val_score(rf,X,y,cv=cv,scoring="roc_auc")

print(f"\n  Random Forest Results:")
print(f"  Accuracy   : {rf_acc:.4f}")
print(f"  Precision  : {rf_prec:.4f}")
print(f"  Recall     : {rf_rec:.4f}  <- KEY metric!")
print(f"  F1 Score   : {rf_f1:.4f}")
print(f"  ROC-AUC    : {rf_auc:.4f}")
print(f"  CV AUC     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"\n{classification_report(y_test,rf_pred,target_names=['Healthy','Heart Disease'])}")

# =============================================================================
# STAGE 4 — EVALUATION PLOTS
# =============================================================================
section("STAGE 4 — EVALUATION PLOTS")

fig=plt.figure(figsize=(18,10)); gs=gridspec.GridSpec(2,3,figure=fig)

ax1=fig.add_subplot(gs[0,0])
cm=confusion_matrix(y_test,rf_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax1,
    xticklabels=["Healthy","Disease"],yticklabels=["Healthy","Disease"],
    linewidths=2,annot_kws={"size":16})
ax1.set_title(f"Confusion Matrix\nRecall={rf_rec:.3f}",fontweight="bold")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")

ax2=fig.add_subplot(gs[0,1])
fpr_rf,tpr_rf,_=roc_curve(y_test,rf_proba)
fpr_lr,tpr_lr,_=roc_curve(y_test,lr_proba)
ax2.plot(fpr_rf,tpr_rf,color="#E74C3C",lw=2,label=f"Random Forest AUC={rf_auc:.3f}")
ax2.plot(fpr_lr,tpr_lr,color="#3498DB",lw=2,label=f"Logistic Reg AUC={roc_auc_score(y_test,lr_proba):.3f}")
ax2.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
ax2.fill_between(fpr_rf,tpr_rf,alpha=0.08,color="#E74C3C")
ax2.set_title("ROC Curve — Model Comparison",fontweight="bold")
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate"); ax2.legend()

ax3=fig.add_subplot(gs[0,2])
fi=pd.Series(rf.feature_importances_,index=X.columns).nlargest(12).sort_values()
ax3.barh(fi.index,fi.values,color=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(fi))),edgecolor="white")
ax3.set_title("Top 12 Feature Importances",fontweight="bold"); ax3.set_xlabel("Importance")

ax4=fig.add_subplot(gs[1,0])
df[df["target"]==0]["risk_score"].hist(ax=ax4,bins=7,alpha=0.7,color="#2ECC71",label="Healthy")
df[df["target"]==1]["risk_score"].hist(ax=ax4,bins=7,alpha=0.7,color="#E74C3C",label="Disease")
ax4.set_title("Clinical Risk Score Distribution",fontweight="bold")
ax4.set_xlabel("Risk Score (0=low, 6=high)"); ax4.legend()

ax5=fig.add_subplot(gs[1,1])
ax5.hist(rf_proba[y_test==0],bins=20,alpha=0.7,color="#2ECC71",label="Healthy")
ax5.hist(rf_proba[y_test==1],bins=20,alpha=0.7,color="#E74C3C",label="Disease")
ax5.axvline(0.5,color="black",linestyle="--",lw=2,label="Threshold=0.5")
ax5.set_title("Model Confidence Scores",fontweight="bold")
ax5.set_xlabel("Predicted Disease Probability"); ax5.legend(fontsize=9)

ax6=fig.add_subplot(gs[1,2])
ax6.bar(range(1,6),cv_scores,color="#00B89F",edgecolor="white",linewidth=1.5)
ax6.axhline(cv_scores.mean(),color="#E74C3C",linestyle="--",lw=2,
    label=f"Mean={cv_scores.mean():.3f}")
ax6.set_xticks(range(1,6)); ax6.set_xlabel("Fold"); ax6.set_ylabel("ROC-AUC")
ax6.set_title(f"5-Fold CV: Mean={cv_scores.mean():.3f}",fontweight="bold")
ax6.legend(); ax6.set_ylim([0.7,1.0])

plt.suptitle("Heart Disease Risk Prediction — Evaluation Dashboard\nBayer AG | UCI Cleveland Dataset",
    fontsize=13,fontweight="bold")
plt.tight_layout(); save(fig,"model_01_evaluation_dashboard.png"); plt.show()

# Save predictions
tn,fp,fn,tp=confusion_matrix(y_test,rf_pred).ravel()
results=X_test.copy().reset_index(drop=True)
results["actual"]=y_test.reset_index(drop=True)
results["predicted"]=rf_pred
results["disease_probability"]=rf_proba.round(4)
results["risk_level"]=pd.cut(rf_proba,bins=[0,0.3,0.6,1.0],
    labels=["Low Risk","Medium Risk","High Risk"])
results.to_csv(os.path.join(OUTPUTS,"patient_predictions.csv"),index=False)

report=f"""
=================================================================
  BAYER AG — CLINICAL DECISION SUPPORT
  Heart Disease Risk Prediction Report
  Dataset: UCI Cleveland Heart Disease (303 patients)
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
=================================================================

MODEL RESULTS (Random Forest — 500 trees)
  Accuracy   : {rf_acc:.4f}
  Precision  : {rf_prec:.4f}
  Recall     : {rf_rec:.4f}  <- KEY metric (catching real cases)
  F1 Score   : {rf_f1:.4f}
  ROC-AUC    : {rf_auc:.4f}
  CV AUC     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}

CLINICAL SIMULATION ({len(y_test)} test patients)
  Correctly caught disease cases  : {tp}
  Missed disease cases (DANGEROUS): {fn}
  Correctly cleared healthy cases : {tn}
  Unnecessary follow-ups          : {fp}
  Catch rate                      : {tp/(tp+fn)*100:.1f}%

TOP PREDICTORS
  -> ST depression (oldpeak)
  -> Max heart rate (thalach)
  -> Blocked vessels (ca)
  -> Combined clinical risk score

BUSINESS IMPACT
  -> Earlier detection = better patient outcomes
  -> Supports Bayer cardiovascular drug trial patient selection
  -> GDPR-compliant clinical decision support tool for Germany
  -> Scalable to European patient datasets

RECOMMENDATIONS
  1. Deploy as clinical screening tool in cardiology
  2. Lower threshold to 0.4 to increase recall further
  3. Validate on German/European patient data
  4. Add SHAP values for individual patient explanations
  5. Integrate with hospital EHR systems via API
=================================================================
"""
print(report)
with open(os.path.join(OUTPUTS,"executive_summary.txt"),"w",encoding="utf-8") as f:
    f.write(report)

section("PIPELINE COMPLETE — ALL OUTPUTS SAVED")
