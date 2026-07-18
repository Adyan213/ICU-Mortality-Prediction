ICU Mortality Prediction
A machine learning pipeline for predicting in-hospital mortality of ICU patients using clinical data, built with scikit-learn and XGBoost.
Dataset
WiDS Datathon 2020 — 91,713 ICU patient records with 186 clinical features including vitals, lab values, demographics, and APACHE severity scores.
Notebook
View Notebook on nbviewer
Pipeline

Dropped 55 features with >70% missing values
Median imputation for numerical features, mode imputation for categorical
One-hot encoding for low cardinality categorical features
Class imbalance handling via scale_pos_weight (1:10 survival:death ratio)
Removed APACHE pre-computed mortality scores to test raw feature predictiveness

Models
Logistic Regression (Baseline)
MetricScoreAUC-ROC0.796Died Recall78%Died Precision29%Macro F10.65
XGBoost (No APACHE Scores)
MetricScoreAUC-ROC0.900Died Recall71%Died Precision38%Macro F10.71
Key Finding
Removing pre-computed Apache mortality probability scores did not degrade performance — the model achieved equivalent AUC-ROC (0.900) using only raw clinical features. This suggests the model independently learned clinically meaningful mortality patterns from vitals, labs, and patient history.
SHAP Analysis
SHAP values confirm the model learned clinically coherent patterns:
Show Image

ventilation requirement — strongest predictor, mechanical ventilation indicates severe illness
age — older patients have significantly higher mortality risk
GCS components — low consciousness level strongly associated with mortality
Day 1 vitals — low SpO2, low blood pressure, high BUN all increase mortality risk
Elective surgery — strongest protective factor, planned procedures indicate healthier baseline

Tech Stack
Python, XGBoost, scikit-learn, SHAP, pandas, matplotlib
