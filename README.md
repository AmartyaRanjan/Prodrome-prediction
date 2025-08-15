# Parkinson’s Disease Conversion Prediction 🧠

Multimodal ML for Parkinson’s disease (PD): this notebook builds diagnostic and prognostic models on PPMI data, contrasting the easier task of **diagnosing established PD** with the harder task of **predicting conversion** among prodromal participants. It uses clinical motor (UPDRS), non-motor (SCOPA), imaging (DaTscan), genetics, and demographics.

## 📝 Table of Contents
- [Project Description](#-project-description)
- [Features](#-features)
- [Data Sources & Preparation](#-data-sources--preparation)
- [Requirements](#-requirements)
- [How to Run](#installation-how-to-run)
- [Usage Example](#-usage-example)
- [Evaluation & Interpretability](#-evaluation--interpretability)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 📜 Project Description
This project underscores the difference between **classification** and **true prognostic prediction**:

- **Diagnostic task**: Separate diagnosed PD from prodromal individuals using multimodal features (target derived from `COHORT` → `{PD:1, PRODROMAL:0}`).
- **Prognostic task**: Predict which prodromal participants **convert to PD during follow-up** (target `Converted_to_PD` built by linking follow-up records; converters identified via clinical progression markers such as `NHY` > 0).

The notebook:
- Merges multiple PPMI tables on `PATNO`.
- Builds baselines (RandomForest / XGBoost / LightGBM).
- Trains an **intermediate-fusion neural net** (Keras Functional API) with separate “towers” for **demographics**, **motor (UPDRS III)**, and **non-motor (SCOPA)** features, concatenated before the classifier head.
- Evaluates with **ROC AUC**, **accuracy**, **confusion matrix**, and uses **McNemar’s test** to compare classifiers.
- Explains predictions with **SHAP**.

## 🚀 Features
- Multimodal feature handling:
  - **Demographics**: `AGE`, `GENDER`
  - **Motor**: `UPDRS_III_TOTAL`
  - **Non-motor**: `SCOPA_TOTAL`
  - **Imaging**: `DATSCAN`
  - **Genetics**: `GBA_Mutation`, `LRRK2_Mutation`, `PATHVAR_COUNT`
- Reproducible split: `train_test_split(..., test_size=0.25, random_state=42, stratify=y)`
- Models: `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, and a Keras **fusion** model (`Input` → dense towers → `concatenate` → sigmoid head with `EarlyStopping`).
- Metrics: ROC AUC, accuracy, ROC curve, confusion matrix.
- Statistical comparison: **McNemar’s test**.
- Interpretability: **SHAP TreeExplainer** + summary/force plots.

## 📂 Data Sources & Preparation
The notebook expects PPMI-like CSVs (paths in the notebook point to Kaggle inputs). Place equivalent files locally and adjust paths near the **data loading** cells.

Files referenced in the notebook include (names may vary by export date):
- `Genetics.csv`
- `MDS-UPDRS_Part_III_24Jun2025.csv`
- `MRIQC_24Jun2025.csv`
- `SCOPA-AUT_24Jul2025.csv`
- `Screening___Demographics-Archived_24Jun2025.csv`
- `Subject_Cohort_History_24Jul2025.csv`
- `DaTscan_Imaging_24Jul2025.csv`
- `Demographics_24Jul2025.csv`

Core keys/columns:
- Join key: `PATNO`
- Targets: `COHORT` (diagnostic PD vs prodromal), derived `Converted_to_PD` (prognostic)
- Features (examples): `AGE_AT_BASELINE`, `GENDER`, `UPDRS_III_TOTAL`, `SCOPA_TOTAL`, `DATSCAN`, `GBA_Mutation`, `LRRK2_Mutation`, `PATHVAR_COUNT`

## 🛠️ Requirements
Python 3.9+ and the following packages:

- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- `xgboost`, `lightgbm`, `statsmodels`
- `tensorflow` (Keras), `shap`

You can create a quick `requirements.txt` like:

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
statsmodels
tensorflow
shap
```
## 🖥️ Installation & How to Run

You can run the notebook **interactively** or in **headless** mode.

### **Option A — Jupyter (Interactive)**

### **Option B — Headless Execution (nbconvert)**

### **Option C — Papermill (Parameterizable Runs)**

---

## 🎯 Usage Example

**Typical workflow inside the notebook:**
1. Prepare features and target (diagnostic or prognostic).
2. Scale where needed using `StandardScaler` and perform a stratified **75/25 split**.
3. Train baseline models and the fusion model.
4. Evaluate metrics (AUC/accuracy) and inspect ROC & confusion matrix.
5. Run **SHAP** for feature importance.

**Example — Executing Headlessly, then Inspecting Results:**

---

## 📊 Evaluation & Interpretability

- **Metrics:**  
  `roc_auc_score`, `accuracy_score`, `roc_curve`, `confusion_matrix`  
- **Model Comparison:**  
  McNemar test on contingency table of correct/incorrect predictions  
- **Feature Importance:**
  - Tree models: `.feature_importances_`
  - Global & local explanations: `shap.TreeExplainer` with summary and force plots  

**Note:** Diagnosis tends to score much higher than prognosis.  
- Expect **strong AUC** for *PD vs prodromal*.  
- Expect **more modest AUC** when predicting future conversion among prodromals.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Open a PR

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file.

---

## 📬 Contact

For questions or collaboration:  
📧 Email: [your.email@example.com]()  
💻 GitHub: [YourUsername](https://github.com/YourUsername)
