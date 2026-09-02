# 🩺 GlycoVision AI: End-to-End Machine Learning Pipeline & Dual-Model Clinical Decision Support System

An end-to-end, production-grade Machine Learning application and interactive comparative GUI built using the **100,000-record Diabetes Prediction Dataset**.

---

## 🌟 Key Highlights & Features

- **Real-World Healthcare Dataset:** 100,000 patient records across 8 physiological biomarkers and demographic features.
- **Full Data Science Pipeline:**
  1. **Data Cleaning & Validation:** Deduplication (3,854 duplicates removed), categorical validation, boundary integrity checks.
  2. **Clinical Feature Engineering:** Interaction terms (`glucose_hba1c_risk`), WHO BMI classifications, age demographics, comorbidity scoring, and high glycemic risk markers.
  3. **Class Balancing (SMOTE):** Addressed 91.5% vs 8.5% class imbalance using Synthetic Minority Over-sampling to prevent majority-class bias.
  4. **Exploratory Data Analysis:** Histograms, KDE distributions, correlation heatmaps, and biomarker stratification.
  5. **Feature Selection:** Mutual Information (Information Gain) analysis and Tree-based Gini importance ranking.
- **Dual-Model Benchmark:**
  - **Algorithm 1:** Regularized Logistic Regression (Linear / Regularized Baseline)
  - **Algorithm 2:** Random Forest Ensemble Classifier (Non-linear Bagging Trees)
- **Hyperparameter Optimization:** 5-Fold Stratified Cross-Validation using `GridSearchCV` scoring on F1-weighted and ROC-AUC.
- **Comprehensive Evaluation:** Accuracy, Precision (PPV), Recall (Sensitivity), Specificity (TNR), F1-Score, ROC-AUC, and Confusion Matrices.
- **Dual-Model Interactive GUI / Web Dashboard:**
  - Enter patient biomarker values via sliders/inputs or choose one of 4 instant clinical presets.
  - Passes inputs simultaneously to both algorithms.
  - Side-by-side gauge meters, probability percentages, and risk categories.
  - Live metric benchmark comparison table, interactive Chart.js bar and ROC curve charts.
  - Automated "Winner Algorithm" recommendation with medical justification.

---

## 📁 Repository Structure

```
ML Project/
│
├── diabetes_prediction_dataset.csv          # Raw & cleaned dataset (100,000 records)
├── diabetes_prediction_pipeline.ipynb       # Complete Google Colab / Jupyter Notebook
├── train_pipeline.py                       # Standalone training & tuning pipeline script
├── app.py                                  # Web Application Server & Dual Inference API
├── streamlit_app.py                        # Streamlit Dashboard Edition
│
├── models/                                 # Exported ML artifacts
│   ├── logistic_regression_model.joblib    # Trained Logistic Regression model
│   ├── random_forest_model.joblib          # Trained Random Forest model
│   ├── preprocessor_pipeline.joblib        # Preprocessing transformer (Scaler + OneHot)
│   ├── model_metadata.json                 # Comprehensive benchmark metrics & winner info
│   └── eda_summary.json                    # Distribution & correlation data
│
├── static/
│   ├── css/
│   │   └── style.css                       # Modern clinical dark theme & glassmorphism
│   └── js/
│       └── app.js                          # Client-side inference, presets & Chart.js charts
│
└── templates/
    └── index.html                          # Single Page Clinical Decision Studio
```

---

## 🚀 How to Run the Project

### Option 1: Run the Interactive Web GUI Dashboard
Start the high-performance local server:
```bash
python app.py
```
Open your browser at: **`http://localhost:5000`**

### Option 2: Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

### Option 3: Deploy to Streamlit Community Cloud (Free 1-Click Hosting)
1. Push this project repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set **Main file path** to `streamlit_app.py`.
4. Click **Deploy!** Your live application URL will be generated in seconds.

### Option 4: Run in Google Colab / Jupyter Notebook
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File > Upload Notebook** and select `diabetes_prediction_pipeline.ipynb`.
3. Upload `diabetes_prediction_dataset.csv` to the Colab session storage.
4. Run all cells sequentially (**Runtime > Run all**).

### Option 5: Re-Train and Re-Tune the ML Pipeline
```bash
python train_pipeline.py
```

---

## 📊 Benchmark Results Summary

| Performance Metric | Algorithm 1: Logistic Regression | Algorithm 2: Random Forest (Winner) | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 95.25% | **97.18%** | Random Forest achieves superior overall diagnostic correctness. |
| **Precision (PPV)** | 69.80% | **93.20%** | Minimizes false positive alarms by 23.4%. |
| **Recall (Sensitivity)** | **74.50%** | 71.80% | Both capture the vast majority of positive diabetic patients. |
| **Specificity (TNR)** | 97.10% | **99.50%** | Exceptional true negative screening rate. |
| **F1-Score (Weighted)** | 72.08% | **81.12%** | Random Forest delivers an optimal precision-recall harmonic balance. |
| **ROC-AUC Score** | 0.9598 | **0.9785** | Outstanding discriminative power separating diabetic vs healthy cohorts. |

---

## 🏆 Diagnostic Winner Determination

**Winning Algorithm:** **Random Forest Ensemble Classifier**

**Clinical Rationale:**
In medical diagnostic screening, the non-linear Random Forest model excels at identifying composite physiological tipping points (e.g., when elevated blood glucose compounds with high BMI and borderline HbA1c). Its significantly higher **ROC-AUC (0.9785)** and **F1-Score (81.12%)** make it the superior model for reliable clinical decision support.
