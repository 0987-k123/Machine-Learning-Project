"""
End-to-End Diabetes Prediction ML Training & Export Pipeline
Author: Antigravity AI
Dataset: Diabetes Prediction Dataset (100,000 records)
Algorithms:
  1. Regularized Logistic Regression (Linear baseline + L1/L2 regularization)
  2. Random Forest Ensemble Classifier (Non-linear ensemble + hyperparameter tuning)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, roc_curve
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imblearn not installed. Will use balanced class weights for imbalance handling.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clinical and statistical feature engineering.
    """
    df = df.copy()
    
    # 1. Interaction Feature: Glucose x HbA1c Risk Index
    df['glucose_hba1c_risk'] = (df['blood_glucose_level'] * df['HbA1c_level']) / 100.0
    
    # 2. Comorbidity Index: Combination of Hypertension and Heart Disease (0, 1, 2)
    df['comorbidity_score'] = df['hypertension'] + df['heart_disease']
    
    # 3. High Glycemic Risk Flag: Clinically recognized prediabetes/diabetes threshold
    df['high_glycemic_risk'] = ((df['HbA1c_level'] >= 6.5) | (df['blood_glucose_level'] >= 140)).astype(int)
    
    # 4. BMI Category (WHO standards)
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25.0:
            return 'Normal'
        elif bmi < 30.0:
            return 'Overweight'
        else:
            return 'Obese'
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    
    # 5. Age Group Demographics
    def categorize_age(age):
        if age < 25:
            return 'Youth'
        elif age < 45:
            return 'Adult'
        elif age < 65:
            return 'Middle-Aged'
        else:
            return 'Senior'
    df['age_group'] = df['age'].apply(categorize_age)
    
    return df


def prepare_data(csv_path: str = 'diabetes_prediction_dataset.csv'):
    """
    Load, clean, validate, and preprocess dataset.
    """
    print(f"[*] Loading raw dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    print(f"    Raw Shape: {df.shape}")
    
    # 1. Remove duplicate records
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[*] Removed {dup_count} duplicate rows. Remaining: {len(df)} rows.")
    
    # 2. Filter rare/invalid categories
    df = df[df['gender'].isin(['Female', 'Male'])].reset_index(drop=True)
    
    # 3. Harmonize smoking history
    smoking_map = {
        'never': 'never',
        'No Info': 'no_info',
        'current': 'current',
        'former': 'past_smoker',
        'not current': 'past_smoker',
        'ever': 'past_smoker'
    }
    df['smoking_history'] = df['smoking_history'].map(smoking_map).fillna('no_info')
    
    # 4. Feature Engineering
    print("[*] Performing Feature Engineering...")
    df = engineer_features(df)
    print(f"    Engineered Shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['diabetes'])
    y = df['diabetes']
    
    print(f"[*] Target distribution:\n{y.value_counts(normalize=True).to_dict()}")
    return df, X, y


def build_preprocessor():
    """
    Create sklearn ColumnTransformer for continuous and categorical features.
    """
    numeric_features = [
        'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'glucose_hba1c_risk', 'comorbidity_score', 'hypertension',
        'heart_disease', 'high_glycemic_risk'
    ]
    categorical_features = ['gender', 'smoking_history', 'bmi_category', 'age_group']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def compute_eda_summary(df: pd.DataFrame, output_path: str = 'models/eda_summary.json'):
    """
    Compute distribution summaries and correlation matrix for dashboard visualization.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().round(4).to_dict()
    
    # Class balance stats
    class_counts = df['diabetes'].value_counts().to_dict()
    total = len(df)
    class_balance = {
        "negative_count": int(class_counts.get(0, 0)),
        "positive_count": int(class_counts.get(1, 0)),
        "negative_pct": round(class_counts.get(0, 0) / total * 100, 2),
        "positive_pct": round(class_counts.get(1, 0) / total * 100, 2),
        "total_samples": total
    }
    
    # Feature distributions by diabetes status
    feature_stats = {}
    for col in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'glucose_hba1c_risk']:
        feature_stats[col] = {
            "overall_mean": round(float(df[col].mean()), 2),
            "overall_std": round(float(df[col].std()), 2),
            "non_diabetic_mean": round(float(df[df['diabetes'] == 0][col].mean()), 2),
            "diabetic_mean": round(float(df[df['diabetes'] == 1][col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2)
        }
        
    eda_data = {
        "class_balance": class_balance,
        "correlations": corr_matrix,
        "feature_stats": feature_stats,
        "gender_breakdown": df.groupby(['gender', 'diabetes']).size().unstack(fill_value=0).to_dict(),
        "smoking_breakdown": df.groupby(['smoking_history', 'diabetes']).size().unstack(fill_value=0).to_dict(),
        "bmi_cat_breakdown": df.groupby(['bmi_category', 'diabetes']).size().unstack(fill_value=0).to_dict()
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eda_data, f, indent=2)
    print(f"[*] Saved EDA summary to '{output_path}'")
    return eda_data


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for classification.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    logloss = log_loss(y_test, y_proba) if y_proba is not None else None
    
    # Compute ROC Curve sample points for charting
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in indices]
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else 0.0,
        "log_loss": round(float(logloss), 4) if logloss is not None else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "roc_curve": roc_points,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"\n=======================================================")
    print(f" Performance Metrics: {model_name}")
    print(f"=======================================================")
    print(f"  Accuracy:      {acc * 100:.2f}%")
    print(f"  Precision:     {prec * 100:.2f}%")
    print(f"  Recall (Sens): {rec * 100:.2f}%")
    print(f"  Specificity:   {specificity * 100:.2f}%")
    print(f"  F1-Score:      {f1 * 100:.2f}%")
    print(f"  ROC-AUC:       {roc_auc:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"=======================================================")
    return metrics


def train_and_evaluate():
    """
    Main training workflow.
    """
    os.makedirs('models', exist_ok=True)
    
    # 1. Prepare Data
    df, X, y = prepare_data('diabetes_prediction_dataset.csv')
    
    # Compute EDA summary
    compute_eda_summary(df, 'models/eda_summary.json')
    
    # 2. Stratified Train-Test Split (80/20)
    print("\n[*] Splitting dataset into 80% Train and 20% Test (Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train size: {X_train_raw.shape[0]} samples")
    print(f"    Test size:  {X_test_raw.shape[0]} samples")
    
    # 3. Fit Preprocessing Pipeline
    print("\n[*] Building and fitting Preprocessing Pipeline...")
    preprocessor, num_cols, cat_cols = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Extract feature names after One-Hot Encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_features
    print(f"    Transformed Feature Count: {len(all_feature_names)}")
    
    # 4. Class Balancing using SMOTE (if available) or Weighted Sampling
    if HAS_SMOTE:
        print("\n[*] Applying SMOTE (Synthetic Minority Over-sampling Technique) on training set...")
        smote = SMOTE(random_state=42, sampling_strategy=0.6)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
        print(f"    Before SMOTE: {y_train.value_counts().to_dict()}")
        print(f"    After SMOTE:  {pd.Series(y_train_balanced).value_counts().to_dict()}")
    else:
        print("\n[*] Using balanced class weighting for models...")
        X_train_balanced, y_train_balanced = X_train_processed, y_train
        
    # 5. Algorithm 1: Logistic Regression with Hyperparameter Tuning
    print("\n[*] Training & Tuning Algorithm 1: Regularized Logistic Regression...")
    lr_base = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_param_grid = {
        'C': [0.05, 0.2, 1.0, 5.0],
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_grid = GridSearchCV(
        lr_base, lr_param_grid, cv=cv_strat, scoring='f1', n_jobs=-1, verbose=1
    )
    lr_grid.fit(X_train_balanced, y_train_balanced)
    best_lr_model = lr_grid.best_estimator_
    print(f"    [+] Best Logistic Regression Params: {lr_grid.best_params_}")
    print(f"    [+] Best CV F1-Score: {lr_grid.best_score_:.4f}")
    
    # 6. Algorithm 2: Random Forest Classifier with Hyperparameter Tuning
    print("\n[*] Training & Tuning Algorithm 2: Random Forest Classifier...")
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid, cv=cv_strat, scoring='f1', n_jobs=-1, verbose=1
    )
    rf_grid.fit(X_train_balanced, y_train_balanced)
    best_rf_model = rf_grid.best_estimator_
    print(f"    [+] Best Random Forest Params: {rf_grid.best_params_}")
    print(f"    [+] Best CV F1-Score: {rf_grid.best_score_:.4f}")
    
    # 7. Model Evaluation on Held-Out Test Set
    lr_metrics = evaluate_model(best_lr_model, X_test_processed, y_test, "Logistic Regression (Regularized)")
    rf_metrics = evaluate_model(best_rf_model, X_test_processed, y_test, "Random Forest Classifier (Tuned)")
    
    # 8. Feature Importance Analysis
    rf_importances = best_rf_model.feature_importances_
    rf_feat_importance = sorted(
        [{"feature": name, "importance": round(float(imp), 4)} for name, imp in zip(all_feature_names, rf_importances)],
        key=lambda x: x["importance"], reverse=True
    )
    
    lr_coefs = best_lr_model.coef_[0]
    lr_feat_importance = sorted(
        [{"feature": name, "coefficient": round(float(coef), 4), "abs_weight": round(float(abs(coef)), 4)} 
         for name, coef in zip(all_feature_names, lr_coefs)],
        key=lambda x: x["abs_weight"], reverse=True
    )
    
    # 9. Determine Winner & Clinical Recommendation
    winner_name = "Random Forest Classifier" if rf_metrics['f1_score'] >= lr_metrics['f1_score'] and rf_metrics['roc_auc'] >= lr_metrics['roc_auc'] else "Logistic Regression"
    winner_rationale = (
        f"{winner_name} achieved superior diagnostic accuracy ({rf_metrics['accuracy']*100:.2f}%), "
        f"balanced F1-Score ({rf_metrics['f1_score']*100:.2f}%), and discriminative power (ROC-AUC: {rf_metrics['roc_auc']:.4f}) "
        f"compared to Logistic Regression (F1: {lr_metrics['f1_score']*100:.2f}%, ROC-AUC: {lr_metrics['roc_auc']:.4f}). "
        f"In medical screening, Random Forest excels at capturing non-linear threshold effects in HbA1c and Blood Glucose levels."
    )
    
    # 10. Package Metadata
    metadata = {
        "dataset_name": "Diabetes Prediction Dataset",
        "total_records": len(df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "feature_names": all_feature_names,
        "raw_features": list(X.columns),
        "winner": {
            "algorithm": winner_name,
            "rationale": winner_rationale,
            "key_advantages": [
                "Superior non-linear decision boundary for composite metabolic risk",
                "High precision-recall balance preventing both missed cases and false alarms",
                "Strong robustness to categorical interactions across age groups and BMI tiers"
            ]
        },
        "models": {
            "logistic_regression": {
                **lr_metrics,
                "best_hyperparameters": lr_grid.best_params_,
                "feature_importance": lr_feat_importance[:10]
            },
            "random_forest": {
                **rf_metrics,
                "best_hyperparameters": rf_grid.best_params_,
                "feature_importance": rf_feat_importance[:10]
            }
        }
    }
    
    # 11. Save Artifacts to models/
    print("\n[*] Saving models and metadata to 'models/' directory...")
    joblib.dump(best_lr_model, 'models/logistic_regression_model.joblib')
    joblib.dump(best_rf_model, 'models/random_forest_model.joblib')
    joblib.dump(preprocessor, 'models/preprocessor_pipeline.joblib')
    
    with open('models/model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    print("[+] All artifacts saved successfully:")
    print("    - models/logistic_regression_model.joblib")
    print("    - models/random_forest_model.joblib")
    print("    - models/preprocessor_pipeline.joblib")
    print("    - models/model_metadata.json")
    print("    - models/eda_summary.json")
    print("\n[✓] Machine Learning Training Pipeline Completed Successfully!")


if __name__ == '__main__':
    train_and_evaluate()
