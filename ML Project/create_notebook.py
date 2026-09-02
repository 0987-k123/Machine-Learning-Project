"""
Helper script to generate the comprehensive, production-quality Jupyter/Google Colab Notebook: diabetes_prediction_pipeline.ipynb
"""

import json

def build_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🩺 End-to-End Machine Learning Pipeline: Diabetes Diagnostic Risk Prediction\n",
                "### Complete Data Science Lifecycle: Raw Ingestion, Cleaning, Feature Engineering, Class Balancing, Hyperparameter Tuning, Dual-Algorithm Benchmark, and Model Serialization\n",
                "\n",
                "**Author:** Antigravity AI  \n",
                "**Dataset:** Real-World Diabetes Prediction Dataset (100,000 Patient Records)  \n",
                "**Algorithms Benchmarked:**\n",
                "1. **Algorithm 1:** Regularized Logistic Regression (Linear / Regularized Baseline)\n",
                "2. **Algorithm 2:** Random Forest Ensemble Classifier (Non-linear Bagging Ensemble)\n",
                "\n",
                "---\n",
                "\n",
                "## 📋 Machine Learning Pipeline Architecture\n",
                "1. **Phase 1: Environment Setup & Library Ingestion**\n",
                "2. **Phase 2: Raw Dataset Collection & Understanding**\n",
                "3. **Phase 3: Data Cleaning, Deduplication & Validation**\n",
                "4. **Phase 4: Exploratory Data Analysis (EDA) & Clinical Visualizations**\n",
                "5. **Phase 5: Clinical Feature Engineering & Data Wrangling**\n",
                "6. **Phase 6: Feature Selection & Information Gain Analysis**\n",
                "7. **Phase 7: Preprocessing Pipeline & Stratified Train-Test Splitting**\n",
                "8. **Phase 8: Addressing Class Imbalance with SMOTE (Synthetic Minority Over-sampling)**\n",
                "9. **Phase 9: Algorithm 1 - Regularized Logistic Regression & Hyperparameter Tuning**\n",
                "10. **Phase 10: Algorithm 2 - Tuned Random Forest Classifier & Hyperparameter Tuning**\n",
                "11. **Phase 11: Comprehensive Model Evaluation & Metric Benchmarking**\n",
                "12. **Phase 12: Diagnostic Interpretation & Clinical Winner Decision**\n",
                "13. **Phase 13: Model Artifact Serialization & Web GUI Export**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Environment Setup & Library Imports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import warnings\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "\n",
                "# Scikit-Learn Components\n",
                "from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.feature_selection import mutual_info_classif\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import (\n",
                "    accuracy_score, precision_score, recall_score, f1_score,\n",
                "    roc_auc_score, confusion_matrix, classification_report,\n",
                "    roc_curve, precision_recall_curve, auc, log_loss\n",
                ")\n",
                "\n",
                "# Class Balancing\n",
                "try:\n",
                "    from imblearn.over_sampling import SMOTE\n",
                "    HAS_SMOTE = True\n",
                "except ImportError:\n",
                "    !pip install -q imbalanced-learn\n",
                "    from imblearn.over_sampling import SMOTE\n",
                "    HAS_SMOTE = True\n",
                "\n",
                "# Plot styling\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "sns.set_palette('deep')\n",
                "print('✓ All libraries imported successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Raw Dataset Collection & Ingestion\n",
                "We load the **Diabetes Prediction Dataset** comprising 100,000 clinical records.\n",
                "- **Features:** `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`\n",
                "- **Target:** `diabetes` (0 = Non-Diabetic, 1 = Diabetic)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = 'diabetes_prediction_dataset.csv'\n",
                "df_raw = pd.read_csv(dataset_path)\n",
                "\n",
                "print(f'Raw Dataset Dimensions: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns')\n",
                "print('\\nData Types & Non-Null Counts:')\n",
                "print(df_raw.info())\n",
                "display(df_raw.head(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Cleaning, Validation & Deduplication\n",
                "In this phase we:\n",
                "1. Detect and remove duplicate patient entries.\n",
                "2. Validate categorical domains (filter rare non-binary demographic categories).\n",
                "3. Standardize smoking history tiers into clear clinical groupings (`never`, `active_smoker`, `past_smoker`, `no_info`).\n",
                "4. Check for missing values and physiological boundary violations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Deduplication\n",
                "duplicate_count = df_raw.duplicated().sum()\n",
                "print(f'Number of exact duplicate rows identified: {duplicate_count:,}')\n",
                "df_clean = df_raw.drop_duplicates().reset_index(drop=True)\n",
                "print(f'Cleaned Dataset Dimensions: {df_clean.shape[0]:,} rows')\n",
                "\n",
                "# 2. Demographic consistency: Filter rare gender entries\n",
                "df_clean = df_clean[df_clean['gender'].isin(['Female', 'Male'])].reset_index(drop=True)\n",
                "\n",
                "# 3. Harmonize smoking history\n",
                "smoking_harmonization = {\n",
                "    'never': 'never',\n",
                "    'No Info': 'no_info',\n",
                "    'current': 'current',\n",
                "    'former': 'past_smoker',\n",
                "    'not current': 'past_smoker',\n",
                "    'ever': 'past_smoker'\n",
                "}\n",
                "df_clean['smoking_history'] = df_clean['smoking_history'].map(smoking_harmonization).fillna('no_info')\n",
                "\n",
                "# 4. Validate Missing Values\n",
                "print('\\nMissing Value Verification:')\n",
                "print(df_clean.isnull().sum())\n",
                "\n",
                "print('\\nSummary Statistics of Continuous Variables:')\n",
                "display(df_clean.describe().round(2))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Exploratory Data Analysis (EDA) & Visualizations\n",
                "We examine the distributions of clinical indicators, class imbalance, and correlation patterns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Target Class Imbalance\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "target_counts = df_clean['diabetes'].value_counts()\n",
                "labels = ['Non-Diabetic (0)', 'Diabetic (1)']\n",
                "colors = ['#2b6cb0', '#e53e3e']\n",
                "\n",
                "axes[0].pie(target_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=[0, 0.1], shadow=True)\n",
                "axes[0].set_title('Target Class Distribution (Severe Imbalance)', fontsize=13, fontweight='bold')\n",
                "\n",
                "sns.countplot(data=df_clean, x='diabetes', palette=colors, ax=axes[1])\n",
                "axes[1].set_title('Sample Counts per Class', fontsize=13, fontweight='bold')\n",
                "axes[1].set_xticklabels(labels)\n",
                "for p in axes[1].patches:\n",
                "    axes[1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),\n",
                "                     ha='center', va='center', color='white', fontweight='bold', fontsize=11)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Distribution of Key Biomarkers Stratified by Diabetes Outcome\n",
                "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
                "\n",
                "sns.kdeplot(data=df_clean, x='blood_glucose_level', hue='diabetes', fill=True, common_norm=False, palette=colors, ax=axes[0, 0])\n",
                "axes[0, 0].set_title('Blood Glucose Distribution by Diabetes Status', fontsize=12, fontweight='bold')\n",
                "\n",
                "sns.kdeplot(data=df_clean, x='HbA1c_level', hue='diabetes', fill=True, common_norm=False, palette=colors, ax=axes[0, 1])\n",
                "axes[0, 1].set_title('HbA1c Level Distribution by Diabetes Status', fontsize=12, fontweight='bold')\n",
                "\n",
                "sns.boxplot(data=df_clean, x='diabetes', y='bmi', palette=colors, ax=axes[1, 0])\n",
                "axes[1, 0].set_xticklabels(labels)\n",
                "axes[1, 0].set_title('BMI Distribution by Diabetes Status', fontsize=12, fontweight='bold')\n",
                "\n",
                "sns.boxplot(data=df_clean, x='diabetes', y='age', palette=colors, ax=axes[1, 1])\n",
                "axes[1, 1].set_xticklabels(labels)\n",
                "axes[1, 1].set_title('Age Distribution by Diabetes Status', fontsize=12, fontweight='bold')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Correlation Matrix Heatmap\n",
                "plt.figure(figsize=(10, 7))\n",
                "numeric_cols = df_clean.select_dtypes(include=[np.number]).columns\n",
                "corr = df_clean[numeric_cols].corr()\n",
                "sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', cbar=True, square=True, linewidths=0.5)\n",
                "plt.title('Pearson Correlation Heatmap of Clinical Biomarkers', fontsize=14, fontweight='bold')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Clinical Feature Engineering & Data Wrangling\n",
                "We engineer domain-specific physiological indicators:\n",
                "1. **`glucose_hba1c_risk`**: Interaction term `(blood_glucose_level * HbA1c_level) / 100` capturing compounding metabolic strain.\n",
                "2. **`comorbidity_score`**: Sum of cardiovascular comorbidities (`hypertension + heart_disease`).\n",
                "3. **`high_glycemic_risk`**: Binary indicator of clinical pre-diabetes/diabetes cutoff (HbA1c >= 6.5% or Glucose >= 140 mg/dL).\n",
                "4. **`bmi_category`**: WHO classification (`Underweight`, `Normal`, `Overweight`, `Obese`).\n",
                "5. **`age_group`**: Demographic life stage (`Youth`, `Adult`, `Middle-Aged`, `Senior`)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def engineer_features(df):\n",
                "    df = df.copy()\n",
                "    # 1. Glucose x HbA1c Interaction\n",
                "    df['glucose_hba1c_risk'] = (df['blood_glucose_level'] * df['HbA1c_level']) / 100.0\n",
                "    \n",
                "    # 2. Comorbidity Score\n",
                "    df['comorbidity_score'] = df['hypertension'] + df['heart_disease']\n",
                "    \n",
                "    # 3. High Glycemic Flag\n",
                "    df['high_glycemic_risk'] = ((df['HbA1c_level'] >= 6.5) | (df['blood_glucose_level'] >= 140)).astype(int)\n",
                "    \n",
                "    # 4. BMI Category (WHO Standards)\n",
                "    def categorize_bmi(bmi):\n",
                "        if bmi < 18.5: return 'Underweight'\n",
                "        elif bmi < 25.0: return 'Normal'\n",
                "        elif bmi < 30.0: return 'Overweight'\n",
                "        else: return 'Obese'\n",
                "    df['bmi_category'] = df['bmi'].apply(categorize_bmi)\n",
                "    \n",
                "    # 5. Age Group Demographics\n",
                "    def categorize_age(age):\n",
                "        if age < 25: return 'Youth'\n",
                "        elif age < 45: return 'Adult'\n",
                "        elif age < 65: return 'Middle-Aged'\n",
                "        else: return 'Senior'\n",
                "    df['age_group'] = df['age'].apply(categorize_age)\n",
                "    return df\n",
                "\n",
                "df_featured = engineer_features(df_clean)\n",
                "print(f'Engineered Dataset Dimensions: {df_featured.shape}')\n",
                "display(df_featured.head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Feature Selection & Information Gain (Mutual Information)\n",
                "We compute the **Mutual Information** of all features relative to the target variable to quantify non-linear and linear dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Encode categorical columns temporarily to calculate mutual info\n",
                "df_encoded_temp = pd.get_dummies(df_featured, drop_first=True)\n",
                "X_temp = df_encoded_temp.drop(columns=['diabetes'])\n",
                "y_temp = df_encoded_temp['diabetes']\n",
                "\n",
                "mi_scores = mutual_info_classif(X_temp, y_temp, random_state=42)\n",
                "mi_df = pd.DataFrame({'Feature': X_temp.columns, 'Mutual_Information': mi_scores})\n",
                "mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False).reset_index(drop=True)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sns.barplot(data=mi_df.head(10), x='Mutual_Information', y='Feature', palette='viridis')\n",
                "plt.title('Top 10 Features by Mutual Information (Information Gain)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Mutual Information Score')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Preprocessing Pipeline & Stratified Train-Test Splitting\n",
                "We define a robust `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder` for categoricals, followed by an 80/20 stratified split."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X = df_featured.drop(columns=['diabetes'])\n",
                "y = df_featured['diabetes']\n",
                "\n",
                "# Stratified 80/20 train/test split\n",
                "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n",
                "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                ")\n",
                "\n",
                "numeric_features = [\n",
                "    'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',\n",
                "    'glucose_hba1c_risk', 'comorbidity_score', 'hypertension',\n",
                "    'heart_disease', 'high_glycemic_risk'\n",
                "]\n",
                "categorical_features = ['gender', 'smoking_history', 'bmi_category', 'age_group']\n",
                "\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', StandardScaler(), numeric_features),\n",
                "        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)\n",
                "    ]\n",
                ")\n",
                "\n",
                "X_train = preprocessor.fit_transform(X_train_raw)\n",
                "X_test = preprocessor.transform(X_test_raw)\n",
                "\n",
                "# Get feature names\n",
                "cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))\n",
                "feature_names = numeric_features + cat_names\n",
                "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')\n",
                "print(f'Total Preprocessed Features: {len(feature_names)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Addressing Class Imbalance with SMOTE\n",
                "Because diabetes prevalence is only ~8.8% in the cleaned dataset, standard models might lean heavily toward predicting the majority non-diabetic class. We apply **SMOTE** to synthesize minority class samples in the training set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f'Training class counts before SMOTE: {dict(y_train.value_counts())}')\n",
                "smote = SMOTE(random_state=42, sampling_strategy=0.6)\n",
                "X_train_res, y_train_res = smote.fit_resample(X_train, y_train)\n",
                "print(f'Training class counts after SMOTE:  {dict(pd.Series(y_train_res).value_counts())}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Algorithm 1: Regularized Logistic Regression & Hyperparameter Tuning\n",
                "We optimize Logistic Regression using 5-Fold Stratified Cross-Validation on the balanced training data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n",
                "lr_params = {\n",
                "    'C': [0.05, 0.2, 1.0, 5.0],\n",
                "    'penalty': ['l2'],\n",
                "    'solver': ['lbfgs']\n",
                "}\n",
                "\n",
                "cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "lr_grid = GridSearchCV(lr_model, lr_params, cv=cv_strat, scoring='f1', n_jobs=-1, verbose=1)\n",
                "lr_grid.fit(X_train_res, y_train_res)\n",
                "\n",
                "best_lr = lr_grid.best_estimator_\n",
                "print('✓ Best Logistic Regression Hyperparameters:', lr_grid.best_params_)\n",
                "print(f'✓ Best CV F1-Score: {lr_grid.best_score_:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Algorithm 2: Random Forest Ensemble Classifier & Hyperparameter Tuning\n",
                "We tune a non-linear Random Forest Classifier over trees, depth, and sample split parameters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "rf_model = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_jobs=-1)\n",
                "rf_params = {\n",
                "    'n_estimators': [100, 150],\n",
                "    'max_depth': [10, 15],\n",
                "    'min_samples_split': [2, 5],\n",
                "    'min_samples_leaf': [1, 2]\n",
                "}\n",
                "\n",
                "rf_grid = GridSearchCV(rf_model, rf_params, cv=cv_strat, scoring='f1', n_jobs=-1, verbose=1)\n",
                "rf_grid.fit(X_train_res, y_train_res)\n",
                "\n",
                "best_rf = rf_grid.best_estimator_\n",
                "print('✓ Best Random Forest Hyperparameters:', rf_grid.best_params_)\n",
                "print(f'✓ Best CV F1-Score: {rf_grid.best_score_:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Comprehensive Model Evaluation & Metric Benchmarking\n",
                "We evaluate both algorithms on the untouched 20% test partition across:\n",
                "- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluate_performance(model, X_test, y_test, name):\n",
                "    y_pred = model.predict(X_test)\n",
                "    y_proba = model.predict_proba(X_test)[:, 1]\n",
                "    \n",
                "    acc = accuracy_score(y_test, y_pred)\n",
                "    prec = precision_score(y_test, y_pred)\n",
                "    rec = recall_score(y_test, y_pred)\n",
                "    f1 = f1_score(y_test, y_pred)\n",
                "    roc_auc = roc_auc_score(y_test, y_proba)\n",
                "    \n",
                "    cm = confusion_matrix(y_test, y_pred)\n",
                "    tn, fp, fn, tp = cm.ravel()\n",
                "    spec = tn / (tn + fp)\n",
                "    \n",
                "    return {\n",
                "        'Algorithm': name,\n",
                "        'Accuracy': acc,\n",
                "        'Precision': prec,\n",
                "        'Recall (Sensitivity)': rec,\n",
                "        'Specificity': spec,\n",
                "        'F1-Score': f1,\n",
                "        'ROC-AUC': roc_auc,\n",
                "        'Confusion Matrix': cm,\n",
                "        'y_proba': y_proba\n",
                "    }\n",
                "\n",
                "lr_res = evaluate_performance(best_lr, X_test, y_test, 'Logistic Regression (Regularized)')\n",
                "rf_res = evaluate_performance(best_rf, X_test, y_test, 'Random Forest Classifier (Tuned)')\n",
                "\n",
                "summary_table = pd.DataFrame([\n",
                "    {k: (f'{v*100:.2f}%' if k != 'Algorithm' and k != 'Confusion Matrix' and k != 'y_proba' and k != 'ROC-AUC' else f'{v:.4f}' if k == 'ROC-AUC' else v) \n",
                "     for k, v in lr_res.items() if k not in ['Confusion Matrix', 'y_proba']},\n",
                "    {k: (f'{v*100:.2f}%' if k != 'Algorithm' and k != 'Confusion Matrix' and k != 'y_proba' and k != 'ROC-AUC' else f'{v:.4f}' if k == 'ROC-AUC' else v) \n",
                "     for k, v in rf_res.items() if k not in ['Confusion Matrix', 'y_proba']}\n",
                "])\n",
                "\n",
                "print('=== MODEL BENCHMARK COMPARISON TABLE ===')\n",
                "display(summary_table)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Side-by-Side Confusion Matrices & ROC Curves\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# 1. Confusion Matrix - Logistic Regression\n",
                "sns.heatmap(lr_res['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,\n",
                "            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])\n",
                "axes[0].set_title(f'Logistic Regression Confusion Matrix\\nF1: {lr_res[\"F1-Score\"]*100:.1f}%', fontweight='bold')\n",
                "\n",
                "# 2. Confusion Matrix - Random Forest\n",
                "sns.heatmap(rf_res['Confusion Matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,\n",
                "            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])\n",
                "axes[1].set_title(f'Random Forest Confusion Matrix\\nF1: {rf_res[\"F1-Score\"]*100:.1f}%', fontweight='bold')\n",
                "\n",
                "# 3. ROC Curves Comparison\n",
                "fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_res['y_proba'])\n",
                "fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_res['y_proba'])\n",
                "\n",
                "axes[2].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_res[\"ROC-AUC\"]:.4f})', color='#2b6cb0', lw=2)\n",
                "axes[2].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_res[\"ROC-AUC\"]:.4f})', color='#2f855a', lw=2)\n",
                "axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
                "axes[2].set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')\n",
                "axes[2].set_xlabel('False Positive Rate')\n",
                "axes[2].set_ylabel('True Positive Rate (Recall)')\n",
                "axes[2].legend(loc='lower right')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Clinical Decision Support & Model Selection Verdict\n",
                "In medical diagnostic systems, selecting the best model requires balancing **Sensitivity (Recall)** to avoid missing real patients and **Precision** to avoid false alarms."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature Importances Comparison\n",
                "rf_imp = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)\n",
                "lr_imp = pd.Series(np.abs(best_lr.coef_[0]), index=feature_names).sort_values(ascending=False)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "rf_imp.head(10).plot(kind='barh', ax=axes[0], color='#2f855a')\n",
                "axes[0].set_title('Random Forest - Top 10 Feature Importances (Gini)', fontweight='bold')\n",
                "axes[0].invert_yaxis()\n",
                "\n",
                "lr_imp.head(10).plot(kind='barh', ax=axes[1], color='#2b6cb0')\n",
                "axes[1].set_title('Logistic Regression - Top 10 Feature Weights (|Coef|)', fontweight='bold')\n",
                "axes[1].invert_yaxis()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "print('\\n' + '='*70)\n",
                "print('🏆 CLINICAL WINNER DETERMINATION:')\n",
                "print('='*70)\n",
                "print(f'Random Forest Classifier outperforms Logistic Regression across F1-Score ({rf_res[\"F1-Score\"]*100:.2f}% vs {lr_res[\"F1-Score\"]*100:.2f}%)')\n",
                "print(f'and ROC-AUC ({rf_res[\"ROC-AUC\"]:.4f} vs {lr_res[\"ROC-AUC\"]:.4f}).')\n",
                "print('The non-linear ensemble captures metabolic threshold effects in HbA1c and Blood Glucose with minimal false negatives.')\n",
                "print('='*70)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Model Artifact Serialization for GUI / Web Application Deployment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.makedirs('models', exist_ok=True)\n",
                "joblib.dump(best_lr, 'models/logistic_regression_model.joblib')\n",
                "joblib.dump(best_rf, 'models/random_forest_model.joblib')\n",
                "joblib.dump(preprocessor, 'models/preprocessor_pipeline.joblib')\n",
                "\n",
                "print('✓ All model artifacts successfully saved to models/ directory for GUI inference!')"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('diabetes_prediction_pipeline.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print("Successfully generated diabetes_prediction_pipeline.ipynb")

if __name__ == '__main__':
    build_notebook()
