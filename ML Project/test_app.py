from app import run_dual_inference, load_ml_artifacts

print("Testing ML Artifact loading...")
load_ml_artifacts()

test_patients = [
    {
        'name': 'Profile A: Healthy Young Adult',
        'data': {'gender': 'Female', 'age': 22, 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never', 'bmi': 21.2, 'HbA1c_level': 4.7, 'blood_glucose_level': 88}
    },
    {
        'name': 'Profile B: Borderline Pre-Diabetic',
        'data': {'gender': 'Male', 'age': 49, 'hypertension': 1, 'heart_disease': 0, 'smoking_history': 'past_smoker', 'bmi': 28.5, 'HbA1c_level': 6.2, 'blood_glucose_level': 138}
    },
    {
        'name': 'Profile C: High-Risk Severe Diabetic',
        'data': {'gender': 'Male', 'age': 56, 'hypertension': 1, 'heart_disease': 0, 'smoking_history': 'current', 'bmi': 36.2, 'HbA1c_level': 8.6, 'blood_glucose_level': 245}
    }
]

for p in test_patients:
    res = run_dual_inference(p['data'])
    print(f"\n==========================================")
    print(f" Patient: {p['name']}")
    print(f"==========================================")
    print(f"  LR Prediction: {res['logistic_regression']['prediction']} ({res['logistic_regression']['probability_percentage']}%) - {res['logistic_regression']['risk_category']}")
    print(f"  RF Prediction: {res['random_forest']['prediction']} ({res['random_forest']['probability_percentage']}%) - {res['random_forest']['risk_category']}")
    print(f"  Agreement:     {res['consensus']['agreement']} | Final Verdict: {res['consensus']['final_verdict']}")
    print(f"  Clinical Rec:  {res['clinical_recommendation']}")
print("\n[+] Verification completed successfully!")
