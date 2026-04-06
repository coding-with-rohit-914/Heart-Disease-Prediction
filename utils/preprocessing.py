"""
Data Preprocessing Utility
============================
Handles: Missing values, outliers, encoding, normalization, SMOTE, PCA
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

SCALER_PATH  = 'models/scaler.pkl'
PCA_PATH     = 'models/pca.pkl'
FEATURE_PATH = 'models/selected_features.pkl'

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_DESCRIPTIONS = {
    'age':      'Age (years)',
    'sex':      'Sex (1=Male, 0=Female)',
    'cp':       'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mmHg)',
    'chol':     'Serum Cholesterol (mg/dl)',
    'fbs':      'Fasting Blood Sugar > 120 mg/dl (1=True)',
    'restecg':  'Resting ECG Results (0-2)',
    'thalach':  'Maximum Heart Rate Achieved',
    'exang':    'Exercise Induced Angina (1=Yes)',
    'oldpeak':  'ST Depression (Exercise vs Rest)',
    'slope':    'Slope of Peak Exercise ST Segment (0-2)',
    'ca':       'Number of Major Vessels (0-3)',
    'thal':     'Thalassemia (1=Normal, 2=Fixed, 3=Reversible)'
}


def load_data(path='data/heart_disease.csv'):
    df = pd.read_csv(path)
    return df


def handle_missing_values(df):
    """Fill missing values with median for numeric cols."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def remove_outliers(df, cols=None):
    """Remove outliers using IQR method."""
    if cols is None:
        cols = ['trestbps', 'chol', 'thalach', 'oldpeak']
    for col in cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df.reset_index(drop=True)


def apply_smote(X, y):
    """Balance classes using SMOTE."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  SMOTE applied: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def scale_features(X_train, X_test=None, fit=True):
    """Standardize features."""
    scaler = StandardScaler()
    if fit:
        X_train_scaled = scaler.fit_transform(X_train)
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test=None, n_components=0.95, fit=True):
    """Apply PCA for dimensionality reduction."""
    if fit:
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        joblib.dump(pca, PCA_PATH)
        print(f"  PCA: {X_train.shape[1]} → {X_train_pca.shape[1]} components")
    else:
        pca = joblib.load(PCA_PATH)
        X_train_pca = pca.transform(X_train)

    X_test_pca = pca.transform(X_test) if X_test is not None else None
    return X_train_pca, X_test_pca, pca


def select_features_rfe(X, y, n_features=10):
    """Feature selection using RFE with Random Forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected = [FEATURE_NAMES[i] for i, s in enumerate(rfe.support_) if s]
    joblib.dump(selected, FEATURE_PATH)
    print(f"  RFE Selected Features: {selected}")
    return selected, rfe


def preprocess_input(input_dict):
    """
    Preprocess a single patient input dict for prediction.
    Returns scaled numpy array ready for model.
    """
    df = pd.DataFrame([input_dict])
    # Ensure correct column order
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df.values)
    return X_scaled


def full_preprocessing_pipeline(df, apply_smote_flag=True, use_pca=False):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. Remove outliers
    3. Split features/target
    4. Scale features
    5. Optionally apply SMOTE
    6. Optionally apply PCA
    Returns X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    df = handle_missing_values(df)
    df = remove_outliers(df)

    X = df[FEATURE_NAMES].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, fit=True)

    if apply_smote_flag:
        X_train_s, y_train = apply_smote(X_train_s, y_train)

    if use_pca:
        X_train_s, X_test_s, _ = apply_pca(X_train_s, X_test_s, fit=True)

    print(f"  Train: {X_train_s.shape}, Test: {X_test_s.shape}")
    return X_train_s, X_test_s, y_train, y_test
