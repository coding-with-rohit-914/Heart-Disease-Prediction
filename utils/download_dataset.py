"""
Dataset Download Helper
========================
Run this script once to download the Cleveland Heart Disease dataset.
Usage: python utils/download_dataset.py
"""
import pandas as pd
import numpy as np
import os

def create_cleveland_dataset():
    """
    Creates Cleveland Heart Disease dataset from UCI ML Repository.
    14 key attributes used for prediction.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]
    
    print("Downloading Cleveland Heart Disease Dataset...")
    try:
        df = pd.read_csv(url, header=None, names=columns, na_values='?')
        # Convert target to binary (0 = No Disease, 1 = Disease)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/heart_disease.csv', index=False)
        print(f"✅ Dataset saved: data/heart_disease.csv | Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Creating synthetic fallback dataset...")
        return create_synthetic_dataset()


def create_synthetic_dataset():
    """
    Creates a realistic synthetic heart disease dataset
    matching Cleveland dataset structure (303 records, 14 features)
    as fallback when network is unavailable.
    """
    np.random.seed(42)
    n = 303

    age         = np.random.randint(29, 77, n)
    sex         = np.random.choice([0, 1], n, p=[0.32, 0.68])
    cp          = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps    = np.random.randint(94, 200, n)
    chol        = np.random.randint(126, 564, n)
    fbs         = np.random.choice([0, 1], n, p=[0.85, 0.15])
    restecg     = np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
    thalach     = np.random.randint(71, 202, n)
    exang       = np.random.choice([0, 1], n, p=[0.67, 0.33])
    oldpeak     = np.round(np.random.uniform(0, 6.2, n), 1)
    slope       = np.random.choice([0, 1, 2], n, p=[0.21, 0.46, 0.33])
    ca          = np.random.choice([0, 1, 2, 3], n, p=[0.59, 0.22, 0.12, 0.07])
    thal        = np.random.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])

    # Generate realistic target based on risk factors
    risk = (
        (age > 55).astype(int) * 2 +
        (sex == 1).astype(int) +
        (cp == 0).astype(int) * 2 +
        (trestbps > 130).astype(int) +
        (chol > 240).astype(int) +
        (exang == 1).astype(int) * 2 +
        (oldpeak > 2).astype(int) * 2 +
        (ca > 0).astype(int) * 2 +
        (thal == 3).astype(int) * 2
    )
    prob = 1 / (1 + np.exp(-(risk - 7) * 0.5))
    target = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'target': target
    })

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/heart_disease.csv', index=False)
    print(f"✅ Synthetic dataset created: data/heart_disease.csv | Shape: {df.shape}")
    return df


if __name__ == '__main__':
    create_cleveland_dataset()
