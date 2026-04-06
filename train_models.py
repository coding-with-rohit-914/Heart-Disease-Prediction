"""
Model Training Script
======================
Trains: Logistic Regression, KNN, SVM, Decision Tree,
        Random Forest, XGBoost, ANN (Neural Network)
Run: python train_models.py
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model      import LogisticRegression
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.svm               import SVC
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from xgboost                   import XGBClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from utils.download_dataset  import create_cleveland_dataset
from utils.preprocessing     import full_preprocessing_pipeline, load_data
from utils.database          import init_db, save_model_performance

# ── Try TensorFlow/Keras (optional) ─────────────────────────────────────────
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not found — ANN will be skipped.")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA SETUP
# ════════════════════════════════════════════════════════════════════════════
def prepare_data():
    if not os.path.exists('data/heart_disease.csv'):
        create_cleveland_dataset()
    df = load_data()
    print(f"\n📊 Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. SKLEARN MODELS DEFINITION
# ════════════════════════════════════════════════════════════════════════════
def get_sklearn_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(
                                   n_estimators=200, max_depth=10,
                                   random_state=42, n_jobs=-1),
        'XGBoost':             XGBClassifier(
                                   n_estimators=200, learning_rate=0.05,
                                   max_depth=6, random_state=42,
                                   eval_metric='logloss', verbosity=0),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. ANN MODEL (TensorFlow/Keras)
# ════════════════════════════════════════════════════════════════════════════
def build_ann(input_dim):
    model = keras.Sequential([
        layers.Dense(64,  activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32,  activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16,  activation='relu'),
        layers.Dense(1,   activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ════════════════════════════════════════════════════════════════════════════
def evaluate_model(name, y_test, y_pred, y_prob=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

    print(f"\n  📈 {name}")
    print(f"     Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
    print(f"     Recall   : {rec:.4f}  |  F1-Score : {f1:.4f}")
    print(f"     AUC-ROC  : {auc:.4f}")
    return acc, prec, rec, f1, auc


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'])
    ax.set_title(f'Confusion Matrix — {name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    fname = name.lower().replace(' ', '_')
    plt.savefig(f'static/images/cm_{fname}.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results):
    models = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    values  = np.array([[results[m][i] for i in range(5)] for m in models])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.15
    colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, values[:, i], width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("\n✅ Comparison chart saved.")


def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx],
                palette='viridis', ax=ax)
    ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance chart saved.")


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════
def train_all():
    print("=" * 60)
    print("   Heart Disease Prediction — Model Training")
    print("=" * 60)

    init_db()
    df = prepare_data()

    print("\n🔧 Preprocessing data...")
    X_train, X_test, y_train, y_test = full_preprocessing_pipeline(
        df, apply_smote_flag=True, use_pca=False
    )

    models   = get_sklearn_models()
    results  = {}
    feature_names = [
        'age','sex','cp','trestbps','chol','fbs',
        'restecg','thalach','exang','oldpeak','slope','ca','thal'
    ]

    print("\n🤖 Training models...\n" + "-" * 40)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, auc = evaluate_model(name, y_test, y_pred, y_prob)
        results[name] = (acc, prec, rec, f1, auc)

        plot_confusion_matrix(name, y_test, y_pred)
        save_model_performance(name, acc, prec, rec, f1, auc)

        fname = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/{fname}.pkl')
        print(f"  💾 Saved: models/{fname}.pkl")

    # Plot feature importance for Random Forest
    rf_model = models['Random Forest']
    plot_feature_importance(rf_model, feature_names)

    # ── ANN ──────────────────────────────────────────────────────────────
    if TF_AVAILABLE:
        print("\n🧠 Training ANN (Neural Network)...")
        ann = build_ann(X_train.shape[1])
        ann.fit(X_train, y_train, epochs=50, batch_size=32,
                validation_split=0.1, verbose=0)
        y_prob_ann = ann.predict(X_test).flatten()
        y_pred_ann = (y_prob_ann >= 0.5).astype(int)

        acc, prec, rec, f1, auc = evaluate_model(
            'ANN', y_test, y_pred_ann, y_prob_ann)
        results['ANN'] = (acc, prec, rec, f1, auc)

        plot_confusion_matrix('ANN', y_test, y_pred_ann)
        save_model_performance('ANN', acc, prec, rec, f1, auc)
        ann.save('models/ann_model.h5')
        print("  💾 Saved: models/ann_model.h5")

    # ── Summary ──────────────────────────────────────────────────────────
    plot_model_comparison(results)

    best_model = max(results, key=lambda k: results[k][0])
    print(f"\n🏆 Best Model: {best_model} "
          f"(Accuracy: {results[best_model][0]:.4f})")
    print("\n✅ All models trained & saved successfully!")


if __name__ == '__main__':
    train_all()
