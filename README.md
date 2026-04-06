# Heart-Disease-Prediction

# ❤️ HeartGuard AI — Heart Disease Risk Prediction System

## 📁 Project Folder Structure

heart_disease_prediction/
│
├── app.py                      ← Main Flask web application
├── train_models.py             ← Train all ML models (run first!)
├── requirements.txt            ← Python dependencies
├── heart_disease.db            ← SQLite database (auto-created)
│
├── data/
│   └── heart_disease.csv       ← Dataset (auto-downloaded)
│
├── models/                     ← Saved trained models (.pkl)
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── ann_model.h5            ← (if TensorFlow installed)
│   ├── scaler.pkl
│   └── selected_features.pkl
│
├── utils/
│   ├── download_dataset.py     ← Dataset download helper
│   ├── preprocessing.py        ← Data preprocessing pipeline
│   └── database.py             ← SQLite database operations
│
├── templates/
│   ├── base.html               ← Base layout
│   ├── index.html              ← Home page
│   ├── predict.html            ← Prediction form
│   ├── result.html             ← Prediction result
│   ├── dashboard.html          ← Model performance dashboard
│   ├── history.html            ← Predictions history
│   └── about.html              ← About page
│
├── static/
│   ├── css/                    ← Custom CSS (optional)
│   ├── js/                     ← Custom JS (optional)
│   └── images/                 ← Charts & confusion matrices (auto-generated)
│
└── notebooks/
    └── exploration.ipynb       ← Jupyter notebook for EDA

## ⚙️ Setup Instructions

### Step 1 — Python Version
Use **Python 3.11** (recommended)

python --version   # should show 3.11.x


### Step 2 — Create Virtual Environment

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate


### Step 3 — Install Dependencies

pip install -r requirements.txt


> 💡 For TensorFlow (ANN model), install separately if needed:

> pip install tensorflow==2.15.0


### Step 4 — Download Dataset

python utils/download_dataset.py

This downloads the Cleveland dataset from UCI. If offline, a synthetic fallback is created automatically.

### Step 5 — Train All Models *(Important — do this before running app)*

python train_models.py

This will:
- Preprocess data (handle missing values, outliers, SMOTE, scaling)
- Train 6+ ML models (LR, KNN, SVM, DT, RF, XGBoost, ANN)
- Save models to `models/` folder
- Save performance charts to `static/images/`
- Store metrics in SQLite DB

Expected output:

📊 Dataset loaded: 303 records, 14 features
🔧 Preprocessing data...
🤖 Training models...
  📈 Logistic Regression  Accuracy: 0.8XXX ...
  📈 Random Forest        Accuracy: 0.91XX ...
  📈 XGBoost              Accuracy: 0.93XX ...
🏆 Best Model: XGBoost (Accuracy: 0.93XX)
✅ All models trained & saved successfully!


### Step 6 — Run the Flask App

python app.py

Open browser: **http://127.0.0.1:5000**



## 🌐 Application Pages

| Page      | URL            | Description                        |
|-----------|----------------|------------------------------------|
| Home      | `/`            | Landing page with stats            |
| Predict   | `/predict`     | Patient data input form            |
| Result    | (auto)         | Prediction result with risk level  |
| Dashboard | `/dashboard`   | Model performance metrics & charts |
| History   | `/history`     | All saved predictions              |
| About     | `/about`       | Project info & tech stack          |
| API       | `/api/predict` | REST API endpoint (POST)           |

## 🔗 Free Datasets

1. **Cleveland Heart Disease (UCI)**
   - URL: https://archive.ics.uci.edu/dataset/45/heart+disease
   - 303 records, 14 attributes, FREE

2. **Heart Failure Prediction (Kaggle)**
   - URL: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
   - 918 records, 12 features, FREE (need Kaggle account)


## 🤖 ML Models Used

| Model                  | Library          | Notes                   |
|------------------------|------------------|-------------------------|
| Logistic Regression    | Scikit-learn     | Baseline model          |
| K-Nearest Neighbors    | Scikit-learn     | k=5                     |
| Support Vector Machine | Scikit-learn     | RBF kernel              |
| Decision Tree          | Scikit-learn     | max_depth=5             |
| Random Forest          | Scikit-learn     | 200 estimators          |
| XGBoost                | XGBoost          | 200 estimators, lr=0.05 |
| ANN (Neural Network)   | TensorFlow/Keras | 4-layer dense net       |

## 📡 REST API Usage

### Predict via API

curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 0, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0,
    "thal": 1, "model": "Random Forest"
  }'


### Response

{
  "prediction": 1,
  "probability": 78.43,
  "risk_level": "High",
  "model_used": "Random Forest",
  "result": "Heart Disease Detected"
}

## ⚠️ Disclaimer
This system is for **educational and research purposes only**.
It does NOT replace professional medical diagnosis.
Always consult a qualified healthcare provider.

*HeartGuard AI*
