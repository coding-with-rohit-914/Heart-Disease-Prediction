"""
Heart Disease Risk Prediction — Flask Application
===================================================
Run: python app.py
"""
import os, warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np

from utils.preprocessing import preprocess_input, FEATURE_DESCRIPTIONS
from utils.database      import (
    init_db, save_prediction, get_all_predictions,
    get_model_performance, get_prediction_stats
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'heart_disease_secret_2025')

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {}
MODEL_FILES = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'KNN':                 'models/knn.pkl',
    'SVM':                 'models/svm.pkl',
    'Decision Tree':       'models/decision_tree.pkl',
    'Random Forest':       'models/random_forest.pkl',
    'XGBoost':             'models/xgboost.pkl',
}

def load_models():
    global MODELS
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
            print(f"  ✅ Loaded: {name}")
        else:
            print(f"  ⚠️  Not found: {name} — run train_models.py first")

    # Try loading ANN
    ann_path = 'models/ann_model.h5'
    if os.path.exists(ann_path):
        try:
            from tensorflow import keras
            MODELS['ANN'] = keras.models.load_model(ann_path)
            print("  ✅ Loaded: ANN")
        except Exception as e:
            print(f"  ⚠️  ANN load failed: {e}")


def get_risk_level(probability):
    if probability < 0.30:  return 'Low',    'success'
    if probability < 0.60:  return 'Medium',  'warning'
    return 'High', 'danger'


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Landing / home page."""
    stats = get_prediction_stats()
    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and result."""
    available_models = list(MODELS.keys()) if MODELS else []

    if request.method == 'POST':
        try:
            input_data = {
                'age':      float(request.form['age']),
                'sex':      float(request.form['sex']),
                'cp':       float(request.form['cp']),
                'trestbps': float(request.form['trestbps']),
                'chol':     float(request.form['chol']),
                'fbs':      float(request.form['fbs']),
                'restecg':  float(request.form['restecg']),
                'thalach':  float(request.form['thalach']),
                'exang':    float(request.form['exang']),
                'oldpeak':  float(request.form['oldpeak']),
                'slope':    float(request.form['slope']),
                'ca':       float(request.form['ca']),
                'thal':     float(request.form['thal']),
            }
            model_name = request.form.get('model', 'Random Forest')

            if model_name not in MODELS:
                return render_template('predict.html',
                    error=f"Model '{model_name}' not loaded. Run train_models.py first.",
                    available_models=available_models,
                    feature_descriptions=FEATURE_DESCRIPTIONS)

            model      = MODELS[model_name]
            X_scaled   = preprocess_input(input_data)

            # Handle ANN separately
            if model_name == 'ANN':
                prob = float(model.predict(X_scaled).flatten()[0])
                pred = int(prob >= 0.5)
            else:
                pred = int(model.predict(X_scaled)[0])
                prob = float(model.predict_proba(X_scaled)[0][1])

            risk_label, risk_color = get_risk_level(prob)

            save_prediction(input_data, model_name, pred, prob, risk_label)

            return render_template('result.html',
                prediction   = pred,
                probability  = round(prob * 100, 2),
                risk_level   = risk_label,
                risk_color   = risk_color,
                model_used   = model_name,
                input_data   = input_data,
                feature_desc = FEATURE_DESCRIPTIONS)

        except Exception as e:
            return render_template('predict.html',
                error=str(e),
                available_models=available_models,
                feature_descriptions=FEATURE_DESCRIPTIONS)

    return render_template('predict.html',
        available_models=available_models,
        feature_descriptions=FEATURE_DESCRIPTIONS)


@app.route('/dashboard')
def dashboard():
    """Model performance dashboard."""
    performances = get_model_performance()
    stats        = get_prediction_stats()
    predictions  = get_all_predictions()[:10]   # latest 10
    return render_template('dashboard.html',
        performances=performances,
        stats=stats,
        predictions=predictions)


@app.route('/history')
def history():
    """Full predictions history."""
    predictions = get_all_predictions()
    return render_template('history.html', predictions=predictions)


@app.route('/about')
def about():
    return render_template('about.html')


# ── API endpoints (JSON) ─────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for prediction."""
    try:
        data       = request.get_json()
        model_name = data.get('model', 'Random Forest')
        input_data = {k: float(v) for k, v in data.items() if k != 'model'}

        if model_name not in MODELS:
            return jsonify({'error': f"Model '{model_name}' not available"}), 400

        model    = MODELS[model_name]
        X_scaled = preprocess_input(input_data)

        if model_name == 'ANN':
            prob = float(model.predict(X_scaled).flatten()[0])
            pred = int(prob >= 0.5)
        else:
            pred = int(model.predict(X_scaled)[0])
            prob = float(model.predict_proba(X_scaled)[0][1])

        risk_label, _ = get_risk_level(prob)
        save_prediction(input_data, model_name, pred, prob, risk_label)

        return jsonify({
            'prediction':  pred,
            'probability': round(prob * 100, 2),
            'risk_level':  risk_label,
            'model_used':  model_name,
            'result':      'Heart Disease Detected' if pred == 1 else 'No Heart Disease'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def api_models():
    return jsonify({'available_models': list(MODELS.keys())})


@app.route('/api/stats')
def api_stats():
    return jsonify(get_prediction_stats())


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n❤️  Heart Disease Prediction System")
    print("=" * 40)
    init_db()
    print("\n📦 Loading ML models...")
    load_models()

    if not MODELS:
        print("\n⚠️  No models loaded! Please run: python train_models.py")
    else:
        print(f"\n✅ {len(MODELS)} model(s) ready.")

    print("\n🚀 Starting Flask server → http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
