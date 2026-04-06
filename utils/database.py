"""
Database Utility (SQLite)
==========================
Handles: predictions log, user sessions, model results storage
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = 'heart_disease.db'


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            age         REAL,
            sex         REAL,
            cp          REAL,
            trestbps    REAL,
            chol        REAL,
            fbs         REAL,
            restecg     REAL,
            thalach     REAL,
            exang       REAL,
            oldpeak     REAL,
            slope       REAL,
            ca          REAL,
            thal        REAL,
            model_used  TEXT,
            prediction  INTEGER,
            probability REAL,
            risk_level  TEXT,
            created_at  TEXT
        )
    ''')

    # Model performance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT UNIQUE,
            accuracy    REAL,
            precision   REAL,
            recall      REAL,
            f1_score    REAL,
            auc_score   REAL,
            trained_at  TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized.")


def save_prediction(input_data: dict, model_used: str, prediction: int,
                    probability: float, risk_level: str):
    """Save a prediction record to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
        (age, sex, cp, trestbps, chol, fbs, restecg, thalach,
         exang, oldpeak, slope, ca, thal, model_used, prediction,
         probability, risk_level, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        input_data.get('age'),      input_data.get('sex'),
        input_data.get('cp'),       input_data.get('trestbps'),
        input_data.get('chol'),     input_data.get('fbs'),
        input_data.get('restecg'),  input_data.get('thalach'),
        input_data.get('exang'),    input_data.get('oldpeak'),
        input_data.get('slope'),    input_data.get('ca'),
        input_data.get('thal'),     model_used,
        prediction, probability, risk_level,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()


def get_all_predictions():
    """Fetch all predictions from DB."""
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_model_performance(name, accuracy, precision, recall, f1, auc):
    """Save or update model performance metrics."""
    conn = get_connection()
    conn.execute('''
        INSERT INTO model_performance
            (model_name, accuracy, precision, recall, f1_score, auc_score, trained_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(model_name) DO UPDATE SET
            accuracy=excluded.accuracy,
            precision=excluded.precision,
            recall=excluded.recall,
            f1_score=excluded.f1_score,
            auc_score=excluded.auc_score,
            trained_at=excluded.trained_at
    ''', (name, accuracy, precision, recall, f1, auc,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()


def get_model_performance():
    """Fetch all model performance records."""
    conn = get_connection()
    rows = conn.execute(
        'SELECT * FROM model_performance ORDER BY accuracy DESC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_stats():
    """Return summary statistics for the dashboard."""
    conn = get_connection()
    total    = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    positive = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction=1"
    ).fetchone()[0]
    negative = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction=0"
    ).fetchone()[0]
    conn.close()
    return {'total': total, 'positive': positive, 'negative': negative}
