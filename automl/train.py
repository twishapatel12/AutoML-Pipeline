# train.py

import os, joblib, base64, warnings, logging, json, time
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from dotenv import load_dotenv

from automl.pipeline import (
    detect_task, build_pipeline, generate_diagnostic_plots,
    generate_shap_summary, generate_report
)

import mlflow

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# === Setup ===
warnings.filterwarnings("ignore")
load_dotenv()

# === Logging with rotation ===
from logging.handlers import RotatingFileHandler
LOG_PATH = os.getenv("LOG_PATH", "app.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.abspath("saved_models"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model(df, target_column, user_id="default_user", task_override=None):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    user_model_dir = os.path.join(MODEL_DIR, user_id)
    os.makedirs(user_model_dir, exist_ok=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    task = task_override or detect_task(y)
    logger.info(f"[{user_id}] Detected task: {task}")

    model_pool = {
        "classification": {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "GradientBoosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True),
            "NaiveBayes": GaussianNB()
        },
        "regression": {
            "RandomForestRegressor": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "GradientBoostingRegressor": GradientBoostingRegressor()
        }
    }

    best_score = float("-inf")
    best_pipeline, best_model_name, best_metrics = None, None, {}
    benchmarking_report = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models_to_train = model_pool[task]

    for name, model in models_to_train.items():
        try:
            pipeline = build_pipeline(model, X)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if task == "classification":
                score = accuracy_score(y_test, y_pred)
                metrics = {
                    "accuracy": score,
                    "f1_score": f1_score(y_test, y_pred, average="weighted"),
                    "cv_mean_accuracy": cross_val_score(pipeline, X, y, cv=5).mean()
                }
            else:
                score = r2_score(y_test, y_pred)
                metrics = {
                    "r2_score": score,
                    "mse": mean_squared_error(y_test, y_pred),
                    "cv_mean_r2": cross_val_score(pipeline, X, y, cv=5).mean()
                }

            benchmarking_report[name] = metrics

            if score > best_score:
                best_score = score
                best_model_name = name
                best_pipeline = pipeline
                best_metrics = metrics

        except Exception as e:
            logger.warning(f"{name} failed: {e}")
            benchmarking_report[name] = {"error": str(e)}

    if not best_pipeline:
        raise RuntimeError("No model successfully trained.")

    # Final training on full data
    best_pipeline.fit(X, y)

    # === Save model with timestamp ===
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(user_model_dir, model_filename)
    joblib.dump(best_pipeline, model_path)

    with open(model_path, "rb") as f:
        model_binary = base64.b64encode(f.read()).decode()

    # === Log to MLflow ===
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("AutoML-Pipeline")
    with mlflow.start_run():
        mlflow.log_params({"model_type": best_model_name, "task_type": task, "target": target_column})
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_pipeline, "model")

    feature_importance = []
    model_step = best_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        fi = model_step.feature_importances_
        feature_names = X.columns
        feature_importance = [{"feature": f, "importance": float(i)} for f, i in zip(feature_names, fi)]

    diag_plot = generate_diagnostic_plots(y_test, best_pipeline.predict(X_test), task)
    shap_plot = generate_shap_summary(best_pipeline, X)

    # === Save report with timestamp ===
    report_filename = f"report_{timestamp}.html"
    report_path = generate_report(
        best_model_name, task, target_column,
        best_metrics, feature_importance,
        diag_plot, shap_plot,
        save_dir=user_model_dir
    )
    report_path = os.path.join(user_model_dir, report_filename)

    # Save benchmarking report
    benchmarking_path = os.path.join(user_model_dir, "benchmarking_report.json")
    with open(benchmarking_path, "w") as f:
        json.dump(benchmarking_report, f, indent=2)

    training_time = round(time.time() - start_time, 2)
    logger.info(f"[{user_id}] Training complete in {training_time}s using model {best_model_name}")

    return {
        "model_path": model_path,
        "model_binary": model_binary,
        "best_model_type": best_model_name,
        "metrics": best_metrics,
        "benchmarking_report": benchmarking_report,
        "benchmarking_path": benchmarking_path,
        "feature_importance": feature_importance,
        "report_html": report_filename,
        "trained_at": timestamp,
        "training_time": training_time
    }

def load_model_for_inference(user_id="default_user"):
    """Load trained model and validate it's present."""
    user_model_dir = os.path.join(MODEL_DIR, user_id)
    model_files = sorted([f for f in os.listdir(user_model_dir) if f.endswith(".pkl")], reverse=True)
    if not model_files:
        raise FileNotFoundError(f"No trained model found for user '{user_id}'.")

    latest_model = model_files[0]
    model_path = os.path.join(user_model_dir, latest_model)
    return joblib.load(model_path)
