import os
import logging
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import ConfusionMatrixDisplay
from datetime import datetime
import plotly.graph_objects as go
from jinja2 import Template
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR", os.path.abspath("./saved_models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# === Set Up Logging ===
logger = logging.getLogger(__name__)

def detect_task(y):
    """Detect machine learning task."""
    task_type = type_of_target(y)
    logger.info(f"Detected task type: {task_type}")
    return "regression" if "continuous" in task_type else "classification"

def build_pipeline(model, X):
    """Construct a pipeline with preprocessing."""
    num_features = X.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def generate_diagnostic_plots(y_true, y_pred, task, save_dir=MODEL_DIR):
    """Generate confusion or residual plots."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "diagnostic_plot.png")
    if task == "classification":
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        disp.figure_.savefig(path)
    else:
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Residual Plot")
        plt.savefig(path)
    plt.close()
    logger.info(f"Saved diagnostic plot to {path}")
    return path

def generate_shap_summary(pipeline, X, save_dir=MODEL_DIR):
    """Generate SHAP summary plot if supported."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)

        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)

        shap_path = os.path.join(save_dir, "shap_summary.png")
        shap.summary_plot(shap_values, X_transformed, show=False)
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot to {shap_path}")
        return shap_path
    except Exception as e:
        logger.warning(f"SHAP plot generation failed: {e}")
        return None

def generate_report(model_name, task, target_column, metrics, feature_importance=None,
                    confusion_matrix_plot=None, shap_plot_path=None, save_dir=MODEL_DIR):
    """Render an HTML report from training artifacts."""
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chart_html = ""

    if feature_importance:
        features = [f["feature"] for f in feature_importance]
        importances = [f["importance"] for f in feature_importance]
        fig = go.Figure([go.Bar(x=features, y=importances)])
        fig.update_layout(title="Feature Importance", xaxis_title="Feature", yaxis_title="Importance")
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Training Report</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            margin: 40px;
            background: #f9f9fb;
            color: #333;
        }

        .container {
            max-width: 960px;
            margin: auto;
            padding: 30px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }

        h1, h2 {
            color: #2c3e50;
            border-bottom: 2px solid #dcdde1;
            padding-bottom: 5px;
        }

        p, li {
            font-size: 16px;
            line-height: 1.6;
        }

        .info-box {
            background: #ecf0f1;
            padding: 15px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
            border-radius: 5px;
        }

        .metrics-list {
            background: #fefefe;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            list-style-type: none;
        }

        .metrics-list li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }

        img, iframe {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        }

    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Training Report</h1>

        <div class="info-box">
            <p><strong>Generated on:</strong> {{ now }}</p>
            <p><strong>Model:</strong> {{ model_name }}</p>
            <p><strong>Task:</strong> {{ task }}</p>
            <p><strong>Target Column:</strong> {{ target_column }}</p>
        </div>

        <h2>✅ Metrics</h2>
        <ul class="metrics-list">
            {% for k, v in metrics.items() %}
                <li><strong>{{ k }}:</strong> {{ "%.4f"|format(v) }}</li>
            {% endfor %}
        </ul>

        {% if chart_html %}
            <h2>📈 Feature Importance</h2>
            {{ chart_html | safe }}
        {% endif %}

        {% if confusion_matrix_plot %}
            <h2>📊 Confusion Matrix / Residuals</h2>
            <img src="{{ confusion_matrix_plot }}">
        {% endif %}

        {% if shap_plot_path %}
            <h2>🔍 SHAP Plot</h2>
            <img src="{{ shap_plot_path }}">
        {% endif %}
    </div>
</body>
</html>
""")
    html = template.render(
        now=now,
        model_name=model_name,
        task=task,
        target_column=target_column,
        metrics=metrics,
        chart_html=chart_html,
        confusion_matrix_plot=confusion_matrix_plot,
        shap_plot_path=shap_plot_path
    )

    report_path = os.path.join(save_dir, "report.html")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html)
    logger.info(f"Saved training report to {report_path}")
    return "report.html"

def cleanup_old_files(base_path=MODEL_DIR, days=7):
    """Clean up old files older than X days."""
    now = time.time()
    cutoff = now - days * 86400
    deleted = []

    for root, dirs, files in os.walk(base_path):
        for name in files:
            filepath = os.path.join(root, name)
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                deleted.append(filepath)

    logger.info(f"Cleanup completed. Deleted {len(deleted)} files.")
    return deleted