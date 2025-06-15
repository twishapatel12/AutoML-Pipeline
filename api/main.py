from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import tempfile, shutil, os, time, logging, base64, io, json
import joblib
import shap
import matplotlib.pyplot as plt
from typing import List
from dotenv import load_dotenv
from automl.train import train_model, load_model_for_inference

import matplotlib
matplotlib.use("Agg")

# === Load env vars and logging setup ===
load_dotenv()
from logging.handlers import RotatingFileHandler

LOG_PATH = os.getenv("LOG_PATH", "api.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3)
logging.basicConfig(handlers=[handler], level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
model_cache = {}

MODEL_DIR = os.getenv("MODEL_DIR", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

class SingleInput(BaseModel):
    data: dict

@app.get("/admin/users", response_model=List[str])
def list_users():
    try:
        return [
            name for name in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, name))
        ]
    except Exception as e:
        logger.exception("Failed to list users")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/user_artifacts")
def list_user_artifacts(user_id: str = Query(...)):
    user_path = os.path.join(MODEL_DIR, user_id)
    if not os.path.exists(user_path):
        raise HTTPException(status_code=404, detail=f"No artifacts found for user '{user_id}'")

    try:
        return [
            os.path.relpath(os.path.join(root, f), MODEL_DIR)
            for root, _, files in os.walk(user_path) for f in files
        ]
    except Exception as e:
        logger.exception("Failed to list user artifacts")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automl/train")
def automl_train(
    file: UploadFile = File(...),
    target: str = Query(...),
    user_id: str = Query("guest_user"),
    session_id: str = Query(None),
    task_type: str = Query(None)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    session_id = session_id or f"session_{int(time.time())}"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found.")

        result = train_model(df, target, user_id=user_id, task_override=task_type)
        result["report_html"] = os.path.basename(result["report_html"])

        model_cache[user_id] = joblib.load(result["model_path"])
        return {**result, "session_id": session_id}

    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/automl/report")
def get_report(filename: str, user_id: str = Query("default_user")):
    path = os.path.join(MODEL_DIR, user_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="text/html", filename=filename)

@app.get("/automl/benchmarking_report")
def get_benchmarking_report(user_id: str = Query("default_user")):
    benchmarking_path = os.path.join(MODEL_DIR, user_id, "benchmarking_report.json")
    if not os.path.exists(benchmarking_path):
        raise HTTPException(status_code=404, detail="Benchmarking report not found")
    return FileResponse(benchmarking_path, media_type="application/json", filename="benchmarking_report.json")

@app.post("/automl/predict")
def predict(file: UploadFile = File(...), user_id: str = Query("default_user")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        if user_id not in model_cache or model_cache[user_id] is None:
            model_cache[user_id] = load_model_for_inference(user_id=user_id)

        model = model_cache[user_id]
        df = pd.read_csv(file.file)

        if df.empty:
            raise HTTPException(status_code=400, detail="Input CSV is empty or invalid.")

        start_time = time.time()
        preds = model.predict(df)
        inference_time = round(time.time() - start_time, 3)

        result = {
            "predictions": pd.DataFrame({"prediction": preds}).to_dict(orient="records"),
            "inference_time": inference_time
        }

        if hasattr(model.named_steps["model"], "predict_proba"):
            try:
                result["probabilities"] = pd.DataFrame(model.predict_proba(df)).to_dict(orient="records")
            except Exception:
                pass

        return result

    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/automl/predict_single")
def predict_single(input: SingleInput, user_id: str = Query("default_user")):
    try:
        if user_id not in model_cache or model_cache[user_id] is None:
            model_cache[user_id] = load_model_for_inference(user_id=user_id)

        model = model_cache[user_id]
        df = pd.DataFrame([input.data])

        start_time = time.time()
        pred = model.predict(df)[0]
        inference_time = round(time.time() - start_time, 3)

        result = {
            "prediction": str(pred),
            "inference_time": inference_time
        }

        if hasattr(model.named_steps["model"], "predict_proba"):
            try:
                probas = model.predict_proba(df)
                result["probabilities"] = probas.tolist()[0]
            except Exception:
                pass

        try:
            transformed = model.named_steps["preprocessor"].transform(df)
            explainer = shap.Explainer(
                model.named_steps["model"],
                transformed,
                check_additivity=False
            )
            shap_values = explainer(transformed)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            result["shap_plot_url"] = f"data:image/png;base64,{b64}"
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            result["shap_plot_url"] = None

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Single prediction failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
