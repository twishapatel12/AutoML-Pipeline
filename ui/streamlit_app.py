import streamlit as st
import requests
import pandas as pd
import base64
import time
import sqlite3
import uuid
from sklearn.utils.multiclass import type_of_target
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Set up page configuration
st.set_page_config(page_title="AutoML Trainer", layout="wide")
st.title("🤖 AutoML Training Dashboard")

# ========= User ID and Session ID ========= #
if 'user_id' not in st.session_state:
    # Generate a simple user-friendly ID (e.g. "user-<short uuid>")
    st.session_state['user_id'] = "user-" + str(uuid.uuid4())[:8]

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

user_id = st.session_state['user_id']
session_id = st.session_state['session_id']

st.sidebar.markdown(f"**User ID:** `{user_id}`")
st.sidebar.markdown(f"**Session ID:** `{session_id}`")

# ========= SQLite Setup ========= #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this file's directory
DB_DIR = os.path.join(BASE_DIR)
DB_PATH = os.path.join(DB_DIR, "automl.db")

def migrate_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(benchmarking_logs)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'user_id' not in columns:
        cursor.execute("ALTER TABLE benchmarking_logs ADD COLUMN user_id TEXT")
        conn.commit()
    conn.close()

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ensure path exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarking_logs (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_id TEXT,
            model_name TEXT,
            dataset_name TEXT,
            target_column TEXT,
            input_data TEXT,
            prediction TEXT,
            inference_time REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_benchmark(entry):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO benchmarking_logs (id, session_id, user_id, model_name, dataset_name, target_column, input_data, prediction, inference_time, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(uuid.uuid4()), session_id, user_id,
        entry['model_name'], entry['dataset_name'], entry['target_column'],
        entry['input_data'], entry['prediction'], entry['inference_time'], entry['timestamp']
    ))
    conn.commit()
    conn.close()

def load_benchmarks():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM benchmarking_logs WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df

# Run migration before initializing DB to ensure schema is up to date
migrate_db()
init_db()

# ========= TABS ========= #
tabs = st.tabs(["📁 Dataset Upload", "⚙️ Train Model", "🔍 Inference", "📊 Benchmarking"])
df = None
auto_target = None
auto_task = None

# ========= TAB 1: Dataset Upload ========= #
with tabs[0]:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Preview of Dataset")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

# ========= TAB 2: Training ========= #
# ========= TAB 2: Training ========= #
with tabs[1]:
    if df is not None:
        num_unique = df.nunique()
        class_candidates = num_unique[num_unique <= 10]
        numeric_cols = df.select_dtypes(include=["int", "float"]).columns
        corr = df[numeric_cols].corr()

        best_corr_col = corr.sum().sort_values(ascending=False).index[0] if not corr.empty else df.columns[-1]
        auto_target = class_candidates.index[0] if not class_candidates.empty else best_corr_col

        st.success(f"Auto-detected target column: **{auto_target}**")
        use_auto_target = st.checkbox("Use auto-detected target column", value=True)
        target_column = auto_target if use_auto_target else st.selectbox("Select Target Column", df.columns)

        try:
            task_type_raw = type_of_target(df[target_column])
            auto_task = "regression" if "continuous" in task_type_raw else "classification"
        except:
            auto_task = "classification"

        st.success(f"Auto-detected task type: **{auto_task}**")
        use_auto_task = st.checkbox("Use auto-detected task type", value=True)
        task_type = auto_task if use_auto_task else st.selectbox("Select Task Type", ["classification", "regression"])

        if st.button("🚀 Train Model"):
            if not uploaded_file:
                st.error("Please upload a dataset first.")
            else:
                with st.spinner("Training..."):
                    uploaded_file.seek(0)
                    params = {
                        "target": target_column,
                        "task_type": task_type,
                        "user_id": user_id
                    }

                    response = requests.post(
                        "http://localhost:8000/automl/train",
                        params=params,
                        files={"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state['model_result'] = result
                    st.session_state['target_column'] = target_column
                    st.session_state['df'] = df

                    st.toast("✅ Model trained successfully!", icon="✅")
                    st.subheader("Model Info")
                    st.write(f"**Best Model:** {result['best_model_type']}")
                    st.json(result["metrics"])

                    if result.get("feature_importance"):
                        st.subheader("Feature Importance")
                        fi_df = pd.DataFrame(result["feature_importance"])
                        st.dataframe(fi_df)

                    report_url = f"http://localhost:8000/automl/report?filename={user_id}/{result['report_html']}"
                    st.markdown(
                        f"""<a href="{report_url}" target="_blank" style="padding:8px 15px; background-color:#4CAF50; color:white; text-decoration:none; border-radius:5px;">📄 Download Report</a>""",
                        unsafe_allow_html=True
                    )

                    model_binary = base64.b64decode(result["model_binary"])
                    b64 = base64.b64encode(model_binary).decode()
                    download_link = f'data:application/octet-stream;base64,{b64}'
                    st.markdown(
                        f"""<a href="{download_link}" download="model.pkl" style="padding:8px 15px; background-color:#4CAF50; color:white; text-decoration:none; border-radius:5px;">⬇️ Download Model</a>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"❌ Training failed: {response.text}")

# ========= TAB 3: Inference ========= #
with tabs[2]:
    if 'model_result' in st.session_state:
        result = st.session_state['model_result']
        df = st.session_state['df']
        target_column = st.session_state['target_column']
        mode = st.radio("Select inference mode", ["CSV Upload", "Manual Input (Single Row)"])

        if mode == "CSV Upload":
            inference_file = st.file_uploader("Upload CSV for prediction (no target column)", type=["csv"], key="inference")
            if inference_file and st.button("Run Inference"):
                inference_file.seek(0)
                start = time.time()
                response = requests.post(
                    "http://localhost:8000/automl/predict",
                    params={"user_id": user_id},
                    files={"file": (inference_file.name, inference_file)}
                )
                end = time.time()

                if response.status_code == 200:
                    result_pred = response.json()
                    preds = pd.DataFrame(result_pred["predictions"])
                    inference_time = result_pred["inference_time"]

                    st.toast("✅ Inference complete!", icon="🧠")
                    st.dataframe(preds)

                    pred_counts = preds[preds.columns[0]].value_counts()
                    st.bar_chart(pred_counts)

                    # Save to DB
                    first_pred_val = preds.iloc[0, 0] if not preds.empty else "N/A"
                    log_benchmark({
                        'model_name': result['best_model_type'],
                        'dataset_name': inference_file.name,
                        'target_column': target_column,
                        'input_data': 'BULK_CSV',
                        'prediction': str(first_pred_val),
                        'inference_time': inference_time,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                    csv = preds.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error(f"Inference failed: {response.text}")

        else:  # Manual Input
            input_df = pd.DataFrame(columns=df.drop(columns=[target_column]).columns)
            input_data = {}
            for col in input_df.columns:
                if df[col].dtype == object:
                    input_data[col] = st.text_input(f"{col}", "")
                else:
                    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()))

            if st.button("Predict Row"):
                start = time.time()
                response = requests.post(
                    "http://localhost:8000/automl/predict_single",
                    params={"user_id": user_id},
                    json={"data": input_data}
                )
                end = time.time()

                if response.status_code == 200:
                    resp = response.json()
                    pred = resp["prediction"]
                    probs = resp.get("probabilities")
                    shap_url = resp.get("shap_plot_url")
                    inference_time = resp["inference_time"]

                    st.toast("✅ Prediction complete!", icon="🤖")
                    st.success(f"Prediction: **{pred}** in {inference_time:.4f}s")
                    if probs:
                        st.json(probs)
                    if shap_url:
                        st.image(shap_url)

                    # Save to DB
                    log_benchmark({
                        'model_name': result['best_model_type'],
                        'dataset_name': 'single_input',
                        'target_column': target_column,
                        'input_data': str(input_data),
                        'prediction': str(pred),
                        'inference_time': inference_time,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                else:
                    st.error("Prediction failed.")

# ========= TAB 4: Benchmarking ========= #
with tabs[3]:
    st.subheader("📊 Benchmarking Results")
    df_benchmark = load_benchmarks()
    if not df_benchmark.empty:
        with st.expander("🔍 Filters"):
            model_filter = st.multiselect("Filter by Model", options=df_benchmark['model_name'].unique())
            dataset_filter = st.multiselect("Filter by Dataset", options=df_benchmark['dataset_name'].unique())

        filtered = df_benchmark.copy()
        if model_filter:
            filtered = filtered[filtered['model_name'].isin(model_filter)]
        if dataset_filter:
            filtered = filtered[filtered['dataset_name'].isin(dataset_filter)]

        def color_inference(val):
            if isinstance(val, float):
                return 'color: green' if val < 0.1 else 'color: red'
            return ''

        st.dataframe(filtered.style.applymap(color_inference, subset=['inference_time']), use_container_width=True)

        st.subheader("⏱ Inference Time by Model")
        fig, ax = plt.subplots()
        filtered.groupby("model_name")["inference_time"].mean().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.subheader("📈 Predictions Distribution")
        fig2, ax2 = plt.subplots()
        filtered['prediction'].value_counts().plot(kind='bar', ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("No benchmark entries yet.")