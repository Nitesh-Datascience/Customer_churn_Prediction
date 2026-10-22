from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# PAGE + THEME
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATHS = [
    ROOT / "data" / "Customer Churn.csv",
    ROOT / "Customer Churn.csv",
    ROOT / "data" / "customer_churn.csv",
]
MODEL_PATHS = [
    ROOT / "models" / "churn_pipeline.joblib",
    ROOT / "churn_pipeline.joblib",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1020; --panel:#111a2e; --line:#273650; --muted:#9aa9c2; --mint:#70e1c1; --blue:#77b8ff; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #172744 0%, #0b1020 42%); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.03em; }
.hero { padding:28px 30px; border:1px solid #2a3b59; border-radius:22px;
        background:linear-gradient(120deg,rgba(25,43,72,.96),rgba(17,24,45,.94) 60%,rgba(32,28,62,.9));
        margin-bottom:18px; box-shadow:0 18px 45px rgba(0,0,0,.16); }
.hero-kicker { color:#70e1c1; font-size:12px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { margin:7px 0 5px; font-size:clamp(28px,3vw,42px); font-weight:700; }
.hero p { color:#aab7cc; margin:0; font-size:15px; }
.section-title { font-family:'Space Grotesk',sans-serif; font-size:23px; font-weight:700; margin:24px 0 5px; }
.section-subtitle { color:#9aa9c2; margin-bottom:13px; font-size:13px; }
div[data-testid="stMetric"] { background:rgba(17,26,46,.92); border:1px solid #2a3954; padding:16px 17px; border-radius:16px; }
div[data-testid="stMetricLabel"] { color:#a9b8ce; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }
div[data-testid="stTabs"] button { border-radius:10px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#293954 !important; border-radius:16px !important; background:rgba(16,25,44,.55); }
.insight { padding:14px 16px; margin:8px 0; border:1px solid #2a3954; border-left:4px solid #70e1c1;
           border-radius:12px; background:rgba(17,26,46,.8); color:#dbe6f5; }
.small-note { color:#9aa9c2; font-size:12px; }
[data-testid="stSidebar"] { background:#0d1527; border-right:1px solid #24324b; }
.stButton>button[kind="primary"] { border:0; background:linear-gradient(90deg,#49c9aa,#6cb8ff); color:#071322; font-weight:800; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#dce6f5"),
    margin=dict(l=12, r=12, t=55, b=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

COLORS = {"Retained": "#70e1c1", "Churned": "#fb7185"}

# ---------------------------------------------------------------------
# DATA + MODEL HELPERS
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_string):
    data = pd.read_csv(path_string)
    data.columns = [str(c).strip() for c in data.columns]
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    if "Churn" not in data.columns:
        raise ValueError("The dataset must contain a 'Churn' column.")
    data["Churn_Label"] = data["Churn"].map({"Yes": "Churned", "No": "Retained"})
    if "tenure" in data.columns:
        data["Tenure Group"] = pd.cut(
            data["tenure"], [-1, 6, 12, 24, 48, 72, 100],
            labels=["0–6", "7–12", "13–24", "25–48", "49–72", "73+"]
        )
    return data
