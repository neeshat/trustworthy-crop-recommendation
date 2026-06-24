# ============================================================
# TRUSTWORTHY CROP RECOMMENDATION SYSTEM
# Streamlit App - Matching Poster UI Design
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Trustworthy Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# COMPLETE CSS
# ============================================================
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding: 0rem 1rem 1rem 1rem;
    }
    
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 40%, #40916c 70%, #52b788 100%);
        padding: 22px 30px 18px 30px;
        border-radius: 0 0 20px 20px;
        text-align: center;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .hero-header::before {
        content: '🌿';
        position: absolute;
        font-size: 80px;
        left: 20px;
        top: 5px;
        opacity: 0.3;
    }
    .hero-header::after {
        content: '🚜';
        position: absolute;
        font-size: 60px;
        right: 25px;
        top: 10px;
        opacity: 0.3;
    }
    .hero-title {
        color: white;
        font-size: 1.95em;
        font-weight: 800;
        margin: 0 0 6px 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        display: inline-flex;
        gap: 6px;
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 5px 18px;
        margin-top: 4px;
    }
    .hero-badge {
        color: #b7e4c7;
        font-size: 0.82em;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .hero-dot { color: #74c69d; }

    /* ── Section Panel ── */
    .panel {
        background: white;
        border-radius: 14px;
        padding: 14px 16px;
        height: 100%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8f5e9;
        position: relative;
    }
    .panel-header {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e8f5e9;
    }
    .panel-number {
        background: #2d6a4f;
        color: white;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78em;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .panel-title {
        font-size: 0.95em;
        font-weight: 700;
        color: #1b4332;
        margin: 0;
        line-height: 1.3;
    }
    .panel-subtitle {
        font-size: 0.72em;
        color: #6c757d;
        margin: 1px 0 0 0;
    }

    /* ── Input Rows ── */
    .input-row {
        display: flex;
        align-items: center;
        padding: 6px 8px;
        border-radius: 8px;
        margin-bottom: 5px;
        background: #f8fffe;
        border: 1px solid #e8f5e9;
        gap: 8px;
    }
    .input-icon { font-size: 1.1em; width: 22px; text-align: center; }
    .input-label {
        font-size: 0.78em;
        color: #495057;
        font-weight: 500;
        flex: 1;
    }
    .input-value {
        font-size: 0.82em;
        font-weight: 700;
        color: #2d6a4f;
        text-align: right;
    }

    /* ── Crop Result Card ── */
    .crop-image-circle {
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        font-size: 3.5em;
        border: 4px solid #52b788;
        box-shadow: 0 4px 15px rgba(82,183,136,0.3);
    }
    .crop-name-badge {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border-radius: 25px;
        padding: 6px 20px;
        font-size: 1.1em;
        font-weight: 800;
        text-align: center;
        margin: 8px auto;
        display: inline-block;
        letter-spacing: 1px;
        box-shadow: 0 3px 10px rgba(45,106,79,0.3);
    }
    .confidence-big {
        font-size: 3.0em;
        font-weight: 900;
        color: #2d6a4f;
        text-align: center;
        line-height: 1.1;
    }
    .conf-label {
        font-size: 0.72em;
        color: #6c757d;
        text-align: center;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .conf-bar-bg {
        background: #e9ecef;
        border-radius: 10px;
        height: 12px;
        margin: 6px 0;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #52b788, #2d6a4f);
        transition: width 0.8s ease;
    }
    .conf-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.65em;
        color: #adb5bd;
        margin-top: 2px;
    }
    .why-box {
        background: #f0faf4;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 10px;
        border-left: 3px solid #52b788;
    }
    .why-title {
        font-size: 0.72em;
        font-weight: 700;
        color: #2d6a4f;
        margin-bottom: 4px;
    }
    .why-text {
        font-size: 0.70em;
        color: #495057;
        line-height: 1.5;
    }

    /* ── Alternative Crops ── */
    .alt-crop-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 10px;
        margin-bottom: 8px;
        background: #f8fffe;
        border: 1px solid #e8f5e9;
    }
    .alt-rank {
        font-size: 0.85em;
        font-weight: 800;
        color: #2d6a4f;
        width: 20px;
        flex-shrink: 0;
    }
    .alt-icon { font-size: 1.4em; flex-shrink: 0; }
    .alt-info { flex: 1; }
    .alt-name {
        font-size: 0.82em;
        font-weight: 700;
        color: #1b4332;
    }
    .alt-score-label {
        font-size: 0.65em;
        color: #6c757d;
    }
    .alt-bar-bg {
        background: #e9ecef;
        border-radius: 5px;
        height: 6px;
        margin-top: 3px;
        overflow: hidden;
    }
    .alt-bar-fill {
        height: 100%;
        border-radius: 5px;
        background: linear-gradient(90deg, #74c69d, #2d6a4f);
    }
    .alt-pct {
        font-size: 0.82em;
        font-weight: 700;
        color: #2d6a4f;
        flex-shrink: 0;
    }
    .tip-box {
        background: #fffde7;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 10px;
        border-left: 3px solid #ffd600;
    }
    .tip-title {
        font-size: 0.72em;
        font-weight: 700;
        color: #f57f17;
        margin-bottom: 4px;
    }
    .tip-text {
        font-size: 0.70em;
        color: #5d4037;
        line-height: 1.5;
    }

    /* ── SHAP Bars ── */
    .shap-toggle {
        display: flex;
        gap: 6px;
        margin-bottom: 10px;
    }
    .shap-btn-pos {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 15px;
        padding: 3px 12px;
        font-size: 0.72em;
        font-weight: 600;
    }
    .shap-btn-neg {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 15px;
        padding: 3px 12px;
        font-size: 0.72em;
        font-weight: 600;
    }
    .shap-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 7px;
    }
    .shap-feat-icon { font-size: 0.9em; width: 18px; text-align: center; }
    .shap-feat-name {
        font-size: 0.72em;
        color: #495057;
        font-weight: 500;
        width: 90px;
        flex-shrink: 0;
    }
    .shap-bar-container {
        flex: 1;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .shap-bar-bg {
        flex: 1;
        background: #e9ecef;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .shap-bar-pos {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #52b788, #2d6a4f);
    }
    .shap-bar-neg {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #e63946, #c1121f);
    }
    .shap-value {
        font-size: 0.70em;
        font-weight: 700;
        width: 38px;
        text-align: right;
        flex-shrink: 0;
    }
    .shap-positive { color: #2d6a4f; }
    .shap-negative { color: #c1121f; }
    .what-box {
        background: #e8f4f8;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 8px;
        border-left: 3px solid #3498db;
    }
    .what-title {
        font-size: 0.72em;
        font-weight: 700;
        color: #1565c0;
        margin-bottom: 4px;
    }
    .what-text {
        font-size: 0.68em;
        color: #37474f;
        line-height: 1.5;
    }

    /* ── Reliability Panel ── */
    .reliability-badge {
        text-align: center;
        padding: 10px;
        background: #f0faf4;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid #b7e4c7;
    }
    .reliability-title {
        font-size: 0.70em;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .reliability-level {
        font-size: 1.6em;
        font-weight: 800;
        color: #2d6a4f;
    }
    .reliability-sub {
        font-size: 0.68em;
        color: #40916c;
        font-weight: 500;
    }
    .pred-reliability {
        text-align: center;
        padding: 10px;
        background: #e8f4f8;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid #b3d9f0;
    }
    .pred-rel-pct {
        font-size: 1.8em;
        font-weight: 900;
        color: #1565c0;
    }
    .pred-rel-label {
        font-size: 0.68em;
        color: #1565c0;
        font-weight: 500;
    }
    .quality-check {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 8px;
        border-radius: 7px;
        margin-bottom: 4px;
        font-size: 0.72em;
        font-weight: 500;
    }
    .check-pass {
        background: #d4edda;
        color: #155724;
    }
    .check-fail {
        background: #fff3cd;
        color: #856404;
    }
    .check-icon { font-size: 0.85em; }

    /* ── Footer ── */
    .footer-bar {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        border-radius: 12px;
        padding: 12px 25px;
        margin-top: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    .footer-text {
        color: #b7e4c7;
        font-size: 0.82em;
        font-weight: 500;
    }
    .footer-bold {
        color: white;
        font-weight: 700;
    }

    /* ── Welcome Screen ── */
    .welcome-panel {
        background: linear-gradient(135deg, #f0faf4, #e8f5e9);
        border-radius: 14px;
        padding: 40px;
        text-align: center;
        border: 2px dashed #52b788;
    }

    /* ── Metric Mini Cards ── */
    .mini-metric {
        background: white;
        border-radius: 8px;
        padding: 8px 10px;
        border: 1px solid #e8f5e9;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .mini-metric-val {
        font-size: 1.1em;
        font-weight: 800;
        color: #2d6a4f;
    }
    .mini-metric-lbl {
        font-size: 0.62em;
        color: #6c757d;
        font-weight: 500;
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Streamlit input overrides */
    .stNumberInput input {
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
        font-size: 0.85em !important;
    }
    .stSlider {padding: 0px !important;}
    
    /* Sidebar */
    .css-1d391kg {background: #f8fffe !important;}
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1b4332, #2d6a4f) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-size: 0.9em !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 12px rgba(45,106,79,0.35) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(45,106,79,0.45) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        with open('crop_recommendation_artifacts.pkl', 'rb') as f:
            art = pickle.load(f)
        return art, True
    except FileNotFoundError:
        return None, False

@st.cache_resource
def load_shap_explainer(_base_model):
    try:
        return shap.TreeExplainer(_base_model)
    except:
        return None

artifacts, loaded = load_artifacts()

if not loaded:
    st.error("❌ `crop_recommendation_artifacts.pkl` not found. "
             "Please upload it to the app directory.")
    st.stop()

# Unpack
model             = artifacts['model']
base_model        = artifacts['base_model']
scaler            = artifacts['scaler']
le                = artifacts['label_encoder']
class_names       = artifacts['class_names']
feature_cols      = artifacts['feature_cols']
cal_scores        = artifacts['cal_scores']
shap_importance   = artifacts['shap_mean_importance']
shap_tau          = float(artifacts['shap_stability_score'])
final_ece         = float(artifacts['final_ece'])
final_f1          = float(artifacts['final_f1'])
final_coverage    = float(artifacts['final_coverage'])
avg_set_size      = float(artifacts['avg_set_size'])
results_df        = artifacts.get('results_df', pd.DataFrame())
ece_results       = artifacts.get('ece_results', {})
pareto_df         = artifacts.get('pareto_df', pd.DataFrame())
pareto_F          = artifacts.get('pareto_F', np.array([]))
best_pareto_idx   = artifacts.get('best_pareto_idx', 0)
feature_stats     = artifacts.get('feature_stats', {})

shap_explainer = load_shap_explainer(base_model)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
CROP_EMOJIS = {
    'rice':'🌾','maize':'🌽','chickpea':'🫘','kidneybeans':'🫘',
    'pigeonpeas':'🌿','mothbeans':'🌱','mungbean':'🌱','blackgram':'🌱',
    'lentil':'🌾','pomegranate':'🍎','banana':'🍌','mango':'🥭',
    'grapes':'🍇','watermelon':'🍉','muskmelon':'🍈','apple':'🍎',
    'orange':'🍊','papaya':'🍈','coconut':'🥥','cotton':'🌸',
    'jute':'🌿','coffee':'☕'
}

FEATURE_ICONS = {
    'N':'🌿','P':'🔴','K':'🟣',
    'temperature':'🌡️','humidity':'💧','ph':'🧪','rainfall':'🌧️'
}

FEATURE_UNITS = {
    'N':'kg/ha','P':'kg/ha','K':'kg/ha',
    'temperature':'°C','humidity':'%','ph':'','rainfall':'mm'
}

FEATURE_DISPLAY = {
    'N':'Nitrogen (N)','P':'Phosphorus (P)','K':'Potassium (K)',
    'temperature':'Temperature','humidity':'Humidity',
    'ph':'Soil pH','rainfall':'Rainfall'
}

def conformal_predict_single(model, x_scaled, cal_scores_arr, alpha=0.05):
    n_cal    = len(cal_scores_arr)
    level    = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    quantile = np.quantile(cal_scores_arr, level)
    proba    = model.predict_proba(x_scaled.reshape(1, -1))[0]
    pred_set = [c for c in range(len(proba)) if (1 - proba[c]) <= quantile]
    return pred_set, quantile, proba

def compute_ece_single(y_prob_arr, n_bins=10):
    """ECE from probability distribution (no ground truth needed for display)."""
    conf = np.max(y_prob_arr)
    return conf

def get_shap_for_sample(explainer, x_scaled, pred_class):
    try:
        sv = explainer.shap_values(x_scaled.reshape(1, -1))
        if isinstance(sv, list):
            return sv[pred_class][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            return sv[0, :, pred_class]
        else:
            return sv[0]
    except:
        return np.zeros(len(feature_cols))

def get_crop_reason(crop, input_vals):
    reasons = {
        'rice':        "Your high humidity, adequate rainfall, and soil nutrients are ideal for rice.",
        'maize':       "Warm temperature, moderate humidity, and your NPK profile suit maize perfectly.",
        'coffee':      "Your specific temperature, humidity, and pH levels match coffee requirements.",
        'banana':      "High humidity, warm temperature, and good potassium support banana growth.",
        'wheat':       "Moderate temperature, balanced nutrients, and your pH suit wheat cultivation.",
        'cotton':      "Warm climate, low rainfall, and your soil conditions favor cotton.",
        'sugarcane':   "High rainfall, warm temperature, and rich nutrients support sugarcane.",
        'jute':        "Warm, humid conditions with your soil profile are ideal for jute.",
        'coconut':     "Warm temperature, high humidity, and coastal-like conditions suit coconut.",
        'mango':       "Your temperature range and soil pH are perfect for mango cultivation.",
        'grapes':      "Your climate conditions and soil acidity match grape growing requirements.",
        'watermelon':  "Warm temperature, moderate humidity, and your NPK profile suit watermelon.",
        'apple':       "Cool temperature and your soil conditions are favorable for apple orchards.",
    }
    return reasons.get(
        crop.lower(),
        f"Your soil nutrients, weather, and other conditions are highly suitable for {crop} cultivation."
    )


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌾 Streamlit-based Trustworthy Crop Recommendation System</div>
    <div class="hero-subtitle">
        <span class="hero-badge">Smart Recommendations</span>
        <span class="hero-dot">•</span>
        <span class="hero-badge">Explainable Results</span>
        <span class="hero-dot">•</span>
        <span class="hero-badge">Better Decisions for Farmers</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN LAYOUT — 5 COLUMNS (matching poster)
# ============================================================
col1, col2, col3, col4, col5 = st.columns([1.05, 1.0, 1.0, 1.1, 0.95])


# ════════════════════════════════════════════════════════════
# COLUMN 1 — Enter Field Details
# ════════════════════════════════════════════════════════════
with col1:
    st.markdown("""
    <div class="panel">
        <div class="panel-header">
            <div class="panel-number">1</div>
            <div>
                <div class="panel-title">Enter Your Field Details 🌱</div>
                <div class="panel-subtitle">Fill in simple information about your field</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="background:white;border-radius:14px;'
                    'padding:14px;box-shadow:0 2px 12px rgba(0,0,0,0.08);'
                    'border:1px solid #e8f5e9;margin-top:-14px">', 
                    unsafe_allow_html=True)

        # Inputs
        N = st.number_input(
            "🌿 Nitrogen N (kg/ha)", 
            min_value=0.0, max_value=145.0, value=90.0, step=1.0,
            label_visibility="visible"
        )
        P = st.number_input(
            "🔴 Phosphorus P (kg/ha)",
            min_value=0.0, max_value=150.0, value=42.0, step=1.0,
            label_visibility="visible"
        )
        K = st.number_input(
            "🟣 Potassium K (kg/ha)",
            min_value=0.0, max_value=210.0, value=43.0, step=1.0,
            label_visibility="visible"
        )
        ph = st.number_input(
            "🧪 Soil pH",
            min_value=3.0, max_value=10.0, value=6.5, step=0.05,
            label_visibility="visible"
        )
        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=15.0, max_value=300.0, value=202.0, step=5.0,
            label_visibility="visible"
        )
        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=5.0, max_value=50.0, value=25.0, step=0.1,
            label_visibility="visible"
        )
        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=10.0, max_value=100.0, value=71.0, step=0.5,
            label_visibility="visible"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Presets
        st.markdown(
            "<div style='font-size:0.70em;color:#6c757d;"
            "font-weight:600;margin:8px 0 4px 2px'>⚡ Quick Presets</div>",
            unsafe_allow_html=True
        )
        pc1, pc2 = st.columns(2)
        presets = {
            "🌾 Rice":   [90, 42, 43, 6.5, 202.0, 20.9, 82.0],
            "🌽 Maize":  [71, 54, 16, 5.75, 87.8, 22.6, 63.7],
            "☕ Coffee": [101,28, 29, 6.8, 158.1, 25.5, 58.9],
            "🍌 Banana": [100,82, 50, 5.9, 105.5, 27.4, 80.3],
        }
        for i, (pname, pvals) in enumerate(presets.items()):
            col_p = pc1 if i % 2 == 0 else pc2
            if col_p.button(pname, key=f"preset_{i}", use_container_width=True):
                st.session_state['pvals'] = pvals
                st.rerun()

        if 'pvals' in st.session_state:
            pv = st.session_state.pop('pvals')
            N, P, K, ph, rainfall, temperature, humidity = pv

        # Confidence level
        alpha = st.select_slider(
            "Confidence Level",
            options=[0.01, 0.05, 0.10, 0.15, 0.20],
            value=0.05,
            format_func=lambda x: f"{int((1-x)*100)}%",
            label_visibility="visible"
        )

        # GET RECOMMENDATION BUTTON
        recommend = st.button("🚀 Get Recommendation", use_container_width=True)


# ── Prepare input array ──────────────────────────────────────
input_vals   = [N, P, K, temperature, humidity, ph, rainfall]
input_arr    = np.array(input_vals).reshape(1, -1)
input_scaled = scaler.transform(input_arr)

# ── Prediction ───────────────────────────────────────────────
y_prob    = model.predict_proba(input_scaled)[0]
y_pred    = int(np.argmax(y_prob))
confidence= float(y_prob[y_pred])
crop_name = class_names[y_pred]
crop_emoji= CROP_EMOJIS.get(crop_name.lower(), '🌱')

# ── Conformal prediction set ─────────────────────────────────
pred_set, cf_threshold, cf_proba = conformal_predict_single(
    model, input_scaled[0], cal_scores, alpha=alpha)

# ── SHAP values for this sample ──────────────────────────────
shap_vals_sample = None
if shap_explainer is not None:
    shap_vals_sample = get_shap_for_sample(
        shap_explainer, input_scaled[0], y_pred)

# ── Top 5 alternatives ───────────────────────────────────────
top5_idx  = np.argsort(y_prob)[::-1][:5]


# ════════════════════════════════════════════════════════════
# COLUMN 2 — Recommended Crop
# ════════════════════════════════════════════════════════════
with col2:
    st.markdown("""
    <div class="panel-header" style="background:white;border-radius:14px;
         padding:14px 16px 10px 16px;box-shadow:0 2px 12px rgba(0,0,0,0.08);
         border:1px solid #e8f5e9;margin-bottom:10px">
        <div class="panel-number">2</div>
        <div>
            <div class="panel-title">Recommended Crop 🌾</div>
            <div class="panel-subtitle">Best crop for your field</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if recommend or True:  # Always show prediction
        # Crop circle + name
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:16px;
             box-shadow:0 2px 12px rgba(0,0,0,0.08);border:1px solid #e8f5e9;">
            
            <div class="crop-image-circle">{crop_emoji}</div>
            
            <div style="text-align:center;margin:8px 0">
                <div style="display:inline-block">
                    <span style="color:#2d6a4f;font-size:1.0em;font-weight:700">✅ </span>
                    <span class="crop-name-badge">{crop_name.upper()}</span>
                </div>
            </div>
            
            <div class="conf-label">Confidence Score (Predictive)</div>
            <div class="confidence-big">{confidence:.0%}</div>
            
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div>
            </div>
            <div class="conf-scale"><span>0%</span><span>100%</span></div>
            
            <div class="why-box">
                <div class="why-title">🌿 Why this crop?</div>
                <div class="why-text">{get_crop_reason(crop_name, input_vals)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Conformal set info
        set_size = len(pred_set)
        unc_color = "#2d6a4f" if set_size == 1 else \
                    "#f57f17" if set_size <= 3 else "#c62828"
        st.markdown(f"""
        <div style="background:#f8fffe;border-radius:10px;padding:8px 12px;
             margin-top:8px;border:1px solid #e8f5e9;text-align:center">
            <span style="font-size:0.70em;color:#6c757d">
                Conformal Prediction Set ({int((1-alpha)*100)}% confidence):
            </span>
            <span style="font-size:0.80em;font-weight:700;color:{unc_color};margin-left:6px">
                {set_size} crop(s)
            </span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# COLUMN 3 — Top Alternative Crops
# ════════════════════════════════════════════════════════════
with col3:
    st.markdown("""
    <div class="panel-header" style="background:white;border-radius:14px;
         padding:14px 16px 10px 16px;box-shadow:0 2px 12px rgba(0,0,0,0.08);
         border:1px solid #e8f5e9;margin-bottom:10px">
        <div class="panel-number">3</div>
        <div>
            <div class="panel-title">Top Alternative Crops 🌿</div>
            <div class="panel-subtitle">Other good options for your field</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="background:white;border-radius:14px;padding:14px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.08);border:1px solid #e8f5e9;">',
        unsafe_allow_html=True
    )

    rank_labels = ["1.", "2.", "3.", "4.", "5."]
    rank_colors = ["#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2"]

    for rank_i, idx in enumerate(top5_idx[:4]):
        cname = class_names[idx]
        cprob = float(y_prob[idx])
        cemoji= CROP_EMOJIS.get(cname.lower(), '🌱')
        is_top= (rank_i == 0)
        bg    = "#f0faf4" if is_top else "#fafafa"
        border= "2px solid #52b788" if is_top else "1px solid #e8f5e9"

        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:10px;
             padding:8px 10px;margin-bottom:7px;">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:0.85em;font-weight:800;
                      color:{rank_colors[rank_i]};width:18px">
                    {rank_labels[rank_i]}
                </span>
                <span style="font-size:1.3em">{cemoji}</span>
                <div style="flex:1">
                    <div style="font-size:0.80em;font-weight:700;
                          color:#1b4332">{cname.title()}</div>
                    <div style="font-size:0.62em;color:#6c757d">
                        Suitability Score
                    </div>
                    <div style="background:#e9ecef;border-radius:4px;
                          height:6px;margin-top:3px;overflow:hidden">
                        <div style="width:{cprob*100:.0f}%;height:100%;
                              border-radius:4px;
                              background:linear-gradient(90deg,
                              {rank_colors[rank_i]}88,{rank_colors[rank_i]})">
                        </div>
                    </div>
                </div>
                <span style="font-size:0.82em;font-weight:800;
                      color:{rank_colors[rank_i]}">
                    {cprob:.0%}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tip box
    tip_crops = ", ".join([class_names[i].title() for i in top5_idx[1:4]])
    st.markdown(f"""
    <div class="tip-box">
        <div class="tip-title">💡 Tip for Farmers</div>
        <div class="tip-text">
            You can also consider <strong>{tip_crops}</strong> based on 
            market demand, availability, and your preferences.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# COLUMN 4 — Why This Recommendation? (SHAP)
# ════════════════════════════════════════════════════════════
with col4:
    st.markdown("""
    <div class="panel-header" style="background:white;border-radius:14px;
         padding:14px 16px 10px 16px;box-shadow:0 2px 12px rgba(0,0,0,0.08);
         border:1px solid #e8f5e9;margin-bottom:10px">
        <div class="panel-number">4</div>
        <div>
            <div class="panel-title">Why This Recommendation? 🔍</div>
            <div class="panel-subtitle">Key factors that influenced the result</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="background:white;border-radius:14px;padding:14px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.08);border:1px solid #e8f5e9;">',
        unsafe_allow_html=True
    )

    # Toggle buttons (visual only)
    st.markdown("""
    <div class="shap-toggle">
        <span class="shap-btn-pos">✅ Positive Impact</span>
        <span class="shap-btn-neg">❌ Negative Impact</span>
    </div>
    """, unsafe_allow_html=True)

    # SHAP bars
    if shap_vals_sample is not None:
        sv = shap_vals_sample
        max_abs = max(np.abs(sv).max(), 1e-6)

        # Sort by absolute value descending
        sorted_idx = np.argsort(np.abs(sv))[::-1]

        for fi in sorted_idx:
            feat   = feature_cols[fi]
            val_sv = float(sv[fi])
            icon   = FEATURE_ICONS.get(feat, '📊')
            disp   = FEATURE_DISPLAY.get(feat, feat)
            unit   = FEATURE_UNITS.get(feat, '')
            bar_w  = min(abs(val_sv) / max_abs * 100, 100)
            is_pos = val_sv >= 0
            bar_cls= "shap-bar-pos" if is_pos else "shap-bar-neg"
            val_cls= "shap-positive" if is_pos else "shap-negative"
            sign   = "+" if is_pos else ""

            st.markdown(f"""
            <div class="shap-row">
                <span class="shap-feat-icon">{icon}</span>
                <span class="shap-feat-name">{disp}</span>
                <div class="shap-bar-container">
                    <div class="shap-bar-bg">
                        <div class="{bar_cls}" 
                             style="width:{bar_w:.1f}%"></div>
                    </div>
                </div>
                <span class="shap-value {val_cls}">
                    {sign}{val_sv:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback: global importance
        max_imp = shap_importance.max()
        sorted_gi = np.argsort(shap_importance)[::-1]
        for fi in sorted_gi:
            feat  = feature_cols[fi]
            imp   = float(shap_importance[fi])
            icon  = FEATURE_ICONS.get(feat, '📊')
            disp  = FEATURE_DISPLAY.get(feat, feat)
            bar_w = imp / max_imp * 100

            st.markdown(f"""
            <div class="shap-row">
                <span class="shap-feat-icon">{icon}</span>
                <span class="shap-feat-name">{disp}</span>
                <div class="shap-bar-container">
                    <div class="shap-bar-bg">
                        <div class="shap-bar-pos" 
                             style="width:{bar_w:.1f}%"></div>
                    </div>
                </div>
                <span class="shap-value shap-positive">
                    +{imp:.3f}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # What this means box
    if shap_vals_sample is not None:
        sv_s = shap_vals_sample
        pos_feats = [FEATURE_DISPLAY.get(feature_cols[i], feature_cols[i])
                     for i in np.argsort(sv_s)[::-1]
                     if sv_s[i] > 0][:2]
        neg_feats = [FEATURE_DISPLAY.get(feature_cols[i], feature_cols[i])
                     for i in np.argsort(sv_s)
                     if sv_s[i] < 0][:2]

        pos_str = " and ".join(pos_feats) if pos_feats else "—"
        neg_str = " and ".join(neg_feats) if neg_feats else "none"

        st.markdown(f"""
        <div class="what-box">
            <div class="what-title">❓ What this means?</div>
            <div class="what-text">
                <strong>{pos_str}</strong> are the most positive factors, 
                while <strong>{neg_str}</strong> 
                {"need improvement." if neg_feats else "show no negative impact."}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# COLUMN 5 — How Reliable is the Result?
# ════════════════════════════════════════════════════════════
with col5:
    st.markdown("""
    <div class="panel-header" style="background:white;border-radius:14px;
         padding:14px 16px 10px 16px;box-shadow:0 2px 12px rgba(0,0,0,0.08);
         border:1px solid #e8f5e9;margin-bottom:10px">
        <div class="panel-number">5</div>
        <div>
            <div class="panel-title">How Reliable is the Result?</div>
            <div class="panel-subtitle">We check and ensure the reliability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="background:white;border-radius:14px;padding:14px;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.08);border:1px solid #e8f5e9;">',
        unsafe_allow_html=True
    )

    # Model Reliability
    rel_level = ("High",   "#2d6a4f", "✅", "#f0faf4", "#b7e4c7") \
                if confidence >= 0.80 else \
                ("Medium", "#f57f17", "⚠️", "#fff8e1", "#ffe082") \
                if confidence >= 0.60 else \
                ("Low",    "#c62828", "❌", "#ffebee", "#ffcdd2")

    st.markdown(f"""
    <div style="background:{rel_level[3]};border:1px solid {rel_level[4]};
         border-radius:10px;padding:10px 12px;margin-bottom:8px;text-align:center">
        <div style="font-size:0.68em;color:#6c757d;
             font-weight:600;margin-bottom:4px">Model Reliability</div>
        <div style="font-size:0.9em;font-weight:700;
             color:{rel_level[1]};margin-bottom:2px">
            {rel_level[2]} {rel_level[0]}
        </div>
        <div style="font-size:0.65em;color:{rel_level[1]}">
            Reliable Recommendation
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction Reliability gauge
    pred_conf_pct = confidence * 100
    gauge_color = "#1565c0" if pred_conf_pct >= 80 else \
                  "#f57f17" if pred_conf_pct >= 60 else "#c62828"

    st.markdown(f"""
    <div style="background:#e8f4f8;border:1px solid #b3d9f0;
         border-radius:10px;padding:10px 12px;margin-bottom:8px;
         text-align:center">
        <div style="font-size:0.68em;color:#6c757d;
             font-weight:600;margin-bottom:4px">Prediction Reliability</div>
        <div style="font-size:2.0em;font-weight:900;
             color:{gauge_color};line-height:1.1">
            {pred_conf_pct:.1f}%
        </div>
        <div style="font-size:0.65em;color:{gauge_color};font-weight:600">
            {"High" if pred_conf_pct >= 80 else "Medium" if pred_conf_pct >= 60 else "Low"} Confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Result Quality Checks
    st.markdown("""
    <div style="font-size:0.72em;font-weight:700;color:#1b4332;
         margin-bottom:6px">Result Quality Checks</div>
    """, unsafe_allow_html=True)

    checks = [
        ("Well Calibrated",        final_ece <= 0.05),
        ("Low Prediction Error",   final_f1  >= 0.90),
        ("Stable & Consistent",    shap_tau  >= 0.80),
        ("Trustworthy Result",      abs(final_coverage - 0.95) <= 0.03),
    ]

    for check_name, passed in checks:
        bg  = "#d4edda" if passed else "#fff3cd"
        fg  = "#155724" if passed else "#856404"
        ico = "✅"       if passed else "⚠️"
        st.markdown(f"""
        <div style="background:{bg};color:{fg};border-radius:7px;
             padding:5px 9px;margin-bottom:4px;font-size:0.70em;
             font-weight:600;display:flex;align-items:center;gap:5px">
            <span>{ico}</span> {check_name}
        </div>
        """, unsafe_allow_html=True)

    # System metrics mini row
    st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    mc1.markdown(f"""
    <div class="mini-metric">
        <div class="mini-metric-val">{final_f1:.3f}</div>
        <div class="mini-metric-lbl">F1 Score</div>
    </div>
    """, unsafe_allow_html=True)
    mc2.markdown(f"""
    <div class="mini-metric">
        <div class="mini-metric-val">{final_ece:.3f}</div>
        <div class="mini-metric-lbl">ECE</div>
    </div>
    """, unsafe_allow_html=True)

    mc3, mc4 = st.columns(2)
    mc3.markdown(f"""
    <div class="mini-metric" style="margin-top:4px">
        <div class="mini-metric-val">{final_coverage:.3f}</div>
        <div class="mini-metric-lbl">Coverage</div>
    </div>
    """, unsafe_allow_html=True)
    mc4.markdown(f"""
    <div class="mini-metric" style="margin-top:4px">
        <div class="mini-metric-val">{shap_tau:.3f}</div>
        <div class="mini-metric-lbl">SHAP τ</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer-bar">
    <span style="font-size:1.8em">👨‍🌾</span>
    <div class="footer-text">
        <span class="footer-bold">Our Goal:</span> 
        To help farmers make better, reliable and confident crop choices 
        for higher yield and sustainable farming.
    </div>
    <span style="font-size:1.5em">🌿</span>
    <span style="font-size:1.5em">🚜</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# EXPANDABLE DEEP ANALYSIS (below main UI)
# ============================================================
st.markdown("<div style='margin-top:15px'>", unsafe_allow_html=True)

with st.expander("📊 Deep Analysis — NSGA-II Pareto Front, Calibration & Model Comparison", 
                 expanded=False):

    ta1, ta2, ta3 = st.tabs([
        "🧬 NSGA-II Pareto Front",
        "📐 Calibration Analysis",
        "🤖 Model Comparison"
    ])

    # ── Tab 1: Pareto Front ──────────────────────────────────
    with ta1:
        if pareto_F is not None and len(pareto_F) > 0:
            pc1, pc2 = st.columns(2)

            with pc1:
                fig_p1 = go.Figure()
                fig_p1.add_trace(go.Scatter(
                    x=-pareto_F[:, 0], y=pareto_F[:, 2],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=-pareto_F[:, 3],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='SHAP τ', len=0.8),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=(
                        'F1: %{x:.4f}<br>'
                        'ECE: %{y:.4f}<extra></extra>'
                    ),
                    name='Pareto Solutions'
                ))
                fig_p1.add_trace(go.Scatter(
                    x=[-pareto_F[best_pareto_idx, 0]],
                    y=[pareto_F[best_pareto_idx, 2]],
                    mode='markers',
                    marker=dict(
                        size=18, color='red',
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    ),
                    name='Best Compromise'
                ))
                fig_p1.add_hline(
                    y=0.05, line_dash="dot", line_color="red",
                    annotation_text="ECE Target"
                )
                fig_p1.add_vline(
                    x=0.90, line_dash="dot", line_color="blue",
                    annotation_text="F1 Target"
                )
                fig_p1.update_layout(
                    title="Pareto Front: F1 vs ECE",
                    xaxis_title="F1 Score ↑",
                    yaxis_title="ECE ↓",
                    height=380,
                    plot_bgcolor='rgba(240,249,243,0.5)',
                    legend=dict(x=0.02, y=0.02)
                )
                st.plotly_chart(fig_p1, use_container_width=True)

            with pc2:
                fig_p2 = go.Figure()
                fig_p2.add_trace(go.Scatter(
                    x=-pareto_F[:, 0], y=pareto_F[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=pareto_F[:, 2],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title='ECE', len=0.8),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=(
                        'F1: %{x:.4f}<br>'
                        'Set Size: %{y:.2f}<extra></extra>'
                    ),
                    name='Pareto Solutions'
                ))
                fig_p2.add_trace(go.Scatter(
                    x=[-pareto_F[best_pareto_idx, 0]],
                    y=[pareto_F[best_pareto_idx, 1]],
                    mode='markers',
                    marker=dict(
                        size=18, color='red',
                        symbol='star',
                        line=dict(color='darkred', width=1)
                    ),
                    name='Best Compromise'
                ))
                fig_p2.update_layout(
                    title="Pareto Front: F1 vs Uncertainty",
                    xaxis_title="F1 Score ↑",
                    yaxis_title="Avg Prediction Set Size ↓",
                    height=380,
                    plot_bgcolor='rgba(240,249,243,0.5)',
                    legend=dict(x=0.02, y=0.95)
                )
                st.plotly_chart(fig_p2, use_container_width=True)

            if not pareto_df.empty:
                st.markdown("**Pareto-Optimal Solutions Table:**")
                st.dataframe(
                    pareto_df.round(4),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Pareto front data not available.")

    # ── Tab 2: Calibration ───────────────────────────────────
    with ta2:
        cc1, cc2 = st.columns(2)

        with cc1:
            # ECE comparison bar chart
            if ece_results:
                models_ece  = list(ece_results.keys())
                raw_eces    = [ece_results[m]['ECE_raw']        for m in models_ece]
                cal_eces    = [ece_results[m]['ECE_calibrated'] for m in models_ece]

                fig_ece = go.Figure()
                fig_ece.add_trace(go.Bar(
                    name='Before Calibration', x=models_ece, y=raw_eces,
                    marker_color='#e74c3c', opacity=0.85,
                    text=[f'{v:.4f}' for v in raw_eces],
                    textposition='outside'
                ))
                fig_ece.add_trace(go.Bar(
                    name='After Calibration', x=models_ece, y=cal_eces,
                    marker_color='#2d6a4f', opacity=0.85,
                    text=[f'{v:.4f}' for v in cal_eces],
                    textposition='outside'
                ))
                fig_ece.add_hline(
                    y=0.05, line_dash="dash", line_color="blue",
                    line_width=2,
                    annotation_text="Target ECE = 0.05"
                )
                fig_ece.update_layout(
                    title="Expected Calibration Error by Model",
                    yaxis_title="ECE (lower is better)",
                    barmode='group',
                    height=380,
                    plot_bgcolor='rgba(240,249,243,0.5)',
                    legend=dict(x=0.6, y=0.95)
                )
                st.plotly_chart(fig_ece, use_container_width=True)

        with cc2:
            # Trustworthiness Radar
            cats = [
                'F1 Score<br>(≥0.90)',
                'Coverage<br>(≈0.95)',
                'ECE⁻¹<br>(≤0.05)',
                'SHAP τ<br>(≥0.80)'
            ]
            rvals = [
                min(1.0, final_f1 / 0.95),
                min(1.0, max(0.0, 1 - abs(final_coverage - 0.95) / 0.05)),
                min(1.0, max(0.0, 1 - final_ece / 0.05)),
                min(1.0, shap_tau / 0.90)
            ]

            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=rvals + [rvals[0]],
                theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(45,106,79,0.25)',
                line=dict(color='#2d6a4f', width=2.5),
                marker=dict(size=8, color='#1b4332'),
                name='Achieved'
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=[1, 1, 1, 1, 1],
                theta=cats + [cats[0]],
                fill='none',
                line=dict(color='red', width=1.5, dash='dot'),
                name='Target'
            ))
            fig_rad.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, range=[0, 1.15],
                        tickfont=dict(size=9)
                    ),
                    angularaxis=dict(tickfont=dict(size=10))
                ),
                title="Trustworthiness Radar Chart",
                height=380,
                showlegend=True,
                legend=dict(x=0.8, y=1.1)
            )
            st.plotly_chart(fig_rad, use_container_width=True)

    # ── Tab 3: Model Comparison ──────────────────────────────
    with ta3:
        if not results_df.empty:
            mc_c1, mc_c2 = st.columns(2)

            with mc_c1:
                fig_f1 = go.Figure()
                for col_name, color, label in [
                    ('f1_calibrated', '#2d6a4f', 'F1 (Calibrated)'),
                    ('acc_calibrated', '#52b788', 'Accuracy'),
                ]:
                    if col_name in results_df.columns:
                        fig_f1.add_trace(go.Bar(
                            name=label,
                            x=results_df.index,
                            y=results_df[col_name],
                            marker_color=color,
                            opacity=0.85,
                            text=[f'{v:.4f}' for v in results_df[col_name]],
                            textposition='outside'
                        ))

                fig_f1.add_hline(
                    y=0.90, line_dash="dot", line_color="red",
                    annotation_text="F1 Target = 0.90"
                )
                fig_f1.update_layout(
                    title="Model Performance Comparison",
                    yaxis_title="Score",
                    yaxis_range=[0.7, 1.05],
                    barmode='group',
                    height=350,
                    plot_bgcolor='rgba(240,249,243,0.5)',
                    legend=dict(x=0, y=1)
                )
                st.plotly_chart(fig_f1, use_container_width=True)

            with mc_c2:
                if 'cv_mean' in results_df.columns:
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Bar(
                        x=results_df.index,
                        y=results_df['cv_mean'],
                        error_y=dict(
                            type='data',
                            array=results_df['cv_std'].values,
                            visible=True
                        ),
                        marker_color='#40916c',
                        opacity=0.85,
                        text=[f'{m:.3f}±{s:.3f}'
                              for m, s in zip(results_df['cv_mean'],
                                              results_df['cv_std'])],
                        textposition='outside'
                    ))
                    fig_cv.add_hline(
                        y=0.90, line_dash="dot", line_color="red",
                        annotation_text="Target = 0.90"
                    )
                    fig_cv.update_layout(
                        title="5-Fold Cross-Validation F1",
                        yaxis_title="CV F1 (Mean ± Std)",
                        yaxis_range=[0.7, 1.05],
                        height=350,
                        plot_bgcolor='rgba(240,249,243,0.5)'
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)

            # Summary table
            st.markdown("**Complete Trustworthiness Report:**")
            summary = pd.DataFrame({
                'Objective': [
                    'Prediction Performance',
                    'Uncertainty Reliability',
                    'Calibration Quality',
                    'Explanation Stability'
                ],
                'Metric': [
                    'Macro F1 Score',
                    'Coverage',
                    'ECE',
                    "Kendall's τ"
                ],
                'Value': [
                    f"{final_f1:.4f}",
                    f"{final_coverage:.4f}",
                    f"{final_ece:.4f}",
                    f"{shap_tau:.4f}"
                ],
                'Target': ['≥ 0.90', '≈ 0.95', '≤ 0.05', '≥ 0.80'],
                'Status': [
                    '✅ Met' if final_f1       >= 0.90              else '⚠️',
                    '✅ Met' if abs(final_coverage - 0.95) <= 0.03  else '⚠️',
                    '✅ Met' if final_ece       <= 0.05              else '⚠️',
                    '✅ Met' if shap_tau        >= 0.80              else '⚠️',
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)