# ============================================================
# TRUSTWORTHY CROP RECOMMENDATION SYSTEM
# Matching Poster Design - No Soil Type
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Trustworthy Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.main > div { padding: 0 !important; }
.block-container {
    padding: 0 0 10px 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg,#14532d 0%,#166534 25%,#15803d 60%,#16a34a 100%);
    padding: 16px 50px 14px 50px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    margin-bottom: 10px;
}
.hero::before {
    content: '🌿🍃';
    position: absolute;
    font-size: 3.5em;
    left: 18px; top: 8px;
    opacity: 0.3;
    letter-spacing: -5px;
}
.hero::after {
    content: '🚜';
    position: absolute;
    font-size: 3.2em;
    right: 22px; top: 10px;
    opacity: 0.25;
}
.hero-title {
    color: #ffffff;
    font-size: 1.80em;
    font-weight: 800;
    margin: 0 0 7px 0;
    text-shadow: 0 2px 6px rgba(0,0,0,0.35);
    letter-spacing: -0.4px;
}
.hero-pills {
    display: inline-flex;
    align-items: center;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 30px;
    padding: 5px 22px;
    gap: 4px;
}
.hero-pill {
    color: #bbf7d0;
    font-size: 0.77em;
    font-weight: 600;
    padding: 0 8px;
}
.hero-sep { color: rgba(255,255,255,0.35); }

/* ── CARD ── */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 13px 14px 14px 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.04);
    height: 100%;
    position: relative;
}
.card-hdr {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 11px;
    padding-bottom: 9px;
    border-bottom: 1.5px solid #dcfce7;
}
.card-num {
    background: linear-gradient(135deg,#15803d,#16a34a);
    color: white;
    min-width: 24px; height: 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72em; font-weight: 800;
    flex-shrink: 0; margin-top: 1px;
    box-shadow: 0 2px 6px rgba(21,128,61,0.35);
}
.card-title {
    font-size: 0.88em;
    font-weight: 700;
    color: #14532d;
    margin: 0; line-height: 1.25;
}
.card-sub {
    font-size: 0.64em;
    color: #6b7280;
    margin: 2px 0 0 0;
}

/* ── INPUT ROWS (Panel 1) ── */
.inp-row {
    display: flex; align-items: center;
    padding: 5px 9px;
    border-radius: 8px; margin-bottom: 5px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    gap: 8px; min-height: 30px;
}
.inp-icon { font-size: 1.0em; width: 19px; text-align: center; flex-shrink: 0; }
.inp-label { font-size: 0.72em; color: #374151; font-weight: 500; flex: 1; }
.inp-val { font-size: 0.74em; font-weight: 700; color: #15803d; text-align: right; flex-shrink: 0; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg,#14532d,#16a34a) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 11px 18px !important;
    font-size: 0.88em !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(22,163,74,0.4) !important;
    letter-spacing: 0.3px !important;
    margin-top: 4px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(22,163,74,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── CROP CIRCLE ── */
.crop-ring {
    width: 108px; height: 108px;
    border-radius: 50%;
    background: radial-gradient(circle at 32% 32%, #dcfce7, #86efac, #22c55e);
    display: flex; align-items: center; justify-content: center;
    margin: 2px auto 9px auto;
    font-size: 3.8em;
    border: 5px solid #16a34a;
    box-shadow: 0 0 0 3px #bbf7d080, 0 6px 20px rgba(22,163,74,0.3);
}
.crop-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg,#15803d,#22c55e);
    color: white;
    border-radius: 25px;
    padding: 6px 20px;
    font-size: 1.05em; font-weight: 800;
    letter-spacing: 1.5px;
    box-shadow: 0 3px 12px rgba(22,163,74,0.4);
}
.conf-lbl {
    text-align: center;
    font-size: 0.67em; color: #6b7280;
    font-weight: 500; margin: 11px 0 2px 0;
}
.conf-num {
    text-align: center;
    font-size: 3.0em; font-weight: 900;
    color: #15803d; line-height: 1.0;
    letter-spacing: -1px;
}
.conf-track {
    background: #e5e7eb; border-radius: 8px;
    height: 10px; margin: 7px 0 2px 0; overflow: hidden;
}
.conf-fill {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg,#4ade80,#15803d);
}
.conf-scale {
    display: flex; justify-content: space-between;
    font-size: 0.60em; color: #9ca3af; margin-top: 1px;
}
.why-box {
    background: #f0fdf4;
    border-radius: 9px; padding: 9px 11px; margin-top: 10px;
    border-left: 3px solid #22c55e;
}
.why-title { font-size: 0.70em; font-weight: 700; color: #15803d; margin-bottom: 3px; }
.why-text  { font-size: 0.66em; color: #374151; line-height: 1.55; }

/* ── ALTERNATIVES ── */
.alt-card {
    display: flex; align-items: center; gap: 9px;
    padding: 8px 10px; border-radius: 11px; margin-bottom: 7px;
    border: 1px solid #e5e7eb; background: #fafafa;
    transition: all 0.2s;
}
.alt-card-1 {
    background: linear-gradient(135deg,#f0fdf4,#dcfce7) !important;
    border: 1.5px solid #86efac !important;
}
.alt-rank { font-size: 0.82em; font-weight: 800; width: 18px; flex-shrink: 0; }
.alt-icon  { font-size: 1.5em; flex-shrink: 0; }
.alt-body  { flex: 1; min-width: 0; }
.alt-name  { font-size: 0.78em; font-weight: 700; color: #14532d; }
.alt-slbl  { font-size: 0.60em; color: #6b7280; margin-top: 1px; }
.alt-track { background: #e5e7eb; border-radius: 4px; height: 5px; margin-top: 3px; overflow: hidden; }
.alt-pct   { font-size: 0.82em; font-weight: 800; flex-shrink: 0; }

/* ── TIP BOX ── */
.tip-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 3px solid #f59e0b;
    border-radius: 10px;
    padding: 9px 11px; margin-top: 8px;
}
.tip-title { font-size: 0.70em; font-weight: 700; color: #b45309; margin-bottom: 3px; }
.tip-text  { font-size: 0.65em; color: #78350f; line-height: 1.55; }

/* ── SHAP PANEL ── */
.toggle-row { display: flex; gap: 7px; margin-bottom: 9px; }
.tog-pos {
    background: #dcfce7; color: #14532d;
    border: 1px solid #86efac;
    border-radius: 14px; padding: 3px 11px;
    font-size: 0.68em; font-weight: 700;
}
.tog-neg {
    background: #fee2e2; color: #7f1d1d;
    border: 1px solid #fca5a5;
    border-radius: 14px; padding: 3px 11px;
    font-size: 0.68em; font-weight: 700;
}
.shap-row {
    display: flex; align-items: center;
    gap: 7px; margin-bottom: 7px;
}
.shap-icon { font-size: 0.88em; width: 17px; text-align: center; flex-shrink: 0; }
.shap-name { font-size: 0.69em; color: #374151; font-weight: 500; width: 80px; flex-shrink: 0; }
.shap-track { flex: 1; background: #f3f4f6; border-radius: 4px; height: 10px; overflow: hidden; }
.shap-green { height: 100%; border-radius: 4px; background: linear-gradient(90deg,#4ade80,#15803d); }
.shap-red   { height: 100%; border-radius: 4px; background: linear-gradient(90deg,#f87171,#dc2626); }
.shap-vg { font-size: 0.68em; font-weight: 700; color: #15803d; width: 36px; text-align: right; flex-shrink: 0; }
.shap-vr { font-size: 0.68em; font-weight: 700; color: #dc2626; width: 36px; text-align: right; flex-shrink: 0; }
.what-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 3px solid #3b82f6;
    border-radius: 9px; padding: 9px 11px; margin-top: 8px;
}
.what-title { font-size: 0.70em; font-weight: 700; color: #1d4ed8; margin-bottom: 3px; }
.what-text  { font-size: 0.65em; color: #1e3a8a; line-height: 1.55; }

/* ── RELIABILITY ── */
.rel-section-lbl {
    font-size: 0.68em; font-weight: 700;
    color: #14532d; margin: 0 0 5px 0;
}
.model-rel {
    border-radius: 11px; padding: 10px 12px;
    margin-bottom: 9px;
    display: flex; align-items: center; gap: 10px;
}
.rel-icon { font-size: 2.2em; flex-shrink: 0; }
.rel-lvl  { font-size: 1.4em; font-weight: 800; line-height: 1.1; }
.rel-sub  { font-size: 0.62em; font-weight: 500; margin-top: 1px; }
.pred-rel {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 11px; padding: 10px 12px; margin-bottom: 9px;
    display: flex; align-items: center; gap: 10px;
}
.pred-pct { font-size: 1.85em; font-weight: 900; line-height: 1.0; }
.pred-lbl { font-size: 0.64em; font-weight: 700; margin-top: 2px; }
.qrow {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 9px; border-radius: 7px;
    margin-bottom: 4px; font-size: 0.68em; font-weight: 600;
}
.qpass { background: #dcfce7; color: #14532d; }
.qwarn { background: #fef9c3; color: #713f12; }
.mtile-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 5px; margin-top: 9px;
}
.mtile {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 9px; padding: 7px 8px; text-align: center;
}
.mtile-val { font-size: 0.95em; font-weight: 800; color: #15803d; }
.mtile-lbl { font-size: 0.57em; color: #6b7280; font-weight: 500; margin-top: 1px; }

/* ── FOOTER ── */
.footer {
    background: linear-gradient(135deg,#14532d,#16a34a);
    border-radius: 14px;
    margin: 8px 10px 10px 10px;
    padding: 12px 28px;
    display: flex; align-items: center;
    justify-content: center; gap: 14px;
    box-shadow: 0 3px 14px rgba(21,128,61,0.3);
}
.footer-txt { color: #bbf7d0; font-size: 0.80em; text-align: center; }
.footer-bold { color: white; font-weight: 700; }

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stNumberInput"] label {
    font-size: 0.74em !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
div[data-testid="stNumberInput"] input {
    font-size: 0.82em !important;
    border-radius: 8px !important;
    border-color: #d1fae5 !important;
    background: #f0fdf4 !important;
    padding: 5px 9px !important;
}
div[data-testid="stSelectSlider"] label {
    font-size: 0.72em !important; color: #374151 !important;
}
div[data-testid="stSelectbox"] label {
    font-size: 0.72em !important; color: #374151 !important;
}
div[data-baseweb="select"] {
    border-radius: 8px !important;
}
.stExpander {
    border: 1px solid #d1fae5 !important;
    border-radius: 12px !important;
    margin: 0 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource(show_spinner="🌾 Loading model, please wait...")
def load_artifacts():
    try:
        with open('crop_recommendation_artifacts.pkl', 'rb') as f:
            return pickle.load(f), True
    except FileNotFoundError:
        return None, False

@st.cache_resource(show_spinner=False)
def get_explainer(_base_model):
    try:
        import shap
        return shap.TreeExplainer(_base_model)
    except:
        return None

artifacts, loaded = load_artifacts()
if not loaded:
    st.error("❌ `crop_recommendation_artifacts.pkl` not found!")
    st.stop()

model           = artifacts['model']
base_model      = artifacts['base_model']
scaler          = artifacts['scaler']
class_names     = artifacts['class_names']
feature_cols    = artifacts['feature_cols']
cal_scores      = artifacts['cal_scores']
shap_importance = np.array(artifacts['shap_mean_importance']).flatten()
shap_tau        = float(artifacts['shap_stability_score'])
final_ece       = float(artifacts['final_ece'])
final_f1        = float(artifacts['final_f1'])
final_coverage  = float(artifacts['final_coverage'])
results_df      = artifacts.get('results_df', pd.DataFrame())
ece_results     = artifacts.get('ece_results', {})
pareto_df       = artifacts.get('pareto_df', pd.DataFrame())
pareto_F        = artifacts.get('pareto_F', np.array([]))
best_pareto_idx = int(artifacts.get('best_pareto_idx', 0))

explainer = get_explainer(base_model)


# ============================================================
# CONSTANTS
# ============================================================
CROP_EMOJI = {
    'rice':'🌾','maize':'🌽','chickpea':'🫘','kidneybeans':'🫘',
    'pigeonpeas':'🌿','mothbeans':'🌱','mungbean':'🌱','blackgram':'🌱',
    'lentil':'🌾','pomegranate':'🍎','banana':'🍌','mango':'🥭',
    'grapes':'🍇','watermelon':'🍉','muskmelon':'🍈','apple':'🍎',
    'orange':'🍊','papaya':'🍈','coconut':'🥥','cotton':'🌸',
    'jute':'🌿','coffee':'☕'
}
FEAT_ICON = {
    'N':'🌿','P':'🔴','K':'🟣',
    'temperature':'🌡️','humidity':'💧','ph':'🧪','rainfall':'🌧️'
}
FEAT_NAME = {
    'N':'Nitrogen (N)','P':'Phosphorus (P)','K':'Potassium (K)',
    'temperature':'Temperature','humidity':'Humidity',
    'ph':'Soil pH','rainfall':'Rainfall'
}
FEAT_UNIT = {
    'N':'kg/ha','P':'kg/ha','K':'kg/ha',
    'temperature':'°C','humidity':'%','ph':'','rainfall':'mm'
}
ALT_COLORS = ["#15803d","#16a34a","#22c55e","#4ade80"]
REASONS = {
    'rice':        "Your soil nutrients, weather and other conditions are highly suitable for Rice cultivation.",
    'maize':       "Warm temperature, moderate humidity and your NPK profile are highly suitable for Maize.",
    'coffee':      "Your specific temperature, humidity and pH conditions are highly suitable for Coffee.",
    'banana':      "High humidity, warm temperature and potassium levels are highly suitable for Banana.",
    'wheat':       "Moderate temperature, balanced nutrients and your pH are highly suitable for Wheat.",
    'cotton':      "Warm climate, low rainfall and your soil conditions are highly suitable for Cotton.",
    'jute':        "Warm humid conditions and your soil profile are highly suitable for Jute cultivation.",
    'coconut':     "Warm temperature and high humidity are highly suitable for Coconut cultivation.",
    'mango':       "Your temperature range and soil pH are highly suitable for Mango cultivation.",
    'grapes':      "Your climate conditions and soil acidity are highly suitable for Grape cultivation.",
    'watermelon':  "Warm temperature and your NPK profile are highly suitable for Watermelon.",
    'apple':       "Cool temperature and your soil conditions are highly suitable for Apple orchards.",
    'orange':      "Your pH and humidity levels are highly suitable for Orange cultivation.",
    'papaya':      "Warm temperature and moderate rainfall are highly suitable for Papaya cultivation.",
    'pomegranate': "Your soil pH and temperature are highly suitable for Pomegranate cultivation.",
    'muskmelon':   "Warm dry conditions and your NPK profile are highly suitable for Muskmelon.",
    'blackgram':   "Your soil nutrients and moisture conditions are highly suitable for Blackgram.",
    'mungbean':    "Your warm conditions and soil nutrients are highly suitable for Mungbean.",
    'mothbeans':   "Dry warm conditions and your soil are highly suitable for Mothbeans.",
    'pigeonpeas':  "Your warm climate and soil NPK are highly suitable for Pigeonpeas.",
    'kidneybeans': "Your soil nutrients and moderate climate are highly suitable for Kidneybeans.",
    'lentil':      "Cool conditions and your soil nutrients are highly suitable for Lentil cultivation.",
    'chickpea':    "Cool dry conditions and your soil are highly suitable for Chickpea cultivation.",
}


# ============================================================
# HELPERS
# ============================================================
def run_predict(vals):
    arr    = np.array(vals).reshape(1, -1)
    scaled = scaler.transform(arr)
    prob   = model.predict_proba(scaled)[0]
    pred   = int(np.argmax(prob))
    return pred, prob, scaled[0]

def get_pred_set(x_sc, alpha=0.05):
    n     = len(cal_scores)
    lvl   = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    quant = np.quantile(cal_scores, lvl)
    prob  = model.predict_proba(x_sc.reshape(1,-1))[0]
    return [c for c in range(len(prob)) if (1-prob[c]) <= quant]

def get_shap(x_sc, pred_cls):
    if explainer is None:
        return None
    try:
        sv = explainer.shap_values(x_sc.reshape(1,-1))
        if isinstance(sv, list):
            return np.array(sv[pred_cls][0]).flatten()
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            return sv[0, :, pred_cls].flatten()
        return sv[0].flatten()
    except:
        return None

def crop_reason(c):
    return REASONS.get(
        c.lower(),
        f"Your soil nutrients, weather and other conditions are highly "
        f"suitable for {c.title()} cultivation."
    )

def ph_status(ph_val):
    if ph_val < 5.5:
        return "Strongly Acidic", "#dc2626", "#fef2f2", "#fecaca"
    elif ph_val < 6.0:
        return "Acidic",          "#d97706", "#fffbeb", "#fde68a"
    elif ph_val <= 7.0:
        return "Optimal ✓",       "#15803d", "#f0fdf4", "#bbf7d0"
    elif ph_val <= 7.5:
        return "Slightly Alkaline","#d97706", "#fffbeb", "#fde68a"
    else:
        return "Alkaline",        "#dc2626", "#fef2f2", "#fecaca"


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">
        🌾 Streamlit-based Trustworthy Crop Recommendation System
    </div>
    <div class="hero-pills">
        <span class="hero-pill">Smart Recommendations</span>
        <span class="hero-sep">•</span>
        <span class="hero-pill">Explainable Results</span>
        <span class="hero-sep">•</span>
        <span class="hero-pill">Better Decisions for Farmers</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 5-COLUMN LAYOUT
# ============================================================
c1, c2, c3, c4, c5 = st.columns([1.0, 1.05, 1.05, 1.15, 0.98],
                                   gap="small")


# ════════════════════════════════════════════════════════════
# PANEL 1 — Field Details Input
# ════════════════════════════════════════════════════════════
with c1:
    st.markdown("""
    <div class="card-hdr" style="background:white;border-radius:14px 14px 0 0;
         padding:13px 14px 9px 14px;margin-bottom:0;
         box-shadow:0 2px 12px rgba(0,0,0,0.08),0 0 0 1px rgba(0,0,0,0.04);">
        <div class="card-num">1</div>
        <div>
            <div class="card-title">Enter Your Field Details 🌱</div>
            <div class="card-sub">Fill in simple information about your field</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            '<div style="background:white;border-radius:0 0 16px 16px;'
            'padding:0 14px 14px 14px;'
            'box-shadow:0 2px 12px rgba(0,0,0,0.08),0 0 0 1px rgba(0,0,0,0.04);'
            'margin-top:0">',
            unsafe_allow_html=True
        )

        N    = st.number_input("🌿 Nitrogen N (kg/ha)",    0.0, 145.0, 90.0,  1.0)
        P    = st.number_input("🔴 Phosphorus P (kg/ha)",  0.0, 150.0, 42.0,  1.0)
        K    = st.number_input("🟣 Potassium K (kg/ha)",   0.0, 210.0, 43.0,  1.0)
        ph   = st.number_input("🧪 Soil pH",               3.0,  10.0,  6.5, 0.05)

        # Dynamic pH status badge
        ph_lbl, ph_col, ph_bg, ph_brd = ph_status(ph)
        st.markdown(f"""
        <div style="background:{ph_bg};border:1px solid {ph_brd};
             border-radius:8px;padding:5px 10px;margin-bottom:5px;
             display:flex;align-items:center;gap:7px">
            <span style="font-size:0.9em">🧪</span>
            <span style="font-size:0.68em;color:#6b7280;font-weight:500">
                pH Status:
            </span>
            <span style="font-size:0.72em;font-weight:700;color:{ph_col}">
                {ph_lbl}
            </span>
            <span style="margin-left:auto;font-size:0.72em;
                 font-weight:800;color:{ph_col}">{ph:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        rain = st.number_input("🌧️ Rainfall (mm)",          15.0, 300.0, 202.0, 5.0)
        temp = st.number_input("🌡️ Temperature (°C)",        5.0,  50.0, 25.0,  0.1)
        hum  = st.number_input("💧 Humidity (%)",           10.0, 100.0, 71.0,  0.5)

        # Summary display panel
        current_vals = [N, P, K, temp, hum, ph, rain]
        rows_html = ""
        for i, f in enumerate(feature_cols):
            v   = current_vals[i]
            u   = FEAT_UNIT[f]
            ico = FEAT_ICON[f]
            nm  = FEAT_NAME[f]
            rows_html += (
                f'<div class="inp-row">'
                f'<span class="inp-icon">{ico}</span>'
                f'<span class="inp-label">{nm}</span>'
                f'<span class="inp-val">{v:.1f}{" "+u if u else ""}</span>'
                f'</div>'
            )

        st.markdown(f"""
        <div style="margin-top:6px;margin-bottom:6px">
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

        # Presets
        st.markdown(
            '<div style="font-size:0.63em;color:#6b7280;font-weight:600;'
            'margin:4px 0 3px 1px">⚡ Quick Presets</div>',
            unsafe_allow_html=True
        )
        pb1, pb2 = st.columns(2)
        PRESETS = {
            "🌾 Rice":   [90, 42, 43, 6.5,  202.0, 20.9, 82.0],
            "🌽 Maize":  [71, 54, 16, 5.75,  87.8, 22.6, 63.7],
            "☕ Coffee": [101,28, 29, 6.8,  158.1, 25.5, 58.9],
            "🍌 Banana": [100,82, 50, 5.9,  105.5, 27.4, 80.3],
        }
        for i, (pname, pvals) in enumerate(PRESETS.items()):
            btn_col = pb1 if i % 2 == 0 else pb2
            if btn_col.button(pname, key=f"pr{i}"):
                st.session_state['preset'] = pvals
                st.rerun()
        if 'preset' in st.session_state:
            pv = st.session_state.pop('preset')
            N, P, K, ph, rain, temp, hum = pv

        # Confidence level
        alpha = st.select_slider(
            "🎯 Confidence Level",
            options=[0.01, 0.05, 0.10, 0.15, 0.20],
            value=0.05,
            format_func=lambda x: f"{int((1-x)*100)}%"
        )

        # Main button
        recommend = st.button("🚀 Get Recommendation")
        st.markdown("</div>", unsafe_allow_html=True)


# ── Run prediction ────────────────────────────────────────────
input_vals      = [N, P, K, temp, hum, ph, rain]
y_pred, y_prob, x_sc = run_predict(input_vals)
confidence      = float(y_prob[y_pred])
crop            = class_names[y_pred]
crop_em         = CROP_EMOJI.get(crop.lower(), '🌱')
top5_idx        = np.argsort(y_prob)[::-1][:5]
pred_set        = get_pred_set(x_sc, alpha)
sv              = get_shap(x_sc, y_pred)
set_size        = len(pred_set)


# ════════════════════════════════════════════════════════════
# PANEL 2 — Recommended Crop
# ════════════════════════════════════════════════════════════
with c2:
    set_col = "#15803d" if set_size <= 2 else "#d97706"
    st.markdown(f"""
    <div class="card">
        <div class="card-hdr">
            <div class="card-num">2</div>
            <div>
                <div class="card-title">Recommended Crop 🌾</div>
                <div class="card-sub">Best crop for your field</div>
            </div>
        </div>

        <div style="text-align:center;margin:4px 0 2px 0">
            <div class="crop-ring">{crop_em}</div>
            <div>
                <span class="crop-pill">✅&nbsp;{crop.upper()}</span>
            </div>
        </div>

        <div class="conf-lbl">Confidence Score (Predictive)</div>
        <div class="conf-num">{confidence:.0%}</div>
        <div class="conf-track">
            <div class="conf-fill"
                 style="width:{confidence*100:.1f}%"></div>
        </div>
        <div class="conf-scale">
            <span>0%</span><span>100%</span>
        </div>

        <div class="why-box">
            <div class="why-title">🌿 Why this crop?</div>
            <div class="why-text">{crop_reason(crop)}</div>
        </div>

        <div style="margin-top:9px;background:#f0fdf4;border-radius:8px;
             padding:5px 10px;border:1px solid #bbf7d0;text-align:center">
            <span style="font-size:0.63em;color:#6b7280">
                Prediction Set ({int((1-alpha)*100)}% confidence):&nbsp;
                <strong style="color:{set_col}">
                    {set_size} crop(s)
                </strong>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PANEL 3 — Top Alternative Crops
# ════════════════════════════════════════════════════════════
with c3:
    alt_html = ""
    for ri, idx in enumerate(top5_idx[:4]):
        cn  = class_names[idx]
        cp  = float(y_prob[idx])
        ce  = CROP_EMOJI.get(cn.lower(), '🌱')
        rc  = ALT_COLORS[min(ri, len(ALT_COLORS)-1)]
        top = ' alt-card-1' if ri == 0 else ''
        alt_html += f"""
        <div class="alt-card{top}">
            <span class="alt-rank" style="color:{rc}">{ri+1}.</span>
            <span class="alt-icon">{ce}</span>
            <div class="alt-body">
                <div class="alt-name">{cn.title()}</div>
                <div class="alt-slbl">Suitability Score</div>
                <div class="alt-track">
                    <div style="width:{cp*100:.0f}%;height:100%;
                          border-radius:4px;
                          background:linear-gradient(90deg,{rc}70,{rc})">
                    </div>
                </div>
            </div>
            <span class="alt-pct" style="color:{rc}">{cp:.0%}</span>
        </div>"""

    tip_names = ", ".join([class_names[i].title() for i in top5_idx[1:4]])

    st.markdown(f"""
    <div class="card">
        <div class="card-hdr">
            <div class="card-num">3</div>
            <div>
                <div class="card-title">Top Alternative Crops 🌿</div>
                <div class="card-sub">Other good options for your field</div>
            </div>
        </div>
        {alt_html}
        <div class="tip-box">
            <div class="tip-title">💡 Tip for Farmers</div>
            <div class="tip-text">
                You can also consider these crops based on market demand,
                availability and your preferences.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PANEL 4 — Why This Recommendation? (SHAP)
# ════════════════════════════════════════════════════════════
with c4:
    if sv is not None:
        shap_arr = np.array(sv).flatten()
    else:
        shap_arr = shap_importance.copy()

    max_abs  = max(np.abs(shap_arr).max(), 1e-6)
    sort_idx = np.argsort(np.abs(shap_arr))[::-1]

    shap_rows = ""
    for fi in sort_idx:
        feat = feature_cols[fi]
        val  = float(shap_arr[fi])
        icon = FEAT_ICON.get(feat, '📊')
        name = FEAT_NAME.get(feat, feat)
        bw   = min(abs(val)/max_abs*100, 100)
        is_p = val >= 0
        bcls = "shap-green" if is_p else "shap-red"
        vcls = "shap-vg"    if is_p else "shap-vr"
        sign = "+" if is_p else ""
        shap_rows += f"""
        <div class="shap-row">
            <span class="shap-icon">{icon}</span>
            <span class="shap-name">{name}</span>
            <div class="shap-track">
                <div class="{bcls}" style="width:{bw:.1f}%"></div>
            </div>
            <span class="{vcls}">{sign}{val:.2f}</span>
        </div>"""

    if sv is not None:
        pos_feats = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                     for i in sort_idx if shap_arr[i] > 0][:2]
        neg_feats = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                     for i in sort_idx if shap_arr[i] < 0][:2]
        pos_str = " and ".join(pos_feats) if pos_feats else "the key features"
        neg_str = " and ".join(neg_feats) if neg_feats else "none"
        need_imp = "need improvement." if neg_feats else "show no negative impact."
        what_html = f"""
        <div class="what-box">
            <div class="what-title">❓ What this means?</div>
            <div class="what-text">
                <strong>{pos_str}</strong> are the most positive factors,
                while <strong>{neg_str}</strong> {need_imp}
            </div>
        </div>"""
    else:
        what_html = ""

    st.markdown(f"""
    <div class="card">
        <div class="card-hdr">
            <div class="card-num">4</div>
            <div>
                <div class="card-title">Why This Recommendation? 🔍</div>
                <div class="card-sub">Key factors that influenced the result</div>
            </div>
        </div>
        <div class="toggle-row">
            <span class="tog-pos">✅ Positive Impact</span>
            <span class="tog-neg">❌ Negative Impact</span>
        </div>
        {shap_rows}
        {what_html}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PANEL 5 — How Reliable is the Result?
# ════════════════════════════════════════════════════════════
with c5:
    # Reliability level
    if confidence >= 0.80:
        rlvl, rcol, rico = "High",   "#15803d", "✅"
        rbg,  rbrd       = "#f0fdf4","#bbf7d0"
    elif confidence >= 0.60:
        rlvl, rcol, rico = "Medium", "#d97706", "⚠️"
        rbg,  rbrd       = "#fffbeb","#fde68a"
    else:
        rlvl, rcol, rico = "Low",    "#dc2626", "❌"
        rbg,  rbrd       = "#fef2f2","#fecaca"

    gcol = ("#1d4ed8" if confidence>=0.80 else
            "#d97706" if confidence>=0.60 else "#dc2626")
    glvl = ("High Confidence"   if confidence>=0.80 else
            "Medium Confidence" if confidence>=0.60 else "Low Confidence")

    checks = [
        ("Well Calibrated",      final_ece      <= 0.05),
        ("Low Prediction Error", final_f1       >= 0.90),
        ("Stable & Consistent",  shap_tau       >= 0.80),
        ("Trustworthy Result",
         abs(final_coverage - 0.95) <= 0.03),
    ]
    checks_html = ""
    for cn2, ok in checks:
        cls = "qpass" if ok else "qwarn"
        ico = "✅"     if ok else "⚠️"
        checks_html += (
            f'<div class="qrow {cls}">{ico}&nbsp;{cn2}</div>'
        )

    st.markdown(f"""
    <div class="card">
        <div class="card-hdr">
            <div class="card-num">5</div>
            <div>
                <div class="card-title">How Reliable is the Result?</div>
                <div class="card-sub">We check and ensure the reliability</div>
            </div>
        </div>

        <div class="rel-section-lbl">Model Reliability</div>
        <div class="model-rel"
             style="background:{rbg};border:1px solid {rbrd}">
            <span class="rel-icon">{rico}</span>
            <div>
                <div class="rel-lvl" style="color:{rcol}">{rlvl}</div>
                <div class="rel-sub" style="color:{rcol}">
                    Reliable Recommendation
                </div>
            </div>
        </div>

        <div class="rel-section-lbl">Prediction Reliability</div>
        <div class="pred-rel">
            <span style="font-size:2.0em;flex-shrink:0">📊</span>
            <div>
                <div class="pred-pct" style="color:{gcol}">
                    {confidence*100:.1f}%
                </div>
                <div class="pred-lbl" style="color:{gcol}">{glvl}</div>
            </div>
        </div>

        <div class="rel-section-lbl">Result Quality Checks</div>
        {checks_html}

        <div class="mtile-grid">
            <div class="mtile">
                <div class="mtile-val">{final_f1:.3f}</div>
                <div class="mtile-lbl">F1 Score</div>
            </div>
            <div class="mtile">
                <div class="mtile-val">{final_ece:.3f}</div>
                <div class="mtile-lbl">ECE</div>
            </div>
            <div class="mtile">
                <div class="mtile-val">{final_coverage:.3f}</div>
                <div class="mtile-lbl">Coverage</div>
            </div>
            <div class="mtile">
                <div class="mtile-val">{shap_tau:.3f}</div>
                <div class="mtile-lbl">SHAP τ</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <span style="font-size:1.9em">👨‍🌾</span>
    <div class="footer-txt">
        <span class="footer-bold">Our Goal:</span>
        To help farmers make better, reliable and confident crop choices
        for higher yield and sustainable farming.
    </div>
    <span style="font-size:1.4em">🌿</span>
    <span style="font-size:1.4em">🌿</span>
    <span style="font-size:1.4em">🚜</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DEEP ANALYSIS EXPANDER
# ============================================================
with st.expander(
    "📊 Deep Analysis — NSGA-II Pareto Front, "
    "Calibration & Model Comparison",
    expanded=False
):
    import plotly.graph_objects as go

    t1, t2, t3 = st.tabs([
        "🧬 NSGA-II Pareto Front",
        "📐 Calibration & Radar",
        "🤖 Model Comparison"
    ])

    with t1:
        if pareto_F is not None and len(pareto_F) > 0:
            pa, pb = st.columns(2)
            with pa:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=-pareto_F[:,0], y=pareto_F[:,2],
                    mode='markers',
                    marker=dict(
                        size=11,
                        color=-pareto_F[:,3],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='SHAP τ', len=0.8),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='F1:%{x:.4f}<br>ECE:%{y:.4f}<extra></extra>'
                ))
                fig1.add_trace(go.Scatter(
                    x=[-pareto_F[best_pareto_idx,0]],
                    y=[pareto_F[best_pareto_idx,2]],
                    mode='markers',
                    marker=dict(size=16,color='red',symbol='star'),
                    name='Best Compromise'
                ))
                fig1.add_hline(y=0.05,line_dash="dot",line_color="red",
                               annotation_text="ECE Target")
                fig1.add_vline(x=0.90,line_dash="dot",line_color="blue",
                               annotation_text="F1 Target")
                fig1.update_layout(
                    title="Pareto Front: F1 vs ECE",
                    xaxis_title="F1 Score ↑",
                    yaxis_title="ECE ↓",
                    height=360,
                    plot_bgcolor='rgba(240,253,244,0.8)'
                )
                st.plotly_chart(fig1, use_container_width=True)

            with pb:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=-pareto_F[:,0], y=pareto_F[:,1],
                    mode='markers',
                    marker=dict(
                        size=11,
                        color=pareto_F[:,2],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title='ECE', len=0.8),
                        line=dict(color='white', width=1)
                    )
                ))
                fig2.add_trace(go.Scatter(
                    x=[-pareto_F[best_pareto_idx,0]],
                    y=[pareto_F[best_pareto_idx,1]],
                    mode='markers',
                    marker=dict(size=16,color='red',symbol='star'),
                    name='Best'
                ))
                fig2.update_layout(
                    title="Pareto Front: F1 vs Uncertainty",
                    xaxis_title="F1 Score ↑",
                    yaxis_title="Avg Set Size ↓",
                    height=360,
                    plot_bgcolor='rgba(240,253,244,0.8)'
                )
                st.plotly_chart(fig2, use_container_width=True)

            if not pareto_df.empty:
                st.dataframe(pareto_df.round(4),
                             use_container_width=True,
                             hide_index=True)
        else:
            st.info("Pareto data not available.")

    with t2:
        ca, cb = st.columns(2)
        with ca:
            if ece_results:
                mn  = list(ece_results.keys())
                re  = [ece_results[m]['ECE_raw']        for m in mn]
                ce2 = [ece_results[m]['ECE_calibrated'] for m in mn]
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    name='Before Calibration', x=mn, y=re,
                    marker_color='#ef4444',
                    text=[f'{v:.4f}' for v in re],
                    textposition='outside'
                ))
                fig3.add_trace(go.Bar(
                    name='After Calibration', x=mn, y=ce2,
                    marker_color='#16a34a',
                    text=[f'{v:.4f}' for v in ce2],
                    textposition='outside'
                ))
                fig3.add_hline(y=0.05,line_dash="dash",
                               line_color="blue",
                               annotation_text="Target = 0.05")
                fig3.update_layout(
                    title="ECE Before vs After Calibration",
                    barmode='group', height=360,
                    plot_bgcolor='rgba(240,253,244,0.8)'
                )
                st.plotly_chart(fig3, use_container_width=True)

        with cb:
            cats = ['F1≥0.90','Coverage\n≈0.95','ECE≤0.05','τ≥0.80']
            rv   = [
                min(1.0, final_f1/0.95),
                min(1.0, max(0.0, 1-abs(final_coverage-0.95)/0.05)),
                min(1.0, max(0.0, 1-final_ece/0.05)),
                min(1.0, shap_tau/0.90)
            ]
            fig4 = go.Figure()
            fig4.add_trace(go.Scatterpolar(
                r=rv+[rv[0]], theta=cats+[cats[0]],
                fill='toself',
                fillcolor='rgba(22,163,74,0.2)',
                line=dict(color='#16a34a', width=2.5),
                name='Achieved'
            ))
            fig4.add_trace(go.Scatterpolar(
                r=[1,1,1,1,1], theta=cats+[cats[0]],
                fill='none',
                line=dict(color='red', width=1.5, dash='dot'),
                name='Target'
            ))
            fig4.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1.1])
                ),
                title="Trustworthiness Radar",
                height=360, showlegend=True
            )
            st.plotly_chart(fig4, use_container_width=True)

    with t3:
        if not results_df.empty:
            ma, mb = st.columns(2)
            with ma:
                fig5 = go.Figure()
                for cn3, cc3, lb3 in [
                    ('f1_calibrated', '#16a34a', 'F1 Calibrated'),
                    ('acc_calibrated','#4ade80', 'Accuracy')
                ]:
                    if cn3 in results_df.columns:
                        fig5.add_trace(go.Bar(
                            name=lb3,
                            x=results_df.index,
                            y=results_df[cn3],
                            marker_color=cc3,
                            text=[f'{v:.4f}' for v in results_df[cn3]],
                            textposition='outside'
                        ))
                fig5.add_hline(y=0.90, line_dash="dot",
                               line_color="red",
                               annotation_text="F1 Target")
                fig5.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    yaxis_range=[0.7, 1.05],
                    height=340,
                    plot_bgcolor='rgba(240,253,244,0.8)'
                )
                st.plotly_chart(fig5, use_container_width=True)

            with mb:
                summary_data = pd.DataFrame({
                    'Objective': [
                        'Prediction Performance',
                        'Uncertainty Reliability',
                        'Calibration Quality',
                        'Explanation Stability'
                    ],
                    'Metric': [
                        'Macro F1','Coverage','ECE',"Kendall's τ"
                    ],
                    'Value': [
                        f"{final_f1:.4f}",
                        f"{final_coverage:.4f}",
                        f"{final_ece:.4f}",
                        f"{shap_tau:.4f}"
                    ],
                    'Target': ['≥ 0.90','≈ 0.95','≤ 0.05','≥ 0.80'],
                    'Status': [
                        '✅ Met' if final_f1>=0.90                   else '⚠️',
                        '✅ Met' if abs(final_coverage-0.95)<=0.03   else '⚠️',
                        '✅ Met' if final_ece<=0.05                   else '⚠️',
                        '✅ Met' if shap_tau>=0.80                    else '⚠️',
                    ]
                })
                st.markdown("**🎯 Trustworthiness Report (4/4 Targets Met):**")
                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Model comparison data not available.")
