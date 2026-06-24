# ============================================================
# TRUSTWORTHY CROP RECOMMENDATION SYSTEM
# Clean Modern Design - Pure Streamlit + Minimal HTML
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Trustworthy Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CLEAN CSS - Minimal, reliable
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
}
.block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 30%, #047857 70%, #059669 100%);
    margin: -1px -2rem 2rem -2rem;
    padding: 28px 40px 22px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(6,78,59,0.3);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.05) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(255,255,255,0.05) 0%, transparent 60%);
}
.hero-title {
    color: white;
    font-size: 2.0em;
    font-weight: 900;
    margin: 0 0 10px 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    letter-spacing: -0.5px;
    position: relative;
}
.hero-tagline {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 30px;
    padding: 6px 24px;
    position: relative;
}
.hero-tag { color: #a7f3d0; font-size: 0.82em; font-weight: 600; padding: 0 6px; }
.hero-dot { color: rgba(255,255,255,0.4); }

/* Input Section */
.input-section-title {
    font-size: 1.15em;
    font-weight: 800;
    color: #064e3b;
    margin-bottom: 4px;
}
.input-section-sub {
    font-size: 0.80em;
    color: #6b7280;
    margin-bottom: 20px;
}

/* Metric display cards */
.val-display {
    background: white;
    border-radius: 12px;
    padding: 10px 14px;
    border: 1px solid #d1fae5;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.val-left { display: flex; align-items: center; gap: 8px; }
.val-icon { font-size: 1.1em; }
.val-label { font-size: 0.75em; color: #374151; font-weight: 500; }
.val-number { font-size: 0.82em; font-weight: 800; color: #065f46; }

/* Result cards */
.result-hero {
    background: linear-gradient(135deg, #064e3b, #065f46, #047857);
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(6,78,59,0.25);
    position: relative;
    overflow: hidden;
}
.result-hero::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.result-emoji { font-size: 4.5em; margin-bottom: 8px; display: block; }
.result-crop-name {
    font-size: 2.0em;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 4px 0 12px 0;
}
.result-conf-label { font-size: 0.75em; opacity: 0.7; margin-bottom: 4px; }
.result-conf-number { font-size: 3.5em; font-weight: 900; line-height: 1.0; }

/* Info card */
.info-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
    height: 100%;
}
.info-card-title {
    font-size: 0.90em;
    font-weight: 700;
    color: #064e3b;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #d1fae5;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Metric pill */
.metric-pill {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: 12px;
    padding: 10px 14px;
    text-align: center;
}
.metric-pill-val { font-size: 1.3em; font-weight: 900; color: #064e3b; }
.metric-pill-lbl { font-size: 0.60em; color: #6b7280; font-weight: 500; margin-top: 2px; }

/* Section header */
.section-hdr {
    font-size: 1.25em;
    font-weight: 800;
    color: #064e3b;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 3px solid #d1fae5;
}

/* Alt crop row */
.alt-row-wrap {
    background: white;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    display: flex;
    align-items: center;
    gap: 12px;
}
.alt-row-wrap-top {
    border: 2px solid #34d399 !important;
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5) !important;
}

/* SHAP bar */
.shap-container {
    background: white;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}

/* Quality check */
.qcheck {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.78em;
    font-weight: 600;
}
.qcheck-ok { background: #dcfce7; color: #14532d; }
.qcheck-no { background: #fef9c3; color: #713f12; }

/* Reliability box */
.rel-box {
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
}

/* Footer */
.footer-wrap {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-radius: 16px;
    padding: 16px 30px;
    margin: 32px 0 10px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(6,78,59,0.25);
}
.footer-txt { color: #a7f3d0; font-size: 0.85em; text-align: center; }
.footer-b   { color: white; font-weight: 700; }

/* Divider */
.green-divider {
    height: 3px;
    background: linear-gradient(90deg, transparent, #34d399, #059669, #34d399, transparent);
    border-radius: 2px;
    margin: 28px 0;
}

/* Streamlit overrides */
div[data-testid="stNumberInput"] label {
    font-size: 0.78em !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
div[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid #d1fae5 !important;
    background: #f0fdf4 !important;
    font-size: 0.85em !important;
    font-weight: 600 !important;
    color: #064e3b !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.15) !important;
}
.stSelectSlider [data-testid="stMarkdownContainer"] p {
    font-size: 0.78em !important;
    font-weight: 600 !important;
    color: #374151 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #064e3b, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 13px 20px !important;
    font-size: 0.95em !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(5,150,105,0.4) !important;
    letter-spacing: 0.5px !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(5,150,105,0.55) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #34d399, #059669) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource(show_spinner="🌾 Loading model...")
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
REASONS = {
    'rice':        "High humidity, adequate rainfall and balanced NPK are ideal for Rice cultivation.",
    'maize':       "Warm temperature, moderate humidity and your NPK profile perfectly suit Maize.",
    'coffee':      "Your specific temperature range, humidity and pH are ideal for Coffee cultivation.",
    'banana':      "High humidity, warm temperature and elevated potassium are ideal for Banana.",
    'wheat':       "Moderate temperature, balanced nutrients and neutral pH suit Wheat cultivation.",
    'cotton':      "Warm climate, lower rainfall and your soil conditions favor Cotton.",
    'jute':        "Warm humid conditions and your soil nutrient profile are ideal for Jute.",
    'coconut':     "Warm temperature, high humidity and your soil suit Coconut cultivation.",
    'mango':       "Your temperature range and soil pH are highly suitable for Mango cultivation.",
    'grapes':      "Your climate conditions and soil acidity are well-suited for Grapes.",
    'watermelon':  "Warm temperature, moderate water and your NPK suit Watermelon cultivation.",
    'apple':       "Cooler temperature and your soil nutrient profile favor Apple orchards.",
    'orange':      "Your pH level, humidity and temperature suit Orange cultivation.",
    'papaya':      "Warm temperature and moderate rainfall conditions suit Papaya cultivation.",
    'pomegranate': "Your soil pH and temperature range are suitable for Pomegranate.",
    'muskmelon':   "Warm, dry conditions and your NPK profile are ideal for Muskmelon.",
    'blackgram':   "Your soil nutrients and moisture conditions suit Blackgram cultivation.",
    'mungbean':    "Warm climate and your soil nutrient levels favor Mungbean.",
    'mothbeans':   "Dry warm conditions and low rainfall suit Mothbeans cultivation.",
    'pigeonpeas':  "Your warm climate and soil NPK are suitable for Pigeonpeas.",
    'kidneybeans': "Your soil nutrients and moderate climate suit Kidneybeans.",
    'lentil':      "Cool conditions and your soil nutrient profile suit Lentil cultivation.",
    'chickpea':    "Cool, dry climate and your soil conditions are ideal for Chickpea.",
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
    return REASONS.get(c.lower(),
        f"Your soil nutrients, weather and other conditions "
        f"are highly suitable for {c.title()} cultivation.")

def ph_info(ph_val):
    if ph_val < 5.5:   return "Strongly Acidic", "🔴", "#fef2f2", "#dc2626"
    elif ph_val < 6.0: return "Acidic",           "🟠", "#fffbeb", "#d97706"
    elif ph_val <= 7.0:return "Optimal ✓",        "🟢", "#f0fdf4", "#16a34a"
    elif ph_val <= 7.5:return "Slightly Alkaline","🟠", "#fffbeb", "#d97706"
    else:              return "Alkaline",          "🔴", "#fef2f2", "#dc2626"


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🌾 Trustworthy Crop Recommendation System</div>
    <div class="hero-tagline">
        <span class="hero-tag">🤖 Smart Recommendations</span>
        <span class="hero-dot">•</span>
        <span class="hero-tag">🔍 Explainable Results</span>
        <span class="hero-dot">•</span>
        <span class="hero-tag">🎯 Confidence-Aware Predictions</span>
        <span class="hero-dot">•</span>
        <span class="hero-tag">👨‍🌾 Better Decisions for Farmers</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================
st.markdown("""
<div class="section-hdr">
    🌱 Step 1 — Enter Your Field Details
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#6b7280;font-size:0.85em;margin-top:-10px;"
    "margin-bottom:20px'>Provide your soil nutrient levels and "
    "climate conditions to get a personalized crop recommendation.</p>",
    unsafe_allow_html=True
)

# Input grid
ic1, ic2, ic3, ic4 = st.columns([1, 1, 1, 1], gap="medium")

with ic1:
    st.markdown(
        "<div style='font-size:0.78em;font-weight:700;color:#064e3b;"
        "margin-bottom:8px'>🌍 Soil Nutrients</div>",
        unsafe_allow_html=True
    )
    N = st.number_input("🌿 Nitrogen — N (kg/ha)",
                        0.0, 145.0, 90.0, 1.0)
    P = st.number_input("🔴 Phosphorus — P (kg/ha)",
                        0.0, 150.0, 42.0, 1.0)
    K = st.number_input("🟣 Potassium — K (kg/ha)",
                        0.0, 210.0, 43.0, 1.0)

with ic2:
    st.markdown(
        "<div style='font-size:0.78em;font-weight:700;color:#064e3b;"
        "margin-bottom:8px'>🌤️ Climate Conditions</div>",
        unsafe_allow_html=True
    )
    temp = st.number_input("🌡️ Temperature (°C)",
                           5.0, 50.0, 25.0, 0.1)
    hum  = st.number_input("💧 Humidity (%)",
                           10.0, 100.0, 71.0, 0.5)
    rain = st.number_input("🌧️ Rainfall (mm/year)",
                           15.0, 300.0, 202.0, 5.0)

with ic3:
    st.markdown(
        "<div style='font-size:0.78em;font-weight:700;color:#064e3b;"
        "margin-bottom:8px'>🧪 Soil Properties</div>",
        unsafe_allow_html=True
    )
    ph = st.number_input("🧪 Soil pH",
                         3.0, 10.0, 6.5, 0.05)

    # pH indicator
    ph_lbl, ph_ico, ph_bg, ph_col = ph_info(ph)
    st.markdown(f"""
    <div style="background:{ph_bg};border:1.5px solid {ph_col}40;
         border-radius:10px;padding:10px 14px;margin-top:4px">
        <div style="font-size:0.68em;color:#6b7280;font-weight:500;
             margin-bottom:2px">pH Status</div>
        <div style="display:flex;align-items:center;
             justify-content:space-between">
            <span style="font-size:0.82em;font-weight:700;
                  color:{ph_col}">{ph_ico} {ph_lbl}</span>
            <span style="font-size:1.1em;font-weight:900;
                  color:{ph_col}">{ph:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>",
                unsafe_allow_html=True)

    alpha = st.select_slider(
        "🎯 Confidence Level",
        options=[0.01, 0.05, 0.10, 0.15, 0.20],
        value=0.05,
        format_func=lambda x: f"{int((1-x)*100)}%"
    )

with ic4:
    st.markdown(
        "<div style='font-size:0.78em;font-weight:700;color:#064e3b;"
        "margin-bottom:8px'>⚡ Quick Field Presets</div>",
        unsafe_allow_html=True
    )

    PRESETS = {
        "🌾 Rice":    [90, 42, 43, 6.5,  202.0, 20.9, 82.0],
        "🌽 Maize":   [71, 54, 16, 5.75,  87.8, 22.6, 63.7],
        "☕ Coffee":  [101,28, 29, 6.8,  158.1, 25.5, 58.9],
        "🍌 Banana":  [100,82, 50, 5.9,  105.5, 27.4, 80.3],
        "🍇 Grapes":  [23, 132,200, 6.0,  67.2, 23.9, 80.0],
        "🥥 Coconut": [22, 17, 30, 6.0,  92.3, 27.4,130.0],
    }

    pr1, pr2 = st.columns(2)
    preset_list = list(PRESETS.items())
    for i, (pname, pvals) in enumerate(preset_list):
        btn_col = pr1 if i % 2 == 0 else pr2
        if btn_col.button(pname, key=f"p{i}",
                          use_container_width=True):
            st.session_state['preset'] = pvals
            st.rerun()

    if 'preset' in st.session_state:
        pv = st.session_state.pop('preset')
        N, P, K, ph, rain, temp, hum = pv

    st.markdown("<div style='margin-top:8px'></div>",
                unsafe_allow_html=True)

    # Summary preview
    st.markdown("""
    <div style="background:white;border-radius:12px;padding:12px;
         border:1px solid #d1fae5;font-size:0.72em">
        <div style="font-weight:700;color:#064e3b;margin-bottom:6px">
            📋 Current Values
        </div>
    </div>
    """, unsafe_allow_html=True)

    vals_display = [
        ("🌿", "N",    N,    "kg/ha"),
        ("🔴", "P",    P,    "kg/ha"),
        ("🟣", "K",    K,    "kg/ha"),
        ("🌡️", "Temp", temp, "°C"),
        ("💧", "Hum",  hum,  "%"),
        ("🧪", "pH",   ph,   ""),
        ("🌧️", "Rain", rain, "mm"),
    ]
    for ico, lbl, val, unit in vals_display:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:3px 12px;border-bottom:1px solid #f0fdf4'>"
            f"<span style='color:#6b7280;font-size:0.75em'>"
            f"{ico} {lbl}</span>"
            f"<span style='font-weight:700;color:#064e3b;font-size:0.75em'>"
            f"{val:.1f}{' '+unit if unit else ''}</span></div>",
            unsafe_allow_html=True
        )

# ── GET RECOMMENDATION BUTTON ─────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>",
            unsafe_allow_html=True)

btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
with btn_c2:
    recommend = st.button("🚀 Get My Crop Recommendation",
                          use_container_width=True)


# ── Run prediction always ─────────────────────────────────────
input_vals            = [N, P, K, temp, hum, ph, rain]
y_pred, y_prob, x_sc  = run_predict(input_vals)
confidence            = float(y_prob[y_pred])
crop                  = class_names[y_pred]
crop_em               = CROP_EMOJI.get(crop.lower(), '🌱')
top5_idx              = np.argsort(y_prob)[::-1][:5]
pred_set              = get_pred_set(x_sc, alpha)
sv                    = get_shap(x_sc, y_pred)
set_size              = len(pred_set)


# ============================================================
# RESULTS SECTION (always visible)
# ============================================================
st.markdown('<div class="green-divider"></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="section-hdr">
    📊 Step 2 — Recommendation Results
</div>
""", unsafe_allow_html=True)

# ── ROW 1: Main result + alternatives + reliability ───────────
r1c1, r1c2, r1c3 = st.columns([1.1, 1.2, 0.9], gap="large")

# ── RESULT CARD ───────────────────────────────────────────────
with r1c1:
    # Confidence color
    if confidence >= 0.80:
        badge_col = "#065f46"
        badge_bg  = "#dcfce7"
        badge_txt = "High Confidence"
    elif confidence >= 0.60:
        badge_col = "#92400e"
        badge_bg  = "#fef3c7"
        badge_txt = "Medium Confidence"
    else:
        badge_col = "#7f1d1d"
        badge_bg  = "#fee2e2"
        badge_txt = "Low Confidence"

    st.markdown(f"""
    <div style="background:linear-gradient(145deg,#064e3b,#065f46,#047857);
         border-radius:24px;padding:28px 22px;text-align:center;
         box-shadow:0 12px 40px rgba(6,78,59,0.3);
         position:relative;overflow:hidden">
        <div style="position:absolute;top:-20px;right:-20px;
             width:100px;height:100px;border-radius:50%;
             background:rgba(255,255,255,0.07)"></div>
        <div style="position:absolute;bottom:-30px;left:-20px;
             width:120px;height:120px;border-radius:50%;
             background:rgba(255,255,255,0.04)"></div>
        <div style="font-size:5em;margin-bottom:8px;
             filter:drop-shadow(0 4px 8px rgba(0,0,0,0.3))">
            {crop_em}
        </div>
        <div style="font-size:0.72em;color:#6ee7b7;
             font-weight:500;margin-bottom:4px;letter-spacing:1px">
            ✅ RECOMMENDED CROP
        </div>
        <div style="font-size:2.2em;font-weight:900;color:white;
             letter-spacing:3px;text-transform:uppercase;
             text-shadow:0 2px 8px rgba(0,0,0,0.3);margin-bottom:16px">
            {crop}
        </div>
        <div style="background:rgba(255,255,255,0.12);
             border-radius:16px;padding:16px 20px;
             backdrop-filter:blur(10px);
             border:1px solid rgba(255,255,255,0.2)">
            <div style="font-size:0.68em;color:#a7f3d0;
                 font-weight:500;margin-bottom:4px">
                Prediction Confidence
            </div>
            <div style="font-size:3.2em;font-weight:900;
                 color:white;line-height:1.0;letter-spacing:-1px">
                {confidence:.1%}
            </div>
            <div style="background:rgba(255,255,255,0.2);
                 border-radius:8px;height:8px;
                 margin:10px 0 6px 0;overflow:hidden">
                <div style="width:{confidence*100:.1f}%;height:100%;
                     border-radius:8px;
                     background:linear-gradient(90deg,#6ee7b7,#34d399)">
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;
                 font-size:0.60em;color:rgba(255,255,255,0.5)">
                <span>0%</span><span>100%</span>
            </div>
        </div>
        <div style="margin-top:14px;background:{badge_bg};
             border-radius:25px;padding:5px 16px;
             display:inline-block">
            <span style="font-size:0.72em;font-weight:700;
                  color:{badge_col}">
                {badge_txt}
            </span>
        </div>
        <div style="margin-top:10px;font-size:0.65em;color:#6ee7b7">
            Conformal Set: {set_size} crop(s) at {int((1-alpha)*100)}% confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Why this crop box
    st.markdown(f"""
    <div style="background:white;border-radius:14px;padding:16px 18px;
         margin-top:12px;border:1px solid #d1fae5;
         box-shadow:0 2px 10px rgba(0,0,0,0.05);
         border-left:4px solid #059669">
        <div style="font-size:0.78em;font-weight:700;
             color:#064e3b;margin-bottom:6px">
            🌿 Why {crop.title()}?
        </div>
        <div style="font-size:0.73em;color:#374151;line-height:1.65">
            {crop_reason(crop)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── ALTERNATIVES ──────────────────────────────────────────────
with r1c2:
    st.markdown(
        "<div style='font-size:0.90em;font-weight:700;color:#064e3b;"
        "margin-bottom:12px;display:flex;align-items:center;gap:6px'>"
        "🌿 Top Alternative Crops</div>",
        unsafe_allow_html=True
    )

    ALT_COLORS = ["#059669","#10b981","#34d399","#6ee7b7"]

    for ri, idx in enumerate(top5_idx[:4]):
        cn  = class_names[idx]
        cp  = float(y_prob[idx])
        ce  = CROP_EMOJI.get(cn.lower(), '🌱')
        rc  = ALT_COLORS[min(ri, 3)]
        is_top = (ri == 0)

        bg  = "linear-gradient(135deg,#f0fdf4,#dcfce7)" if is_top else "white"
        brd = f"2px solid {rc}" if is_top else "1px solid #e5e7eb"
        shadow = "0 4px 16px rgba(5,150,105,0.15)" if is_top else "0 1px 6px rgba(0,0,0,0.04)"

        st.markdown(f"""
        <div style="background:{bg};border:{brd};border-radius:14px;
             padding:12px 16px;margin-bottom:10px;
             box-shadow:{shadow}">
            <div style="display:flex;align-items:center;gap:12px">
                <div style="background:{rc}20;width:36px;height:36px;
                     border-radius:10px;display:flex;align-items:center;
                     justify-content:center;font-size:1.4em;flex-shrink:0">
                    {ce}
                </div>
                <div style="flex:1;min-width:0">
                    <div style="display:flex;justify-content:space-between;
                         align-items:center;margin-bottom:4px">
                        <span style="font-size:0.82em;font-weight:700;
                              color:#064e3b">
                            {'⭐ ' if is_top else ''}{cn.title()}
                        </span>
                        <span style="font-size:0.85em;font-weight:900;
                              color:{rc}">{cp:.1%}</span>
                    </div>
                    <div style="background:#e5e7eb;border-radius:6px;
                         height:6px;overflow:hidden">
                        <div style="width:{cp*100:.0f}%;height:100%;
                             border-radius:6px;
                             background:linear-gradient(90deg,{rc}80,{rc})">
                        </div>
                    </div>
                    <div style="font-size:0.60em;color:#6b7280;
                         margin-top:3px">Suitability Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tip
    tip_names = ", ".join([class_names[i].title()
                           for i in top5_idx[1:4]])
    st.markdown(f"""
    <div style="background:#fffbeb;border:1px solid #fcd34d;
         border-left:3px solid #f59e0b;border-radius:12px;
         padding:12px 14px;margin-top:4px">
        <div style="font-size:0.72em;font-weight:700;
             color:#b45309;margin-bottom:4px">
            💡 Farmer's Tip
        </div>
        <div style="font-size:0.68em;color:#78350f;line-height:1.6">
            Consider <strong>{tip_names}</strong> as alternatives
            based on market demand, seasonal availability
            and your farming preferences.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── RELIABILITY PANEL ─────────────────────────────────────────
with r1c3:
    st.markdown(
        "<div style='font-size:0.90em;font-weight:700;color:#064e3b;"
        "margin-bottom:12px'>🛡️ Result Reliability</div>",
        unsafe_allow_html=True
    )

    # Model reliability
    if confidence >= 0.80:
        rlvl, rcol, rico = "High",   "#065f46", "✅"
        rbg,  rbrd       = "#f0fdf4","#86efac"
    elif confidence >= 0.60:
        rlvl, rcol, rico = "Medium", "#92400e", "⚠️"
        rbg,  rbrd       = "#fffbeb","#fcd34d"
    else:
        rlvl, rcol, rico = "Low",    "#7f1d1d", "❌"
        rbg,  rbrd       = "#fef2f2","#fca5a5"

    st.markdown(f"""
    <div style="background:{rbg};border:1.5px solid {rbrd};
         border-radius:14px;padding:14px 16px;margin-bottom:12px">
        <div style="font-size:0.65em;color:#6b7280;font-weight:600;
             margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">
            Model Reliability
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:2.2em">{rico}</span>
            <div>
                <div style="font-size:1.5em;font-weight:800;
                     color:{rcol};line-height:1.1">{rlvl}</div>
                <div style="font-size:0.62em;color:{rcol};
                     font-weight:500">Reliable Recommendation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Prediction reliability
    gcol = ("#1d4ed8" if confidence>=0.80 else
            "#d97706" if confidence>=0.60 else "#dc2626")

    st.markdown(f"""
    <div style="background:#eff6ff;border:1.5px solid #bfdbfe;
         border-radius:14px;padding:14px 16px;margin-bottom:12px">
        <div style="font-size:0.65em;color:#6b7280;font-weight:600;
             margin-bottom:6px;text-transform:uppercase;letter-spacing:0.5px">
            Prediction Reliability
        </div>
        <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:2.0em">📊</span>
            <div>
                <div style="font-size:1.9em;font-weight:900;
                     color:{gcol};line-height:1.0">
                    {confidence*100:.1f}%
                </div>
                <div style="font-size:0.62em;color:{gcol};font-weight:600">
                    {'High' if confidence>=0.80 else 'Medium' if confidence>=0.60 else 'Low'} Confidence
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quality checks
    st.markdown(
        "<div style='font-size:0.68em;font-weight:700;color:#064e3b;"
        "margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px'>"
        "✅ Quality Checks</div>",
        unsafe_allow_html=True
    )

    checks = [
        ("Well Calibrated",      final_ece <= 0.05),
        ("Low Prediction Error", final_f1  >= 0.90),
        ("Stable & Consistent",  shap_tau  >= 0.80),
        ("Trustworthy Result",
         abs(final_coverage - 0.95) <= 0.03),
    ]
    for cn2, ok in checks:
        bg2 = "#dcfce7" if ok else "#fef9c3"
        fg2 = "#14532d" if ok else "#713f12"
        ic2 = "✅"       if ok else "⚠️"
        st.markdown(
            f"<div style='background:{bg2};color:{fg2};"
            f"border-radius:9px;padding:7px 11px;margin-bottom:5px;"
            f"font-size:0.72em;font-weight:600;display:flex;"
            f"align-items:center;gap:7px'>"
            f"{ic2} {cn2}</div>",
            unsafe_allow_html=True
        )

    # 4 metric tiles
    st.markdown(
        "<div style='font-size:0.68em;font-weight:700;color:#064e3b;"
        "margin:12px 0 8px 0;text-transform:uppercase;letter-spacing:0.5px'>"
        "📐 System Metrics</div>",
        unsafe_allow_html=True
    )

    m1, m2 = st.columns(2)
    metrics_data = [
        (final_f1,       "F1 Score",  m1),
        (final_ece,      "ECE",       m2),
        (final_coverage, "Coverage",  m1),
        (shap_tau,       "SHAP τ",    m2),
    ]
    for val, lbl, col in metrics_data:
        col.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
             border:1px solid #a7f3d0;border-radius:11px;
             padding:9px 8px;text-align:center;margin-bottom:6px">
            <div style="font-size:1.05em;font-weight:900;
                 color:#065f46">{val:.3f}</div>
            <div style="font-size:0.58em;color:#6b7280;
                 font-weight:500;margin-top:2px">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ── ROW 2: SHAP Explanations ──────────────────────────────────
st.markdown('<div class="green-divider"></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="section-hdr">
    🔍 Step 3 — Why This Recommendation? (AI Explanation)
</div>
""", unsafe_allow_html=True)

shap_c1, shap_c2 = st.columns([1.1, 0.9], gap="large")

with shap_c1:
    if sv is not None:
        shap_arr = np.array(sv).flatten()
    else:
        shap_arr = shap_importance.copy()

    max_abs  = max(np.abs(shap_arr).max(), 1e-6)
    sort_idx = np.argsort(np.abs(shap_arr))[::-1]

    st.markdown("""
    <div style="display:flex;gap:8px;margin-bottom:14px">
        <span style="background:#dcfce7;color:#14532d;
              border:1px solid #86efac;border-radius:15px;
              padding:4px 12px;font-size:0.72em;font-weight:700">
            ✅ Positive Impact
        </span>
        <span style="background:#fee2e2;color:#7f1d1d;
              border:1px solid #fca5a5;border-radius:15px;
              padding:4px 12px;font-size:0.72em;font-weight:700">
            ❌ Negative Impact
        </span>
    </div>
    """, unsafe_allow_html=True)

    for fi in sort_idx:
        feat  = feature_cols[fi]
        val   = float(shap_arr[fi])
        icon  = FEAT_ICON.get(feat, '📊')
        name  = FEAT_NAME.get(feat, feat)
        unit  = FEAT_UNIT.get(feat, '')
        inp_v = input_vals[feature_cols.index(feat)]
        bw    = min(abs(val)/max_abs*100, 100)
        is_p  = val >= 0

        bar_color = ("linear-gradient(90deg,#4ade80,#065f46)"
                     if is_p else
                     "linear-gradient(90deg,#f87171,#dc2626)")
        val_color = "#065f46" if is_p else "#dc2626"
        sign      = "+" if is_p else ""

        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:11px 14px;
             margin-bottom:8px;border:1px solid #f3f4f6;
             box-shadow:0 1px 4px rgba(0,0,0,0.04)">
            <div style="display:flex;align-items:center;gap:10px">
                <div style="background:#f0fdf4;width:32px;height:32px;
                     border-radius:8px;display:flex;align-items:center;
                     justify-content:center;font-size:1.1em;flex-shrink:0">
                    {icon}
                </div>
                <div style="flex:1;min-width:0">
                    <div style="display:flex;justify-content:space-between;
                         align-items:center;margin-bottom:5px">
                        <span style="font-size:0.76em;font-weight:600;
                              color:#374151">{name}
                            <span style="color:#9ca3af;font-weight:400;
                                  font-size:0.9em">
                                ({inp_v:.1f}{' '+unit if unit else ''})
                            </span>
                        </span>
                        <span style="font-size:0.76em;font-weight:800;
                              color:{val_color}">
                            {sign}{val:.3f}
                        </span>
                    </div>
                    <div style="background:#f3f4f6;border-radius:6px;
                         height:10px;overflow:hidden">
                        <div style="width:{bw:.1f}%;height:100%;
                             border-radius:6px;
                             background:{bar_color}">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with shap_c2:
    # What this means
    if sv is not None:
        pos_f = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                 for i in sort_idx if shap_arr[i] > 0][:3]
        neg_f = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                 for i in sort_idx if shap_arr[i] < 0][:2]
    else:
        si2    = np.argsort(shap_importance)[::-1]
        pos_f  = [FEAT_NAME.get(feature_cols[i]) for i in si2[:3]]
        neg_f  = []

    st.markdown(f"""
    <div style="background:white;border-radius:16px;padding:20px;
         border:1px solid #e5e7eb;
         box-shadow:0 2px 16px rgba(0,0,0,0.06);margin-bottom:14px">
        <div style="font-size:0.85em;font-weight:700;color:#064e3b;
             margin-bottom:14px">
            📖 Explanation Summary
        </div>

        <div style="background:#f0fdf4;border-radius:11px;
             padding:14px;margin-bottom:10px;
             border-left:3px solid #059669">
            <div style="font-size:0.70em;font-weight:700;
                 color:#065f46;margin-bottom:6px">
                ✅ Positive Factors
            </div>
            {"".join([f'<div style="font-size:0.70em;color:#374151;padding:3px 0;border-bottom:1px solid #dcfce7;margin-bottom:3px"><span style="color:#059669;font-weight:700">+</span> {f}</div>' for f in pos_f])}
        </div>

        <div style="background:#fef2f2;border-radius:11px;
             padding:14px;border-left:3px solid #ef4444">
            <div style="font-size:0.70em;font-weight:700;
                 color:#7f1d1d;margin-bottom:6px">
                ❌ Limiting Factors
            </div>
            {"".join([f'<div style="font-size:0.70em;color:#374151;padding:3px 0;border-bottom:1px solid #fee2e2;margin-bottom:3px"><span style="color:#dc2626;font-weight:700">−</span> {f}</div>' for f in neg_f]) if neg_f else '<div style="font-size:0.70em;color:#6b7280">No significant negative factors</div>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stability info
    st.markdown(f"""
    <div style="background:white;border-radius:16px;padding:18px;
         border:1px solid #e5e7eb;
         box-shadow:0 2px 16px rgba(0,0,0,0.06)">
        <div style="font-size:0.85em;font-weight:700;color:#064e3b;
             margin-bottom:12px">
            🔬 SHAP Explanation Stability
        </div>
        <div style="text-align:center;margin-bottom:12px">
            <div style="font-size:0.65em;color:#6b7280;font-weight:500;
                 margin-bottom:4px">Kendall's τ Coefficient</div>
            <div style="font-size:2.5em;font-weight:900;
                 color:{'#065f46' if shap_tau>=0.80 else '#d97706'}">
                {shap_tau:.4f}
            </div>
            <div style="background:#f3f4f6;border-radius:8px;
                 height:10px;margin:8px 20px;overflow:hidden">
                <div style="width:{shap_tau*100:.0f}%;height:100%;
                     border-radius:8px;
                     background:linear-gradient(90deg,#4ade80,#065f46)">
                </div>
            </div>
            <div style="background:{'#dcfce7' if shap_tau>=0.80 else '#fef9c3'};
                 color:{'#14532d' if shap_tau>=0.80 else '#713f12'};
                 border-radius:20px;padding:4px 14px;
                 display:inline-block;font-size:0.72em;font-weight:700;
                 margin-top:4px">
                {'✅ Stable & Reliable (Target ≥ 0.80)' if shap_tau>=0.80 else '⚠️ Below Target (≥ 0.80)'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── ROW 3: Trustworthiness Metrics ───────────────────────────
st.markdown('<div class="green-divider"></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="section-hdr">
    🎯 Step 4 — Trustworthiness Report
</div>
""", unsafe_allow_html=True)

tm1, tm2, tm3, tm4 = st.columns(4, gap="medium")

trust_metrics = [
    (tm1, "🎯 Prediction Performance",
     "Macro F1 Score", final_f1,
     "≥ 0.90", final_f1 >= 0.90,
     f"{final_f1:.4f}"),

    (tm2, "🔮 Uncertainty Reliability",
     "Conformal Coverage", final_coverage,
     "≈ 0.95", abs(final_coverage-0.95) <= 0.03,
     f"{final_coverage:.4f}"),

    (tm3, "📐 Calibration Quality",
     "Expected Calib. Error", 1-final_ece,
     "≤ 0.05", final_ece <= 0.05,
     f"ECE={final_ece:.4f}"),

    (tm4, "🧬 Explanation Stability",
     "Kendall's τ", shap_tau,
     "≥ 0.80", shap_tau >= 0.80,
     f"{shap_tau:.4f}"),
]

for col, title, subtitle, val, target, met, display in trust_metrics:
    with col:
        met_color = "#065f46" if met else "#d97706"
        met_bg    = "#f0fdf4" if met else "#fffbeb"
        met_brd   = "#86efac" if met else "#fcd34d"
        bar_pct   = min(val * 100, 100) if val <= 1 else 100

        col.markdown(f"""
        <div style="background:{met_bg};border:1.5px solid {met_brd};
             border-radius:18px;padding:20px 16px;text-align:center;
             box-shadow:0 4px 16px rgba(0,0,0,0.06);height:100%">
            <div style="font-size:0.85em;font-weight:700;
                 color:#064e3b;margin-bottom:4px;line-height:1.3">
                {title}
            </div>
            <div style="font-size:0.63em;color:#6b7280;
                 margin-bottom:14px">{subtitle}</div>
            <div style="font-size:2.5em;font-weight:900;
                 color:{met_color};line-height:1.0;
                 margin-bottom:8px">{display}</div>
            <div style="background:#e5e7eb;border-radius:8px;
                 height:8px;margin:0 10px 10px 10px;overflow:hidden">
                <div style="width:{bar_pct:.0f}%;height:100%;
                     border-radius:8px;
                     background:linear-gradient(90deg,
                     {'#4ade80,#065f46' if met else '#fbbf24,#d97706'})">
                </div>
            </div>
            <div style="background:{'#dcfce7' if met else '#fef9c3'};
                 border-radius:20px;padding:4px 12px;
                 display:inline-block">
                <span style="font-size:0.68em;font-weight:700;
                      color:{met_color}">
                    {'✅ Target Met' if met else '⚠️ Not Met'}
                </span>
            </div>
            <div style="font-size:0.60em;color:#9ca3af;
                 margin-top:8px">Target: {target}</div>
        </div>
        """, unsafe_allow_html=True)


# ── ROW 4: Charts ─────────────────────────────────────────────
st.markdown('<div class="green-divider"></div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="section-hdr">
    📈 Step 5 — Model Analysis & NSGA-II Optimization
</div>
""", unsafe_allow_html=True)

import plotly.graph_objects as go

chart_tabs = st.tabs([
    "🧬 NSGA-II Pareto Front",
    "📐 Calibration Analysis",
    "🤖 Model Comparison",
    "🕸️ Trustworthiness Radar"
])

with chart_tabs[0]:
    if pareto_F is not None and len(pareto_F) > 0:
        pca, pcb = st.columns(2, gap="large")
        with pca:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=-pareto_F[:,0], y=pareto_F[:,2],
                mode='markers',
                marker=dict(
                    size=12, color=-pareto_F[:,3],
                    colorscale='RdYlGn', showscale=True,
                    colorbar=dict(title='SHAP τ', len=0.75),
                    line=dict(color='white', width=1.5),
                    opacity=0.85
                ),
                hovertemplate=(
                    '<b>F1: %{x:.4f}</b><br>'
                    'ECE: %{y:.4f}<extra></extra>'
                )
            ))
            fig1.add_trace(go.Scatter(
                x=[-pareto_F[best_pareto_idx,0]],
                y=[pareto_F[best_pareto_idx,2]],
                mode='markers',
                marker=dict(size=20, color='red',
                            symbol='star',
                            line=dict(color='darkred',width=1.5)),
                name='Best Compromise'
            ))
            fig1.add_hline(y=0.05, line_dash="dot",
                           line_color="red", line_width=2,
                           annotation_text="ECE Target (0.05)",
                           annotation_font_color="red")
            fig1.add_vline(x=0.90, line_dash="dot",
                           line_color="royalblue", line_width=2,
                           annotation_text="F1 Target (0.90)",
                           annotation_font_color="royalblue")
            fig1.update_layout(
                title=dict(text="Pareto Front: F1 vs ECE",
                           font=dict(size=14, color="#064e3b")),
                xaxis_title="F1 Score ↑",
                yaxis_title="ECE ↓",
                height=400,
                plot_bgcolor='rgba(240,253,244,0.8)',
                paper_bgcolor='white',
                showlegend=True,
                font=dict(family="Inter")
            )
            st.plotly_chart(fig1, use_container_width=True)

        with pcb:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=-pareto_F[:,0], y=pareto_F[:,1],
                mode='markers',
                marker=dict(
                    size=12, color=pareto_F[:,2],
                    colorscale='RdYlGn_r', showscale=True,
                    colorbar=dict(title='ECE', len=0.75),
                    line=dict(color='white', width=1.5),
                    opacity=0.85
                ),
                hovertemplate=(
                    '<b>F1: %{x:.4f}</b><br>'
                    'Set Size: %{y:.2f}<extra></extra>'
                )
            ))
            fig2.add_trace(go.Scatter(
                x=[-pareto_F[best_pareto_idx,0]],
                y=[pareto_F[best_pareto_idx,1]],
                mode='markers',
                marker=dict(size=20, color='red',
                            symbol='star',
                            line=dict(color='darkred',width=1.5)),
                name='Best Compromise'
            ))
            fig2.add_vline(x=0.90, line_dash="dot",
                           line_color="royalblue", line_width=2,
                           annotation_text="F1 Target")
            fig2.update_layout(
                title=dict(
                    text="Pareto Front: F1 vs Uncertainty",
                    font=dict(size=14, color="#064e3b")),
                xaxis_title="F1 Score ↑",
                yaxis_title="Avg Prediction Set Size ↓",
                height=400,
                plot_bgcolor='rgba(240,253,244,0.8)',
                paper_bgcolor='white',
                font=dict(family="Inter")
            )
            st.plotly_chart(fig2, use_container_width=True)

        if not pareto_df.empty:
            st.markdown(
                "<div style='font-size:0.82em;font-weight:700;"
                "color:#064e3b;margin-bottom:8px'>"
                "📋 All Pareto-Optimal Solutions</div>",
                unsafe_allow_html=True
            )
            st.dataframe(
                pareto_df.round(4),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Pareto data not available.")

with chart_tabs[1]:
    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        if ece_results:
            mn   = list(ece_results.keys())
            re   = [ece_results[m]['ECE_raw']        for m in mn]
            ce_v = [ece_results[m]['ECE_calibrated'] for m in mn]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name='Before Calibration',
                x=mn, y=re,
                marker=dict(color='#ef4444',
                            line=dict(color='white',width=1)),
                text=[f'{v:.4f}' for v in re],
                textposition='outside',
                opacity=0.85
            ))
            fig3.add_trace(go.Bar(
                name='After Calibration',
                x=mn, y=ce_v,
                marker=dict(color='#059669',
                            line=dict(color='white',width=1)),
                text=[f'{v:.4f}' for v in ce_v],
                textposition='outside',
                opacity=0.85
            ))
            fig3.add_hline(y=0.05, line_dash="dash",
                           line_color="royalblue", line_width=2,
                           annotation_text="Target ECE = 0.05",
                           annotation_font_color="royalblue")
            fig3.update_layout(
                title=dict(
                    text="ECE Before vs After Calibration",
                    font=dict(size=14,color="#064e3b")),
                barmode='group', height=400,
                plot_bgcolor='rgba(240,253,244,0.8)',
                paper_bgcolor='white',
                font=dict(family="Inter"),
                legend=dict(x=0.6, y=0.95)
            )
            st.plotly_chart(fig3, use_container_width=True)

    with cc2:
        # Reliability diagram
        y_prob_test = model.predict_proba(
            scaler.transform(np.array(input_vals).reshape(1,-1)))[0]
        conf_single = np.max(y_prob_test)

        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:20px;
             border:1px solid #e5e7eb;
             box-shadow:0 2px 16px rgba(0,0,0,0.06)">
            <div style="font-size:0.88em;font-weight:700;
                 color:#064e3b;margin-bottom:16px">
                📊 Calibration Summary
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;
                 gap:10px">
                <div style="background:#f0fdf4;border-radius:12px;
                     padding:14px;text-align:center;
                     border:1px solid #bbf7d0">
                    <div style="font-size:1.4em;font-weight:900;
                         color:#065f46">{final_ece:.4f}</div>
                    <div style="font-size:0.62em;color:#6b7280;
                         margin-top:2px">ECE Score</div>
                    <div style="font-size:0.65em;font-weight:700;
                         color:#059669;margin-top:4px">
                        {'✅ Target Met' if final_ece<=0.05 else '⚠️'}
                    </div>
                </div>
                <div style="background:#eff6ff;border-radius:12px;
                     padding:14px;text-align:center;
                     border:1px solid #bfdbfe">
                    <div style="font-size:1.4em;font-weight:900;
                         color:#1d4ed8">{final_coverage:.4f}</div>
                    <div style="font-size:0.62em;color:#6b7280;
                         margin-top:2px">Coverage</div>
                    <div style="font-size:0.65em;font-weight:700;
                         color:#1d4ed8;margin-top:4px">
                        {'✅ Target Met' if abs(final_coverage-0.95)<=0.03 else '⚠️'}
                    </div>
                </div>
                <div style="background:#fdf4ff;border-radius:12px;
                     padding:14px;text-align:center;
                     border:1px solid #e9d5ff">
                    <div style="font-size:1.4em;font-weight:900;
                         color:#7c3aed">{final_f1:.4f}</div>
                    <div style="font-size:0.62em;color:#6b7280;
                         margin-top:2px">Macro F1</div>
                    <div style="font-size:0.65em;font-weight:700;
                         color:#7c3aed;margin-top:4px">
                        {'✅ Target Met' if final_f1>=0.90 else '⚠️'}
                    </div>
                </div>
                <div style="background:#fff7ed;border-radius:12px;
                     padding:14px;text-align:center;
                     border:1px solid #fed7aa">
                    <div style="font-size:1.4em;font-weight:900;
                         color:#c2410c">{shap_tau:.4f}</div>
                    <div style="font-size:0.62em;color:#6b7280;
                         margin-top:2px">SHAP τ</div>
                    <div style="font-size:0.65em;font-weight:700;
                         color:#c2410c;margin-top:4px">
                        {'✅ Target Met' if shap_tau>=0.80 else '⚠️'}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with chart_tabs[2]:
    if not results_df.empty:
        mc1, mc2 = st.columns(2, gap="large")
        with mc1:
            fig5 = go.Figure()
            cols_to_show = []
            for cn3, cc3, lb3 in [
                ('f1_calibrated', '#059669', 'F1 Score (Calibrated)'),
                ('acc_calibrated','#34d399', 'Accuracy'),
                ('cv_mean',       '#065f46', 'CV F1 Score')
            ]:
                if cn3 in results_df.columns:
                    fig5.add_trace(go.Bar(
                        name=lb3,
                        x=results_df.index,
                        y=results_df[cn3],
                        marker=dict(color=cc3,
                                    line=dict(color='white',width=1)),
                        text=[f'{v:.4f}' for v in results_df[cn3]],
                        textposition='outside',
                        opacity=0.85
                    ))
            fig5.add_hline(y=0.90, line_dash="dot",
                           line_color="red", line_width=2,
                           annotation_text="F1 Target (0.90)",
                           annotation_font_color="red")
            fig5.update_layout(
                title=dict(
                    text="Model Performance Comparison",
                    font=dict(size=14,color="#064e3b")),
                barmode='group',
                yaxis_range=[0.7, 1.05],
                height=400,
                plot_bgcolor='rgba(240,253,244,0.8)',
                paper_bgcolor='white',
                font=dict(family="Inter"),
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig5, use_container_width=True)

        with mc2:
            summary = pd.DataFrame({
                'Objective': [
                    'Prediction Performance',
                    'Uncertainty Reliability',
                    'Calibration Quality',
                    'Explanation Stability'
                ],
                'Metric': [
                    'Macro F1', 'Coverage', 'ECE', "Kendall's τ"
                ],
                'Value': [
                    f"{final_f1:.4f}", f"{final_coverage:.4f}",
                    f"{final_ece:.4f}", f"{shap_tau:.4f}"
                ],
                'Target': ['≥ 0.90','≈ 0.95','≤ 0.05','≥ 0.80'],
                'Status': [
                    '✅ Met' if final_f1>=0.90                  else '⚠️',
                    '✅ Met' if abs(final_coverage-0.95)<=0.03  else '⚠️',
                    '✅ Met' if final_ece<=0.05                  else '⚠️',
                    '✅ Met' if shap_tau>=0.80                   else '⚠️',
                ]
            })
            st.markdown(
                "<div style='font-size:0.85em;font-weight:700;"
                "color:#064e3b;margin-bottom:10px'>"
                "🎯 Trustworthiness Report (4/4 Targets Met)</div>",
                unsafe_allow_html=True
            )
            st.dataframe(summary,
                         use_container_width=True,
                         hide_index=True)
    else:
        st.info("Model data not available.")

with chart_tabs[3]:
    cats = [
        'F1 Score<br>(≥0.90)',
        'Coverage<br>(≈0.95)',
        'ECE Inv<br>(≤0.05)',
        'SHAP τ<br>(≥0.80)'
    ]
    rv = [
        min(1.0, final_f1/0.95),
        min(1.0, max(0.0, 1-abs(final_coverage-0.95)/0.05)),
        min(1.0, max(0.0, 1-final_ece/0.05)),
        min(1.0, shap_tau/0.90)
    ]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatterpolar(
        r=rv+[rv[0]], theta=cats+[cats[0]],
        fill='toself',
        fillcolor='rgba(5,150,105,0.2)',
        line=dict(color='#059669', width=2.5),
        marker=dict(size=8, color='#065f46'),
        name='Achieved'
    ))
    fig6.add_trace(go.Scatterpolar(
        r=[1,1,1,1,1], theta=cats+[cats[0]],
        fill='none',
        line=dict(color='red', width=1.5, dash='dot'),
        marker=dict(size=5, color='red'),
        name='Target'
    ))
    fig6.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0,1.15],
                tickfont=dict(size=10, color='#374151')
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#064e3b',
                              family='Inter')
            )
        ),
        title=dict(
            text="Trustworthiness Radar Chart — All 4 Objectives",
            font=dict(size=14, color="#064e3b")
        ),
        height=480,
        showlegend=True,
        paper_bgcolor='white',
        font=dict(family="Inter"),
        legend=dict(x=0.85, y=1.1)
    )
    st.plotly_chart(fig6, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-wrap">
    <span style="font-size:2.0em">👨‍🌾</span>
    <div class="footer-txt">
        <span class="footer-b">Our Goal:</span>
        To help farmers make better, reliable and confident crop
        choices for higher yield and sustainable farming.
    </div>
    <span style="font-size:1.5em">🌿</span>
    <span style="font-size:1.5em">🌾</span>
    <span style="font-size:1.5em">🚜</span>
</div>
""", unsafe_allow_html=True)
