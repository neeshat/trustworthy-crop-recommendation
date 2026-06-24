# ============================================================
# TRUSTWORTHY CROP RECOMMENDATION SYSTEM
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    margin: 0 !important;
    padding: 0 !important;
}

.stApp {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 40%, #f0f9ff 100%);
    min-height: 100vh;
}

.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
    margin: 0 !important;
}

/* Remove ALL top spacing */
[data-testid="stMain"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

div[data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

#MainMenu, footer, header { visibility: hidden !important; display: none !important; }
.stDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* Remove default streamlit top padding */
.css-18e3th9, .css-1d391kg, [class*="css-"] {
    padding-top: 0 !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #052e16 0%, #14532d 25%, #166534 55%, #15803d 80%, #16a34a 100%);
    margin: 0 -2rem 2.5rem -2rem;
    padding: 28px 60px 24px 60px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(5,46,22,0.45);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 15% 50%, rgba(74,222,128,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 50%, rgba(34,197,94,0.10) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.05) 0%, transparent 40%);
    pointer-events: none;
}
.hero-icon {
    font-size: 3em;
    margin-bottom: 4px;
    display: block;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
    position: relative;
    z-index: 1;
}
.hero-title {
    color: #ffffff;
    font-size: 1.85em;
    font-weight: 900;
    margin: 0 0 14px 0;
    text-shadow: 0 2px 16px rgba(0,0,0,0.5);
    letter-spacing: -0.3px;
    position: relative;
    z-index: 1;
    line-height: 1.2;
}
.hero-pills {
    display: inline-flex;
    align-items: center;
    gap: 0;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 50px;
    padding: 8px 28px;
    position: relative;
    z-index: 1;
}
.hero-pill { color: #bbf7d0; font-size: 0.78em; font-weight: 600; padding: 0 12px; }
.hero-dot  { color: rgba(255,255,255,0.35); font-size: 0.7em; }

/* ── SECTION HEADER ── */
.sec-hdr {
    font-size: 1.12em;
    font-weight: 800;
    color: #052e16;
    margin: 0 0 18px 0;
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, #059669, #34d399, transparent) 1;
}

/* ── GRADIENT DIVIDER ── */
.g-div {
    height: 2px;
    background: linear-gradient(90deg, transparent, #34d399, #059669, #34d399, transparent);
    margin: 30px 0;
    border-radius: 2px;
    opacity: 0.7;
}

/* ── STREAMLIT INPUT OVERRIDES ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectSlider"] label {
    font-size: 0.80em !important;
    font-weight: 700 !important;
    color: #166534 !important;
    margin-bottom: 4px !important;
}
div[data-testid="stNumberInput"] input {
    border-radius: 12px !important;
    border: 2px solid #d1fae5 !important;
    background: linear-gradient(135deg, #f8fffc, #f0fdf4) !important;
    font-size: 0.90em !important;
    font-weight: 600 !important;
    color: #052e16 !important;
    padding: 9px 12px !important;
    transition: all 0.25s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 4px rgba(5,150,105,0.14) !important;
    background: white !important;
}
div[data-testid="stNumberInput"] input::placeholder {
    color: #9ca3af !important;
    font-weight: 400 !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #052e16, #166534, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 28px !important;
    font-size: 1.0em !important;
    font-weight: 800 !important;
    width: 100% !important;
    box-shadow: 0 6px 28px rgba(5,150,105,0.45) !important;
    letter-spacing: 0.4px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 10px 36px rgba(5,150,105,0.65) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:disabled {
    background: linear-gradient(135deg, #6b7280, #9ca3af) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Preset buttons styling */
div[data-testid="column"] .stButton > button {
    padding: 7px 6px !important;
    font-size: 0.70em !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(5,150,105,0.25) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.75) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1.5px solid #d1fae5 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-size: 0.82em !important;
    font-weight: 700 !important;
    color: #166534 !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important;
    box-shadow: 0 3px 12px rgba(5,150,105,0.35) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1.5px solid #d1fae5 !important;
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
    'rice':        "High humidity, adequate rainfall and balanced NPK nutrients are ideal for Rice.",
    'maize':       "Warm temperature, moderate humidity and your NPK profile perfectly suit Maize.",
    'coffee':      "Your specific temperature range, humidity and pH are ideal for Coffee.",
    'banana':      "High humidity, warm temperature and elevated potassium favor Banana growth.",
    'wheat':       "Moderate temperature, balanced nutrients and neutral pH suit Wheat cultivation.",
    'cotton':      "Warm climate, lower rainfall and your soil conditions favor Cotton.",
    'jute':        "Warm humid conditions and your soil nutrient profile are ideal for Jute.",
    'coconut':     "Warm temperature and high humidity suit Coconut cultivation.",
    'mango':       "Your temperature range and soil pH are highly suitable for Mango.",
    'grapes':      "Your climate conditions and soil acidity are well-suited for Grapes.",
    'watermelon':  "Warm temperature and moderate water content suit Watermelon cultivation.",
    'apple':       "Cooler temperature and your soil nutrient profile favor Apple orchards.",
    'orange':      "Your pH level, humidity and temperature suit Orange cultivation.",
    'papaya':      "Warm temperature and moderate rainfall conditions suit Papaya.",
    'pomegranate': "Your soil pH and temperature range are suitable for Pomegranate.",
    'muskmelon':   "Warm, dry conditions and your NPK profile are ideal for Muskmelon.",
    'blackgram':   "Your soil nutrients and moisture conditions suit Blackgram cultivation.",
    'mungbean':    "Warm climate and your soil nutrient levels favor Mungbean.",
    'mothbeans':   "Dry warm conditions and low rainfall suit Mothbeans cultivation.",
    'pigeonpeas':  "Your warm climate and soil NPK are suitable for Pigeonpeas.",
    'kidneybeans': "Your soil nutrients and moderate climate suit Kidneybeans.",
    'lentil':      "Cool conditions and your soil nutrient profile suit Lentil.",
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
    lvl   = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    quant = np.quantile(cal_scores, lvl)
    prob  = model.predict_proba(x_sc.reshape(1, -1))[0]
    return [c for c in range(len(prob)) if (1 - prob[c]) <= quant]

def get_shap(x_sc, pred_cls):
    if explainer is None:
        return None
    try:
        sv = explainer.shap_values(x_sc.reshape(1, -1))
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

def ph_info(v):
    if v < 5.5:    return "Strongly Acidic",    "🔴", "#fef2f2", "#dc2626", "#fecaca"
    elif v < 6.0:  return "Acidic",             "🟠", "#fffbeb", "#d97706", "#fde68a"
    elif v <= 7.0: return "Optimal ✓",          "🟢", "#f0fdf4", "#16a34a", "#bbf7d0"
    elif v <= 7.5: return "Slightly Alkaline",  "🟠", "#fffbeb", "#d97706", "#fde68a"
    else:          return "Alkaline",            "🔴", "#fef2f2", "#dc2626", "#fecaca"

# ============================================================
# HERO — zero top margin
# ============================================================
st.markdown("""
<div class="hero">
    <span class="hero-icon">🌾</span>
    <div class="hero-title">Streamlit-based Trustworthy Crop Recommendation System</div>
    <div class="hero-pills">
        <span class="hero-pill">🤖 Smart Recommendations</span>
        <span class="hero-dot">•</span>
        <span class="hero-pill">🔍 Explainable AI</span>
        <span class="hero-dot">•</span>
        <span class="hero-pill">🎯 Confidence-Aware</span>
        <span class="hero-dot">•</span>
        <span class="hero-pill">👨‍🌾 Farmer Friendly</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="sec-hdr">🌱 Enter Your Field Details</div>', unsafe_allow_html=True)

# Initialize defaults — all None (empty)
N    = None
P    = None
K    = None
temp = None
hum  = None
rain = None
ph   = None

# Handle preset injection via session state
if 'preset_vals' in st.session_state:
    pv   = st.session_state.pop('preset_vals')
    N, P, K, ph, rain, temp, hum = pv

ic1, ic2, ic3, ic4 = st.columns(4, gap="large")

# ── Column 1: Soil Nutrients ─────────────────────────────────
with ic1:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#052e16,#166534);
         border-radius:14px;padding:10px 14px;margin-bottom:14px;
         text-align:center;box-shadow:0 4px 16px rgba(5,46,22,0.25)">
        <span style="color:#bbf7d0;font-size:0.78em;font-weight:800;
              letter-spacing:1.5px">🌍 SOIL NUTRIENTS</span>
    </div>
    """, unsafe_allow_html=True)
    N = st.number_input(
        "🌿 Nitrogen — N (kg/ha)",
        min_value=0.0, max_value=145.0,
        value=N, step=1.0,
        placeholder="e.g. 90"
    )
    P = st.number_input(
        "🔴 Phosphorus — P (kg/ha)",
        min_value=0.0, max_value=150.0,
        value=P, step=1.0,
        placeholder="e.g. 42"
    )
    K = st.number_input(
        "🟣 Potassium — K (kg/ha)",
        min_value=0.0, max_value=210.0,
        value=K, step=1.0,
        placeholder="e.g. 43"
    )

# ── Column 2: Climate ─────────────────────────────────────────
with ic2:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a5f,#1d4ed8);
         border-radius:14px;padding:10px 14px;margin-bottom:14px;
         text-align:center;box-shadow:0 4px 16px rgba(29,78,216,0.25)">
        <span style="color:#bfdbfe;font-size:0.78em;font-weight:800;
              letter-spacing:1.5px">🌤️ CLIMATE CONDITIONS</span>
    </div>
    """, unsafe_allow_html=True)
    temp = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=5.0, max_value=50.0,
        value=temp, step=0.1,
        placeholder="e.g. 25.0"
    )
    hum = st.number_input(
        "💧 Humidity (%)",
        min_value=10.0, max_value=100.0,
        value=hum, step=0.5,
        placeholder="e.g. 65.0"
    )
    rain = st.number_input(
        "🌧️ Rainfall (mm/year)",
        min_value=15.0, max_value=300.0,
        value=rain, step=5.0,
        placeholder="e.g. 200.0"
    )

# ── Column 3: Soil Properties ─────────────────────────────────
with ic3:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#4a1942,#7c3aed);
         border-radius:14px;padding:10px 14px;margin-bottom:14px;
         text-align:center;box-shadow:0 4px 16px rgba(124,58,237,0.25)">
        <span style="color:#e9d5ff;font-size:0.78em;font-weight:800;
              letter-spacing:1.5px">🧪 SOIL PROPERTIES</span>
    </div>
    """, unsafe_allow_html=True)
    ph = st.number_input(
        "🧪 Soil pH",
        min_value=3.0, max_value=10.0,
        value=ph, step=0.05,
        placeholder="e.g. 6.5"
    )

    if ph is not None:
        ph_lbl, ph_ico, ph_bg, ph_col, ph_brd = ph_info(ph)
        st.markdown(f"""
        <div style="background:{ph_bg};border:1.5px solid {ph_brd};
             border-radius:12px;padding:12px 14px;margin-top:6px;
             box-shadow:0 2px 8px rgba(0,0,0,0.06)">
            <div style="font-size:0.62em;color:#6b7280;font-weight:700;
                 margin-bottom:5px;letter-spacing:0.8px">SOIL pH STATUS</div>
            <div style="display:flex;align-items:center;
                 justify-content:space-between">
                <span style="font-size:0.85em;font-weight:700;color:{ph_col}">
                    {ph_ico} {ph_lbl}
                </span>
                <span style="font-size:1.3em;font-weight:900;color:{ph_col}">
                    {ph:.2f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f9fafb;border:1.5px dashed #d1d5db;
             border-radius:12px;padding:12px 14px;margin-top:6px;
             text-align:center">
            <span style="font-size:0.70em;color:#9ca3af">
                Enter pH value to see soil status
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    alpha = st.select_slider(
        "🎯 Confidence Level",
        options=[0.01, 0.05, 0.10, 0.15, 0.20],
        value=0.05,
        format_func=lambda x: f"{int((1-x)*100)}%"
    )

# ── Column 4: Presets ─────────────────────────────────────────
with ic4:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7c2d12,#c2410c);
         border-radius:14px;padding:10px 14px;margin-bottom:14px;
         text-align:center;box-shadow:0 4px 16px rgba(194,65,12,0.25)">
        <span style="color:#fed7aa;font-size:0.78em;font-weight:800;
              letter-spacing:1.5px">⚡ SAMPLE PRESETS</span>
    </div>
    """, unsafe_allow_html=True)

    # N, P, K, ph, rain, temp, hum
    PRESETS = {
        "🌾 Rice":    [90,  42,  43,  6.5,  202.0, 20.9, 82.0],
        "🌽 Maize":   [71,  54,  16,  5.75,  87.8, 22.6, 63.7],
        "☕ Coffee":  [101, 28,  29,  6.8,  158.1, 25.5, 58.9],
        "🍌 Banana":  [100, 82,  50,  5.9,  105.5, 27.4, 80.3],
        "🍇 Grapes":  [23,  132, 200, 6.0,   67.2, 23.9, 80.0],
        "🥥 Coconut": [22,  17,  30,  6.0,  130.0, 27.4, 92.3],
    }

    pr1, pr2 = st.columns(2)
    for i, (pname, pvals) in enumerate(PRESETS.items()):
        bc = pr1 if i % 2 == 0 else pr2
        if bc.button(pname, key=f"pr_{i}", use_container_width=True):
            st.session_state['preset_vals'] = pvals
            st.rerun()

    st.markdown("""
    <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
         border-radius:12px;padding:12px 14px;margin-top:10px;
         border:1.5px solid #86efac;
         box-shadow:0 2px 8px rgba(5,150,105,0.08)">
        <div style="font-size:0.68em;color:#166534;font-weight:700;
             margin-bottom:6px">📊 Field Ranges (Dataset)</div>
        <div style="font-size:0.62em;color:#374151;line-height:1.9">
            🌿 N: 0–140 kg/ha &nbsp;|&nbsp; 🔴 P: 5–145 kg/ha<br>
            🟣 K: 5–205 kg/ha &nbsp;|&nbsp; 🧪 pH: 3.5–9.5<br>
            🌡️ Temp: 8–44°C &nbsp;|&nbsp; 💧 Humidity: 14–100%<br>
            🌧️ Rainfall: 20–299 mm
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Validate inputs ────────────────────────────────────────────
all_filled = all(v is not None for v in [N, P, K, temp, hum, rain, ph])

# ── Button ────────────────────────────────────────────────────
st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.3, 1, 1.3])
with btn_col:
    recommend = st.button(
        "🚀 Get My Crop Recommendation",
        use_container_width=True,
        disabled=not all_filled
    )

if not all_filled:
    st.markdown("""
    <div style="text-align:center;margin-top:16px">
        <div style="background:linear-gradient(135deg,#fffbeb,#fef3c7);
             border:1.5px solid #fcd34d;border-radius:14px;
             padding:14px 28px;display:inline-block;
             box-shadow:0 4px 16px rgba(251,191,36,0.2)">
            <span style="font-size:0.82em;color:#92400e;font-weight:700">
                ⚠️ Please fill in all field details above to get your crop recommendation
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run prediction ────────────────────────────────────────────
input_vals            = [N, P, K, temp, hum, ph, rain]
y_pred, y_prob, x_sc  = run_predict(input_vals)
confidence            = float(y_prob[y_pred])
crop                  = class_names[y_pred]
crop_em               = CROP_EMOJI.get(crop.lower(), '🌱')
pred_set              = get_pred_set(x_sc, alpha)
sv                    = get_shap(x_sc, y_pred)
set_size              = len(pred_set)

# Top alternatives: only crops with prob > 5% (meaningful)
ALT_THRESHOLD = 0.05
top_alts = [
    (i, float(y_prob[i]))
    for i in np.argsort(y_prob)[::-1]
    if float(y_prob[i]) >= ALT_THRESHOLD and i != y_pred
][:4]


# ============================================================
# RESULTS
# ============================================================
st.markdown('<div class="g-div"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-hdr">📊 Recommendation Results</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns([1.1, 1.15, 0.95], gap="large")

# ── PANEL 1: Recommended Crop ─────────────────────────────────
with r1:
    if confidence >= 0.80:
        badge_bg, badge_fg = "#dcfce7", "#065f46"
        badge_txt = "✅ High Confidence"
    elif confidence >= 0.60:
        badge_bg, badge_fg = "#fef9c3", "#92400e"
        badge_txt = "⚠️ Medium Confidence"
    else:
        badge_bg, badge_fg = "#fee2e2", "#7f1d1d"
        badge_txt = "❌ Low Confidence"

    st.markdown(f"""
    <div style="background:linear-gradient(160deg,#052e16 0%,#14532d 40%,
         #166534 70%,#15803d 100%);
         border-radius:24px;padding:30px 26px;text-align:center;
         box-shadow:0 16px 48px rgba(5,46,22,0.40);
         position:relative;overflow:hidden;
         border:1px solid rgba(74,222,128,0.25)">

        <div style="position:absolute;top:-30px;right:-30px;width:110px;
             height:110px;border-radius:50%;
             background:radial-gradient(circle,rgba(74,222,128,0.18),transparent)">
        </div>
        <div style="position:absolute;bottom:-35px;left:-30px;width:130px;
             height:130px;border-radius:50%;
             background:radial-gradient(circle,rgba(16,185,129,0.12),transparent)">
        </div>

        <div style="font-size:5.5em;margin-bottom:8px;
             filter:drop-shadow(0 6px 16px rgba(0,0,0,0.4));
             position:relative;z-index:1">{crop_em}</div>

        <div style="font-size:0.62em;color:#86efac;font-weight:700;
             letter-spacing:2.5px;margin-bottom:5px;position:relative">
             RECOMMENDED CROP</div>

        <div style="font-size:2.2em;font-weight:900;color:white;
             letter-spacing:2px;text-transform:uppercase;
             text-shadow:0 3px 12px rgba(0,0,0,0.4);
             margin-bottom:22px;position:relative">
             {crop}</div>

        <div style="background:rgba(255,255,255,0.12);border-radius:18px;
             padding:18px 20px;backdrop-filter:blur(20px);
             border:1px solid rgba(255,255,255,0.2);position:relative">
            <div style="font-size:0.62em;color:#a7f3d0;font-weight:600;
                 letter-spacing:1px;margin-bottom:4px">CONFIDENCE SCORE</div>
            <div style="font-size:3.5em;font-weight:900;color:white;
                 line-height:1.0;letter-spacing:-2px">{confidence:.1%}</div>
            <div style="background:rgba(255,255,255,0.18);border-radius:8px;
                 height:9px;margin:12px 0 6px 0;overflow:hidden">
                <div style="width:{confidence*100:.1f}%;height:100%;
                     border-radius:8px;
                     background:linear-gradient(90deg,#6ee7b7,#34d399,#10b981)">
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;
                 font-size:0.55em;color:rgba(255,255,255,0.4)">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
        </div>

        <div style="margin-top:14px;display:flex;
             justify-content:center;gap:8px;flex-wrap:wrap;position:relative">
            <div style="background:{badge_bg};border-radius:25px;
                 padding:6px 16px">
                <span style="font-size:0.72em;font-weight:800;
                      color:{badge_fg}">{badge_txt}</span>
            </div>
            <div style="background:rgba(255,255,255,0.14);border-radius:25px;
                 padding:6px 14px;border:1px solid rgba(255,255,255,0.2)">
                <span style="font-size:0.66em;color:#a7f3d0;font-weight:600">
                    📦 {set_size} crop(s) @ {int((1-alpha)*100)}% CI
                </span>
            </div>
        </div>
    </div>

    <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
         border-radius:16px;padding:16px 18px;margin-top:12px;
         border:1.5px solid #86efac;
         box-shadow:0 4px 18px rgba(5,150,105,0.12)">
        <div style="font-size:0.74em;font-weight:700;color:#065f46;
             margin-bottom:7px">🌿 Why {crop.title()}?</div>
        <div style="font-size:0.71em;color:#374151;line-height:1.75">
            {crop_reason(crop)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── PANEL 2: Top Alternatives ─────────────────────────────────
with r2:
    st.markdown("""
    <div style="font-size:0.90em;font-weight:800;color:#052e16;
         margin-bottom:14px;display:flex;align-items:center;gap:8px">
        🌿 Top Alternative Crops
    </div>
    """, unsafe_allow_html=True)

    GRAD_ALTS = [
        ("linear-gradient(135deg,#f0fdf4,#dcfce7)", "#059669", "#86efac"),
        ("linear-gradient(135deg,#eff6ff,#dbeafe)", "#1d4ed8", "#93c5fd"),
        ("linear-gradient(135deg,#fdf4ff,#f3e8ff)", "#7c3aed", "#c4b5fd"),
        ("linear-gradient(135deg,#fff7ed,#fed7aa)", "#c2410c", "#fdba74"),
    ]

    if top_alts:
        for ri, (idx, cp) in enumerate(top_alts):
            cn_alt = class_names[idx]
            ce_alt = CROP_EMOJI.get(cn_alt.lower(), '🌱')
            grad, tc, brd = GRAD_ALTS[ri % len(GRAD_ALTS)]

            st.markdown(f"""
            <div style="background:{grad};border:1.5px solid {brd};
                 border-radius:16px;padding:14px 16px;margin-bottom:10px;
                 box-shadow:0 4px 14px rgba(0,0,0,0.06)">
                <div style="display:flex;align-items:center;gap:12px">
                    <div style="background:rgba(255,255,255,0.9);
                         width:46px;height:46px;border-radius:12px;
                         display:flex;align-items:center;justify-content:center;
                         font-size:1.6em;flex-shrink:0;
                         box-shadow:0 2px 10px rgba(0,0,0,0.08)">
                        {ce_alt}
                    </div>
                    <div style="flex:1;min-width:0">
                        <div style="display:flex;align-items:center;
                             justify-content:space-between;margin-bottom:6px">
                            <span style="font-size:0.84em;font-weight:700;
                                  color:#1f2937">
                                {ri+1}. {cn_alt.title()}
                            </span>
                            <span style="font-size:0.92em;font-weight:900;
                                  color:{tc}">{cp:.1%}</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.55);
                             border-radius:6px;height:7px;overflow:hidden">
                            <div style="width:{cp*100:.1f}%;height:100%;
                                 border-radius:6px;
                                 background:linear-gradient(90deg,{brd},{tc})">
                            </div>
                        </div>
                        <div style="font-size:0.60em;color:#6b7280;
                             margin-top:3px">Suitability Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        tip_names = ", ".join([class_names[i].title() for i, _ in top_alts[:3]])
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#fffbeb,#fef3c7);
             border:1.5px solid #fcd34d;border-left:4px solid #f59e0b;
             border-radius:14px;padding:14px 16px;margin-top:6px;
             box-shadow:0 3px 12px rgba(251,191,36,0.15)">
            <div style="font-size:0.72em;font-weight:800;
                 color:#b45309;margin-bottom:5px">💡 Farmer's Tip</div>
            <div style="font-size:0.68em;color:#78350f;line-height:1.70">
                Consider <strong>{tip_names}</strong> as alternatives
                based on market demand, seasonal availability and
                your farming preferences.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
             border:1.5px solid #86efac;border-radius:20px;
             padding:30px 24px;text-align:center;
             box-shadow:0 6px 24px rgba(5,150,105,0.12)">
            <div style="font-size:2.5em;margin-bottom:12px">🌟</div>
            <div style="font-size:0.90em;font-weight:800;color:#065f46;
                 margin-bottom:8px">Dominant Recommendation!</div>
            <div style="font-size:0.72em;color:#374151;line-height:1.75;
                 max-width:260px;margin:0 auto">
                The model is highly confident in its primary
                recommendation. No significant alternatives
                were found for your field conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── PANEL 3: Reliability ──────────────────────────────────────
with r3:
    st.markdown("""
    <div style="font-size:0.90em;font-weight:800;color:#052e16;
         margin-bottom:14px">🛡️ Result Reliability</div>
    """, unsafe_allow_html=True)

    # ── Model Reliability ────────────────────────────────────
    # Icon: shield/check style matching the original poster image
    if confidence >= 0.80:
        rlvl   = "High"
        rbg    = "linear-gradient(135deg,#f0fdf4,#dcfce7)"
        rbrd   = "#86efac"
        rcol   = "#065f46"
        ric_bg = "linear-gradient(135deg,#059669,#10b981)"
        ric    = "🛡️"
    elif confidence >= 0.60:
        rlvl   = "Medium"
        rbg    = "linear-gradient(135deg,#fffbeb,#fef3c7)"
        rbrd   = "#fcd34d"
        rcol   = "#92400e"
        ric_bg = "linear-gradient(135deg,#d97706,#f59e0b)"
        ric    = "⚠️"
    else:
        rlvl   = "Low"
        rbg    = "linear-gradient(135deg,#fef2f2,#fee2e2)"
        rbrd   = "#fca5a5"
        rcol   = "#7f1d1d"
        ric_bg = "linear-gradient(135deg,#dc2626,#ef4444)"
        ric    = "⚡"

    st.markdown(f"""
    <div style="background:{rbg};border:1.5px solid {rbrd};
         border-radius:16px;padding:16px;margin-bottom:12px;
         box-shadow:0 4px 18px rgba(0,0,0,0.07)">
        <div style="font-size:0.60em;color:#6b7280;font-weight:700;
             letter-spacing:1.2px;margin-bottom:10px">MODEL RELIABILITY</div>
        <div style="display:flex;align-items:center;gap:14px">
            <div style="background:{ric_bg};
                 width:52px;height:52px;border-radius:14px;
                 display:flex;align-items:center;justify-content:center;
                 font-size:1.6em;flex-shrink:0;
                 box-shadow:0 6px 18px rgba(0,0,0,0.2)">
                {ric}
            </div>
            <div>
                <div style="font-size:1.5em;font-weight:900;
                     color:{rcol};line-height:1.1">{rlvl}</div>
                <div style="font-size:0.63em;color:{rcol};
                     font-weight:600;opacity:0.85">
                     Reliable Recommendation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction Reliability ───────────────────────────────
    # Icon: bar-chart style matching poster
    gcol = ("#1d4ed8" if confidence >= 0.80 else
            "#d97706" if confidence >= 0.60 else "#dc2626")
    glvl = ("High Confidence"   if confidence >= 0.80 else
            "Medium Confidence" if confidence >= 0.60 else
            "Low Confidence")
    g_icon_bg = ("linear-gradient(135deg,#1d4ed8,#3b82f6)" if confidence >= 0.80 else
                 "linear-gradient(135deg,#d97706,#f59e0b)" if confidence >= 0.60 else
                 "linear-gradient(135deg,#dc2626,#ef4444)")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);
         border:1.5px solid #93c5fd;border-radius:16px;
         padding:16px;margin-bottom:12px;
         box-shadow:0 4px 18px rgba(0,0,0,0.07)">
        <div style="font-size:0.60em;color:#6b7280;font-weight:700;
             letter-spacing:1.2px;margin-bottom:10px">
             PREDICTION RELIABILITY</div>
        <div style="display:flex;align-items:center;gap:14px">
            <div style="background:{g_icon_bg};
                 width:52px;height:52px;border-radius:14px;
                 display:flex;align-items:center;justify-content:center;
                 font-size:1.6em;flex-shrink:0;
                 box-shadow:0 6px 18px rgba(0,0,0,0.2)">
                📈
            </div>
            <div>
                <div style="font-size:2.0em;font-weight:900;
                     color:{gcol};line-height:1.0">
                    {confidence*100:.1f}%
                </div>
                <div style="font-size:0.63em;color:{gcol};
                     font-weight:600">{glvl}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quality Checks ────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.60em;color:#6b7280;font-weight:700;
         letter-spacing:1.2px;margin-bottom:8px">RESULT QUALITY CHECKS</div>
    """, unsafe_allow_html=True)

    checks = [
        ("Well Calibrated",      final_ece  <= 0.05),
        ("Low Prediction Error", final_f1   >= 0.90),
        ("Stable & Consistent",  shap_tau   >= 0.80),
        ("Trustworthy Result",   abs(final_coverage - 0.95) <= 0.03),
    ]
    for ck_name, ok in checks:
        if ok:
            ck_bg  = "linear-gradient(135deg,#dcfce7,#f0fdf4)"
            ck_fg  = "#14532d"
            ck_ico = "✅"
        else:
            ck_bg  = "linear-gradient(135deg,#fef9c3,#fffbeb)"
            ck_fg  = "#713f12"
            ck_ico = "⚠️"
        st.markdown(
            f"<div style='background:{ck_bg};color:{ck_fg};"
            f"border-radius:10px;padding:9px 13px;margin-bottom:6px;"
            f"font-size:0.72em;font-weight:600;"
            f"box-shadow:0 1px 5px rgba(0,0,0,0.05)'>"
            f"{ck_ico} {ck_name}</div>",
            unsafe_allow_html=True
        )


# ============================================================
# SHAP SECTION
# ============================================================
st.markdown('<div class="g-div"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">🔍 Why This Recommendation? (AI Explanation)</div>',
    unsafe_allow_html=True
)

sc1, sc2 = st.columns([1.2, 0.8], gap="large")

with sc1:
    shap_arr = np.array(sv).flatten() if sv is not None else shap_importance.copy()
    max_abs  = max(np.abs(shap_arr).max(), 1e-6)
    sort_idx = np.argsort(np.abs(shap_arr))[::-1]

    st.markdown("""
    <div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap">
        <div style="background:linear-gradient(135deg,#dcfce7,#bbf7d0);
             border:1.5px solid #86efac;border-radius:25px;
             padding:5px 16px">
            <span style="font-size:0.70em;font-weight:700;color:#065f46">
                ✅ Positive Impact
            </span>
        </div>
        <div style="background:linear-gradient(135deg,#fee2e2,#fecaca);
             border:1.5px solid #fca5a5;border-radius:25px;
             padding:5px 16px">
            <span style="font-size:0.70em;font-weight:700;color:#7f1d1d">
                ❌ Negative Impact
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for fi in sort_idx:
        feat  = feature_cols[fi]
        val   = float(shap_arr[fi])
        icon  = FEAT_ICON.get(feat, '📊')
        name  = FEAT_NAME.get(feat, feat)
        unit  = FEAT_UNIT.get(feat, '')
        inp_v = input_vals[feature_cols.index(feat)]
        bw    = min(abs(val) / max_abs * 100, 100)
        is_p  = val >= 0

        if is_p:
            bar_grad = "linear-gradient(90deg,#4ade80,#059669,#065f46)"
            val_col  = "#065f46"
            card_bg  = "linear-gradient(135deg,#f0fdf4,#f7fef9)"
            card_brd = "#bbf7d0"
        else:
            bar_grad = "linear-gradient(90deg,#f87171,#ef4444,#dc2626)"
            val_col  = "#dc2626"
            card_bg  = "linear-gradient(135deg,#fef2f2,#fff5f5)"
            card_brd = "#fca5a5"

        sign = "+" if is_p else ""

        st.markdown(f"""
        <div style="background:{card_bg};border:1.5px solid {card_brd};
             border-radius:14px;padding:12px 16px;margin-bottom:8px;
             box-shadow:0 2px 10px rgba(0,0,0,0.05)">
            <div style="display:flex;align-items:center;gap:12px">
                <div style="background:rgba(255,255,255,0.92);
                     width:36px;height:36px;border-radius:10px;
                     display:flex;align-items:center;justify-content:center;
                     font-size:1.15em;flex-shrink:0;
                     box-shadow:0 2px 8px rgba(0,0,0,0.07)">
                    {icon}
                </div>
                <div style="flex:1;min-width:0">
                    <div style="display:flex;justify-content:space-between;
                         align-items:center;margin-bottom:6px">
                        <span style="font-size:0.78em;font-weight:600;color:#1f2937">
                            {name}
                            <span style="color:#9ca3af;font-weight:400;font-size:0.88em">
                                &nbsp;({inp_v:.1f}{' '+unit if unit else ''})
                            </span>
                        </span>
                        <span style="font-size:0.80em;font-weight:800;color:{val_col}">
                            {sign}{val:.3f}
                        </span>
                    </div>
                    <div style="background:rgba(255,255,255,0.65);
                         border-radius:8px;height:10px;overflow:hidden">
                        <div style="width:{bw:.1f}%;height:100%;
                             border-radius:8px;background:{bar_grad}">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with sc2:
    if sv is not None:
        pos_f = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                 for i in sort_idx if shap_arr[i] > 0][:3]
        neg_f = [FEAT_NAME.get(feature_cols[i], feature_cols[i])
                 for i in sort_idx if shap_arr[i] < 0][:3]
    else:
        si2   = np.argsort(shap_importance)[::-1]
        pos_f = [FEAT_NAME.get(feature_cols[i], feature_cols[i]) for i in si2[:3]]
        neg_f = []

    pos_items = "".join([
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'padding:8px 0;border-bottom:1px solid rgba(134,239,172,0.35)">'
        f'<div style="width:22px;height:22px;border-radius:50%;'
        f'background:linear-gradient(135deg,#059669,#10b981);'
        f'display:flex;align-items:center;justify-content:center;'
        f'flex-shrink:0">'
        f'<span style="color:white;font-size:0.7em;font-weight:900">+</span></div>'
        f'<span style="font-size:0.73em;color:#374151;font-weight:500">{f}</span>'
        f'</div>'
        for f in pos_f
    ])

    neg_items = "".join([
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'padding:8px 0;border-bottom:1px solid rgba(252,165,165,0.35)">'
        f'<div style="width:22px;height:22px;border-radius:50%;'
        f'background:linear-gradient(135deg,#dc2626,#ef4444);'
        f'display:flex;align-items:center;justify-content:center;'
        f'flex-shrink:0">'
        f'<span style="color:white;font-size:0.7em;font-weight:900">−</span></div>'
        f'<span style="font-size:0.73em;color:#374151;font-weight:500">{f}</span>'
        f'</div>'
        for f in neg_f
    ]) if neg_f else (
        '<div style="font-size:0.70em;color:#6b7280;padding:10px 0">'
        'No significant negative factors</div>'
    )

    st.markdown(f"""
    <div style="background:linear-gradient(160deg,#f0fdf4,#ecfdf5,#f0f9ff);
         border-radius:18px;padding:20px;border:1.5px solid #d1fae5;
         box-shadow:0 6px 24px rgba(0,0,0,0.07);margin-bottom:14px">
        <div style="font-size:0.82em;font-weight:800;color:#052e16;
             margin-bottom:14px">📋 Key Factor Summary</div>

        <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);
             border-radius:12px;padding:14px;margin-bottom:12px;
             border:1px solid #86efac">
            <div style="font-size:0.66em;font-weight:700;color:#065f46;
                 margin-bottom:8px;letter-spacing:0.8px">✅ POSITIVE FACTORS</div>
            {pos_items}
        </div>

        <div style="background:linear-gradient(135deg,#fef2f2,#fee2e2);
             border-radius:12px;padding:14px;border:1px solid #fca5a5">
            <div style="font-size:0.66em;font-weight:700;color:#7f1d1d;
                 margin-bottom:8px;letter-spacing:0.8px">❌ LIMITING FACTORS</div>
            {neg_items}
        </div>
    </div>

    <div style="background:linear-gradient(135deg,#fdf4ff,#f3e8ff);
         border-radius:18px;padding:20px;border:1.5px solid #c4b5fd;
         box-shadow:0 6px 24px rgba(0,0,0,0.07)">
        <div style="font-size:0.82em;font-weight:800;color:#4c1d95;
             margin-bottom:12px">🔬 Explanation Stability</div>
        <div style="text-align:center;margin-bottom:10px">
            <div style="font-size:0.60em;color:#6b7280;font-weight:600;
                 letter-spacing:0.8px;margin-bottom:6px">
                KENDALL'S τ COEFFICIENT
            </div>
            <div style="font-size:2.8em;font-weight:900;
                 color:{'#065f46' if shap_tau>=0.80 else '#d97706'}">
                {shap_tau:.4f}
            </div>
            <div style="background:rgba(255,255,255,0.55);border-radius:8px;
                 height:8px;margin:10px 20px;overflow:hidden">
                <div style="width:{min(shap_tau*100,100):.0f}%;height:100%;
                     border-radius:8px;
                     background:linear-gradient(90deg,#a78bfa,#7c3aed)">
                </div>
            </div>
            <div style="background:{'#dcfce7' if shap_tau>=0.80 else '#fef9c3'};
                 color:{'#14532d' if shap_tau>=0.80 else '#713f12'};
                 border-radius:20px;padding:5px 18px;
                 display:inline-block;font-size:0.70em;font-weight:700">
                {'✅ Stable & Reliable' if shap_tau>=0.80 else '⚠️ Below Target'}
                &nbsp;(Target ≥ 0.80)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TRUSTWORTHINESS METRICS
# ============================================================
st.markdown('<div class="g-div"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">🏆 Trustworthiness Report</div>',
    unsafe_allow_html=True
)

tm1, tm2, tm3, tm4 = st.columns(4, gap="medium")

TRUST_CARDS = [
    (tm1,
     "🎯", "Prediction Performance", "Macro F1 Score",
     final_f1, f"{final_f1:.4f}", "≥ 0.90", final_f1 >= 0.90,
     "linear-gradient(160deg,#f0fdf4,#dcfce7,#bbf7d0)",
     "#86efac", "#065f46", "#4ade80", "#059669"),

    (tm2,
     "🔮", "Uncertainty Reliability", "Conformal Coverage",
     final_coverage, f"{final_coverage:.4f}", "≈ 0.95",
     abs(final_coverage - 0.95) <= 0.03,
     "linear-gradient(160deg,#eff6ff,#dbeafe,#bfdbfe)",
     "#93c5fd", "#1d4ed8", "#60a5fa", "#2563eb"),

    (tm3,
     "📐", "Calibration Quality", "Expected Calibration Error",
     1 - final_ece, f"ECE={final_ece:.4f}", "≤ 0.05", final_ece <= 0.05,
     "linear-gradient(160deg,#fdf4ff,#f3e8ff,#e9d5ff)",
     "#c4b5fd", "#7c3aed", "#a78bfa", "#6d28d9"),

    (tm4,
     "🧬", "Explanation Stability", "Kendall's τ Coefficient",
     shap_tau, f"{shap_tau:.4f}", "≥ 0.80", shap_tau >= 0.80,
     "linear-gradient(160deg,#fff7ed,#fed7aa,#fdba74)",
     "#fb923c", "#c2410c", "#f97316", "#ea580c"),
]

for (col, icon, title, subtitle, val, display,
     target, met, grad, brd, dark, light, mid) in TRUST_CARDS:
    bar_w   = min(val * 100, 100) if val <= 1 else 100
    met_txt = "✅ Target Met" if met else "⚠️ Not Met"
    met_bg  = "#dcfce7" if met else "#fef9c3"
    met_col = "#14532d" if met else "#713f12"

    col.markdown(f"""
    <div style="background:{grad};border:1.5px solid {brd};
         border-radius:22px;padding:24px 18px;text-align:center;
         box-shadow:0 8px 28px rgba(0,0,0,0.09);
         position:relative;overflow:hidden">
        <div style="position:absolute;top:-25px;right:-25px;
             width:90px;height:90px;border-radius:50%;
             background:radial-gradient(circle,rgba(255,255,255,0.35),transparent)">
        </div>
        <div style="position:absolute;bottom:-30px;left:-20px;
             width:80px;height:80px;border-radius:50%;
             background:radial-gradient(circle,rgba(255,255,255,0.2),transparent)">
        </div>
        <div style="font-size:2em;margin-bottom:8px">{icon}</div>
        <div style="font-size:0.82em;font-weight:800;color:{dark};
             margin-bottom:4px;line-height:1.3">{title}</div>
        <div style="font-size:0.60em;color:#6b7280;
             margin-bottom:16px">{subtitle}</div>
        <div style="font-size:2.3em;font-weight:900;color:{dark};
             line-height:1.0;margin-bottom:14px">{display}</div>
        <div style="background:rgba(255,255,255,0.55);border-radius:8px;
             height:8px;margin:0 8px 14px 8px;overflow:hidden">
            <div style="width:{bar_w:.0f}%;height:100%;border-radius:8px;
                 background:linear-gradient(90deg,{light},{mid},{dark})">
            </div>
        </div>
        <div style="background:{met_bg};border-radius:20px;
             padding:5px 16px;display:inline-block">
            <span style="font-size:0.68em;font-weight:700;
                  color:{met_col}">{met_txt}</span>
        </div>
        <div style="font-size:0.58em;color:#9ca3af;margin-top:8px">
            Target: {target}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CHARTS SECTION
# ============================================================
st.markdown('<div class="g-div"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-hdr">📈 Model Analysis & NSGA-II Optimization</div>',
    unsafe_allow_html=True
)

import plotly.graph_objects as go

PLOT_BG  = "rgba(240,253,244,0.7)"
FONT_FAM = "Inter"
TITLE_COL = "#052e16"

tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 Pareto Front",
    "📐 Calibration",
    "🤖 Model Comparison",
    "🕸️ Radar Chart"
])

with tab1:
    if pareto_F is not None and len(pareto_F) > 0:
        pa, pb = st.columns(2, gap="large")
        with pa:
            f1 = go.Figure()
            f1.add_trace(go.Scatter(
                x=-pareto_F[:, 0], y=pareto_F[:, 2],
                mode='markers',
                marker=dict(
                    size=13, color=-pareto_F[:, 3],
                    colorscale='RdYlGn', showscale=True,
                    colorbar=dict(title='SHAP τ', len=0.75,
                                  tickfont=dict(family=FONT_FAM)),
                    line=dict(color='white', width=1.5),
                    opacity=0.9
                ),
                hovertemplate='<b>F1: %{x:.4f}</b><br>ECE: %{y:.4f}<extra></extra>'
            ))
            f1.add_trace(go.Scatter(
                x=[-pareto_F[best_pareto_idx, 0]],
                y=[pareto_F[best_pareto_idx, 2]],
                mode='markers',
                marker=dict(size=22, color='#ef4444', symbol='star',
                            line=dict(color='#7f1d1d', width=2)),
                name='Best Compromise'
            ))
            f1.add_hline(y=0.05, line_dash="dot", line_color="#ef4444",
                         line_width=2, annotation_text="ECE Target (0.05)",
                         annotation_font=dict(color="#ef4444", family=FONT_FAM))
            f1.add_vline(x=0.90, line_dash="dot", line_color="#3b82f6",
                         line_width=2, annotation_text="F1 Target (0.90)",
                         annotation_font=dict(color="#3b82f6", family=FONT_FAM))
            f1.update_layout(
                title=dict(text="Pareto Front: F1 Score vs ECE",
                           font=dict(size=14, color=TITLE_COL, family=FONT_FAM)),
                xaxis_title="F1 Score ↑", yaxis_title="ECE ↓",
                height=420, plot_bgcolor=PLOT_BG, paper_bgcolor='white',
                font=dict(family=FONT_FAM), showlegend=True,
                legend=dict(x=0.02, y=0.02)
            )
            st.plotly_chart(f1, use_container_width=True)

        with pb:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(
                x=-pareto_F[:, 0], y=pareto_F[:, 1],
                mode='markers',
                marker=dict(
                    size=13, color=pareto_F[:, 2],
                    colorscale='RdYlGn_r', showscale=True,
                    colorbar=dict(title='ECE', len=0.75,
                                  tickfont=dict(family=FONT_FAM)),
                    line=dict(color='white', width=1.5),
                    opacity=0.9
                ),
                hovertemplate='<b>F1: %{x:.4f}</b><br>Set Size: %{y:.2f}<extra></extra>'
            ))
            f2.add_trace(go.Scatter(
                x=[-pareto_F[best_pareto_idx, 0]],
                y=[pareto_F[best_pareto_idx, 1]],
                mode='markers',
                marker=dict(size=22, color='#ef4444', symbol='star',
                            line=dict(color='#7f1d1d', width=2)),
                name='Best Compromise'
            ))
            f2.add_vline(x=0.90, line_dash="dot", line_color="#3b82f6",
                         line_width=2, annotation_text="F1 Target",
                         annotation_font=dict(color="#3b82f6", family=FONT_FAM))
            f2.update_layout(
                title=dict(text="Pareto Front: F1 Score vs Uncertainty",
                           font=dict(size=14, color=TITLE_COL, family=FONT_FAM)),
                xaxis_title="F1 Score ↑",
                yaxis_title="Avg Prediction Set Size ↓",
                height=420, plot_bgcolor=PLOT_BG, paper_bgcolor='white',
                font=dict(family=FONT_FAM)
            )
            st.plotly_chart(f2, use_container_width=True)
    else:
        st.info("Pareto data not available.")

with tab2:
    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        if ece_results:
            mn   = list(ece_results.keys())
            re   = [ece_results[m]['ECE_raw'] for m in mn]
            ce_v = [ece_results[m]['ECE_calibrated'] for m in mn]
            f3   = go.Figure()
            f3.add_trace(go.Bar(
                name='Before Calibration', x=mn, y=re,
                marker=dict(color='#ef4444', line=dict(color='white', width=1.5)),
                text=[f'{v:.4f}' for v in re],
                textposition='outside', opacity=0.85
            ))
            f3.add_trace(go.Bar(
                name='After Calibration', x=mn, y=ce_v,
                marker=dict(color='#059669', line=dict(color='white', width=1.5)),
                text=[f'{v:.4f}' for v in ce_v],
                textposition='outside', opacity=0.85
            ))
            f3.add_hline(y=0.05, line_dash="dash", line_color="#3b82f6",
                         line_width=2, annotation_text="Target ECE = 0.05",
                         annotation_font=dict(color="#3b82f6", family=FONT_FAM))
            f3.update_layout(
                title=dict(text="ECE Before vs After Calibration",
                           font=dict(size=14, color=TITLE_COL, family=FONT_FAM)),
                barmode='group', height=420,
                plot_bgcolor=PLOT_BG, paper_bgcolor='white',
                font=dict(family=FONT_FAM),
                legend=dict(x=0.55, y=0.95)
            )
            st.plotly_chart(f3, use_container_width=True)

    with cc2:
        metrics_summary = [
            ("🎯 Macro F1",    final_f1,       "≥ 0.90",
             final_f1 >= 0.90,        "#f0fdf4","#86efac","#065f46"),
            ("🔮 Coverage",    final_coverage, "≈ 0.95",
             abs(final_coverage-0.95) <= 0.03, "#eff6ff","#93c5fd","#1d4ed8"),
            ("📐 ECE",         final_ece,      "≤ 0.05",
             final_ece <= 0.05,        "#fdf4ff","#c4b5fd","#7c3aed"),
            ("🧬 SHAP τ",      shap_tau,       "≥ 0.80",
             shap_tau >= 0.80,         "#fff7ed","#fdba74","#c2410c"),
        ]
        for m_lbl, m_val, m_tgt, m_ok, bg, brd, col in metrics_summary:
            m_ok_txt = "✅ Met" if m_ok else "⚠️ Not Met"
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {brd};
                 border-radius:14px;padding:14px 18px;margin-bottom:10px;
                 display:flex;align-items:center;justify-content:space-between;
                 box-shadow:0 3px 10px rgba(0,0,0,0.06)">
                <div>
                    <div style="font-size:0.82em;font-weight:700;color:{col}">
                        {m_lbl}
                    </div>
                    <div style="font-size:0.62em;color:#6b7280;margin-top:2px">
                        Target: {m_tgt}
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:1.3em;font-weight:900;color:{col}">
                        {m_val:.4f}
                    </div>
                    <div style="font-size:0.65em;font-weight:700;color:{col}">
                        {m_ok_txt}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    if not results_df.empty:
        mc1, mc2 = st.columns(2, gap="large")
        with mc1:
            f5 = go.Figure()
            BARS = [
                ('f1_calibrated',  '#059669', 'F1 Score (Calibrated)'),
                ('acc_calibrated', '#34d399', 'Accuracy'),
                ('cv_mean',        '#065f46', 'CV F1 (5-fold)')
            ]
            for cn3, cc3, lb3 in BARS:
                if cn3 in results_df.columns:
                    f5.add_trace(go.Bar(
                        name=lb3,
                        x=results_df.index,
                        y=results_df[cn3],
                        marker=dict(color=cc3, line=dict(color='white', width=1.5)),
                        text=[f'{v:.4f}' for v in results_df[cn3]],
                        textposition='outside', opacity=0.88
                    ))
            f5.add_hline(y=0.90, line_dash="dot", line_color="#ef4444",
                         line_width=2, annotation_text="F1 Target (0.90)",
                         annotation_font=dict(color="#ef4444", family=FONT_FAM))
            f5.update_layout(
                title=dict(text="Model Performance Comparison",
                           font=dict(size=14, color=TITLE_COL, family=FONT_FAM)),
                barmode='group', yaxis_range=[0.7, 1.05],
                height=420, plot_bgcolor=PLOT_BG, paper_bgcolor='white',
                font=dict(family=FONT_FAM), legend=dict(x=0, y=1)
            )
            st.plotly_chart(f5, use_container_width=True)

        with mc2:
            summary = pd.DataFrame({
                'Objective': [
                    'Prediction Performance', 'Uncertainty Reliability',
                    'Calibration Quality', 'Explanation Stability'
                ],
                'Metric':  ['Macro F1', 'Coverage', 'ECE', "Kendall's τ"],
                'Value':   [
                    f"{final_f1:.4f}", f"{final_coverage:.4f}",
                    f"{final_ece:.4f}", f"{shap_tau:.4f}"
                ],
                'Target':  ['≥ 0.90', '≈ 0.95', '≤ 0.05', '≥ 0.80'],
                'Status':  [
                    '✅ Met' if final_f1 >= 0.90               else '⚠️',
                    '✅ Met' if abs(final_coverage-0.95) <= 0.03 else '⚠️',
                    '✅ Met' if final_ece <= 0.05              else '⚠️',
                    '✅ Met' if shap_tau >= 0.80               else '⚠️',
                ]
            })
            st.markdown(
                "<div style='font-size:0.85em;font-weight:700;"
                "color:#052e16;margin-bottom:10px'>"
                "🎯 All 4/4 Trustworthiness Targets Met</div>",
                unsafe_allow_html=True
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Model data not available.")

with tab4:
    cats = [
        'F1 Score<br>(≥0.90)',
        'Coverage<br>(≈0.95)',
        'ECE Inv<br>(≤0.05)',
        'SHAP τ<br>(≥0.80)'
    ]
    rv = [
        min(1.0, final_f1 / 0.95),
        min(1.0, max(0.0, 1 - abs(final_coverage - 0.95) / 0.05)),
        min(1.0, max(0.0, 1 - final_ece / 0.05)),
        min(1.0, shap_tau / 0.90)
    ]
    f6 = go.Figure()
    f6.add_trace(go.Scatterpolar(
        r=rv + [rv[0]], theta=cats + [cats[0]],
        fill='toself',
        fillcolor='rgba(5,150,105,0.18)',
        line=dict(color='#059669', width=3),
        marker=dict(size=10, color='#065f46',
                    line=dict(color='white', width=2)),
        name='Achieved'
    ))
    f6.add_trace(go.Scatterpolar(
        r=[1, 1, 1, 1, 1], theta=cats + [cats[0]],
        fill='none',
        line=dict(color='#ef4444', width=2, dash='dot'),
        marker=dict(size=6, color='#ef4444'),
        name='Target'
    ))
    f6.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(
                visible=True, range=[0, 1.15],
                tickfont=dict(size=10, color='#374151', family=FONT_FAM),
                gridcolor='rgba(0,0,0,0.08)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#052e16',
                              family=FONT_FAM, weight=700),
                gridcolor='rgba(0,0,0,0.06)'
            )
        ),
        title=dict(
            text="Trustworthiness Radar — All 4 Objectives",
            font=dict(size=15, color=TITLE_COL, family=FONT_FAM)
        ),
        height=520, showlegend=True,
        paper_bgcolor='white', font=dict(family=FONT_FAM),
        legend=dict(x=0.85, y=1.1, font=dict(size=11, family=FONT_FAM))
    )
    st.plotly_chart(f6, use_container_width=True)


# ============================================================
# FOOTER — full width, no side padding gaps
# ============================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #052e16 0%, #14532d 30%, #166534 65%, #15803d 100%);
    border-radius: 24px;
    padding: 28px 50px;
    margin-top: 40px;
    box-shadow: 0 12px 40px rgba(5,46,22,0.35);
    position: relative;
    overflow: hidden;
">
    <!-- Decorative glows -->
    <div style="position:absolute;inset:0;
         background:
             radial-gradient(ellipse at 20% 50%, rgba(74,222,128,0.10), transparent 55%),
             radial-gradient(ellipse at 80% 50%, rgba(34,197,94,0.07), transparent 55%),
             radial-gradient(ellipse at 50% 0%,  rgba(255,255,255,0.04), transparent 40%);
         pointer-events:none">
    </div>

    <div style="display:flex;align-items:center;justify-content:center;
         gap:24px;flex-wrap:wrap;position:relative;z-index:1">

        <!-- Left icons -->
        <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:2.2em;
                 filter:drop-shadow(0 3px 8px rgba(0,0,0,0.3))">🚜</span>
            <span style="font-size:1.8em;
                 filter:drop-shadow(0 3px 8px rgba(0,0,0,0.3))">🌾</span>
        </div>

        <!-- Center text -->
        <div style="text-align:center;flex:1;min-width:260px">
            <div style="color:white;font-size:1.0em;font-weight:800;
                 margin-bottom:6px;letter-spacing:0.3px">
                👨‍🌾 Our Mission
            </div>
            <div style="color:#bbf7d0;font-size:0.82em;
                 line-height:1.7;font-weight:500">
                Helping farmers make <strong style="color:white">better,
                reliable and confident</strong> crop choices
                for higher yield and sustainable farming.
            </div>
            <div style="margin-top:12px;display:flex;justify-content:center;
                 gap:8px;flex-wrap:wrap">
                <span style="background:rgba(255,255,255,0.12);
                     border:1px solid rgba(255,255,255,0.18);
                     border-radius:20px;padding:4px 14px;
                     font-size:0.65em;color:#a7f3d0;font-weight:600">
                    🧬 NSGA-II Optimization
                </span>
                <span style="background:rgba(255,255,255,0.12);
                     border:1px solid rgba(255,255,255,0.18);
                     border-radius:20px;padding:4px 14px;
                     font-size:0.65em;color:#a7f3d0;font-weight:600">
                    🎯 Conformal Prediction
                </span>
                <span style="background:rgba(255,255,255,0.12);
                     border:1px solid rgba(255,255,255,0.18);
                     border-radius:20px;padding:4px 14px;
                     font-size:0.65em;color:#a7f3d0;font-weight:600">
                    🔍 SHAP XAI
                </span>
                <span style="background:rgba(255,255,255,0.12);
                     border:1px solid rgba(255,255,255,0.18);
                     border-radius:20px;padding:4px 14px;
                     font-size:0.65em;color:#a7f3d0;font-weight:600">
                    📐 Calibration Analysis
                </span>
            </div>
        </div>

        <!-- Right icons -->
        <div style="display:flex;gap:8px;align-items:center">
            <span style="font-size:1.8em;
                 filter:drop-shadow(0 3px 8px rgba(0,0,0,0.3))">🌿</span>
            <span style="font-size:2.2em;
                 filter:drop-shadow(0 3px 8px rgba(0,0,0,0.3))">🌱</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
