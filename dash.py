"""
CampaignAI — Marketing Campaign Performance Predictor
Brands : Nykaa · Purplle · Tira
Models : XGBoost Regressor (Revenue)  +  Logistic Regression + SMOTE (Profit / Loss)

Run:  streamlit run dash.py
Needs regression_model.pkl and classification_model.pkl next to this file
(or in C:\\Users\\LENOVO\\Documents\\Project_3_Market_Analysis).
"""

from pathlib import Path
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CampaignAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(r"C:\Users\LENOVO\Documents\Project_3_Market_Analysis")

CHANNEL_NAMES = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
BRANDS = ["Nykaa", "Purplle", "Tira"]

# Revenue is unknown before launch, so Updated_ROI is estimated from
# conversions using an average revenue per conversion.
REVENUE_PER_CONVERSION = 450

# ── Green palette ─────────────────────────────────────
G_BG = "#0B3B2C"        # page background (deep emerald)
G_SURFACE = "#0F4D3A"   # input boxes
G_ACCENT = "#34D399"    # emerald accent
G_DEEP = "#047857"      # button gradient end
G_MINT = "#A7F3D0"      # labels
G_SOFT = "#D1FAE5"      # secondary text
G_PROFIT = "#6EE7B7"    # profit
G_LOSS = "#CBD5E1"      # loss (neutral slate — no red)


# ══════════════════════════════════════════════════════
# HELPERS — rendering
# ══════════════════════════════════════════════════════
def html(markup: str):
    """Render HTML safely (flattened so Markdown never shows it as code)."""
    clean = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def stretch(element, *args, **kwargs):
    """Full-width element on both old and new Streamlit versions."""
    try:
        return element(*args, width="stretch", **kwargs)
    except TypeError:
        return element(*args, use_container_width=True, **kwargs)


def inr(value: float) -> str:
    """Indian digit grouping: ₹12,34,567."""
    value = float(value)
    sign = "-" if value < 0 else ""
    s = f"{abs(value):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return f"{sign}₹{s}"


def notice(text: str):
    """Green-themed message box (used instead of the red st.error)."""
    html(f'<div class="notice">⚠️ {text}</div>')


# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Nunito+Sans:wght@300;400;600;700&display=swap');

.stApp {{
    background:
        radial-gradient(ellipse at 10% 0%,  rgba(52,211,153,0.18) 0%, transparent 45%),
        radial-gradient(ellipse at 90% 100%, rgba(16,185,129,0.16) 0%, transparent 50%),
        linear-gradient(160deg, #0E4535 0%, {G_BG} 55%, #0A3527 100%);
    color:#FFFFFF;
}}
html, body, [class*="css"], .stApp, .stApp p, .stApp label, .stApp input, .stApp button {{
    font-family:'Nunito Sans',sans-serif;
}}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:1.8rem; max-width:1200px; }}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family:'Poppins',sans-serif; color:#FFFFFF; }}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] strong {{ color:#FFFFFF; }}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {{ color:{G_SOFT} !important; }}

/* ── Hero ── */
.hero {{ text-align:center; padding:0.5rem 0 0.2rem; }}
.hero-badge {{
    display:inline-block; font-size:0.7rem; letter-spacing:0.22em;
    text-transform:uppercase; color:{G_MINT}; font-weight:700;
    border:1px solid rgba(167,243,208,0.45); border-radius:100px;
    padding:0.3rem 1rem; margin-bottom:0.9rem;
    background:rgba(52,211,153,0.10);
}}
.hero-title {{
    font-family:'Poppins',sans-serif; font-size:2.9rem; font-weight:700;
    line-height:1.1; color:#FFFFFF; margin:0; letter-spacing:-0.02em;
}}
.hero-title .acc {{ color:{G_ACCENT}; }}
.hero-sub {{ font-size:1.02rem; color:{G_SOFT}; margin-top:0.6rem; }}

/* ── KPI strip ── */
div[data-testid="stMetric"] {{
    background:rgba(255,255,255,0.07);
    border:1px solid rgba(167,243,208,0.22);
    border-radius:14px; padding:1.1rem 1.4rem;
    transition:border-color 0.2s, transform 0.2s;
}}
div[data-testid="stMetric"]:hover {{ border-color:{G_ACCENT}; transform:translateY(-2px); }}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
    color:{G_MINT} !important; font-size:0.8rem !important; font-weight:600 !important;
}}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div {{
    font-family:'Poppins',sans-serif !important; font-weight:600 !important;
    color:#FFFFFF !important; font-size:1.85rem !important;
}}

/* ── Tabs (old + new Streamlit markup) ── */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{
    gap:0.4rem; background:rgba(255,255,255,0.06);
    padding:0.35rem; border-radius:14px;
    border:1px solid rgba(167,243,208,0.18);
}}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {{
    height:44px; padding:0 1.6rem; background:transparent;
    border-radius:10px; border:none;
}}
.stTabs [role="tab"] p {{ color:{G_SOFT}; font-weight:600; font-size:0.93rem; }}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,{G_ACCENT},{G_DEEP}) !important;
    box-shadow:0 4px 18px rgba(52,211,153,0.35);
}}
.stTabs [aria-selected="true"] p {{ color:#FFFFFF !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
.stTabs .react-aria-SelectionIndicator {{ display:none !important; }}

/* ── Input labels ── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label,
[data-testid="stWidgetLabel"] p {{
    color:{G_SOFT} !important; font-size:0.85rem !important; font-weight:600 !important;
}}

/* ── Input boxes: solid green surface + white text (visible while typing) ── */
.stApp input, .stApp textarea {{
    background:{G_SURFACE} !important;
    color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
    caret-color:{G_ACCENT} !important; opacity:1 !important;
}}
.stApp input::placeholder {{ color:{G_MINT} !important; -webkit-text-fill-color:{G_MINT} !important; }}
.stApp [data-testid="stNumberInputContainer"],
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] div:has(> input),
.stApp [data-testid="stMultiSelect"] div:has(> input),
.stApp [data-testid="stMultiSelect"] div:has(> div > input),
.stApp [data-testid="stMultiSelect"] div:has(> div > div > input) {{
    background:{G_SURFACE} !important;
    border:1px solid rgba(167,243,208,0.28) !important;
    border-radius:10px !important;
}}
.stApp [data-testid="stNumberInputContainer"]:focus-within,
.stApp [data-baseweb="select"] > div:focus-within {{ border-color:{G_ACCENT} !important; }}
.stApp [data-baseweb="select"] div,
.stApp [data-testid="stSelectbox"] div:has(> input) div {{
    color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
}}
.stApp [data-baseweb="select"] svg, .stApp [data-testid="stNumberInputContainer"] svg,
.stApp [data-testid="stNumberInputContainer"] button {{
    fill:{G_MINT} !important; color:{G_MINT} !important; background:transparent !important;
}}
/* Dropdown option list */
[data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"] {{ background:{G_SURFACE} !important; }}
[data-baseweb="popover"] li, [role="listbox"] [role="option"],
[data-baseweb="popover"] li *, [role="listbox"] [role="option"] * {{ color:#FFFFFF !important; }}
[data-baseweb="popover"] li:hover, [role="listbox"] [role="option"]:hover,
[role="listbox"] [aria-selected="true"] {{ background:rgba(52,211,153,0.30) !important; }}
/* Channel chips */
.stApp [data-baseweb="tag"], .stApp [data-tag] {{ background:{G_ACCENT} !important; }}
.stApp [data-baseweb="tag"] *, .stApp [data-tag] * {{ color:#063D2B !important; fill:#063D2B !important; }}
/* Slider (fallback if config.toml is missing) */
.stApp [data-testid="stSlider"] [role="slider"] {{ background:{G_ACCENT} !important; border-color:{G_ACCENT} !important; }}
.stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"] {{ color:{G_MINT} !important; }}
/* Inline code chips ("Active: Email") */
.stApp [data-testid="stMarkdownContainer"] code {{
    background:rgba(52,211,153,0.18) !important; color:{G_MINT} !important;
    border-radius:6px; padding:2px 8px;
}}

/* ── Group label ── */
.grp {{
    font-family:'Poppins',sans-serif;
    font-size:0.7rem; letter-spacing:0.18em; text-transform:uppercase;
    color:{G_ACCENT}; font-weight:600; margin-bottom:0.5rem;
    padding-bottom:0.4rem; border-bottom:1px solid rgba(167,243,208,0.18);
}}

/* ── Button ── */
.stButton > button {{
    background:linear-gradient(135deg,{G_ACCENT},{G_DEEP}) !important;
    color:#FFFFFF !important; border:none !important;
    border-radius:12px !important; font-weight:700 !important;
    box-shadow:0 6px 22px rgba(52,211,153,0.30);
    transition:all 0.2s ease !important;
}}
.stButton > button p {{ color:#FFFFFF !important; font-size:1rem !important; font-weight:700 !important; }}
.stButton > button:hover {{
    transform:translateY(-2px) !important;
    box-shadow:0 10px 30px rgba(52,211,153,0.45) !important;
}}

/* ── Result cards ── */
.rcard {{
    background:rgba(255,255,255,0.07);
    border:1px solid rgba(167,243,208,0.22);
    border-radius:18px; padding:1.6rem; margin-top:0.5rem;
}}
.rcard-profit {{ border:1px solid {G_PROFIT}; background:linear-gradient(160deg,rgba(110,231,183,0.20),rgba(110,231,183,0.05)); }}
.rcard-loss   {{ border:1px dashed {G_LOSS};  background:linear-gradient(160deg,rgba(203,213,225,0.16),rgba(203,213,225,0.04)); }}
.big-val {{ font-family:'Poppins',sans-serif; font-size:2.5rem; font-weight:700; line-height:1.1; margin:0.25rem 0; color:#FFFFFF; }}
.mlabel {{ font-size:0.7rem; letter-spacing:0.14em; text-transform:uppercase; color:{G_MINT}; font-weight:700; }}
.desc   {{ color:{G_SOFT}; margin:0.4rem 0 1.1rem; font-size:0.92rem; }}
.stat   {{ color:#FFFFFF; font-size:1.3rem; font-weight:700; margin-top:2px; }}
.stat-sm{{ color:#FFFFFF; font-size:0.98rem; font-weight:600; }}
.stats  {{ display:flex; gap:2.5rem; flex-wrap:wrap; }}
.grid2  {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}

/* ── Insight + notice ── */
.insight, .notice {{
    background:rgba(255,255,255,0.06);
    border-left:3px solid {G_ACCENT};
    border-radius:0 12px 12px 0;
    padding:1rem 1.2rem; margin-top:1rem;
    color:{G_SOFT}; font-size:0.95rem; line-height:1.6;
}}
.insight strong, .notice strong {{ color:#FFFFFF; }}
.notice {{ border-left-color:{G_MINT}; color:#FFFFFF; margin-bottom:0.5rem; }}

/* ── Section title ── */
.stab-title {{ font-family:'Poppins',sans-serif; font-size:1.55rem; font-weight:600; color:#FFFFFF; margin:0.2rem 0 0.1rem; }}
.stab-sub   {{ color:{G_SOFT}; font-size:0.92rem; margin-bottom:1rem; }}

/* ── Expander / tables ── */
[data-testid="stExpander"] details {{
    background:rgba(255,255,255,0.05) !important; border-radius:10px !important;
    border:1px solid rgba(167,243,208,0.2) !important;
}}
[data-testid="stExpander"] summary p {{ color:{G_SOFT} !important; }}
hr {{ border-color:rgba(167,243,208,0.15); }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
def locate(filename):
    for folder in (APP_DIR, PROJECT_DIR, APP_DIR / "models", APP_DIR.parent / "models"):
        path = folder / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"{filename} not found. Put it next to dash.py or in {PROJECT_DIR}")


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    with locate("regression_model.pkl").open("rb") as f:
        reg = pickle.load(f)
    with locate("classification_model.pkl").open("rb") as f:
        clf = pickle.load(f)
    # Compatibility: LogisticRegression pickled on a different sklearn version
    model = clf.get("model")
    if model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return reg, clf


try:
    reg_pkg, clf_pkg = load_models()
    models_loaded = True
except Exception as e:  # noqa: BLE001
    models_loaded = False
    load_error = f"{type(e).__name__}: {e}"

REG_FEATURES = list(reg_pkg["features"]) if models_loaded else []
CLF_FEATURES = list(clf_pkg["features"]) if models_loaded else []


def options_for(group):
    """Dropdown options read from the trained feature names (always in sync)."""
    prefix = group + "_"
    return [f[len(prefix):] for f in CLF_FEATURES if f.startswith(prefix)]


# ══════════════════════════════════════════════════════
# HERO + KPI
# ══════════════════════════════════════════════════════
html("""
<div class="hero">
    <span class="hero-badge">✦ Nykaa · Purplle · Tira</span>
    <h1 class="hero-title">AI-powered campaign <span class="acc">intelligence</span></h1>
    <p class="hero-sub">Forecast revenue and predict profitability before your campaign goes live</p>
</div>
""")

st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Campaigns", "166,665")
k2.metric("Revenue R² Score", f"{reg_pkg['r2_score']*100:.2f}%" if models_loaded else "99.49%")
k3.metric("Profit Accuracy", f"{clf_pkg['accuracy']*100:.2f}%" if models_loaded else "97.27%")
k4.metric("Loss Recall", f"{clf_pkg['loss_recall']*100:.2f}%" if models_loaded else "97.11%")
st.markdown("<br>", unsafe_allow_html=True)

if not models_loaded:
    notice(f"Could not load models — {load_error}")
    st.caption("Check both .pkl files are in the project folder and run: "
               "pip install streamlit pandas numpy scikit-learn xgboost plotly")
    st.stop()


# ══════════════════════════════════════════════════════
# HELPERS — model input
# ══════════════════════════════════════════════════════
def build_features(inputs, feature_cols):
    """Turn form inputs into the exact column order the model was trained on."""
    row = {col: 0.0 for col in feature_cols}
    numeric = {
        "Duration": inputs["duration"], "Impressions": inputs["impressions"],
        "Clicks": inputs["clicks"], "Leads": inputs["leads"],
        "Conversions": inputs["conversions"], "Acquisition_Cost": inputs["acq_cost"],
        "Engagement_Score": inputs["engagement"],
    }
    for k, v in numeric.items():
        if k in row:
            row[k] = float(v)

    # Updated_ROI = (Revenue - Acquisition_Cost) / Acquisition_Cost (as in the notebook),
    # with Revenue estimated from conversions before launch.
    if "Updated_ROI" in row:
        est_revenue = inputs["conversions"] * REVENUE_PER_CONVERSION
        row["Updated_ROI"] = (est_revenue - inputs["acq_cost"]) / max(inputs["acq_cost"], 1)

    for ch in inputs["channels"]:
        if ch in row:
            row[ch] = 1.0

    # Brand is used only if the model was trained with Brand_ columns
    for group, key in [("Brand", "brand"), ("Campaign_Type", "campaign_type"),
                       ("Target_Audience", "target_audience"), ("Language", "language"),
                       ("Customer_Segment", "customer_segment")]:
        col = f"{group}_{inputs[key]}"
        if col in row:
            row[col] = 1.0

    return pd.DataFrame([row])[feature_cols]


def input_form(prefix):
    col1, col2 = st.columns(2, gap="large")
    with col1:
        html('<div class="grp">Brand & Campaign</div>')
        brand = st.selectbox("Brand", BRANDS, key=f"{prefix}_b")
        campaign_type = st.selectbox("Campaign Type", options_for("Campaign_Type"), key=f"{prefix}_ct")
        target_audience = st.selectbox("Target Audience", options_for("Target_Audience"), key=f"{prefix}_ta")
        language = st.selectbox("Language", options_for("Language"), key=f"{prefix}_lang")
        customer_segment = st.selectbox("Customer Segment", options_for("Customer_Segment"), key=f"{prefix}_seg")

        html('<div class="grp" style="margin-top:1rem;">Performance Metrics</div>')
        a, b = st.columns(2)
        impressions = a.number_input("Impressions", 10000, 100000, 55000, step=1000, key=f"{prefix}_imp")
        clicks = b.number_input("Clicks", 200, 15000, 3900, step=100, key=f"{prefix}_clk")
        c, d = st.columns(2)
        leads = c.number_input("Leads", 50, 9000, 1476, step=50, key=f"{prefix}_lds")
        conversions = d.number_input("Conversions", 10, 7000, 775, step=50, key=f"{prefix}_conv")

    with col2:
        html('<div class="grp">Campaign Settings</div>')
        acq_cost = st.number_input("Acquisition Cost (₹)", 8.0, 15000.0, 376.0, step=10.0, key=f"{prefix}_cost")
        duration = st.slider("Duration (days)", 5, 30, 11, key=f"{prefix}_dur")
        engagement = st.slider("Engagement Score", 0.0, 30.0, 13.5, step=0.1, key=f"{prefix}_eng")

        html('<div class="grp" style="margin-top:1rem;">Marketing Channels</div>')
        channels = st.multiselect(
            "Select channels used",
            [c for c in CHANNEL_NAMES if c in CLF_FEATURES],
            key=f"{prefix}_ch",
        )
        if channels:
            st.markdown("**Active:** " + "  ".join(f"`{ch}`" for ch in channels))
        else:
            st.caption("No channels selected yet")

    return dict(brand=brand, campaign_type=campaign_type, target_audience=target_audience,
                language=language, customer_segment=customer_segment, duration=duration,
                impressions=impressions, clicks=clicks, leads=leads, conversions=conversions,
                acq_cost=acq_cost, engagement=engagement, channels=channels)


def validate(inputs):
    if not inputs["channels"]:
        notice("Please select at least one marketing channel"); return False
    if inputs["conversions"] > inputs["leads"]:
        notice("Conversions cannot exceed Leads"); return False
    if inputs["leads"] > inputs["clicks"]:
        notice("Leads cannot exceed Clicks"); return False
    if inputs["clicks"] > inputs["impressions"]:
        notice("Clicks cannot exceed Impressions"); return False
    if inputs["acq_cost"] <= 0:
        notice("Acquisition Cost must be greater than 0"); return False
    return True


def summary_table(inputs):
    df = pd.DataFrame({
        "Feature": ["Brand", "Campaign Type", "Target Audience", "Language", "Customer Segment",
                    "Duration", "Impressions", "Clicks", "Leads", "Conversions",
                    "Acquisition Cost", "Engagement Score", "Channels"],
        "Value": [inputs["brand"], inputs["campaign_type"], inputs["target_audience"],
                  inputs["language"], inputs["customer_segment"], f"{inputs['duration']} days",
                  f"{inputs['impressions']:,}", f"{inputs['clicks']:,}", f"{inputs['leads']:,}",
                  f"{inputs['conversions']:,}", f"₹{inputs['acq_cost']:,.2f}",
                  f"{inputs['engagement']}", ", ".join(inputs["channels"]) or "None"],
    })
    stretch(st.dataframe, df, hide_index=True)


# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "💰  Revenue Prediction",
    "📈  Profit / Loss Prediction",
    "ℹ️  Model Insights",
])

# ════════════════════════════════════════
# TAB 1 — REVENUE
# ════════════════════════════════════════
with tab1:
    html('<div class="stab-title">Revenue Forecast</div>')
    html(f'<div class="stab-sub">Estimate how much revenue this campaign will generate using '
         f'XGBoost (R² {reg_pkg["r2_score"]*100:.2f}%)</div>')

    inputs = input_form("reg")
    st.markdown("<br>", unsafe_allow_html=True)

    if stretch(st.button, "🚀  Forecast Revenue", key="reg_btn"):
        if validate(inputs):
            with st.spinner("Running XGBoost model..."):
                X = build_features(inputs, REG_FEATURES)
                revenue = max(0.0, float(reg_pkg["model"].predict(X)[0]))

            st.markdown("---")
            st.markdown("### 🎯 Revenue Prediction Result")
            r1, r2 = st.columns(2, gap="large")

            with r1:
                roi = (revenue - inputs["acq_cost"]) / max(inputs["acq_cost"], 1)
                if revenue < 100000:
                    tier = "Low revenue · under ₹1 Lakh"
                elif revenue < 500000:
                    tier = "Average revenue · ₹1L – ₹5L"
                elif revenue < 1000000:
                    tier = "Good revenue · ₹5L – ₹10L"
                else:
                    tier = "High revenue · above ₹10 Lakh"

                html(f"""
                <div class="rcard rcard-profit">
                    <div class="mlabel">Predicted Revenue</div>
                    <div class="big-val">{inr(revenue)}</div>
                    <p class="desc">{tier}</p>
                    <div class="stats">
                        <div><div class="mlabel">Model R²</div>
                             <div class="stat">{reg_pkg['r2_score']*100:.2f}%</div></div>
                        <div><div class="mlabel">Estimated ROI</div>
                             <div class="stat">{roi:,.2f}×</div></div>
                        <div><div class="mlabel">Avg Model Error</div>
                             <div class="stat">{inr(reg_pkg['rmse'])}</div></div>
                    </div>
                </div>""")

                if roi > 2:
                    msg = "✅ <strong>Profitable campaign.</strong> Proceed with confidence — estimated ROI is strong."
                elif roi > 0:
                    msg = "✅ <strong>Profitable campaign.</strong> Campaign is profitable but consider optimising channels."
                else:
                    msg = "❌ <strong>Loss-making campaign.</strong> Review acquisition cost and channel strategy."
                html(f'<div class="insight">{msg}</div>')

            with r2:
                gauge_max = max(2500000, revenue * 1.15)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=revenue,
                    number={"prefix": "₹", "valueformat": ",.0f", "font": {"color": "#FFFFFF", "size": 28}},
                    gauge={"axis": {"range": [0, gauge_max], "tickcolor": G_MINT,
                                    "tickfont": {"color": G_MINT, "size": 9}},
                           "bar": {"color": G_ACCENT}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                           "steps": [{"range": [0, 100000], "color": "rgba(167,243,208,0.08)"},
                                     {"range": [100000, 500000], "color": "rgba(167,243,208,0.16)"},
                                     {"range": [500000, 1000000], "color": "rgba(110,231,183,0.26)"},
                                     {"range": [1000000, gauge_max], "color": "rgba(52,211,153,0.38)"}]}))
                fig.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30),
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="#FFFFFF",
                                  font_family="Nunito Sans")
                stretch(st.plotly_chart, fig, key="reg_gauge")

            with st.expander("📋  Campaign Input Summary"):
                summary_table(inputs)

# ════════════════════════════════════════
# TAB 2 — PROFIT / LOSS
# ════════════════════════════════════════
with tab2:
    html('<div class="stab-title">Profit / Loss Analysis</div>')
    html(f'<div class="stab-sub">Predict whether this campaign will be profitable using '
         f'Logistic Regression + SMOTE ({clf_pkg["accuracy"]*100:.2f}% accuracy)</div>')

    inputs_clf = input_form("clf")
    st.markdown("<br>", unsafe_allow_html=True)

    if stretch(st.button, "📊  Analyse Profitability", key="clf_btn"):
        if validate(inputs_clf):
            with st.spinner("Running Logistic Regression model..."):
                X = build_features(inputs_clf, CLF_FEATURES)
                # scaler was fitted on a NumPy array → pass NumPy
                X_sc = clf_pkg["scaler"].transform(X.to_numpy(dtype=float))
                prob = clf_pkg["model"].predict_proba(X_sc)[0]
                classes = list(clf_pkg["model"].classes_)
                p_profit = round(float(prob[classes.index(1)]) * 100, 1)
                p_loss = round(float(prob[classes.index(0)]) * 100, 1)

            st.markdown("---")
            st.markdown("### 🎯 Profit / Loss Result")
            r1, r2 = st.columns(2, gap="large")

            is_profit = p_profit >= 50
            with r1:
                card_cls = "rcard rcard-profit" if is_profit else "rcard rcard-loss"
                verdict = "Profitable ✓" if is_profit else "Loss ✗"
                conf = p_profit if is_profit else p_loss
                desc = ("This campaign is expected to generate profit." if is_profit
                        else "This campaign is expected to result in a loss.")

                html(f"""
                <div class="{card_cls}">
                    <div class="mlabel">Prediction</div>
                    <div class="big-val">{verdict}</div>
                    <p class="desc">{desc}</p>
                    <div class="stats">
                        <div><div class="mlabel">Confidence</div>
                             <div class="stat">{conf}%</div></div>
                        <div><div class="mlabel">Model Accuracy</div>
                             <div class="stat">{clf_pkg['accuracy']*100:.1f}%</div></div>
                        <div><div class="mlabel">Loss Recall</div>
                             <div class="stat">{clf_pkg['loss_recall']*100:.1f}%</div></div>
                    </div>
                </div>""")

                if is_profit:
                    msg = ("✅ <strong>Go ahead!</strong> Campaign shows strong profit signals. "
                           "Consider scaling budget on top channels.")
                else:
                    msg = ("⚠️ <strong>Review required.</strong> Lower the acquisition cost or "
                           "improve conversions, then re-check.")
                html(f'<div class="insight">{msg}</div>')

            with r2:
                fig = go.Figure(go.Pie(
                    labels=["Profit", "Loss"], values=[p_profit, p_loss], hole=0.62,
                    marker_colors=[G_ACCENT, G_LOSS], sort=False,
                    textinfo="label+percent",
                    textfont={"color": "#063D2B", "size": 13},
                    hovertemplate="%{label}: %{value}%<extra></extra>"))
                fig.update_layout(
                    height=300, margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font_family="Nunito Sans",
                    annotations=[{"text": "Profit" if is_profit else "Loss",
                                  "x": 0.5, "y": 0.5, "font_size": 22,
                                  "font_color": "#FFFFFF", "showarrow": False}])
                stretch(st.plotly_chart, fig, key="clf_donut")

            with st.expander("📋  Campaign Input Summary"):
                summary_table(inputs_clf)

# ════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════
with tab3:
    html('<div class="stab-title">Model Insights</div>')
    html('<div class="stab-sub">Performance metrics, training details and dataset overview</div>')

    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("#### 💰 Regression Model")
        html(f"""
        <div class="rcard">
            <div class="mlabel">Model</div>
            <div class="stat" style="margin-bottom:1rem;">{reg_pkg['model_name']}</div>
            <div class="grid2">
                <div><div class="mlabel">R² Score</div><div class="stat">{reg_pkg['r2_score']*100:.2f}%</div></div>
                <div><div class="mlabel">RMSE</div><div class="stat">{inr(reg_pkg['rmse'])}</div></div>
                <div><div class="mlabel">MAE</div><div class="stat">{inr(reg_pkg['mae'])}</div></div>
                <div><div class="mlabel">Train/Test Gap</div><div class="stat">{reg_pkg['gap']:.4f}</div></div>
                <div><div class="mlabel">Trained On</div><div class="stat-sm">{reg_pkg['trained_on']}</div></div>
                <div><div class="mlabel">Features</div><div class="stat-sm">{len(REG_FEATURES)} columns</div></div>
            </div>
        </div>""")

    with m2:
        st.markdown("#### 📈 Classification Model")
        html(f"""
        <div class="rcard">
            <div class="mlabel">Model</div>
            <div class="stat" style="margin-bottom:1rem;">{clf_pkg['model_name']} + SMOTE</div>
            <div class="grid2">
                <div><div class="mlabel">Accuracy</div><div class="stat">{clf_pkg['accuracy']*100:.2f}%</div></div>
                <div><div class="mlabel">F1 Score</div><div class="stat">{clf_pkg['f1_score']*100:.2f}%</div></div>
                <div><div class="mlabel">ROC-AUC</div><div class="stat">{clf_pkg['roc_auc']:.4f}</div></div>
                <div><div class="mlabel">Loss Recall</div><div class="stat">{clf_pkg['loss_recall']*100:.2f}%</div></div>
                <div><div class="mlabel">Trained On</div><div class="stat-sm">{clf_pkg['trained_on']}</div></div>
                <div><div class="mlabel">SMOTE Used</div><div class="stat-sm">{'Yes ✅' if clf_pkg.get('smote_used') else 'No'}</div></div>
            </div>
        </div>""")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Dataset Overview")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Campaigns", "166,665")
    d2.metric("Brands", "3")
    d3.metric("Features", f"{len(CLF_FEATURES)}")
    d4.metric("Profit Campaigns", "95.01%")
    d5.metric("Loss Campaigns", "4.99%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🏆 All Models Compared")

    # Results from Market _Analysis_Prediction.ipynb
    reg_df = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression",
                  "Decision Tree", "Random Forest", "XGBoost ✅"],
        "R² Score": ["77.11%", "77.11%", "77.11%", "99.22%", "99.65%", "99.49%"],
        "RMSE": ["₹2,22,877", "₹2,22,877", "₹2,22,877", "₹41,012", "₹27,373", "₹33,326"],
        "Selected": ["", "", "", "", "", "✅ Winner"],
    })
    st.markdown("**Regression Models**")
    stretch(st.dataframe, reg_df, hide_index=True)

    clf_df = pd.DataFrame({
        "Model": ["Logistic Regression ✅", "Decision Tree", "Random Forest",
                  "XGBoost", "KNN", "Naive Bayes", "Gradient Boosting",
                  "AdaBoost", "SVM", "LightGBM"],
        "Accuracy": ["97.27%", "100%*", "100%*", "99.99%", "82.98%",
                     "100%*", "100%*", "100%*", "98.91%", "100%*"],
        "Loss Recall": ["97.11%", "100%*", "100%*", "99.88%", "18.10%",
                        "100%*", "100%*", "100%*", "98.02%", "100%*"],
        "Note": ["✅ Selected", "⚠️ Data leakage", "⚠️ Data leakage",
                 "Good", "Poor recall", "⚠️ Leakage", "⚠️ Leakage",
                 "⚠️ Leakage", "Good", "⚠️ Leakage"],
    })
    st.markdown("**Classification Models (after SMOTE)** *(\\* = Updated_ROI leakage)*")
    stretch(st.dataframe, clf_df, hide_index=True)