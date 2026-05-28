"""
app.py — Streamlit Dashboard
Boxing Performance Prediction System
Hala'o: streamlit run dashboard/app.py
Autór: Joao Nuno Urle Pereira Boavida
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(
    page_title="Boxing Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Animasaun + Style ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Animasaun tombol predisaun ── */
div.stButton > button {
    background: linear-gradient(135deg, #1B3A6B, #2980B9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(41,128,185,0.35);
}
div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(41,128,185,0.50);
    background: linear-gradient(135deg, #2471A3, #1B3A6B);
}
div.stButton > button:active {
    transform: translateY(1px) scale(0.98);
    box-shadow: 0 2px 8px rgba(41,128,185,0.30);
}

/* ── Animasaun metric cards ── */
div[data-testid="metric-container"] {
    background: #f8fbff;
    border: 1px solid #d6e8f7;
    border-radius: 12px;
    padding: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 20px rgba(41,128,185,0.15);
}

/* ── Animasaun hasil predisaun masuk ── */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(46,204,113,0.5); }
    70%  { box-shadow: 0 0 0 16px rgba(46,204,113,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,204,113,0); }
}
.result-card {
    animation: slideIn 0.45s ease, pulse 1.2s ease 0.45s;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-bottom: 18px;
}

/* ── Animasaun tab hover ── */
button[data-baseweb="tab"] {
    transition: color 0.2s ease;
}
button[data-baseweb="tab"]:hover {
    color: #2980B9 !important;
}

/* ── Sidebar style ── */
section[data-testid="stSidebar"] {
    background: #f0f4f9;
}

/* ── Footer kredit ── */
.footer-credit {
    text-align: center;
    padding: 18px 0 6px;
    color: #7f8c8d;
    font-size: 13px;
}
.footer-credit span {
    color: #1B3A6B;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

COLORS     = {"Low": "#e74c3c", "Medium": "#f39c12", "Peak": "#2ecc71"}
PHASE_ORDER= ["GPP", "SPP", "Competition", "Taper"]
LABEL_ORDER= ["Low", "Medium", "Peak"]

@st.cache_data
def load_data():
    path = "data/raw/boxing_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    from data_generator import generate_dataset
    df = generate_dataset()
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(path, index=False)
    return df

@st.cache_resource
def load_model():
    path = "models/random_forest_model.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Boxing Performance Prediction Dashboard")
st.markdown(
    "Sistema predisaun nível performansa atleta boxe uza **Random Forest** "
    "bazeia ba markadór biokimiku no indikadór performansa fíziku "
    "iha **20-Week Macrocycle**."
)

df    = load_data()
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filter Dadus")
st.sidebar.caption("Hili atleta no faze treinamentu")

selected_athletes = st.sidebar.multiselect(
    "Atleta (Athlete ID)",
    options=sorted(df["athlete_id"].unique()),
    default=sorted(df["athlete_id"].unique())[:5],
)
selected_phases = st.sidebar.multiselect(
    "Faze Treinamentu (Training Phase)",
    options=PHASE_ORDER,
    default=PHASE_ORDER,
)

df_filtered = df[
    (df["athlete_id"].isin(selected_athletes)) &
    (df["phase"].isin(selected_phases))
]

st.sidebar.divider()
st.sidebar.caption("Informasaun Modelu")
st.sidebar.info(
    "Algoritmu: Random Forest\n\n"
    "Accuracy: 90.83%\n\n"
    "AUC-ROC: 97.83%\n\n"
    "Dataset: 600 rekorde"
)

st.sidebar.divider()
st.sidebar.markdown(
    "<div style='text-align:center;font-size:12px;color:#7f8c8d'>"
    "Dezenvolvidu husi<br>"
    "<strong style='color:#1B3A6B;font-size:14px'>Digo Pereira</strong><br>"
    "Siénsia Komputadór · 2024/2025"
    "</div>",
    unsafe_allow_html=True,
)

# ── Metric Cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rekorde",  len(df_filtered))
col2.metric("Atleta Hili",    len(selected_athletes))
col3.metric("Peak Performance", int((df_filtered["performance_level"] == "Peak").sum()))
col4.metric("Avg HRV (ms)",   f"{df_filtered['hrv_ms'].mean():.1f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA — Analiza Dadus",
    "Tren Semana (Weekly Trends)",
    "Korelasaun (Correlation)",
    "Rezultadu Modelu (Model Results)",
    "Predisaun (Prediction)",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Distribuisaun Nível Performansa")
    col_a, col_b = st.columns(2)

    with col_a:
        avail  = [l for l in LABEL_ORDER if l in df_filtered["performance_level"].unique()]
        counts = df_filtered["performance_level"].value_counts()[avail]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values,
                      color=[COLORS[k] for k in counts.index],
                      edgecolor="white", width=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 3, str(val),
                    ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_title("Performance Level Distribution")
        ax.set_xlabel("Performance Level")
        ax.set_ylabel("Nú. Rekorde")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Distribuisaun per Faze Treinamentu**")
        perf_phase = (
            df_filtered
            .groupby(["phase", "performance_level"])
            .size()
            .unstack(fill_value=0)
            .reindex(PHASE_ORDER)
        )
        st.dataframe(perf_phase, use_container_width=True)

    st.markdown("**Estatístika Deskritiva — Descriptive Statistics**")
    numeric_cols = [
        "cortisol_nmol_L", "testosterone_nmol_L", "ck_u_L",
        "lactate_mmol_L", "hrv_ms", "vo2max_ml_kg_min",
        "reaction_time_ms", "punch_speed_m_s",
        "endurance_score", "strength_index",
    ]
    st.dataframe(df_filtered[numeric_cols].describe().round(2), use_container_width=True)

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Tren Feature per Semana (Weekly Average Trend)")
    feature_labels = {
        "cortisol_nmol_L":     "Kortisol (nmol/L)",
        "hrv_ms":              "HRV (ms)",
        "vo2max_ml_kg_min":    "VO2max (ml/kg/min)",
        "punch_speed_m_s":     "Punch Speed (m/s)",
        "endurance_score":     "Endurance Score",
        "training_load":       "Training Load",
        "testosterone_nmol_L": "Testosterona (nmol/L)",
        "ck_u_L":              "CK (U/L)",
    }
    feature_sel = st.selectbox(
        "Hili Feature (Select Feature)",
        options=list(feature_labels.keys()),
        format_func=lambda x: feature_labels[x],
    )
    weekly = df_filtered.groupby("week")[feature_sel].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly["week"], weekly[feature_sel],
            marker="o", linewidth=2, markersize=5, color="#2980B9")
    ax.fill_between(weekly["week"], weekly[feature_sel], alpha=0.08, color="#2980B9")
    for xv, lbl, clr in [(5.5,"GPP | SPP","#3498db"),(12.5,"SPP | Competition","#9b59b6"),(17.5,"Competition | Taper","#e67e22")]:
        ax.axvline(xv, color=clr, ls="--", lw=1.2, alpha=0.7)
        ax.text(xv + 0.2, ax.get_ylim()[1] * 0.97, lbl, fontsize=8, color=clr, va="top")
    ax.set_xlabel("Semana (Week)")
    ax.set_ylabel(feature_labels[feature_sel])
    ax.set_title(f"Média {feature_labels[feature_sel]} per Semana")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Korelasaun Feature — Correlation Heatmap")
    st.caption("Valor positiva = korelasaun direta | Valor negativa = korelasaun inversa")
    corr_cols = [
        "cortisol_nmol_L", "testosterone_nmol_L", "ck_u_L",
        "lactate_mmol_L", "hrv_ms", "vo2max_ml_kg_min",
        "reaction_time_ms", "punch_speed_m_s",
        "endurance_score", "strength_index", "training_load",
    ]
    corr = df_filtered[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Feature Biokimiku no Fíziku")
    st.pyplot(fig)
    plt.close()

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Rezultadu Avaliasaun — Random Forest Model Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",       "90.83%")
    m2.metric("F1-Score",       "90.77%")
    m3.metric("AUC-ROC",        "97.83%")
    m4.metric("CV F1 (5-fold)", "84.48%")

    results_path = "outputs/model_results.txt"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Modelu seidauk treinu. Hala'o `python src/model.py` uluk.")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if os.path.exists("outputs/05_feature_importance.png"):
            st.image("outputs/05_feature_importance.png",
                     caption="Feature Importance — Importánsia Feature")
    with img_col2:
        if os.path.exists("outputs/06_confusion_matrix.png"):
            st.image("outputs/06_confusion_matrix.png",
                     caption="Confusion Matrix — Matris Konfusaun")

# ── Tab 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Predisaun Nível Performansa — Predict Athlete Performance Level")
    st.caption("Input dadus atleta foun hodi hetan predisaun nível performansa")

    if model is None:
        st.warning("Modelu seidauk treinu. Hala'o `python src/model.py` uluk.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Markadór Biokimiku (Biochemical Markers)**")
            cortisol     = st.slider("Kortisol / Cortisol (nmol/L)",          300.0, 800.0, 490.0, 5.0)
            testosterone = st.slider("Testosterona / Testosterone (nmol/L)",    8.0,  35.0,  22.0, 0.5)
            ck           = st.slider("Creatine Kinase — CK (U/L)",             80.0, 500.0, 200.0, 5.0)
            lactate      = st.slider("Laktatu / Lactate (mmol/L)",              1.5,   8.0,   3.5, 0.1)
            hrv          = st.slider("HRV — Heart Rate Variability (ms)",      20.0,  90.0,  55.0, 1.0)
        with col2:
            st.markdown("**Indikadór Fíziku (Physical Performance)**")
            vo2max      = st.slider("VO2max (ml/kg/min)",                      35.0,  75.0,  60.0, 0.5)
            reaction    = st.slider("Tempu Reaksaun / Reaction Time (ms)",    150.0, 300.0, 195.0, 1.0)
            punch_speed = st.slider("Velosidade Tuku / Punch Speed (m/s)",      6.0,  14.0,  11.0, 0.1)
            endurance   = st.slider("Endurance Score",                         20.0, 100.0,  78.0, 1.0)
            strength    = st.slider("Strength Index",                          20.0, 100.0,  77.0, 1.0)
        with col3:
            st.markdown("**Periodizasaun (Periodization)**")
            training_load = st.slider("Training Load",  10.0, 100.0,  50.0, 1.0)
            week          = st.slider("Semana / Week",  1, 20, 15)
            phase         = st.selectbox("Faze / Phase", PHASE_ORDER, index=2)

        st.divider()
        if st.button("Hala'o Predisaun — Run Prediction", use_container_width=True):
            phase_enc = {"GPP": 0, "SPP": 1, "Competition": 2, "Taper": 3}[phase]
            t_c_ratio = testosterone / (cortisol + 1e-6)
            recovery  = ((hrv/90)*0.4 + (1-ck/500)*0.3 + (1-cortisol/800)*0.3)*100
            physical  = (vo2max*0.3 + (1-reaction/300)*100*0.2 +
                         punch_speed*0.2 + endurance*0.15 + strength*0.15)
            cum_load  = training_load * week

            input_data = pd.DataFrame([{
                "cortisol_nmol_L": cortisol, "testosterone_nmol_L": testosterone,
                "ck_u_L": ck, "lactate_mmol_L": lactate, "hrv_ms": hrv,
                "vo2max_ml_kg_min": vo2max, "reaction_time_ms": reaction,
                "punch_speed_m_s": punch_speed, "endurance_score": endurance,
                "strength_index": strength, "training_load": training_load,
                "week": week, "t_c_ratio": t_c_ratio,
                "recovery_score": recovery, "physical_composite": physical,
                "cumulative_load": cum_load, "phase_encoded": phase_enc,
            }])

            try:
                scaler        = joblib.load("models/scaler.joblib")
                input_scaled  = scaler.transform(input_data)
                prediction    = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                label         = LABEL_ORDER[prediction]

                color_map = {"Low": "#e74c3c", "Medium": "#f39c12", "Peak": "#2ecc71"}
                label_tet = {"Low": "Kraik",   "Medium": "Médiu",   "Peak": "Peak"}
                desc_map  = {
                    "Low":    "Karga aas ka rekuperasaun inadekuadu — High load or inadequate recovery",
                    "Medium": "Estadus tranzisaun — Transitional state, progressing well",
                    "Peak":   "Kondisaun otimál ba kompetisaun — Optimal condition for competition",
                }

                st.markdown(
                    f"""
                    <div class='result-card' style='background:{color_map[label]}'>
                        <p style='color:white;margin:0;font-size:13px;opacity:0.88;letter-spacing:1px;
                                  text-transform:uppercase'>
                            Nível Performansa Predita — Predicted Performance Level
                        </p>
                        <h1 style='color:white;margin:10px 0 6px;font-size:42px;font-weight:700'>
                            {label_tet[label]} ({label})
                        </h1>
                        <p style='color:white;margin:0;font-size:13px;opacity:0.85'>
                            {desc_map[label]}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Probabilidade per Klas — Class Probabilities**")
                    prob_df = pd.DataFrame({
                        "Klas (Class)": ["Kraik (Low)", "Médiu (Medium)", "Peak (Peak)"],
                        "Probabilidade": [f"{p:.1%}" for p in probabilities],
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                with rc2:
                    st.markdown("**Feature Kalkuladu — Computed Features**")
                    st.dataframe(pd.DataFrame({
                        "Feature":  ["T/C Ratio", "Recovery Score", "Physical Composite", "Cumulative Load"],
                        "Valor":    [f"{t_c_ratio:.5f}", f"{recovery:.2f}", f"{physical:.2f}", f"{cum_load:.1f}"],
                    }), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Erru predisaun — Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div class='footer-credit'>"
    "Dezenvolvidu husi <span>Joao Nuno Urle Pereira Boavida</span> &nbsp;|&nbsp; "
    "Teze Siénsia Komputadór &nbsp;|&nbsp; "
    "Random Forest Boxing Performance Prediction &nbsp;|&nbsp; "
    "2024 / 2025"
    "</div>",
    unsafe_allow_html=True,
)
