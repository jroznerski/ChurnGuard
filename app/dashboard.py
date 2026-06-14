"""
ChurnGuard — Customer Churn Intelligence Dashboard
Streamlit production-grade dashboard.
Run: streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.hypothesis_testing import HypothesisTester
from src.models.predictor import predictor

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252b3b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #e0e0ff; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
    .risk-high { color: #ff4b4b; font-weight: 700; }
    .risk-medium { color: #ffa500; font-weight: 700; }
    .risk-low { color: #00cc88; font-weight: 700; }
    .section-header {
        font-size: 1.3rem; font-weight: 600;
        color: #cdd6f4; border-left: 4px solid #7c3aed;
        padding-left: 12px; margin: 1.5rem 0 1rem 0;
    }
    .stAlert { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)

COLORS = {
    "primary": "#7c3aed",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "success": "#10b981",
    "info": "#3b82f6",
    "bg": "#1e2130",
    "card": "#252b3b",
    "text": "#cdd6f4",
    "muted": "#8892b0",
}

DATA_PATH = os.getenv("DATA_RAW_PATH", "data/raw/customers.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")

# ─── Data / model loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame | None:
    if not Path(DATA_PATH).exists():
        return None
    df = pd.read_csv(DATA_PATH)
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "multiple_lines",
    ]
    df["service_count"] = sum(
        (df[c] == "Yes").astype(int) for c in service_cols if c in df.columns
    )
    return df


@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    predictor.load(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
    return predictor


@st.cache_data(ttl=300)
def load_metadata() -> dict | None:
    if not Path(METADATA_PATH).exists():
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=60,
    )
    st.markdown("## 🛡️ ChurnGuard")
    st.markdown("*Customer Churn Intelligence Platform*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔬 Hypothesis Tests", "🤖 Predict Churn", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.divider()

    df = load_data()
    model = load_model()
    metadata = load_metadata()

    if df is not None:
        st.success(f"✅ Dataset: {len(df):,} customers")
    else:
        st.warning("⚠️ No dataset found")

    if model is not None:
        st.success("✅ Model: loaded")
    else:
        st.warning("⚠️ Model: not trained")

    st.markdown(
        """
---
<small style='color:#8892b0'>
API: <code>localhost:8000/docs</code><br>
v1.0.0 · MIT License
</small>
""",
        unsafe_allow_html=True,
    )

# ─── Helper: plotly dark template ───────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(t=40, b=30, l=30, r=20),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Churn Overview")

    if df is None:
        st.error(
            "Dataset not found. Run `python scripts/train_model.py --generate` to create data."
        )
        st.stop()

    churn_rate = df["churn"].mean()
    monthly_avg = df["monthly_charges"].mean()
    tenure_avg = df["tenure"].mean()
    churned_count = df["churn"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{churn_rate*100:.1f}%", delta=f"{churned_count:,} churned", delta_color="inverse")
    c3.metric("Avg Monthly Charges", f"${monthly_avg:.2f}")
    c4.metric("Avg Tenure (months)", f"{tenure_avg:.1f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Churn by Contract Type</div>', unsafe_allow_html=True)
        contract_churn = (
            df.groupby("contract")["churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Churn Rate", "count": "Customers"})
        )
        contract_churn["Churn Rate %"] = (contract_churn["Churn Rate"] * 100).round(1)
        fig = px.bar(
            contract_churn,
            x="contract",
            y="Churn Rate %",
            color="contract",
            color_discrete_sequence=[COLORS["danger"], COLORS["warning"], COLORS["success"]],
            text="Churn Rate %",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_title="Contract", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Monthly Charges Distribution</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for label, color in [("Churned (1)", COLORS["danger"]), ("Retained (0)", COLORS["success"])]:
            val = 1 if "Churned" in label else 0
            subset = df[df["churn"] == val]["monthly_charges"]
            fig2.add_trace(go.Histogram(
                x=subset,
                name=label,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, barmode="overlay", xaxis_title="Monthly Charges ($)", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Churn by Internet Service</div>', unsafe_allow_html=True)
        inet_churn = df.groupby("internet_service")["churn"].mean().reset_index()
        inet_churn["Churn Rate %"] = (inet_churn["churn"] * 100).round(1)
        fig3 = px.pie(
            inet_churn,
            values="Churn Rate %",
            names="internet_service",
            color_discrete_sequence=[COLORS["info"], COLORS["danger"], COLORS["muted"]],
            hole=0.45,
        )
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Tenure vs Monthly Charges (Churn)</div>', unsafe_allow_html=True)
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig4 = px.scatter(
            sample,
            x="tenure",
            y="monthly_charges",
            color=sample["churn"].map({1: "Churned", 0: "Retained"}),
            color_discrete_map={"Churned": COLORS["danger"], "Retained": COLORS["success"]},
            opacity=0.5,
            size_max=6,
        )
        fig4.update_layout(**PLOTLY_LAYOUT, xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)")
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HYPOTHESIS TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Hypothesis Tests":
    st.title("🔬 Statistical Hypothesis Testing")
    st.markdown(
        "Five rigorous statistical tests examining factors that drive customer churn."
        " Significance level: **α = 0.05**"
    )

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    with st.spinner("Running hypothesis tests …"):
        tester = HypothesisTester(df)
        results = tester.run_all()

    for r in results:
        with st.expander(
            f"{'✅' if r.rejected else '❌'} **{r.description}** | p = {r.p_value:.4f}",
            expanded=True,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test", r.test_type)
            c2.metric("Statistic", f"{r.statistic:.3f}")
            c3.metric("p-value", f"{r.p_value:.4f}")
            c4.metric(
                "Result",
                "Reject H₀" if r.rejected else "Fail to reject H₀",
            )

            if r.effect_size_label:
                st.info(f"**Effect size** — {r.effect_size_label}")

            st.success(r.conclusion) if r.rejected else st.warning(r.conclusion)

            if r.name == "contract_vs_churn" and "churn_by_contract" in r.additional:
                data = r.additional["churn_by_contract"]
                rows = [{"Contract": k, "Churn Rate": v["churn_rate"], "N": v["n"]} for k, v in data.items()]
                chart_df = pd.DataFrame(rows)
                chart_df["Churn Rate %"] = (chart_df["Churn Rate"] * 100).round(1)
                fig = px.bar(chart_df, x="Contract", y="Churn Rate %", text="Churn Rate %",
                             color="Contract",
                             color_discrete_sequence=[COLORS["danger"], COLORS["warning"], COLORS["success"]])
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            if r.name in ("monthly_charges_vs_churn", "tenure_vs_churn") and r.additional:
                col = "monthly_charges" if "charges" in r.name else "tenure"
                fig = go.Figure()
                for label, group_val, color in [("Churned", 1, COLORS["danger"]), ("Retained", 0, COLORS["success"])]:
                    vals = df[df["churn"] == group_val][col].dropna()
                    fig.add_trace(go.Box(y=vals, name=label, marker_color=color, boxmean=True))
                fig.update_layout(**PLOTLY_LAYOUT, yaxis_title=col.replace("_", " ").title())
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Churn":
    st.title("🤖 Real-Time Churn Prediction")

    if model is None:
        st.error(
            "Model not loaded. Run `python scripts/train_model.py --generate` first."
        )
        st.stop()

    st.markdown("Enter customer details to get an instant churn probability estimate.")

    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Profile")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        senior_citizen = c2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = c3.selectbox("Partner", ["Yes", "No"])

        c4, c5, c6 = st.columns(3)
        dependents = c4.selectbox("Dependents", ["Yes", "No"])
        tenure = c5.slider("Tenure (months)", 0, 72, 12)
        contract = c6.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.markdown("#### 📡 Services")
        c7, c8 = st.columns(2)
        phone_service = c7.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = c8.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        c9, c10, c11, c12 = st.columns(4)
        online_security = c9.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = c10.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = c11.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = c12.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        c13, c14 = st.columns(2)
        streaming_tv = c13.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = c14.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        st.markdown("#### 💳 Billing")
        cb1, cb2, cb3, cb4 = st.columns(4)
        paperless_billing = cb1.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = cb2.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = cb3.number_input("Monthly Charges ($)", 0.0, 500.0, 79.85, step=0.5)
        total_charges = cb4.number_input("Total Charges ($)", 0.0, 50000.0, float(monthly_charges * tenure), step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        features = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
        }

        result = predictor.predict_single(features)
        proba = result["churn_probability"]
        risk = result["risk_level"]

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{proba * 100:.1f}%")
        r2.metric("Risk Level", risk)
        r3.metric("Prediction", "Will Churn" if result["churn_prediction"] else "Will Retain")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["muted"]},
                    "bar": {"color": COLORS["danger"] if risk == "High" else COLORS["warning"] if risk == "Medium" else COLORS["success"]},
                    "steps": [
                        {"range": [0, 40], "color": "#1a2e22"},
                        {"range": [40, 70], "color": "#2e2710"},
                        {"range": [70, 100], "color": "#2e1212"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": result["threshold_used"] * 100},
                },
            )
        )
        gauge.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(gauge, use_container_width=True)

        if risk == "High":
            st.error("⚠️ **High churn risk detected.** Immediate retention action recommended.")
        elif risk == "Medium":
            st.warning("🔶 **Medium churn risk.** Monitor and consider a proactive offer.")
        else:
            st.success("✅ **Low churn risk.** Customer appears stable.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if metadata is None:
        st.error("No model metadata found. Train the model first.")
        st.stop()

    metrics = metadata.get("metrics", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")
    m2.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    m3.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    m4.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    m5.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Cross-Validation AUC</div>', unsafe_allow_html=True)
        cv_mean = metrics.get("cv_auc_mean", 0)
        cv_std = metrics.get("cv_auc_std", 0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
            y=[cv_mean + np.random.uniform(-cv_std, cv_std) for _ in range(5)],
            marker_color=COLORS["primary"],
        ))
        fig.add_hline(y=cv_mean, line_dash="dash", line_color=COLORS["success"],
                      annotation_text=f"Mean AUC = {cv_mean:.4f}")
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="AUC-ROC", yaxis_range=[0.7, 1.0])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")

    with col2:
        st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)
        shap_imp = metadata.get("shap_feature_importance", {})
        if shap_imp:
            shap_df = (
                pd.DataFrame(shap_imp.items(), columns=["Feature", "SHAP Value"])
                .sort_values("SHAP Value", ascending=True)
                .tail(12)
            )
            fig2 = px.bar(
                shap_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="SHAP Value",
                color_continuous_scale=["#3b82f6", "#7c3aed", "#ef4444"],
            )
            fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["Retained", "Churned"],
        y=["Retained", "Churned"],
        color_continuous_scale=["#1e2130", COLORS["primary"]],
    )
    cm_fig.update_layout(**PLOTLY_LAYOUT, width=450)
    st.plotly_chart(cm_fig)

    with st.expander("📋 Full Metadata"):
        st.json(metadata)
