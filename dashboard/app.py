"""
Streamlit Risk Analyst Dashboard for Real-Time Transaction Scoring.

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import requests
import streamlit as st
import pandas as pd

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")
METRICS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics_summary.json")

st.set_page_config(
    page_title="Payment Risk Engine | Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    .badge-allow {
        background-color: #064E3B;
        color: #34D399;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #059669;
        margin-bottom: 8px;
    }
    .badge-challenge {
        background-color: #78350F;
        color: #FBBF24;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #D97706;
        margin-bottom: 8px;
    }
    .badge-block {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #DC2626;
        margin-bottom: 8px;
    }
    .metrics-category-header {
        font-size: 1.1rem;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        margin-top: 8px;
    }
    .cat-ml { background: linear-gradient(135deg, #064E3B, #065F46); color: #A7F3D0; border: 1px solid #10B981; }
    .cat-fin { background: linear-gradient(135deg, #1E3A5F, #1E40AF); color: #93C5FD; border: 1px solid #3B82F6; }
    .cat-funnel { background: linear-gradient(135deg, #78350F, #92400E); color: #FDE68A; border: 1px solid #F59E0B; }
    .cat-ops { background: linear-gradient(135deg, #4C1D95, #5B21B6); color: #C4B5FD; border: 1px solid #8B5CF6; }
    .big-metric {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.75;
        margin-bottom: 4px;
    }
    .model-winner {
        background: linear-gradient(135deg, #064E3B, #065F46);
        border: 1px solid #10B981;
        border-radius: 8px;
        padding: 12px 16px;
        color: #A7F3D0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load Verified Presets
# ---------------------------------------------------------------------------
if os.path.exists(PRESETS_FILE):
    with open(PRESETS_FILE, "r") as f:
        PRESETS = json.load(f)
else:
    PRESETS = {
        "allow": {"TransactionAmt": 161.0, "card1": 2377, "ProductCD": "W", "card4": "visa", "card6": "debit", "hour": 14},
        "challenge": {"TransactionAmt": 1265.5, "card1": 18227, "ProductCD": "W", "card4": "visa", "card6": "credit", "hour": 22},
        "block": {"TransactionAmt": 50.0, "card1": 15627, "ProductCD": "H", "card4": "mastercard", "card6": "debit", "hour": 3},
    }

# Session State Initialization
if "active_payload" not in st.session_state:
    st.session_state.active_payload = PRESETS["allow"].copy()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

def select_preset(key):
    st.session_state.active_payload = PRESETS[key].copy()
    st.session_state.last_result = None


# ---------------------------------------------------------------------------
# Sidebar: Backend Health & System Stats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Engine Telemetry")
    st.caption(f"Backend Target: `{API_BASE_URL}`")

    try:
        r_health = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if r_health.status_code == 200 and r_health.json().get("status") == "ok":
            st.success("● Backend: Online (Ready)")
            st.caption(f"Model: `{r_health.json().get('model_version')}`")
        else:
            st.error("● Backend: Degraded / Error")
    except Exception:
        st.error("● Backend: Offline / Unreachable")

    st.markdown("---")
    st.subheader("⚡ Rolling Latency SLA (Last 100 Requests)")
    try:
        r_stats = requests.get(f"{API_BASE_URL}/stats", timeout=2)
        if r_stats.status_code == 200:
            stats = r_stats.json()
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("Requests", stats["request_count"])
                st.metric("p95 Latency", f"{stats['p95_latency_ms']} ms")
            with col_s2:
                st.metric("p50 Latency", f"{stats['p50_latency_ms']} ms")
                st.metric("p99 Latency", f"{stats['p99_latency_ms']} ms")
    except Exception:
        st.caption("Unable to fetch latency telemetry.")

    st.markdown("---")
    st.caption("Architecture: XGBoost + Isotonic Calibration + 3-Tier Operational Routing.")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("💳 Real-Time Payment Fraud Risk Engine")
st.markdown("Sub-50ms transaction decisioning with **calibrated probabilities**, **operational routing**, and **local SHAP explainability**.")

tab_sim, tab_metrics, tab_defense, tab_rzp = st.tabs([
    "🧪 Interactive Transaction Simulator",
    "📊 Model Performance Metrics",
    "🛡️ Real-Time Defense & Incident Monitor",
    "💳 Live Razorpay Webhook Monitor"
])

with tab_sim:
    st.markdown("### ⚡ Quick Demo Presets")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        if st.button("🟢 Scenario 1: Low Risk (Grocery)", use_container_width=True):
            select_preset("allow")
            st.rerun()
        st.caption("Frictionless baseline transaction ($161.00) $\\rightarrow$ **`ALLOW`**")

    with col_p2:
        if st.button("🟡 Scenario 2: Elevated Risk Spike", use_container_width=True):
            select_preset("challenge")
            st.rerun()
        st.caption("High value transaction ($1,265.50) $\\rightarrow$ **`CHALLENGE (2FA / OTP)`**")

    with col_p3:
        if st.button("🔴 Scenario 3: High-Confidence Fraud", use_container_width=True):
            select_preset("block")
            st.rerun()
        st.caption("Carding / identity mismatch attack $\\rightarrow$ **`HARD_BLOCK`**")

    st.markdown("---")

    # Editable Form
    st.subheader("📝 Transaction Parameters")
    curr = st.session_state.active_payload

    with st.form("transaction_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tx_amt = st.number_input("Transaction Amount ($)", min_value=0.50, max_value=25000.0, value=float(curr.get("TransactionAmt", 100.0)), step=5.0)
            card1 = st.number_input("Card ID (card1)", min_value=1000, max_value=30000, value=int(curr.get("card1", 10000)))
            prod_options = ["W", "C", "R", "H", "S"]
            p_val = curr.get("ProductCD", "W")
            prod_idx = prod_options.index(p_val) if p_val in prod_options else 0
            prod_cd = st.selectbox("Product Code", prod_options, index=prod_idx)

        with col2:
            c4_options = ["visa", "mastercard", "discover", "amex"]
            c4_val = str(curr.get("card4", "visa")).lower()
            c4_idx = c4_options.index(c4_val) if c4_val in c4_options else 0
            card4 = st.selectbox("Card Brand", c4_options, index=c4_idx)

            c6_options = ["debit", "credit"]
            c6_val = str(curr.get("card6", "debit")).lower()
            c6_idx = c6_options.index(c6_val) if c6_val in c6_options else 0
            card6 = st.selectbox("Card Type", c6_options, index=c6_idx)

            hour = st.slider("Hour of Day (0-23)", 0, 23, value=int(curr.get("hour", 12)))

        with col3:
            tx_1h = st.number_input("1h Velocity (Tx Count)", min_value=1, max_value=50, value=int(curr.get("card_tx_count_1h", 1)))
            tx_24h = st.number_input("24h Velocity (Tx Count)", min_value=1, max_value=100, value=int(curr.get("card_tx_count_24h", 1)))
            time_since = st.number_input("Seconds Since Last Tx", min_value=0.0, max_value=864000.0, value=float(curr.get("card_time_since_last_tx", 86400.0)), step=60.0)

        with col4:
            p_email = st.text_input("Purchaser Email Domain", value=str(curr.get("P_emaildomain", "gmail.com")))
            addr1 = st.number_input("Billing Region Code (addr1)", min_value=100.0, max_value=600.0, value=float(curr.get("addr1", 299.0)))
            has_id = st.checkbox("Has Identity Record", value=bool(curr.get("has_identity", False)))

        submitted = st.form_submit_button("⚡ Evaluate Transaction Risk", type="primary", use_container_width=True)

    if submitted:
        # Merge edited fields into full active payload
        payload = curr.copy()
        payload.update({
            "TransactionAmt": float(tx_amt),
            "card1": int(card1),
            "ProductCD": str(prod_cd),
            "card4": str(card4),
            "card6": str(card6),
            "hour": int(hour),
            "card_tx_count_1h": int(tx_1h),
            "card_tx_count_24h": int(tx_24h),
            "card_time_since_last_tx": float(time_since),
            "P_emaildomain": str(p_email),
            "addr1": float(addr1),
            "has_identity": bool(has_id),
        })

        with st.spinner("Scoring transaction via FastAPI microservice..."):
            try:
                res = requests.post(f"{API_BASE_URL}/score", json=payload, timeout=5)
                if res.status_code == 200:
                    st.session_state.last_result = res.json()
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as exc:
                st.error(f"Failed to connect to API at `{API_BASE_URL}`: {exc}")

    if st.session_state.last_result:
        data = st.session_state.last_result
        prob = data["fraud_probability"]
        tier = data["risk_tier"]
        reasons = data["reasons"]
        lat = data["latency_ms"]

        st.markdown("---")
        st.subheader("🎯 Decision & Risk Assessment")

        col_res1, col_res2, col_res3 = st.columns([1.2, 1.8, 1])

        with col_res1:
            st.markdown("**Operational Action Tier:**")
            if tier == "ALLOW":
                st.markdown('<div class="badge-allow">● ALLOW (APPROVED)</div>', unsafe_allow_html=True)
                st.caption("Frictionless checkout path. Zero cardholder friction.")
            elif tier == "CHALLENGE":
                st.markdown('<div class="badge-challenge">▲ CHALLENGE (2FA / OTP)</div>', unsafe_allow_html=True)
                st.caption("Step-up authentication routed to customer to recover honest GMV.")
            else:
                st.markdown('<div class="badge-block">✕ HARD BLOCK (REJECTED)</div>', unsafe_allow_html=True)
                st.caption("Confirmed fraud signature. Immediate authorization reject.")

            st.markdown(f"**API Latency:** `{lat} ms` *(Sub-50ms SLA)*")

        with col_res2:
            st.markdown(f"**Calibrated Fraud Probability: `{prob * 100:.2f}%`**")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption("Risk Thresholds: ALLOW (< 3%) | CHALLENGE (3% – 30%) | HARD BLOCK (≥ 30%)")

        with col_res3:
            st.metric(label="Calculated Risk Score", value=f"{prob:.4f}", delta=f"{tier}")

        st.markdown("#### 🔍 Primary Risk Drivers (SHAP Local Explainability)")
        for r in reasons:
            st.markdown(f"- ⚠️ **{r}**")


# =====================================================================
# 📊 MODEL PERFORMANCE METRICS TAB
# =====================================================================
with tab_metrics:
    st.subheader("📊 Comprehensive Model Performance Metrics")
    st.markdown("Production metrics computed on the **sealed test set** (never seen during training). Updated after each training pipeline run.")

    # Load metrics from JSON
    metrics = None
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    # Also try fetching from API
    if metrics is None:
        try:
            r = requests.get(f"{API_BASE_URL}/metrics", timeout=3)
            if r.status_code == 200:
                metrics = r.json()
        except Exception:
            pass

    if metrics is None:
        st.warning("⚠️ No metrics found. Run the training pipeline (`python fraud_detection.py`) to generate metrics.")
    else:
        version = metrics.get("pipeline_version", "unknown")
        st.caption(f"Pipeline Version: **`{version}`**")

        st.markdown("---")

        # --- Optimized Routing Tiers Banner ---
        th = metrics.get("thresholds", {})
        if th:
            p_low = th.get("p_low_allow_challenge", 0.0804)
            p_high = th.get("p_high_challenge_block", 0.7495)
            val_m = metrics.get("val_metrics", {})
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E293B, #0F172A); border: 1px solid #3B82F6; border-radius: 12px; padding: 18px 24px; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #60A5FA;">🎯 Optimised Business Routing Tiers (Tuned on Validation Set)</span>
                    <span style="font-size: 0.85rem; color: #94A3B8;">Validation: Challenge &lt; 6% (<b>{val_m.get('challenge_rate_pct', 5.84):.2f}%</b>) | Auto-Block &gt; 90% (<b>{val_m.get('auto_block_precision_pct', 90.33):.2f}%</b>)</span>
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div style="background: #064E3B; padding: 8px 16px; border-radius: 8px; border: 1px solid #059669;">
                        <strong style="color: #34D399;">🟢 ALLOW</strong> &nbsp;<code style="color: #A7F3D0;">prob &lt; {p_low:.4f}</code>
                        <div style="font-size: 0.8rem; color: #6EE7B7; margin-top: 2px;">Frictionless Approval</div>
                    </div>
                    <div style="background: #78350F; padding: 8px 16px; border-radius: 8px; border: 1px solid #D97706;">
                        <strong style="color: #FBBF24;">🟡 CHALLENGE</strong> &nbsp;<code style="color: #FDE68A;">{p_low:.4f} &le; prob &lt; {p_high:.4f}</code>
                        <div style="font-size: 0.8rem; color: #FCD34D; margin-top: 2px;">Step-Up Friction (2FA / OTP)</div>
                    </div>
                    <div style="background: #7F1D1D; padding: 8px 16px; border-radius: 8px; border: 1px solid #DC2626;">
                        <strong style="color: #F87171;">🔴 HARD_BLOCK</strong> &nbsp;<code style="color: #FECACA;">prob &ge; {p_high:.4f}</code>
                        <div style="font-size: 0.8rem; color: #FCA5A5; margin-top: 2px;">Automated Decline</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Row 1: ML Detection Metrics ---
        ml = metrics.get("ml_detection", {})
        st.markdown('<div class="metrics-category-header cat-ml">🤖 ML Detection Performance (Sealed Test Set)</div>', unsafe_allow_html=True)
        ml_c1, ml_c2, ml_c3, ml_c4 = st.columns(4)
        with ml_c1:
            st.metric("Precision", f"{ml.get('precision_pct', 0):.2f}%")
        with ml_c2:
            st.metric("Recall", f"{ml.get('recall_pct', 0):.2f}%")
        with ml_c3:
            st.metric("PR-AUC", f"{ml.get('pr_auc', 0):.4f}")
        with ml_c4:
            st.metric("ROC-AUC", f"{ml.get('roc_auc', 0):.4f}")

        st.markdown("")

        # --- Row 2: Financial Cost Metrics ---
        fin = metrics.get("financial_cost", {})
        st.markdown('<div class="metrics-category-header cat-fin">💰 Financial Impact</div>', unsafe_allow_html=True)
        fin_c1, fin_c2, fin_c3, fin_c4 = st.columns(4)
        with fin_c1:
            savings_lakhs = fin.get("net_merchant_savings_inr_lakhs", 0)
            savings_usd = fin.get("net_merchant_savings_usd", 0)
            st.metric("Net Merchant Savings", f"₹{savings_lakhs:.2f} Lakhs", delta=f"${savings_usd:,.0f} USD")
        with fin_c2:
            st.metric("Value-Weighted Recall", f"{fin.get('value_weighted_recall_pct', 0):.2f}%")
        with fin_c3:
            st.metric("Fraud $ Caught", f"${fin.get('fraud_caught_usd', 0):,.0f}")
        with fin_c4:
            st.metric("Fraud $ Missed", f"${fin.get('fraud_missed_usd', 0):,.0f}", delta="Reduce", delta_color="inverse")

        st.markdown("")

        # --- Row 3: Merchant Funnel Metrics ---
        funnel = metrics.get("merchant_funnel", {})
        st.markdown('<div class="metrics-category-header cat-funnel">🏪 Merchant Funnel / Routing</div>', unsafe_allow_html=True)
        fun_c1, fun_c2, fun_c3, fun_c4 = st.columns(4)
        with fun_c1:
            st.metric("Auto-Block Precision", f"{funnel.get('auto_block_precision_pct', 0):.2f}%")
        with fun_c2:
            st.metric("Challenge Rate", f"{funnel.get('challenge_rate_pct', 0):.2f}%")
        with fun_c3:
            st.metric("Allow Rate (Frictionless)", f"{funnel.get('allow_rate_pct', 0):.2f}%")
        with fun_c4:
            st.metric("Block Rate", f"{funnel.get('block_rate_pct', 0):.2f}%")

        st.markdown("")

        # --- Row 4: Operations Metrics ---
        ops = metrics.get("operations", {})
        st.markdown('<div class="metrics-category-header cat-ops">⚡ Operations & Latency</div>', unsafe_allow_html=True)
        ops_c1, ops_c2, ops_c3 = st.columns(3)
        with ops_c1:
            st.metric("Inference Latency (p50)", f"{ops.get('inference_latency_p50_ms', 0):.2f} ms")
        with ops_c2:
            st.metric("Inference Latency (p95)", f"{ops.get('inference_latency_p95_ms', 0):.2f} ms")
        with ops_c3:
            st.metric("Inference Latency (p99)", f"{ops.get('inference_latency_p99_ms', 0):.2f} ms")

        st.markdown("---")

        # --- Formatted Table (User's Exact Request) ---
        st.markdown("### 📋 Metrics Summary Table")
        table_data = [
            {"Metric Category": "ML Detection", "Metric": "Precision", "Actual Value": f"{ml.get('precision_pct', 0):.2f}%"},
            {"Metric Category": "ML Detection", "Metric": "Recall", "Actual Value": f"{ml.get('recall_pct', 0):.2f}%"},
            {"Metric Category": "ML Detection", "Metric": "PR-AUC", "Actual Value": f"{ml.get('pr_auc', 0):.4f}"},
            {"Metric Category": "ML Detection", "Metric": "ROC-AUC", "Actual Value": f"{ml.get('roc_auc', 0):.4f}"},
            {"Metric Category": "Financial Cost", "Metric": "Net Merchant Savings", "Actual Value": f"₹{fin.get('net_merchant_savings_inr_lakhs', 0):.2f} Lakhs"},
            {"Metric Category": "Financial Cost", "Metric": "Value-Weighted Recall", "Actual Value": f"{fin.get('value_weighted_recall_pct', 0):.2f}%"},
            {"Metric Category": "Merchant Funnel", "Metric": "Auto-Block Precision", "Actual Value": f"{funnel.get('auto_block_precision_pct', 0):.2f}%"},
            {"Metric Category": "Merchant Funnel", "Metric": "Challenge Rate", "Actual Value": f"{funnel.get('challenge_rate_pct', 0):.2f}%"},
            {"Metric Category": "Operations", "Metric": "Inference Latency", "Actual Value": f"{ops.get('inference_latency_p50_ms', 0):.2f} ms"},
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Model Comparison Table ---
        st.markdown("### 🏆 Model Comparison (Sealed Test Set)")
        model_comp = metrics.get("model_comparison", {})
        if model_comp:
            comp_rows = []
            model_display = {
                "logistic_regression": "📉 Logistic Regression",
                "mlp_neural_net": "🧠 Neural Net (MLP)",
                "xgboost_calibrated": "🌲 XGBoost (Calibrated)",
                "lightgbm_calibrated": "⚡ LightGBM (Calibrated)",
                "ensemble_blend": "🏆 Ensemble Blend",
            }
            best_pr_auc = 0
            best_model = ""
            for key, display_name in model_display.items():
                if key in model_comp:
                    m = model_comp[key]
                    pr = m.get("pr_auc", 0)
                    comp_rows.append({
                        "Model": display_name,
                        "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                        "PR-AUC": f"{pr:.4f}",
                    })
                    if pr > best_pr_auc:
                        best_pr_auc = pr
                        best_model = display_name

            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            if best_model:
                st.markdown(f'<div class="model-winner">🥇 Best Production Model: <strong>{best_model}</strong> (PR-AUC: {best_pr_auc:.4f})</div>', unsafe_allow_html=True)

            # Show ensemble weights if available
            ensemble = model_comp.get("ensemble_blend", {})
            weights = ensemble.get("weights", {})
            if weights:
                st.markdown("#### 🔀 Ensemble Blend Weights")
                w_cols = st.columns(len(weights))
                for idx, (model_name, weight) in enumerate(weights.items()):
                    with w_cols[idx]:
                        st.metric(model_name, f"{float(weight)*100:.0f}%")

        # --- Confusion Matrix ---
        cm = metrics.get("confusion_matrix", {})
        if cm:
            st.markdown("---")
            st.markdown(f"### 🔢 Confusion Matrix (threshold = {cm.get('threshold', 0.30)})")
            cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
            with cm_c1:
                st.metric("True Positives (TP)", f"{cm.get('TP', 0):,}")
            with cm_c2:
                st.metric("False Positives (FP)", f"{cm.get('FP', 0):,}")
            with cm_c3:
                st.metric("False Negatives (FN)", f"{cm.get('FN', 0):,}")
            with cm_c4:
                st.metric("True Negatives (TN)", f"{cm.get('TN', 0):,}")



# =====================================================================
# 🛡️ REAL-TIME DEFENSE & INCIDENT MONITOR TAB
# =====================================================================
with tab_defense:
    st.subheader("🛡️ Real-Time Fraud Spike Monitoring & Automated Defense Circuit Breaker")
    st.markdown("Continuous gateway sliding-window monitoring, autonomous defense routing, and in-app incident management.")

    # Fetch defense telemetry from API
    defense_data = None
    try:
        r_def = requests.get(f"{API_BASE_URL}/defense/status", timeout=3)
        if r_def.status_code == 200:
            defense_data = r_def.json()
    except Exception:
        pass

    if not defense_data:
        st.warning("⚠️ Could not reach Defense API at `/defense/status`. Ensure the FastAPI backend is running.")
    else:
        cb = defense_data.get("circuit_breaker", {})
        telem = defense_data.get("sliding_window_telemetry", {})
        active_inc = defense_data.get("active_incident")
        supp_count = defense_data.get("suppressed_entities_count", 0)
        supp_list = defense_data.get("suppressed_entities", [])
        cb_state = cb.get("state", "NORMAL")

        # 1. State Banner
        state_color = {
            "NORMAL": {
                "bg": "linear-gradient(135deg, #064E3B, #022C22)",
                "border": "#059669",
                "badge": "🟢 NORMAL: Baseline Routing Active",
                "desc": "Gateway traffic is healthy. Standard routing thresholds are actively protecting checkout.",
            },
            "DEFENSE_ACTIVE": {
                "bg": "linear-gradient(135deg, #7F1D1D, #450A0A)",
                "border": "#DC2626",
                "badge": "🔴 DEFENSE ACTIVE: Circuit Breaker Engaged",
                "desc": "Fraud spike detected! Autonomous defense has tightened routing thresholds to intercept elevated risk.",
            },
            "COOLDOWN": {
                "bg": "linear-gradient(135deg, #78350F, #451A03)",
                "border": "#D97706",
                "badge": "🟡 COOLDOWN: Traffic Stabilizing",
                "desc": "Evaluating consecutive clean transactions. System will automatically recover to NORMAL once cooldown threshold is reached.",
            },
        }.get(cb_state, {"bg": "#1E293B", "border": "#334155", "badge": cb_state, "desc": ""})

        act_th = cb.get("active_thresholds", {})
        st.markdown(f"""
        <div style="background: {state_color['bg']}; border: 1px solid {state_color['border']}; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">{state_color['badge']}</span>
                <span style="font-size: 0.9rem; color: #E2E8F0; background: rgba(0,0,0,0.3); padding: 4px 12px; border-radius: 6px;">Spike Severity: <strong>{telem.get('spike_severity', 'NORMAL')}</strong> | Drift: <strong>{telem.get('score_drift_status', 'STABLE')}</strong></span>
            </div>
            <div style="font-size: 0.95rem; color: #CBD5E1; margin-bottom: 12px;">{state_color['desc']}</div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                <div style="background: rgba(0,0,0,0.25); padding: 6px 14px; border-radius: 6px; font-size: 0.85rem;">
                    Active Tiers: <code>ALLOW &lt; {act_th.get('p_low', 0.0804):.4f}</code> &bull; <code>CHALLENGE &lt; {act_th.get('p_high', 0.7495):.4f}</code> &bull; <code>BLOCK &ge; {act_th.get('p_high', 0.7495):.4f}</code>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 6px 14px; border-radius: 6px; font-size: 0.85rem;">
                    Active Suppression Blacklist: <strong>{supp_count}</strong> entities
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Sliding Window KPIs
        st.markdown("### ⏱️ Gateway Sliding-Window Telemetry (Rolling 5 Minutes)")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.metric("5m Transaction Volume", telem.get("tx_count", 0), delta=f"₹{telem.get('total_volume_amt', 0):,.2f}")
        with col_t2:
            st.metric("High-Risk Rate", f"{telem.get('high_risk_rate_pct', 0.0):.1f}%", delta=f"{telem.get('high_risk_count', 0)} blocked", delta_color="inverse")
        with col_t3:
            st.metric("Challenge Rate", f"{telem.get('challenge_rate_pct', 0.0):.1f}%", delta=f"{telem.get('challenge_count', 0)} 3DS/OTP")
        with col_t4:
            st.metric("Rolling Mean Risk Score", f"{telem.get('mean_risk_prob', 0.0):.4f}", delta=f"{telem.get('burst_velocity_60s', 0)} in last 60s", delta_color="inverse")

        st.markdown("---")

        # 3. Active Incident or All Clear
        st.markdown("### 🚨 In-App Incident Management")
        if active_inc:
            st.error(f"⚠️ **ACTIVE FRAUD SPIKE INCIDENT DETECTED** [{active_inc.get('incident_id')}]")
            inc_c1, inc_c2, inc_c3 = st.columns(3)
            with inc_c1:
                st.markdown(f"**Severity:** `{active_inc.get('severity')}`")
                st.markdown(f"**Status:** `{active_inc.get('status')}`")
            with inc_c2:
                st.markdown(f"**Triggered At:** `{active_inc.get('started_at')[:19]}`")
                st.markdown(f"**Affected Transactions:** `{active_inc.get('affected_transactions_count', 0)}`")
            with inc_c3:
                res_btn = st.button("✅ Manually Resolve Incident", use_container_width=True)
                if res_btn:
                    requests.post(f"{API_BASE_URL}/defense/incidents/{active_inc.get('incident_id')}/resolve")
                    st.success("Incident resolved. Reloading...")
                    st.rerun()

            st.json(active_inc.get("trigger_metrics", {}))
        else:
            st.success("● Gateway Incident Status: **ALL CLEAR** (No active fraud surges)")

        # Historical Incidents Log
        with st.expander("📋 View Incident Audit Trail", expanded=False):
            try:
                r_inc = requests.get(f"{API_BASE_URL}/defense/incidents?limit=10", timeout=2)
                if r_inc.status_code == 200:
                    inc_list = r_inc.json().get("incidents", [])
                    if inc_list:
                        inc_df = pd.DataFrame([{
                            "ID": i.get("incident_id"),
                            "Severity": i.get("severity"),
                            "Status": i.get("status"),
                            "Started At": i.get("started_at")[:19] if i.get("started_at") else "-",
                            "Duration": f"{i.get('duration_seconds', 0)}s" if i.get("duration_seconds") else "Ongoing",
                            "Affected Txns": i.get("affected_transactions_count", 0),
                            "Resolution": i.get("resolution_reason", "Active"),
                        } for i in inc_list])
                        st.dataframe(inc_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No historical incidents recorded.")
            except Exception:
                st.caption("Unable to fetch incident history.")

        st.markdown("---")

        # 4. Temporary Entity Suppression List
        st.markdown("### 🚫 Temporary Entity Suppression List (Auto-Blacklist)")
        st.caption("Repeated attackers (3+ HARD_BLOCK events in 10 minutes) are automatically suppressed with a 30-minute TTL. Suppression is reversible and can be unblocked manually below:")

        if not supp_list:
            st.info("No entities currently suppressed. Gateway blacklist is clean.")
        else:
            for s in supp_list:
                eid = s.get("entity_id")
                rem_m = round(s.get("remaining_ttl_seconds", 0) / 60.0, 1)
                col_s1, col_s2, col_s3 = st.columns([3, 2, 1])
                with col_s1:
                    st.markdown(f"**Entity:** `{eid}`")
                    st.caption(f"Reason: {s.get('reason')}")
                with col_s2:
                    st.markdown(f"**Remaining TTL:** `{rem_m} mins` ({int(s.get('remaining_ttl_seconds', 0))}s)")
                    st.caption(f"Violations: {s.get('violation_count')} in window")
                with col_s3:
                    if st.button("🔓 Unblock", key=f"unblock_{eid}", use_container_width=True):
                        requests.post(f"{API_BASE_URL}/defense/suppression-list/remove", json={"entity_id": eid})
                        st.success(f"Unblocked {eid}")
                        st.rerun()

        st.markdown("---")

        # 5. Operator Emergency Controls
        st.markdown("### 🛠️ Operator Manual Defense Controls")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            st.caption("Emergency manual circuit breaker engagement:")
            if st.button("🚨 Emergency Trip Circuit Breaker", use_container_width=True):
                requests.post(f"{API_BASE_URL}/defense/circuit-breaker/trip", json={"reason": "Manual operator emergency engagement from dashboard", "severity": "HIGH"})
                st.warning("Circuit breaker manually tripped to DEFENSE_ACTIVE.")
                st.rerun()
        with col_ctrl2:
            st.caption("Reset circuit breaker back to standard parameters:")
            if st.button("🔄 Reset Circuit Breaker to NORMAL", use_container_width=True):
                requests.post(f"{API_BASE_URL}/defense/circuit-breaker/reset")
                st.success("Circuit breaker reset to NORMAL.")
                st.rerun()


with tab_rzp:
    st.subheader("💳 Live Razorpay Webhook Payment Monitor")
    st.markdown("Real-time stream of captured test payments from Razorpay (`payment.captured`), scored dynamically by the calibrated fraud model with live velocity and audit trail logging.")

    col_btn, col_info = st.columns([1.2, 3.8])
    with col_btn:
        if st.button("🔄 Refresh Audit Feed", use_container_width=True):
            st.rerun()

    # Fetch logs from API or read from disk
    audit_data = []
    try:
        res = requests.get(f"{API_BASE_URL}/razorpay/audit-logs?limit=50", timeout=3)
        if res.status_code == 200:
            audit_data = res.json().get("logs", [])
    except Exception:
        # Fallback to direct file read if API is momentarily unreachable
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "razorpay_audit_log.jsonl")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = [json.loads(line.strip()) for line in f if line.strip()]
                    audit_data = lines[-50:][::-1]
            except Exception:
                audit_data = []

    if not audit_data:
        st.info("No Razorpay test payments logged yet. Trigger a payment using the demo script (`python scripts/demo_razorpay_flow.py`) or configure your Razorpay Webhook to point to `POST /webhook/razorpay`.")
    else:
        # KPI Cards
        n_total = len(audit_data)
        n_allow = sum(1 for e in audit_data if e.get("risk_tier") == "ALLOW")
        n_challenge = sum(1 for e in audit_data if e.get("risk_tier") == "CHALLENGE")
        n_block = sum(1 for e in audit_data if e.get("risk_tier") == "HARD_BLOCK")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Webhook Events", n_total)
        with kpi2:
            st.metric("Frictionless (ALLOW)", f"{n_allow} ({n_allow/n_total*100:.1f}%)")
        with kpi3:
            st.metric("Step-Up (CHALLENGE)", f"{n_challenge} ({n_challenge/n_total*100:.1f}%)")
        with kpi4:
            st.metric("Decline (HARD_BLOCK)", f"{n_block} ({n_block/n_total*100:.1f}%)")

        st.markdown("---")
        st.markdown("### 📋 Captured Payments Audit Stream")

        for idx, entry in enumerate(audit_data):
            p_id = entry.get("payment_id", "Unknown")
            amt = entry.get("amount", 0.0)
            curr_code = entry.get("currency", "INR")
            prob = entry.get("fraud_probability", 0.0)
            tier = entry.get("risk_tier", "ALLOW")
            ts = entry.get("timestamp", "")
            cust = entry.get("customer_identifier", "anonymous")
            reasons = entry.get("reasons", [])
            real_feat = entry.get("features_real", {})
            def_feat = entry.get("features_defaulted", [])

            tier_badge = {
                "ALLOW": "🟢 ALLOW",
                "CHALLENGE": "🟡 CHALLENGE",
                "HARD_BLOCK": "🔴 HARD_BLOCK"
            }.get(tier, tier)

            with st.expander(f"{tier_badge} | {p_id} | ₹{amt:,.2f} {curr_code} | Prob: {prob*100:.2f}% | Customer: {cust} ({ts[:19]})", expanded=(idx == 0)):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.markdown("#### ⚡ Real Features Derived Live")
                    st.json(real_feat)
                    st.markdown(f"**Calculated Probability:** `{prob:.4f}` ({prob*100:.2f}%)")
                    st.markdown(f"**Operational Decision:** `{tier}`")
                    st.markdown(f"**Inference Latency:** `{entry.get('latency_ms', 0)} ms`")

                with col_e2:
                    st.markdown("#### 🛡️ SHAP Alert Reasons")
                    if reasons:
                        for r in reasons:
                            st.markdown(f"- ⚠️ **{r}**")
                    else:
                        st.caption("No positive risk alerts triggered.")

                    st.markdown("#### ⚙️ Features Defaulted / Imputed")
                    st.caption("To prevent synthetic fabrication, unseen IEEE-CIS features use existing pipeline defaults:")
                    for df_item in def_feat:
                        st.markdown(f"- `{df_item}`")

