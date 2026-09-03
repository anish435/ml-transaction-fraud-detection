"""
Streamlit Risk Analyst Dashboard for Real-Time Transaction Scoring.

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import requests
import streamlit as st

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")

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
