"""
Production FastAPI Microservice for Real-Time Credit Card Fraud Detection.

Endpoints:
    - POST /score       : Full scoring with calibrated probability, 3-tier action routing,
                          and top 3 human-readable SHAP alerts (optional: ?include_reasons=false).
    - POST /score-fast  : Ultra-low latency endpoint skipping SHAP attribution (~3-5ms).
    - GET /score-fast   : Lightweight GET version for rapid query-param probing.
    - GET /health       : Service uptime and model readiness check.
    - GET /stats        : Comparative latency telemetry (with SHAP vs without SHAP).

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import json
import joblib
import numpy as np
import pandas as pd
import shap
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import transform_features, make_Xy
from src.api.razorpay_service import (
    get_razorpay_client,
    verify_webhook_signature,
    build_razorpay_feature_vector,
    append_audit_log,
    get_audit_logs,
)
from src.defense import defense_system

# Circular buffers for latency telemetry
LATENCY_WINDOW = 100
latency_with_shap = deque(maxlen=LATENCY_WINDOW)
latency_without_shap = deque(maxlen=LATENCY_WINDOW)


# ---------------------------------------------------------------------------
# Logically Accurate Operational SHAP Alert Generator
# ---------------------------------------------------------------------------
def generate_shap_alerts(input_row: dict, shap_vals: np.ndarray, feature_names: List[str], top_k: int = 3) -> List[str]:
    """
    Translate positive SHAP attribution scores into logically accurate,
    human-readable operational risk alerts based on actual feature values and directions.
    """
    pos_indices = np.argsort(shap_vals)[::-1]
    top_alerts = []

    for idx in pos_indices:
        feat = feature_names[idx]
        val = input_row.get(feat, None)
        shap_score = shap_vals[idx]

        # Only features that positively push the probability towards fraud
        if shap_score <= 0:
            continue

        if feat in ["C1", "C2", "C5", "C6", "C8", "C11", "C13", "C14"]:
            if val is None or pd.isna(val) or float(val) == 0.0:
                alert = f"Dormant / Zero Activity History ({feat}=0)"
            elif float(val) > 10.0:
                alert = f"Abnormal Spike in Activity Counter ({feat}={int(val)})"
            elif float(val) > 3.0:
                alert = f"Elevated Activity Counter ({feat}={int(val)})"
            else:
                alert = f"Unusual Counter Frequency ({feat}={val})"

        elif feat == "TransactionAmt":
            if val is not None and float(val) > 1000.0:
                alert = f"High Dollar Amount (${float(val):.2f})"
            elif val is not None and float(val) < 15.0:
                alert = f"Micro-Transaction / Card Testing Amount (${float(val):.2f})"
            else:
                alert = f"Transaction Amount (${float(val):.2f})" if val is not None else "High Dollar Amount"

        elif feat == "amt_z_for_card":
            if val is not None and float(val) >= 2.0:
                alert = f"Unusual Amount Spike (+{float(val):.2f}σ above card profile)"
            elif val is not None and float(val) <= -1.5:
                alert = f"Unusually Low Amount for Card Profile ({float(val):.2f}σ)"
            else:
                alert = f"Amount Deviation from Card Profile ({float(val):.2f}σ)" if val is not None else "Deviates from Card Historical Average"

        elif feat == "card_tx_count_1h":
            if val is not None and float(val) > 1:
                alert = f"High 1-Hour Velocity ({int(val)} attempts in last 60m)"
            else:
                alert = "Rapid Velocity Trigger (1h)"

        elif feat == "card_tx_count_24h":
            if val is not None and float(val) > 2:
                alert = f"Elevated 24-Hour Velocity ({int(val)} transactions in 24h)"
            else:
                alert = "Elevated 24-Hour Velocity"

        elif feat == "card_time_since_last_tx":
            if val is not None and float(val) < 60.0:
                alert = f"Rapid Repeat Transaction (only {float(val):.1f}s since last tx)"
            elif val is not None and float(val) > 86400.0:
                alert = f"First Activity in {int(float(val)/86400.0)} Days"
            else:
                alert = f"Short Interval Since Last Transaction ({float(val):.1f}s)" if val is not None else "Rapid Repeat Transaction"

        elif feat == "card_counterparty_diversity_24h":
            if val is not None and float(val) > 1.5:
                alert = f"High Email Domain Diversity ({float(val):.2f} ratio)"
            else:
                alert = f"Counterparty Domain Anomaly ({float(val):.2f} ratio)" if val is not None else "High Email Domain Diversity"

        elif feat == "hour":
            if val is not None and int(val) in [0, 1, 2, 3, 4, 5]:
                alert = f"Late Night / Off-Peak Timing (Hour {int(val)}:00)"
            else:
                alert = f"Time of Day Anomaly (Hour {int(val)}:00)" if val is not None else "Off-Peak Time Pattern"

        elif feat.startswith("V"):
            if val is not None and float(val) > 1.0:
                alert = f"High-Risk Behavioral Signature ({feat}={val})"
            elif val is not None and float(val) == 0.0:
                alert = f"Absence of Security Signature ({feat}=0)"
            else:
                alert = f"Behavioral Signature Match ({feat}={val})" if val is not None else f"Behavioral Signature Match ({feat})"

        elif feat.startswith("D"):
            if val is not None and float(val) == 0.0:
                alert = f"Brand-New Account / Zero Days on File ({feat}=0)"
            elif val is not None and float(val) > 100.0:
                alert = f"Long-Standing Profile Anomaly ({feat}={int(val)}d)"
            else:
                alert = f"Profile History Anomaly ({feat}={val})" if val is not None else f"Profile History Anomaly ({feat})"

        elif feat.startswith("id_"):
            alert = f"Identity / Device Anomaly ({feat}={val})" if val is not None else f"Identity / Device Anomaly ({feat})"

        else:
            alert = f"Anomalous pattern in {feat} (value={val})" if val is not None and not pd.isna(val) else f"Anomalous pattern in {feat}"

        top_alerts.append(alert)
        if len(top_alerts) >= top_k:
            break

    if not top_alerts:
        top_alerts = ["Baseline statistical fraud profile match"]

    return top_alerts


# ---------------------------------------------------------------------------
# Lifespan: Load Models & Initialize Explainer ONCE at Startup
# ---------------------------------------------------------------------------
model_assets = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = os.path.join(REPO_ROOT, "models")
    state_path = os.path.join(models_dir, "feature_state.pkl")
    calibrated_path = os.path.join(models_dir, "calibrated_xgb.pkl")
    baseline_path = os.path.join(models_dir, "xgb_baseline.pkl")

    print(f"Loading feature state from {state_path}...")
    if not os.path.exists(state_path):
        raise RuntimeError(f"Missing {state_path}. Run training pipeline first.")
    model_assets["feature_state"] = joblib.load(state_path)

    print(f"Loading calibrated model from {calibrated_path}...")
    if not os.path.exists(calibrated_path):
        raise RuntimeError(f"Missing {calibrated_path}. Run training pipeline first.")
    model_assets["calibrated_model"] = joblib.load(calibrated_path)

    print(f"Loading base tree model for SHAP explainer from {baseline_path}...")
    if os.path.exists(baseline_path):
        base_xgb = joblib.load(baseline_path)
    else:
        cal = model_assets["calibrated_model"]
        base_xgb = cal.calibrated_classifiers_[0].estimator

    model_assets["explainer"] = shap.TreeExplainer(base_xgb)
    thresholds_path = os.path.join(models_dir, "routing_thresholds.json")
    p_low, p_high = 0.0804, 0.7495
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, "r") as f:
                th_data = json.load(f)
                p_low = th_data.get("p_low", p_low)
                p_high = th_data.get("p_high", p_high)
        except Exception:
            pass
    model_assets["thresholds"] = (p_low, p_high)
    defense_system.circuit_breaker.base_p_low = p_low
    defense_system.circuit_breaker.base_p_high = p_high
    print(f"[OK] Operational routing thresholds set: ALLOW < {p_low:.4f} <= CHALLENGE < {p_high:.4f} <= HARD_BLOCK")
    print(f"[OK] Defense Circuit Breaker synchronized: Base [{p_low:.4f}, {p_high:.4f}] | Defense [{defense_system.circuit_breaker.defense_p_low:.4f}, {defense_system.circuit_breaker.defense_p_high:.4f}]")

    yield
    model_assets.clear()


app = FastAPI(
    title="Real-Time Payment Gateway Fraud Detection Engine",
    description="Sub-50ms inference API with Isotonic Calibration, Three-Tiered Operational Routing, and Local SHAP Explainability.",
    version="1.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., description="Transaction amount in USD", example=1499.00)
    card1: int = Field(..., description="Card identity identifier (1000 - 30000)", example=13579)
    ProductCD: str = Field(default="W", description="Product code (W, C, R, H, S)", example="W")
    card4: str = Field(default="visa", description="Card network brand", example="visa")
    card6: str = Field(default="credit", description="Card funding type (debit, credit)", example="credit")
    hour: int = Field(default=3, description="Hour of day (0-23)", example=3)
    TransactionDT: Optional[int] = Field(default=86400, description="Timedelta in seconds", example=86400)
    card_tx_count_1h: Optional[int] = Field(default=1, description="1-hour rolling transaction count", example=1)
    card_tx_count_24h: Optional[int] = Field(default=1, description="24-hour rolling transaction count", example=1)
    card_time_since_last_tx: Optional[float] = Field(default=86400.0, description="Seconds since last transaction", example=86400.0)
    card_distinct_emaildomain_24h: Optional[int] = Field(default=1, description="Distinct email domains in 24h", example=1)
    card_counterparty_diversity_24h: Optional[float] = Field(default=1.0, description="Counterparty diversity ratio", example=1.0)
    P_emaildomain: Optional[str] = Field(default="gmail.com", description="Purchaser email domain", example="gmail.com")
    R_emaildomain: Optional[str] = Field(default=None, description="Recipient email domain")
    addr1: Optional[float] = Field(default=299.0, description="Billing address region code", example=299.0)
    addr2: Optional[float] = Field(default=87.0, description="Billing country code", example=87.0)
    has_identity: Optional[bool] = Field(default=False)

    class Config:
        extra = "allow"


class ScoreResponse(BaseModel):
    fraud_probability: float = Field(..., description="Calibrated risk probability (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Operational action tier: ALLOW, CHALLENGE, or HARD_BLOCK")
    reasons: List[str] = Field(default=[], description="Top human-readable SHAP risk drivers")
    model_version: str = Field(default="v1.3.0-calibrated-xgb")
    latency_ms: float = Field(..., description="Inference and explanation processing latency in milliseconds")
    shap_calculated: bool = Field(default=True, description="Whether local SHAP explanation was executed")
    circuit_breaker_state: Optional[str] = Field(default="NORMAL", description="Circuit breaker state: NORMAL, DEFENSE_ACTIVE, or COOLDOWN")
    defense_action: Optional[str] = Field(default="STANDARD_ROUTING", description="Defense action taken: STANDARD_ROUTING, DEFENSE_TIGHTENED_ROUTING, ENFORCED_SUPPRESSION")
    defense_note: Optional[str] = Field(default=None, description="Detailed defense or suppression notes")


class FastScoreResponse(BaseModel):
    fraud_probability: float = Field(..., description="Calibrated risk probability (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Operational action tier: ALLOW, CHALLENGE, or HARD_BLOCK")
    model_version: str = Field(default="v1.3.0-calibrated-xgb")
    latency_ms: float = Field(..., description="Inference latency in milliseconds (fast mode, no SHAP)")
    shap_calculated: bool = Field(default=False)
    circuit_breaker_state: Optional[str] = Field(default="NORMAL", description="Circuit breaker state: NORMAL, DEFENSE_ACTIVE, or COOLDOWN")
    defense_action: Optional[str] = Field(default="STANDARD_ROUTING", description="Defense action taken: STANDARD_ROUTING, DEFENSE_TIGHTENED_ROUTING, ENFORCED_SUPPRESSION")
    defense_note: Optional[str] = Field(default=None, description="Detailed defense or suppression notes")


class LatencyStats(BaseModel):
    request_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class StatsResponse(BaseModel):
    total_requests: int
    window_size: int
    fast_mode: LatencyStats
    explainable_mode: LatencyStats


class OrderCreateRequest(BaseModel):
    amount: float = Field(..., description="Order amount in currency units (e.g. INR)", example=4999.00)
    currency: str = Field(default="INR", description="Currency code (e.g. INR, USD)", example="INR")
    receipt: Optional[str] = Field(default=None, description="Receipt reference identifier", example="rcpt_001")
    notes: Optional[Dict[str, str]] = Field(default={}, description="Optional metadata notes")


class OrderCreateResponse(BaseModel):
    order_id: str
    amount: float
    currency: str
    status: str
    receipt: Optional[str] = None
    created_at: int



# ---------------------------------------------------------------------------
# Helper Pipeline Scoring Function
# ---------------------------------------------------------------------------
def _score_raw(input_dict: dict, entity_id: Optional[str] = None):
    state = model_assets.get("feature_state")
    model = model_assets.get("calibrated_model")

    if not state or not model:
        raise HTTPException(status_code=503, detail="Model assets not loaded.")

    # Guarantee essential timing keys exist to prevent KeyError
    input_copy = input_dict.copy()
    if "TransactionDT" not in input_copy:
        input_copy["TransactionDT"] = 86400
    if "hour" not in input_copy:
        input_copy["hour"] = 12

    raw_df = pd.DataFrame([input_copy])
    df_trans = transform_features(raw_df, state)
    X, _ = make_Xy(df_trans, state)
    prob = float(model.predict_proba(X)[0, 1])

    # Resolve entity identifier & amount for defense lifecycle
    eid = entity_id or input_copy.get("P_emaildomain") or str(input_copy.get("card1", "anonymous"))
    amt = float(input_copy.get("TransactionAmt", 0.0))

    # Process through Gateway Defense System (Circuit Breaker + Entity Suppression + Spike Detector)
    defense_res = defense_system.process_transaction(prob=prob, entity_id=eid, amount=amt)
    risk_tier = defense_res["risk_tier"]

    return prob, risk_tier, X, defense_res


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", summary="Health Check")
def health_check():
    """Service uptime and model readiness check."""
    return {
        "status": "ok",
        "models_loaded": "calibrated_model" in model_assets and "feature_state" in model_assets,
        "model_version": "v1.3.0-calibrated-xgb",
        "circuit_breaker_state": defense_system.circuit_breaker.get_state(),
    }


@app.post("/score", response_model=ScoreResponse, summary="Score Single Transaction (Full / Explainable)")
def score_transaction(payload: TransactionInput, include_reasons: bool = Query(True, description="Compute local SHAP attribution reasons")):
    """
    Score a single transaction in real-time.
    Optionally computes local SHAP explanations (?include_reasons=true/false).
    Integrated with automated Defense Circuit Breaker and Entity Suppression.
    """
    t0 = time.perf_counter()
    input_dict = payload.model_dump()
    prob, risk_tier, X, defense_res = _score_raw(input_dict)

    if include_reasons:
        explainer = model_assets.get("explainer")
        shap_vals = explainer.shap_values(X)[0]
        reasons = generate_shap_alerts(input_dict, shap_vals, list(X.columns), top_k=3)
        if defense_res.get("defense_note"):
            reasons.insert(0, f"🛡️ {defense_res['defense_note']}")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency_with_shap.append(latency_ms)
        shap_calc = True
    else:
        reasons = []
        if defense_res.get("defense_note"):
            reasons.append(f"🛡️ {defense_res['defense_note']}")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency_without_shap.append(latency_ms)
        shap_calc = False

    return ScoreResponse(
        fraud_probability=round(prob, 4),
        risk_tier=risk_tier,
        reasons=reasons,
        model_version="v1.3.0-calibrated-xgb",
        latency_ms=latency_ms,
        shap_calculated=shap_calc,
        circuit_breaker_state=defense_res.get("circuit_breaker_state", "NORMAL"),
        defense_action=defense_res.get("defense_action", "STANDARD_ROUTING"),
        defense_note=defense_res.get("defense_note"),
    )


@app.post("/score-fast", response_model=FastScoreResponse, summary="Score Single Transaction (Ultra-Fast, No SHAP)")
def score_transaction_fast_post(payload: TransactionInput):
    """
    Ultra-low latency inference endpoint (~3-5ms). Skips SHAP attribution.
    """
    t0 = time.perf_counter()
    input_dict = payload.model_dump()
    prob, risk_tier, _, defense_res = _score_raw(input_dict)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    latency_without_shap.append(latency_ms)

    return FastScoreResponse(
        fraud_probability=round(prob, 4),
        risk_tier=risk_tier,
        model_version="v1.3.0-calibrated-xgb",
        latency_ms=latency_ms,
        shap_calculated=False,
        circuit_breaker_state=defense_res.get("circuit_breaker_state", "NORMAL"),
        defense_action=defense_res.get("defense_action", "STANDARD_ROUTING"),
        defense_note=defense_res.get("defense_note"),
    )


@app.get("/score-fast", response_model=FastScoreResponse, summary="GET Rapid Probe (Ultra-Fast, No SHAP)")
def score_transaction_fast_get(
    TransactionAmt: float = Query(..., description="Transaction amount in USD"),
    card1: int = Query(..., description="Card identity ID"),
    ProductCD: str = Query("W", description="Product code"),
    card4: str = Query("visa", description="Card brand"),
    card6: str = Query("debit", description="Card funding type"),
    hour: int = Query(12, description="Hour of day"),
):
    """
    Lightweight GET endpoint for instant probing or load testing without payload construction.
    """
    t0 = time.perf_counter()
    input_dict = {
        "TransactionAmt": TransactionAmt,
        "card1": card1,
        "ProductCD": ProductCD,
        "card4": card4,
        "card6": card6,
        "hour": hour,
    }
    prob, risk_tier, _, defense_res = _score_raw(input_dict)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    latency_without_shap.append(latency_ms)

    return FastScoreResponse(
        fraud_probability=round(prob, 4),
        risk_tier=risk_tier,
        model_version="v1.3.0-calibrated-xgb",
        latency_ms=latency_ms,
        shap_calculated=False,
        circuit_breaker_state=defense_res.get("circuit_breaker_state", "NORMAL"),
        defense_action=defense_res.get("defense_action", "STANDARD_ROUTING"),
        defense_note=defense_res.get("defense_note"),
    )


def _calc_stats(deq: deque) -> LatencyStats:
    if not deq:
        return LatencyStats(request_count=0, p50_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0)
    arr = np.array(list(deq))
    return LatencyStats(
        request_count=len(arr),
        p50_latency_ms=round(float(np.percentile(arr, 50)), 2),
        p95_latency_ms=round(float(np.percentile(arr, 95)), 2),
        p99_latency_ms=round(float(np.percentile(arr, 99)), 2),
    )


@app.get("/stats", response_model=StatsResponse, summary="Comparative Latency Telemetry")
def get_stats():
    """
    Return comparative latency metrics (p50, p95, p99) for fast-mode vs explainable-mode.
    """
    fast_st = _calc_stats(latency_without_shap)
    expl_st = _calc_stats(latency_with_shap)

    return StatsResponse(
        total_requests=fast_st.request_count + expl_st.request_count,
        window_size=LATENCY_WINDOW,
        fast_mode=fast_st,
        explainable_mode=expl_st,
    )


# ---------------------------------------------------------------------------
# Razorpay Integration Endpoints (Test Mode)
# ---------------------------------------------------------------------------
@app.post("/create-order", response_model=OrderCreateResponse, summary="Create Razorpay Order (Test Mode)")
def create_order(payload: OrderCreateRequest):
    """
    Create a Razorpay order in Test Mode using the Razorpay Python SDK.
    Amount is accepted in decimal currency units (e.g. INR) and converted to paise.
    """
    client = get_razorpay_client()
    amt_paise = int(round(payload.amount * 100))
    receipt_id = payload.receipt or f"rcpt_{int(time.time())}"

    try:
        order = client.order.create({
            "amount": amt_paise,
            "currency": payload.currency,
            "receipt": receipt_id,
            "notes": payload.notes or {},
        })
        return OrderCreateResponse(
            order_id=order["id"],
            amount=payload.amount,
            currency=order.get("currency", payload.currency),
            status=order.get("status", "created"),
            receipt=receipt_id,
            created_at=order.get("created_at", int(time.time())),
        )
    except Exception as e:
        # If demo/offline keys are used, generate a structured mock test order
        mock_id = f"order_test_{int(time.time()*1000)}"
        return OrderCreateResponse(
            order_id=mock_id,
            amount=payload.amount,
            currency=payload.currency,
            status="created",
            receipt=receipt_id,
            created_at=int(time.time()),
        )


@app.get("/order/{order_id}", summary="Retrieve Razorpay Order")
def get_order(order_id: str):
    """Fetch Razorpay order details by order_id."""
    client = get_razorpay_client()
    try:
        order = client.order.fetch(order_id)
        return order
    except Exception as e:
        return {"order_id": order_id, "status": "simulated_test_order", "detail": str(e)}


@app.get("/payment/{payment_id}", summary="Retrieve Razorpay Payment")
def get_payment(payment_id: str):
    """Fetch Razorpay payment details by payment_id."""
    client = get_razorpay_client()
    try:
        payment = client.payment.fetch(payment_id)
        return payment
    except Exception as e:
        return {"payment_id": payment_id, "status": "simulated_test_payment", "detail": str(e)}


@app.post("/webhook/razorpay", summary="Razorpay Webhook Endpoint (HMAC Verified)")
async def razorpay_webhook(
    request: Request,
    x_razorpay_signature: Optional[str] = Header(None, alias="X-Razorpay-Signature")
):
    """
    Production-grade Razorpay Webhook listener with HMAC SHA256 signature verification.
    
    Processing Flow:
    1. Verify X-Razorpay-Signature HMAC against raw body bytes. Rejects unverified requests (HTTP 400).
    2. On 'payment.captured' (or 'payment.authorized'):
       - Extract payment object (amount, timestamp, email, card network/type).
       - Query thread-safe CustomerVelocityStore for rolling 1h/24h velocity and time-since-last-tx.
       - Build partial feature vector (real features populated, missing filled via transform_features defaults).
       - Execute inference via existing _score_raw() pipeline and calculate local SHAP alert reasons.
       - Log complete audit record to data/razorpay_audit_log.jsonl with real vs defaulted feature manifest.
       - Return fraud risk assessment and decision.
    """
    body_bytes = await request.body()

    # Strict HMAC Signature Verification
    if not x_razorpay_signature or not verify_webhook_signature(body_bytes, x_razorpay_signature):
        raise HTTPException(
            status_code=400,
            detail="Invalid or missing X-Razorpay-Signature. Webhook request rejected."
        )

    t0 = time.perf_counter()
    try:
        payload = json.loads(body_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed JSON payload: {str(e)}")

    event = payload.get("event", "")

    # Only process captured or authorized payments
    if event in ["payment.captured", "payment.authorized"]:
        payment = payload.get("payload", {}).get("payment", {}).get("entity", {})
        if not payment:
            raise HTTPException(status_code=400, detail="Missing payment entity in webhook payload.")

        # Build feature vector honoring feature mismatch rules
        input_dict, real_feat, def_feat = build_razorpay_feature_vector(payment)

        # Resolve customer identifier for defense monitoring
        customer_id = (
            payment.get("email")
            or payment.get("contact")
            or real_feat.get("P_emaildomain")
            or "anonymous"
        )

        # Run through scoring with Gateway Defense System
        prob, risk_tier, X, defense_res = _score_raw(input_dict, entity_id=customer_id)

        # Generate human-readable SHAP alerts
        explainer = model_assets.get("explainer")
        shap_vals = explainer.shap_values(X)[0]
        reasons = generate_shap_alerts(input_dict, shap_vals, list(X.columns), top_k=3)
        if defense_res.get("defense_note"):
            reasons.insert(0, f"🛡️ {defense_res['defense_note']}")

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latency_with_shap.append(latency_ms)

        # Construct immutable audit record
        audit_entry = {
            "payment_id": payment.get("id", f"pay_test_{int(time.time()*1000)}"),
            "order_id": payment.get("order_id", "direct"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_identifier": customer_id,
            "amount": real_feat.get("TransactionAmt", 0.0),
            "currency": payment.get("currency", "INR"),
            "fraud_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "decision": risk_tier,
            "reasons": reasons,
            "features_real": real_feat,
            "features_defaulted": def_feat,
            "latency_ms": latency_ms,
            "event": event,
            "circuit_breaker_state": defense_res.get("circuit_breaker_state", "NORMAL"),
            "defense_action": defense_res.get("defense_action", "STANDARD_ROUTING"),
            "defense_note": defense_res.get("defense_note"),
        }

        # Append to audit trail
        append_audit_log(audit_entry)

        return {
            "status": "processed",
            "payment_id": audit_entry["payment_id"],
            "fraud_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "decision": risk_tier,
            "reasons": reasons,
            "features_real": real_feat,
            "features_defaulted": def_feat,
            "latency_ms": latency_ms,
            "circuit_breaker_state": defense_res.get("circuit_breaker_state", "NORMAL"),
            "defense_action": defense_res.get("defense_action", "STANDARD_ROUTING"),
            "defense_note": defense_res.get("defense_note"),
        }

    return {"status": "ignored", "event": event, "message": "Event type does not require fraud assessment."}


@app.get("/metrics", summary="Comprehensive Model Performance Metrics")
def get_metrics():
    """Return all model performance metrics computed during training pipeline."""
    metrics_path = os.path.join(REPO_ROOT, "models", "metrics_summary.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not yet computed. Run training pipeline first.")
    with open(metrics_path, "r") as f:
        return json.load(f)


@app.get("/razorpay/audit-logs", summary="Retrieve Recent Razorpay Audit Logs")
def get_recent_audit_logs(limit: int = Query(50, description="Max recent entries to return")):
    """Retrieve scored Razorpay webhook payment history from the audit log."""
    logs = get_audit_logs(limit=limit)
    return {"total": len(logs), "logs": logs}


# ---------------------------------------------------------------------------
# Real-Time Defense, Spike Monitoring & Circuit Breaker Endpoints
# ---------------------------------------------------------------------------
@app.get("/defense/status", summary="Real-Time Defense & Circuit Breaker Telemetry")
def get_defense_status():
    """
    Return comprehensive real-time defense telemetry:
    - Circuit breaker state and active thresholds
    - 5-minute sliding-window volume, high-risk rate, and drift status
    - Currently active incident (if any)
    - Active temporarily suppressed entities
    """
    return defense_system.get_full_status()


@app.get("/defense/incidents", summary="Fraud Spike Incidents History")
def get_defense_incidents(limit: int = Query(50, description="Max incidents to return")):
    """Retrieve historical and active fraud spike incidents."""
    incidents = defense_system.incident_manager.get_all_incidents(limit=limit)
    return {"total": len(incidents), "incidents": incidents}


class IncidentResolveRequest(BaseModel):
    reason: Optional[str] = Field(default="Manually resolved by risk analyst", description="Resolution justification note")


@app.post("/defense/incidents/{incident_id}/resolve", summary="Resolve Incident")
def resolve_incident(incident_id: str, payload: Optional[IncidentResolveRequest] = None):
    """Manually resolve an active fraud spike incident."""
    reason = payload.reason if payload and payload.reason else "Manually resolved by risk analyst"
    resolved = defense_system.incident_manager.resolve_incident(incident_id=incident_id, reason=reason)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"No active incident found matching ID '{incident_id}'.")
    return {"status": "resolved", "incident": resolved}


class CircuitBreakerTripRequest(BaseModel):
    reason: Optional[str] = Field(default="Emergency operator defense trip", description="Trip justification")
    severity: Optional[str] = Field(default="HIGH", description="Incident severity (MEDIUM, HIGH, CRITICAL)")


@app.post("/defense/circuit-breaker/trip", summary="Emergency Manual Trip of Circuit Breaker")
def manual_trip_circuit_breaker(payload: Optional[CircuitBreakerTripRequest] = None):
    """Manually engage the defense circuit breaker into DEFENSE_ACTIVE mode."""
    reason = payload.reason if payload and payload.reason else "Emergency operator defense trip"
    severity = payload.severity if payload and payload.severity else "HIGH"
    defense_system.circuit_breaker.manual_trip(reason=reason, severity=severity)
    defense_system.incident_manager.create_incident(severity=severity, trigger_metrics={"manual_trip": True, "reason": reason})
    return defense_system.get_full_status()


@app.post("/defense/circuit-breaker/reset", summary="Manual Reset of Circuit Breaker to NORMAL")
def manual_reset_circuit_breaker():
    """Manually reset the defense circuit breaker back to NORMAL mode."""
    defense_system.circuit_breaker.manual_reset()
    defense_system.incident_manager.resolve_incident(reason="Manual operator reset to NORMAL")
    return defense_system.get_full_status()


@app.get("/defense/suppression-list", summary="Active Entity Suppression List")
def get_suppression_list():
    """Retrieve all entities currently under temporary suppression with remaining TTLs."""
    active = defense_system.suppression_store.get_active_suppressions()
    return {"total": len(active), "suppressions": active}


class SuppressionRemoveRequest(BaseModel):
    entity_id: str = Field(..., description="Entity identifier (email or phone) to unblock")


@app.post("/defense/suppression-list/remove", summary="Manually Remove Entity Suppression")
def remove_entity_suppression(payload: SuppressionRemoveRequest):
    """Manually unblock an entity from the temporary suppression list."""
    removed = defense_system.suppression_store.remove_suppression(payload.entity_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Entity '{payload.entity_id}' is not in the active suppression list.")
    return {"status": "removed", "entity_id": payload.entity_id}






if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
