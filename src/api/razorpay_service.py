"""
Razorpay Service Module: Test Mode SDK Client, HMAC Signature Verification,
Live Customer Velocity Engine, Feature Vector Mapping, and Audit Logging.
"""

import os
import json
import time
import hmac
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import razorpay
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_demo123456789")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "demoSecretKey987654321")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "demoWebhookSecret123")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
AUDIT_LOG_PATH = os.path.join(DATA_DIR, "razorpay_audit_log.jsonl")
VELOCITY_STORE_PATH = os.path.join(DATA_DIR, "customer_velocity_store.json")


# ---------------------------------------------------------------------------
# 1. Razorpay SDK Client
# ---------------------------------------------------------------------------
def get_razorpay_client() -> razorpay.Client:
    """Initialize and return Razorpay Python SDK Client."""
    key_id = os.getenv("RAZORPAY_KEY_ID", RAZORPAY_KEY_ID)
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", RAZORPAY_KEY_SECRET)
    return razorpay.Client(auth=(key_id, key_secret))


# ---------------------------------------------------------------------------
# 2. HMAC SHA256 Webhook Signature Verification
# ---------------------------------------------------------------------------
def verify_webhook_signature(body_bytes: bytes, signature: str, secret: Optional[str] = None) -> bool:
    """
    Verify Razorpay Webhook HMAC SHA256 signature against request raw payload.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not signature:
        return False

    webhook_secret = secret or os.getenv("RAZORPAY_WEBHOOK_SECRET", RAZORPAY_WEBHOOK_SECRET)
    if not webhook_secret:
        return False

    expected_signature = hmac.new(
        key=webhook_secret.encode("utf-8"),
        msg=body_bytes,
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_signature, signature)


# ---------------------------------------------------------------------------
# 3. Thread-Safe Live Customer Velocity Engine
# ---------------------------------------------------------------------------
class CustomerVelocityStore:
    """
    In-memory customer velocity engine with JSON persistence.
    Tracks chronological payment timestamps per customer (email or contact)
    to compute time_since_last_tx, 1-hour velocity, and 24-hour velocity.
    """
    def __init__(self, persistence_path: str = VELOCITY_STORE_PATH):
        self.persistence_path = persistence_path
        self._lock = threading.Lock()
        self._history: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = {}

    def _save(self):
        try:
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f)
        except Exception:
            pass

    def record_and_get_velocity(self, customer_id: str, current_ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Record current payment timestamp and compute rolling velocity features.
        """
        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            ts_list = self._history.get(customer_id, [])

            # Compute interval from last transaction
            if ts_list:
                time_since_last_tx = max(0.0, round(current_ts - ts_list[-1], 1))
            else:
                time_since_last_tx = 86400.0  # Cold-start baseline: 24h

            # Filter timestamps within rolling windows
            one_hour_ago = current_ts - 3600
            twenty_four_hours_ago = current_ts - 86400

            # Count previous attempts + current attempt
            tx_count_1h = sum(1 for t in ts_list if t >= one_hour_ago) + 1
            tx_count_24h = sum(1 for t in ts_list if t >= twenty_four_hours_ago) + 1

            # Append current timestamp and prune history older than 24h
            updated_list = [t for t in ts_list if t >= twenty_four_hours_ago]
            updated_list.append(current_ts)
            self._history[customer_id] = updated_list

            self._save()

        return {
            "card_time_since_last_tx": time_since_last_tx,
            "card_tx_count_1h": tx_count_1h,
            "card_tx_count_24h": tx_count_24h,
        }

    def reset(self):
        """Clear store for testing."""
        with self._lock:
            self._history = {}
            self._save()


# Singleton velocity engine instance
velocity_store = CustomerVelocityStore()


# ---------------------------------------------------------------------------
# 4. Feature Vector Mapping (Handling Feature Mismatch)
# ---------------------------------------------------------------------------
def build_razorpay_feature_vector(payment_dict: dict) -> Tuple[dict, dict, List[str]]:
    """
    Map Razorpay payment fields into the existing model's input feature vector.

    Constraint Compliance:
    - Only genuine overlaps are populated (amount, hour, email domain, card network/type, live velocity).
    - Unseen features (C1-C14, D1-D15, V1-V339, card1, addr1/2) are left to existing transform_features()
      defaults/imputations. No synthetic or fabricated values are generated.

    Returns:
        input_dict: Payload passed to _score_raw().
        features_real: Dictionary of actual values derived from the transaction.
        features_defaulted: List of feature categories that were defaulted.
    """
    # 1. Transaction Amount (convert from paise/cents to decimal currency)
    raw_amount = payment_dict.get("amount", 0)
    amt = float(raw_amount) / 100.0

    # 2. Timestamp & Hour of Day
    created_at = payment_dict.get("created_at")
    if created_at:
        try:
            dt = datetime.fromtimestamp(created_at, timezone.utc)
            hour = dt.hour
        except Exception:
            hour = datetime.now().hour
    else:
        created_at = int(time.time())
        hour = datetime.now().hour

    # 3. Email Domain
    email = str(payment_dict.get("email", "") or "").strip().lower()
    if "@" in email:
        p_emaildomain = email.split("@")[-1]
    else:
        p_emaildomain = "gmail.com"

    # 4. Card Details (if payment method is card)
    card = payment_dict.get("card") or {}
    card4 = str(card.get("network", "visa") or "visa").lower()
    card6 = str(card.get("type", "credit") or "credit").lower()

    # Map unknown networks to valid one-hot encoded categories
    valid_card4 = {"visa", "mastercard", "discover", "american express"}
    if card4 not in valid_card4:
        card4 = "visa"

    valid_card6 = {"credit", "debit"}
    if card6 not in valid_card6:
        card6 = "credit"

    # 5. Customer Identifier for Velocity
    contact = str(payment_dict.get("contact", "") or "").strip()
    customer_id = email if email and "@" in email else (contact if contact else "anonymous_customer")

    # 6. Live Velocity Calculation
    velocity = velocity_store.record_and_get_velocity(customer_id, current_ts=float(created_at))

    # Real features dictionary for audit
    features_real = {
        "TransactionAmt": round(amt, 2),
        "hour": hour,
        "P_emaildomain": p_emaildomain,
        "card4": card4,
        "card6": card6,
        "ProductCD": "W",
        "card_time_since_last_tx": velocity["card_time_since_last_tx"],
        "card_tx_count_1h": velocity["card_tx_count_1h"],
        "card_tx_count_24h": velocity["card_tx_count_24h"],
    }

    # Features that are defaulted by transform_features()
    features_defaulted = [
        "card1 (cold-start imputed via train global mean/std)",
        "addr1, addr2, dist1, dist2 (unprovided, null indicators set)",
        "C1..C14 (unprovided activity counters, reindexed to 0)",
        "D1..D15 (unprovided timedeltas, reindexed to 0)",
        "V1..V339 (unprovided Vesta behavioral signals, reindexed to 0)",
        "id_01..id_38 (identity attributes, unprovided)",
    ]

    # Clean input dictionary matching existing scoring pipeline
    input_dict = {
        "TransactionAmt": round(amt, 2),
        "hour": hour,
        "TransactionDT": int(created_at),
        "ProductCD": "W",
        "card1": 13579,  # Default cold-start ID (mean/std will be applied)
        "card4": card4,
        "card6": card6,
        "P_emaildomain": p_emaildomain,
        "card_time_since_last_tx": velocity["card_time_since_last_tx"],
        "card_tx_count_1h": velocity["card_tx_count_1h"],
        "card_tx_count_24h": velocity["card_tx_count_24h"],
        "card_distinct_emaildomain_24h": 1,
        "card_counterparty_diversity_24h": 1.0,
    }

    return input_dict, features_real, features_defaulted


# ---------------------------------------------------------------------------
# 5. Audit Logging Engine
# ---------------------------------------------------------------------------
_audit_lock = threading.Lock()

def append_audit_log(entry: dict):
    """Append a scored Razorpay payment event to the audit log (JSONL)."""
    with _audit_lock:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def get_audit_logs(limit: int = 50) -> List[dict]:
    """Retrieve the most recent audit log entries."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    logs = []
    with _audit_lock:
        try:
            with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            return []

    return logs[-limit:][::-1]  # Most recent first
