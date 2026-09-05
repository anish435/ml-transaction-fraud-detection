"""
End-to-End Demo Script: Razorpay Test Mode Order Creation & Webhook Fraud Scoring.

Demonstrates:
1. Creating a live test order via POST /create-order.
2. Simulating legitimate consumer checkout (capturing payment, signing webhook HMAC, verifying ALLOW).
3. Simulating a rapid-fire card testing velocity attack (multiple payments in seconds, observing risk tier escalation).
4. Demonstrating security rejection of tampered/unverified webhook signatures.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "demoWebhookSecret123")


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def send_signed_webhook(event_payload: dict, secret: str = WEBHOOK_SECRET) -> requests.Response:
    """Helper to HMAC-sign and POST a webhook payload."""
    body_bytes = json.dumps(event_payload).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
    }
    return requests.post(f"{BASE_URL}/webhook/razorpay", data=body_bytes, headers=headers)


def main():
    print_banner("RAZORPAY TEST MODE & FRAUD ENGINE INTEGRATION DEMO")
    print(f"Target API Endpoint: {BASE_URL}")

    # 1. Health Check
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=3)
        if res.status_code != 200:
            print(f"[ERROR] API is unhealthy (status {res.status_code}).")
            sys.exit(1)
        print(f"[OK] Backend Online: {res.json()}")
    except Exception as e:
        print(f"[ERROR] Could not reach API at {BASE_URL}: {e}")
        print("Make sure uvicorn is running: uvicorn src.api.main:app --port 8000")
        sys.exit(1)

    # 2. Step 1: Create a Razorpay Order
    print_banner("STEP 1: Create Test Order via POST /create-order")
    order_req = {
        "amount": 1499.00,
        "currency": "INR",
        "receipt": f"rcpt_demo_{int(time.time())}",
        "notes": {"merchant_category": "electronics", "channel": "mobile_app"}
    }
    res_order = requests.post(f"{BASE_URL}/create-order", json=order_req)
    print(f"Status Code: {res_order.status_code}")
    order_data = res_order.json()
    order_id = order_data.get("order_id", "order_mock_001")
    print(f"Created Order: ID={order_id}, Amount=INR {order_data.get('amount')}, Status={order_data.get('status')}")

    # 3. Step 2: Normal Legitimate Payment (ALLOW)
    print_banner("STEP 2: Simulate Normal Legitimate Customer Payment")
    legit_payment_id = f"pay_legit_{int(time.time())}"
    now_ts = int(time.time())
    legit_webhook = {
        "entity": "event",
        "account_id": "acc_demo_test",
        "event": "payment.captured",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": legit_payment_id,
                    "amount": 149900,  # ₹1,499.00
                    "currency": "INR",
                    "status": "captured",
                    "order_id": order_id,
                    "method": "card",
                    "card": {
                        "network": "Visa",
                        "type": "debit",
                        "issuer": "HDFC"
                    },
                    "email": "priya.sharma@gmail.com",
                    "contact": "+919876543210",
                    "created_at": now_ts
                }
            }
        }
    }
    print(f"Sending HMAC-signed webhook for {legit_payment_id} (Customer: priya.sharma@gmail.com)...")
    res_legit = send_signed_webhook(legit_webhook)
    print(f"Webhook Status: {res_legit.status_code}")
    legit_res = res_legit.json()
    print(f"-> Decision: [{legit_res.get('decision')}] | Fraud Probability: {legit_res.get('fraud_probability')*100:.2f}% | Latency: {legit_res.get('latency_ms')}ms")
    print(f"-> SHAP Alerts: {legit_res.get('reasons')}")
    print(f"-> Real Features Extracted: {legit_res.get('features_real')}")

    # 4. Step 3: Rapid Velocity Card Testing Attack
    print_banner("STEP 3: Simulate Rapid-Fire Velocity Card Testing Attack")
    bot_email = "carder.bot77@throwawaymail.org"
    print(f"Simulating bot attack firing 4 rapid payment attempts for {bot_email}...")

    amounts = [4900, 4900, 4900, 1499900]  # Three micro-charges (₹49) followed by ₹14,999 spike!
    for idx, amt in enumerate(amounts, 1):
        ts = now_ts + (idx * 4)  # 4 seconds apart
        attack_payment_id = f"pay_attack_{idx}_{int(time.time())}"
        attack_webhook = {
            "entity": "event",
            "account_id": "acc_demo_test",
            "event": "payment.captured",
            "contains": ["payment"],
            "payload": {
                "payment": {
                    "entity": {
                        "id": attack_payment_id,
                        "amount": amt,
                        "currency": "INR",
                        "status": "captured",
                        "order_id": f"order_bot_{idx}",
                        "method": "card",
                        "card": {
                            "network": "MasterCard",
                            "type": "credit",
                            "issuer": "UNKNOWN"
                        },
                        "email": bot_email,
                        "contact": "+919999988888",
                        "created_at": ts
                    }
                }
            }
        }
        res_att = send_signed_webhook(attack_webhook)
        att_res = res_att.json()
        real_v = att_res.get("features_real", {})
        print(f"  Attempt {idx}: Amount=INR {real_v.get('TransactionAmt'):,.2f} | Interval={real_v.get('card_time_since_last_tx')}s | 1h Count={real_v.get('card_tx_count_1h')} -> Decision: [{att_res.get('decision')}] (Prob: {att_res.get('fraud_probability')*100:.2f}%)")
        time.sleep(0.3)

    # 5. Step 4: Security Verification (Tampered Signature Rejection)
    print_banner("STEP 4: Security Verification — Tampered Signature Rejection")
    raw_body = json.dumps({"event": "payment.captured", "tampered": True}).encode("utf-8")
    headers_bad = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": "bad_tampered_signature_hex_12345"
    }
    res_bad = requests.post(f"{BASE_URL}/webhook/razorpay", data=raw_body, headers=headers_bad)
    print(f"Tampered Webhook HTTP Status: {res_bad.status_code}")
    print(f"Server Rejection Response: {res_bad.json()}")
    assert res_bad.status_code == 400, "Security failure: Tampered signature was not rejected!"
    print("[PASS] Tampered webhook successfully blocked by HMAC verification.")

    # 6. Audit Feed Verification
    print_banner("STEP 5: Live Audit Log Verification")
    res_logs = requests.get(f"{BASE_URL}/razorpay/audit-logs?limit=5")
    logs_json = res_logs.json()
    print(f"Total Audit Entries in Store: {logs_json.get('total')}")
    print("Latest 3 Audit Entries:")
    for entry in logs_json.get("logs", [])[:3]:
        print(f"  * {entry.get('timestamp')[:19]} | ID: {entry.get('payment_id')} | INR {entry.get('amount')} | Decision: {entry.get('decision')} | Prob: {entry.get('fraud_probability')*100:.2f}%")

    print_banner("DEMO COMPLETED SUCCESSFULLY")
    print("Now open the Streamlit Dashboard at http://localhost:8501 and click the")
    print("'Live Razorpay Webhook Monitor' tab to inspect the visual audit trail live!")


if __name__ == "__main__":
    main()
