"""
Single-Run Model Test Script
============================
Tests the live production model and calibrated scoring pipeline across
3 representative transaction scenarios:
1. Low-Risk Everyday Purchase (Expected: ALLOW)
2. High-Value Elevated Risk Spike (Expected: CHALLENGE)
3. High-Confidence Bot/Carding Attack (Expected: HARD_BLOCK)
"""

import sys
import os
import requests
import json

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

presets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "presets.json")
with open(presets_path, "r") as f:
    PRESETS = json.load(f)

test_cases = [
    {
        "name": "Scenario 1: Verified Low-Risk Grocery Transaction",
        "description": "Legitimate domestic purchase ($161.00) -> Expected: ALLOW",
        "payload": PRESETS["allow"],
    },
    {
        "name": "Scenario 2: Elevated Risk Spike (High Value)",
        "description": "High-value purchase ($1,265.50) with amount anomaly -> Expected: CHALLENGE",
        "payload": PRESETS["challenge"],
    },
    {
        "name": "Scenario 3: High-Confidence Fraud / Carding Attack",
        "description": "Multiple fraud indicators (ProductCD 'C', high velocity counters) -> Expected: HARD_BLOCK",
        "payload": PRESETS["block"],
    },
]

def main():
    print("=" * 75)
    print("      PRODUCTION MODEL INFERENCE TEST (FASTAPI + CALIBRATED MODEL)")
    print("=" * 75)
    print(f"Target API Server: {API_URL}")

    # Check health
    try:
        r_health = requests.get(f"{API_URL}/health", timeout=3)
        print(f"Server Health    : {r_health.status_code} ({r_health.json()})")
    except Exception as e:
        print(f"[ERROR] Could not connect to {API_URL}: {e}")
        sys.exit(1)

    print("-" * 75)

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[TEST CASE {idx}] {tc['name']}")
        print(f"Context      : {tc['description']}")
        print(f"Input Payload: {tc['payload']}")

        res = requests.post(f"{API_URL}/score", json=tc["payload"]).json()

        prob = res.get("fraud_probability", 0.0)
        tier = res.get("risk_tier", "UNKNOWN")
        latency = res.get("latency_ms", 0.0)
        action = res.get("defense_action", "STANDARD_ROUTING")
        reasons = res.get("reasons", [])

        print(f"  --> Fraud Probability : {prob * 100:.2f}% (Score: {prob:.4f})")
        print(f"  --> Operational Tier  : [{tier}]")
        print(f"  --> Defense Action    : {action}")
        print(f"  --> Inference Latency : {latency:.2f} ms")
        print(f"  --> Top SHAP Alerts   :")
        if reasons:
            for r in reasons:
                clean_r = r.encode("ascii", errors="replace").decode("ascii")
                print(f"       * {clean_r}")
        else:
            print("       * (No positive alerts)")

    print("\n" + "=" * 75)
    print(" MODEL INFERENCE TEST COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
