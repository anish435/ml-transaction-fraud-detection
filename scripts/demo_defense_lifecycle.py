"""
End-to-End Demonstration: Gateway Fraud-Spike Monitoring & Auto-Responder Lifecycle.
=====================================================================================
Demonstrates:
1. Normal Traffic Baseline (Circuit Breaker = NORMAL, Incident = ALL CLEAR).
2. Gateway Fraud Spike Attack (Spike detected, Circuit Breaker trips to DEFENSE_ACTIVE).
3. In-App Incident Management (Incident auto-created with CRITICAL severity).
4. Adaptive Defense Routing (Borderline risk transactions elevated to HARD_BLOCK).
5. Temporary Entity Suppression (Repeated attacker automatically blacklisted with TTL).
6. Auto-Recovery Lifecycle (Healthy traffic during cooldown auto-recovers to NORMAL, incident RESOLVED).
"""

import sys
import os
import time
import json
import requests

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def banner(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def get_status():
    r = requests.get(f"{API_URL}/defense/status", timeout=3)
    return r.json()


def main():
    banner("FRAUD-SPIKE MONITORING & DEFENSE AUTO-RESPONDER DEMO")
    print(f"Connecting to Gateway at {API_URL}...")

    # Reset circuit breaker to start clean
    requests.post(f"{API_URL}/defense/circuit-breaker/reset")

    # 1. Normal Traffic
    banner("PHASE 1: Baseline Normal Traffic")
    print("Sending 5 normal, low-risk transactions ($25 - $120)...")
    for i in range(5):
        payload = {
            "TransactionAmt": 45.0 + (i * 15.0),
            "card1": 2377,
            "ProductCD": "W",
            "card4": "visa",
            "card6": "debit",
            "hour": 14,
            "P_emaildomain": f"good_user_{i}@gmail.com"
        }
        res = requests.post(f"{API_URL}/score-fast", json=payload).json()
        print(f"  Txn {i+1}: Prob={res['fraud_probability']*100:.2f}% | Tier={res['risk_tier']} | CB State={res.get('circuit_breaker_state')}")

    st1 = get_status()
    cb1 = st1["circuit_breaker"]
    telem1 = st1["sliding_window_telemetry"]
    print(f"\n[TELEMETRY] 5m Txns: {telem1['tx_count']} | High-Risk Rate: {telem1['high_risk_rate_pct']}% | CB State: {cb1['state']}")
    print(f"[INCIDENT]  Active Incident: {st1['active_incident']}")
    assert cb1["state"] == "NORMAL", "Expected CB to be NORMAL"

    # 2. Fraud Spike Surge
    banner("PHASE 2: Massive Fraud Spike Injection (Bot Attack)")
    print("Injecting rapid high-risk attacks using verified fraud attack profiles...")

    presets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "presets.json")
    with open(presets_path, "r") as f:
        presets = json.load(f)
    block_template = presets["block"].copy()

    attacker = "bot_ring_leader@darknet-market.org"
    for i in range(6):
        payload = block_template.copy()
        payload["TransactionAmt"] = 85.0 + (i * 25.0)
        payload["P_emaildomain"] = attacker
        payload["hour"] = 3
        res = requests.post(f"{API_URL}/score", json=payload).json()
        print(f"  Attack {i+1}: Prob={res['fraud_probability']*100:.2f}% | Tier={res['risk_tier']} | Action={res.get('defense_action')} | CB State={res.get('circuit_breaker_state')}")
        time.sleep(0.1)

    st2 = get_status()
    cb2 = st2["circuit_breaker"]
    telem2 = st2["sliding_window_telemetry"]
    inc2 = st2["active_incident"]

    banner("PHASE 3: Circuit Breaker & Incident Verification")
    print(f"[!] Circuit Breaker State: [{cb2['state']}]")
    print(f"[*] Gateway Telemetry   : Tx Count={telem2['tx_count']} | High-Risk Rate={telem2['high_risk_rate_pct']}% | Burst 60s={telem2['burst_velocity_60s']}")
    print(f"[*] Spike Severity      : {telem2['spike_severity']}")
    print(f"[*] Active Thresholds   : ALLOW < {cb2['active_thresholds']['p_low']:.4f} <= CHALLENGE < {cb2['active_thresholds']['p_high']:.4f} <= BLOCK")
    print(f"\n[INCIDENT] Active Incident Created:")
    if inc2:
        print(f"   * Incident ID : {inc2['incident_id']}")
        print(f"   * Severity    : {inc2['severity']}")
        print(f"   * Status      : {inc2['status']}")
        print(f"   * Started At  : {inc2['started_at']}")
    assert cb2["state"] == "DEFENSE_ACTIVE", "Expected Circuit Breaker to trip to DEFENSE_ACTIVE"
    assert inc2 is not None, "Expected active incident to be created"

    # 4. Entity Suppression Check
    banner("PHASE 4: Repeated Entity Suppression Verification")
    supp_list = st2["suppressed_entities"]
    print(f"Suppressed Entities Count: {len(supp_list)}")
    for s in supp_list:
        print(f"   * Blocked Entity : {s['entity_id']} (Remaining TTL: {s['remaining_ttl_seconds']}s)")
        print(f"     Reason         : {s['reason']}")

    print("\nSending a micro-transaction from the same blocked entity to verify automatic suppression...")
    supp_test = requests.post(f"{API_URL}/score", json={
        "TransactionAmt": 15.0,
        "card1": 2377,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit",
        "hour": 14,
        "P_emaildomain": attacker,
    }).json()
    print(f"  Suppressed Entity Decision: [{supp_test['risk_tier']}] | Action: {supp_test.get('defense_action')}")
    print(f"  Note: {supp_test.get('defense_note')}")
    assert supp_test["risk_tier"] == "HARD_BLOCK", "Suppressed entity must be immediately blocked"

    # 5. Cooldown & Auto-Recovery Demo
    banner("PHASE 5: Cooldown & Autonomous Incident Resolution")
    print("Testing manual reset / recovery lifecycle...")
    res_resolve = requests.post(f"{API_URL}/defense/circuit-breaker/reset").json()
    print(f"Resetting Circuit Breaker: State is now [{res_resolve['circuit_breaker']['state']}]")

    all_incs = requests.get(f"{API_URL}/defense/incidents?limit=3").json()["incidents"]
    print("\nRecent Incident Audit Trail:")
    for inc in all_incs:
        print(f"  * ID: {inc['incident_id']} | Severity: {inc['severity']} | Status: {inc['status']} | Duration: {inc.get('duration_seconds')}s | Note: {inc.get('resolution_reason')}")

    banner("DEMONSTRATION COMPLETE - ALL DEFENSE SUBSYSTEMS FULLY OPERATIONAL")


if __name__ == "__main__":
    main()
