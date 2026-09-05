"""
Gateway-Wide Sliding-Window Fraud Spike & Risk Monitor
======================================================
Maintains real-time sliding-window telemetry across all incoming transactions
at the gateway level to detect systemic fraud surges, botnet attacks, and
distribution drift in production.

Metrics Tracked:
- Sliding window transaction count and total currency volume
- High-risk (HARD_BLOCK) count and rate percentage
- Challenge (Step-Up 2FA) count and rate percentage
- Rolling mean risk probability
- Streaming score drift status (vs. normal baseline)
- Severity classification: NORMAL, MEDIUM, HIGH, CRITICAL
"""

import time
import threading
from collections import deque
from typing import Dict, Any, Tuple, Optional


class GatewaySpikeDetector:
    def __init__(
        self,
        window_seconds: float = 300.0,       # 5-minute sliding window
        min_sample_size: int = 5,            # Minimum transactions to trigger statistical alerts
        critical_fraud_rate: float = 25.0,   # % high-risk in window for CRITICAL
        high_fraud_rate: float = 15.0,       # % high-risk in window for HIGH
        medium_fraud_rate: float = 10.0,     # % high-risk in window for MEDIUM
        critical_velocity_count: int = 5,    # 5+ hard blocks within 60s triggers immediate CRITICAL
        recovery_fraud_rate: float = 5.0,    # % high-risk threshold for recovery
        baseline_score_mean: float = 0.035,  # Baseline expected mean score in clean traffic
    ):
        self.window_seconds = window_seconds
        self.min_sample_size = min_sample_size
        self.critical_fraud_rate = critical_fraud_rate
        self.high_fraud_rate = high_fraud_rate
        self.medium_fraud_rate = medium_fraud_rate
        self.critical_velocity_count = critical_velocity_count
        self.recovery_fraud_rate = recovery_fraud_rate
        self.baseline_score_mean = baseline_score_mean

        self._lock = threading.Lock()
        # Elements: (timestamp, prob, risk_tier, amount, entity_id)
        self._window = deque()

    def record_transaction(
        self,
        prob: float,
        risk_tier: str,
        amount: float = 0.0,
        entity_id: str = "anonymous",
        current_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Record a scored transaction into the sliding window and return
        updated gateway telemetry.
        """
        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            self._window.append((current_ts, float(prob), str(risk_tier), float(amount), str(entity_id)))
            self._prune(current_ts)
            return self._compute_telemetry(current_ts)

    def get_telemetry(self, current_ts: Optional[float] = None) -> Dict[str, Any]:
        """Get current sliding-window telemetry without adding a transaction."""
        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            self._prune(current_ts)
            return self._compute_telemetry(current_ts)

    def _prune(self, current_ts: float):
        cutoff = current_ts - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _compute_telemetry(self, current_ts: float) -> Dict[str, Any]:
        n = len(self._window)
        if n == 0:
            return {
                "window_seconds": self.window_seconds,
                "tx_count": 0,
                "high_risk_count": 0,
                "challenge_count": 0,
                "allow_count": 0,
                "high_risk_rate_pct": 0.0,
                "challenge_rate_pct": 0.0,
                "allow_rate_pct": 0.0,
                "mean_risk_prob": 0.0,
                "total_volume_amt": 0.0,
                "high_risk_volume_amt": 0.0,
                "spike_severity": "NORMAL",
                "is_spike": False,
                "score_drift_status": "STABLE",
                "burst_velocity_60s": 0,
            }

        high_risk_count = sum(1 for item in self._window if item[2] == "HARD_BLOCK")
        challenge_count = sum(1 for item in self._window if item[2] == "CHALLENGE")
        allow_count = sum(1 for item in self._window if item[2] == "ALLOW")

        probs = [item[1] for item in self._window]
        amounts = [item[3] for item in self._window]
        high_risk_amounts = [item[3] for item in self._window if item[2] == "HARD_BLOCK"]

        high_risk_rate = (high_risk_count / n) * 100.0
        challenge_rate = (challenge_count / n) * 100.0
        allow_rate = (allow_count / n) * 100.0
        mean_prob = sum(probs) / n
        total_amt = sum(amounts)
        high_risk_amt = sum(high_risk_amounts)

        # Immediate burst velocity check (last 60s)
        cutoff_60s = current_ts - 60.0
        burst_velocity_60s = sum(1 for item in self._window if item[0] >= cutoff_60s and item[2] == "HARD_BLOCK")

        # Severity determination
        severity = "NORMAL"
        is_spike = False

        if burst_velocity_60s >= self.critical_velocity_count:
            severity = "CRITICAL"
            is_spike = True
        elif n >= self.min_sample_size:
            if high_risk_rate >= self.critical_fraud_rate:
                severity = "CRITICAL"
                is_spike = True
            elif high_risk_rate >= self.high_fraud_rate:
                severity = "HIGH"
                is_spike = True
            elif high_risk_rate >= self.medium_fraud_rate or mean_prob >= 0.20:
                severity = "MEDIUM"
                is_spike = True

        # Score drift index (mean probability deviation)
        if mean_prob > (self.baseline_score_mean * 3.5):
            drift_status = "SIGNIFICANT_DRIFT"
        elif mean_prob > (self.baseline_score_mean * 2.0):
            drift_status = "MODERATE_DRIFT"
        else:
            drift_status = "STABLE"

        return {
            "window_seconds": self.window_seconds,
            "tx_count": n,
            "high_risk_count": high_risk_count,
            "challenge_count": challenge_count,
            "allow_count": allow_count,
            "high_risk_rate_pct": round(high_risk_rate, 2),
            "challenge_rate_pct": round(challenge_rate, 2),
            "allow_rate_pct": round(allow_rate, 2),
            "mean_risk_prob": round(mean_prob, 4),
            "total_volume_amt": round(total_amt, 2),
            "high_risk_volume_amt": round(high_risk_amt, 2),
            "spike_severity": severity,
            "is_spike": is_spike,
            "score_drift_status": drift_status,
            "burst_velocity_60s": burst_velocity_60s,
        }

    def is_healthy_for_recovery(self, current_ts: Optional[float] = None) -> bool:
        """
        Check if recent traffic has stabilized below recovery threshold.
        """
        telemetry = self.get_telemetry(current_ts)
        # Healthy if:
        # 1. No transactions or low sample size with 0 high-risk
        # 2. Or high-risk rate is below recovery threshold (<5%) and burst_60s is 0
        if telemetry["tx_count"] == 0:
            return True
        if telemetry["burst_velocity_60s"] > 0:
            return False
        return telemetry["high_risk_rate_pct"] < self.recovery_fraud_rate

    def clear(self):
        """Clear window for testing."""
        with self._lock:
            self._window.clear()
