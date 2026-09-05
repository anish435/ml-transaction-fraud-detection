"""
Automated Defense-Only Circuit Breaker
======================================
Autonomous safety controller that temporarily adjusts operational routing
thresholds when a systemic fraud spike is detected, and automatically recovers
once traffic returns to normal.

Guarantees:
- Purely defensive: Adjusts routing thresholds (ALLOW / CHALLENGE / HARD_BLOCK).
  Does NOT trigger automatic refunds or irreversible financial actions.
- Reversible: Restores exact standard baseline thresholds automatically after cooldown.
- Configurable & Logged: Full audit history of state changes and trigger conditions.
"""

import time
import threading
from typing import Dict, Tuple, Optional, Any


class DefenseCircuitBreaker:
    def __init__(
        self,
        base_p_low: float = 0.0804,          # Standard ALLOW / CHALLENGE boundary
        base_p_high: float = 0.7495,         # Standard CHALLENGE / HARD_BLOCK boundary
        defense_p_low: float = 0.0400,       # Tightened ALLOW / CHALLENGE boundary during spike
        defense_p_high: float = 0.4500,      # Tightened CHALLENGE / HARD_BLOCK boundary during spike
        cooldown_seconds: float = 60.0,      # Cooldown duration required before auto-recovery
        min_healthy_tx_count: int = 10,      # Number of consecutive healthy transactions for recovery
    ):
        self.base_p_low = base_p_low
        self.base_p_high = base_p_high
        self.defense_p_low = defense_p_low
        self.defense_p_high = defense_p_high
        self.cooldown_seconds = cooldown_seconds
        self.min_healthy_tx_count = min_healthy_tx_count

        self._lock = threading.Lock()
        # States: "NORMAL", "DEFENSE_ACTIVE", "COOLDOWN"
        self._state = "NORMAL"
        self._last_state_change = time.time()
        self._trip_reason: Optional[str] = None
        self._trip_severity: Optional[str] = None
        self._healthy_streak: int = 0
        self._cooldown_started_at: Optional[float] = None

    def get_state(self) -> str:
        with self._lock:
            return self._state

    def get_active_thresholds(self) -> Tuple[float, float, str]:
        """
        Return the currently applicable (p_low, p_high, state).
        """
        with self._lock:
            if self._state in ("DEFENSE_ACTIVE", "COOLDOWN"):
                return self.defense_p_low, self.defense_p_high, self._state
            return self.base_p_low, self.base_p_high, self._state

    def evaluate_traffic_and_update(
        self,
        telemetry: Dict[str, Any],
        is_healthy: bool,
        current_ts: Optional[float] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluates current gateway telemetry and updates circuit breaker state.
        Returns: (new_state, transition_event_or_None)
        """
        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            is_spike = telemetry.get("is_spike", False)
            severity = telemetry.get("spike_severity", "NORMAL")

            # 1. State: NORMAL -> check for trip
            if self._state == "NORMAL":
                if is_spike:
                    self._state = "DEFENSE_ACTIVE"
                    self._last_state_change = current_ts
                    self._trip_reason = f"Automated Trip: {severity} fraud spike detected ({telemetry.get('high_risk_rate_pct')}% fraud rate, {telemetry.get('burst_velocity_60s')} bursts in 60s)"
                    self._trip_severity = severity
                    self._healthy_streak = 0
                    self._cooldown_started_at = None
                    return self._state, "TRIPPED"
                return self._state, None

            # 2. State: DEFENSE_ACTIVE -> check if ready to enter COOLDOWN
            elif self._state == "DEFENSE_ACTIVE":
                if is_spike:
                    # Still spiking; keep active
                    self._healthy_streak = 0
                    self._cooldown_started_at = None
                    return self._state, None
                elif is_healthy:
                    # Healthy traffic observed -> begin cooldown
                    self._state = "COOLDOWN"
                    self._cooldown_started_at = current_ts
                    self._last_state_change = current_ts
                    self._healthy_streak = 1
                    return self._state, "ENTER_COOLDOWN"
                return self._state, None

            # 3. State: COOLDOWN -> evaluate recovery or re-trip
            elif self._state == "COOLDOWN":
                if is_spike:
                    # Spike returned during cooldown -> re-trip immediately
                    self._state = "DEFENSE_ACTIVE"
                    self._last_state_change = current_ts
                    self._trip_reason = f"Re-tripped during cooldown: {severity} spike recurred"
                    self._trip_severity = severity
                    self._healthy_streak = 0
                    self._cooldown_started_at = None
                    return self._state, "RE_TRIPPED"

                if is_healthy:
                    self._healthy_streak += 1

                time_in_cooldown = current_ts - (self._cooldown_started_at or current_ts)
                if time_in_cooldown >= self.cooldown_seconds and self._healthy_streak >= self.min_healthy_tx_count:
                    # Fully recovered!
                    self._state = "NORMAL"
                    self._last_state_change = current_ts
                    self._trip_reason = None
                    self._trip_severity = None
                    self._healthy_streak = 0
                    self._cooldown_started_at = None
                    return self._state, "AUTO_RECOVERED"

                return self._state, None

            return self._state, None

    def manual_trip(self, reason: str = "Manual operator emergency trip", severity: str = "HIGH", current_ts: Optional[float] = None):
        """Force circuit breaker into DEFENSE_ACTIVE."""
        if current_ts is None:
            current_ts = time.time()
        with self._lock:
            self._state = "DEFENSE_ACTIVE"
            self._last_state_change = current_ts
            self._trip_reason = reason
            self._trip_severity = severity
            self._healthy_streak = 0
            self._cooldown_started_at = None

    def manual_reset(self, current_ts: Optional[float] = None):
        """Force circuit breaker back into NORMAL state."""
        if current_ts is None:
            current_ts = time.time()
        with self._lock:
            self._state = "NORMAL"
            self._last_state_change = current_ts
            self._trip_reason = None
            self._trip_severity = None
            self._healthy_streak = 0
            self._cooldown_started_at = None

    def get_status(self, current_ts: Optional[float] = None) -> Dict[str, Any]:
        """Return comprehensive circuit breaker diagnostic status."""
        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            p_low, p_high = (self.defense_p_low, self.defense_p_high) if self._state != "NORMAL" else (self.base_p_low, self.base_p_high)
            cooldown_progress = 0.0
            if self._state == "COOLDOWN" and self._cooldown_started_at:
                elapsed = current_ts - self._cooldown_started_at
                cooldown_progress = min(100.0, round((elapsed / max(self.cooldown_seconds, 1)) * 100, 1))

            return {
                "state": self._state,
                "is_defense_active": self._state in ("DEFENSE_ACTIVE", "COOLDOWN"),
                "active_thresholds": {
                    "p_low": p_low,
                    "p_high": p_high,
                },
                "standard_thresholds": {
                    "p_low": self.base_p_low,
                    "p_high": self.base_p_high,
                },
                "defense_thresholds": {
                    "p_low": self.defense_p_low,
                    "p_high": self.defense_p_high,
                },
                "last_state_change_epoch": self._last_state_change,
                "seconds_in_current_state": round(current_ts - self._last_state_change, 1),
                "trip_reason": self._trip_reason,
                "trip_severity": self._trip_severity,
                "healthy_streak": self._healthy_streak,
                "cooldown_progress_pct": cooldown_progress,
            }
