"""
Defense Subsystem Package
==========================
Coordinates Real-Time Gateway Spike Detection, Automated Circuit Breaking,
In-App Incident Management, and Temporary Entity Suppression.
"""

from typing import Dict, Any, Optional, Tuple

from src.defense.spike_detector import GatewaySpikeDetector
from src.defense.circuit_breaker import DefenseCircuitBreaker
from src.defense.incident_manager import IncidentManager
from src.defense.entity_suppression import EntitySuppressionStore


class DefenseSystem:
    """
    Central coordinator integrating all defensive components.
    Singleton-friendly and thread-safe.
    """
    def __init__(
        self,
        base_p_low: float = 0.0804,
        base_p_high: float = 0.7495,
        defense_p_low: float = 0.0400,
        defense_p_high: float = 0.4500,
        cooldown_seconds: float = 60.0,
    ):
        self.spike_detector = GatewaySpikeDetector()
        self.circuit_breaker = DefenseCircuitBreaker(
            base_p_low=base_p_low,
            base_p_high=base_p_high,
            defense_p_low=defense_p_low,
            defense_p_high=defense_p_high,
            cooldown_seconds=cooldown_seconds,
        )
        self.incident_manager = IncidentManager()
        self.suppression_store = EntitySuppressionStore()

    def process_transaction(
        self,
        prob: float,
        entity_id: str = "anonymous",
        amount: float = 0.0,
        current_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a scored transaction through the defense lifecycle:
        1. Check temporary entity suppression.
        2. Resolve operational tier against active circuit breaker thresholds.
        3. Feed sliding-window spike detector.
        4. Track violations for repeated abuse suppression.
        5. Trigger automated incident creation or auto-recovery.

        Returns comprehensive defense decision metadata.
        """
        # Step 1: Check entity suppression first
        is_supp, supp_details = self.suppression_store.is_suppressed(entity_id, current_ts)
        if is_supp:
            risk_tier = "HARD_BLOCK"
            defense_note = f"TEMPORARY_SUPPRESSION_ACTIVE ({supp_details.get('reason')})"
            # Record in detector
            telemetry = self.spike_detector.record_transaction(
                prob=max(prob, 0.90),
                risk_tier=risk_tier,
                amount=amount,
                entity_id=entity_id,
                current_ts=current_ts,
            )
            return {
                "risk_tier": risk_tier,
                "circuit_breaker_state": self.circuit_breaker.get_state(),
                "is_entity_suppressed": True,
                "suppression_details": supp_details,
                "defense_action": "ENFORCED_SUPPRESSION",
                "defense_note": defense_note,
                "telemetry": telemetry,
            }

        # Step 2: Determine operational routing tier based on active circuit breaker thresholds
        p_low, p_high, cb_state = self.circuit_breaker.get_active_thresholds()
        if prob < p_low:
            risk_tier = "ALLOW"
        elif prob < p_high:
            risk_tier = "CHALLENGE"
        else:
            risk_tier = "HARD_BLOCK"

        defense_action = "STANDARD_ROUTING"
        defense_note = None

        if cb_state in ("DEFENSE_ACTIVE", "COOLDOWN"):
            defense_action = "DEFENSE_TIGHTENED_ROUTING"
            defense_note = f"Active Circuit Breaker ({cb_state}): Tightened thresholds [ALLOW < {p_low:.4f} <= CHALLENGE < {p_high:.4f} <= BLOCK]"
            self.incident_manager.record_affected_transaction()

        # Step 3: Record transaction in gateway-wide sliding window
        telemetry = self.spike_detector.record_transaction(
            prob=prob,
            risk_tier=risk_tier,
            amount=amount,
            entity_id=entity_id,
            current_ts=current_ts,
        )

        # Step 4: Track entity violation if HARD_BLOCK
        if risk_tier == "HARD_BLOCK":
            newly_suppressed = self.suppression_store.record_violation(
                entity_id=entity_id,
                violation_type="HARD_BLOCK",
                current_ts=current_ts,
            )
            if newly_suppressed:
                defense_note = (defense_note + " | " if defense_note else "") + f"Entity '{entity_id}' placed on Temporary Suppression List (3+ violations in window)"

        # Step 5: Evaluate gateway traffic and handle circuit breaker state transitions
        is_healthy = self.spike_detector.is_healthy_for_recovery(current_ts)
        new_state, transition = self.circuit_breaker.evaluate_traffic_and_update(
            telemetry=telemetry,
            is_healthy=is_healthy,
            current_ts=current_ts,
        )

        if transition in ("TRIPPED", "RE_TRIPPED"):
            self.incident_manager.create_incident(
                severity=telemetry["spike_severity"],
                trigger_metrics=telemetry,
                current_ts=current_ts,
            )
        elif transition == "AUTO_RECOVERED":
            self.incident_manager.resolve_incident(
                reason="Auto-recovered: gateway traffic returned to normal thresholds for sustained cooldown window",
                current_ts=current_ts,
            )

        return {
            "risk_tier": risk_tier,
            "circuit_breaker_state": new_state,
            "is_entity_suppressed": False,
            "suppression_details": None,
            "defense_action": defense_action,
            "defense_note": defense_note,
            "active_thresholds": {"p_low": p_low, "p_high": p_high},
            "telemetry": telemetry,
        }

    def get_full_status(self, current_ts: Optional[float] = None) -> Dict[str, Any]:
        """Get snapshot of entire defense subsystem."""
        cb_status = self.circuit_breaker.get_status(current_ts)
        telemetry = self.spike_detector.get_telemetry(current_ts)
        active_inc = self.incident_manager.get_active_incident()
        suppressions = self.suppression_store.get_active_suppressions(current_ts)

        return {
            "circuit_breaker": cb_status,
            "sliding_window_telemetry": telemetry,
            "active_incident": active_inc,
            "suppressed_entities_count": len(suppressions),
            "suppressed_entities": suppressions,
        }


# Global singleton instance
defense_system = DefenseSystem()
