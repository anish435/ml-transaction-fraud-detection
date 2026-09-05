"""
Unit & Integration Tests: Defense Circuit Breaker, Gateway Spike Detector,
In-App Incident Manager, and Entity Suppression.
"""

import sys
import os
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.defense import DefenseSystem


class TestDefenseCircuitBreaker(unittest.TestCase):
    def setUp(self):
        # Create dedicated DefenseSystem instance with quick cooldown for testing
        self.defense = DefenseSystem(
            base_p_low=0.0804,
            base_p_high=0.7495,
            defense_p_low=0.0400,
            defense_p_high=0.4500,
            cooldown_seconds=2.0,   # 2s cooldown for rapid testing
        )
        self.defense.circuit_breaker.min_healthy_tx_count = 3
        self.defense.spike_detector.clear()
        self.defense.incident_manager.clear()
        self.defense.suppression_store.clear()

    def test_normal_traffic_baseline(self):
        """Test that normal, low-risk traffic maintains NORMAL state with no alerts or incidents."""
        t0 = 1000.0
        for i in range(10):
            res = self.defense.process_transaction(
                prob=0.01,
                entity_id=f"user_{i}@legit.com",
                amount=150.0,
                current_ts=t0 + i,
            )
            self.assertEqual(res["risk_tier"], "ALLOW")
            self.assertEqual(res["circuit_breaker_state"], "NORMAL")
            self.assertFalse(res["is_entity_suppressed"])

        status = self.defense.get_full_status(current_ts=t0 + 10)
        self.assertEqual(status["circuit_breaker"]["state"], "NORMAL")
        self.assertEqual(status["sliding_window_telemetry"]["spike_severity"], "NORMAL")
        self.assertIsNone(status["active_incident"])

    def test_spike_detection_and_circuit_breaker_trip(self):
        """Test that a surge of high-risk transactions trips the circuit breaker and creates an incident."""
        t0 = 2000.0
        # Send 2 normal
        for i in range(2):
            self.defense.process_transaction(0.01, f"user_{i}@domain.com", 100.0, t0 + i)

        # Inject 6 high-risk fraud attacks in 10 seconds
        for i in range(6):
            res = self.defense.process_transaction(
                prob=0.88,
                entity_id=f"attacker_{i}@botnet.org",
                amount=2500.0,
                current_ts=t0 + 10 + i,
            )

        # Circuit breaker should now be DEFENSE_ACTIVE
        self.assertEqual(res["circuit_breaker_state"], "DEFENSE_ACTIVE")

        status = self.defense.get_full_status(current_ts=t0 + 20)
        self.assertEqual(status["circuit_breaker"]["state"], "DEFENSE_ACTIVE")
        self.assertTrue(status["circuit_breaker"]["is_defense_active"])
        self.assertIn(status["sliding_window_telemetry"]["spike_severity"], ("HIGH", "CRITICAL"))

        # Incident should be active
        active_inc = status["active_incident"]
        self.assertIsNotNone(active_inc)
        self.assertEqual(active_inc["status"], "MITIGATING")
        self.assertIn(active_inc["severity"], ("HIGH", "CRITICAL"))

    def test_defense_routing_tightened_thresholds(self):
        """
        Test that during DEFENSE_ACTIVE, a borderline transaction (e.g. prob=0.50)
        which would normally be CHALLENGE is elevated to HARD_BLOCK!
        """
        t0 = 3000.0
        # Force trip or send attack
        self.defense.circuit_breaker.manual_trip("Unit test attack simulation", "CRITICAL", t0)
        self.assertEqual(self.defense.circuit_breaker.get_state(), "DEFENSE_ACTIVE")

        # Standard: 0.50 < 0.7495 -> would be CHALLENGE
        # Defense : 0.50 >= 0.4500 -> must be elevated to HARD_BLOCK
        res = self.defense.process_transaction(
            prob=0.50,
            entity_id="victim_customer@domain.com",
            amount=800.0,
            current_ts=t0 + 5,
        )
        self.assertEqual(res["risk_tier"], "HARD_BLOCK")
        self.assertEqual(res["defense_action"], "DEFENSE_TIGHTENED_ROUTING")
        self.assertIn("Tightened thresholds", res["defense_note"])

    def test_temporary_entity_suppression_and_manual_unblock(self):
        """Test that repeated abuse by the same entity triggers temporary suppression with TTL."""
        t0 = 4000.0
        bad_actor = "fraud_ring_boss@darkweb.io"

        # 1st violation
        self.defense.process_transaction(0.85, bad_actor, 1000.0, t0 + 1)
        is_supp, _ = self.defense.suppression_store.is_suppressed(bad_actor, t0 + 2)
        self.assertFalse(is_supp)

        # 2nd violation
        self.defense.process_transaction(0.92, bad_actor, 1200.0, t0 + 10)
        is_supp, _ = self.defense.suppression_store.is_suppressed(bad_actor, t0 + 11)
        self.assertFalse(is_supp)

        # 3rd violation -> should trigger suppression!
        res3 = self.defense.process_transaction(0.95, bad_actor, 1500.0, t0 + 20)
        is_supp, details = self.defense.suppression_store.is_suppressed(bad_actor, t0 + 21)
        self.assertTrue(is_supp)
        self.assertIn("Repeated violations", details["reason"])

        # 4th transaction (even if low amount) must be automatically blocked by suppression
        res4 = self.defense.process_transaction(0.05, bad_actor, 50.0, t0 + 30)
        self.assertEqual(res4["risk_tier"], "HARD_BLOCK")
        self.assertTrue(res4["is_entity_suppressed"])
        self.assertEqual(res4["defense_action"], "ENFORCED_SUPPRESSION")

        # Test manual unblock
        unblocked = self.defense.suppression_store.remove_suppression(bad_actor)
        self.assertTrue(unblocked)
        is_supp_after, _ = self.defense.suppression_store.is_suppressed(bad_actor, t0 + 35)
        self.assertFalse(is_supp_after)

    def test_cooldown_and_auto_recovery_cycle(self):
        """Test the full lifecycle: Trip -> Cooldown -> Auto-Recovery -> Incident Resolved."""
        t0 = 5000.0
        # 1. Trip circuit breaker with attack
        for i in range(6):
            self.defense.process_transaction(0.88, f"bot_{i}@net.com", 2000.0, t0 + i)

        self.assertEqual(self.defense.circuit_breaker.get_state(), "DEFENSE_ACTIVE")
        active_inc = self.defense.incident_manager.get_active_incident()
        self.assertIsNotNone(active_inc)
        self.assertEqual(active_inc["status"], "MITIGATING")

        # 2. Advance time past the sliding window so the attack drops out of the active window
        t_clean = t0 + 350.0  # 350s later (> 300s window)

        # 3. Stream healthy transactions into the system
        for i in range(5):
            res = self.defense.process_transaction(
                prob=0.01,
                entity_id=f"clean_user_{i}@good.org",
                amount=120.0,
                current_ts=t_clean + i,
            )

        # After healthy transactions, state should transition to COOLDOWN
        state = self.defense.circuit_breaker.get_state()
        self.assertIn(state, ("COOLDOWN", "NORMAL"))

        # Wait past cooldown_seconds (cooldown_seconds=2.0) and feed healthy transactions
        t_recovered = t_clean + 10.0
        for i in range(5):
            res = self.defense.process_transaction(
                prob=0.01,
                entity_id=f"clean_user_rec_{i}@good.org",
                amount=120.0,
                current_ts=t_recovered + i,
            )

        # Should now be fully AUTO_RECOVERED back to NORMAL
        self.assertEqual(self.defense.circuit_breaker.get_state(), "NORMAL")

        # Active incident should now be RESOLVED
        status = self.defense.get_full_status(t_recovered + 10)
        self.assertIsNone(status["active_incident"])

        all_incidents = self.defense.incident_manager.get_all_incidents(limit=5)
        self.assertTrue(len(all_incidents) >= 1)
        self.assertEqual(all_incidents[0]["status"], "RESOLVED")
        self.assertIn("Auto-recovered", all_incidents[0]["resolution_reason"])


if __name__ == "__main__":
    unittest.main()
