"""
In-App Incident Management Engine
===================================
Tracks, persists, and manages structured fraud spike incidents throughout
their lifecycle (ACTIVE -> MITIGATING -> RESOLVED).

Features:
- Unique incident tracking with severity, trigger snapshot, and affected transactions.
- Automated creation upon spike detection and automated resolution upon recovery.
- Thread-safe persistence to data/incidents.jsonl.
- Manual resolution / notes capability for compliance and auditability.
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
INCIDENTS_LOG = os.path.join(DATA_DIR, "incidents.jsonl")


class IncidentManager:
    def __init__(self, persistence_path: str = INCIDENTS_LOG):
        self.persistence_path = persistence_path
        self._lock = threading.Lock()
        self._active_incident: Optional[Dict[str, Any]] = None
        self._load_active()

    def _load_active(self):
        if not os.path.exists(self.persistence_path):
            return

        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            inc = json.loads(line)
                            if inc.get("status") in ("ACTIVE", "MITIGATING"):
                                self._active_incident = inc
                        except Exception:
                            continue
        except Exception:
            self._active_incident = None

    def _append_record(self, record: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        with open(self.persistence_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def create_incident(
        self,
        severity: str,
        trigger_metrics: Dict[str, Any],
        current_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create and record a new incident if one is not already active.
        If an incident is already active, escalate severity if higher.
        """
        if current_ts is None:
            current_ts = time.time()

        now_iso = datetime.fromtimestamp(current_ts, timezone.utc).isoformat()

        with self._lock:
            if self._active_incident:
                # If existing incident, check for escalation
                order = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
                curr_sev = self._active_incident.get("severity", "MEDIUM")
                if order.get(severity, 1) > order.get(curr_sev, 1):
                    self._active_incident["severity"] = severity
                    self._active_incident["last_escalated_at"] = now_iso
                    self._append_record(self._active_incident)
                return self._active_incident

            inc_id = f"inc_{int(current_ts)}_{int(current_ts*1000)%1000}"
            incident = {
                "incident_id": inc_id,
                "status": "MITIGATING",  # Mitigation / circuit breaker engaged
                "severity": severity,
                "started_at": now_iso,
                "started_ts": current_ts,
                "resolved_at": None,
                "resolved_ts": None,
                "duration_seconds": None,
                "trigger_metrics": trigger_metrics,
                "affected_transactions_count": 0,
                "resolution_reason": None,
            }
            self._active_incident = incident
            self._append_record(incident)
            return incident

    def record_affected_transaction(self):
        """Increment count of transactions evaluated under active mitigation."""
        with self._lock:
            if self._active_incident:
                self._active_incident["affected_transactions_count"] = (
                    self._active_incident.get("affected_transactions_count", 0) + 1
                )

    def resolve_incident(
        self,
        incident_id: Optional[str] = None,
        reason: str = "Auto-recovered: gateway traffic returned to normal thresholds",
        current_ts: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve the active incident (or specific incident_id).
        """
        if current_ts is None:
            current_ts = time.time()

        now_iso = datetime.fromtimestamp(current_ts, timezone.utc).isoformat()

        with self._lock:
            if not self._active_incident:
                return None

            if incident_id and self._active_incident.get("incident_id") != incident_id:
                return None

            inc = self._active_incident
            inc["status"] = "RESOLVED"
            inc["resolved_at"] = now_iso
            inc["resolved_ts"] = current_ts
            inc["duration_seconds"] = round(current_ts - inc.get("started_ts", current_ts), 1)
            inc["resolution_reason"] = reason

            self._append_record(inc)
            resolved = inc.copy()
            self._active_incident = None
            return resolved

    def get_active_incident(self) -> Optional[Dict[str, Any]]:
        """Return the current active incident if any."""
        with self._lock:
            return self._active_incident.copy() if self._active_incident else None

    def get_all_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent incidents from storage."""
        if not os.path.exists(self.persistence_path):
            return []

        incidents_map = {}
        with self._lock:
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                inc = json.loads(line)
                                # Map by ID to keep latest updated state
                                incidents_map[inc["incident_id"]] = inc
                            except Exception:
                                continue
            except Exception:
                return []

        all_incs = list(incidents_map.values())
        all_incs.sort(key=lambda x: x.get("started_ts", 0), reverse=True)
        return all_incs[:limit]

    def clear(self):
        """Clear for tests."""
        with self._lock:
            self._active_incident = None
            if os.path.exists(self.persistence_path):
                try:
                    os.remove(self.persistence_path)
                except Exception:
                    pass
