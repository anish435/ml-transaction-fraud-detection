"""
Entity Suppression & Temporary Blacklist Engine
=================================================
Maintains thread-safe temporary suppression/blacklisting for entities
(e.g., customer email, contact phone, card fingerprint) that exhibit
repeated high-risk or HARD_BLOCK violations within a sliding time window.

Safety Guarantees:
- Fully reversible: Entries automatically expire via configurable TTL.
- Manual override: Immediate unblock capability via API or Dashboard.
- Thread-safe: Synchronized across concurrent webhook requests.
- Persistent: State persisted to data/suppression_list.json.
"""

import os
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
SUPPRESSION_FILE = os.path.join(DATA_DIR, "suppression_list.json")


class EntitySuppressionStore:
    def __init__(
        self,
        persistence_path: str = SUPPRESSION_FILE,
        violation_window_seconds: float = 600.0,   # 10 minutes
        violation_threshold: int = 3,              # 3 hard blocks trigger suppression
        default_ttl_seconds: float = 1800.0,       # 30 minutes TTL
    ):
        self.persistence_path = persistence_path
        self.violation_window = violation_window_seconds
        self.violation_threshold = violation_threshold
        self.default_ttl = default_ttl_seconds
        self._lock = threading.Lock()

        # {entity_id: [ts1, ts2, ...]}
        self._violations: Dict[str, List[float]] = {}
        # {entity_id: {"suppressed_at": ts, "expires_at": ts, "reason": str, "violation_count": int}}
        self._suppressions: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._suppressions = data.get("suppressions", {})
            except Exception:
                self._suppressions = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump({"suppressions": self._suppressions}, f, indent=2)
        except Exception:
            pass

    def record_violation(
        self,
        entity_id: str,
        violation_type: str = "HARD_BLOCK",
        current_ts: Optional[float] = None,
        custom_ttl: Optional[float] = None,
    ) -> bool:
        """
        Record a high-risk violation. If violations >= threshold in window,
        suppress entity with TTL. Returns True if newly suppressed.
        """
        if not entity_id or entity_id in ("anonymous", "unknown", "anonymous_customer"):
            return False

        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            # Check if already suppressed
            if entity_id in self._suppressions:
                if self._suppressions[entity_id]["expires_at"] > current_ts:
                    return False  # Already actively suppressed

            # Prune and record violation timestamp
            cutoff = current_ts - self.violation_window
            history = [ts for ts in self._violations.get(entity_id, []) if ts >= cutoff]
            history.append(current_ts)
            self._violations[entity_id] = history

            if len(history) >= self.violation_threshold:
                ttl = custom_ttl or self.default_ttl
                self._suppressions[entity_id] = {
                    "entity_id": entity_id,
                    "suppressed_at": current_ts,
                    "expires_at": current_ts + ttl,
                    "ttl_seconds": ttl,
                    "reason": f"Repeated violations ({len(history)} {violation_type} events within {int(self.violation_window/60)}m)",
                    "violation_count": len(history),
                }
                self._save()
                return True

        return False

    def is_suppressed(self, entity_id: str, current_ts: Optional[float] = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if entity is currently suppressed. Automatically prunes expired entries.
        Returns: (is_suppressed, suppression_details)
        """
        if not entity_id or entity_id in ("anonymous", "unknown", "anonymous_customer"):
            return False, None

        if current_ts is None:
            current_ts = time.time()

        with self._lock:
            if entity_id in self._suppressions:
                entry = self._suppressions[entity_id]
                if entry["expires_at"] > current_ts:
                    remaining = max(0.0, round(entry["expires_at"] - current_ts, 1))
                    details = entry.copy()
                    details["remaining_ttl_seconds"] = remaining
                    return True, details
                else:
                    # Expired -> prune
                    del self._suppressions[entity_id]
                    self._save()

        return False, None

    def remove_suppression(self, entity_id: str) -> bool:
        """Manually unblock/remove suppression for an entity."""
        with self._lock:
            if entity_id in self._suppressions:
                del self._suppressions[entity_id]
                self._save()
                return True
        return False

    def get_active_suppressions(self, current_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return list of all currently active suppressed entities."""
        if current_ts is None:
            current_ts = time.time()

        active = []
        with self._lock:
            expired_keys = []
            for eid, entry in self._suppressions.items():
                if entry["expires_at"] > current_ts:
                    item = entry.copy()
                    item["remaining_ttl_seconds"] = max(0.0, round(entry["expires_at"] - current_ts, 1))
                    active.append(item)
                else:
                    expired_keys.append(eid)

            if expired_keys:
                for eid in expired_keys:
                    del self._suppressions[eid]
                self._save()

        return sorted(active, key=lambda x: x["remaining_ttl_seconds"], reverse=True)

    def clear(self):
        """Clear store for tests."""
        with self._lock:
            self._violations = {}
            self._suppressions = {}
            self._save()
