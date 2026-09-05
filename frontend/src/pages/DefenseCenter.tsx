import React, { useState } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Flame,
  Activity,
  UserX,
  RotateCcw,
  Sliders,
  CheckCircle2,
  Lock,
  Unlock,
  Info,
} from "lucide-react";
import { DefenseStatus, SuppressedEntity } from "../types/api";
import { api } from "../services/api";

interface DefenseCenterProps {
  defense: DefenseStatus | null;
  onRefresh: () => void;
  onNotification?: (msg: { text: string; type: "success" | "warning" | "info" | "error" }) => void;
}

export const DefenseCenter: React.FC<DefenseCenterProps> = ({
  defense,
  onRefresh,
  onNotification,
}) => {
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  if (!defense) {
    return (
      <div className="p-8 text-center text-slate-400">Loading Defense Telemetry...</div>
    );
  }

  const { circuit_breaker, sliding_window_telemetry, active_incident, suppressed_entities } = defense;
  const isDefenseActive = circuit_breaker.state === "DEFENSE_ACTIVE";

  const handleManualTrip = async () => {
    if (!confirm("Confirm: Manually trigger Circuit Breaker defense tightening?")) return;
    setActionLoading("trip");
    try {
      await api.tripCircuitBreaker("Emergency operator trip from web console", "HIGH");
      onNotification?.({
        text: "Circuit Breaker manually engaged to DEFENSE_ACTIVE.",
        type: "warning",
      });
      onRefresh();
    } catch (err: any) {
      onNotification?.({
        text: `Error engaging circuit breaker: ${err.message}`,
        type: "error",
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleManualReset = async () => {
    setActionLoading("reset");
    try {
      await api.resetCircuitBreaker();
      onNotification?.({
        text: "Circuit Breaker manually restored to NORMAL mode.",
        type: "info",
      });
      onRefresh();
    } catch (err: any) {
      onNotification?.({
        text: `Error resetting circuit breaker: ${err.message}`,
        type: "error",
      });
    } finally {
      setActionLoading(null);
    }
  };

  const handleUnblockEntity = async (entityId: string) => {
    try {
      await api.removeSuppression(entityId);
      onNotification?.({
        text: `Entity '${entityId}' unblocked from temporary suppression list.`,
        type: "success",
      });
      onRefresh();
    } catch (err: any) {
      onNotification?.({
        text: `Error removing suppression: ${err.message}`,
        type: "error",
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            Adaptive Gateway Defense Center
            {isDefenseActive ? (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse">
                DEFENSE ENGAGED
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ACTIVE MONITORING
              </span>
            )}
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Real-time sliding window telemetry monitor and dynamic circuit breaker.
            When elevated risk or botnet bursts are detected, the gateway automatically tightens
            routing thresholds to isolate malicious traffic without performing irreversible financial actions.
          </p>
        </div>

        {/* Emergency Manual Controls */}
        <div className="flex items-center gap-2">
          {isDefenseActive ? (
            <button
              onClick={handleManualReset}
              disabled={actionLoading !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition-all disabled:opacity-50"
            >
              <RotateCcw className="w-3.5 h-3.5 text-emerald-400" />
              <span>Reset to NORMAL</span>
            </button>
          ) : (
            <button
              onClick={handleManualTrip}
              disabled={actionLoading !== null}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/40 text-xs font-semibold transition-all disabled:opacity-50"
            >
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>Emergency Defense Trip</span>
            </button>
          )}
        </div>
      </div>

      {/* Grid: Circuit Breaker Telemetry + Risk Spike Monitor */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Card 1: Adaptive Circuit Breaker */}
        <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-dark-700/60">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-brand-cyan/10 border border-brand-cyan/30 flex items-center justify-center text-brand-cyan">
                <Sliders className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">Circuit Breaker Routing State</h3>
                <p className="text-xs text-slate-400">Dynamic operational threshold control</p>
              </div>
            </div>
            <span className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold uppercase ${
              isDefenseActive
                ? "bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse"
                : circuit_breaker.state === "COOLDOWN"
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            }`}>
              {circuit_breaker.state}
            </span>
          </div>

          <div className="bg-dark-900/70 p-4 rounded-lg border border-dark-700/50 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-dark-800 border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block font-medium">Standard Thresholds</span>
                <div className="mt-1 text-xs text-slate-200 font-mono">
                  <span className="text-emerald-400 font-semibold">ALLOW</span> &lt; {circuit_breaker.standard_thresholds.p_low.toFixed(4)}
                  <br />
                  <span className="text-rose-400 font-semibold">BLOCK</span> &ge; {circuit_breaker.standard_thresholds.p_high.toFixed(4)}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-dark-800 border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block font-medium">Tightened Defense Thresholds</span>
                <div className="mt-1 text-xs text-slate-200 font-mono">
                  <span className="text-emerald-400 font-semibold">ALLOW</span> &lt; {circuit_breaker.defense_thresholds.p_low.toFixed(4)}
                  <br />
                  <span className="text-rose-400 font-semibold">BLOCK</span> &ge; {circuit_breaker.defense_thresholds.p_high.toFixed(4)}
                </div>
              </div>
            </div>

            <div className="flex items-start gap-2 text-xs text-slate-400 pt-1">
              <Info className="w-4 h-4 text-brand-cyan shrink-0 mt-0.5" />
              <span>
                Adaptive defense temporarily tightens routing thresholds when elevated risk is detected.
                This is a defense-only adaptive routing mechanism and does not trigger financial cancellations.
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div className="bg-dark-800 p-2.5 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 block uppercase font-medium">Active Allow Tier</span>
              <span className="text-xs font-mono font-bold text-emerald-400 mt-0.5 block">
                p &lt; {circuit_breaker.active_thresholds.p_low.toFixed(4)}
              </span>
            </div>
            <div className="bg-dark-800 p-2.5 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 block uppercase font-medium">Active Block Tier</span>
              <span className="text-xs font-mono font-bold text-rose-400 mt-0.5 block">
                p &ge; {circuit_breaker.active_thresholds.p_high.toFixed(4)}
              </span>
            </div>
            <div className="bg-dark-800 p-2.5 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 block uppercase font-medium">Time In State</span>
              <span className="text-xs font-mono font-bold text-slate-200 mt-0.5 block">
                {Math.round(circuit_breaker.seconds_in_current_state)}s
              </span>
            </div>
          </div>
        </div>

        {/* Card 2: Risk Spike Monitor */}
        <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-dark-700/60">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400">
                <Flame className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">Sliding-Window Risk Spike Monitor</h3>
                <p className="text-xs text-slate-400">5-minute rolling transaction telemetry</p>
              </div>
            </div>
            <span className={`px-2.5 py-1 rounded-md text-xs font-bold ${
              sliding_window_telemetry.spike_severity === "CRITICAL"
                ? "bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse"
                : sliding_window_telemetry.spike_severity === "HIGH"
                ? "bg-orange-500/20 text-orange-400 border border-orange-500/40"
                : sliding_window_telemetry.spike_severity === "MEDIUM"
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            }`}>
              {sliding_window_telemetry.spike_severity}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-dark-800 p-3 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 uppercase font-medium">Window Volume</span>
              <span className="text-base font-mono font-bold text-white mt-1 block">
                {sliding_window_telemetry.tx_count}
              </span>
            </div>

            <div className="bg-dark-800 p-3 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 uppercase font-medium">High-Risk Rate</span>
              <span className={`text-base font-mono font-bold mt-1 block ${
                sliding_window_telemetry.high_risk_rate_pct > 15 ? "text-rose-400" : "text-slate-200"
              }`}>
                {sliding_window_telemetry.high_risk_rate_pct.toFixed(1)}%
              </span>
            </div>

            <div className="bg-dark-800 p-3 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 uppercase font-medium">Hard Block Burst</span>
              <span className="text-base font-mono font-bold text-rose-400 mt-1 block">
                {sliding_window_telemetry.burst_velocity_60s} <span className="text-[10px] text-slate-400 font-normal">/ 60s</span>
              </span>
            </div>

            <div className="bg-dark-800 p-3 rounded-lg border border-dark-700/60">
              <span className="text-[10px] text-slate-400 uppercase font-medium">Drift Status</span>
              <span className="text-xs font-bold text-brand-cyan mt-1 block truncate">
                {sliding_window_telemetry.score_drift_status}
              </span>
            </div>
          </div>

          <div className="space-y-2 pt-2 border-t border-dark-700/50 text-xs">
            <div className="flex justify-between items-center text-slate-300">
              <span>Rolling Mean Risk Probability</span>
              <span className="font-mono font-bold text-slate-100">
                {(sliding_window_telemetry.mean_risk_prob * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span>Window Currency Volume</span>
              <span className="font-mono font-bold text-slate-100">
                ${sliding_window_telemetry.total_volume_amt.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
            <div className="flex justify-between items-center text-slate-300">
              <span>High-Risk Intercepted Volume</span>
              <span className="font-mono font-bold text-rose-400">
                ${sliding_window_telemetry.high_risk_volume_amt.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Temporary Entity Suppression Table */}
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-dark-700/60">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand-purple/10 border border-brand-purple/30 flex items-center justify-center text-brand-purple">
              <UserX className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Temporary Entity Suppression List</h3>
              <p className="text-xs text-slate-400">
                Reversible rate-limiting for entities with 3+ hard block violations in 10 minutes
              </p>
            </div>
          </div>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-dark-800 border border-dark-700 text-slate-300">
            {suppressed_entities.length} active
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-dark-700/60 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                <th className="py-2.5 px-3">Entity Identifier</th>
                <th className="py-2.5 px-3">Reason</th>
                <th className="py-2.5 px-3">Violations</th>
                <th className="py-2.5 px-3">Remaining TTL</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-750/50 text-slate-300">
              {suppressed_entities.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6 text-center text-slate-500 italic">
                    ✓ No entities currently under temporary suppression.
                  </td>
                </tr>
              ) : (
                suppressed_entities.map((ent: SuppressedEntity) => (
                  <tr key={ent.entity_id} className="hover:bg-dark-800/50">
                    <td className="py-3 px-3 font-mono font-medium text-slate-200">
                      {ent.entity_id}
                    </td>
                    <td className="py-3 px-3 text-slate-400">{ent.reason}</td>
                    <td className="py-3 px-3 font-mono text-rose-400 font-bold">
                      {ent.violations}
                    </td>
                    <td className="py-3 px-3 font-mono text-brand-cyan">
                      {Math.round(ent.remaining_ttl_seconds)}s
                    </td>
                    <td className="py-3 px-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 uppercase">
                        {ent.status}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => handleUnblockEntity(ent.entity_id)}
                        className="px-2.5 py-1 rounded bg-dark-800 hover:bg-dark-700 text-slate-300 hover:text-white border border-dark-700 text-[11px] font-medium transition-all"
                      >
                        Unblock
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="p-3 bg-dark-900/60 rounded-lg border border-dark-700/50 text-[11px] text-slate-400 flex items-center gap-2">
          <Info className="w-3.5 h-3.5 text-brand-cyan shrink-0" />
          <span>
            Temporary suppression is completely reversible, automatically expires via TTL, and enforces HARD_BLOCK only on repetitive suspicious entities.
          </span>
        </div>
      </div>
    </div>
  );
};
