import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, Shield, TrendingUp, Users, Flame } from "lucide-react";
import { DefenseStatus, SeverityType } from "../../types/api";

interface DefensePanelProps {
  defense: DefenseStatus | null;
  onNavigateToDefense?: () => void;
}

export const DefensePanel: React.FC<DefensePanelProps> = ({
  defense,
  onNavigateToDefense,
}) => {
  if (!defense) {
    return (
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 animate-pulse h-full">
        <div className="h-4 bg-dark-750 rounded w-1/3 mb-4" />
        <div className="h-20 bg-dark-750 rounded mb-4" />
      </div>
    );
  }

  const { circuit_breaker, sliding_window_telemetry, active_incident, suppressed_entities_count } = defense;
  const isDefenseActive = circuit_breaker.state === "DEFENSE_ACTIVE";
  const severity: SeverityType = sliding_window_telemetry.spike_severity || "NORMAL";

  const getSeverityBadge = () => {
    switch (severity) {
      case "CRITICAL":
        return <span className="px-2 py-0.5 rounded text-[10px] font-extrabold bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse">🔴 CRITICAL SPIKE</span>;
      case "HIGH":
        return <span className="px-2 py-0.5 rounded text-[10px] font-extrabold bg-orange-500/20 text-orange-400 border border-orange-500/40">🟠 HIGH SEVERITY</span>;
      case "MEDIUM":
        return <span className="px-2 py-0.5 rounded text-[10px] font-extrabold bg-amber-500/20 text-amber-400 border border-amber-500/40">🟡 MEDIUM ALERT</span>;
      case "NORMAL":
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">🟢 NORMAL</span>;
    }
  };

  return (
    <div className={`rounded-xl p-5 border transition-all shadow-xl flex flex-col justify-between ${
      isDefenseActive
        ? "bg-gradient-to-b from-dark-850 to-rose-950/30 border-rose-500/50 shadow-glow-red/20"
        : "bg-dark-850 border-dark-700/80"
    }`}>
      <div>
        {/* Panel Header */}
        <div className="flex items-center justify-between pb-3 border-b border-dark-700/60">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isDefenseActive ? "bg-rose-500/20 text-rose-400" : "bg-brand-cyan/10 text-brand-cyan"
            }`}>
              {isDefenseActive ? <Flame className="w-4 h-4 text-rose-400" /> : <ShieldCheck className="w-4 h-4 text-brand-cyan" />}
            </div>
            <div>
              <h3 className="font-bold text-xs uppercase tracking-wider text-white">Real-Time Defense</h3>
              <p className="text-[11px] text-slate-400">Adaptive Circuit Breaker</p>
            </div>
          </div>
          {getSeverityBadge()}
        </div>

        {/* Hero State Banner */}
        <div className={`my-4 p-3.5 rounded-lg border text-xs ${
          isDefenseActive
            ? "bg-rose-500/10 border-rose-500/30 text-rose-200"
            : circuit_breaker.state === "COOLDOWN"
            ? "bg-amber-500/10 border-amber-500/30 text-amber-200"
            : "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
        }`}>
          <div className="flex items-center justify-between font-bold text-xs">
            <span className="flex items-center gap-1.5">
              {isDefenseActive ? (
                <>🔴 DEFENSE ACTIVE — SPIKE DETECTED</>
              ) : circuit_breaker.state === "COOLDOWN" ? (
                <>🟡 COOLDOWN RECOVERY</>
              ) : (
                <>🟢 SYSTEM NORMAL</>
              )}
            </span>
            <span className="font-mono text-[11px] opacity-80 uppercase">
              {circuit_breaker.state}
            </span>
          </div>

          <p className="mt-1 text-[11px] text-slate-400 leading-tight">
            {isDefenseActive
              ? "Routing thresholds tightened dynamically: elevated risk traffic challenged/blocked."
              : circuit_breaker.state === "COOLDOWN"
              ? "Traffic stabilizing; evaluating healthy cooldown streak for auto-recovery."
              : "Standard operational routing active with continuous statistical drift monitoring."}
          </p>
        </div>

        {/* Live Defense Key Metrics */}
        <div className="space-y-2.5 text-xs">
          <div className="flex items-center justify-between py-1 border-b border-dark-700/40">
            <span className="text-slate-400 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-slate-500" /> High-Risk Rate (5m)
            </span>
            <span className={`font-mono font-bold ${
              sliding_window_telemetry.high_risk_rate_pct > 15
                ? "text-rose-400"
                : sliding_window_telemetry.high_risk_rate_pct > 5
                ? "text-amber-400"
                : "text-slate-200"
            }`}>
              {sliding_window_telemetry.high_risk_rate_pct.toFixed(1)}%
            </span>
          </div>

          <div className="flex items-center justify-between py-1 border-b border-dark-700/40">
            <span className="text-slate-400">Score Drift Status</span>
            <span className={`font-semibold text-[11px] ${
              sliding_window_telemetry.score_drift_status === "SIGNIFICANT_DRIFT"
                ? "text-rose-400"
                : sliding_window_telemetry.score_drift_status === "MODERATE_DRIFT"
                ? "text-amber-400"
                : "text-emerald-400"
            }`}>
              {sliding_window_telemetry.score_drift_status}
            </span>
          </div>

          <div className="flex items-center justify-between py-1 border-b border-dark-700/40">
            <span className="text-slate-400">Active Thresholds</span>
            <span className="font-mono text-[11px] text-brand-cyan">
              p_low: {circuit_breaker.active_thresholds.p_low.toFixed(3)} | p_high: {circuit_breaker.active_thresholds.p_high.toFixed(3)}
            </span>
          </div>

          <div className="flex items-center justify-between py-1 border-b border-dark-700/40">
            <span className="text-slate-400">Active Incident</span>
            <span className="font-medium text-slate-200">
              {active_incident ? (
                <span className="text-rose-400 font-mono text-[11px]">#{active_incident.incident_id}</span>
              ) : (
                <span className="text-slate-500 text-[11px]">None</span>
              )}
            </span>
          </div>

          <div className="flex items-center justify-between py-1">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-slate-500" /> Suppressed Entities
            </span>
            <span className="font-mono font-bold text-slate-200">
              {suppressed_entities_count}
            </span>
          </div>
        </div>
      </div>

      {onNavigateToDefense && (
        <div className="pt-4 mt-3 border-t border-dark-700/60">
          <button
            onClick={onNavigateToDefense}
            className="w-full py-2 rounded-lg bg-dark-800 hover:bg-dark-750 text-brand-cyan hover:text-white border border-dark-700 text-xs font-semibold transition-all flex items-center justify-center gap-1.5"
          >
            <span>Open Defense Center</span>
            <span className="text-xs">&rarr;</span>
          </button>
        </div>
      )}
    </div>
  );
};
