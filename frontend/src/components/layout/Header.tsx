import React from "react";
import { RefreshCw, Shield, AlertTriangle, ShieldAlert } from "lucide-react";
import { CircuitBreakerMode } from "../../types/api";

interface HeaderProps {
  title: string;
  subtitle?: string;
  circuitBreakerState: CircuitBreakerMode;
  isSpikeActive: boolean;
  lastUpdatedSecondsAgo: number;
  onRefresh: () => void;
  loading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  title,
  subtitle,
  circuitBreakerState,
  isSpikeActive,
  lastUpdatedSecondsAgo,
  onRefresh,
  loading,
}) => {
  const getDefenseBadge = () => {
    if (circuitBreakerState === "DEFENSE_ACTIVE") {
      return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-risk-block/15 border border-risk-block/40 text-risk-block text-xs font-semibold shadow-glow-red animate-pulse">
          <ShieldAlert className="w-4 h-4 text-risk-block" />
          <span>🔴 DEFENSE ACTIVE — SPIKE DETECTED</span>
        </div>
      );
    }
    if (circuitBreakerState === "COOLDOWN") {
      return (
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-risk-challenge/15 border border-risk-challenge/40 text-risk-challenge text-xs font-semibold shadow-glow-amber">
          <AlertTriangle className="w-4 h-4 text-risk-challenge" />
          <span>🟡 COOLDOWN RECOVERY</span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
        <Shield className="w-4 h-4 text-emerald-400" />
        <span>🟢 GATEWAY OPERATIONAL</span>
      </div>
    );
  };

  return (
    <header className="h-16 px-6 border-b border-dark-700/60 bg-dark-950/80 backdrop-blur-md flex items-center justify-between sticky top-0 z-30">
      <div>
        <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
          {title}
        </h2>
        {subtitle && <p className="text-xs text-slate-400 font-medium">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-4">
        {/* Defense State Pill */}
        {getDefenseBadge()}

        {/* Model Tag */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-dark-800 border border-dark-700 text-xs text-slate-300 font-mono">
          <span className="w-1.5 h-1.5 rounded-full bg-brand-purple" />
          <span>XGBoost + LightGBM</span>
        </div>

        {/* Refresh & Timer */}
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="hidden sm:inline">
            {lastUpdatedSecondsAgo === 0
              ? "Just now"
              : `${lastUpdatedSecondsAgo}s ago`}
          </span>
          <button
            id="header-refresh-btn"
            onClick={onRefresh}
            disabled={loading}
            className="p-2 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-300 hover:text-white border border-dark-700/80 transition-all disabled:opacity-50"
            title="Refresh telemetry"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-brand-cyan" : ""}`} />
          </button>
        </div>
      </div>
    </header>
  );
};
