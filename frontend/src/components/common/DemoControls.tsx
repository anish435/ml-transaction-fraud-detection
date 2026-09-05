import React, { useState } from "react";
import { Play, Flame, CheckCircle, RotateCcw, Loader2 } from "lucide-react";
import { api } from "../../services/api";

interface DemoControlsProps {
  onActionComplete: () => void;
  onNotification?: (msg: { text: string; type: "success" | "warning" | "info" | "error" }) => void;
}

export const DemoControls: React.FC<DemoControlsProps> = ({
  onActionComplete,
  onNotification,
}) => {
  const [activeAction, setActiveAction] = useState<string | null>(null);

  const handleSimulateNormal = async () => {
    setActiveAction("normal");
    try {
      const res = await api.simulateNormal();
      onNotification?.({
        text: `Normal Payment: Scored ${res.transaction.decision} ($${res.transaction.amount.toFixed(2)}, Risk: ${(res.transaction.fraud_probability * 100).toFixed(1)}%)`,
        type: "success",
      });
      onActionComplete();
    } catch (err: any) {
      onNotification?.({
        text: `Error simulating normal payment: ${err.message}`,
        type: "error",
      });
    } finally {
      setActiveAction(null);
    }
  };

  const handleSimulateSpike = async () => {
    setActiveAction("spike");
    try {
      const res = await api.simulateSpike();
      onNotification?.({
        text: `🚨 High-Risk Surge! ${res.transactions_scored} suspicious transactions detected. Circuit Breaker engaged: ${res.defense_status.circuit_breaker.state}`,
        type: "warning",
      });
      onActionComplete();
    } catch (err: any) {
      onNotification?.({
        text: `Error simulating spike: ${err.message}`,
        type: "error",
      });
    } finally {
      setActiveAction(null);
    }
  };

  const handleSimulateRecovery = async () => {
    setActiveAction("recovery");
    try {
      const res = await api.simulateRecovery();
      onNotification?.({
        text: `Traffic normalized. Circuit Breaker restored to ${res.circuit_breaker_state}.`,
        type: "info",
      });
      onActionComplete();
    } catch (err: any) {
      onNotification?.({
        text: `Error simulating recovery: ${err.message}`,
        type: "error",
      });
    } finally {
      setActiveAction(null);
    }
  };

  const handleReset = async () => {
    setActiveAction("reset");
    try {
      await api.resetDemo();
      onNotification?.({
        text: `Demo state reset to clean baseline (Circuit Breaker: NORMAL).`,
        type: "info",
      });
      onActionComplete();
    } catch (err: any) {
      onNotification?.({
        text: `Error resetting demo: ${err.message}`,
        type: "error",
      });
    } finally {
      setActiveAction(null);
    }
  };

  return (
    <div className="bg-dark-850/90 border border-dark-700/80 rounded-xl p-3 sm:p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-lg">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-brand-cyan/10 border border-brand-cyan/30 flex items-center justify-center text-brand-cyan">
          <Play className="w-4 h-4 fill-brand-cyan/20" />
        </div>
        <div>
          <h4 className="text-xs font-bold text-white tracking-wide uppercase flex items-center gap-2">
            Interactive Presentation Controls
            <span className="text-[10px] font-medium text-slate-400 lowercase">(real pipeline execution)</span>
          </h4>
          <p className="text-[11px] text-slate-400">
            Drive live transaction scoring, fraud surge detection, circuit breaker defense, and auto-recovery.
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* Simulate Normal */}
        <button
          id="btn-demo-normal"
          onClick={handleSimulateNormal}
          disabled={activeAction !== null}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition-all disabled:opacity-50"
        >
          {activeAction === "normal" ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
          )}
          <span>Normal Payment</span>
        </button>

        {/* Simulate Risk Spike */}
        <button
          id="btn-demo-spike"
          onClick={handleSimulateSpike}
          disabled={activeAction !== null}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-risk-block/15 hover:bg-risk-block/25 text-rose-300 border border-risk-block/40 text-xs font-semibold shadow-glow-red/50 transition-all disabled:opacity-50"
        >
          {activeAction === "spike" ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Flame className="w-3.5 h-3.5 text-rose-400" />
          )}
          <span>Simulate Risk Spike</span>
        </button>

        {/* Simulate Recovery */}
        <button
          id="btn-demo-recovery"
          onClick={handleSimulateRecovery}
          disabled={activeAction !== null}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-cyan/10 hover:bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/30 text-xs font-semibold transition-all disabled:opacity-50"
        >
          {activeAction === "recovery" ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <CheckCircle className="w-3.5 h-3.5 text-brand-cyan" />
          )}
          <span>Simulate Recovery</span>
        </button>

        {/* Reset */}
        <button
          id="btn-demo-reset"
          onClick={handleReset}
          disabled={activeAction !== null}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-400 hover:text-slate-200 border border-dark-700 text-xs font-medium transition-all disabled:opacity-50"
          title="Reset to clean baseline"
        >
          {activeAction === "reset" ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <RotateCcw className="w-3.5 h-3.5" />
          )}
        </button>
      </div>
    </div>
  );
};
