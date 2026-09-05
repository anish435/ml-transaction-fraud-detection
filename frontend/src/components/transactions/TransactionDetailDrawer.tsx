import React from "react";
import { X, ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Clock, CreditCard, Mail, ArrowUpRight } from "lucide-react";
import { TransactionRecord } from "../../types/api";

interface TransactionDetailDrawerProps {
  transaction: TransactionRecord | null;
  onClose: () => void;
}

export const TransactionDetailDrawer: React.FC<TransactionDetailDrawerProps> = ({
  transaction,
  onClose,
}) => {
  if (!transaction) return null;

  const getDecisionBadge = (decision: string) => {
    switch (decision) {
      case "ALLOW":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <ShieldCheck className="w-3.5 h-3.5" /> ALLOW
          </span>
        );
      case "CHALLENGE":
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <AlertTriangle className="w-3.5 h-3.5" /> CHALLENGE (2FA)
          </span>
        );
      case "HARD_BLOCK":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-rose-500/15 text-rose-400 border border-rose-500/30">
            <ShieldAlert className="w-3.5 h-3.5" /> HARD_BLOCK
          </span>
        );
    }
  };

  const probPercent = (transaction.fraud_probability * 100).toFixed(1);
  const isHighRisk = transaction.fraud_probability >= 0.7495;
  const isMediumRisk = transaction.fraud_probability >= 0.0804 && !isHighRisk;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-dark-900 border-l border-dark-700/80 h-full flex flex-col shadow-2xl overflow-y-auto">
        {/* Drawer Header */}
        <div className="p-5 border-b border-dark-700/80 flex items-center justify-between bg-dark-950/70 sticky top-0 z-10">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase text-brand-cyan tracking-wider">
                Risk Analysis
              </span>
              <span className="text-xs text-slate-500 font-mono">#{transaction.id}</span>
            </div>
            <h3 className="text-base font-bold text-white mt-0.5">Transaction Inspection</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-400 hover:text-white border border-dark-700 transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Drawer Body */}
        <div className="p-6 space-y-6">
          {/* Amount & Decision Hero Card */}
          <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/70 space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400 font-medium">Transaction Amount</p>
                <div className="text-3xl font-extrabold text-white mt-1">
                  {transaction.currency === "INR" ? "₹" : "$"}
                  {transaction.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  <span className="text-xs font-medium text-slate-400 ml-1.5">{transaction.currency}</span>
                </div>
              </div>
              {getDecisionBadge(transaction.decision)}
            </div>

            {/* Fraud Probability Bar */}
            <div className="space-y-1.5 pt-2 border-t border-dark-700/50">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400 font-medium">Calibrated Fraud Risk</span>
                <span
                  className={`font-mono font-bold ${
                    isHighRisk
                      ? "text-rose-400"
                      : isMediumRisk
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }`}
                >
                  {probPercent}%
                </span>
              </div>
              <div className="w-full h-2.5 bg-dark-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isHighRisk
                      ? "bg-rose-500 shadow-glow-red"
                      : isMediumRisk
                      ? "bg-amber-500 shadow-glow-amber"
                      : "bg-emerald-500"
                  }`}
                  style={{ width: `${Math.max(Number(probPercent), 2)}%` }}
                />
              </div>
            </div>
          </div>

          {/* AI Signals / SHAP Reasons */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-brand-purple/20 text-brand-purple flex items-center justify-center">
                <Cpu className="w-3.5 h-3.5" />
              </div>
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                AI Signals & Operational Attributions
              </h4>
            </div>

            {transaction.reasons && transaction.reasons.length > 0 ? (
              <div className="space-y-2">
                {transaction.reasons.map((reason, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-dark-850/80 border border-dark-700/60 text-xs text-slate-200 flex items-start gap-2.5 font-medium"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-brand-purple mt-1.5 shrink-0" />
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-dark-850 border border-dark-700/50 text-xs text-slate-400 italic">
                No explanation signals returned.
              </div>
            )}
          </div>

          {/* Operational Routing & Defense Metadata */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Inference & Defense Telemetry
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-dark-850 p-3 rounded-lg border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block">Model Engine</span>
                <span className="text-xs font-semibold text-slate-200 mt-0.5 block">
                  XGBoost + LightGBM
                </span>
              </div>
              <div className="bg-dark-850 p-3 rounded-lg border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block">Inference Latency</span>
                <span className="text-xs font-mono font-semibold text-brand-cyan mt-0.5 block">
                  {transaction.latency_ms.toFixed(1)} ms
                </span>
              </div>
              <div className="bg-dark-850 p-3 rounded-lg border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block">Circuit Breaker</span>
                <span className="text-xs font-semibold text-slate-200 mt-0.5 block">
                  {transaction.circuit_breaker_state || "NORMAL"}
                </span>
              </div>
              <div className="bg-dark-850 p-3 rounded-lg border border-dark-700/60">
                <span className="text-[11px] text-slate-400 block">Defense Action</span>
                <span className="text-xs font-semibold text-slate-200 mt-0.5 block truncate" title={transaction.defense_action}>
                  {transaction.defense_action || "STANDARD_ROUTING"}
                </span>
              </div>
            </div>
          </div>

          {/* Entity & Transaction Details */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Identity & Payment Card
            </h4>
            <div className="space-y-2 bg-dark-850 p-4 rounded-lg border border-dark-700/60 text-xs">
              <div className="flex justify-between items-center py-1 border-b border-dark-700/40">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <CreditCard className="w-3.5 h-3.5 text-slate-500" /> Card Network / Type
                </span>
                <span className="text-slate-200 font-medium capitalize">
                  {transaction.card4 || "Visa"} ({transaction.card6 || "Credit"})
                </span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-dark-700/40">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Mail className="w-3.5 h-3.5 text-slate-500" /> Customer / Entity
                </span>
                <span className="text-slate-200 font-mono truncate max-w-[200px]" title={transaction.email}>
                  {transaction.email || "anonymous"}
                </span>
              </div>
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-slate-500" /> Timestamp
                </span>
                <span className="text-slate-300 font-mono text-[11px]">
                  {new Date(transaction.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-dark-700/80 bg-dark-950/70 mt-auto flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-300 hover:text-white border border-dark-700 text-xs font-semibold transition-all"
          >
            Close Analysis
          </button>
        </div>
      </div>
    </div>
  );
};
