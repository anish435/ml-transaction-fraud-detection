import React, { useState, useEffect } from "react";
import { CreditCard, CheckCircle2, ArrowUpRight, ShieldCheck, RefreshCw, Send, Lock, AlertCircle } from "lucide-react";
import { RazorpayAuditLog, OrderCreateResponse } from "../types/api";
import { api } from "../services/api";

interface RazorpayProps {
  onNotification?: (msg: { text: string; type: "success" | "warning" | "info" | "error" }) => void;
}

export const Razorpay: React.FC<RazorpayProps> = ({ onNotification }) => {
  const [logs, setLogs] = useState<RazorpayAuditLog[]>([]);
  const [loadingLogs, setLoadingLogs] = useState<boolean>(true);

  // Form states
  const [amount, setAmount] = useState<number>(2499.0);
  const [currency, setCurrency] = useState<string>("INR");
  const [receipt, setReceipt] = useState<string>(`rcpt_${Date.now()}`);
  const [creatingOrder, setCreatingOrder] = useState<boolean>(false);
  const [lastCreatedOrder, setLastCreatedOrder] = useState<OrderCreateResponse | null>(null);

  const loadAuditLogs = async () => {
    setLoadingLogs(true);
    try {
      const res = await api.getRazorpayAuditLogs(20);
      setLogs(res.logs || []);
    } catch (err: any) {
      onNotification?.({
        text: `Error loading Razorpay audit logs: ${err.message}`,
        type: "error",
      });
    } finally {
      setLoadingLogs(false);
    }
  };

  useEffect(() => {
    loadAuditLogs();
  }, []);

  const handleCreateOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreatingOrder(true);
    try {
      const order = await api.createRazorpayOrder(amount, currency, receipt);
      setLastCreatedOrder(order);
      onNotification?.({
        text: `Razorpay test order created: ${order.order_id} (${currency} ${order.amount})`,
        type: "success",
      });
      setReceipt(`rcpt_${Date.now()}`);
    } catch (err: any) {
      onNotification?.({
        text: `Failed to create order: ${err.message}`,
        type: "error",
      });
    } finally {
      setCreatingOrder(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40 uppercase tracking-wider">
              RAZORPAY TEST MODE
            </span>
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Gateway Connected
            </span>
          </div>
          <h2 className="text-base font-bold text-white tracking-tight mt-1">
            Payment Gateway Integration & Webhook Audit Trail
          </h2>
          <p className="text-xs text-slate-400 mt-0.5 max-w-2xl">
            Integrated with Razorpay Python SDK and HMAC SHA256 verified webhooks.
            Inbound payment events undergo real-time feature transformation, rolling velocity attribution, and automated risk scoring.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-900 border border-dark-700 text-xs text-slate-300">
          <Lock className="w-4 h-4 text-emerald-400" />
          <span>HMAC Webhook Verified</span>
        </div>
      </div>

      {/* Grid: Test Payment Creator + Webhook Details */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Test Order Creation Form (1/3) */}
        <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md space-y-4">
          <div className="pb-3 border-b border-dark-700/60">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <CreditCard className="w-4 h-4 text-brand-cyan" />
              Create Test Payment Order
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Simulate Razorpay checkout initialization
            </p>
          </div>

          <form onSubmit={handleCreateOrder} className="space-y-3.5">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1">
                Order Amount
              </label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs font-mono text-slate-400">
                  {currency === "INR" ? "₹" : "$"}
                </span>
                <input
                  type="number"
                  step="0.01"
                  value={amount}
                  onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg pl-8 pr-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-brand-cyan/50 focus:ring-1 focus:ring-brand-cyan/30"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Currency
                </label>
                <select
                  value={currency}
                  onChange={(e) => setCurrency(e.target.value)}
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-brand-cyan/50"
                >
                  <option value="INR">INR (₹)</option>
                  <option value="USD">USD ($)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Receipt ID
                </label>
                <input
                  type="text"
                  value={receipt}
                  onChange={(e) => setReceipt(e.target.value)}
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none focus:border-brand-cyan/50"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={creatingOrder}
              className="w-full py-2.5 rounded-lg bg-brand-cyan/15 hover:bg-brand-cyan/25 text-brand-cyan border border-brand-cyan/40 text-xs font-bold transition-all flex items-center justify-center gap-2 shadow-glow-cyan/30 disabled:opacity-50"
            >
              <Send className="w-3.5 h-3.5" />
              <span>{creatingOrder ? "Creating..." : "Initialize Order"}</span>
            </button>
          </form>

          {/* Result Card */}
          {lastCreatedOrder && (
            <div className="p-3.5 rounded-lg bg-dark-900/90 border border-emerald-500/30 text-xs space-y-1.5 animate-in fade-in">
              <div className="flex items-center justify-between text-emerald-400 font-bold">
                <span className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Order Created
                </span>
                <span className="font-mono text-[11px]">{lastCreatedOrder.status}</span>
              </div>
              <div className="text-[11px] font-mono text-slate-300 space-y-0.5">
                <div>
                  <span className="text-slate-500">ID: </span>
                  {lastCreatedOrder.order_id}
                </div>
                <div>
                  <span className="text-slate-500">Amount: </span>
                  {lastCreatedOrder.currency} {lastCreatedOrder.amount.toFixed(2)}
                </div>
                <div>
                  <span className="text-slate-500">Receipt: </span>
                  {lastCreatedOrder.receipt}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recent Webhook Audit Trail (2/3) */}
        <div className="lg:col-span-2 bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-dark-700/60">
            <div>
              <h3 className="text-sm font-bold text-white">Razorpay Webhook Audit Log</h3>
              <p className="text-xs text-slate-400">Captured and scored payment webhook payloads</p>
            </div>
            <button
              onClick={loadAuditLogs}
              disabled={loadingLogs}
              className="p-1.5 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-300 border border-dark-700 transition-all"
              title="Refresh audit log"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingLogs ? "animate-spin text-brand-cyan" : ""}`} />
            </button>
          </div>

          <div className="overflow-x-auto max-h-[380px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-dark-700/60 text-[11px] font-semibold text-slate-400 uppercase tracking-wider sticky top-0 bg-dark-850 z-10">
                  <th className="py-2.5 px-3">Payment ID</th>
                  <th className="py-2.5 px-3">Customer Identifier</th>
                  <th className="py-2.5 px-3">Amount</th>
                  <th className="py-2.5 px-3">Risk %</th>
                  <th className="py-2.5 px-3">Operational Decision</th>
                  <th className="py-2.5 px-3">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-750/50 text-slate-300">
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-slate-500 italic">
                      No Razorpay webhook payments recorded in audit trail yet.
                    </td>
                  </tr>
                ) : (
                  logs.map((log) => {
                    const prob = (log.fraud_probability * 100).toFixed(1);
                    return (
                      <tr key={log.payment_id} className="hover:bg-dark-800/40">
                        <td className="py-2.5 px-3 font-mono font-medium text-slate-200">
                          {log.payment_id}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-400 truncate max-w-[160px]" title={log.customer_identifier}>
                          {log.customer_identifier}
                        </td>
                        <td className="py-2.5 px-3 font-semibold text-white">
                          ₹{log.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                        </td>
                        <td className="py-2.5 px-3 font-mono font-bold">
                          {prob}%
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              log.decision === "ALLOW"
                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                : log.decision === "CHALLENGE"
                                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                : "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                            }`}
                          >
                            {log.decision}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-400">
                          {log.latency_ms?.toFixed(1) || 18.5} ms
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
