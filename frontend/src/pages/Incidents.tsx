import React, { useState, useEffect } from "react";
import { AlertTriangle, ShieldCheck, Clock, CheckCircle2, Flame, RefreshCw } from "lucide-react";
import { IncidentRecord } from "../types/api";
import { api } from "../services/api";

interface IncidentsProps {
  onNotification?: (msg: { text: string; type: "success" | "warning" | "info" | "error" }) => void;
}

export const Incidents: React.FC<IncidentsProps> = ({ onNotification }) => {
  const [incidents, setIncidents] = useState<IncidentRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterStatus, setFilterStatus] = useState<string>("ALL");

  const loadIncidents = async () => {
    setLoading(true);
    try {
      const res = await api.getIncidents();
      setIncidents(res.incidents || []);
    } catch (err: any) {
      onNotification?.({
        text: `Failed to load incidents: ${err.message}`,
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadIncidents();
  }, []);

  const handleResolve = async (incidentId: string) => {
    try {
      await api.resolveIncident(incidentId, "Resolved from Web Console audit");
      onNotification?.({
        text: `Incident #${incidentId} marked as RESOLVED.`,
        type: "success",
      });
      loadIncidents();
    } catch (err: any) {
      onNotification?.({
        text: `Error resolving incident: ${err.message}`,
        type: "error",
      });
    }
  };

  const filtered = incidents.filter(
    (inc) => filterStatus === "ALL" || inc.status === filterStatus
  );

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "MITIGATING":
      case "ACTIVE":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-rose-500/15 text-rose-400 border border-rose-500/40 animate-pulse">
            <Flame className="w-3.5 h-3.5" /> {status}
          </span>
        );
      case "RESOLVED":
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3.5 h-3.5" /> RESOLVED
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            Gateway Incident Management
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-dark-750 text-slate-300 border border-dark-700">
              {incidents.length} total
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Audit log of systemic fraud spikes, gateway surges, and automated circuit breaker mitigation lifecycles.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Status Filter */}
          <div className="flex items-center bg-dark-800 p-0.5 rounded-lg border border-dark-700 text-xs">
            {["ALL", "MITIGATING", "RESOLVED"].map((st) => (
              <button
                key={st}
                onClick={() => setFilterStatus(st)}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  filterStatus === st
                    ? "bg-dark-700 text-brand-cyan shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {st}
              </button>
            ))}
          </div>

          <button
            onClick={loadIncidents}
            disabled={loading}
            className="p-2 rounded-lg bg-dark-800 hover:bg-dark-750 text-slate-300 hover:text-white border border-dark-700 transition-all"
            title="Refresh incidents"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin text-brand-cyan" : ""}`} />
          </button>
        </div>
      </div>

      {/* Incidents Feed */}
      {filtered.length === 0 ? (
        <div className="bg-dark-850 rounded-xl p-12 border border-dark-700/80 text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mx-auto">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-bold text-white">No active incidents recorded</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            All gateway traffic is routing under baseline parameters without active mitigation triggers.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {filtered.map((inc) => (
            <div
              key={inc.incident_id}
              className={`rounded-xl p-5 border transition-all ${
                inc.status === "MITIGATING"
                  ? "bg-gradient-to-r from-dark-850 via-dark-850 to-rose-950/20 border-rose-500/50 shadow-glow-red/20"
                  : "bg-dark-850 border-dark-700/80"
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-dark-700/60">
                <div className="flex items-center gap-3">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                    inc.status === "MITIGATING"
                      ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                      : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                  }`}>
                    {inc.status === "MITIGATING" ? <Flame className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-mono font-bold text-sm text-white">
                        #{inc.incident_id}
                      </h3>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-dark-750 text-slate-300 border border-dark-600">
                        {inc.severity} SEVERITY
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Triggered at: {new Date(inc.started_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {getStatusBadge(inc.status)}
                  {inc.status === "MITIGATING" && (
                    <button
                      onClick={() => handleResolve(inc.incident_id)}
                      className="px-3 py-1 rounded-lg bg-dark-800 hover:bg-dark-750 text-emerald-400 hover:text-emerald-300 border border-emerald-500/30 text-xs font-semibold transition-all"
                    >
                      Resolve Incident
                    </button>
                  )}
                </div>
              </div>

              {/* Metrics Breakdown */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-3 text-xs border-b border-dark-700/40">
                <div className="bg-dark-800/80 p-2.5 rounded-lg border border-dark-700/50">
                  <span className="text-[10px] text-slate-400 block font-medium">Affected Transactions</span>
                  <span className="text-sm font-mono font-bold text-slate-200 mt-0.5 block">
                    {inc.affected_transactions_count}
                  </span>
                </div>
                <div className="bg-dark-800/80 p-2.5 rounded-lg border border-dark-700/50">
                  <span className="text-[10px] text-slate-400 block font-medium">Trigger Fraud Rate</span>
                  <span className="text-sm font-mono font-bold text-rose-400 mt-0.5 block">
                    {inc.trigger_metrics?.high_risk_rate_pct ?? "N/A"}%
                  </span>
                </div>
                <div className="bg-dark-800/80 p-2.5 rounded-lg border border-dark-700/50">
                  <span className="text-[10px] text-slate-400 block font-medium">Burst Velocity (60s)</span>
                  <span className="text-sm font-mono font-bold text-amber-400 mt-0.5 block">
                    {inc.trigger_metrics?.burst_velocity_60s ?? 0}
                  </span>
                </div>
                <div className="bg-dark-800/80 p-2.5 rounded-lg border border-dark-700/50">
                  <span className="text-[10px] text-slate-400 block font-medium">Duration</span>
                  <span className="text-sm font-mono font-bold text-brand-cyan mt-0.5 block">
                    {inc.duration_seconds ? `${Math.round(inc.duration_seconds)}s` : "Active"}
                  </span>
                </div>
              </div>

              {/* Resolution Reason */}
              {inc.resolution_reason && (
                <div className="pt-2 text-xs text-slate-400">
                  <span className="font-semibold text-slate-300">Resolution Note: </span>
                  {inc.resolution_reason}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
