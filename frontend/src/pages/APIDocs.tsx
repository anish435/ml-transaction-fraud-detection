import React from "react";
import { Code2, ExternalLink, CheckCircle2, Copy } from "lucide-react";

export const APIDocs: React.FC = () => {
  const swaggerUrl = "http://127.0.0.1:8000/docs";

  const endpoints = [
    {
      method: "POST",
      path: "/score",
      tag: "INFERENCE",
      desc: "Full transaction scoring with calibrated probability, 3-tier action routing, and top SHAP attribution alerts.",
    },
    {
      method: "POST",
      path: "/score-fast",
      tag: "INFERENCE",
      desc: "Ultra-low latency inference endpoint (~3-5ms). Skips SHAP attribution for high-throughput pipelines.",
    },
    {
      method: "GET",
      path: "/score-fast",
      tag: "INFERENCE",
      desc: "Lightweight query parameter endpoint for rapid probing or load-testing.",
    },
    {
      method: "GET",
      path: "/health",
      tag: "SYSTEM",
      desc: "Service readiness check, model loading verification, and circuit breaker state.",
    },
    {
      method: "GET",
      path: "/stats",
      tag: "TELEMETRY",
      desc: "Comparative latency distribution (p50, p95, p99) between fast-mode and explainable-mode.",
    },
    {
      method: "GET",
      path: "/metrics",
      tag: "ANALYTICS",
      desc: "Returns complete sealed-test evaluation summary, confusion matrix, and model comparison metrics.",
    },
    {
      method: "GET",
      path: "/transactions/recent",
      tag: "FEED",
      desc: "Retrieves circular in-memory buffer of live scored transactions.",
    },
    {
      method: "POST",
      path: "/webhook/razorpay",
      tag: "RAZORPAY",
      desc: "Production webhook listener with HMAC SHA256 signature verification and automatic transaction audit logging.",
    },
    {
      method: "POST",
      path: "/create-order",
      tag: "RAZORPAY",
      desc: "Creates a Razorpay payment order in Test Mode using the Razorpay Python SDK.",
    },
    {
      method: "GET",
      path: "/razorpay/audit-logs",
      tag: "RAZORPAY",
      desc: "Retrieves recent captured payment events from the append-only JSONL audit log.",
    },
    {
      method: "GET",
      path: "/defense/status",
      tag: "DEFENSE",
      desc: "Returns comprehensive circuit breaker state, 5m sliding window volume, high-risk rate, and drift telemetry.",
    },
    {
      method: "GET",
      path: "/defense/incidents",
      tag: "DEFENSE",
      desc: "Lists all active and resolved fraud spike incidents.",
    },
    {
      method: "POST",
      path: "/defense/incidents/{incident_id}/resolve",
      tag: "DEFENSE",
      desc: "Manually resolves an active fraud incident with an analyst note.",
    },
    {
      method: "POST",
      path: "/defense/circuit-breaker/trip",
      tag: "DEFENSE",
      desc: "Emergency manual trip of the circuit breaker into DEFENSE_ACTIVE mode.",
    },
    {
      method: "POST",
      path: "/defense/circuit-breaker/reset",
      tag: "DEFENSE",
      desc: "Resets circuit breaker back to NORMAL mode and standard routing thresholds.",
    },
    {
      method: "GET",
      path: "/defense/suppression-list",
      tag: "DEFENSE",
      desc: "Lists all entities currently under temporary suppression with remaining TTLs.",
    },
    {
      method: "POST",
      path: "/demo/simulate-normal",
      tag: "DEMO",
      desc: "Scores an authentic low-risk transaction payload and records telemetry.",
    },
    {
      method: "POST",
      path: "/demo/simulate-spike",
      tag: "DEMO",
      desc: "Triggers a high-risk transaction burst, tripping the circuit breaker and creating an incident.",
    },
    {
      method: "POST",
      path: "/demo/simulate-recovery",
      tag: "DEMO",
      desc: "Feeds clean traffic to transition circuit breaker back to NORMAL mode.",
    },
    {
      method: "POST",
      path: "/demo/reset",
      tag: "DEMO",
      desc: "Resets circuit breaker and demo state to baseline clean conditions.",
    },
  ];

  const getMethodBadge = (method: string) => {
    switch (method) {
      case "POST":
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-blue-500/20 text-blue-400 border border-blue-500/30">POST</span>;
      case "GET":
      default:
        return <span className="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">GET</span>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              API STATUS: ONLINE
            </span>
            <span className="text-xs text-slate-500 font-mono">FastAPI Microservice (Port 8000)</span>
          </div>
          <h2 className="text-base font-bold text-white tracking-tight mt-1">
            Developer API & Integration Reference
          </h2>
          <p className="text-xs text-slate-400 mt-0.5 max-w-2xl">
            Integrate RiskGuard AI directly into your checkout flow or payment gateway microservices.
            Open interactive Swagger UI to test payloads directly in browser.
          </p>
        </div>

        <a
          href={swaggerUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-brand-cyan/15 hover:bg-brand-cyan/25 text-brand-cyan border border-brand-cyan/40 text-xs font-bold transition-all shadow-glow-cyan/30"
        >
          <span>Open API Documentation (Swagger)</span>
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>

      {/* Endpoints Table */}
      <div className="bg-dark-850 rounded-xl border border-dark-700/80 shadow-md overflow-hidden">
        <div className="p-4 border-b border-dark-700/60 bg-dark-900/60 flex justify-between items-center">
          <h3 className="text-sm font-bold text-white">Exposed Microservice Endpoints</h3>
          <span className="text-xs text-slate-400 font-mono">{endpoints.length} active routes</span>
        </div>

        <div className="divide-y divide-dark-750/50">
          {endpoints.map((ep, idx) => (
            <div key={idx} className="p-4 hover:bg-dark-800/40 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-start sm:items-center gap-3">
                <div className="w-16 shrink-0">{getMethodBadge(ep.method)}</div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-white text-xs">{ep.path}</span>
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-semibold bg-dark-750 text-slate-400 border border-dark-600">
                      {ep.tag}
                    </span>
                  </div>
                  <p className="text-slate-400 text-xs mt-0.5">{ep.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
