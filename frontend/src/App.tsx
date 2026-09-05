import React, { useState, useEffect, useCallback } from "react";
import { Sidebar } from "./components/layout/Sidebar";
import { Header } from "./components/layout/Header";
import { Overview } from "./pages/Overview";
import { Transactions } from "./pages/Transactions";
import { DefenseCenter } from "./pages/DefenseCenter";
import { Incidents } from "./pages/Incidents";
import { ModelPerformance } from "./pages/ModelPerformance";
import { Razorpay } from "./pages/Razorpay";
import { APIDocs } from "./pages/APIDocs";
import { api } from "./services/api";
import { usePolling } from "./hooks/usePolling";
import {
  HealthResponse,
  StatsResponse,
  DefenseStatus,
  ModelMetricsSummary,
  TransactionRecord,
} from "./types/api";
import { AlertCircle, CheckCircle2, AlertTriangle, Info, X } from "lucide-react";

interface Notification {
  id: number;
  text: string;
  type: "success" | "warning" | "info" | "error";
}

export function App() {
  const [currentTab, setCurrentTab] = useState<string>("overview");
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [metrics, setMetrics] = useState<ModelMetricsSummary | null>(null);

  // Poll Health
  const { data: healthData, error: healthError } = usePolling<HealthResponse>(
    useCallback(() => api.getHealth(), []),
    5000
  );

  // Poll Defense Status
  const {
    data: defenseData,
    loading: defenseLoading,
    secondsAgo,
    refresh: refreshDefense,
  } = usePolling<DefenseStatus>(
    useCallback(() => api.getDefenseStatus(), []),
    2500
  );

  // Poll Latency Stats
  const { data: statsData } = usePolling<StatsResponse>(
    useCallback(() => api.getStats(), []),
    5000
  );

  // Poll Live Transactions
  const {
    data: txResponse,
    refresh: refreshTransactions,
  } = usePolling<{ total: number; transactions: TransactionRecord[] }>(
    useCallback(() => api.getRecentTransactions(50), []),
    3000
  );

  // Load Model Metrics once on startup
  useEffect(() => {
    api
      .getMetrics()
      .then((m) => setMetrics(m))
      .catch((err) => console.warn("Could not load initial metrics", err));
  }, []);

  const addNotification = (msg: {
    text: string;
    type: "success" | "warning" | "info" | "error";
  }) => {
    const id = Date.now();
    setNotifications((prev) => [...prev, { id, ...msg }]);
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 5000);
  };

  const removeNotification = (id: number) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const handleGlobalRefresh = () => {
    refreshDefense();
    refreshTransactions();
  };

  const isBackendOnline = !healthError && !!healthData;
  const transactionsList = txResponse?.transactions || [];
  const circuitBreakerState = defenseData?.circuit_breaker?.state || "NORMAL";
  const isSpikeActive = defenseData?.sliding_window_telemetry?.is_spike || false;

  const getPageTitle = () => {
    switch (currentTab) {
      case "transactions":
        return { title: "Live Transactions", subtitle: "Real-time scored stream & SHAP risk signals" };
      case "defense":
        return { title: "Defense Center", subtitle: "Automated circuit breaker & spike mitigation" };
      case "incidents":
        return { title: "Incident Log", subtitle: "Gateway systemic fraud surge lifecycle audit" };
      case "models":
        return { title: "Model Performance", subtitle: "Sealed-test ML benchmark & business routing metrics" };
      case "razorpay":
        return { title: "Razorpay Test Mode", subtitle: "Payment gateway integration & webhook audit" };
      case "api":
        return { title: "API Reference", subtitle: "Exposed endpoints & Swagger documentation" };
      case "overview":
      default:
        return { title: "Risk Operations Console", subtitle: "Payment gateway fraud scoring & adaptive defense" };
    }
  };

  const pageInfo = getPageTitle();

  return (
    <div className="flex h-screen bg-dark-900 text-slate-100 overflow-hidden font-sans antialiased">
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={(tab) => setCurrentTab(tab)}
        isBackendOnline={isBackendOnline}
        modelVersion={healthData?.model_version || "v1.3.0"}
        incidentCount={defenseData?.active_incident ? 1 : 0}
      />

      {/* Main App Canvas */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Top Header */}
        <Header
          title={pageInfo.title}
          subtitle={pageInfo.subtitle}
          circuitBreakerState={circuitBreakerState}
          isSpikeActive={isSpikeActive}
          lastUpdatedSecondsAgo={secondsAgo}
          onRefresh={handleGlobalRefresh}
          loading={defenseLoading}
        />

        {/* Offline Warning Banner */}
        {!isBackendOnline && (
          <div className="bg-rose-500/20 border-b border-rose-500/40 px-6 py-2.5 flex items-center justify-between text-xs text-rose-300">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400" />
              <span>
                <b>⚠ BACKEND UNAVAILABLE</b> — Connecting to <code>http://127.0.0.1:8000</code>... Retrying automatically.
              </span>
            </div>
            <button
              onClick={handleGlobalRefresh}
              className="px-2.5 py-1 rounded bg-rose-500/30 hover:bg-rose-500/40 text-white font-semibold transition-all"
            >
              Retry Now
            </button>
          </div>
        )}

        {/* Scrollable Page Content */}
        <main className="flex-1 overflow-y-auto p-6 space-y-6">
          {currentTab === "overview" && (
            <Overview
              defense={defenseData}
              metrics={metrics}
              stats={statsData}
              transactions={transactionsList}
              onRefresh={handleGlobalRefresh}
              onNavigateToDefense={() => setCurrentTab("defense")}
              onNotification={addNotification}
            />
          )}

          {currentTab === "transactions" && (
            <Transactions
              transactions={transactionsList}
              onRefresh={refreshTransactions}
            />
          )}

          {currentTab === "defense" && (
            <DefenseCenter
              defense={defenseData}
              onRefresh={handleGlobalRefresh}
              onNotification={addNotification}
            />
          )}

          {currentTab === "incidents" && (
            <Incidents onNotification={addNotification} />
          )}

          {currentTab === "models" && (
            <ModelPerformance metrics={metrics} />
          )}

          {currentTab === "razorpay" && (
            <Razorpay onNotification={addNotification} />
          )}

          {currentTab === "api" && <APIDocs />}
        </main>
      </div>

      {/* Floating Notification Toasts */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
        {notifications.map((n) => {
          const isError = n.type === "error";
          const isWarning = n.type === "warning";
          const isSuccess = n.type === "success";

          return (
            <div
              key={n.id}
              className={`p-3.5 rounded-xl border shadow-xl flex items-start gap-2.5 text-xs pointer-events-auto backdrop-blur-md transition-all animate-in slide-in-from-bottom-3 ${
                isError
                  ? "bg-rose-950/90 border-rose-500/50 text-rose-200"
                  : isWarning
                  ? "bg-amber-950/90 border-amber-500/50 text-amber-200"
                  : isSuccess
                  ? "bg-emerald-950/90 border-emerald-500/50 text-emerald-200"
                  : "bg-dark-850/90 border-dark-700 text-slate-200"
              }`}
            >
              {isError && <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />}
              {isWarning && <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />}
              {isSuccess && <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />}
              {!isError && !isWarning && !isSuccess && <Info className="w-4 h-4 text-brand-cyan shrink-0 mt-0.5" />}

              <div className="flex-1 font-medium leading-tight">{n.text}</div>

              <button
                onClick={() => removeNotification(n.id)}
                className="p-1 rounded hover:bg-white/10 opacity-70 hover:opacity-100 transition-opacity"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default App;
