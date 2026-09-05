import React, { useState, useEffect } from "react";
import { KPICards } from "../components/dashboard/KPICards";
import { RiskActivityChart, TelemetryPoint } from "../components/charts/RiskActivityChart";
import { RiskRoutingDonut } from "../components/charts/RiskRoutingDonut";
import { DefensePanel } from "../components/defense/DefensePanel";
import { TransactionTable } from "../components/transactions/TransactionTable";
import { TransactionDetailDrawer } from "../components/transactions/TransactionDetailDrawer";
import { DemoControls } from "../components/common/DemoControls";
import { DefenseStatus, ModelMetricsSummary, StatsResponse, TransactionRecord } from "../types/api";

interface OverviewProps {
  defense: DefenseStatus | null;
  metrics: ModelMetricsSummary | null;
  stats: StatsResponse | null;
  transactions: TransactionRecord[];
  onRefresh: () => void;
  onNavigateToDefense: () => void;
  onNotification?: (msg: { text: string; type: "success" | "warning" | "info" | "error" }) => void;
}

export const Overview: React.FC<OverviewProps> = ({
  defense,
  metrics,
  stats,
  transactions,
  onRefresh,
  onNavigateToDefense,
  onNotification,
}) => {
  const [selectedTx, setSelectedTx] = useState<TransactionRecord | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryPoint[]>([]);

  // Collect rolling telemetry points for the chart
  useEffect(() => {
    if (!defense) return;
    const now = new Date();
    const timeLabel = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const newPoint: TelemetryPoint = {
      time: timeLabel,
      meanRisk: (defense.sliding_window_telemetry.mean_risk_prob || 0.0) * 100,
      highRiskRate: defense.sliding_window_telemetry.high_risk_rate_pct || 0.0,
      volume: defense.sliding_window_telemetry.tx_count || 0,
    };

    setTelemetryHistory((prev) => {
      const updated = [...prev, newPoint];
      return updated.slice(-15); // keep last 15 points
    });
  }, [defense]);

  // Compute live action counts
  const allowCount = transactions.filter((t) => t.decision === "ALLOW").length;
  const challengeCount = transactions.filter((t) => t.decision === "CHALLENGE").length;
  const blockCount = transactions.filter((t) => t.decision === "HARD_BLOCK").length;

  return (
    <div className="space-y-6">
      {/* Demo Controls Area */}
      <DemoControls onActionComplete={onRefresh} onNotification={onNotification} />

      {/* Top 5 KPI Cards */}
      <KPICards
        defense={defense}
        metrics={metrics}
        stats={stats}
        liveTxCount={transactions.length}
        liveBlockCount={blockCount}
      />

      {/* Main Monitoring Row: Risk Activity Chart (2/3) + Real-Time Defense Panel (1/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <RiskActivityChart
            data={
              telemetryHistory.length > 0
                ? telemetryHistory
                : [
                    { time: "12:00:00", meanRisk: 3.5, highRiskRate: 0.0, volume: 1 },
                    { time: "12:00:10", meanRisk: 4.1, highRiskRate: 0.0, volume: 2 },
                  ]
            }
          />
        </div>
        <div className="lg:col-span-1">
          <DefensePanel defense={defense} onNavigateToDefense={onNavigateToDefense} />
        </div>
      </div>

      {/* Transactions & Routing Breakdown Row: Live Feed (2/3) + Donut (1/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2">
          <TransactionTable
            transactions={transactions}
            onSelectTransaction={(tx) => setSelectedTx(tx)}
            maxRows={8}
          />
        </div>
        <div className="lg:col-span-1">
          <RiskRoutingDonut
            allowCount={allowCount}
            challengeCount={challengeCount}
            blockCount={blockCount}
            thresholds={
              defense?.circuit_breaker?.active_thresholds || {
                p_low: 0.0804,
                p_high: 0.7495,
              }
            }
          />
        </div>
      </div>

      {/* Transaction Detail Drawer */}
      <TransactionDetailDrawer
        transaction={selectedTx}
        onClose={() => setSelectedTx(null)}
      />
    </div>
  );
};
