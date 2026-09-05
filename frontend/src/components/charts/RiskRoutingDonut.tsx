import React from "react";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from "recharts";
import { HelpCircle } from "lucide-react";

interface RiskRoutingDonutProps {
  allowCount: number;
  challengeCount: number;
  blockCount: number;
  thresholds?: { p_low: number; p_high: number };
}

export const RiskRoutingDonut: React.FC<RiskRoutingDonutProps> = ({
  allowCount,
  challengeCount,
  blockCount,
  thresholds = { p_low: 0.0804, p_high: 0.7495 },
}) => {
  const total = allowCount + challengeCount + blockCount || 1;
  const allowPct = ((allowCount / total) * 100).toFixed(1);
  const challengePct = ((challengeCount / total) * 100).toFixed(1);
  const blockPct = ((blockCount / total) * 100).toFixed(1);

  const data = [
    { name: "ALLOW", value: allowCount || 0.1, color: "#10B981", percent: allowPct },
    { name: "CHALLENGE", value: challengeCount || 0, color: "#F59E0B", percent: challengePct },
    { name: "HARD_BLOCK", value: blockCount || 0, color: "#EF4444", percent: blockPct },
  ];

  return (
    <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-lg flex flex-col justify-between h-[360px]">
      <div>
        <div className="flex items-center justify-between">
          <h3 className="font-bold text-sm text-white tracking-tight">Risk Routing Distribution</h3>
          <div className="group relative">
            <HelpCircle className="w-4 h-4 text-slate-400 hover:text-slate-300 cursor-pointer" />
            <div className="absolute right-0 bottom-full mb-2 w-56 p-2.5 bg-dark-950 border border-dark-700 rounded-lg text-[11px] text-slate-300 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
              Thresholds tuned on validation set and evaluated on sealed test:
              <br />
              <b className="text-emerald-400">ALLOW:</b> p &lt; {thresholds.p_low.toFixed(4)}
              <br />
              <b className="text-amber-400">CHALLENGE:</b> {thresholds.p_low.toFixed(4)} ≤ p &lt; {thresholds.p_high.toFixed(4)}
              <br />
              <b className="text-rose-400">HARD_BLOCK:</b> p ≥ {thresholds.p_high.toFixed(4)}
            </div>
          </div>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">3-tier action routing split</p>
      </div>

      {/* Donut Chart */}
      <div className="relative flex-1 w-full flex items-center justify-center my-2">
        <ResponsiveContainer width="100%" height={170}>
          <PieChart>
            <Tooltip
              contentStyle={{
                backgroundColor: "#0B0F17",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#F8FAFC",
              }}
              formatter={(value: any, name: any) => [
                `${Number(value) >= 1 ? value : 0} tx`,
                name,
              ]}
            />
            <Pie
              data={data}
              innerRadius={48}
              outerRadius={70}
              paddingAngle={3}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-xs text-slate-400 font-medium">Decisions</span>
          <span className="text-lg font-bold text-white font-mono">
            {allowCount + challengeCount + blockCount}
          </span>
        </div>
      </div>

      {/* Legend & Thresholds */}
      <div className="space-y-1.5 pt-2 border-t border-dark-700/60 text-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-slate-300 font-medium">ALLOW</span>
            <span className="text-[10px] text-slate-500 font-mono">
              (p &lt; {thresholds.p_low.toFixed(4)})
            </span>
          </div>
          <span className="font-mono font-bold text-slate-200">{allowPct}%</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            <span className="text-slate-300 font-medium">CHALLENGE</span>
            <span className="text-[10px] text-slate-500 font-mono">
              (to {thresholds.p_high.toFixed(4)})
            </span>
          </div>
          <span className="font-mono font-bold text-slate-200">{challengePct}%</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span className="text-slate-300 font-medium">HARD_BLOCK</span>
            <span className="text-[10px] text-slate-500 font-mono">
              (p ≥ {thresholds.p_high.toFixed(4)})
            </span>
          </div>
          <span className="font-mono font-bold text-slate-200">{blockPct}%</span>
        </div>
      </div>
    </div>
  );
};
