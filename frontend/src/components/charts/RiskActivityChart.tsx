import React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

export interface TelemetryPoint {
  time: string;
  meanRisk: number;       // 0 to 100%
  highRiskRate: number;   // 0 to 100%
  volume: number;         // Count or relative volume
}

interface RiskActivityChartProps {
  data: TelemetryPoint[];
}

export const RiskActivityChart: React.FC<RiskActivityChartProps> = ({ data }) => {
  return (
    <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-lg flex flex-col h-[360px]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-bold text-sm text-white tracking-tight flex items-center gap-2">
            Real-Time Risk Activity
            <span className="w-2 h-2 rounded-full bg-brand-cyan animate-pulse" />
          </h3>
          <p className="text-xs text-slate-400">
            5-minute sliding window average risk score & high-risk burst telemetry
          </p>
        </div>

        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-brand-cyan" />
            <span className="text-slate-300">Mean Risk %</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span className="text-slate-300">High-Risk Rate %</span>
          </div>
        </div>
      </div>

      <div className="flex-1 w-full min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="cyanGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#00E5FF" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#EF4444" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1E293B" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#64748B"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: "#1E293B" }}
            />
            <YAxis
              stroke="#64748B"
              fontSize={11}
              domain={[0, 100]}
              tickFormatter={(val) => `${val}%`}
              tickLine={false}
              axisLine={{ stroke: "#1E293B" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0B0F17",
                borderColor: "#334155",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#F8FAFC",
                boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.5)",
              }}
              formatter={(value: any, name: any) => {
                const label = name === "meanRisk" ? "Mean Risk" : "High-Risk Rate";
                return [`${Number(value).toFixed(1)}%`, label];
              }}
            />
            <Area
              type="monotone"
              dataKey="meanRisk"
              stroke="#00E5FF"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#cyanGradient)"
              isAnimationActive={true}
            />
            <Area
              type="monotone"
              dataKey="highRiskRate"
              stroke="#EF4444"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#redGradient)"
              isAnimationActive={true}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
