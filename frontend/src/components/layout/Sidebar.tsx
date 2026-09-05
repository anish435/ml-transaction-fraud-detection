import React from "react";
import {
  ShieldAlert,
  LayoutDashboard,
  ArrowLeftRight,
  ShieldCheck,
  Cpu,
  AlertTriangle,
  CreditCard,
  Code2,
  Activity,
  CheckCircle2,
  XCircle,
} from "lucide-react";

interface SidebarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
  isBackendOnline: boolean;
  modelVersion: string;
  incidentCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  isBackendOnline,
  modelVersion,
  incidentCount = 0,
}) => {
  const navItems = [
    {
      group: "MONITOR",
      items: [
        { id: "overview", label: "Overview", icon: LayoutDashboard },
        { id: "transactions", label: "Transactions", icon: ArrowLeftRight },
        { id: "defense", label: "Defense Center", icon: ShieldCheck },
      ],
    },
    {
      group: "ANALYTICS",
      items: [
        { id: "models", label: "Model Performance", icon: Cpu },
        {
          id: "incidents",
          label: "Incidents",
          icon: AlertTriangle,
          badge: incidentCount > 0 ? incidentCount : undefined,
        },
      ],
    },
    {
      group: "INTEGRATION",
      items: [
        { id: "razorpay", label: "Razorpay", icon: CreditCard },
        { id: "api", label: "API", icon: Code2 },
      ],
    },
  ];

  return (
    <aside className="w-64 bg-dark-950 border-r border-dark-700/60 flex flex-col h-screen select-none shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-dark-700/60">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-cyan/20 to-brand-blue/20 border border-brand-cyan/40 flex items-center justify-center text-brand-cyan shadow-glow-cyan">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white flex items-center gap-1.5">
              RiskGuard AI
              <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30">
                PRO
              </span>
            </h1>
            <p className="text-xs text-slate-400 font-medium">AI Payment Risk Manager</p>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        {navItems.map((group) => (
          <div key={group.group}>
            <p className="px-3 text-[11px] font-semibold text-slate-400 tracking-wider mb-1.5 uppercase">
              {group.group}
            </p>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = currentTab === item.id;
                return (
                  <button
                    key={item.id}
                    id={`nav-${item.id}`}
                    onClick={() => onSelectTab(item.id)}
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? "bg-brand-cyan/10 text-brand-cyan border border-brand-cyan/30 shadow-sm"
                        : "text-slate-400 hover:text-slate-200 hover:bg-dark-850 border border-transparent"
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <Icon
                        className={`w-4 h-4 ${
                          isActive ? "text-brand-cyan" : "text-slate-400"
                        }`}
                      />
                      <span>{item.label}</span>
                    </div>
                    {item.badge !== undefined && (
                      <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-risk-block/20 text-risk-block border border-risk-block/40">
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Telemetry */}
      <div className="p-3 border-t border-dark-700/60 bg-dark-900/60">
        <div className="bg-dark-850 rounded-lg p-3 border border-dark-700/50 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium">Engine API</span>
            <div className="flex items-center gap-1.5 font-semibold">
              {isBackendOnline ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-emerald-400 text-[11px]">ONLINE</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  <span className="text-rose-400 text-[11px]">OFFLINE</span>
                </>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>Model Version</span>
            <span className="font-mono text-slate-300 font-medium truncate max-w-[120px]" title={modelVersion}>
              {modelVersion || "v1.3.0"}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
