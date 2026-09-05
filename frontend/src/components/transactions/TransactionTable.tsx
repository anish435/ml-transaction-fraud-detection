import React, { useState, useMemo } from "react";
import { Search, Filter, ArrowUpRight, ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { TransactionRecord, DecisionType } from "../../types/api";

interface TransactionTableProps {
  transactions: TransactionRecord[];
  onSelectTransaction: (tx: TransactionRecord) => void;
  maxRows?: number;
}

export const TransactionTable: React.FC<TransactionTableProps> = ({
  transactions,
  onSelectTransaction,
  maxRows,
}) => {
  const [filterDecision, setFilterDecision] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  const filtered = useMemo(() => {
    return transactions.filter((tx) => {
      const matchesFilter =
        filterDecision === "ALL" || tx.decision === filterDecision;
      const matchesSearch =
        searchQuery === "" ||
        tx.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (tx.email && tx.email.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesFilter && matchesSearch;
    });
  }, [transactions, filterDecision, searchQuery]);

  const displayList = maxRows ? filtered.slice(0, maxRows) : filtered;

  const renderBadge = (decision: DecisionType) => {
    switch (decision) {
      case "ALLOW":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            ALLOW
          </span>
        );
      case "CHALLENGE":
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            CHALLENGE
          </span>
        );
      case "HARD_BLOCK":
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/15 text-rose-400 border border-rose-500/30">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
            HARD_BLOCK
          </span>
        );
    }
  };

  return (
    <div className="bg-dark-850 rounded-xl border border-dark-700/80 overflow-hidden shadow-lg">
      {/* Table Controls */}
      <div className="p-4 border-b border-dark-700/70 flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-dark-900/60">
        <div className="flex items-center gap-2">
          <h3 className="font-bold text-sm text-white tracking-tight">Live Transactions Feed</h3>
          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-dark-750 text-slate-400 border border-dark-700">
            {filtered.length} captured
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search ID / email..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-dark-800 border border-dark-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-brand-cyan/50 focus:ring-1 focus:ring-brand-cyan/30 transition-all w-44"
            />
          </div>

          {/* Filter Pills */}
          <div className="flex items-center bg-dark-800 p-0.5 rounded-lg border border-dark-700 text-xs">
            {["ALL", "ALLOW", "CHALLENGE", "HARD_BLOCK"].map((tab) => (
              <button
                key={tab}
                onClick={() => setFilterDecision(tab)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                  filterDecision === tab
                    ? "bg-dark-700 text-brand-cyan shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-dark-700/60 bg-dark-900/40 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              <th className="py-3 px-4">Time</th>
              <th className="py-3 px-4">Transaction ID</th>
              <th className="py-3 px-4">Amount</th>
              <th className="py-3 px-4">Risk Probability</th>
              <th className="py-3 px-4">Decision</th>
              <th className="py-3 px-4">Latency</th>
              <th className="py-3 px-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-750/50 text-xs text-slate-300">
            {displayList.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-500 italic">
                  No transactions recorded matching the current filter.
                </td>
              </tr>
            ) : (
              displayList.map((tx) => {
                const prob = (tx.fraud_probability * 100).toFixed(1);
                const timeFormatted = new Date(tx.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });
                return (
                  <tr
                    key={tx.id}
                    onClick={() => onSelectTransaction(tx)}
                    className="hover:bg-dark-800/60 transition-colors cursor-pointer group"
                  >
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                      {timeFormatted}
                    </td>
                    <td className="py-3 px-4 font-mono font-medium text-slate-200 group-hover:text-brand-cyan transition-colors">
                      {tx.id}
                    </td>
                    <td className="py-3 px-4 font-semibold text-white">
                      {tx.currency === "INR" ? "₹" : "$"}
                      {tx.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-slate-200 font-bold w-12">
                          {prob}%
                        </span>
                        <div className="w-16 h-1.5 bg-dark-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              tx.fraud_probability >= 0.7495
                                ? "bg-rose-500"
                                : tx.fraud_probability >= 0.0804
                                ? "bg-amber-500"
                                : "bg-emerald-500"
                            }`}
                            style={{ width: `${Math.max(Number(prob), 4)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4">{renderBadge(tx.decision)}</td>
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                      {tx.latency_ms.toFixed(1)} ms
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTransaction(tx);
                        }}
                        className="p-1.5 rounded-md hover:bg-dark-700 text-slate-400 hover:text-brand-cyan transition-all"
                        title="Inspect AI reasons"
                      >
                        <ArrowUpRight className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
