import React, { useState } from "react";
import { TransactionTable } from "../components/transactions/TransactionTable";
import { TransactionDetailDrawer } from "../components/transactions/TransactionDetailDrawer";
import { TransactionRecord } from "../types/api";

interface TransactionsProps {
  transactions: TransactionRecord[];
  onRefresh: () => void;
}

export const Transactions: React.FC<TransactionsProps> = ({
  transactions,
  onRefresh,
}) => {
  const [selectedTx, setSelectedTx] = useState<TransactionRecord | null>(null);

  return (
    <div className="space-y-5">
      <div className="bg-dark-850 rounded-xl p-5 border border-dark-700/80 shadow-md">
        <h2 className="text-base font-bold text-white tracking-tight">
          Transaction Risk Stream
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl">
          Every inbound payment transaction is evaluated in sub-50ms using our calibrated XGBoost + LightGBM ensemble.
          Click any transaction row to inspect the top positive local SHAP attributions, feature anomalies, and operational routing decisions.
        </p>
      </div>

      <TransactionTable
        transactions={transactions}
        onSelectTransaction={(tx) => setSelectedTx(tx)}
      />

      <TransactionDetailDrawer
        transaction={selectedTx}
        onClose={() => setSelectedTx(null)}
      />
    </div>
  );
};
