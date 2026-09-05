import {
  HealthResponse,
  StatsResponse,
  DefenseStatus,
  IncidentRecord,
  SuppressedEntity,
  ModelMetricsSummary,
  TransactionRecord,
  RazorpayAuditLog,
  OrderCreateResponse,
} from "../types/api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function fetchJSON<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      ...options,
    });
    if (!res.ok) {
      let errorMsg = `HTTP ${res.status} ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) errorMsg = errJson.detail;
      } catch {
        // use default errorMsg
      }
      throw new Error(errorMsg);
    }
    return await res.json();
  } catch (err: any) {
    console.error(`API request failed: ${endpoint}`, err);
    throw err;
  }
}

export const api = {
  // Health & Stats
  async getHealth(): Promise<HealthResponse> {
    return fetchJSON<HealthResponse>("/health");
  },

  async getStats(): Promise<StatsResponse> {
    return fetchJSON<StatsResponse>("/stats");
  },

  async getMetrics(): Promise<ModelMetricsSummary> {
    return fetchJSON<ModelMetricsSummary>("/metrics");
  },

  // Defense Center
  async getDefenseStatus(): Promise<DefenseStatus> {
    return fetchJSON<DefenseStatus>("/defense/status");
  },

  async getIncidents(limit = 50): Promise<{ total: number; incidents: IncidentRecord[] }> {
    return fetchJSON<{ total: number; incidents: IncidentRecord[] }>(`/defense/incidents?limit=${limit}`);
  },

  async resolveIncident(incidentId: string, reason = "Manually resolved from web console"): Promise<{ status: string; incident: IncidentRecord }> {
    return fetchJSON<{ status: string; incident: IncidentRecord }>(`/defense/incidents/${incidentId}/resolve`, {
      method: "POST",
      body: JSON.stringify({ reason }),
    });
  },

  async tripCircuitBreaker(reason = "Manual operator trip from dashboard", severity = "HIGH"): Promise<DefenseStatus> {
    return fetchJSON<DefenseStatus>("/defense/circuit-breaker/trip", {
      method: "POST",
      body: JSON.stringify({ reason, severity }),
    });
  },

  async resetCircuitBreaker(): Promise<DefenseStatus> {
    return fetchJSON<DefenseStatus>("/defense/circuit-breaker/reset", {
      method: "POST",
    });
  },

  async getSuppressionList(): Promise<{ total: number; suppressions: SuppressedEntity[] }> {
    return fetchJSON<{ total: number; suppressions: SuppressedEntity[] }>("/defense/suppression-list");
  },

  async removeSuppression(entityId: string): Promise<{ status: string; entity_id: string }> {
    return fetchJSON<{ status: string; entity_id: string }>("/defense/suppression-list/remove", {
      method: "POST",
      body: JSON.stringify({ entity_id: entityId }),
    });
  },

  // Transactions Feed & Scoring
  async getRecentTransactions(limit = 50): Promise<{ total: number; transactions: TransactionRecord[] }> {
    return fetchJSON<{ total: number; transactions: TransactionRecord[] }>(`/transactions/recent?limit=${limit}`);
  },

  async scoreTransaction(payload: any, includeReasons = true): Promise<any> {
    return fetchJSON<any>(`/score?include_reasons=${includeReasons}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  // Razorpay Test Mode
  async createRazorpayOrder(amount: number, currency = "INR", receipt?: string): Promise<OrderCreateResponse> {
    return fetchJSON<OrderCreateResponse>("/create-order", {
      method: "POST",
      body: JSON.stringify({ amount, currency, receipt }),
    });
  },

  async getRazorpayAuditLogs(limit = 50): Promise<{ total: number; logs: RazorpayAuditLog[] }> {
    return fetchJSON<{ total: number; logs: RazorpayAuditLog[] }>(`/razorpay/audit-logs?limit=${limit}`);
  },

  async getRazorpayOrder(orderId: string): Promise<any> {
    return fetchJSON<any>(`/order/${orderId}`);
  },

  // Demo Controls
  async simulateNormal(): Promise<{ status: string; transaction: TransactionRecord; defense_status: DefenseStatus }> {
    return fetchJSON<{ status: string; transaction: TransactionRecord; defense_status: DefenseStatus }>("/demo/simulate-normal", {
      method: "POST",
    });
  },

  async simulateSpike(): Promise<{ status: string; transactions_scored: number; recent_burst: TransactionRecord[]; defense_status: DefenseStatus }> {
    return fetchJSON<{ status: string; transactions_scored: number; recent_burst: TransactionRecord[]; defense_status: DefenseStatus }>("/demo/simulate-spike", {
      method: "POST",
    });
  },

  async simulateRecovery(): Promise<{ status: string; circuit_breaker_state: string; defense_status: DefenseStatus }> {
    return fetchJSON<{ status: string; circuit_breaker_state: string; defense_status: DefenseStatus }>("/demo/simulate-recovery", {
      method: "POST",
    });
  },

  async resetDemo(): Promise<{ status: string; defense_status: DefenseStatus }> {
    return fetchJSON<{ status: string; defense_status: DefenseStatus }>("/demo/reset", {
      method: "POST",
    });
  },
};
