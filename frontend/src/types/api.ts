export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  model_version: string;
  circuit_breaker_state: string;
}

export interface LatencyStats {
  request_count: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
}

export interface StatsResponse {
  total_requests: number;
  window_size: number;
  fast_mode: LatencyStats;
  explainable_mode: LatencyStats;
}

export type DecisionType = "ALLOW" | "CHALLENGE" | "HARD_BLOCK";
export type CircuitBreakerMode = "NORMAL" | "DEFENSE_ACTIVE" | "COOLDOWN";
export type SeverityType = "NORMAL" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface TransactionRecord {
  id: string;
  timestamp: string;
  amount: number;
  currency: string;
  fraud_probability: number;
  risk_tier: DecisionType;
  decision: DecisionType;
  reasons: string[];
  latency_ms: number;
  card4?: string;
  card6?: string;
  email?: string;
  circuit_breaker_state?: string;
  defense_action?: string;
  defense_note?: string;
}

export interface ThresholdValues {
  p_low: number;
  p_high: number;
}

export interface CircuitBreakerStatus {
  state: CircuitBreakerMode;
  is_defense_active: boolean;
  active_thresholds: ThresholdValues;
  standard_thresholds: ThresholdValues;
  defense_thresholds: ThresholdValues;
  last_state_change_epoch?: number;
  seconds_in_current_state: number;
  trip_reason?: string | null;
  trip_severity?: SeverityType | null;
  healthy_streak: number;
  cooldown_progress_pct: number;
}

export interface SlidingWindowTelemetry {
  window_seconds: number;
  tx_count: number;
  high_risk_count: number;
  challenge_count: number;
  allow_count: number;
  high_risk_rate_pct: number;
  challenge_rate_pct: number;
  allow_rate_pct: number;
  mean_risk_prob: number;
  total_volume_amt: number;
  high_risk_volume_amt: number;
  spike_severity: SeverityType;
  is_spike: boolean;
  score_drift_status: "STABLE" | "MODERATE_DRIFT" | "SIGNIFICANT_DRIFT";
  burst_velocity_60s: number;
}

export interface IncidentRecord {
  incident_id: string;
  status: "ACTIVE" | "MITIGATING" | "RESOLVED";
  severity: SeverityType;
  started_at: string;
  started_ts?: number;
  resolved_at?: string | null;
  resolved_ts?: number | null;
  duration_seconds?: number | null;
  trigger_metrics?: Record<string, any>;
  affected_transactions_count: number;
  resolution_reason?: string | null;
}

export interface SuppressedEntity {
  entity_id: string;
  reason: string;
  violations: number;
  suppressed_at: number;
  expires_at: number;
  remaining_ttl_seconds: number;
  status: string;
}

export interface DefenseStatus {
  circuit_breaker: CircuitBreakerStatus;
  sliding_window_telemetry: SlidingWindowTelemetry;
  active_incident: IncidentRecord | null;
  suppressed_entities_count: number;
  suppressed_entities: SuppressedEntity[];
}

export interface ModelMetricsSummary {
  thresholds: {
    p_low_allow_challenge: number;
    p_high_challenge_block: number;
    optimization_objective: string;
    tuned_on: string;
    evaluated_on: string;
    constraints_applied?: {
      challenge_rate_max_pct: number;
      auto_block_precision_min_pct: number;
    };
  };
  val_metrics: {
    challenge_rate_pct: number;
    auto_block_precision_pct: number;
    net_savings_lakhs: number;
    ml_recall_pct: number;
    vw_recall_pct?: number;
    ml_precision_pct?: number;
  };
  ml_detection: {
    precision_pct: number;
    recall_pct: number;
    pr_auc: number;
    roc_auc: number;
  };
  financial_cost: {
    net_merchant_savings_inr_lakhs: number;
    net_merchant_savings_usd: number;
    fraud_caught_usd: number;
    fraud_missed_usd: number;
    value_weighted_recall_pct?: number;
  };
  merchant_funnel: {
    auto_block_precision_pct: number;
    challenge_rate_pct: number;
    allow_rate_pct: number;
    block_rate_pct: number;
  };
  operations: {
    inference_latency_p50_ms: number;
    inference_latency_p95_ms: number;
    inference_latency_p99_ms: number;
  };
  confusion_matrix: {
    TP: number;
    FP: number;
    FN: number;
    TN: number;
    threshold: number;
  };
  ensemble_weights: {
    w_lr: number;
    w_nn: number;
    w_xgb: number;
    w_lgbm: number;
  };
  pipeline_version: string;
  model_comparison: Record<string, {
    roc_auc: number;
    pr_auc: number;
    weights?: Record<string, number>;
  }>;
}

export interface RazorpayAuditLog {
  payment_id: string;
  order_id?: string;
  timestamp: string;
  customer_identifier: string;
  amount: number;
  currency: string;
  fraud_probability: number;
  risk_tier: DecisionType;
  decision: DecisionType;
  reasons: string[];
  features_real?: Record<string, any>;
  features_defaulted?: string[];
  latency_ms: number;
  event?: string;
  circuit_breaker_state?: string;
  defense_action?: string;
}

export interface OrderCreateResponse {
  order_id: string;
  amount: number;
  currency: string;
  status: string;
  receipt?: string;
  created_at: number;
}
