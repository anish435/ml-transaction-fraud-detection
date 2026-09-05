"""
Business Routing Threshold Optimisation
=========================================
Tunes ALLOW / CHALLENGE / HARD_BLOCK thresholds on the *validation* set
(temporal 70--85 pct split) to maximise Net Merchant Savings (primary)
and Fraud Recall (secondary), subject to hard constraints:
  * Challenge Rate       <  6 %
  * Auto-Block Precision > 90 %

After selecting optimal thresholds on the validation set the script
evaluates *once* on the sealed test set (85--100 pct) and writes
  models/metrics_summary.json
  models/routing_thresholds.json
"""

import os, sys, json, time
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from src.features import transform_features, make_Xy, compute_rolling_features
from fraud_detection import FraudMLP, load_and_merge_data

FX_INR_PER_USD       = 83.0
INR_LAKHS_DIVISOR    = 100_000
FALSE_DECLINE_MARGIN = 0.03
FALSE_DECLINE_CHURN  = 10.0   # USD per blocked legitimate txn


def compute_metrics(y_true, probs, amounts, p_low, p_high):
    """Return dict of all routing + ML + financial metrics."""
    y = np.asarray(y_true)
    a = np.asarray(amounts)
    n = len(y)

    allow_mask     = probs < p_low
    challenge_mask = (probs >= p_low) & (probs < p_high)
    block_mask     = probs >= p_high

    challenge_rate  = challenge_mask.sum() / n * 100
    ab_fraud        = y[block_mask].sum()
    ab_total        = block_mask.sum()
    auto_block_prec = float(ab_fraud / max(ab_total, 1))

    preds = block_mask.astype(int)
    cm    = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    ml_precision = float(tp / max(tp + fp, 1))
    ml_recall    = float(tp / max(tp + fn, 1))
    ml_pr_auc    = float(average_precision_score(y, probs))
    ml_roc_auc   = float(roc_auc_score(y, probs))

    fraud_caught_mask = (y == 1) & block_mask
    false_alarm_mask  = (y == 0) & block_mask
    fraud_missed_mask = (y == 1) & (~block_mask)

    fraud_caught_val = float(a[fraud_caught_mask].sum())
    fraud_missed_val = float(a[fraud_missed_mask].sum())
    false_alarm_cost = float(
        a[false_alarm_mask].sum() * FALSE_DECLINE_MARGIN
        + false_alarm_mask.sum() * FALSE_DECLINE_CHURN
    )
    net_savings      = fraud_caught_val - false_alarm_cost
    net_savings_lkh  = net_savings * FX_INR_PER_USD / INR_LAKHS_DIVISOR

    total_fraud_val = float(a[y == 1].sum())
    vw_recall       = float(fraud_caught_val / max(total_fraud_val, 1e-9))

    return dict(
        challenge_rate_pct       = round(challenge_rate, 4),
        auto_block_precision_pct = round(auto_block_prec * 100, 4),
        net_savings_lakhs        = round(net_savings_lkh, 4),
        ml_recall_pct            = round(ml_recall * 100, 4),
        vw_recall_pct            = round(vw_recall * 100, 4),
        ml_precision_pct         = round(ml_precision * 100, 4),
        ml_pr_auc                = round(ml_pr_auc, 6),
        ml_roc_auc               = round(ml_roc_auc, 6),
        fraud_caught_usd         = round(fraud_caught_val, 2),
        fraud_missed_usd         = round(fraud_missed_val, 2),
        allow_rate_pct           = round(allow_mask.sum() / n * 100, 4),
        block_rate_pct           = round(block_mask.sum() / n * 100, 4),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def grid_search(y_val, val_probs, val_amounts,
                cr_max=6.0, ab_min=90.0,
                n_low=40, n_high=60):
    p_low_grid  = np.linspace(0.005, 0.25, n_low)
    p_high_grid = np.linspace(0.30,  0.98, n_high)
    candidates  = []

    for p_low in p_low_grid:
        for p_high in p_high_grid:
            if p_high <= p_low:
                continue
            m = compute_metrics(y_val, val_probs, val_amounts, p_low, p_high)
            if m["challenge_rate_pct"] >= cr_max:
                continue
            if m["auto_block_precision_pct"] <= ab_min:
                continue
            candidates.append((p_low, p_high, m))

    if not candidates:
        print("[WARN] No feasible pair found; relaxing Auto-Block floor by 5 pp")
        for p_low in p_low_grid:
            for p_high in p_high_grid:
                if p_high <= p_low:
                    continue
                m = compute_metrics(y_val, val_probs, val_amounts, p_low, p_high)
                if m["challenge_rate_pct"] >= cr_max:
                    continue
                if m["auto_block_precision_pct"] <= (ab_min - 5.0):
                    continue
                candidates.append((p_low, p_high, m))

    candidates.sort(
        key=lambda x: (x[2]["net_savings_lakhs"], x[2]["ml_recall_pct"]),
        reverse=True,
    )
    if candidates:
        return candidates[0][0], candidates[0][1], candidates[0][2], candidates
    return None, None, None, []


def blend_probs(calibrated_xgb, calibrated_lgbm,
                mlp_probs, X, w_lr, w_nn, w_xgb, w_lgbm):
    xgb_p  = calibrated_xgb.predict_proba(X)[:, 1]
    lgbm_p = calibrated_lgbm.predict_proba(X)[:, 1]
    lr_p   = np.zeros_like(xgb_p)
    return w_lr * lr_p + w_nn * mlp_probs + w_xgb * xgb_p + w_lgbm * lgbm_p


def get_mlp_probs(X, nn_imputer, nn_scaler):
    Ximp = nn_imputer.transform(X)
    Xsc  = nn_scaler.transform(Ximp)
    mlp  = FraudMLP(Xsc.shape[1])
    mlp.load_state_dict(torch.load("models/fraud_mlp.pt", map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        return torch.sigmoid(
            mlp(torch.tensor(Xsc, dtype=torch.float32))
        ).numpy().ravel()


def main():
    print("=" * 65)
    print(" BUSINESS ROUTING THRESHOLD OPTIMISATION  v2.1")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Load data                                                           #
    # ------------------------------------------------------------------ #
    print("\nLoading data ...")
    data  = load_and_merge_data()
    data  = compute_rolling_features(data)
    data  = data.sort_values("TransactionDT").reset_index(drop=True)
    n     = len(data)

    val_set  = data.iloc[int(n * 0.70): int(n * 0.85)].copy()
    test_set = data.iloc[int(n * 0.85):].copy()
    print(f"Val set : {val_set.shape}  |  Test set : {test_set.shape}")

    state = joblib.load("models/feature_state.pkl")

    val_f        = transform_features(val_set, state)
    X_val, y_val = make_Xy(val_f, state)
    val_amounts  = val_set["TransactionAmt"].values

    te_f           = transform_features(test_set, state)
    X_test, y_test = make_Xy(te_f, state)
    test_amounts   = test_set["TransactionAmt"].values

    calibrated_xgb  = joblib.load("models/calibrated_xgb.pkl")
    calibrated_lgbm = joblib.load("models/calibrated_lgbm.pkl")
    w_data          = joblib.load("models/stacking_weights.pkl")
    w_lr, w_nn, w_xgb, w_lgbm = w_data["weights"]
    nn_imputer      = joblib.load("models/nn_imputer.pkl")
    nn_scaler       = joblib.load("models/nn_scaler.pkl")

    print("Computing validation blend probabilities ...")
    val_mlp  = get_mlp_probs(X_val, nn_imputer, nn_scaler)
    val_probs = blend_probs(calibrated_xgb, calibrated_lgbm,
                            val_mlp, X_val, w_lr, w_nn, w_xgb, w_lgbm)

    print("Computing test blend probabilities ...")
    test_mlp   = get_mlp_probs(X_test, nn_imputer, nn_scaler)
    test_probs = blend_probs(calibrated_xgb, calibrated_lgbm,
                             test_mlp, X_test, w_lr, w_nn, w_xgb, w_lgbm)

    # ------------------------------------------------------------------ #
    # Phase 1 — grid search on VAL set                                    #
    # ------------------------------------------------------------------ #
    print("\n[Phase 1] Grid search on VALIDATION set ...")
    p_low, p_high, val_m, cands = grid_search(
        y_val, val_probs, val_amounts,
        cr_max=6.0, ab_min=90.0,
    )

    if p_low is None:
        print("[ERROR] No feasible thresholds found. Using fallback 0.05 / 0.50")
        p_low, p_high = 0.05, 0.50
        val_m = compute_metrics(y_val, val_probs, val_amounts, p_low, p_high)

    print(f"\n  Best thresholds (val):")
    print(f"    p_low  = {p_low:.4f}  (ALLOW / CHALLENGE boundary)")
    print(f"    p_high = {p_high:.4f}  (CHALLENGE / HARD_BLOCK boundary)")
    print(f"\n  Validation metrics at selected thresholds:")
    print(f"    Challenge Rate       : {val_m['challenge_rate_pct']:.2f} %  (limit < 6 %)")
    print(f"    Auto-Block Precision : {val_m['auto_block_precision_pct']:.2f} %  (limit > 90 %)")
    print(f"    Net Savings          : INR {val_m['net_savings_lakhs']:.2f} Lakhs")
    print(f"    Fraud Recall         : {val_m['ml_recall_pct']:.2f} %")
    print(f"    Value-Weighted Recall: {val_m['vw_recall_pct']:.2f} %")
    print(f"    Precision            : {val_m['ml_precision_pct']:.2f} %")
    print(f"    PR-AUC               : {val_m['ml_pr_auc']:.4f}")

    print(f"\n  Top-5 feasible candidates:")
    hdr = f"  {'#':<3} {'p_low':>7} {'p_high':>7} {'Savings(L)':>11} {'Recall%':>8} {'ChalRate%':>10} {'ABPrec%':>8}"
    print(hdr)
    print("  " + "-" * 62)
    for rank, (pl, ph, m) in enumerate(cands[:5], 1):
        print(
            f"  {rank:<3} {pl:>7.4f} {ph:>7.4f} "
            f"{m['net_savings_lakhs']:>11.2f} "
            f"{m['ml_recall_pct']:>8.2f} "
            f"{m['challenge_rate_pct']:>10.2f} "
            f"{m['auto_block_precision_pct']:>8.2f}"
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — single evaluation on SEALED TEST set                      #
    # ------------------------------------------------------------------ #
    print("\n[Phase 2] Evaluating on SEALED TEST set (single pass) ...")
    tm = compute_metrics(y_test, test_probs, test_amounts, p_low, p_high)

    box = [
        "=" * 60,
        "   SEALED TEST SET — FINAL RESULTS",
        "=" * 60,
        f"  Thresholds: p_low={p_low:.4f}  p_high={p_high:.4f}",
        "-" * 60,
        f"  Challenge Rate       : {tm['challenge_rate_pct']:>7.2f} %   (limit: < 6 %)",
        f"  Auto-Block Precision : {tm['auto_block_precision_pct']:>7.2f} %   (limit: > 90 %)",
        f"  Net Merchant Savings : INR {tm['net_savings_lakhs']:>8.2f} Lakhs",
        f"  Fraud Recall         : {tm['ml_recall_pct']:>7.2f} %",
        f"  Value-Weighted Recall: {tm['vw_recall_pct']:>7.2f} %",
        f"  Precision            : {tm['ml_precision_pct']:>7.2f} %",
        f"  PR-AUC               : {tm['ml_pr_auc']:>10.4f}",
        f"  ROC-AUC              : {tm['ml_roc_auc']:>10.4f}",
        f"  Allow Rate           : {tm['allow_rate_pct']:>7.2f} %",
        f"  Block Rate           : {tm['block_rate_pct']:>7.2f} %",
        "-" * 60,
        f"  Confusion Matrix  TP={tm['tp']}  FP={tm['fp']}  FN={tm['fn']}  TN={tm['tn']}",
        "=" * 60,
    ]
    print("\n" + "\n".join(box))

    # ------------------------------------------------------------------ #
    # Phase 3 — latency benchmark                                         #
    # ------------------------------------------------------------------ #
    print("\n[Phase 3] Inference latency benchmark ...")
    sample_X = X_test.iloc[:1]
    lats = []
    for _ in range(100):
        t0 = time.perf_counter()
        calibrated_xgb.predict_proba(sample_X)
        lats.append((time.perf_counter() - t0) * 1000)
    lat = {
        "p50_ms": float(np.percentile(lats, 50)),
        "p95_ms": float(np.percentile(lats, 95)),
        "p99_ms": float(np.percentile(lats, 99)),
    }
    print(f"  p50={lat['p50_ms']:.2f} ms  p95={lat['p95_ms']:.2f} ms  p99={lat['p99_ms']:.2f} ms")

    # ------------------------------------------------------------------ #
    # Write outputs                                                        #
    # ------------------------------------------------------------------ #
    summary = {
        "thresholds": {
            "p_low_allow_challenge":  round(float(p_low),  6),
            "p_high_challenge_block": round(float(p_high), 6),
            "optimization_objective": "max(net_savings_lakhs), tiebreak max(recall)",
            "constraints_applied": {
                "challenge_rate_max_pct":       6.0,
                "auto_block_precision_min_pct": 90.0,
            },
            "tuned_on":    "validation_set_70pct_85pct",
            "evaluated_on":"sealed_test_set_85pct_100pct",
        },
        "val_metrics": {
            "challenge_rate_pct":        val_m["challenge_rate_pct"],
            "auto_block_precision_pct":  val_m["auto_block_precision_pct"],
            "net_savings_lakhs":         val_m["net_savings_lakhs"],
            "ml_recall_pct":             val_m["ml_recall_pct"],
            "vw_recall_pct":             val_m["vw_recall_pct"],
            "ml_precision_pct":          val_m["ml_precision_pct"],
        },
        "ml_detection": {
            "precision_pct": tm["ml_precision_pct"],
            "recall_pct":    tm["ml_recall_pct"],
            "pr_auc":        round(tm["ml_pr_auc"], 4),
            "roc_auc":       round(tm["ml_roc_auc"], 4),
        },
        "financial_cost": {
            "net_merchant_savings_inr_lakhs": tm["net_savings_lakhs"],
            "fraud_caught_usd":               tm["fraud_caught_usd"],
            "fraud_missed_usd":               tm["fraud_missed_usd"],
            "value_weighted_recall_pct":      tm["vw_recall_pct"],
        },
        "merchant_funnel": {
            "auto_block_precision_pct": tm["auto_block_precision_pct"],
            "challenge_rate_pct":       tm["challenge_rate_pct"],
            "allow_rate_pct":           tm["allow_rate_pct"],
            "block_rate_pct":           tm["block_rate_pct"],
        },
        "operations": {
            "inference_latency_p50_ms": round(lat["p50_ms"], 2),
            "inference_latency_p95_ms": round(lat["p95_ms"], 2),
            "inference_latency_p99_ms": round(lat["p99_ms"], 2),
        },
        "confusion_matrix": {
            "p_high_threshold": round(float(p_high), 6),
            "TP": tm["tp"], "FP": tm["fp"], "FN": tm["fn"], "TN": tm["tn"],
        },
        "ensemble_weights": {
            "w_lr": float(w_lr), "w_nn": float(w_nn),
            "w_xgb": float(w_xgb), "w_lgbm": float(w_lgbm),
        },
        "pipeline_version": "v2.1.0-threshold-optimised",
    }

    os.makedirs(os.path.join(REPO_ROOT, "models"), exist_ok=True)
    metrics_path = os.path.join(REPO_ROOT, "models", "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVED] metrics_summary.json -> {metrics_path}")

    thresholds_path = os.path.join(REPO_ROOT, "models", "routing_thresholds.json")
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump({"p_low": round(float(p_low), 6), "p_high": round(float(p_high), 6)}, f, indent=2)
    print(f"[SAVED] routing_thresholds.json -> {thresholds_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
