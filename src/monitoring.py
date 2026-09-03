"""
Population Stability Index (PSI) Monitoring Module.

Provides functions to compute, evaluate, and interpret Population Stability Index (PSI)
for monitoring score drift and feature drift across time-based splits, batches, or production windows.

Mathematical Definition:
-----------------------
    PSI = sum_{i=1}^k (Actual%_i - Expected%_i) * ln(Actual%_i / Expected%_i)

Standard Industry Benchmarks:
----------------------------
    PSI < 0.10: Stable (No significant shift; model is safe).
    0.10 <= PSI < 0.25: Moderate shift (Slight drift; monitor closely).
    PSI >= 0.25: Significant shift / Alert (Action required / Retraining needed).
"""

import numpy as np
import pandas as pd


def compute_psi(expected, actual, bins=10, epsilon=1e-4):
    """
    Compute Population Stability Index (PSI) between an expected (baseline)
    distribution and an actual (monitoring/target) distribution.

    Parameters:
    -----------
    expected : array-like
        Baseline / reference distribution (e.g. training set scores or features).
    actual : array-like
        Current / target distribution (e.g. validation, test, walk-forward, or batch).
    bins : int, default=10
        Number of quantile bins (deciles by default) to segment expected data.
    epsilon : float, default=1e-4
        Small constant added to zero-count bins to prevent division by zero or ln(0).

    Returns:
    --------
    float : The computed Population Stability Index (PSI).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Filter out NaNs / Infs
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bin edges on expected distribution
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, quantiles)

    # Remove duplicates in bin edges (common for discrete counts with many ties)
    bin_edges = np.unique(bin_edges)

    # If all values are identical or only 1 bin edge exists, create a basic 2-bin split
    if len(bin_edges) < 2:
        val = bin_edges[0] if len(bin_edges) == 1 else expected[0]
        bin_edges = np.array([val - 1.0, val, val + 1.0])

    # Extend boundary edges to infinity to capture all out-of-range actual values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute frequency counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert counts to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Epsilon smoothing for zero counts
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    # Re-normalize proportions so they sum to 1.0
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)

    # Compute PSI
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def interpret_psi(psi_val):
    """
    Interpret a PSI value according to standard risk management thresholds.

    Parameters:
    -----------
    psi_val : float
        Calculated Population Stability Index value.

    Returns:
    --------
    status : str
        'STABLE', 'MODERATE_DRIFT', or 'SIGNIFICANT_DRIFT'
    is_alert : bool
        True if PSI >= 0.25 (significant drift requiring attention), False otherwise.
    """
    if psi_val < 0.10:
        return "STABLE", False
    elif psi_val < 0.25:
        return "MODERATE_DRIFT", False
    else:
        return "SIGNIFICANT_DRIFT", True


def compute_feature_psi_table(expected_df, actual_dict, features, score_col=None, bins=10):
    """
    Compute PSI across multiple evaluation datasets/windows for both model scores and features.

    Parameters:
    -----------
    expected_df : pd.DataFrame
        Baseline reference dataframe (e.g. train set) with features (and optional score_col).
    actual_dict : dict
        Mapping of {dataset_name: pd.DataFrame} for target evaluation sets.
    features : list of str
        List of feature column names to monitor.
    score_col : str, optional
        Column name containing model prediction probabilities if present in DataFrames.
    bins : int, default=10
        Number of decile bins.

    Returns:
    --------
    pd.DataFrame : Formatted table of PSI values and drift flags.
    """
    rows = []
    cols_to_eval = []
    if score_col and score_col in expected_df.columns:
        cols_to_eval.append(("Model Score (Calibrated Prob)", score_col))
    for f in features:
        if f in expected_df.columns:
            cols_to_eval.append((f, f))

    for target_name, actual_df in actual_dict.items():
        for label, col in cols_to_eval:
            if col not in actual_df.columns:
                continue
            e_vals = expected_df[col].values
            a_vals = actual_df[col].values
            psi = compute_psi(e_vals, a_vals, bins=bins)
            status, is_alert = interpret_psi(psi)
            flag = "[!] DRIFT (>=0.25)" if is_alert else ("MODERATE" if status == "MODERATE_DRIFT" else "STABLE")
            rows.append({
                "Target Dataset": target_name,
                "Metric / Feature": label,
                "PSI Value": round(psi, 4),
                "Status": status,
                "Alert": flag,
            })

    return pd.DataFrame(rows)
