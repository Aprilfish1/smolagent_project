"""
CCAR backtesting metric calculation tools.

Supports:
  - Regression metrics: RMSE, MAE, MAPE, R²
  - Rank-order metrics: Gini coefficient, KS statistic, AUC-ROC
  - Bias metrics: Mean Error (over/under prediction)
  - Segmented metrics: all of the above broken out by any grouping column(s)
"""
import json
import numpy as np
import pandas as pd
from smolagents import tool
from .data_tools import get_df


def _safe_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2*AUC - 1 computed without sklearn (works for regression scores too)."""
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true > y_true.mean(), y_score)
        return round(2 * auc - 1, 4)
    except Exception:
        return float("nan")


def _safe_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic between the score distributions of
    positives (above-median actuals) and negatives."""
    try:
        from scipy.stats import ks_2samp
        median = np.median(y_true)
        pos_scores = y_score[y_true > median]
        neg_scores = y_score[y_true <= median]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return float("nan")
        ks_stat, _ = ks_2samp(pos_scores, neg_scores)
        return round(float(ks_stat), 4)
    except Exception:
        return float("nan")


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Core metric computation for a single segment."""
    n = len(actual)
    if n == 0:
        return {}

    err = predicted - actual
    abs_err = np.abs(err)
    sq_err = err ** 2

    rmse = float(np.sqrt(np.mean(sq_err)))
    mae = float(np.mean(abs_err))
    mean_err = float(np.mean(err))

    # MAPE — avoid division by zero
    nonzero = actual != 0
    mape = float(np.mean(abs_err[nonzero] / np.abs(actual[nonzero]))) * 100 if nonzero.any() else float("nan")

    # R²
    ss_res = np.sum(sq_err)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    gini = _safe_gini(actual, predicted)
    ks = _safe_ks(actual, predicted)

    return {
        "n": n,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Mean_Error (bias)": round(mean_err, 4),
        "MAPE (%)": round(mape, 4) if not np.isnan(mape) else "n/a",
        "R2": round(r2, 4) if not np.isnan(r2) else "n/a",
        "Gini": gini,
        "KS": ks,
    }


@tool
def calculate_backtesting_metrics(
    actual_column: str,
    predicted_column: str,
    dataset_name: str = "main",
    group_by_columns: str = "",
) -> str:
    """Calculate key CCAR model backtesting performance metrics.

    Computes RMSE, MAE, Bias (Mean Error), MAPE, R², Gini, and KS statistic.
    When group_by_columns is provided the metrics are broken out per segment.

    Args:
        actual_column: Column name containing the observed/actual values.
        predicted_column: Column name containing the model-predicted values.
        dataset_name: Name of the dataset to use (must be loaded first). Defaults to "main".
        group_by_columns: Comma-separated column names to segment results by (e.g. "product_type,vintage_year"). Leave empty for portfolio-level only.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    for col in [actual_column, predicted_column]:
        if col not in df.columns:
            return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    actual = df[actual_column].dropna().values.astype(float)
    predicted = df[predicted_column].dropna().values.astype(float)

    lines = [f"=== Backtesting Metrics — Dataset: '{dataset_name}' ===\n"]

    # Portfolio-level
    portfolio_metrics = _compute_metrics(actual, predicted)
    lines.append("── Portfolio Level ──")
    for k, v in portfolio_metrics.items():
        lines.append(f"  {k}: {v}")

    # Segmented
    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    if group_cols:
        for col in group_cols:
            if col not in df.columns:
                lines.append(f"\nWARNING: Group column '{col}' not found — skipped.")
                continue

        lines.append(f"\n── Segmented by: {', '.join(group_cols)} ──")
        # Drop rows missing actual or predicted
        df_clean = df[[actual_column, predicted_column] + group_cols].dropna()
        grouped = df_clean.groupby(group_cols)

        results = []
        for key, grp in grouped:
            key_label = key if isinstance(key, str) else " | ".join(str(k) for k in key)
            m = _compute_metrics(
                grp[actual_column].values.astype(float),
                grp[predicted_column].values.astype(float),
            )
            m["segment"] = key_label
            results.append(m)

        # Display as a table
        result_df = pd.DataFrame(results).set_index("segment")
        lines.append(result_df.to_string())

    return "\n".join(lines)
