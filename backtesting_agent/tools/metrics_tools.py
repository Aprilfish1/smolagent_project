"""
CCAR backtesting metric calculation tools.

Metrics
-------
  AMPE   – Absolute Mean Percentage Error = mean(|actual-pred|/|actual|) × 100
  MPE    – Mean Percentage Error           = mean((pred-actual)/actual)   × 100
           positive MPE → model over-predicts; negative → under-predicts
  Abs Dollar Error  – mean(|actual - predicted|)
  Mean Dollar Error – mean(actual - predicted)  (signed, same sign as MPE)
  RMSE, R², Gini, KS  — rank-order / regression metrics

Two measurement levels
----------------------
  Portfolio level  : metrics computed on portfolio totals
                     (sum of actuals vs sum of predicted across all rows / segment)
  Per-account level: metrics computed per row, then averaged
                     OR if n_accounts_column exists: total / n_accounts

Multiple target pairs
---------------------
  Pass target_pairs = "actual_col:predicted_col" for a single target, or
  "col1_a:col1_p,col2_a:col2_p" for multiple targets in one call.
"""
import numpy as np
import pandas as pd
from smolagents import tool
from .data_tools import get_df


# ─────────────────────────────────────────────────────────────────────────────
# Rank-order helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true > y_true.mean(), y_score)
        return round(2 * auc - 1, 4)
    except Exception:
        return float("nan")


def _safe_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
        median = np.median(y_true)
        pos = y_score[y_true > median]
        neg = y_score[y_true <= median]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        stat, _ = ks_2samp(pos, neg)
        return round(float(stat), 4)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _compute_account_level(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Account-level metrics: computed row-by-row, then averaged.
    Error = predicted - actual  (positive = over-prediction, negative = under-prediction)
    """
    n = len(actual)
    if n == 0:
        return {}

    diff = predicted - actual          # positive = over-prediction
    abs_diff = np.abs(diff)

    nonzero = actual != 0
    ampe = float(np.mean(abs_diff[nonzero] / np.abs(actual[nonzero])) * 100) if nonzero.any() else float("nan")
    mpe  = float(np.mean(diff[nonzero]     / actual[nonzero])          * 100) if nonzero.any() else float("nan")

    sq = diff ** 2
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = float(1 - np.sum(sq) / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "n_rows":              n,
        "AMPE (%)":            round(ampe, 4) if not np.isnan(ampe) else float("nan"),
        "MPE (%)":             round(mpe,  4) if not np.isnan(mpe)  else float("nan"),
        "Abs Error (per row)": round(float(np.mean(abs_diff)), 6),
        "Mean Error (per row)":round(float(np.mean(diff)),     6),
        "RMSE":                round(float(np.sqrt(np.mean(sq))), 6),
        "R2":                  round(r2, 4) if not np.isnan(r2) else float("nan"),
        "Gini":                _safe_gini(actual, predicted),
        "KS":                  _safe_ks(actual, predicted),
    }


def _compute_portfolio_level(actual: np.ndarray, predicted: np.ndarray,
                             n_accounts: int | None = None) -> dict:
    """
    Portfolio-level metrics: compare the SUM of actuals vs SUM of predicted.
    If n_accounts is provided, also show per-account dollar values.
    """
    n = len(actual)
    if n == 0:
        return {}

    total_actual    = float(np.sum(actual))
    total_predicted = float(np.sum(predicted))
    total_diff      = total_predicted - total_actual   # positive = over-prediction
    total_abs_diff  = abs(total_diff)

    pct_err     = (total_diff / total_actual * 100) if total_actual != 0 else float("nan")
    abs_pct_err = (total_abs_diff / abs(total_actual) * 100) if total_actual != 0 else float("nan")

    result = {
        "n_rows":                   n,
        "Total Actual":             round(total_actual, 4),
        "Total Predicted":          round(total_predicted, 4),
        "Portfolio AMPE (%)":       round(abs_pct_err, 4) if not np.isnan(abs_pct_err) else float("nan"),
        "Portfolio MPE (%)":        round(pct_err, 4)     if not np.isnan(pct_err)     else float("nan"),
        "Portfolio Abs Error ($)":  round(total_abs_diff, 4),
        "Portfolio Mean Error ($)": round(total_diff, 4),
    }

    if n_accounts and n_accounts > 0:
        result["Per-Account Actual"]       = round(total_actual    / n_accounts, 6)
        result["Per-Account Predicted"]    = round(total_predicted / n_accounts, 6)
        result["Per-Account Abs Error ($)"]= round(total_abs_diff  / n_accounts, 6)
        result["Per-Account Error ($)"]    = round(total_diff       / n_accounts, 6)

    return result


def _metrics_block(actual: np.ndarray, predicted: np.ndarray,
                   n_accounts: int | None, level: str) -> str:
    lines = []
    if level in ("account", "both"):
        m = _compute_account_level(actual, predicted)
        lines.append("  [Account level]")
        for k, v in m.items():
            lines.append(f"    {k}: {_fmt(v)}")
    if level in ("portfolio", "both"):
        m = _compute_portfolio_level(actual, predicted, n_accounts)
        lines.append("  [Portfolio level]")
        for k, v in m.items():
            lines.append(f"    {k}: {_fmt(v)}")
    return "\n".join(lines)


def _parse_pairs(target_pairs: str) -> list[tuple[str, str]]:
    pairs = []
    for token in target_pairs.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"Invalid target pair '{token}'. "
                "Format: actual_col:predicted_col"
            )
        a, p = token.split(":", 1)
        pairs.append((a.strip(), p.strip()))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Public tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculate_backtesting_metrics(
    target_pairs: str,
    dataset_name: str = "main",
    group_by_columns: str = "",
    level: str = "both",
    n_accounts_column: str = "",
) -> str:
    """Calculate CCAR model backtesting metrics for one or more target variables.

    Metrics computed
    ----------------
    Account level  : AMPE, MPE, Abs Error per row, Mean Error per row, RMSE, R², Gini, KS
    Portfolio level: Portfolio AMPE/MPE, total & per-account dollar errors

    Error sign convention: error = predicted - actual
    MPE > 0  → model over-predicts  (predicted > actual)
    MPE < 0  → model under-predicts (predicted < actual)

    Args:
        target_pairs: One or more "actual_col:predicted_col" pairs, comma-separated.
                      Example: "actual_pd:pred_pd"
                      Example: "actual_pd:pred_pd,actual_lgd:pred_lgd,actual_ead:pred_ead"
        dataset_name: Name of the loaded/aggregated dataset. Defaults to "main".
        group_by_columns: Comma-separated columns to segment results by
                          (e.g. "product_type,statement_month"). Leave empty for overall only.
        level: Which measurement level to show — "account", "portfolio", or "both". Defaults to "both".
        n_accounts_column: Column holding the number of accounts per row (produced by
                           aggregation tools as "n_accounts"). Used to compute per-account
                           dollar metrics at portfolio level. Leave empty if not available.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    # Parse target pairs
    try:
        pairs = _parse_pairs(target_pairs)
    except ValueError as e:
        return f"ERROR: {e}"

    if not pairs:
        return "ERROR: target_pairs is empty. Provide at least one 'actual:predicted' pair."

    # Validate all columns exist
    for actual_col, pred_col in pairs:
        for col in (actual_col, pred_col):
            if col not in df.columns:
                return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    n_acc_col = n_accounts_column.strip() or None
    if n_acc_col and n_acc_col not in df.columns:
        return f"ERROR: n_accounts_column '{n_acc_col}' not found. Available: {list(df.columns)}"

    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    for col in group_cols:
        if col not in df.columns:
            return f"ERROR: Group column '{col}' not found. Available: {list(df.columns)}"

    lines = [f"=== Backtesting Metrics — Dataset: '{dataset_name}' | Level: {level} ===\n"]

    for actual_col, pred_col in pairs:
        lines.append(f"{'─'*60}")
        lines.append(f"Target: {actual_col}  vs  {pred_col}")
        lines.append(f"{'─'*60}")

        df_pair = df[[actual_col, pred_col]
                     + (group_cols)
                     + ([n_acc_col] if n_acc_col else [])].dropna(subset=[actual_col, pred_col])

        actual    = df_pair[actual_col].values.astype(float)
        predicted = df_pair[pred_col].values.astype(float)
        n_acc     = int(df_pair[n_acc_col].sum()) if n_acc_col else None

        lines.append("\n── Overall ──")
        lines.append(_metrics_block(actual, predicted, n_acc, level))

        if group_cols:
            lines.append(f"\n── Segmented by: {', '.join(group_cols)} ──")

            # Build a compact summary table (account-level metrics only for readability)
            rows = []
            for key, grp in df_pair.groupby(group_cols):
                key_label = key if isinstance(key, str) else " | ".join(str(k) for k in key)
                act = grp[actual_col].values.astype(float)
                prd = grp[pred_col].values.astype(float)
                n_a = int(grp[n_acc_col].sum()) if n_acc_col else None

                row = {"segment": key_label}

                if level in ("account", "both"):
                    m = _compute_account_level(act, prd)
                    row["n_rows"]       = m.get("n_rows")
                    row["AMPE (%)"]     = _fmt(m.get("AMPE (%)"))
                    row["MPE (%)"]      = _fmt(m.get("MPE (%)"))
                    row["Abs Err/row"]  = _fmt(m.get("Abs Error (per row)"))
                    row["Mean Err/row"] = _fmt(m.get("Mean Error (per row)"))
                    row["RMSE"]         = _fmt(m.get("RMSE"))
                    row["R2"]           = _fmt(m.get("R2"))
                    row["Gini"]         = _fmt(m.get("Gini"))

                if level in ("portfolio", "both"):
                    mp = _compute_portfolio_level(act, prd, n_a)
                    row["Port AMPE (%)"] = _fmt(mp.get("Portfolio AMPE (%)"))
                    row["Port MPE (%)"]  = _fmt(mp.get("Portfolio MPE (%)"))
                    row["Port Err ($)"]  = _fmt(mp.get("Portfolio Mean Error ($)"))
                    if n_acc_col:
                        row["Acct Err ($)"] = _fmt(mp.get("Per-Account Error ($)"))

                rows.append(row)

            result_df = pd.DataFrame(rows).set_index("segment")
            lines.append(result_df.to_string())

        lines.append("")

    return "\n".join(lines)
