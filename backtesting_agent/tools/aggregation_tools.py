"""
Credit card balance sheet backtesting — aggregation tool.

Single entry point: aggregate_credit_card()

Supports all analysis dimensions (statement_month, vintage, horizon,
performance_month, feature_bins) with correct domain logic:
  - flow  (Payment, PurchaseVolume): sum across ALL horizons
  - stock (EOS): filter to MAX horizon per statement_month, then sum
"""

from __future__ import annotations

import os
import polars as pl
import pandas as pd
from smolagents import tool
from .data_tools import _store


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_filter(lf: pl.LazyFrame, row_filter: str) -> pl.LazyFrame:
    """Apply an optional pandas-style filter string. Empty = no filter."""
    row_filter = row_filter.strip()
    if not row_filter:
        return lf
    df_pd = lf.collect().to_pandas()
    try:
        df_pd = df_pd.query(row_filter)
    except Exception as e:
        raise ValueError(
            f"Filter expression failed: {e}\n"
            f"Expression: '{row_filter}'\n"
            "Tip: use pandas query syntax, e.g. \"product_type == 'Credit Card'\""
        )
    return pl.from_pandas(df_pd).lazy()


def _check_columns(file_path: str, needed: list[str]) -> str | None:
    """Return an error string if any column is missing from the parquet schema."""
    available = pl.scan_parquet(file_path).collect_schema().names()
    missing = [c for c in needed if c not in available]
    if missing:
        return f"ERROR: Columns not found in parquet: {missing}\nAvailable: {available}"
    return None


def _cc_filter_stock(lf: pl.LazyFrame, stmt_col: str, horizon_col: str) -> pl.LazyFrame:
    """EOS pre-filter: keep only rows at the max horizon per statement_month."""
    max_h = lf.group_by(stmt_col).agg(pl.col(horizon_col).max().alias("_max_h"))
    return (
        lf.join(max_h, on=stmt_col, how="left")
        .filter(pl.col(horizon_col) == pl.col("_max_h"))
        .drop("_max_h")
    )


def _cc_compute_metrics(df: pd.DataFrame, actual_col: str, pred_col: str, level: str) -> pd.DataFrame:
    """
    Add portfolio totals, per-account values, and MPE/AMPE to the aggregated DataFrame.

    All calculations are on AGGREGATED group sums, never on individual rows.

    Columns added:
      total_actual, total_predicted       — group sums
      per_account_actual, per_account_predicted — total ÷ n_rows
      plot_actual, plot_predicted         — the right values for the requested level:
                                            portfolio → total_*
                                            account   → per_account_*
      MPE  = (plot_predicted - plot_actual) / plot_actual
      AMPE = |MPE|
    """
    import numpy as np

    total_actual    = df[actual_col]
    total_predicted = df[pred_col]
    n               = df["n_rows"]

    df["total_actual"]          = total_actual
    df["total_predicted"]       = total_predicted
    df["per_account_actual"]    = total_actual    / n
    df["per_account_predicted"] = total_predicted / n

    if level == "account":
        plot_actual    = df["per_account_actual"]
        plot_predicted = df["per_account_predicted"]
    else:
        plot_actual    = total_actual
        plot_predicted = total_predicted

    df["plot_actual"]    = plot_actual
    df["plot_predicted"] = plot_predicted

    error      = plot_predicted - plot_actual
    df["MPE"]  = np.where(plot_actual != 0, error / plot_actual, float("nan"))
    df["AMPE"] = df["MPE"].abs()
    return df


def _cc_summary(df: pd.DataFrame, dataset_name: str, metric_type: str,
                dimension: str, level: str, group_col: str) -> str:
    """Return a human-readable result summary."""
    metric_desc = "sum across all horizons" if metric_type == "flow" else "last horizon per stmt_month"
    try:
        range_str = f"{df[group_col].min()} → {df[group_col].max()}"
    except Exception:
        range_str = "n/a"
    plot_actual_src    = "per_account_actual"    if level == "account" else "total_actual"
    plot_predicted_src = "per_account_predicted" if level == "account" else "total_predicted"
    return (
        f"Credit card aggregation complete → dataset '{dataset_name}'\n"
        f"  dimension   : {dimension}\n"
        f"  metric_type : {metric_type} ({metric_desc})\n"
        f"  level       : {level}\n"
        f"  groups      : {len(df)}  (range: {range_str})\n"
        f"  columns     : {list(df.columns)}\n"
        f"  Mean MPE    : {df['MPE'].mean():.4f}\n"
        f"  Mean AMPE   : {df['AMPE'].mean():.4f}\n"
        f"\nColumn guide for charts and reporting:\n"
        f"  x-axis      : '{group_col}'\n"
        f"  actual      : 'plot_actual'    (= {plot_actual_src})\n"
        f"  predicted   : 'plot_predicted' (= {plot_predicted_src})\n"
        f"  error metr. : 'MPE', 'AMPE'\n"
        f"\nFirst 5 rows:\n{df.head().to_string(index=False)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def aggregate_credit_card(
    file_path: str,
    actual_column: str,
    predicted_column: str,
    metric_type: str,
    dimension: str,
    dataset_name: str,
    level: str = "portfolio",
    statement_month_column: str = "statement_month",
    horizon_column: str = "horizon",
    performance_month_column: str = "performance_month",
    vintage_column: str = "origination_date",
    vintage_period: str = "year",
    feature_column: str = "",
    n_bins: int = 10,
    bin_method: str = "quantile",
    bin_thresholds: str = "",
    row_filter: str = "",
) -> str:
    """Aggregate credit card backtesting data with domain-specific flow/stock logic.

    METRIC TYPE:
      "flow"  — Payment, PurchaseVolume: sum actual/predicted across ALL horizons.
      "stock" — EOS: filter to MAX horizon per statement_month first, then sum.

    DIMENSION:
      "statement_month"   — group by forecast origination month
      "vintage"           — group by account origination cohort (year/quarter/month)
      "horizon"           — group by forecast horizon number (1–28)
      "performance_month" — group by calendar outcome month
      "feature_bins"      — bin a numeric column then group (requires feature_column)

    LEVEL:
      "portfolio" — plot_actual/plot_predicted = group totals
      "account"   — plot_actual/plot_predicted = group totals ÷ n_accounts

    OUTPUT always includes:
      total_actual, total_predicted     (portfolio sums)
      per_account_actual, per_account_predicted
      plot_actual, plot_predicted       (use these for ALL charts)
      MPE  = (plot_predicted - plot_actual) / plot_actual
      AMPE = |MPE|

    Args:
        file_path: Path to the .parquet file.
        actual_column: Exact column name for actual values.
        predicted_column: Exact column name for predicted values.
        metric_type: "flow" or "stock".
        dimension: "statement_month", "vintage", "horizon", "performance_month",
                   or "feature_bins".
        dataset_name: Name to store the result under for later viz tools.
        level: "portfolio" or "account". Defaults to "portfolio".
        statement_month_column: Defaults to "statement_month".
        horizon_column: Defaults to "horizon".
        performance_month_column: Defaults to "performance_month".
        vintage_column: Defaults to "origination_date".
        vintage_period: "year", "quarter", or "month". Defaults to "year".
        feature_column: Required for dimension="feature_bins".
        n_bins: Number of bins used when bin_thresholds is empty. Defaults to 10.
        bin_method: "quantile" or "equal". Used only when bin_thresholds is empty.
                    Defaults to "quantile".
        bin_thresholds: Comma-separated numeric cut-points for custom bins, e.g.
                        "610,650,695,720,770". When provided, n_bins and bin_method
                        are ignored. Values outside the range are captured in edge
                        bins (−∞ to first threshold, last threshold to +∞).
        row_filter: Optional pandas query string applied before aggregation.

    Returns:
        Summary with group range, mean MPE, mean AMPE, column guide, and first 5 rows.
    """
    VALID_DIMENSIONS = ("statement_month", "vintage", "horizon", "performance_month", "feature_bins")

    if metric_type not in ("flow", "stock"):
        return f"ERROR: metric_type must be 'flow' or 'stock', got '{metric_type}'"
    if dimension not in VALID_DIMENSIONS:
        return f"ERROR: dimension must be one of {VALID_DIMENSIONS}, got '{dimension}'"
    if level not in ("portfolio", "account"):
        return f"ERROR: level must be 'portfolio' or 'account', got '{level}'"
    if dimension == "feature_bins" and not feature_column.strip():
        return "ERROR: feature_column is required when dimension='feature_bins'"
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    # Columns needed from parquet
    needed = {actual_column, predicted_column, statement_month_column, horizon_column}
    if dimension == "vintage":
        needed.add(vintage_column)
    elif dimension == "performance_month":
        needed.add(performance_month_column)
    elif dimension == "feature_bins":
        needed.add(feature_column.strip())

    err = _check_columns(file_path, list(needed))
    if err:
        return err

    try:
        lf = pl.scan_parquet(file_path).select(list(needed))
        lf = _apply_filter(lf, row_filter)

        # EOS pre-filter: keep max-horizon row per statement_month
        if metric_type == "stock":
            lf = _cc_filter_stock(lf, statement_month_column, horizon_column)

        # Build the group-by column
        _PERIOD_MAP = {"year": "1y", "quarter": "1q", "month": "1mo"}

        if dimension == "statement_month":
            lf = lf.with_columns(
                pl.col(statement_month_column).cast(pl.Date).dt.truncate("1mo").alias("_group")
            )
            group_col, out_col = "_group", "statement_month"

        elif dimension == "vintage":
            trunc = _PERIOD_MAP.get(vintage_period, "1y")
            lf = lf.with_columns(
                pl.col(vintage_column).cast(pl.Date).dt.truncate(trunc).alias("_group")
            )
            group_col, out_col = "_group", f"vintage_{vintage_period}"

        elif dimension == "horizon":
            group_col, out_col = horizon_column, horizon_column

        elif dimension == "performance_month":
            lf = lf.with_columns(
                pl.col(performance_month_column).cast(pl.Date).dt.truncate("1mo").alias("_group")
            )
            group_col, out_col = "_group", "performance_month"

        elif dimension == "feature_bins":
            fc = feature_column.strip()
            df_collected = lf.collect().to_pandas()
            thresholds_str = bin_thresholds.strip()
            if thresholds_str:
                # Custom thresholds: parse comma-separated numbers and add ±inf edges
                import math
                cuts = sorted(float(v.strip()) for v in thresholds_str.split(",") if v.strip())
                bins = [-math.inf] + cuts + [math.inf]
                df_collected["_group"] = pd.cut(
                    df_collected[fc], bins=bins, duplicates="drop"
                ).astype(str)
            elif bin_method == "quantile":
                df_collected["_group"] = pd.qcut(
                    df_collected[fc], q=n_bins, duplicates="drop"
                ).astype(str)
            else:
                df_collected["_group"] = pd.cut(
                    df_collected[fc], bins=n_bins, duplicates="drop"
                ).astype(str)
            lf = pl.from_pandas(df_collected).lazy()
            group_col, out_col = "_group", f"{fc}_bin"

        # Aggregate
        agg_df = (
            lf.group_by(group_col)
            .agg([
                pl.col(actual_column).sum().alias("total_actual"),
                pl.col(predicted_column).sum().alias("total_predicted"),
                pl.len().alias("n_rows"),
            ])
            .sort(group_col)
            .collect()
        )

        df = agg_df.to_pandas().rename(columns={group_col: out_col})
        df = _cc_compute_metrics(df, "total_actual", "total_predicted", level)
        _store(dataset_name, df)

    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR during aggregation: {e}"

    return _cc_summary(df, dataset_name, metric_type, dimension, level, out_col)
