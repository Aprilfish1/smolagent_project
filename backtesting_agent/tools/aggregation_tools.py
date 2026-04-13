"""
Pre-aggregation tools for large account-level parquet datasets.

Strategy
--------
- Read and aggregate entirely in Polars (fast, memory-efficient, lazy evaluation)
- Convert the compact aggregated result to pandas and store it by name
- All downstream tools (metrics, viz, comparison) work on the stored pandas DataFrame

Supported aggregation dimensions
---------------------------------
  aggregate_by_statement_month  – group by a statement/performance date column
  aggregate_by_vintage          – group by origination cohort date
  aggregate_by_feature_bins     – bin a continuous column then group
  aggregate_parquet             – arbitrary custom group-by on any columns
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

def _read_lazy(file_path: str, columns: list[str]) -> pl.LazyFrame:
    """Open a parquet file as a Polars LazyFrame, selecting only needed columns."""
    return pl.scan_parquet(file_path).select(columns)


def _apply_filter(lf: pl.LazyFrame, row_filter: str) -> pl.LazyFrame:
    """Apply a Polars SQL-style filter expression string. Empty = no filter."""
    row_filter = row_filter.strip()
    if not row_filter:
        return lf
    try:
        return lf.filter(pl.Expr.deserialize(row_filter.encode(), format="json"))
    except Exception:
        # Fall back to pandas query on collected frame if Polars expr parse fails
        df_pd = lf.collect().to_pandas()
        try:
            df_pd = df_pd.query(row_filter)
        except Exception as e:
            raise ValueError(
                f"Filter expression failed: {e}\n"
                f"Expression: '{row_filter}'\n"
                f"Tip: use pandas query syntax, e.g. \"product_type == 'Mortgage' and vintage_year >= 2020\""
            )
        return pl.from_pandas(df_pd).lazy()


def _agg_exprs(actual: str, predicted: str, exposure: str | None) -> list[pl.Expr]:
    """Build the standard set of Polars aggregation expressions."""
    exprs = [
        pl.len().alias("n_accounts"),
        pl.col(actual).mean().alias(f"mean_{actual}"),
        pl.col(predicted).mean().alias(f"mean_{predicted}"),
    ]
    if exposure:
        # exposure-weighted mean = sum(value * weight) / sum(weight)
        exprs += [
            pl.col(exposure).sum().alias("total_exposure"),
            (
                (pl.col(actual) * pl.col(exposure)).sum()
                / pl.col(exposure).sum()
            ).alias(f"ew_{actual}"),
            (
                (pl.col(predicted) * pl.col(exposure)).sum()
                / pl.col(exposure).sum()
            ).alias(f"ew_{predicted}"),
        ]
    return exprs


def _schema_names(file_path: str) -> list[str]:
    return pl.scan_parquet(file_path).collect_schema().names()


def _check_columns(file_path: str, needed: list[str]) -> str | None:
    """Return error string if any column is missing, else None."""
    available = _schema_names(file_path)
    missing = [c for c in needed if c not in available]
    if missing:
        return f"ERROR: Columns not found in parquet: {missing}\nAvailable: {available}"
    return None


def _finish(lf: pl.LazyFrame, dataset_name: str, sort_col: str | None = None) -> pl.DataFrame:
    """Collect LazyFrame, optionally sort, and return Polars DataFrame."""
    df = lf.collect()
    if sort_col and sort_col in df.columns:
        df = df.sort(sort_col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def aggregate_parquet(
    file_path: str,
    actual_column: str,
    predicted_column: str,
    group_by_columns: str,
    dataset_name: str = "agg",
    exposure_column: str = "",
    row_filter: str = "",
) -> str:
    """Read a large account-level parquet file using Polars and aggregate by any columns.

    The file is read lazily (only needed columns, no full load into RAM).
    The aggregated result — typically a small DataFrame — is stored as a
    named pandas dataset ready for metrics and visualization tools.

    Args:
        file_path: Path to the .parquet file.
        actual_column: Column name for observed values (e.g. actual_pd).
        predicted_column: Column name for model-predicted values (e.g. predicted_pd).
        group_by_columns: Comma-separated columns to group by (e.g. "product_type,risk_segment").
        dataset_name: Name to store the aggregated result under. Defaults to "agg".
        exposure_column: Column for exposure/EAD weighting. Leave empty to skip.
        row_filter: Optional pandas-style filter string applied before aggregation
                    (e.g. "product_type == 'Mortgage' and vintage_year >= 2020").
    """
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    if not group_cols:
        return "ERROR: group_by_columns must not be empty."

    exposure_col = exposure_column.strip() or None
    needed = list(dict.fromkeys(
        group_cols + [actual_column, predicted_column]
        + ([exposure_col] if exposure_col else [])
    ))

    err = _check_columns(file_path, needed)
    if err:
        return err

    try:
        lf = _read_lazy(file_path, needed)
        lf = _apply_filter(lf, row_filter)
        agg_lf = lf.group_by(group_cols).agg(_agg_exprs(actual_column, predicted_column, exposure_col))
        agg_df = _finish(agg_lf, dataset_name).sort(group_cols)
        _store(dataset_name, agg_df.to_pandas())
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR during aggregation: {e}"

    preview = agg_df.head(5).to_pandas().to_string(index=False)
    return (
        f"Aggregated '{file_path}' → dataset '{dataset_name}'\n"
        f"Shape: {agg_df.shape[0]:,} groups × {agg_df.shape[1]} columns\n"
        f"Columns: {agg_df.columns}\n\n"
        f"First 5 rows:\n{preview}"
    )


@tool
def aggregate_by_statement_month(
    file_path: str,
    actual_column: str,
    predicted_column: str,
    date_column: str,
    dataset_name: str = "agg_monthly",
    period: str = "month",
    extra_group_columns: str = "",
    exposure_column: str = "",
    row_filter: str = "",
) -> str:
    """Read a large parquet file with Polars and aggregate by statement date period.

    Groups by the truncated statement/performance date (month, quarter, or year),
    plus any extra categorical columns. Good for time-series views of model performance.

    Args:
        file_path: Path to the .parquet file.
        actual_column: Column with observed values.
        predicted_column: Column with model predictions.
        date_column: Name of the statement/performance date column.
        dataset_name: Name to store the result under. Defaults to "agg_monthly".
        period: Time granularity — "month", "quarter", or "year". Defaults to "month".
        extra_group_columns: Comma-separated additional columns to group by (e.g. "product_type").
        exposure_column: Column for exposure weighting (optional).
        row_filter: Optional pandas-style filter string applied before aggregation.
    """
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    period = period.lower().strip()
    if period not in ("month", "quarter", "year"):
        return "ERROR: period must be 'month', 'quarter', or 'year'."

    extra_cols = [c.strip() for c in extra_group_columns.split(",") if c.strip()]
    exposure_col = exposure_column.strip() or None
    period_col = f"statement_{period}"

    needed = list(dict.fromkeys(
        [date_column, actual_column, predicted_column]
        + extra_cols + ([exposure_col] if exposure_col else [])
    ))
    err = _check_columns(file_path, needed)
    if err:
        return err

    try:
        lf = _read_lazy(file_path, needed)
        lf = _apply_filter(lf, row_filter)

        # Truncate date to period
        if period == "month":
            lf = lf.with_columns(
                pl.col(date_column).cast(pl.Date).dt.truncate("1mo").alias(period_col)
            )
        elif period == "quarter":
            lf = lf.with_columns(
                pl.col(date_column).cast(pl.Date).dt.truncate("1q").alias(period_col)
            )
        elif period == "year":
            lf = lf.with_columns(
                pl.col(date_column).cast(pl.Date).dt.year().alias(period_col)
            )

        group_cols = [period_col] + extra_cols
        agg_lf = lf.group_by(group_cols).agg(_agg_exprs(actual_column, predicted_column, exposure_col))
        agg_df = _finish(agg_lf, dataset_name, sort_col=period_col)
        _store(dataset_name, agg_df.to_pandas())
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR during aggregation: {e}"

    preview = agg_df.head(5).to_pandas().to_string(index=False)
    period_vals = agg_df[period_col]
    return (
        f"Statement-{period} aggregation complete → dataset '{dataset_name}'\n"
        f"Shape: {agg_df.shape[0]:,} periods × {agg_df.shape[1]} columns\n"
        f"Date range: {period_vals.min()} → {period_vals.max()}\n\n"
        f"First 5 rows:\n{preview}"
    )


@tool
def aggregate_by_vintage(
    file_path: str,
    actual_column: str,
    predicted_column: str,
    origination_date_column: str,
    dataset_name: str = "agg_vintage",
    vintage_period: str = "quarter",
    extra_group_columns: str = "",
    exposure_column: str = "",
    row_filter: str = "",
) -> str:
    """Read a large parquet file with Polars and aggregate by origination vintage cohort.

    Vintage = truncated origination date (month, quarter, or year). This is the
    standard CCAR approach: compare actual vs predicted performance across cohorts
    grouped by when accounts were originated.

    Args:
        file_path: Path to the .parquet file.
        actual_column: Column with observed values.
        predicted_column: Column with model predictions.
        origination_date_column: Column containing account origination/booking date.
        dataset_name: Name to store the result under. Defaults to "agg_vintage".
        vintage_period: Cohort granularity — "month", "quarter", or "year". Defaults to "quarter".
        extra_group_columns: Additional columns to cross-cut vintages by (e.g. "product_type").
        exposure_column: Column for exposure weighting (optional).
        row_filter: Optional pandas-style filter string applied before aggregation.
    """
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    vintage_period = vintage_period.lower().strip()
    if vintage_period not in ("month", "quarter", "year"):
        return "ERROR: vintage_period must be 'month', 'quarter', or 'year'."

    extra_cols = [c.strip() for c in extra_group_columns.split(",") if c.strip()]
    exposure_col = exposure_column.strip() or None
    vintage_col = f"vintage_{vintage_period}"

    needed = list(dict.fromkeys(
        [origination_date_column, actual_column, predicted_column]
        + extra_cols + ([exposure_col] if exposure_col else [])
    ))
    err = _check_columns(file_path, needed)
    if err:
        return err

    try:
        lf = _read_lazy(file_path, needed)
        lf = _apply_filter(lf, row_filter)

        if vintage_period == "month":
            lf = lf.with_columns(
                pl.col(origination_date_column).cast(pl.Date).dt.truncate("1mo").alias(vintage_col)
            )
        elif vintage_period == "quarter":
            lf = lf.with_columns(
                pl.col(origination_date_column).cast(pl.Date).dt.truncate("1q").alias(vintage_col)
            )
        elif vintage_period == "year":
            lf = lf.with_columns(
                pl.col(origination_date_column).cast(pl.Date).dt.year().alias(vintage_col)
            )

        group_cols = [vintage_col] + extra_cols
        agg_lf = lf.group_by(group_cols).agg(_agg_exprs(actual_column, predicted_column, exposure_col))
        agg_df = _finish(agg_lf, dataset_name, sort_col=vintage_col)
        _store(dataset_name, agg_df.to_pandas())
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR during aggregation: {e}"

    preview = agg_df.head(5).to_pandas().to_string(index=False)
    vintage_vals = agg_df[vintage_col]
    return (
        f"Vintage-{vintage_period} aggregation complete → dataset '{dataset_name}'\n"
        f"Shape: {agg_df.shape[0]:,} cohorts × {agg_df.shape[1]} columns\n"
        f"Vintage range: {vintage_vals.min()} → {vintage_vals.max()}\n\n"
        f"First 5 rows:\n{preview}"
    )


@tool
def aggregate_by_feature_bins(
    file_path: str,
    actual_column: str,
    predicted_column: str,
    bin_column: str,
    dataset_name: str = "agg_bins",
    n_bins: int = 10,
    bin_method: str = "quantile",
    extra_group_columns: str = "",
    exposure_column: str = "",
    row_filter: str = "",
) -> str:
    """Read a large parquet file with Polars and aggregate by binning a continuous feature.

    Bins the specified continuous column into n_bins buckets (quantile or equal-width),
    then aggregates actual vs predicted per bin. Useful for score-band, LTV tier,
    balance bucket, or any numeric segmentation analysis.

    Args:
        file_path: Path to the .parquet file.
        actual_column: Column with observed values.
        predicted_column: Column with model predictions.
        bin_column: Continuous column to bin (e.g. credit_score, ltv_ratio, balance).
        dataset_name: Name to store the result under. Defaults to "agg_bins".
        n_bins: Number of bins to create. Defaults to 10.
        bin_method: "quantile" for equal-frequency bins or "equal_width" for equal-range bins. Defaults to "quantile".
        extra_group_columns: Additional categorical columns to cross-cut bins by.
        exposure_column: Column for exposure weighting (optional).
        row_filter: Optional pandas-style filter string applied before aggregation.
    """
    if not os.path.exists(file_path):
        return f"ERROR: File not found: {file_path}"

    bin_method = bin_method.lower().strip()
    if bin_method not in ("quantile", "equal_width"):
        return "ERROR: bin_method must be 'quantile' or 'equal_width'."

    extra_cols = [c.strip() for c in extra_group_columns.split(",") if c.strip()]
    exposure_col = exposure_column.strip() or None
    bin_label_col = f"{bin_column}_bin"

    needed = list(dict.fromkeys(
        [bin_column, actual_column, predicted_column]
        + extra_cols + ([exposure_col] if exposure_col else [])
    ))
    err = _check_columns(file_path, needed)
    if err:
        return err

    try:
        # Collect only needed columns — binning requires full column for quantile computation
        df = pl.scan_parquet(file_path).select(needed).collect()

        if bin_method == "quantile":
            df = df.with_columns(
                pl.col(bin_column).qcut(n_bins, labels=[str(i) for i in range(1, n_bins + 1)],
                                        allow_duplicates=True).alias(bin_label_col)
            )
        else:
            df = df.with_columns(
                pl.col(bin_column).cut(n_bins, labels=[str(i) for i in range(1, n_bins + 1)]).alias(bin_label_col)
            )

        group_cols = [bin_label_col] + extra_cols
        agg_df = (
            df.group_by(group_cols)
            .agg(
                _agg_exprs(actual_column, predicted_column, exposure_col)
                + [pl.col(bin_column).mean().round(4).alias(f"{bin_column}_midpoint")]
            )
            .sort(bin_label_col)
        )
        _store(dataset_name, agg_df.to_pandas())
    except ValueError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR during aggregation: {e}"

    preview = agg_df.to_pandas().to_string(index=False)
    return (
        f"Feature-bin aggregation on '{bin_column}' ({bin_method}, {n_bins} bins) "
        f"→ dataset '{dataset_name}'\n"
        f"Shape: {agg_df.shape[0]:,} bins × {agg_df.shape[1]} columns\n\n"
        f"{preview}"
    )
