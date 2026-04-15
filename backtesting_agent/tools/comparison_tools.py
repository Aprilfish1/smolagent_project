"""
CCAR round-to-round comparison tools.

Compares two backtesting datasets (e.g. CCAR 2024 vs CCAR 2025) across:
  - Portfolio-level metrics delta
  - Segment-level metric deltas
  - Visual side-by-side charts
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import tool
from .data_tools import get_df
from .metrics_tools import _compute_account_level as _compute_metrics

sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "./backtesting_output"


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, filename: str, output_dir: str) -> str:
    _ensure_output_dir(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


@tool
def compare_ccar_rounds(
    actual_column: str,
    predicted_column: str,
    dataset_name_round1: str = "round1",
    dataset_name_round2: str = "round2",
    round1_label: str = "CCAR Round 1",
    round2_label: str = "CCAR Round 2",
    group_by_columns: str = "",
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Compare backtesting performance between two CCAR rounds across metrics and segments.

    Both datasets must be loaded first (via load_dataset) and given distinct names.

    Produces:
      - A text table showing metric deltas (Round 2 - Round 1) at portfolio level
      - Segment-level metric comparison tables (if group_by_columns provided)
      - Side-by-side bar charts for each segment dimension
      - Scatter overlay chart comparing distributions

    Args:
        actual_column: Column name for observed values (must exist in both datasets).
        predicted_column: Column name for model predictions (must exist in both datasets).
        dataset_name_round1: Name of the first CCAR round dataset. Defaults to "round1".
        dataset_name_round2: Name of the second CCAR round dataset. Defaults to "round2".
        round1_label: Human-readable label for round 1 (used in charts). Defaults to "CCAR Round 1".
        round2_label: Human-readable label for round 2 (used in charts). Defaults to "CCAR Round 2".
        group_by_columns: Comma-separated column names to segment the comparison by (e.g. "product_type,vintage_year").
        output_dir: Directory to save output charts.
    """
    # Load datasets
    try:
        df1 = get_df(dataset_name_round1)
        df2 = get_df(dataset_name_round2)
    except KeyError as e:
        return f"ERROR: {e}"

    for col in [actual_column, predicted_column]:
        for label, df in [(round1_label, df1), (round2_label, df2)]:
            if col not in df.columns:
                return f"ERROR: Column '{col}' not found in '{label}'. Available: {list(df.columns)}"

    saved_charts = []
    lines = [
        f"=== CCAR Round Comparison ===",
        f"  Round 1: {round1_label} ({dataset_name_round1})  —  {df1.shape[0]:,} rows",
        f"  Round 2: {round2_label} ({dataset_name_round2})  —  {df2.shape[0]:,} rows",
        "",
    ]

    # ── Portfolio-level comparison ────────────────────────────────────────────
    m1 = _compute_metrics(
        df1[actual_column].dropna().values.astype(float),
        df1[predicted_column].dropna().values.astype(float),
    )
    m2 = _compute_metrics(
        df2[actual_column].dropna().values.astype(float),
        df2[predicted_column].dropna().values.astype(float),
    )

    lines.append("── Portfolio-Level Metrics ──")
    header = f"{'Metric':<25} {round1_label:>20} {round2_label:>20} {'Delta (R2-R1)':>20}"
    lines.append(header)
    lines.append("-" * len(header))

    numeric_metrics = ["RMSE", "MAE", "Mean_Error (bias)", "R2", "Gini", "KS"]
    for metric in numeric_metrics:
        v1 = m1.get(metric, "n/a")
        v2 = m2.get(metric, "n/a")
        try:
            delta = round(float(v2) - float(v1), 4)
            delta_str = f"{delta:+.4f}"
        except (TypeError, ValueError):
            delta_str = "n/a"
        lines.append(f"{metric:<25} {str(v1):>20} {str(v2):>20} {delta_str:>20}")

    # Portfolio-level bar chart for key metrics
    fig, axes = plt.subplots(1, len(numeric_metrics), figsize=(3 * len(numeric_metrics), 4))
    if len(numeric_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, numeric_metrics):
        v1 = m1.get(metric, float("nan"))
        v2 = m2.get(metric, float("nan"))
        try:
            vals = [float(v1), float(v2)]
        except (TypeError, ValueError):
            vals = [float("nan"), float("nan")]
        colors = ["steelblue", "darkorange"]
        ax.bar([round1_label, round2_label], vals, color=colors, edgecolor="white")
        ax.set_title(metric, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    fig.suptitle("Portfolio-Level Metric Comparison", fontweight="bold")
    fig.tight_layout()
    saved_charts.append(_save(fig, "cmp_01_portfolio_metrics.png", output_dir))

    # ── Segment-level comparison ──────────────────────────────────────────────
    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    for seg_col in group_cols:
        miss = [l for l, df in [(round1_label, df1), (round2_label, df2)] if seg_col not in df.columns]
        if miss:
            lines.append(f"\nWARNING: Column '{seg_col}' missing in: {miss} — skipped.")
            continue

        lines.append(f"\n── Segment Comparison: {seg_col} ──")

        def seg_metrics(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for key, grp in df.groupby(seg_col):
                m = _compute_metrics(
                    grp[actual_column].dropna().values.astype(float),
                    grp[predicted_column].dropna().values.astype(float),
                )
                m[seg_col] = key
                rows.append(m)
            return pd.DataFrame(rows).set_index(seg_col) if rows else pd.DataFrame()

        seg1 = seg_metrics(df1)
        seg2 = seg_metrics(df2)

        if seg1.empty or seg2.empty:
            lines.append("  (No segment data available)")
            continue

        # Align on common segments
        common = seg1.index.intersection(seg2.index)
        seg1_c = seg1.loc[common]
        seg2_c = seg2.loc[common]

        delta_df = pd.DataFrame(index=common)
        for col_ in numeric_metrics:
            if col_ in seg1_c.columns and col_ in seg2_c.columns:
                try:
                    delta_df[col_] = (
                        seg2_c[col_].astype(float) - seg1_c[col_].astype(float)
                    ).round(4)
                except Exception:
                    pass

        lines.append(f"  Segments only in {round1_label}: {list(seg1.index.difference(common))}")
        lines.append(f"  Segments only in {round2_label}: {list(seg2.index.difference(common))}")
        lines.append(f"\n  {round1_label} metrics by {seg_col}:")
        lines.append(seg1_c[numeric_metrics].to_string())
        lines.append(f"\n  {round2_label} metrics by {seg_col}:")
        lines.append(seg2_c[numeric_metrics].to_string())
        lines.append(f"\n  Delta ({round2_label} - {round1_label}) by {seg_col}:")
        lines.append(delta_df.to_string())

        # Side-by-side bar chart for RMSE & MAE by segment
        for metric in ["RMSE", "MAE", "Gini"]:
            if metric not in seg1_c.columns:
                continue
            try:
                vals1 = seg1_c[metric].astype(float)
                vals2 = seg2_c[metric].astype(float)
            except Exception:
                continue

            x = np.arange(len(common))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(8, len(common) * 0.8), 5))
            ax.bar(x - width / 2, vals1, width, label=round1_label, color="steelblue")
            ax.bar(x + width / 2, vals2, width, label=round2_label, color="darkorange")
            ax.set_xticks(x)
            ax.set_xticklabels(common, rotation=45, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by {seg_col}: {round1_label} vs {round2_label}", fontweight="bold")
            ax.legend()
            fig.tight_layout()
            saved_charts.append(_save(fig, f"cmp_{seg_col}_{metric}.png", output_dir))

    # ── Scatter overlay ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df1[predicted_column], df1[actual_column],
               alpha=0.4, label=round1_label, color="steelblue", edgecolors="none")
    ax.scatter(df2[predicted_column], df2[actual_column],
               alpha=0.4, label=round2_label, color="darkorange", edgecolors="none")
    all_vals = pd.concat([df1[[actual_column, predicted_column]], df2[[actual_column, predicted_column]]])
    mn = all_vals.min().min()
    mx = all_vals.max().max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect fit")
    ax.set_xlabel(f"Predicted ({predicted_column})")
    ax.set_ylabel(f"Actual ({actual_column})")
    ax.set_title("Actual vs Predicted Overlay", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    saved_charts.append(_save(fig, "cmp_scatter_overlay.png", output_dir))

    lines.append(f"\n── Charts saved ({len(saved_charts)}) ──")
    for p in saved_charts:
        lines.append(f"  {p}")

    return "\n".join(lines)
