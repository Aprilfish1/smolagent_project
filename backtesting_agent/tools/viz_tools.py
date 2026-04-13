"""
Visualization tools for CCAR backtesting analysis.

Supported chart types:
  bar          – grouped / stacked bar chart
  line         – line chart (time-series friendly)
  scatter      – actual vs predicted scatter
  box          – box plot by segment
  heatmap      – correlation or pivot heatmap
  histogram    – distribution of a single column
  residual     – residual plot (predicted - actual)
"""
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for agents
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import tool
from .data_tools import get_df

sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "./backtesting_output"


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, filename: str, output_dir: str) -> str:
    _ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


@tool
def generate_chart(
    chart_type: str,
    x_column: str,
    y_column: str,
    dataset_name: str = "main",
    group_by_column: str = "",
    title: str = "",
    output_dir: str = OUTPUT_DIR,
    filename: str = "",
) -> str:
    """Generate a single visualization chart from a loaded dataset.

    Supported chart_type values:
      - bar       : bar chart (x must be categorical)
      - line      : line chart (good for time series)
      - scatter   : scatter plot; use actual vs predicted columns to see fit
      - box       : box-plot distribution grouped by x_column
      - heatmap   : correlation matrix (x_column and y_column are ignored)
      - histogram : distribution of y_column (x_column ignored)
      - residual  : residual plot — x_column=predicted, y_column=actual

    Args:
        chart_type: One of bar, line, scatter, box, heatmap, histogram, residual.
        x_column: Column to use on the X axis (or grouping for box/bar).
        y_column: Column to use on the Y axis (or single column for histogram).
        dataset_name: Name of the loaded dataset. Defaults to "main".
        group_by_column: Optional column for color/hue grouping (e.g. segment).
        title: Chart title. Auto-generated if empty.
        output_dir: Directory to save the PNG. Defaults to ./backtesting_output.
        filename: Output filename (without path). Auto-generated if empty.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    chart_type = chart_type.lower().strip()
    hue = group_by_column.strip() or None

    # Validate columns
    required_cols = []
    if chart_type not in ("heatmap",):
        required_cols.append(y_column)
    if chart_type not in ("histogram", "heatmap"):
        required_cols.append(x_column)
    if hue:
        required_cols.append(hue)
    for col in required_cols:
        if col not in df.columns:
            return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    fig, ax = plt.subplots(figsize=(10, 6))
    chart_title = title or f"{chart_type.title()} — {dataset_name}"

    try:
        if chart_type == "bar":
            agg = df.groupby([x_column] + ([hue] if hue else []))[y_column].mean().reset_index()
            if hue:
                sns.barplot(data=agg, x=x_column, y=y_column, hue=hue, ax=ax)
            else:
                sns.barplot(data=agg, x=x_column, y=y_column, ax=ax)
            ax.set_xlabel(x_column)
            ax.set_ylabel(f"Mean {y_column}")
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "line":
            if hue:
                for grp_key, grp_df in df.groupby(hue):
                    sorted_grp = grp_df.sort_values(x_column)
                    ax.plot(sorted_grp[x_column], sorted_grp[y_column], marker="o", label=str(grp_key))
                ax.legend(title=hue)
            else:
                sorted_df = df.sort_values(x_column)
                ax.plot(sorted_df[x_column], sorted_df[y_column], marker="o")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "scatter":
            if hue:
                sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue, alpha=0.6, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_column, y=y_column, alpha=0.6, ax=ax)
            # Perfect-fit reference line
            mn = min(df[x_column].min(), df[y_column].min())
            mx = max(df[x_column].max(), df[y_column].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect fit")
            ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)

        elif chart_type == "box":
            if hue:
                sns.boxplot(data=df, x=x_column, y=y_column, hue=hue, ax=ax)
            else:
                sns.boxplot(data=df, x=x_column, y=y_column, ax=ax)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include="number")
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)

        elif chart_type == "histogram":
            sns.histplot(data=df, x=y_column, hue=hue, kde=True, ax=ax)
            ax.set_xlabel(y_column)

        elif chart_type == "residual":
            residuals = df[y_column] - df[x_column]   # actual - predicted
            if hue:
                scatter_kw = {"c": pd.Categorical(df[hue]).codes, "cmap": "tab10", "alpha": 0.6}
                sc = ax.scatter(df[x_column], residuals, **scatter_kw)
            else:
                ax.scatter(df[x_column], residuals, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Predicted ({x_column})")
            ax.set_ylabel(f"Residual (actual - predicted)")

        else:
            plt.close(fig)
            return f"ERROR: Unknown chart_type '{chart_type}'. Choose from: bar, line, scatter, box, heatmap, histogram, residual."

    except Exception as e:
        plt.close(fig)
        return f"ERROR generating chart: {e}"

    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = filename or f"{chart_type}_{x_column}_{y_column}_{dataset_name}.png"
    saved_path = _save(fig, fname, output_dir)
    return f"Chart saved to: {saved_path}"


@tool
def generate_full_report(
    actual_column: str,
    predicted_column: str,
    dataset_name: str = "main",
    group_by_columns: str = "",
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Generate a comprehensive visual backtesting report (multiple charts) for one dataset.

    Creates the following charts automatically:
      1. Actual vs Predicted scatter plot (portfolio level)
      2. Residual plot
      3. Residual distribution histogram
      4. If group_by_columns provided: bar chart of mean actual vs predicted per segment

    Args:
        actual_column: Column with observed values.
        predicted_column: Column with model-predicted values.
        dataset_name: Name of the loaded dataset. Defaults to "main".
        group_by_columns: Comma-separated columns for segmented bar charts (e.g. "product_type,vintage_year").
        output_dir: Directory where all PNGs will be saved.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    for col in [actual_column, predicted_column]:
        if col not in df.columns:
            return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    _ensure_output_dir(output_dir)
    saved = []

    # 1. Scatter: actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[predicted_column], df[actual_column], alpha=0.5, edgecolors="none")
    mn = min(df[predicted_column].min(), df[actual_column].min())
    mx = max(df[predicted_column].max(), df[actual_column].max())
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect fit")
    ax.set_xlabel(f"Predicted ({predicted_column})")
    ax.set_ylabel(f"Actual ({actual_column})")
    ax.set_title(f"Actual vs Predicted — {dataset_name}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    saved.append(_save(fig, f"01_scatter_{dataset_name}.png", output_dir))

    # 2. Residual plot
    residuals = df[actual_column] - df[predicted_column]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[predicted_column], residuals, alpha=0.5, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(f"Predicted ({predicted_column})")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residual Plot — {dataset_name}", fontweight="bold")
    fig.tight_layout()
    saved.append(_save(fig, f"02_residual_{dataset_name}.png", output_dir))

    # 3. Residual histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_title(f"Residual Distribution — {dataset_name}", fontweight="bold")
    fig.tight_layout()
    saved.append(_save(fig, f"03_residual_hist_{dataset_name}.png", output_dir))

    # 4. Segmented bar charts
    group_cols = [c.strip() for c in group_by_columns.split(",") if c.strip()]
    for seg_col in group_cols:
        if seg_col not in df.columns:
            continue
        agg = (
            df.groupby(seg_col)[[actual_column, predicted_column]]
            .mean()
            .reset_index()
            .melt(id_vars=seg_col, var_name="type", value_name="value")
        )
        fig, ax = plt.subplots(figsize=(max(8, len(df[seg_col].unique()) * 1.2), 5))
        sns.barplot(data=agg, x=seg_col, y="value", hue="type", ax=ax)
        ax.set_title(f"Mean Actual vs Predicted by {seg_col} — {dataset_name}", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        saved.append(_save(fig, f"04_segment_{seg_col}_{dataset_name}.png", output_dir))

    return (
        f"Full report generated for '{dataset_name}'. {len(saved)} charts saved:\n"
        + "\n".join(f"  {p}" for p in saved)
    )
