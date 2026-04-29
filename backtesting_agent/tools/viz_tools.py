"""
Visualization tools for CCAR backtesting analysis.

Supported chart types (generate_chart):
  bar          – grouped / stacked bar chart
  line         – line chart (time-series friendly)
  scatter      – actual vs predicted scatter
  box          – box plot by segment
  heatmap      – correlation or pivot heatmap
  histogram    – distribution of a single column
  residual     – residual plot (predicted - actual)

Dedicated tools:
  plot_trend   – overlay multiple columns over a time or horizon axis;
                 designed for statement_month / performance_month / horizon views
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as mticker
matplotlib.use("Agg")          # non-interactive backend — safe for agents
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import tool
from .data_tools import get_df

sns.set_theme(style="whitegrid", palette="muted")

# Absolute path so charts always land in the same folder regardless of
# which directory the script is launched from.
OUTPUT_DIR = str(Path(__file__).parent.parent.parent / "backtesting_output")


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig: plt.Figure, filename: str, output_dir: str) -> str:
    _ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return save_path


def _apply_y_unit(ax: plt.Axes, y_unit: str, y_label: str) -> str:
    """
    Apply y-axis unit formatting and return an updated y-axis label.

    y_unit values:
      "millions" — divide tick values by 1e6, append 'M' to label
      "percent"  — multiply tick values by 100, append '%' to label
      ""         — no transformation
    """
    y_unit = (y_unit or "").strip().lower()
    if y_unit in ("millions", "million", "m"):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.2f}M")
        )
        return (y_label + " (Millions)").strip()
    elif y_unit in ("percent", "percentage", "%", "pct"):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%")
        )
        return (y_label + " (%)").strip()
    return y_label


def _resolve_labels(y_cols: list[str], y_labels_str: str) -> list[str]:
    """
    Map internal column names to display labels.

    If y_labels_str is provided (comma-separated), use those.
    Otherwise replace 'plot_actual' → 'Actual' and 'plot_predicted' → 'Predicted'.
    """
    if y_labels_str and y_labels_str.strip():
        custom = [l.strip() for l in y_labels_str.split(",")]
        # Pad with column names if fewer labels than columns
        while len(custom) < len(y_cols):
            custom.append(y_cols[len(custom)])
        return custom[: len(y_cols)]

    result = []
    for col in y_cols:
        if col == "plot_actual":
            result.append("Actual")
        elif col == "plot_predicted":
            result.append("Predicted")
        elif col.startswith("plot_"):
            result.append(col[5:].replace("_", " ").title())
        else:
            result.append(col.replace("_", " ").title())
    return result


@tool
def generate_chart(
    chart_type: str,
    x_column: str,
    y_column: str,
    dataset_name: str = "main",
    group_by_column: str = "",
    title: str = "",
    y_labels: str = "",
    y_unit: str = "",
    output_dir: str = OUTPUT_DIR,
    filename: str = "",
) -> str:
    """Generate a single visualization chart from a loaded dataset.

    Supported chart_type values:
      - bar       : bar chart (x must be categorical). y_column accepts comma-separated
                    column names to produce grouped side-by-side bars
                    (e.g. y_column="plot_actual,plot_predicted").
      - line      : line chart (good for time series). y_column also accepts
                    comma-separated names to overlay multiple lines.
      - scatter   : scatter plot; use actual vs predicted columns to see fit
      - box       : box-plot distribution grouped by x_column
      - heatmap   : correlation matrix (x_column and y_column are ignored)
      - histogram : distribution of y_column (x_column ignored)
      - residual  : residual plot — x_column=predicted, y_column=actual

    Args:
        chart_type: One of bar, line, scatter, box, heatmap, histogram, residual.
        x_column: Column to use on the X axis (or grouping for box/bar).
        y_column: Column(s) for the Y axis. For bar/line charts, pass comma-separated
                  names to plot multiple series side-by-side or as multiple lines
                  (e.g. "plot_actual,plot_predicted").
        dataset_name: Name of the loaded dataset. Defaults to "main".
        group_by_column: Optional column for color/hue grouping (e.g. segment).
                         Not used when y_column contains multiple columns.
        title: Chart title. Auto-generated if empty.
        y_labels: Comma-separated display labels for each y series, in the same
                  order as y_column. Example: "Actual EOS,Predicted EOS".
                  Defaults to human-readable versions of column names.
        y_unit: Y-axis unit formatting. Use "millions" for dollar values
                (Payment, PurchaseVolume, EOS) or "percent" for MPE/AMPE.
                Leave empty for no transformation.
        output_dir: Directory to save the PNG. Defaults to ./backtesting_output.
        filename: Output filename (without path). Auto-generated if empty.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    chart_type = chart_type.lower().strip()
    hue = group_by_column.strip() or None

    # Support comma-separated y_columns for bar and line charts
    y_cols = [c.strip() for c in y_column.split(",") if c.strip()]
    multi_y = len(y_cols) > 1
    display_labels = _resolve_labels(y_cols, y_labels)

    # Validate columns
    required_cols = []
    if chart_type not in ("heatmap",):
        required_cols.extend(y_cols)
    if chart_type not in ("histogram", "heatmap"):
        required_cols.append(x_column)
    if hue and not multi_y:
        required_cols.append(hue)
    for col in required_cols:
        if col not in df.columns:
            return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    fig, ax = plt.subplots(figsize=(10, 6))
    chart_title = title or f"{chart_type.title()} — {dataset_name}"

    try:
        if chart_type == "bar":
            if multi_y:
                # Grouped side-by-side bars: melt multiple y columns into long form,
                # then rename the series values to display labels before plotting.
                agg = df.groupby(x_column)[y_cols].mean().reset_index()
                # Rename columns to display labels so the legend shows them directly
                rename_map = dict(zip(y_cols, display_labels))
                agg = agg.rename(columns=rename_map)
                melted = agg.melt(id_vars=x_column, value_vars=display_labels,
                                  var_name="Series", value_name="_value")
                sns.barplot(data=melted, x=x_column, y="_value", hue="Series", ax=ax)
                ax.set_xlabel(x_column)
                ax.set_ylabel(_apply_y_unit(ax, y_unit, "Mean Value"))
                ax.tick_params(axis="x", rotation=45)
            else:
                y_col = y_cols[0]
                lbl = display_labels[0]
                agg = df.groupby([x_column] + ([hue] if hue else []))[y_col].mean().reset_index()
                agg = agg.rename(columns={y_col: lbl})
                if hue:
                    sns.barplot(data=agg, x=x_column, y=lbl, hue=hue, ax=ax)
                else:
                    sns.barplot(data=agg, x=x_column, y=lbl, ax=ax)
                ax.set_xlabel(x_column)
                ax.set_ylabel(_apply_y_unit(ax, y_unit, f"Mean {lbl}"))
                ax.tick_params(axis="x", rotation=45)

        elif chart_type == "line":
            palette = plt.get_cmap("tab10").colors  # type: ignore
            if multi_y:
                sorted_df = df.sort_values(x_column)
                for i, (y_col, lbl) in enumerate(zip(y_cols, display_labels)):
                    ls = "--" if "predicted" in y_col.lower() or "pred" in y_col.lower() else "-"
                    ax.plot(sorted_df[x_column], sorted_df[y_col],
                            marker="o", markersize=3, linestyle=ls,
                            color=palette[i % len(palette)], label=lbl)
                ax.legend()
            elif hue:
                for i, (grp_key, grp_df) in enumerate(df.groupby(hue)):
                    sorted_grp = grp_df.sort_values(x_column)
                    ax.plot(sorted_grp[x_column], sorted_grp[y_cols[0]],
                            marker="o", color=palette[i % len(palette)], label=str(grp_key))
                ax.legend(title=hue)
            else:
                sorted_df = df.sort_values(x_column)
                ax.plot(sorted_df[x_column], sorted_df[y_cols[0]],
                        marker="o", label=display_labels[0])
                ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(_apply_y_unit(ax, y_unit, ", ".join(display_labels)))
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "scatter":
            y_col = y_cols[0]
            if hue:
                sns.scatterplot(data=df, x=x_column, y=y_col, hue=hue, alpha=0.6, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_column, y=y_col, alpha=0.6, ax=ax)
            mn = min(df[x_column].min(), df[y_col].min())
            mx = max(df[x_column].max(), df[y_col].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect fit")
            ax.legend()
            ax.set_xlabel(x_column)
            ax.set_ylabel(_apply_y_unit(ax, y_unit, display_labels[0]))

        elif chart_type == "box":
            y_col = y_cols[0]
            if hue:
                sns.boxplot(data=df, x=x_column, y=y_col, hue=hue, ax=ax)
            else:
                sns.boxplot(data=df, x=x_column, y=y_col, ax=ax)
            ax.set_ylabel(_apply_y_unit(ax, y_unit, display_labels[0]))
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include="number")
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)

        elif chart_type == "histogram":
            y_col = y_cols[0]
            sns.histplot(data=df, x=y_col, hue=hue, kde=True, ax=ax)
            ax.set_xlabel(_apply_y_unit(ax, y_unit, display_labels[0]))

        elif chart_type == "residual":
            y_col = y_cols[0]
            residuals = df[x_column] - df[y_col]   # predicted - actual
            if hue:
                scatter_kw = {"c": pd.Categorical(df[hue]).codes, "cmap": "tab10", "alpha": 0.6}
                ax.scatter(df[x_column], residuals, **scatter_kw)
            else:
                ax.scatter(df[x_column], residuals, alpha=0.6)
            ax.axhline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(f"Predicted ({x_column})")
            ax.set_ylabel(_apply_y_unit(ax, y_unit, "Error (predicted - actual)"))

        else:
            plt.close(fig)
            return (
                f"ERROR: Unknown chart_type '{chart_type}'. "
                "Choose from: bar, line, scatter, box, heatmap, histogram, residual."
            )

    except Exception as e:
        plt.close(fig)
        return f"ERROR generating chart: {e}"

    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    safe_y = y_column.replace(",", "_").replace(" ", "")[:40]
    fname = filename or f"{chart_type}_{x_column}_{safe_y}_{dataset_name}.png"
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

    # 2. Error plot
    residuals = df[predicted_column] - df[actual_column]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[predicted_column], residuals, alpha=0.5, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(f"Predicted ({predicted_column})")
    ax.set_ylabel("Error (predicted - actual)")
    ax.set_title(f"Error Plot — {dataset_name}", fontweight="bold")
    fig.tight_layout()
    saved.append(_save(fig, f"02_residual_{dataset_name}.png", output_dir))

    # 3. Error distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (predicted - actual)")
    ax.set_title(f"Error Distribution — {dataset_name}", fontweight="bold")
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
            .rename(columns={actual_column: "Actual", predicted_column: "Predicted"})
            .melt(id_vars=seg_col, value_vars=["Actual", "Predicted"],
                  var_name="Series", value_name="value")
        )
        fig, ax = plt.subplots(figsize=(max(8, len(df[seg_col].unique()) * 1.2), 5))
        sns.barplot(data=agg, x=seg_col, y="value", hue="Series", ax=ax)
        ax.set_title(f"Mean Actual vs Predicted by {seg_col} — {dataset_name}", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        saved.append(_save(fig, f"04_segment_{seg_col}_{dataset_name}.png", output_dir))

    return (
        f"Full report generated for '{dataset_name}'. {len(saved)} charts saved:\n"
        + "\n".join(f"  {p}" for p in saved)
    )


@tool
def plot_trend(
    x_column: str,
    y_columns: str,
    dataset_name: str = "main",
    group_by_column: str = "",
    title: str = "",
    y_label: str = "",
    y_labels: str = "",
    y_unit: str = "",
    output_dir: str = OUTPUT_DIR,
    filename: str = "",
) -> str:
    """Plot one or more columns as a trend over a time or horizon axis.

    Designed for CCAR time-dimension analysis:
      - Trend over statement_month  (how forecast accuracy changes by vintage)
      - Trend over performance_month (how accuracy changes over calendar time)
      - Trend over horizon_months   (how accuracy degrades with forecast horizon)
      - Overlaying actual vs predicted columns on the same chart

    Multiple y_columns are plotted as separate lines on the same axes, making it
    easy to compare actual vs predicted, or multiple target variables, side by side.

    Args:
        x_column: Column for the X axis (e.g. statement_month, performance_month, horizon_months).
        y_columns: Comma-separated column names to plot on the Y axis
                   (e.g. "plot_actual,plot_predicted").
        dataset_name: Name of the loaded/aggregated dataset. Defaults to "main".
        group_by_column: Optional column to split into separate lines per group
                         (e.g. "product_type" produces one line per product).
                         When set, each y_column × group value gets its own line.
        title: Chart title. Auto-generated if empty.
        y_label: Y-axis label. Auto-generated from column names if empty.
        y_labels: Comma-separated display labels for each y series, in the same
                  order as y_columns. Example: "Actual EOS,Predicted EOS".
                  Defaults to human-readable versions of column names.
        y_unit: Y-axis unit formatting. Use "millions" for dollar values
                (Payment, PurchaseVolume, EOS) or "percent" for MPE/AMPE.
                Leave empty for no transformation.
        output_dir: Directory to save the PNG. Defaults to ./backtesting_output.
        filename: Output filename. Auto-generated if empty.
    """
    try:
        df = get_df(dataset_name)
    except KeyError as e:
        return f"ERROR: {e}"

    y_cols = [c.strip() for c in y_columns.split(",") if c.strip()]
    if not y_cols:
        return "ERROR: y_columns is empty."

    for col in [x_column] + y_cols + ([group_by_column] if group_by_column else []):
        if col not in df.columns:
            return f"ERROR: Column '{col}' not found. Available: {list(df.columns)}"

    display_labels = _resolve_labels(y_cols, y_labels)

    df = df.copy()
    # Try to parse x as datetime ONLY if the column is not already numeric.
    # Numeric columns (e.g. horizon 1-28) must NOT be converted: pd.to_datetime()
    # interprets integers as nanoseconds since epoch, turning 1-28 → 1970-01-01.
    x_is_date = False
    if not pd.api.types.is_numeric_dtype(df[x_column]):
        try:
            df[x_column] = pd.to_datetime(df[x_column])
            x_is_date = True
        except Exception:
            x_is_date = False

    # Colour palette — enough distinct colours for many lines
    palette = plt.get_cmap("tab10").colors  # type: ignore

    fig, ax = plt.subplots(figsize=(12, 6))
    color_idx = 0

    group_col = group_by_column.strip() or None

    if group_col:
        groups = sorted(df[group_col].dropna().unique())
        for y_col, lbl in zip(y_cols, display_labels):
            for grp in groups:
                grp_df = df[df[group_col] == grp].sort_values(x_column)
                line_label = f"{lbl} | {grp}"
                linestyle = "--" if "predicted" in y_col.lower() or "pred" in y_col.lower() else "-"
                ax.plot(grp_df[x_column], grp_df[y_col],
                        marker="o", markersize=3,
                        linestyle=linestyle,
                        color=palette[color_idx % len(palette)],
                        label=line_label)
                color_idx += 1
    else:
        for y_col, lbl in zip(y_cols, display_labels):
            sorted_df = df.sort_values(x_column)
            linestyle = "--" if "predicted" in y_col.lower() or "pred" in y_col.lower() else "-"
            ax.plot(sorted_df[x_column], sorted_df[y_col],
                    marker="o", markersize=3,
                    linestyle=linestyle,
                    color=palette[color_idx % len(palette)],
                    label=lbl)
            color_idx += 1

    # Apply y-axis unit formatting
    axis_y_label = _apply_y_unit(ax, y_unit, y_label or ", ".join(display_labels))

    ax.set_xlabel(x_column)
    ax.set_ylabel(axis_y_label)
    ax.set_title(title or f"Trend: {', '.join(display_labels)} over {x_column} — {dataset_name}",
                 fontsize=12, fontweight="bold")

    if x_is_date:
        fig.autofmt_xdate(rotation=45)
    else:
        ax.tick_params(axis="x", rotation=45)

    # Build legend: data lines + style note for actual (solid) vs predicted (dashed)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    style_notes = [
        Line2D([0], [0], color="gray", linestyle="-",  label="solid = actual"),
        Line2D([0], [0], color="gray", linestyle="--", label="dashed = predicted"),
    ]
    ax.legend(handles=handles + style_notes,
              labels=labels + ["solid = actual", "dashed = predicted"],
              loc="best", fontsize=7)

    fig.tight_layout()
    fname = filename or f"trend_{x_column}_{dataset_name}.png"
    saved_path = _save(fig, fname, output_dir)
    return f"Trend chart saved to: {saved_path}"
