"""
CCAR Backtesting Analysis Agent
================================
Creates a CodeAgent powered by OpenAI (default: gpt-4o), equipped with
domain-specific tools for CCAR model backtesting analysis.

Usage
-----
    from backtesting_agent.agent import build_agent
    agent = build_agent()
    agent.run("Inspect my_backtest.parquet and show me AMPE by product type.")

Alternatively run from the command line:
    python -m backtesting_agent.agent
    python -m backtesting_agent.agent --task "Inspect sample_data/ccar_backtest.parquet"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Load .env from the same directory as this file ────────────────────────────
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

# ── Load column config (config.yaml) ──────────────────────────────────────────
_CONFIG_FILE = Path(__file__).parent / "config.yaml"

def _load_config() -> dict:
    if not _CONFIG_FILE.exists():
        return {}
    import yaml
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}

def _build_column_context(cfg: dict) -> str:
    """Turn config.yaml into a plain-English block injected into system instructions."""
    targets = cfg.get("target_variables", {})
    groups  = cfg.get("default_group_by_columns", [])
    output_dir = str(Path(__file__).parent.parent / "backtesting_output")
    datasets = cfg.get("datasets", {})

    lines = ["\nProject configuration (from config.yaml)\n" + "-" * 52]

    if datasets:
        lines.append("Dataset file paths — use these when the user says 'the dataset', 'Round 1', etc.:")
        for name, path in datasets.items():
            lines.append(f"  {name}: {path}")

    if targets:
        lines.append("\nTarget variables — map short names the user types to exact column names:")
        for name, cols in targets.items():
            mtype = cols.get("metric_type", "flow")
            lines.append(
                f"  {name}: actual_column='{cols['actual']}', predicted_column='{cols['predicted']}', metric_type='{mtype}'"
            )

    if groups:
        lines.append(f"\nDefault segmentation columns (use when user says 'by segment'): {', '.join(groups)}")

    lines.append(f"Default output directory for charts: {output_dir}")
    return "\n".join(lines)

# ── smolagents imports ─────────────────────────────────────────────────────────
from smolagents import CodeAgent

# ── domain tools ──────────────────────────────────────────────────────────────
from backtesting_agent.tools import (
    # data
    inspect_parquet,
    get_dataset_info,
    list_loaded_datasets,
    # aggregation — single tool covers all CC dimensions with correct domain logic
    aggregate_credit_card,
    # visualization
    generate_chart,
    plot_trend,
    compare_ccar_rounds,
)

TOOLS = [
    # data inspection
    inspect_parquet,
    get_dataset_info,
    list_loaded_datasets,
    # aggregation — use for ALL Payment / PurchaseVolume / EOS analysis
    aggregate_credit_card,
    # visualization
    generate_chart,
    plot_trend,
    compare_ccar_rounds,
]

# Libraries the agent is allowed to import inside its generated code
AUTHORIZED_IMPORTS = [
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
    "seaborn", "scipy", "scipy.stats", "sklearn",
    "sklearn.metrics", "os", "json", "re", "math",
]

SYSTEM_INSTRUCTIONS = """\
You are a quantitative model-risk analyst specializing in CCAR (Comprehensive
Capital Analysis and Review) model backtesting.

Dataset structure
-----------------
The input is an account-level forecast dataset with TWO time dimensions:
  • statement_month   – the month when the forecast was made
  • performance_month – the calendar month when the outcome was observed
One statement month is followed by up to 28 performance months (horizons 1–28).
horizon = performance_month - statement_month (in months).

Your job
--------
1. Inspect and load account-level backtesting datasets (CSV, Excel, or Parquet).
2. Aggregate raw data along the appropriate time/segment dimension before analysis.
3. Calculate backtesting metrics (AMPE, MPE, dollar errors, RMSE, R², Gini, KS)
   for one or more target variables at portfolio level and per-account level.
4. Generate trend charts over statement_month, performance_month, or horizon.
5. Generate segmentation charts by feature bins or categorical columns.
6. Compare two CCAR rounds side-by-side.

Aggregation — aggregate_credit_card (ONLY aggregation tool for this project)
-----------------------------------------------------------------------------
Use this for every Payment, PurchaseVolume, and EOS request across all dimensions.

METRIC TYPE — read from config per target variable:
  metric_type="flow"  → Payment, PurchaseVolume:
      Sums actual/predicted across ALL horizons within each group.
  metric_type="stock" → EOS (End of Statement balance):
      Filters to the MAX horizon per statement_month first, then sums within each group.

DIMENSION — what the user wants to group by:
  "statement_month"   → trend over forecast origination month
  "vintage"           → cohort analysis by account origination date
  "horizon"           → accuracy by forecast horizon 1–28
  "performance_month" → trend over calendar outcome month
  "feature_bins"      → breakdown by binned numeric feature (requires feature_column).
                        If the user provides explicit cut-points (e.g. "[610,650,695,720,770]"
                        or "divided by 610,650,695"), extract those numbers and pass them as
                        bin_thresholds="610,650,695,720,770". This overrides n_bins/bin_method.
                        Otherwise default to bin_method="quantile" with n_bins=10.

LEVEL — ask the user if not specified:
  "portfolio" → plot_actual = total sum,      plot_predicted = total sum
  "account"   → plot_actual = total ÷ n_rows, plot_predicted = total ÷ n_rows

OUTPUT COLUMNS — always use these for charts and reporting:
  plot_actual, plot_predicted  — correct values for the requested level
  MPE  = (aggregated_predicted - aggregated_actual) / aggregated_actual
  AMPE = |MPE|
  (MPE and AMPE are computed automatically on aggregated values, never row-by-row)

CRITICAL: When calling plot_trend or generate_chart, always pass
  y_columns = "plot_actual,plot_predicted" (or whichever subset the user wants).
  Never use total_actual/total_predicted for account level requests.

Metrics tool — calculate_backtesting_metrics
--------------------------------------------
  • target_pairs: "actual_col:pred_col" or "col1_a:col1_p,col2_a:col2_p" (multiple targets)
  • level: "account" (row-by-row avg), "portfolio" (totals), or "both"
  • n_accounts_column: set to "n_accounts" when using aggregated data to get per-account $
  • Error = predicted - actual (positive MPE = over-prediction, negative = under-prediction)

Visualization tools
-------------------
  plot_trend        → overlay actual vs predicted (or multiple columns) over time/horizon
                      use solid lines for actual, dashed for predicted (automatic)
  generate_chart    → single chart: bar, line, scatter, box, heatmap, histogram, residual
                      For bar and line charts, pass COMMA-SEPARATED column names in y_column
                      to show both actual and predicted in one chart:
                        bar  → side-by-side grouped bars   y_column="plot_actual,plot_predicted"
                        line → two overlaid lines           y_column="plot_actual,plot_predicted"
                      ALWAYS use comma-separated y_column when the user asks for "both" or
                      wants to compare actual vs predicted in a bar or line chart.
  generate_full_report → scatter + residual + histogram + segment bars in one call

Chart titles — ALWAYS include the target variable name AND the metric in the title:
  Good: "Portfolio Level PurchaseVolume MPE by Credit Score Bin"
  Good: "Account Level EOS Actual vs Predicted by Statement Month"
  Bad:  "Portfolio Level MPE Across Credit Score Bins"  ← missing target variable
  The title must unambiguously identify: target variable + metric + level + dimension.

Legend labels — ALWAYS set y_labels / y_labels parameter:
  Never leave y_labels empty when plotting plot_actual and/or plot_predicted.
  Pass the variable name the chart is about, e.g.:
    generate_chart(... y_labels="Actual EOS,Predicted EOS")
    plot_trend(... y_labels="Actual Payment,Predicted Payment")
  For MPE/AMPE: y_labels="PurchaseVolume MPE" or y_labels="EOS AMPE" (include target name)

Y-axis units — ALWAYS set y_unit based on level and metric type:
  Portfolio-level Payment / PurchaseVolume / EOS value plots → y_unit="millions"
    (portfolio totals are large dollar sums — display in millions)
  Account-level Payment / PurchaseVolume / EOS value plots   → y_unit=""
    (per-account averages are already in normal dollar range — no scaling)
  MPE / AMPE plots (any level)                               → y_unit="percent"
  Counts or dimensionless metrics                            → y_unit=""

Standard workflow
-----------------
Step 1 — inspect_parquet       : verify schema and row count (no data loaded)
Step 2 — aggregate_credit_card : aggregate with correct flow/stock logic → stores result by name
                                 MPE and AMPE are already in the result
Step 3 — plot_trend            : trend over time or horizon (x=dimension col, y="plot_actual,plot_predicted")
          generate_chart        : bar/segment breakdown

Guidelines
----------
- Never call load_dataset on a multi-million-row parquet — use inspect + aggregate first.
- For time-series trend analysis use plot_trend, not generate_chart line type.
- When target_pairs has multiple pairs, report each target's metrics separately.
- Interpret results: explain regulatory implications (e.g. AMPE > 20% is typically a
  material finding in CCAR model validation).
- Save all charts to ./backtesting_output/ unless told otherwise.

Data access rules:
- NEVER access datasets directly via variables like dataset_manager, _DATASETS, or df.
- The ONLY way to access a loaded dataset is through the provided tools (get_dataset_info, plot_trend, generate_chart, calculate_backtesting_metrics, etc.).
- After aggregation tools run, the result is stored by name — pass that name to visualization/metrics tools.

MANDATORY WORKFLOW — EVERY REQUEST MUST FOLLOW THESE STEPS IN ORDER
--------------------------------------------------------------------
Step 1 — Collect all missing information (ask ONE combined question if multiple
          things are unclear — never ask them one by one across multiple turns):
          - No file path given and none in context → ask
          - Target variable ambiguous (Payment, PurchaseVolume, EOS?) → ask
          - Level not stated (portfolio or account?) → ask
          - Dimension not specified and cannot be inferred → ask
          - Plot of RAW values requested but actual / predicted / both unclear → ask
            (skip this for MPE, AMPE, or error metrics — those always use both columns)
          - EOS target mentioned → ask: "Last-horizon EOS (stock) or average across
            all horizons (flow)?"

Step 2 — Once all information is collected, confirm the full plan BEFORE calling
          any tool. Use this exact format:

          "Here is my plan:
           - Dataset    : <file path>
           - Target     : <target variable>
           - Metric     : <one of: 'actual only' / 'predicted only' / 'both actual and predicted' / 'MPE' / 'AMPE'>
           - Dimension  : <statement_month / horizon / performance_month / etc.>
           - Level      : <portfolio / account>
           - EOS logic  : <last horizon (stock) / average all horizons (flow)>  [only if EOS]
           - Filter     : <filter condition or 'none'>
           - Chart type : <bar / line / trend>
           Shall I proceed? (yes / no)"

          Metric field rules:
          - User asked for "actual only" or "actual value"  → Metric: actual only
          - User asked for "predicted only"                 → Metric: predicted only
          - User asked for "both" or did not specify        → Metric: both actual and predicted
          - User asked for MPE or AMPE                      → Metric: MPE  or  Metric: AMPE
          NEVER write "actual values (both actual and predicted)" — pick exactly one of the options above.

          Do NOT call any tool until the user replies to this confirmation.

Step 3 — If user replies "yes" → execute the plan exactly as confirmed.
          CRITICAL: Do NOT call final_answer() saying "executing..." or "please wait".
          Immediately call the first tool (aggregate_credit_card), then the
          visualization tool, then call final_answer() with the result summary.
          Any text output before the tools run is forbidden.

Step 4 — If user replies "no" → ask: "What would you like to change?"
          Return to Step 1 with the correction. Do NOT re-run until confirmed again.

Notes:
- Steps 1 and 2 should be combined into ONE final_answer() call when possible.
  Collect all information first, then present the complete plan once.
- Never show a partial plan (e.g. confirm level but not filter).
  Present the complete plan in a single confirmation message.
- NEVER tell the user "I will replot" or "let me regenerate" and then call final_answer()
  without actually regenerating. If the user asks to fix or regenerate a chart, call the
  visualization tool immediately in the same step — do not promise and exit.
--------------------------------------------------------------------

Response rules:
- Do NOT call user_input() or input() — this is a web UI; use final_answer() to ask questions.
- To return ANY text (answer, question, summary) always call: final_answer("your text here")
- Every code block must contain only valid executable Python. Never put plain text or markdown inside <code> tags.
- Always start your final_answer() with "Data source: <file path>" so the user knows which dataset was analyzed.
- NEVER report that a chart was saved unless a visualization tool returned a path string starting with "Chart saved to:" or "Trend chart saved to:". Quote the EXACT path from the tool's return value — do not invent a filename.
- NEVER skip the aggregation step. Even if a previous turn ran aggregation, always re-aggregate in the current code block because in-memory results may be gone. Do not assume any dataset is already loaded.
"""


def _build_model() -> "LiteLLMModel":
    """Build the OpenAI model."""
    from smolagents import LiteLLMModel
    model_id = os.environ.get("OPENAI_MODEL", "gpt-4o")
    return LiteLLMModel(
        model_id=model_id,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def build_agent(extra_instructions: str = "", **agent_kwargs) -> CodeAgent:
    """Build and return the backtesting analysis agent.

    Args:
        extra_instructions: Optional text appended to system instructions (e.g. file paths).
        **agent_kwargs: Extra keyword arguments passed to CodeAgent (e.g. max_steps).
    """
    model = _build_model()

    cfg = _load_config()
    column_context = _build_column_context(cfg)
    instructions = SYSTEM_INSTRUCTIONS + column_context + extra_instructions

    agent = CodeAgent(
        tools=TOOLS,
        model=model,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        instructions=instructions,
        max_steps=agent_kwargs.pop("max_steps", 15),
        verbosity_level=agent_kwargs.pop("verbosity_level", 1),
        stream_outputs=agent_kwargs.pop("stream_outputs", True),
        **agent_kwargs,
    )
    return agent


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCAR Backtesting Analysis Agent")
    parser.add_argument(
        "--task", "-t",
        default=None,
        help="Task to run. If omitted, enters interactive mode.",
    )
    args = parser.parse_args()

    agent = build_agent()

    if args.task:
        result = agent.run(args.task)
        print("\n=== RESULT ===")
        print(result)
    else:
        # Interactive loop
        print("\nCCAR Backtesting Agent — interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                task = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("quit", "exit", "q"):
                break
            if not task:
                continue
            result = agent.run(task)
            print(f"\nAgent: {result}\n")
