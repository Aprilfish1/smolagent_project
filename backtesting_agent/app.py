"""
CCAR Backtesting Agent — Gradio Web UI
=======================================
Two-tab layout:
  Tab 1 — Portfolio Dashboard : KPIs, period performance, EOS stats, averages, MPE trend
  Tab 2 — Analysis Chat       : interactive agent chatbox

Run with:
    python backtesting_agent/app.py
"""
from __future__ import annotations

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))

import gradio as gr
from backtesting_agent.agent import build_agent, _load_config

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_DIR  = Path(__file__).parent / "sample_data"
ROUND1_PATH = str(SAMPLE_DIR / "ccar_round1.parquet")
ROUND2_PATH = str(SAMPLE_DIR / "ccar_round2.parquet")
OUTPUT_DIR  = str(Path(__file__).parent.parent / "backtesting_output")

# ── Build agent once at startup ───────────────────────────────────────────────
_PATH_HINT = f"""
Available sample data files (tell the user these paths if they ask):
  Round 1: {ROUND1_PATH}
  Round 2: {ROUND2_PATH}

IMPORTANT: Never call user_input() or input(). If a file path is not
specified by the user, ask them which dataset they want to use.
"""
agent = build_agent(extra_instructions=_PATH_HINT)


# ── Analysis periods ──────────────────────────────────────────────────────────
PERIODS = {
    "Great Recession (2007/11 – 2009/06)": (date(2007, 11, 1), date(2009, 6, 30)),
    "Recent (2021/07 – latest)":           (date(2021, 7, 1),  None),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _scan_path(path: str) -> str:
    """Return glob pattern for partitioned parquet folders, else the path itself."""
    return str(Path(path) / "*.parquet") if Path(path).is_dir() else path


def _base_lf(sp: str, needed: list[str], row_filter: str):
    """Scan parquet, select needed columns, apply optional row filter."""
    import polars as pl
    lf = pl.scan_parquet(sp).select(needed)
    if row_filter.strip():
        df_pd = lf.collect().to_pandas()
        df_pd = df_pd.query(row_filter)
        lf = pl.from_pandas(df_pd).lazy()
    return lf


def _apply_stock_filter(lf, stmt_col: str, horizon_col: str):
    """Keep only the max-horizon row per statement_month (EOS stock logic)."""
    import polars as pl
    max_h = lf.group_by(stmt_col).agg(pl.col(horizon_col).max().alias("_max_h"))
    return (lf.join(max_h, on=stmt_col, how="left")
              .filter(pl.col(horizon_col) == pl.col("_max_h"))
              .drop("_max_h"))


def _aggregate_by_stmt(lf, stmt_col: str, actual_col: str, pred_col: str) -> pd.DataFrame:
    """Group by statement_month, sum actual/predicted, return pandas DataFrame."""
    import polars as pl
    return (
        lf.with_columns(pl.col(stmt_col).cast(pl.Date).dt.truncate("1mo").alias("_grp"))
          .group_by("_grp")
          .agg([pl.col(actual_col).sum().alias("act"),
                pl.col(pred_col).sum().alias("pred"),
                pl.len().alias("n_rows")])
          .sort("_grp")
          .collect()
          .to_pandas()
    )


def _mpe_stats(agg: pd.DataFrame, pct: bool = True) -> dict:
    """Compute MPE/AMPE stats from an aggregated DataFrame."""
    scale = 100 if pct else 1
    fmt   = (lambda v: f"{v:.2f}%") if pct else (lambda v: f"{v:.4f}")
    mpe   = np.where(agg["act"] != 0,
                     (agg["pred"] - agg["act"]) / agg["act"],
                     float("nan")) * scale
    ampe  = np.abs(mpe)
    return {
        "Stmt Months": len(agg),
        "Mean MPE":    fmt(float(np.nanmean(mpe))),
        "Mean AMPE":   fmt(float(np.nanmean(ampe))),
        "Min MPE":     fmt(float(np.nanmin(mpe))),
        "Max MPE":     fmt(float(np.nanmax(mpe))),
    }


def _fmt_millions(v: float) -> str:
    return f"${v / 1e6:,.2f}M"


def _fmt_dollars(v: float) -> str:
    return f"${v:,.2f}"


# ── Dashboard section functions ───────────────────────────────────────────────

def _resolve_primary_path(cfg: dict, path_override: str) -> str:
    """Return path_override if given, else the first dataset path from config."""
    p = path_override.strip()
    if p:
        return p
    datasets = cfg.get("datasets", {})
    return next(iter(datasets.values()), "") if datasets else ""


def _dashboard_overview(cfg: dict, path_override: str = "") -> pd.DataFrame:
    """Fast schema + date-range overview using polars lazy scan."""
    try:
        import polars as pl
    except ImportError:
        return pd.DataFrame([{"Error": "polars not installed"}])

    time_cols = cfg.get("time_columns", {})
    stmt_col  = time_cols.get("statement_month", "statement_month")

    primary = _resolve_primary_path(cfg, path_override)
    if path_override.strip():
        # User supplied a custom path — show just that one
        datasets = {"custom": path_override.strip()}
    else:
        datasets = cfg.get("datasets", {})

    if not datasets:
        return pd.DataFrame([{"Dataset": "—", "Rows": "—", "Columns": "—",
                               "Stmt Month Range": "No datasets in config.yaml"}])
    rows = []
    for name, path in datasets.items():
        if not os.path.exists(path):
            rows.append({"Dataset": name, "Rows": "—", "Columns": "—",
                         "Stmt Month Range": f"NOT FOUND: {path}"})
            continue
        try:
            sp     = _scan_path(path)
            lf     = pl.scan_parquet(sp)
            schema = lf.collect_schema()
            n_rows = lf.select(pl.len()).collect().item()
            if stmt_col in schema.names():
                d = lf.select([pl.col(stmt_col).min().alias("mn"),
                               pl.col(stmt_col).max().alias("mx")]
                              ).collect().to_pandas()
                dr = f"{d['mn'].iloc[0]} → {d['mx'].iloc[0]}"
            else:
                dr = "N/A"
            rows.append({"Dataset": name, "Rows": f"{n_rows:,}",
                         "Columns": len(schema), "Stmt Month Range": dr})
        except Exception as e:
            rows.append({"Dataset": name, "Rows": "ERROR", "Columns": "—",
                         "Stmt Month Range": str(e)[:80]})
    return pd.DataFrame(rows)


def _dashboard_performance_by_period(cfg: dict, path_override: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    MPE / AMPE per target variable for each analysis period.
    Returns (recession_df, recent_df).
    """
    try:
        import polars as pl
    except ImportError:
        err = pd.DataFrame([{"Error": "polars not installed"}])
        return err, err

    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    row_filter  = cfg.get("default_row_filter", "")
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")

    primary_path = _resolve_primary_path(cfg, path_override)
    if not primary_path or not targets:
        empty = pd.DataFrame([{"Info": "No dataset or targets configured"}])
        return empty, empty

    if not os.path.exists(primary_path):
        err = pd.DataFrame([{"Error": f"Dataset not found: {primary_path}"}])
        return err, err

    sp = _scan_path(primary_path)
    period_results: dict[str, list] = {p: [] for p in PERIODS}

    for target_name, cols in targets.items():
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")
        mtype      = cols.get("metric_type", "flow")

        needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})

        for period_label, (p_start, p_end) in PERIODS.items():
            try:
                lf = _base_lf(sp, needed, row_filter)

                # Period filter on statement_month
                lf = lf.filter(pl.col(stmt_col).cast(pl.Date) >= pl.lit(p_start))
                if p_end:
                    lf = lf.filter(pl.col(stmt_col).cast(pl.Date) <= pl.lit(p_end))

                if mtype == "stock":
                    lf = _apply_stock_filter(lf, stmt_col, horizon_col)

                agg = _aggregate_by_stmt(lf, stmt_col, actual_col, pred_col)

                if agg.empty:
                    stats = {"Stmt Months": 0, "Mean MPE": "No data",
                             "Mean AMPE": "—", "Min MPE": "—", "Max MPE": "—"}
                else:
                    stats = _mpe_stats(agg)

                period_results[period_label].append({"Target": target_name,
                                                     "Type": mtype, **stats})
            except Exception as e:
                period_results[period_label].append({
                    "Target": target_name, "Type": mtype,
                    "Stmt Months": "—", "Mean MPE": f"ERROR: {str(e)[:50]}",
                    "Mean AMPE": "—", "Min MPE": "—", "Max MPE": "—"})

    dfs = [pd.DataFrame(rows) if rows else pd.DataFrame([{"Info": "No data"}])
           for rows in period_results.values()]
    return dfs[0], dfs[1]


def _dashboard_eos_neg_stats(cfg: dict, path_override: str = "") -> pd.DataFrame:
    """
    Percentage of statement months where portfolio-level actual / predicted
    EOS balance (at last horizon) is negative.
    """
    try:
        import polars as pl
    except ImportError:
        return pd.DataFrame([{"Error": "polars not installed"}])

    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    row_filter  = cfg.get("default_row_filter", "")
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")

    stock_targets = {n: c for n, c in targets.items()
                     if c.get("metric_type") == "stock"}
    primary_path = _resolve_primary_path(cfg, path_override)
    if not stock_targets or not primary_path:
        return pd.DataFrame([{"Info": "No stock/EOS targets in config.yaml"}])

    if not os.path.exists(primary_path):
        return pd.DataFrame([{"Error": f"Dataset not found: {primary_path}"}])

    sp = _scan_path(primary_path)
    results = []

    for target_name, cols in stock_targets.items():
        actual_col = cols["actual"]
        pred_col   = cols["predicted"]
        needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})

        try:
            lf  = _base_lf(sp, needed, row_filter)
            lf  = _apply_stock_filter(lf, stmt_col, horizon_col)
            agg = _aggregate_by_stmt(lf, stmt_col, actual_col, pred_col)

            n   = len(agg)
            na  = int((agg["act"]  < 0).sum())
            np_ = int((agg["pred"] < 0).sum())

            results.append({
                "Target":                  target_name,
                "Total Stmt Months":       n,
                "% Neg Actual EOS":        f"{na / n * 100:.1f}%  ({na}/{n})" if n else "—",
                "% Neg Predicted EOS":     f"{np_ / n * 100:.1f}%  ({np_}/{n})" if n else "—",
            })
        except Exception as e:
            results.append({
                "Target":              target_name,
                "Total Stmt Months":   "—",
                "% Neg Actual EOS":    f"ERROR: {str(e)[:60]}",
                "% Neg Predicted EOS": "—",
            })

    return pd.DataFrame(results) if results else pd.DataFrame([{"Info": "No results"}])


def _dashboard_avg_values(cfg: dict, path_override: str = "") -> pd.DataFrame:
    """
    Per target variable:
      Flow  → avg across stmt months of (portfolio total, account-level mean)
              for the sum of ALL horizons per stmt_month
      Stock → avg across stmt months of (portfolio total, account-level mean)
              at the LAST horizon per stmt_month
    Values in millions for portfolio, dollars for account level.
    """
    try:
        import polars as pl
    except ImportError:
        return pd.DataFrame([{"Error": "polars not installed"}])

    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    row_filter  = cfg.get("default_row_filter", "")
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")

    primary_path = _resolve_primary_path(cfg, path_override)
    if not primary_path or not targets:
        return pd.DataFrame([{"Info": "No dataset or targets configured"}])

    if not os.path.exists(primary_path):
        return pd.DataFrame([{"Error": f"Dataset not found: {primary_path}"}])

    sp = _scan_path(primary_path)
    results = []

    for target_name, cols in targets.items():
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")
        mtype      = cols.get("metric_type", "flow")
        agg_desc   = "Last Horizon (stock)" if mtype == "stock" else "Sum All Horizons (flow)"
        needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})

        try:
            lf = _base_lf(sp, needed, row_filter)
            if mtype == "stock":
                lf = _apply_stock_filter(lf, stmt_col, horizon_col)

            agg = _aggregate_by_stmt(lf, stmt_col, actual_col, pred_col)

            agg["acct_actual"] = agg["act"]  / agg["n_rows"]
            agg["acct_pred"]   = agg["pred"] / agg["n_rows"]

            results.append({
                "Target":                   target_name,
                "Aggregation":              agg_desc,
                "Avg Portfolio Actual":     _fmt_millions(agg["act"].mean()),
                "Avg Portfolio Predicted":  _fmt_millions(agg["pred"].mean()),
                "Avg Account Actual":       _fmt_dollars(agg["acct_actual"].mean()),
                "Avg Account Predicted":    _fmt_dollars(agg["acct_pred"].mean()),
            })
        except Exception as e:
            results.append({
                "Target":                  target_name,
                "Aggregation":             agg_desc,
                "Avg Portfolio Actual":    f"ERROR: {str(e)[:50]}",
                "Avg Portfolio Predicted": "—",
                "Avg Account Actual":      "—",
                "Avg Account Predicted":   "—",
            })

    return pd.DataFrame(results) if results else pd.DataFrame([{"Info": "No results"}])


def _dashboard_trend_chart(cfg: dict, path_override: str = ""):
    """MPE trend over statement months for all targets — returns a matplotlib Figure."""
    try:
        import polars as pl
    except ImportError:
        return None

    targets     = cfg.get("target_variables", {})
    time_cols   = cfg.get("time_columns", {})
    row_filter  = cfg.get("default_row_filter", "")
    stmt_col    = time_cols.get("statement_month", "statement_month")
    horizon_col = time_cols.get("horizon", "horizon")
    perf_col    = time_cols.get("performance_month", "performance_month")

    primary_path = _resolve_primary_path(cfg, path_override)
    if not primary_path or not targets:
        return None
    if not os.path.exists(primary_path):
        return None

    sp  = _scan_path(primary_path)
    fig, ax = plt.subplots(figsize=(13, 4))

    for target_name, cols in targets.items():
        actual_col = cols.get("actual", "")
        pred_col   = cols.get("predicted", "")
        mtype      = cols.get("metric_type", "flow")
        needed = list({actual_col, pred_col, stmt_col, horizon_col, perf_col})

        try:
            lf = _base_lf(sp, needed, row_filter)
            if mtype == "stock":
                lf = _apply_stock_filter(lf, stmt_col, horizon_col)
            agg = _aggregate_by_stmt(lf, stmt_col, actual_col, pred_col)
            mpe_pct = np.where(agg["act"] != 0,
                               (agg["pred"] - agg["act"]) / agg["act"] * 100,
                               float("nan"))
            ax.plot(agg["_grp"], mpe_pct, marker="o", markersize=3,
                    linewidth=1.5, label=target_name)
        except Exception:
            continue

    # Shade analysis periods
    colors = ["#fff3cd", "#d4edda"]
    for (label, (p_start, p_end)), color in zip(PERIODS.items(), colors):
        p_end_dt = p_end or date.today()
        ax.axvspan(pd.Timestamp(p_start), pd.Timestamp(p_end_dt),
                   alpha=0.25, color=color, label=label)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Statement Month")
    ax.set_ylabel("MPE (%)")
    ax.set_title("Portfolio-Level MPE by Statement Month (all targets)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.canvas.draw()
    labels = ax.get_xticklabels()
    n = len(labels)
    if n > 18:
        step = max(1, n // 18)
        for i, lbl in enumerate(labels):
            lbl.set_visible(i % step == 0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return fig


def load_dashboard(path_override: str = ""):
    """Called when Load/Refresh is clicked — computes all dashboard sections."""
    cfg = _load_config()
    p   = path_override.strip()
    status_path = f" ({p})" if p else f" ({_resolve_primary_path(cfg, '')})"
    overview             = _dashboard_overview(cfg, p)
    rec_perf, rec_recent = _dashboard_performance_by_period(cfg, p)
    eos_stats            = _dashboard_eos_neg_stats(cfg, p)
    avg_vals             = _dashboard_avg_values(cfg, p)
    trend_fig            = _dashboard_trend_chart(cfg, p)
    return overview, rec_perf, rec_recent, eos_stats, avg_vals, trend_fig, f"✅ Dashboard loaded{status_path}"


# ── Example prompts ───────────────────────────────────────────────────────────
EXAMPLES = [
    f"Inspect the file at '{ROUND1_PATH}' and summarise its schema.",
    f"Aggregate '{ROUND1_PATH}' by statement_month (actual_pd vs predicted_pd), "
    f"store as 'r1_stmt'. Calculate metrics at portfolio and account level.",
    f"Aggregate '{ROUND1_PATH}' by horizon (actual_pd vs predicted_pd), store as "
    f"'r1_horizon'. Plot the trend of mean_actual_pd vs mean_predicted_pd over horizon_months.",
    f"Aggregate '{ROUND1_PATH}' by product_type and risk_segment (actual_pd vs predicted_pd), "
    f"store as 'r1_seg'. Generate a bar chart of mean_actual_pd by product_type.",
    f"Compare Round 1 ('{ROUND1_PATH}') and Round 2 ('{ROUND2_PATH}'): aggregate both by "
    f"horizon (actual_pd vs predicted_pd), then compare them side by side.",
]

# ── Ambiguity pre-flight check ────────────────────────────────────────────────
def _build_clarify_system() -> str:
    """Build the pre-flight system prompt, injecting known dataset names from config.yaml."""
    cfg = _load_config()
    datasets = cfg.get("datasets", {})
    targets  = cfg.get("target_variables", {})

    dataset_block = ""
    if datasets:
        lines = ["Known dataset names (treat any of these as a valid file reference):"]
        for name, path in datasets.items():
            lines.append(f"  - \"{name}\" → {path}")
        dataset_block = "\n" + "\n".join(lines)

    target_names = ", ".join(targets.keys()) if targets else "Payment, PurchaseVolume, EOS"

    return f"""You are a request parser for a backtesting analysis tool.
Review the FULL conversation and decide if any critical information is STILL unanswered.
Consider what the user has already provided in prior messages before asking anything.
{dataset_block}

The available target variables are: {target_names}.

Critical missing information means ANY of the following:
1. No dataset/file path is provided AND the user has not referenced a known dataset name.
   If the user mentions a known dataset name (e.g. "UM actmt", "round1"), treat it as resolved.
2. The target variable is ambiguous — not stated and not obvious from context.
3. The analysis level is not specified (portfolio or account?).
4. A plot of raw values is requested and it is unclear whether to show actual,
   predicted, or both — skip this if the user is asking for MPE or AMPE.

Do NOT ask about EOS stock vs flow logic — the agent will handle that.

RULES (follow strictly):
- Check ALL four points above in one pass.
- If ONE OR MORE pieces are missing, ask about ALL of them in a SINGLE combined
  message. Never ask one question now and save others for later.
- Keep the combined question under 80 words.
- If nothing is missing, reply with exactly: PROCEED"""

_CLARIFY_SYSTEM = _build_clarify_system()


def _check_ambiguity(user_message: str, history: list) -> str | None:
    try:
        import litellm
        api_key = os.environ.get("OPENAI_API_KEY", "")
        model   = os.environ.get("OPENAI_MODEL", "gpt-4o")
        messages = [{"role": "system", "content": _CLARIFY_SYSTEM}]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})
        resp = litellm.completion(model=model, messages=messages, api_key=api_key,
                                  max_tokens=150, temperature=0)
        answer = resp.choices[0].message.content.strip()
        return None if answer.upper().startswith("PROCEED") else answer
    except Exception:
        return None


def _collect_charts() -> list[str]:
    pngs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.png"), recursive=True)
    return sorted(pngs, key=os.path.getmtime, reverse=True)


_CONFIRM_WORDS = {"yes", "y", "yes.", "yes!", "yep", "proceed", "go", "go ahead",
                  "ok", "okay", "sure", "correct", "confirmed", "confirm", "do it"}


def _is_confirmation(msg: str) -> bool:
    return msg.strip().lower().rstrip(".!") in _CONFIRM_WORDS


def _build_task(user_message: str, history: list) -> str:
    prior = history[:-1]
    if not prior:
        return user_message
    recent = prior[-6:]
    lines = ["Conversation so far (use this as full context for the request below):"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"  {role}: {msg['content']}")
    lines.append(f"\nUser's latest message: {user_message}")

    if _is_confirmation(user_message):
        # Find the last confirmed plan from history
        plan_text = ""
        for m in reversed(prior):
            if m["role"] == "assistant" and "Here is my plan" in m.get("content", ""):
                plan_text = m["content"]
                break

        # Inject exact column names from config so the model doesn't guess
        cfg = _load_config()
        targets  = cfg.get("target_variables", {})
        col_info = []
        for tname, tcols in targets.items():
            col_info.append(
                f"  {tname}: actual_column='{tcols.get('actual','')}', "
                f"predicted_column='{tcols.get('predicted','')}', "
                f"metric_type='{tcols.get('metric_type','flow')}'"
            )
        col_block = "\n".join(col_info) if col_info else "  (see config.yaml)"

        return (
            f"The user confirmed the plan below. Execute it NOW using the provided tools.\n\n"
            f"{plan_text}\n\n"
            f"EXACT COLUMN NAMES (from config.yaml — use these, do not guess):\n"
            f"{col_block}\n\n"
            f"MANDATORY EXECUTION STEPS — write these exact tool calls:\n"
            f"Step 1 (code block 1):\n"
            f"  result = aggregate_credit_card(\n"
            f"      file_path=<dataset path from plan>,\n"
            f"      actual_column=<actual_column from above>,\n"
            f"      predicted_column=<predicted_column from above>,\n"
            f"      metric_type=<metric_type from above>,\n"
            f"      dimension=<dimension from plan>,\n"
            f"      dataset_name='agg_result',\n"
            f"      level=<level from plan>,\n"
            f"      row_filter=<filter from plan or empty string>,\n"
            f"  )\n"
            f"  print(result)\n\n"
            f"Step 2 (code block 2):\n"
            f"  chart_path = plot_trend(\n"
            f"      x_column=<dimension column from plan>,\n"
            f"      y_columns='plot_actual,plot_predicted',\n"
            f"      dataset_name='agg_result',\n"
            f"      title=<descriptive title>,\n"
            f"      y_unit='millions',\n"
            f"  )\n"
            f"  print(chart_path)\n\n"
            f"Step 3: final_answer('Data source: <path>\\nChart saved to: <chart_path>')\n\n"
            f"RULES:\n"
            f"- Use ONLY aggregate_credit_card and plot_trend — do NOT write raw pandas or matplotlib code.\n"
            f"- Do NOT re-present the plan. Do NOT ask 'Shall I proceed?' again.\n"
            f"- Do NOT call final_answer() before Steps 1 and 2 are complete."
        )
    else:
        lines.append(
            "\nComplete the user's latest request using all context above. "
            "Do not ask again for clarifications already answered in the conversation. "
            "However, you MUST still follow the MANDATORY WORKFLOW: present the full "
            "execution plan and ask 'Shall I proceed? (yes/no)' before calling any tool, "
            "even if all information is already clear from context."
        )
    return "\n".join(lines)


def _agent_already_engaged(history: list) -> bool:
    """True if the agent has already responded in this conversation (plan, result, etc.).
    Pre-flight clarification should only run before the agent is first invoked."""
    for m in history:
        if m["role"] != "assistant":
            continue
        c = m["content"]
        if (len(c) > 200
                or "Here is my plan" in c
                or "Shall I proceed" in c
                or "Data source:" in c
                or "Chart saved" in c):
            return True
    return False


def run_agent(user_message: str, history: list) -> tuple:
    if not user_message.strip():
        return history, "Enter a request above.", _collect_charts()
    history = history + [{"role": "user", "content": user_message}]

    # Run pre-flight only before the agent has been engaged for the first time.
    # Once the agent has presented a plan / given any response, all subsequent
    # messages (yes/no, corrections, new filters) go straight to the agent.
    if not _agent_already_engaged(history[:-1]):
        clarifying_question = _check_ambiguity(user_message, history[:-1])
        if clarifying_question:
            history = history + [{"role": "assistant", "content": clarifying_question}]
            return history, "Waiting for clarification…", _collect_charts()
    charts_before = set(_collect_charts())
    task = _build_task(user_message, history)

    # When user confirms with "yes", reset agent memory so it starts fresh
    # with only the execution instruction — prevents the model from getting
    # confused by the long plan-confirmation history and re-presenting the plan.
    if _is_confirmation(user_message):
        try:
            agent.memory.reset()
        except Exception:
            pass

    try:
        result = agent.run(task)
    except Exception as e:
        result = f"ERROR: {e}"
    response = str(result)
    charts_after = _collect_charts()
    new_charts = [c for c in charts_after if c not in charts_before]
    if new_charts:
        new_names = ", ".join(os.path.basename(c) for c in new_charts)
        response += f"\n\n📊 New chart(s) generated: {new_names}"
    history = history + [{"role": "assistant", "content": response}]
    status = (f"✅ Done — {len(new_charts)} new chart(s) generated."
              if new_charts else "✅ Done — no new charts.")
    return history, status, charts_after


def clear_all():
    try:
        agent.memory.reset()
    except Exception:
        pass
    return [], "Ready.", []


# ── Build the Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(title="CCAR Backtesting Agent",
               theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:

    gr.Markdown("# CCAR Backtesting Analysis Agent")

    with gr.Tabs():

        # ── Tab 1: Portfolio Dashboard ────────────────────────────────────────
        with gr.Tab("📊 Portfolio Dashboard"):
            gr.Markdown("### Dataset Selection")
            gr.Markdown(
                "Enter the **absolute path** to your `.parquet` file on the server, "
                "then click **Load / Refresh Dashboard**."
            )
            _default_path = str(SAMPLE_DIR / "ccar_round1.parquet")
            dataset_path_box = gr.Textbox(
                value=_default_path,
                label="Dataset path (absolute path to your .parquet file on the server)",
                placeholder="/home/jovyan/your_data/ccar_backtest.parquet",
                interactive=True,
            )
            with gr.Row():
                load_btn    = gr.Button("🔄 Load / Refresh Dashboard", variant="primary")
                dash_status = gr.Textbox(value="Edit the path above if needed, then click Load.",
                                         label="Status", interactive=False, scale=4)

            # Section 1: Dataset overview
            gr.Markdown("---\n### 📁 Dataset Overview")
            overview_table = gr.Dataframe(interactive=False, wrap=True)

            # Section 2: Period performance
            gr.Markdown("---\n### 📉 Performance by Period (portfolio level, based on statement month)")
            gr.Markdown(f"**{list(PERIODS.keys())[0]}**")
            rec_perf_table = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown(f"**{list(PERIODS.keys())[1]}**")
            rec_recent_table = gr.Dataframe(interactive=False, wrap=True)

            # Section 3: EOS negative balance stats
            gr.Markdown("---\n### 🔴 EOS Negative Balance Statistics (last horizon, portfolio level)")
            gr.Markdown(
                "Percentage of statement months where the portfolio-level "
                "EOS balance (at the final forecast horizon) is negative."
            )
            eos_neg_table = gr.Dataframe(interactive=False, wrap=True)

            # Section 4: Average portfolio / account level values
            gr.Markdown("---\n### 💰 Average Portfolio & Account Level Values")
            gr.Markdown(
                "Flow targets (Payment, PurchaseVolume): averaged across statement months "
                "of the **sum across all horizons** per month.  \n"
                "Stock targets (EOS): averaged across statement months of the "
                "**last-horizon value** per month."
            )
            avg_table = gr.Dataframe(interactive=False, wrap=True)

            # Section 5: MPE trend chart
            gr.Markdown("---\n### 📈 MPE Trend by Statement Month")
            gr.Markdown("Shaded regions mark the Great Recession and Recent analysis periods.")
            trend_chart = gr.Plot()

            load_btn.click(
                fn=load_dashboard,
                inputs=[dataset_path_box],
                outputs=[overview_table, rec_perf_table, rec_recent_table,
                         eos_neg_table, avg_table, trend_chart, dash_status],
            )

        # ── Tab 2: Analysis Chat ──────────────────────────────────────────────
        with gr.Tab("💬 Analysis Chat"):
            gr.Markdown(
                "Ask questions about your backtesting data in plain English. "
                "The agent will aggregate your parquet file, compute metrics, "
                "and generate charts."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=520)
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="e.g. Aggregate by horizon and show PD trend…",
                            label="Your request", scale=5, lines=2, autofocus=True)
                        with gr.Column(scale=1, min_width=120):
                            submit_btn = gr.Button("▶ Run", variant="primary")
                            clear_btn  = gr.Button("🗑 Clear")
                    status_box = gr.Textbox(label="Status", interactive=False, lines=1)
                    gr.Markdown("### Quick-start examples")
                    for ex in EXAMPLES:
                        gr.Button(ex[:90] + "…", size="sm").click(
                            fn=lambda e=ex: e, outputs=msg_box)

                with gr.Column(scale=2):
                    gr.Markdown("### Generated Charts")
                    gallery = gr.Gallery(label="Charts (newest first)", columns=2,
                                         height=560, object_fit="contain", show_label=False)
                    refresh_btn = gr.Button("🔄 Refresh Gallery", size="sm")

            with gr.Accordion("📁 Sample data file paths", open=False):
                gr.Markdown(f"""
Copy these paths into your requests:

| Round | Path |
|---|---|
| CCAR Round 1 | `{ROUND1_PATH}` |
| CCAR Round 2 | `{ROUND2_PATH}` |

Output charts saved to: `{os.path.abspath(OUTPUT_DIR)}`
""")

            submit_btn.click(fn=run_agent, inputs=[msg_box, chatbot],
                             outputs=[chatbot, status_box, gallery]
                             ).then(fn=lambda: "", outputs=msg_box)
            msg_box.submit(fn=run_agent, inputs=[msg_box, chatbot],
                           outputs=[chatbot, status_box, gallery]
                           ).then(fn=lambda: "", outputs=msg_box)
            clear_btn.click(fn=clear_all, outputs=[chatbot, status_box, gallery])
            refresh_btn.click(fn=_collect_charts, outputs=gallery)


if __name__ == "__main__":
    import signal, subprocess, platform
    # Kill any process already using port 7860 (cross-platform)
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["lsof", "-ti", "tcp:7860"], capture_output=True, text=True)
            for pid in result.stdout.strip().splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
        else:
            # Linux: use fuser (kills the process directly)
            subprocess.run(["fuser", "-k", "7860/tcp"], capture_output=True)
    except Exception:
        pass

    # JupyterHub proxy support: set root_path so Gradio generates correct URLs
    # JUPYTERHUB_SERVICE_PREFIX is set automatically by JupyterHub, e.g. /user/jovyan/
    _jupyter_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "").rstrip("/")
    _root_path = f"{_jupyter_prefix}/proxy/7860" if _jupyter_prefix else ""

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        root_path=_root_path,
        share=False,
        inbrowser=False,   # headless server — no browser to open
    )
